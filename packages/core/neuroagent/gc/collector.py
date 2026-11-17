"""Sandbox and Sensory memory garbage collectors."""

from __future__ import annotations

import logging
import mimetypes
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx
from opentelemetry import metrics

from neuroagent.artifacts.client import StorageArtifactClient

logger = logging.getLogger(__name__)

_meter = metrics.get_meter("neuroagent.gc")

_sandbox_soft_counter = _meter.create_counter(
    "neuroagent_sandbox_soft_released_total",
    description="Total sandboxes soft-released (paused)",
    unit="1",
)
_sandbox_hard_counter = _meter.create_counter(
    "neuroagent_sandbox_hard_released_total",
    description="Total sandboxes hard-released (deleted)",
    unit="1",
)
_sensory_expired_counter = _meter.create_counter(
    "neuroagent_sensory_expired_total",
    description="Total sensory memory entries expired",
    unit="1",
)

@dataclass(frozen=True, slots=True)
class SandboxInfo:
    id: str
    tenant_id: str
    last_activity_ts: float


TimeProvider = Callable[[], float]
SandboxListProvider = Callable[[], Awaitable[list[SandboxInfo]]]
AuditCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


@dataclass
class GCRunStats:
    last_run_ts: float = 0.0
    last_sandbox_soft_count: int = 0
    last_sandbox_hard_count: int = 0
    last_sensory_expired_count: int = 0
    last_longterm_expired_count: int = 0
    last_longterm_evicted_count: int = 0


async def _noop_audit(event: str, data: dict[str, Any]) -> None:
    logger.info("audit: %s %s", event, data)


def _wall_time() -> float:
    import time

    return time.time()


class SandboxCollector:
    """Identifies idle sandboxes and performs soft/hard release."""

    def __init__(
        self,
        *,
        supervisor_base_url: str,
        sandbox_list_provider: SandboxListProvider,
        storage_client: StorageArtifactClient | None = None,
        supervisor_token: str | None = None,
        soft_threshold_seconds: float = 1800.0,
        hard_threshold_seconds: float = 604800.0,
        time_provider: TimeProvider | None = None,
        audit_callback: AuditCallback | None = None,
    ) -> None:
        self._supervisor_url = supervisor_base_url.rstrip("/")
        self._list_provider = sandbox_list_provider
        self._storage_client = storage_client
        self._supervisor_token = supervisor_token
        self._soft_threshold = soft_threshold_seconds
        self._hard_threshold = hard_threshold_seconds
        self._now = time_provider or _wall_time
        self._audit = audit_callback or _noop_audit
        self._stats = GCRunStats()

    @property
    def stats(self) -> GCRunStats:
        return self._stats

    async def collect(self) -> GCRunStats:
        now = self._now()
        sandboxes = await self._list_provider()

        soft_count = 0
        hard_count = 0

        for sb in sandboxes:
            idle = now - sb.last_activity_ts
            if idle > self._hard_threshold:
                if await self._hard_release(sb):
                    hard_count += 1
            elif idle > self._soft_threshold:
                if await self._soft_release(sb):
                    soft_count += 1

        self._stats.last_run_ts = now
        self._stats.last_sandbox_soft_count = soft_count
        self._stats.last_sandbox_hard_count = hard_count
        return self._stats

    async def _soft_release(self, sb: SandboxInfo) -> bool:
        url = f"{self._supervisor_url}/v2/sandboxes/{sb.id}/pause"
        headers = self._auth_headers()
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, headers=headers)
                if resp.status_code == 501:
                    del_url = (
                        f"{self._supervisor_url}/v2/sandboxes/{sb.id}?grace=300"
                    )
                    resp = await client.delete(del_url, headers=headers)
                resp.raise_for_status()
        except Exception:
            logger.exception("soft release failed for sandbox %s", sb.id)
            return False

        _sandbox_soft_counter.add(1, {"tenant": sb.tenant_id})
        await self._audit("sandbox.softReleased", {
            "sandbox_id": sb.id,
            "tenant_id": sb.tenant_id,
        })
        logger.info("soft released sandbox %s", sb.id)
        return True

    async def _hard_release(self, sb: SandboxInfo) -> bool:
        if self._storage_client is not None:
            try:
                await self._export_outputs(sb)
            except Exception:
                logger.warning(
                    "export failed for sandbox %s — skipping hard release",
                    sb.id,
                    exc_info=True,
                )
                return False

        url = f"{self._supervisor_url}/v2/sandboxes/{sb.id}?force=true"
        headers = self._auth_headers()
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.delete(url, headers=headers)
                resp.raise_for_status()
        except Exception:
            logger.exception("hard release delete failed for sandbox %s", sb.id)
            return False

        _sandbox_hard_counter.add(1, {"tenant": sb.tenant_id})
        await self._audit("sandbox.hardReleased", {
            "sandbox_id": sb.id,
            "tenant_id": sb.tenant_id,
        })
        logger.info("hard released sandbox %s", sb.id)
        return True

    async def _export_outputs(self, sb: SandboxInfo) -> None:
        assert self._storage_client is not None

        list_url = f"{self._supervisor_url}/v2/sandboxes/{sb.id}/files/list"
        headers = self._auth_headers()
        params: dict[str, str] = {"path": "outputs/", "recursive": "true"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(list_url, headers=headers, params=params)
            resp.raise_for_status()
            entries = resp.json().get("entries", [])

            for entry in entries:
                if entry.get("type") != "file":
                    continue
                file_path = entry["path"]
                filename = file_path.rsplit("/", 1)[-1]

                file_url = f"{self._supervisor_url}/v2/sandboxes/{sb.id}/files"
                resp = await client.get(
                    file_url, headers=headers, params={"path": file_path},
                )
                resp.raise_for_status()
                content = resp.content

                content_type, _ = mimetypes.guess_type(filename)
                if content_type is None:
                    content_type = "application/octet-stream"

                await self._storage_client.register_sandbox_artifact(
                    tenant_id=sb.tenant_id,
                    content_bytes=content,
                    content_type=content_type,
                    filename=filename,
                    run_id=sb.id,
                    origin_path=file_path,
                )

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._supervisor_token:
            headers["Authorization"] = f"Bearer {self._supervisor_token}"
        return headers


class SensoryCollector:
    """Scans Redis for expired sensory memory keys and emits OTel metric."""

    def __init__(
        self,
        redis_client: Any | None = None,
        key_pattern: str = "sensory:*",
        time_provider: TimeProvider | None = None,
    ) -> None:
        self._redis = redis_client
        self._pattern = key_pattern
        self._now = time_provider or _wall_time
        self._expired_count = 0

    @property
    def expired_count(self) -> int:
        return self._expired_count

    async def collect(self) -> int:
        if self._redis is None:
            return 0

        expired = 0
        cursor: int = 0
        while True:
            cursor, keys = await self._redis.scan(
                cursor=cursor, match=self._pattern, count=100,
            )
            for key in keys:
                ttl = await self._redis.ttl(key)
                if ttl is not None and ttl <= 0:
                    expired += 1
            if cursor == 0:
                break

        self._expired_count = expired
        if expired > 0:
            _sensory_expired_counter.add(expired, {"tenant": "default"})
        return expired
