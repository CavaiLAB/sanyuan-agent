"""Storage artifact client — uploads to the C# Storage service via S2S JWT."""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

_S2S_TOKEN_PROVIDER = Callable[[], Awaitable[str]] | Callable[[], str]
_MAX_SIZE_BYTES = 50 * 1024 * 1024
_ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset()
_ALLOWED_CONTENT_PREFIXES: tuple[str, ...] = (
    "text/",
    "image/",
    "application/json",
    "application/pdf",
)


def _is_allowed_content_type(content_type: str) -> bool:
    ct = content_type.lower().strip()
    if ct in _ALLOWED_CONTENT_TYPES:
        return True
    return any(ct.startswith(prefix) for prefix in _ALLOWED_CONTENT_PREFIXES)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Reference to a stored artifact."""

    id: str
    url: str
    sha256: str
    filename: str
    content_type: str


class StorageArtifactClient:
    """Async client for the Storage service artifact API.

    *base_url* is the root of the Storage service (e.g.
    ``https://storage.internal``).

    *s2s_token_provider* is a callable that returns a valid S2S JWT string.
    It may be sync or async.
    """

    def __init__(
        self,
        base_url: str,
        s2s_token_provider: _S2S_TOKEN_PROVIDER,
        *,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token_provider = s2s_token_provider
        self._timeout = timeout

    async def _token(self) -> str:
        result = self._token_provider()
        if isinstance(result, Awaitable):
            return await result
        return result

    async def create_artifact(
        self,
        tenant_id: str,
        content_bytes: bytes,
        content_type: str,
        filename: str,
    ) -> ArtifactRef:
        """Upload *content_bytes* as a new artifact and return its reference.

        Raises:
            ValueError: if *content_type* is not in the whitelist or the payload
                exceeds 50 MB.
            httpx.HTTPStatusError: if the Storage service returns an error
                after one retry.
        """
        if not _is_allowed_content_type(content_type):
            raise ValueError(f"Content type not allowed: {content_type}")

        if len(content_bytes) > _MAX_SIZE_BYTES:
            raise ValueError(
                f"Payload too large: {len(content_bytes)} bytes "
                f"(max {_MAX_SIZE_BYTES})",
            )

        digest = _sha256(content_bytes)
        token = await self._token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # Step 1 — request upload slot
        meta_payload: dict[str, Any] = {
            "tenant_id": tenant_id,
            "filename": filename,
            "content_type": content_type,
            "size": len(content_bytes),
            "sha256": digest,
        }

        meta_resp = await self._post_with_retry(
            f"{self._base_url}/api/v1/storage/artifacts",
            json=meta_payload,
            headers=headers,
        )
        meta_resp.raise_for_status()
        meta = meta_resp.json()
        artifact_id: str = meta["artifact_id"]
        upload_url: str = meta["upload_url"]

        # Step 2 — upload bytes to presigned URL
        upload_resp = await self._put_with_retry(
            upload_url,
            content=content_bytes,
            headers={"Content-Type": content_type},
        )
        upload_resp.raise_for_status()

        # Step 3 — complete
        complete_resp = await self._post_with_retry(
            f"{self._base_url}/api/v1/storage/artifacts/{artifact_id}/complete",
            json={"sha256": digest},
            headers=headers,
        )
        complete_resp.raise_for_status()
        complete = complete_resp.json()

        return ArtifactRef(
            id=artifact_id,
            url=complete.get("download_url", upload_url),
            sha256=digest,
            filename=filename,
            content_type=content_type,
        )

    async def _post_with_retry(
        self,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """POST with one retry on network-level failures."""
        last_exc: Exception | None = None
        for _attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    return await client.post(url, json=json, headers=headers)
            except (httpx.NetworkError, httpx.TimeoutException) as exc:
                last_exc = exc
        raise last_exc  # type: ignore[misc]

    async def _put_with_retry(
        self,
        url: str,
        *,
        content: bytes,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """PUT with one retry on network-level failures."""
        last_exc: Exception | None = None
        for _attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    return await client.put(url, content=content, headers=headers)
            except (httpx.NetworkError, httpx.TimeoutException) as exc:
                last_exc = exc
        raise last_exc  # type: ignore[misc]
