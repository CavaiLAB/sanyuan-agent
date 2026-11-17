"""Storage artifact client — uploads to the C# Storage service via S2S JWT."""

from __future__ import annotations

import base64
import hashlib
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

_S2S_TOKEN_PROVIDER = Callable[[], Awaitable[str]] | Callable[[], str]
_MAX_SIZE_BYTES = 50 * 1024 * 1024
_ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset()
# Prefix allow-list. ``video/`` and ``audio/`` are allowed so multimodal media
# artifacts can persist; ``application/`` stays restricted to specific subtypes
# (json/pdf) so arbitrary binaries like ``application/octet-stream`` remain rejected.
_ALLOWED_CONTENT_PREFIXES: tuple[str, ...] = (
    "text/",
    "image/",
    "video/",
    "audio/",
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


def _sha256_b64(data: bytes) -> str:
    """Base64 of the RAW digest bytes — the wire shape the C# ``byte[] Sha256``
    field round-trips (System.Text.Json serialises ``byte[]`` as base64). Sending
    the hex string here silently stores the wrong bytes."""
    return base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")


def _is_guid(value: str | None) -> bool:
    """True iff *value* parses as a GUID. Storage's ``ConversationId`` is a
    ``Guid?``; a non-GUID (dev ids like ``c-demo-1``, or a run-id fallback) would
    fail model binding, so callers omit it and let Storage default it to null."""
    if not value:
        return False
    try:
        uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return False
    return True


def _object_key(filename: str, *, run_id: str | None) -> str:
    """Storage-unique key for the stored object. ``ObjectKey`` is required and is
    the UNIQUE index on the artifacts table, so two runs producing ``report.md``
    must not collide — prefix with the run id (or a random id when absent). The
    filename stays the leaf so downloads keep the original name."""
    prefix = run_id or uuid.uuid4().hex
    return f"artifacts/{prefix}/{filename}"


def _apply_ownership_meta(
    meta_payload: dict[str, Any],
    *,
    ownership: dict[str, str] | None,
    kind: str | None,
    origin: str | None,
) -> None:
    """Fold additive C0.1 data-centralization tags into the artifact metadata.

    ``ownership`` carries owner/department/tenant (forwarded verbatim); ``kind``
    and ``origin`` classify the artifact. All optional — absent keys are omitted
    so the request stays backward-compatible.
    """
    if ownership:
        if ownership.get("owner"):
            meta_payload["owner"] = ownership["owner"]
        if ownership.get("department"):
            meta_payload["department"] = ownership["department"]
    if kind is not None:
        meta_payload["kind"] = kind
    if origin is not None:
        meta_payload["origin"] = origin


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

    async def register_sandbox_artifact(
        self,
        tenant_id: str,
        content_bytes: bytes,
        content_type: str,
        filename: str,
        *,
        run_id: str | None = None,
        user_id: str | None = None,
        origin_path: str | None = None,
        conversation_id: str | None = None,
        ownership: dict[str, str] | None = None,
        kind: str | None = None,
        origin: str | None = None,
    ) -> ArtifactRef:
        """Upload a sandbox output file as an artifact.

        Unlike ``create_artifact`` this method skips the content-type
        whitelist (the caller validates via extension blacklist) and
        enforces a **100 MB** size limit.

        *conversation_id* scopes the artifact to a conversation so the file
        panel (which lists by ``conversation_id``) shows it — forwarded only
        when it is a GUID (Storage's ``ConversationId`` is a ``Guid?``).
        *ownership* (owner/department), *kind* (``produced``/``uploaded``) and
        *origin* (``cloud``/``computer``) are additive data-centralization tags.
        ``tenant_id`` is carried by the S2S token (the row's ``TenantId`` comes
        from the token's tenant claim, not the body).

        Raises:
            ValueError: payload exceeds 100 MB.
            httpx.HTTPStatusError: Storage service error after one retry.
        """
        _sandbox_max = 100 * 1024 * 1024
        if len(content_bytes) > _sandbox_max:
            raise ValueError(
                f"Payload too large: {len(content_bytes)} bytes "
                f"(max {_sandbox_max})",
            )
        return await self._upload(
            content_bytes,
            content_type=content_type,
            filename=filename,
            run_id=run_id,
            conversation_id=conversation_id,
            ownership=ownership,
            kind=kind,
            origin=origin,
        )

    async def create_artifact(
        self,
        tenant_id: str,
        content_bytes: bytes,
        content_type: str,
        filename: str,
        *,
        conversation_id: str | None = None,
        ownership: dict[str, str] | None = None,
        kind: str | None = None,
        origin: str | None = None,
    ) -> ArtifactRef:
        """Upload *content_bytes* as a new artifact and return its reference.

        *conversation_id* scopes the artifact (forwarded only when a GUID).
        *ownership*/*kind*/*origin* are additive data-centralization tags. The
        row's tenant comes from the S2S token, not *tenant_id* in the body.

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
        return await self._upload(
            content_bytes,
            content_type=content_type,
            filename=filename,
            run_id=None,
            conversation_id=conversation_id,
            ownership=ownership,
            kind=kind,
            origin=origin,
        )

    async def _upload(
        self,
        content_bytes: bytes,
        *,
        content_type: str,
        filename: str,
        run_id: str | None,
        conversation_id: str | None,
        ownership: dict[str, str] | None,
        kind: str | None,
        origin: str | None,
    ) -> ArtifactRef:
        """The 3-step Storage upload: create metadata (presigned PUT slot) →
        PUT bytes to object storage → complete (marks Ready). Wire shape matches
        the deployed C# ``ArtifactEndpoints`` (camelCase, ``objectKey`` required,
        ``sha256`` = base64 of raw digest, ``id``/``uploadUrl`` in responses)."""
        token = await self._token()
        headers: dict[str, str] = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        object_key = _object_key(filename, run_id=run_id)
        meta_payload: dict[str, Any] = {
            "objectKey": object_key,
            "sizeBytes": len(content_bytes),
            "sha256": _sha256_b64(content_bytes),
            "contentType": content_type,
        }
        if _is_guid(conversation_id):
            meta_payload["conversationId"] = conversation_id
        _apply_ownership_meta(meta_payload, ownership=ownership, kind=kind, origin=origin)

        # Step 1 — create metadata row + presigned PUT URL
        meta_resp = await self._post_with_retry(
            f"{self._base_url}/api/v1/artifacts",
            json=meta_payload,
            headers=headers,
        )
        meta_resp.raise_for_status()
        meta = meta_resp.json()
        artifact_id: str = meta["id"]
        upload_url: str = meta["uploadUrl"]

        # Step 2 — PUT bytes to the presigned object-storage URL (no S2S header;
        # the presign carries its own auth)
        upload_resp = await self._put_with_retry(
            upload_url,
            content=content_bytes,
            headers={"Content-Type": content_type},
        )
        upload_resp.raise_for_status()

        # Step 3 — complete (no body; id is the route param). Marks the row Ready.
        complete_resp = await self._post_with_retry(
            f"{self._base_url}/api/v1/artifacts/{artifact_id}/complete",
            json=None,
            headers=headers,
        )
        complete_resp.raise_for_status()

        return ArtifactRef(
            id=artifact_id,
            url=await self._download_url(artifact_id, headers),
            sha256=_sha256(content_bytes),
            filename=filename,
            content_type=content_type,
        )

    async def _download_url(self, artifact_id: str, headers: dict[str, str]) -> str:
        """Fetch a browser-usable presigned GET URL for the SSE artifact frame.
        Best-effort: if the download endpoint errors, fall back to the artifact
        route so the frame still carries a resolvable id."""
        try:
            resp = await self._get_with_retry(
                f"{self._base_url}/api/v1/artifacts/{artifact_id}/download",
                headers=headers,
            )
            resp.raise_for_status()
            return str(resp.json().get("url") or "")
        except (httpx.HTTPError, KeyError, ValueError):
            return f"{self._base_url}/api/v1/artifacts/{artifact_id}/download"

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

    async def _get_with_retry(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """GET with one retry on network-level failures."""
        last_exc: Exception | None = None
        for _attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    return await client.get(url, headers=headers)
            except (httpx.NetworkError, httpx.TimeoutException) as exc:
                last_exc = exc
        raise last_exc  # type: ignore[misc]
