"""ReaderRouter plus HTTP and Exa Contents backend mappings."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from time import monotonic, perf_counter
from typing import Any
from urllib.parse import urlsplit

from neuroagent.tools.block_detect import detect_block
from neuroagent.tools.web_extract import extract_http_document
from neuroagent.tools.web_models import WebStatus
from neuroagent.tools.web_policy import URLClassifier
from neuroagent.tools.web_reader_models import (
    BrowserReaderSessionFactory,
    ExaContentsClient,
    FetchAttempt,
    FetchBackendError,
    FetchDocument,
    FetchRequest,
    FirecrawlClient,
    HttpFetchRequest,
    HttpReaderClient,
    ReaderBackend,
    SourceAdapter,
)

ReaderObserver = Callable[[FetchAttempt], None]
_DOCUMENT_EXTENSIONS = (
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
)
_LOGIN_WALL_RE = re.compile(r"\b(sign[ -]?in|log[ -]?in|authentication required)\b|登录|登入", re.I)


class _PageBudget:
    """Concurrency-safe per-router page budget (one router is one app run)."""

    def __init__(self, limit: int) -> None:
        if limit < 0:
            raise ValueError("page budget must not be negative")
        self._limit = limit
        self._used = 0
        self._lock = asyncio.Lock()

    async def reserve(self) -> bool:
        async with self._lock:
            if self._used >= self._limit:
                return False
            self._used += 1
            return True


class FirecrawlReaderBackend:
    """Cost-bounded Firecrawl v2 Scrape fallback for difficult public pages."""

    name = "firecrawl"
    third_party = True

    def __init__(
        self,
        client: FirecrawlClient | None,
        *,
        max_pages: int = 3,
        max_concurrency: int = 2,
        request_timeout_seconds: float = 20.0,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("Firecrawl max_concurrency must be positive")
        if request_timeout_seconds <= 0:
            raise ValueError("Firecrawl request timeout must be positive")
        self._client = client
        self._budget = _PageBudget(max_pages)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._request_timeout_seconds = request_timeout_seconds

    @property
    def available(self) -> bool:
        return self._client is not None

    async def read(self, request: FetchRequest, *, deadline: float) -> FetchDocument:
        if self._client is None:
            return FetchDocument(
                url=request.url,
                final_url=request.url,
                provider=self.name,
                status="provider_unavailable",
                error_code="firecrawl_client_unavailable",
            )
        if urlsplit(request.url).path.lower().endswith(_DOCUMENT_EXTENSIONS):
            return FetchDocument(
                url=request.url,
                final_url=request.url,
                provider=self.name,
                status="unsupported_content",
                error_code="document_conversion_deferred",
            )
        if not await self._budget.reserve():
            return FetchDocument(
                url=request.url,
                final_url=request.url,
                provider=self.name,
                status="rate_limited",
                error_code="firecrawl_run_page_budget_exhausted",
            )

        remaining = min(self._request_timeout_seconds, deadline - monotonic())
        if remaining <= 0:
            return FetchDocument(
                url=request.url,
                final_url=request.url,
                provider=self.name,
                status="timeout",
                error_code="chain_deadline_exhausted",
            )
        payload: dict[str, Any] = {
            "url": request.url,
            "formats": ["markdown"],
            "onlyMainContent": True,
            "timeout": max(1, int(remaining * 1000)),
        }
        async with self._semaphore:
            raw = await asyncio.wait_for(
                self._client.scrape(payload), timeout=max(0.001, remaining)
            )

        data = raw.get("data")
        item = data if isinstance(data, Mapping) else {}
        metadata_value = item.get("metadata")
        metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
        content = _optional_string(item.get("markdown")) or ""
        title = _optional_string(metadata.get("title")) or ""
        final_url = (
            _optional_string(metadata.get("sourceURL"))
            or _optional_string(metadata.get("url"))
            or request.url
        )
        request_id = (
            _optional_string(item.get("scrapeId"))
            or _optional_string(metadata.get("scrapeId"))
            or _optional_string(raw.get("id"))
        )
        credits_used = _nonnegative_int(raw.get("creditsUsed"))
        if _looks_like_login(title, content):
            return FetchDocument(
                url=request.url,
                final_url=final_url,
                provider=self.name,
                status="login_required",
                title=title,
                bytes=len(content.encode()),
                request_id=request_id,
                credits_used=credits_used,
                error_code="firecrawl_login_wall",
                block_reason="login_required",
            )
        blocked, block_reason = detect_block(url=final_url, title=title, content=content)
        if blocked:
            status: WebStatus = "captcha" if "验证" in (block_reason or "") else "blocked"
            return FetchDocument(
                url=request.url,
                final_url=final_url,
                provider=self.name,
                status=status,
                title=title,
                bytes=len(content.encode()),
                request_id=request_id,
                credits_used=credits_used,
                error_code="firecrawl_soft_block",
                block_reason=block_reason,
            )
        if len(content.strip()) < request.min_content_chars:
            return FetchDocument(
                url=request.url,
                final_url=final_url,
                provider=self.name,
                status="empty_content",
                title=title,
                bytes=len(content.encode()),
                request_id=request_id,
                credits_used=credits_used,
                error_code="firecrawl_content_below_quality_threshold",
            )
        return FetchDocument(
            url=request.url,
            final_url=final_url,
            provider=self.name,
            status="success",
            title=title,
            content=content,
            content_type="text/markdown",
            bytes=len(content.encode()),
            request_id=request_id,
            credits_used=credits_used,
        )


class BrowserReaderBackend:
    """Last-resort renderer using one isolated BrowserSession per app run."""

    name = "browser"
    third_party = True

    def __init__(
        self,
        session_factory: BrowserReaderSessionFactory | None,
        *,
        run_key: str,
        max_sessions: int = 1,
        max_concurrency: int = 1,
        page_wait_seconds: float = 8.0,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("Browser max_concurrency must be positive")
        if page_wait_seconds <= 0:
            raise ValueError("Browser page wait must be positive")
        self._session_factory = session_factory
        self._run_key = run_key
        self._budget = _PageBudget(max_sessions)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._page_wait_seconds = page_wait_seconds

    @property
    def available(self) -> bool:
        return self._session_factory is not None and bool(self._run_key)

    async def read(self, request: FetchRequest, *, deadline: float) -> FetchDocument:
        if self._session_factory is None or not self._run_key:
            return FetchDocument(
                url=request.url,
                final_url=request.url,
                provider=self.name,
                status="provider_unavailable",
                error_code="browser_session_factory_unavailable",
            )
        if not await self._budget.reserve():
            return FetchDocument(
                url=request.url,
                final_url=request.url,
                provider=self.name,
                status="rate_limited",
                error_code="browser_run_session_budget_exhausted",
            )
        remaining = min(self._page_wait_seconds, deadline - monotonic())
        if remaining <= 0:
            return FetchDocument(
                url=request.url,
                final_url=request.url,
                provider=self.name,
                status="timeout",
                error_code="chain_deadline_exhausted",
            )

        session = None
        async with self._semaphore:
            try:
                session = await self._session_factory(self._run_key)
                navigation = await session.navigate(request.url)
                if navigation.get("policy_denied"):
                    return FetchDocument(
                        url=request.url,
                        final_url=request.url,
                        provider=self.name,
                        status="policy_denied",
                        error_code=str(navigation.get("error_code") or "browser_policy_denied"),
                    )
                if not navigation.get("success"):
                    return FetchDocument(
                        url=request.url,
                        final_url=str(navigation.get("url") or request.url),
                        provider=self.name,
                        status="provider_unavailable",
                        error_code="browser_navigation_failed",
                    )
                await session.wait_until_ready(max(1, int(remaining * 1000)))
                dom = await session.read_dom("body")
                if not dom.get("success"):
                    return FetchDocument(
                        url=request.url,
                        final_url=str(dom.get("url") or navigation.get("url") or request.url),
                        provider=self.name,
                        status="empty_content",
                        error_code="browser_dom_unavailable",
                    )
                final_url = str(dom.get("url") or navigation.get("url") or request.url)
                title = str(dom.get("title") or navigation.get("title") or "")
                content = str(dom.get("text") or "")
                html = str(dom.get("html") or "")
                if _looks_like_login(title, content):
                    return FetchDocument(
                        url=request.url,
                        final_url=final_url,
                        provider=self.name,
                        status="login_required",
                        title=title,
                        bytes=len(content.encode()),
                        error_code="browser_login_wall",
                        block_reason="login_required",
                    )
                blocked, block_reason = detect_block(
                    url=final_url, title=title, content=content, html=html
                )
                if blocked:
                    status: WebStatus = "captcha" if "验证" in (block_reason or "") else "blocked"
                    return FetchDocument(
                        url=request.url,
                        final_url=final_url,
                        provider=self.name,
                        status=status,
                        title=title,
                        bytes=len(content.encode()),
                        error_code="browser_soft_block",
                        block_reason=block_reason,
                    )
                if len(content.strip()) < request.min_content_chars:
                    return FetchDocument(
                        url=request.url,
                        final_url=final_url,
                        provider=self.name,
                        status="empty_content",
                        title=title,
                        bytes=len(content.encode()),
                        error_code="browser_content_below_quality_threshold",
                    )
                return FetchDocument(
                    url=request.url,
                    final_url=final_url,
                    provider=self.name,
                    status="success",
                    title=title,
                    content=content,
                    content_type="text/plain",
                    bytes=len(content.encode()),
                )
            finally:
                if session is not None:
                    await asyncio.shield(session.close())


class HttpReaderBackend:
    """Local public-URL reader over an app-injected bounded HTTP client."""

    name = "http"
    third_party = False

    def __init__(self, client: HttpReaderClient | None) -> None:
        self._client = client

    @property
    def available(self) -> bool:
        return self._client is not None

    async def read(self, request: FetchRequest, *, deadline: float) -> FetchDocument:
        if self._client is None:
            return FetchDocument(
                url=request.url,
                final_url=request.url,
                provider=self.name,
                status="provider_unavailable",
                error_code="http_client_unavailable",
            )
        response = await self._client.fetch(
            HttpFetchRequest(url=request.url, max_bytes=request.max_bytes, deadline=deadline)
        )
        return extract_http_document(request, response)


class ExaContentsBackend:
    """Exa Contents mapping; document conversion remains deferred to W4."""

    name = "exa"
    third_party = True

    def __init__(self, client: ExaContentsClient | None) -> None:
        self._client = client

    @property
    def available(self) -> bool:
        return self._client is not None

    async def read(self, request: FetchRequest, *, deadline: float) -> FetchDocument:
        if self._client is None:
            return FetchDocument(
                url=request.url,
                final_url=request.url,
                provider=self.name,
                status="provider_unavailable",
                error_code="exa_client_unavailable",
            )
        if urlsplit(request.url).path.lower().endswith(_DOCUMENT_EXTENSIONS):
            return FetchDocument(
                url=request.url,
                final_url=request.url,
                provider=self.name,
                status="unsupported_content",
                error_code="document_conversion_deferred",
            )
        if monotonic() >= deadline:
            return FetchDocument(
                url=request.url,
                final_url=request.url,
                provider=self.name,
                status="timeout",
                error_code="chain_deadline_exhausted",
            )

        raw = await self._client.contents({"urls": [request.url], "text": True})
        request_id = _optional_string(raw.get("requestId"))
        cost_usd = _exa_cost(raw)
        raw_items = raw.get("results")
        items = raw_items if isinstance(raw_items, list) else []
        if not items or not isinstance(items[0], Mapping):
            return FetchDocument(
                url=request.url,
                final_url=request.url,
                provider=self.name,
                status="empty_content",
                request_id=request_id,
                cost_usd=cost_usd,
                error_code=_exa_status_code(raw) or "exa_empty_results",
            )
        item = items[0]
        content = _optional_string(item.get("text")) or ""
        title = _optional_string(item.get("title")) or ""
        final_url = _optional_string(item.get("url")) or request.url
        blocked, block_reason = detect_block(url=final_url, title=title, content=content)
        if blocked:
            return FetchDocument(
                url=request.url,
                final_url=final_url,
                provider=self.name,
                status="blocked",
                title=title,
                bytes=len(content.encode()),
                request_id=request_id,
                cost_usd=cost_usd,
                error_code="exa_soft_block",
                block_reason=block_reason,
            )
        if len(content.strip()) < request.min_content_chars:
            return FetchDocument(
                url=request.url,
                final_url=final_url,
                provider=self.name,
                status="empty_content",
                title=title,
                bytes=len(content.encode()),
                request_id=request_id,
                cost_usd=cost_usd,
                error_code="exa_content_below_quality_threshold",
            )
        return FetchDocument(
            url=request.url,
            final_url=final_url,
            provider=self.name,
            status="success",
            title=title,
            content=content,
            content_type="text/plain",
            bytes=len(content.encode()),
            request_id=request_id,
            cost_usd=cost_usd,
        )


class ReaderRouter:
    """Apply URL policy, a chain deadline, and ordered provider fallback."""

    def __init__(
        self,
        backends: Sequence[ReaderBackend],
        classifier: URLClassifier,
        *,
        chain_timeout_seconds: float = 25.0,
        observer: ReaderObserver | None = None,
    ) -> None:
        if not backends:
            raise ValueError("ReaderRouter requires at least one backend")
        if chain_timeout_seconds <= 0:
            raise ValueError("chain_timeout_seconds must be positive")
        self._backends = tuple(backends)
        self._classifier = classifier
        self._chain_timeout_seconds = chain_timeout_seconds
        self._observer = observer

    async def fetch(self, request: FetchRequest) -> FetchDocument:
        classification = await self._classifier.classify(request.url)
        if not classification.allowed:
            attempt = FetchAttempt(
                provider="policy",
                status="policy_denied",
                latency_ms=0,
                error_code=classification.reason,
            )
            self._record(attempt)
            return FetchDocument(
                url=classification.redacted_url,
                final_url=classification.redacted_url,
                provider="policy",
                status="policy_denied",
                error_code=classification.reason,
                attempts=(attempt,),
            )

        deadline = monotonic() + min(request.timeout_seconds, self._chain_timeout_seconds)
        attempts: list[FetchAttempt] = []
        fallback_reason: str | None = None
        last_document: FetchDocument | None = None
        for backend in self._backends:
            if isinstance(backend, SourceAdapter) and not backend.supports(request, last_document):
                continue
            if monotonic() >= deadline:
                attempt = FetchAttempt(
                    provider=backend.name,
                    status="timeout",
                    latency_ms=0,
                    fallback_reason=fallback_reason,
                    error_code="chain_deadline_exhausted",
                )
                attempts.append(attempt)
                self._record(attempt)
                break
            if not backend.available:
                attempt = FetchAttempt(
                    provider=backend.name,
                    status="provider_unavailable",
                    latency_ms=0,
                    fallback_reason=fallback_reason,
                    error_code="startup_preflight_unavailable",
                )
                attempts.append(attempt)
                self._record(attempt)
                fallback_reason = attempt.status
                continue

            # Re-resolve immediately before a third-party handoff. A changed DNS
            # answer is treated as rebinding and the raw URL never reaches the vendor.
            if backend.third_party:
                vendor_classification = await self._classifier.classify(request.url)
                if (
                    not vendor_classification.allowed
                    or vendor_classification.resolved_ips != classification.resolved_ips
                ):
                    attempt = FetchAttempt(
                        provider=backend.name,
                        status="policy_denied",
                        latency_ms=0,
                        fallback_reason=fallback_reason,
                        error_code=(
                            vendor_classification.reason
                            if not vendor_classification.allowed
                            else "dns_rebinding_detected"
                        ),
                    )
                    attempts.append(attempt)
                    self._record(attempt)
                    return FetchDocument(
                        url=classification.redacted_url,
                        final_url=classification.redacted_url,
                        provider=backend.name,
                        status="policy_denied",
                        error_code=attempt.error_code,
                        attempts=tuple(attempts),
                    )

            started = perf_counter()
            remaining = max(0.001, deadline - monotonic())
            try:
                document = await asyncio.wait_for(
                    backend.read(request, deadline=deadline), timeout=remaining
                )
            except FetchBackendError as exc:
                document = FetchDocument(
                    url=exc.error.safe_url or classification.redacted_url,
                    final_url=exc.error.safe_url or classification.redacted_url,
                    provider=backend.name,
                    status=exc.error.status,
                    error_code=exc.error.code,
                )
            except TimeoutError:
                document = FetchDocument(
                    url=classification.redacted_url,
                    final_url=classification.redacted_url,
                    provider=backend.name,
                    status="timeout",
                    error_code="backend_timeout",
                )
            except Exception as exc:
                document = FetchDocument(
                    url=classification.redacted_url,
                    final_url=classification.redacted_url,
                    provider=backend.name,
                    status="provider_unavailable",
                    error_code=type(exc).__name__,
                )
            if (
                document.status == "success"
                and len(document.content.strip()) < request.min_content_chars
            ):
                document = replace(
                    document,
                    status="empty_content",
                    content="",
                    error_code="content_below_quality_threshold",
                )
            attempt = FetchAttempt(
                provider=backend.name,
                status=document.status,
                latency_ms=(perf_counter() - started) * 1000,
                bytes=document.bytes,
                retry_count=document.retry_count,
                fallback_reason=fallback_reason,
                error_code=document.error_code,
                request_id=document.request_id,
                cost_usd=document.cost_usd,
                credits_used=document.credits_used,
            )
            attempts.append(attempt)
            self._record(attempt)
            last_document = document
            if document.status == "success":
                return replace(document, attempts=tuple(attempts))
            if document.status in {"policy_denied", "login_required"}:
                return replace(document, attempts=tuple(attempts))
            fallback_reason = document.status

        if last_document is not None:
            return replace(last_document, attempts=tuple(attempts))
        if attempts:
            last_attempt = attempts[-1]
            return FetchDocument(
                url=classification.redacted_url,
                final_url=classification.redacted_url,
                provider=last_attempt.provider,
                status=last_attempt.status,
                error_code=last_attempt.error_code,
                attempts=tuple(attempts),
            )
        return FetchDocument(
            url=classification.redacted_url,
            final_url=classification.redacted_url,
            provider="router",
            status="timeout",
            error_code="chain_deadline_exhausted",
            attempts=tuple(attempts),
        )

    def _record(self, attempt: FetchAttempt) -> None:
        if self._observer is None:
            return
        try:
            self._observer(attempt)
        except Exception:
            return


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _looks_like_login(title: str, content: str) -> bool:
    return len(content) < 5000 and bool(_LOGIN_WALL_RE.search(f"{title}\n{content[:2000]}"))


def _nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _exa_cost(data: Mapping[str, Any]) -> float | None:
    raw = data.get("costDollars")
    if not isinstance(raw, Mapping):
        return None
    total = raw.get("total")
    return float(total) if isinstance(total, int | float) and total >= 0 else None


def _exa_status_code(data: Mapping[str, Any]) -> str | None:
    statuses = data.get("statuses")
    if not isinstance(statuses, list) or not statuses or not isinstance(statuses[0], Mapping):
        return None
    value = statuses[0].get("status")
    return f"exa_{value}" if isinstance(value, str) and value else None


__all__ = [
    "BrowserReaderBackend",
    "ExaContentsBackend",
    "FirecrawlReaderBackend",
    "HttpReaderBackend",
    "ReaderObserver",
    "ReaderRouter",
]
