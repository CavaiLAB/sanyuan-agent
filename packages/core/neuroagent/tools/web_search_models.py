"""Provider-neutral models and protocols for Web Search routing.

The harness owns request/result semantics but performs no Exa network I/O.
Concrete HTTP clients are injected by the app layer so the standalone harness
keeps its dependency boundary and tests can use pure fakes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

WebStatus = Literal[
    "success",
    "blocked",
    "captcha",
    "login_required",
    "timeout",
    "rate_limited",
    "empty_content",
    "unsupported_content",
    "provider_unavailable",
    "policy_denied",
]


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """Internal search request; provider selection is intentionally absent."""

    query: str
    max_results: int = 5
    include_domains: tuple[str, ...] = ()
    exclude_domains: tuple[str, ...] = ()
    start_published_at: datetime | None = None
    end_published_at: datetime | None = None
    content_results: int = 3

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("search query must not be empty")
        if not 1 <= self.max_results <= 100:
            raise ValueError("max_results must be between 1 and 100")
        if not 0 <= self.content_results <= 100:
            raise ValueError("content_results must be between 0 and 100")
        for boundary in (self.start_published_at, self.end_published_at):
            if boundary is not None and boundary.tzinfo is None:
                raise ValueError("published date filters must include a timezone")
        if (
            self.start_published_at is not None
            and self.end_published_at is not None
            and self.start_published_at > self.end_published_at
        ):
            raise ValueError("start_published_at must be before end_published_at")


@dataclass(frozen=True, slots=True)
class SearchItem:
    """One normalized search result with optional retrieved evidence."""

    title: str
    url: str
    snippet: str = ""
    published_date: str | None = None
    content: str = ""
    highlights: tuple[str, ...] = ()
    provider: str = ""
    request_id: str | None = None
    cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class SearchBackendResult:
    """Structured backend response before Router filtering and deduplication."""

    items: tuple[SearchItem, ...]
    request_id: str | None = None
    cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class SearchProviderError:
    """Serializable error metadata safe for observations and logs."""

    provider: str
    status: WebStatus
    code: str
    retryable: bool = False


class SearchBackendError(RuntimeError):
    """Backend failure carrying a provider-neutral, secret-free error model."""

    def __init__(self, error: SearchProviderError) -> None:
        super().__init__(f"{error.provider}: {error.code} ({error.status})")
        self.error = error


@dataclass(frozen=True, slots=True)
class SearchAttempt:
    """One backend attempt emitted by SearchRouter for observability."""

    provider: str
    status: WebStatus
    latency_ms: float
    result_count: int = 0
    retry_count: int = 0
    fallback_reason: str | None = None
    error_code: str | None = None
    request_id: str | None = None
    cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """Final routed search response plus every attempted backend."""

    items: tuple[SearchItem, ...]
    attempts: tuple[SearchAttempt, ...] = ()
    final_provider: str | None = None


@runtime_checkable
class SearchBackend(Protocol):
    """Provider implementation consumed by SearchRouter."""

    name: str

    @property
    def available(self) -> bool:
        """Whether startup preflight found the backend usable."""
        ...

    async def search(self, request: SearchRequest) -> SearchBackendResult:
        """Execute one provider request or raise SearchBackendError."""
        ...


@runtime_checkable
class ExaSearchClient(Protocol):
    """App-injected transport boundary for Exa's Search endpoint."""

    async def search(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """POST a search payload and return decoded JSON."""
        ...


@dataclass(slots=True)
class SearchTrace:
    """Simple in-memory observer useful for tests and embedding runtimes."""

    attempts: list[SearchAttempt] = field(default_factory=list)

    def __call__(self, attempt: SearchAttempt) -> None:
        self.attempts.append(attempt)
