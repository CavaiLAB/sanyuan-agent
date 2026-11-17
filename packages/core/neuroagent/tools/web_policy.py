"""Fail-closed URL classification shared by HTTP and third-party readers."""

from __future__ import annotations

import ipaddress
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from neuroagent.tools.web_models import WebStatus

_PRIVATE_HOST_SUFFIXES = (".internal", ".local", ".localhost", ".home", ".lan")
_SENSITIVE_QUERY_NAMES = frozenset(
    {
        "access_token",
        "apikey",
        "api_key",
        "auth",
        "authorization",
        "key",
        "ossaccesskeyid",
        "password",
        "secret",
        "sig",
        "signature",
        "token",
        "x-amz-credential",
        "x-amz-security-token",
        "x-amz-signature",
    }
)


@runtime_checkable
class HostResolver(Protocol):
    """Resolve one hostname into concrete IP strings."""

    async def resolve(self, host: str) -> Sequence[str]: ...


@dataclass(frozen=True, slots=True)
class URLClassification:
    """Stable policy verdict safe for Router observations."""

    allowed: bool
    status: WebStatus
    reason: str
    redacted_url: str
    resolved_ips: tuple[str, ...] = ()


def is_public_ip(value: str) -> bool:
    """Return True only for globally routable IPv4/IPv6 addresses."""
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return address.is_global


def safe_log_url(url: str) -> str:
    """Strip query, fragment, and userinfo before a URL reaches logs/traces."""
    try:
        parsed = urlsplit(url.strip())
        host = parsed.hostname or ""
        port = parsed.port
    except ValueError:
        return "<invalid-url>"
    netloc = host.lower()
    if port:
        netloc = f"{netloc}:{port}"
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", "", ""))


class URLClassifier:
    """Classify destinations before any HTTP or vendor request is attempted."""

    def __init__(
        self,
        resolver: HostResolver | None,
        *,
        private_domains: Sequence[str] = (),
    ) -> None:
        self._resolver = resolver
        self._private_domains = tuple(
            domain.strip().lower().lstrip(".")
            for domain in private_domains
            if domain.strip()
        )

    async def classify(self, url: str) -> URLClassification:
        redacted = safe_log_url(url)
        try:
            parsed = urlsplit(url.strip())
            host = (parsed.hostname or "").lower().rstrip(".")
            port = parsed.port
        except ValueError:
            return self._deny("invalid_url", redacted)
        if parsed.scheme.lower() not in {"http", "https"}:
            return self._deny("unsupported_scheme", redacted)
        if not host:
            return self._deny("missing_host", redacted)
        if parsed.username is not None or parsed.password is not None:
            return self._deny("url_userinfo_forbidden", redacted)
        if port is not None and not 1 <= port <= 65535:
            return self._deny("invalid_port", redacted)
        if host == "localhost" or host.endswith(_PRIVATE_HOST_SUFFIXES):
            return self._deny("private_hostname", redacted)
        if any(host == domain or host.endswith(f".{domain}") for domain in self._private_domains):
            return self._deny("tenant_private_domain", redacted)

        query_names = {name.casefold() for name, _ in parse_qsl(parsed.query, keep_blank_values=True)}
        if query_names & _SENSITIVE_QUERY_NAMES or any(
            name.startswith("x-amz-") for name in query_names
        ):
            return self._deny("credential_query_forbidden", redacted)

        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            literal = None
        if literal is not None:
            if not literal.is_global:
                return self._deny("non_public_ip", redacted)
            return URLClassification(
                allowed=True,
                status="success",
                reason="public_literal_ip",
                redacted_url=redacted,
                resolved_ips=(str(literal),),
            )

        if self._resolver is None:
            return self._deny("dns_resolver_unavailable", redacted)
        try:
            resolved = tuple(sorted(set(await self._resolver.resolve(host))))
        except Exception:
            return self._deny("dns_resolution_failed", redacted)
        if not resolved:
            return self._deny("dns_empty", redacted)
        if not all(is_public_ip(value) for value in resolved):
            return self._deny("dns_non_public_ip", redacted, resolved)
        return URLClassification(
            allowed=True,
            status="success",
            reason="public_hostname",
            redacted_url=redacted,
            resolved_ips=resolved,
        )

    @staticmethod
    def _deny(
        reason: str,
        redacted_url: str,
        resolved_ips: tuple[str, ...] = (),
    ) -> URLClassification:
        return URLClassification(
            allowed=False,
            status="policy_denied",
            reason=reason,
            redacted_url=redacted_url,
            resolved_ips=resolved_ips,
        )


__all__ = [
    "HostResolver",
    "URLClassification",
    "URLClassifier",
    "is_public_ip",
    "safe_log_url",
]
