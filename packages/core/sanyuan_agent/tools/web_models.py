"""Shared provider-neutral Web Research primitives."""

from __future__ import annotations

from typing import Literal

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

__all__ = ["WebStatus"]
