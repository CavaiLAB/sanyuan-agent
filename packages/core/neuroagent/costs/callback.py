"""LiteLLM success callback that enqueues token-usage records into UsageTracker."""

from __future__ import annotations

import logging
import time
from typing import Any

import litellm

from neuroagent.costs.tracker import LLMCallUsage, UsageTracker

logger = logging.getLogger(__name__)

# Module-level tracker reference — set via install_usage_callback().
_tracker: UsageTracker | None = None
_default_tenant_id: str = ""
_default_user_id: str = ""
_default_run_id: str = ""


def install_usage_callback(
    tracker: UsageTracker,
    *,
    default_tenant_id: str = "",
    default_user_id: str = "",
    default_run_id: str = "",
) -> None:
    """Register the LiteLLM success callback.

    Call once during application startup::

        tracker = UsageTracker(flush_callback=my_flush)
        install_usage_callback(tracker, default_tenant_id="t-001")
    """
    global _tracker, _default_tenant_id, _default_user_id, _default_run_id
    _tracker = tracker
    _default_tenant_id = default_tenant_id
    _default_user_id = default_user_id
    _default_run_id = default_run_id
    if _on_success not in litellm.success_callback:
        litellm.success_callback.append(_on_success)


def set_run_context(run_id: str, tenant_id: str, user_id: str = "") -> None:
    """Update the current run context (call per-request in the app layer)."""
    global _default_run_id, _default_tenant_id, _default_user_id
    _default_run_id = run_id
    _default_tenant_id = tenant_id
    _default_user_id = user_id


def _on_success(
    kwargs: dict[str, Any],
    completion_response: Any,
    start_time: Any,
    end_time: Any,
) -> None:
    """LiteLLM ``success_callback`` — enqueues a token-usage record.

    Signature matches ``litellm.success_callback`` convention.
    Runs synchronously in LiteLLM's thread; ``UsageTracker.enqueue`` is
    thread-safe.
    """
    if _tracker is None:
        return

    try:
        model = _safe_str(kwargs.get("model", "") or getattr(completion_response, "model", ""))

        # Token usage
        usage = getattr(completion_response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0

        # Provider inference from model string
        provider = _infer_provider(model)

        # Per-call context travels in litellm metadata (set by _call_llm);
        # module-level defaults are only a fallback for legacy callers.
        litellm_params = kwargs.get("litellm_params") or {}
        metadata = litellm_params.get("metadata") or {}

        record = LLMCallUsage(
            run_id=metadata.get("run_id") or kwargs.get("run_id", _default_run_id),
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            timestamp=time.time(),
            tenant_id=metadata.get("tenant_id")
            or kwargs.get("tenant_id", _default_tenant_id),
        )
        _tracker.enqueue(record)
    except Exception:
        logger.exception("usage tracking callback error")


def _safe_str(val: Any) -> str:
    if isinstance(val, str):
        return val
    return str(val) if val is not None else ""


def _infer_provider(model: str) -> str:
    """Infer provider from LiteLLM model string (e.g. 'openai/gpt-4')."""
    if "/" in model:
        return model.split("/", 1)[0]
    return "unknown"
