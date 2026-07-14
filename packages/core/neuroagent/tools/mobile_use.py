"""Mobile-use tool provider (drive a mobile device/app) over an injected client.

VeADK-style ``mobile_use`` exposed as a :class:`ToolProvider` shell. Two
boundaries mirror the multimodal provider:

1. **No real device / app / desktop / agent-link / network in the harness.** The
   provider talks only to an injected :class:`MobileUseClient` (a Protocol). A
   real client (device farm / agent-link + credentials) is supplied later; tests
   pass a fake. harness imports nothing from ``neuroagent_app`` /
   any host-side package and reads no secret.
2. **Structured result, never a raw device handle or path.** The tool returns a
   :class:`MobileUseResult` shape (``status`` / ``summary`` / ``actions`` /
   ``app`` / ``session_id``) - a logical session id, not a device handle.

The tool carries provider metadata (``category="mobile"``,
``side_effect="external"``, ``policy_hint="confirm"``). As in PR1 that metadata
is descriptive: it informs policy building / UI / audit but never gates execution
on its own. This shell declares no ``required_secrets`` - secret/credential
declaration arrives with a real client.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from neuroagent.tools.provider import ToolSpec, spec_from_callable


@dataclass(frozen=True, slots=True)
class MobileUseResult:
    """A mobile_use run outcome - structured, never a raw device handle/path.

    *session_id* is a logical session label, not a device/connection handle.
    *actions* lists the high-level steps taken, in order.
    """

    status: str
    summary: str
    actions: tuple[str, ...] = ()
    app: str = ""
    session_id: str = ""


@runtime_checkable
class MobileUseClient(Protocol):
    """The device-facing surface the provider needs.

    ``run`` returns a :class:`MobileUseResult`. Real implementations drive a
    device via a farm / agent-link with credentials; tests pass a fake. Async -
    device interaction is I/O-bound.
    """

    async def run(
        self, *, instruction: str, app: str, session_id: str
    ) -> MobileUseResult: ...


class FakeMobileUseClient:
    """Deterministic in-process MobileUseClient - no device / network / secret.

    Echoes the request into a predictable :class:`MobileUseResult` so the provider
    can be exercised end-to-end without a device. The real client is later work.
    """

    async def run(
        self, *, instruction: str, app: str, session_id: str
    ) -> MobileUseResult:
        return MobileUseResult(
            status="completed",
            summary=f"simulated mobile_use: {instruction}",
            actions=("parse_instruction", "plan_steps", "simulate_execute"),
            app=app,
            session_id=session_id,
        )


def _result(result: MobileUseResult) -> dict[str, Any]:
    """Normalize a result into the tool's JSON-serializable result dict."""
    return dataclasses.asdict(result)


class MobileUseToolProvider:
    """Yields the single ``mobile_use`` tool bound to an injected *client*."""

    def __init__(self, client: MobileUseClient) -> None:
        self._client = client

    def provide(self) -> list[ToolSpec]:
        client = self._client

        async def mobile_use(
            instruction: str, app: str = "", session_id: str = "default"
        ) -> dict[str, Any]:
            """Drive a mobile device/app via a natural-language instruction.

            Returns a structured run result (status / summary / actions); never a
            raw device handle or path.
            """
            return _result(
                await client.run(instruction=instruction, app=app, session_id=session_id)
            )

        return [
            spec_from_callable(
                mobile_use,
                category="mobile",
                capabilities=("mobile", "operate"),
                side_effect="external",
                policy_hint="confirm",
            )
        ]
