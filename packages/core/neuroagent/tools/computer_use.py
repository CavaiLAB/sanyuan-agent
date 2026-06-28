"""Computer-use tool (drive a desktop computer via NL) over an injected client.

VeADK-style ``computer_use`` exposed as a :class:`ToolProvider` shell, the
desktop counterpart of ``mobile_use``. Two boundaries mirror the mobile / multi-
modal providers:

1. **No real desktop / VM / agent-link / network in the harness.** The provider
   talks only to an injected :class:`ComputerUseClient` (a Protocol). A real
   client (a cloud desktop / VNC bridge + credentials) is supplied later; tests
   pass a fake. harness imports nothing from ``neuroagent_app`` /
   ``neuroagent_sandbox_supervisor`` and reads no secret.
2. **Structured result, never a raw desktop handle or path.** The tool returns a
   :class:`ComputerUseResult` (``status`` / ``summary`` / ``actions`` /
   ``session_id``) — a logical session label, not a connection handle.

The tool carries provider metadata (``category="computer"``,
``side_effect="external"``, ``policy_hint="confirm"``): descriptive only, it
informs policy building / UI / audit but never gates execution on its own. This
shell declares no ``required_secrets`` — credential declaration arrives with a
real client.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from neuroagent.tools.provider import ToolSpec, spec_from_callable


@dataclass(frozen=True, slots=True)
class ComputerUseResult:
    """A computer_use run outcome — structured, never a raw desktop handle/path.

    *session_id* is a logical session label, not a VM / connection handle.
    *actions* lists the high-level steps taken, in order.
    """

    status: str
    summary: str
    actions: tuple[str, ...] = ()
    session_id: str = ""


@runtime_checkable
class ComputerUseClient(Protocol):
    """The desktop-facing surface the provider needs.

    ``run`` returns a :class:`ComputerUseResult`. Real implementations drive a
    desktop via a cloud-VM / VNC bridge with credentials; tests pass a fake.
    Async — desktop interaction is I/O-bound.
    """

    async def run(self, *, instruction: str, session_id: str) -> ComputerUseResult: ...


class FakeComputerUseClient:
    """Deterministic in-process ComputerUseClient — no desktop / network / secret.

    Echoes the request into a predictable :class:`ComputerUseResult` so the
    provider can be exercised end-to-end without a desktop. The real client is
    later work.
    """

    async def run(self, *, instruction: str, session_id: str) -> ComputerUseResult:
        return ComputerUseResult(
            status="completed",
            summary=f"simulated computer_use: {instruction}",
            actions=("parse_instruction", "plan_steps", "simulate_execute"),
            session_id=session_id,
        )


def _result(result: ComputerUseResult) -> dict[str, Any]:
    """Normalize a result into the tool's JSON-serializable result dict."""
    return dataclasses.asdict(result)


class ComputerUseToolProvider:
    """Yields the single ``computer_use`` tool bound to an injected *client*."""

    def __init__(self, client: ComputerUseClient) -> None:
        self._client = client

    def provide(self) -> list[ToolSpec]:
        client = self._client

        async def computer_use(instruction: str, session_id: str = "default") -> dict[str, Any]:
            """Drive a desktop computer via a natural-language instruction.

            Args:
                instruction: What to do on the computer, in natural language.
                session_id: Logical session label to reuse across calls.

            Returns a structured run result (status / summary / actions); never a
            raw desktop handle or path.
            """
            return _result(await client.run(instruction=instruction, session_id=session_id))

        return [
            spec_from_callable(
                computer_use,
                category="computer",
                capabilities=("computer", "operate"),
                side_effect="external",
                policy_hint="confirm",
            )
        ]
