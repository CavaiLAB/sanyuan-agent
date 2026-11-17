"""Tool Provider protocol: normalize tool declaration upstream of the registry.

A :class:`ToolSpec` is the single artifact every tool source yields — a name,
an executable callable, an OpenAI-style function schema, plus optional
non-executing metadata (category / capabilities / side-effect class / required
secrets / whether it emits artifacts / a policy hint). A :class:`ToolProvider`
is anything that yields a batch of specs.

This sits *above* :class:`~sanyuan_agent.tools.registry.ToolRegistry`: providers
produce specs, helpers register specs into the registry. It does not replace
the executor, the MCP bridge, ``ToolPolicy`` or ``RiskGate`` — those remain the
runtime authority. The metadata fields are descriptive only: they never gate
execution by themselves (a ``policy_hint`` is a *suggestion* for building a
``ToolPolicy``, not an enforcement point).

Backward compatibility: the existing ``build_*_tool`` builders and the
``ToolRegistry.register(name, callable, schema)`` path are untouched. The
provider path is purely additive — :func:`spec_from_callable` adapts any
existing callable (including MCP proxies carrying ``__tool_schema__``) into a
spec, so old builders can be wrapped without changing their public API.

Harness stays standalone: this module imports nothing from ``neuroagent_app``
or other host-side packages.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from sanyuan_agent.security.tool_policy import PolicyRule, ToolPolicy
from sanyuan_agent.tools.registry import ToolRegistry, callable_to_schema

# A policy hint mirrors ToolPolicy's action vocabulary, but is advisory only —
# it is consumed when *building* a policy, never when enforcing one.
PolicyHint = str  # "allow" | "deny" | "confirm"
_POLICY_HINTS: frozenset[str] = frozenset({"allow", "deny", "confirm"})


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A normalized, registry-ready tool declaration.

    *name*, *callable*, and *schema* are the executable core (what the registry
    needs). The rest is descriptive metadata that downstream code (policy
    builders, the front-end, audit tags) may read; none of it affects how the
    tool executes.

    - *category*: coarse grouping, e.g. ``"web"`` / ``"sandbox"`` / ``"artifact"``
      / ``"browser"`` / ``"multimodal"``.
    - *capabilities*: finer tags, e.g. ``("search",)`` or ``("image", "generate")``.
    - *side_effect*: ``"read"`` / ``"write"`` / ``"external"`` / ``"destructive"``.
    - *required_secrets*: env-var / secret-scope *names* the tool needs at
      runtime. Declaration only — harness never loads real secrets here.
    - *artifact_output*: True if the tool emits artifact/workspace references
      (vs. inline text). Multimodal tools must set this.
    - *policy_hint*: advisory ``ToolPolicy`` action a deployment *may* default
      to (e.g. external generators → ``"confirm"``).
    """

    name: str
    callable: Callable[..., Any] | Callable[..., Awaitable[Any]]
    schema: dict[str, Any]
    category: str = ""
    capabilities: tuple[str, ...] = ()
    side_effect: str = "read"
    required_secrets: tuple[str, ...] = ()
    artifact_output: bool = False
    policy_hint: PolicyHint | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ToolSpec.name must be non-empty")
        if not callable(self.callable):
            raise ValueError(f"ToolSpec.callable for {self.name!r} is not callable")
        if not isinstance(self.schema, dict):
            raise ValueError(f"ToolSpec.schema for {self.name!r} must be a dict")
        # The registry's executor validates arguments against schema["parameters"];
        # a non-dict here would crash at call time, so reject it now.
        params = self.schema.get("parameters")
        if params is not None and not isinstance(params, dict):
            raise ValueError(f"ToolSpec.schema['parameters'] for {self.name!r} must be a dict")
        if self.policy_hint is not None and self.policy_hint not in _POLICY_HINTS:
            raise ValueError(
                f"unknown policy_hint {self.policy_hint!r} for {self.name!r}; "
                f"expected one of {sorted(_POLICY_HINTS)}"
            )


@runtime_checkable
class ToolProvider(Protocol):
    """Anything that yields a batch of :class:`ToolSpec`.

    Synchronous on purpose: a provider assembles already-bound callables (it
    does not perform I/O). Discovery-based sources that *must* do async I/O —
    e.g. an MCP server — adapt their result into specs via
    :func:`specs_from_callables` after awaiting their own discovery, rather than
    making this protocol async.
    """

    def provide(self) -> list[ToolSpec]:
        """Return the specs this provider contributes."""
        ...


def spec_from_callable(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    schema: dict[str, Any] | None = None,
    category: str = "",
    capabilities: tuple[str, ...] = (),
    side_effect: str = "read",
    required_secrets: tuple[str, ...] = (),
    artifact_output: bool = False,
    policy_hint: PolicyHint | None = None,
    metadata: dict[str, Any] | None = None,
) -> ToolSpec:
    """Adapt an existing callable into a :class:`ToolSpec`.

    The schema is resolved in priority order, matching the agent builder's
    ``_to_tool_def`` so the provider path and the legacy
    ``build_simple_agent(tools=[callable])`` path stay equivalent:

    1. an explicit *schema* argument, then
    2. the callable's ``__tool_schema__`` (MCP proxies carry this), then
    3. reflection via :func:`~sanyuan_agent.tools.registry.callable_to_schema`.

    *name* defaults to the resolved schema name (which is the callable's
    ``__name__`` for reflected tools, or the qualified name for MCP proxies).
    """
    resolved = schema
    if resolved is None:
        carried = getattr(fn, "__tool_schema__", None)
        resolved = carried if isinstance(carried, dict) else callable_to_schema(fn)

    resolved_name = name or resolved.get("name") or getattr(fn, "__name__", "")
    return ToolSpec(
        name=resolved_name,
        callable=fn,
        schema=resolved,
        category=category,
        capabilities=capabilities,
        side_effect=side_effect,
        required_secrets=required_secrets,
        artifact_output=artifact_output,
        policy_hint=policy_hint,
        metadata=metadata or {},
    )


def specs_from_callables(
    fns: list[Callable[..., Any]],
    *,
    category: str = "",
    capabilities: tuple[str, ...] = (),
    side_effect: str = "read",
    required_secrets: tuple[str, ...] = (),
    artifact_output: bool = False,
    policy_hint: PolicyHint | None = None,
) -> list[ToolSpec]:
    """Adapt a list of callables into specs sharing the same metadata.

    Useful for builders that return several callables of one family — e.g. the
    browser or sandbox tool sets, or the result of ``make_mcp_tools``.
    """
    return [
        spec_from_callable(
            fn,
            category=category,
            capabilities=capabilities,
            side_effect=side_effect,
            required_secrets=required_secrets,
            artifact_output=artifact_output,
            policy_hint=policy_hint,
        )
        for fn in fns
    ]


def register_tool_specs(
    registry: ToolRegistry,
    specs: list[ToolSpec],
    *,
    overwrite: bool = False,
) -> list[str]:
    """Register each spec into *registry*; return the registered names.

    Raises :class:`ValueError` on a duplicate name within *registry* unless
    *overwrite* is True. (The registry's own ``register`` silently overwrites;
    this guard surfaces accidental name collisions across providers, which is
    the common real-world failure — e.g. two providers both exposing
    ``web_search``.)
    """
    registered: list[str] = []
    for spec in specs:
        if not overwrite and registry.has_tool(spec.name):
            raise ValueError(
                f"tool {spec.name!r} already registered; pass overwrite=True to replace it"
            )
        registry.register(spec.name, spec.callable, spec.schema)
        registered.append(spec.name)
    return registered


def register_provider(
    registry: ToolRegistry,
    provider: ToolProvider,
    *,
    overwrite: bool = False,
) -> list[str]:
    """Register every spec a *provider* yields into *registry*."""
    return register_tool_specs(registry, provider.provide(), overwrite=overwrite)


# ---------------------------------------------------------------------------
# Metadata consumers — where the descriptive fields finally do work.
#
# These read the advisory metadata (``policy_hint`` / ``required_secrets``) to
# *build* deployment defaults. They remain advisory: ``ToolPolicy`` / ``RiskGate``
# are still the runtime authority, and a deployment may adopt, edit, or ignore
# what these produce. The harness still reads no real secret here.
# ---------------------------------------------------------------------------


def default_policy_from_specs(
    specs: Iterable[ToolSpec],
    *,
    default: str = "allow",
) -> ToolPolicy:
    """Build a :class:`ToolPolicy` from each spec's ``policy_hint``.

    A spec carrying ``policy_hint="confirm"`` (external generators, sandbox /
    bash, file writes) yields a confirm rule on that exact tool name; ``"deny"``
    a deny rule; ``"allow"`` an explicit allow rule. Specs with no hint
    contribute nothing and fall through to *default*. Rules are emitted in spec
    order; since each pattern is an exact tool name they don't shadow each other,
    so order only matters relative to *default*.

    This is a sane starting policy a deployment may adopt or override — it does
    not change that ``ToolPolicy`` / ``RiskGate`` enforce at runtime.
    """
    rules = [
        PolicyRule(spec.name, spec.policy_hint)
        for spec in specs
        if spec.policy_hint is not None
    ]
    return ToolPolicy(rules, default=default)


def missing_required_secrets(
    specs: Iterable[ToolSpec],
    env: Mapping[str, str],
) -> dict[str, tuple[str, ...]]:
    """Return ``{tool_name: missing_secret_names}`` for under-configured specs.

    A startup preflight: surface a tool whose declared ``required_secrets`` are
    absent (or blank) in *env* *before* the model calls it and it crashes
    mid-run. The harness never reads real secrets — *env* is supplied by the
    caller (e.g. ``dict(os.environ)``), and only presence / non-emptiness is
    checked, never the value. A spec with no declared secrets never appears.
    """
    report: dict[str, tuple[str, ...]] = {}
    for spec in specs:
        missing = tuple(
            name for name in spec.required_secrets if not env.get(name, "").strip()
        )
        if missing:
            report[spec.name] = missing
    return report
