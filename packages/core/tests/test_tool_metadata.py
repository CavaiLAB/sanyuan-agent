"""Metadata consumers: where ToolSpec's advisory fields finally do work.

`default_policy_from_specs` turns `policy_hint` into a default ToolPolicy;
`missing_required_secrets` is a startup preflight over `required_secrets`.
Both remain advisory — ToolPolicy / RiskGate are still the runtime authority.
"""

from __future__ import annotations

from neuroagent.tools.provider import (
    ToolSpec,
    default_policy_from_specs,
    missing_required_secrets,
)


def _spec(name: str, *, policy_hint=None, required_secrets=()) -> ToolSpec:
    return ToolSpec(
        name=name,
        callable=lambda: None,
        schema={"name": name, "parameters": {"type": "object", "properties": {}}},
        policy_hint=policy_hint,
        required_secrets=required_secrets,
    )


class TestDefaultPolicyFromSpecs:
    def test_hints_become_rules_else_default(self) -> None:
        specs = [
            _spec("web_search"),  # no hint
            _spec("bash", policy_hint="confirm"),
            _spec("danger", policy_hint="deny"),
        ]
        policy = default_policy_from_specs(specs, default="allow")
        assert policy.decide("web_search") == "allow"  # falls through to default
        assert policy.decide("bash") == "confirm"
        assert policy.decide("danger") == "deny"

    def test_unmatched_tool_uses_default(self) -> None:
        policy = default_policy_from_specs([_spec("x", policy_hint="confirm")], default="deny")
        assert policy.decide("not_listed") == "deny"


class TestMissingRequiredSecrets:
    def test_reports_only_missing_or_blank(self) -> None:
        specs = [
            _spec("img", required_secrets=("MODEL_IMAGE_NAME", "ARK_KEY")),
            _spec("free", required_secrets=()),
        ]
        env = {"MODEL_IMAGE_NAME": "x", "ARK_KEY": "  "}  # blank counts as missing
        report = missing_required_secrets(specs, env)
        assert report == {"img": ("ARK_KEY",)}

    def test_all_present_yields_empty(self) -> None:
        specs = [_spec("img", required_secrets=("A",))]
        assert missing_required_secrets(specs, {"A": "v"}) == {}
