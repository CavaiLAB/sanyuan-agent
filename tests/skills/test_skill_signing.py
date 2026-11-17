"""Tests for Ed25519 custom-skill signing + the pending/active load gate."""

from pathlib import Path

import yaml
from sanyuan_agent.skills.registry import (
    SkillRegistry,
    load_custom_skill,
    parse_skill,
)
from sanyuan_agent.skills.signing import (
    SkillVerifier,
    canonical_payload,
    generate_keypair,
    sign,
    verify,
)

_NAME = "market-research"
_VERSION = "1.0.0"
_BODY = "# Market Research\n\nResearch a topic and summarise."


def _write_skill(dir_: Path, *, signature: str | None) -> Path:
    meta: dict[str, object] = {
        "name": _NAME,
        "version": _VERSION,
        "description": "Research things",
        "trigger_keywords": ["research"],
    }
    if signature is not None:
        meta["signature"] = signature
    content = "---\n" + yaml.safe_dump(meta, sort_keys=True) + "---\n" + _BODY
    skill_dir = dir_ / _NAME
    skill_dir.mkdir()
    path = skill_dir / "SKILL.md"
    path.write_text(content, encoding="utf-8")
    return path


class TestSignRoundtrip:
    def test_sign_then_verify(self) -> None:
        priv, pub = generate_keypair()
        sig = sign(priv, _NAME, _VERSION, _BODY)
        assert verify(pub, _NAME, _VERSION, _BODY, sig)

    def test_tampered_body_fails(self) -> None:
        priv, pub = generate_keypair()
        sig = sign(priv, _NAME, _VERSION, _BODY)
        assert not verify(pub, _NAME, _VERSION, _BODY + " tampered", sig)

    def test_wrong_key_fails(self) -> None:
        priv, _ = generate_keypair()
        _, other_pub = generate_keypair()
        sig = sign(priv, _NAME, _VERSION, _BODY)
        assert not verify(other_pub, _NAME, _VERSION, _BODY, sig)

    def test_canonical_payload_is_deterministic(self) -> None:
        assert canonical_payload(_NAME, _VERSION, _BODY) == canonical_payload(
            _NAME, _VERSION, _BODY
        )


class TestCustomSkillGate:
    def test_signed_skill_loads_active(self, tmp_path: Path) -> None:
        priv, pub = generate_keypair()
        sig = sign(priv, _NAME, _VERSION, _BODY)
        path = _write_skill(tmp_path, signature=sig)

        registry = SkillRegistry()
        result = load_custom_skill(path, registry, SkillVerifier(pub))

        assert result.accepted is True
        assert result.reason == "signed"
        assert result.skill.status == "active"
        assert result.skill.trust == "custom"
        assert _NAME in registry.list()

    def test_unsigned_skill_is_pending_and_unregistered(self, tmp_path: Path) -> None:
        _, pub = generate_keypair()
        path = _write_skill(tmp_path, signature=None)

        registry = SkillRegistry()
        result = load_custom_skill(path, registry, SkillVerifier(pub))

        assert result.accepted is False
        assert result.reason == "unsigned"
        assert result.skill.status == "pending"
        assert _NAME not in registry.list()

    def test_tampered_signature_is_rejected(self, tmp_path: Path) -> None:
        priv, pub = generate_keypair()
        sig = sign(priv, _NAME, _VERSION, _BODY)
        path = _write_skill(tmp_path, signature=sig)
        # Tamper the body on disk *after* signing.
        path.write_text(
            path.read_text(encoding="utf-8").replace("summarise", "exfiltrate"),
            encoding="utf-8",
        )

        registry = SkillRegistry()
        result = load_custom_skill(path, registry, SkillVerifier(pub))

        assert result.accepted is False
        assert result.reason == "bad_signature"
        assert _NAME not in registry.list()


class TestBuiltinUnaffected:
    def test_builtin_skills_load_without_signature(self) -> None:
        builtin = (
            Path(__file__).resolve().parents[2]
            / "packages/core/sanyuan_agent/skills/public/web-research/SKILL.md"
        )
        skill = parse_skill(builtin)
        # Built-in skills are trusted by provenance: active, no signature needed.
        assert skill.trust == "builtin"
        assert skill.status == "active"
        assert skill.signature is None
