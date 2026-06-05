"""SKILL.md parser and in-memory skill registry."""

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from .signing import SkillVerifier


@dataclass(frozen=True, slots=True)
class Skill:
    """A loaded skill definition.

    ``trust`` is ``builtin`` for first-party skills (trusted by provenance) or
    ``custom`` for user-uploaded ones. ``status`` is ``active`` (loadable) or
    ``pending`` (awaiting a valid signature — never injected into prompts).
    """

    name: str
    version: str
    description: str
    tools: list[str]
    prompt_template: str
    source_path: Path
    trigger_keywords: list[str] = field(default_factory=list)
    signature: str | None = None
    status: str = "active"
    trust: str = "builtin"


@dataclass(frozen=True, slots=True)
class SkillLoadResult:
    """Outcome of attempting to load a custom skill."""

    skill: Skill
    accepted: bool
    reason: str  # "signed" | "unsigned" | "bad_signature"


class SkillRegistry:
    """In-memory registry for skills."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        if skill.name in self._skills:
            raise ValueError(
                f"Skill {skill.name!r} is already registered "
                f"(from {self._skills[skill.name].source_path})"
            )
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise KeyError(f"Skill {name!r} not found") from exc

    def list(self) -> list[str]:
        return sorted(self._skills.keys())

    def __len__(self) -> int:
        return len(self._skills)

    def __iter__(self) -> Iterator[Skill]:
        return iter(self._skills.values())

    def match_trigger(self, text: str) -> Skill | None:
        """Return the first skill whose trigger_keywords match *text* (case-insensitive)."""
        text_lower = text.lower()
        for skill in self._skills.values():
            for kw in skill.trigger_keywords:
                if kw.lower() in text_lower:
                    return skill
        return None

    def load_prompt(self, skill: Skill) -> str:
        """Load the ``prompt.md`` file adjacent to a skill's SKILL.md."""
        prompt_path = skill.source_path.parent / "prompt.md"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return skill.prompt_template


def parse_skill(path: Path) -> Skill:
    """Parse a SKILL.md file with YAML frontmatter + markdown body."""
    text = path.read_text(encoding="utf-8")

    if not text.startswith("---"):
        raise ValueError(f"Missing YAML frontmatter in {path}")

    # Split on the second '---' (closing frontmatter delimiter)
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"Malformed frontmatter in {path}")

    front_yaml = parts[1].strip()
    body = parts[2].strip()

    meta = yaml.safe_load(front_yaml) or {}

    return Skill(
        name=meta.get("name", path.stem),
        version=str(meta.get("version", "0.0.0")),
        description=meta.get("description", ""),
        tools=list(meta.get("tools", [])),
        prompt_template=body,
        source_path=path,
        trigger_keywords=list(meta.get("trigger_keywords", [])),
        signature=meta.get("signature"),
    )


def load_custom_skill(
    path: Path,
    registry: SkillRegistry,
    verifier: SkillVerifier,
) -> SkillLoadResult:
    """Parse + verify a user-uploaded skill, registering it only if signed.

    A skill with a valid Ed25519 signature is registered as ``active``;
    unsigned or tampered skills are returned as ``pending`` and NOT registered,
    so they can never be injected into a prompt.
    """
    parsed = parse_skill(path)
    if parsed.signature and verifier.verify(
        parsed.name, parsed.version, parsed.prompt_template, parsed.signature
    ):
        skill = replace(parsed, trust="custom", status="active")
        registry.register(skill)
        return SkillLoadResult(skill, accepted=True, reason="signed")

    reason = "unsigned" if not parsed.signature else "bad_signature"
    pending = replace(parsed, trust="custom", status="pending")
    return SkillLoadResult(pending, accepted=False, reason=reason)


def load_custom_skills_dir(
    directory: Path,
    registry: SkillRegistry,
    verifier: SkillVerifier,
) -> list[SkillLoadResult]:
    """Load every ``*/SKILL.md`` under *directory* through signature verification."""
    results: list[SkillLoadResult] = []
    if not directory.exists():
        return results
    for subdir in sorted(d for d in directory.iterdir() if d.is_dir()):
        skill_file = subdir / "SKILL.md"
        if skill_file.exists():
            results.append(load_custom_skill(skill_file, registry, verifier))
    return results


def load_skills_dir(directory: Path, registry: SkillRegistry) -> None:
    """Scan *directory* for *.skill.md files and subdirectory SKILL.md files."""
    if not directory.exists():
        return

    # Direct *.skill.md files
    for path in sorted(directory.glob("*.skill.md")):
        registry.register(parse_skill(path))

    # Subdirectory SKILL.md files
    for subdir in sorted(d for d in directory.iterdir() if d.is_dir()):
        skill_file = subdir / "SKILL.md"
        if skill_file.exists():
            registry.register(parse_skill(skill_file))
