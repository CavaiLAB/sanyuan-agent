"""SKILL.md parser and in-memory skill registry."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import yaml


@dataclass(frozen=True, slots=True)
class Skill:
    """A loaded skill definition."""

    name: str
    version: str
    description: str
    tools: list[str]
    prompt_template: str
    source_path: Path


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
    )


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
