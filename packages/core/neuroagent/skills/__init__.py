"""Skill registry — parse SKILL.md files and manage loaded skills."""

from pathlib import Path

from neuroagent.skills.registry import Skill, SkillRegistry, load_skills_dir, parse_skill

__all__ = ["Skill", "SkillRegistry", "load_skills_dir", "parse_skill"]

# Auto-load built-in public skills on import
_PUBLIC_DIR = Path(__file__).parent / "public"


def load_public_skills(registry: SkillRegistry) -> None:
    """Load all built-in public skills from the ``skills/public/`` directory."""
    load_skills_dir(_PUBLIC_DIR, registry)
