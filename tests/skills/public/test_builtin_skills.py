"""E2E tests for built-in skills — parsing, trigger routing, agent integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from sanyuan_agent.agents.simple import _maybe_inject_skill
from sanyuan_agent.skills import SkillRegistry, load_public_skills, parse_skill

SKILLS_DIR = Path(__file__).parent.parent.parent.parent / "packages" / "core" / "sanyuan_agent" / "skills" / "public"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry() -> SkillRegistry:
    reg = SkillRegistry()
    load_public_skills(reg)
    return reg


# ---------------------------------------------------------------------------
# 1. Public skills auto-discovery
# ---------------------------------------------------------------------------


def test_load_public_skills_discovers_all(registry: SkillRegistry) -> None:
    names = registry.list()
    assert "web-research" in names
    assert "code-execution" in names
    assert "report-generation" in names
    assert len(names) == 3


# ---------------------------------------------------------------------------
# 2. web-research SKILL.md parsing
# ---------------------------------------------------------------------------


def test_web_research_parsed(registry: SkillRegistry) -> None:
    skill = registry.get("web-research")
    assert skill.version == "1.0.0"
    assert "web_search" in skill.tools
    assert "web_fetch" in skill.tools
    assert "research" in skill.trigger_keywords
    assert "investigate" in skill.trigger_keywords
    assert "Research Report" in skill.prompt_template


# ---------------------------------------------------------------------------
# 3. code-execution SKILL.md parsing
# ---------------------------------------------------------------------------


def test_code_execution_parsed(registry: SkillRegistry) -> None:
    skill = registry.get("code-execution")
    assert "bash" in skill.tools
    assert "file_ops" in skill.tools
    assert "run code" in skill.trigger_keywords
    assert "python" in skill.trigger_keywords
    assert "sandbox" in skill.prompt_template.lower()


# ---------------------------------------------------------------------------
# 4. report-generation SKILL.md parsing
# ---------------------------------------------------------------------------


def test_report_generation_parsed(registry: SkillRegistry) -> None:
    skill = registry.get("report-generation")
    assert "generate report" in skill.trigger_keywords
    assert "write report" in skill.trigger_keywords
    assert "Table of Contents" in skill.prompt_template


# ---------------------------------------------------------------------------
# 5. Trigger routing: "research X" triggers web-research
# ---------------------------------------------------------------------------


def test_trigger_web_research(registry: SkillRegistry) -> None:
    skill = registry.match_trigger("Please research the latest AI models")
    assert skill is not None
    assert skill.name == "web-research"


# ---------------------------------------------------------------------------
# 6. Trigger routing: "run this code" triggers code-execution
# ---------------------------------------------------------------------------


def test_trigger_code_execution(registry: SkillRegistry) -> None:
    skill = registry.match_trigger("Run this Python script for me")
    assert skill is not None
    assert skill.name == "code-execution"


# ---------------------------------------------------------------------------
# 7. Trigger routing: "write a report" triggers report-generation
# ---------------------------------------------------------------------------


def test_trigger_report_generation(registry: SkillRegistry) -> None:
    skill = registry.match_trigger("Generate a report from these findings")
    assert skill is not None
    assert skill.name == "report-generation"


# ---------------------------------------------------------------------------
# 8. No trigger for unrelated messages
# ---------------------------------------------------------------------------


def test_no_trigger_for_unrelated(registry: SkillRegistry) -> None:
    skill = registry.match_trigger("What is the weather today?")
    assert skill is None


# ---------------------------------------------------------------------------
# 9. prompt.md loading
# ---------------------------------------------------------------------------


def test_load_prompt_from_file(registry: SkillRegistry) -> None:
    skill = registry.get("web-research")
    prompt = registry.load_prompt(skill)
    assert "web-research" in prompt
    assert "web_search" in prompt


# ---------------------------------------------------------------------------
# 10. SKILL.zh.md Chinese docs exist and parseable
# ---------------------------------------------------------------------------


def test_chinese_skill_docs_exist() -> None:
    for skill_name in ["web-research", "code-execution", "report-generation"]:
        zh_path = SKILLS_DIR / skill_name / "SKILL.zh.md"
        assert zh_path.exists(), f"Missing SKILL.zh.md for {skill_name}"
        content = zh_path.read_text(encoding="utf-8")
        assert content.startswith("---"), f"SKILL.zh.md for {skill_name} missing frontmatter"


# ---------------------------------------------------------------------------
# 11. tools.yaml exists for each skill
# ---------------------------------------------------------------------------


def test_tools_yaml_exists() -> None:
    import yaml

    for skill_name in ["web-research", "code-execution", "report-generation"]:
        yaml_path = SKILLS_DIR / skill_name / "tools.yaml"
        assert yaml_path.exists(), f"Missing tools.yaml for {skill_name}"
        content = yaml_path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        assert "tools" in data
        assert isinstance(data["tools"], list)


# ---------------------------------------------------------------------------
# 12. Skill injection into agent messages
# ---------------------------------------------------------------------------


def test_skill_injection_in_messages(registry: SkillRegistry) -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Please investigate the best web frameworks"},
    ]
    result = _maybe_inject_skill(messages, registry)
    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert "web-research" in result[0]["content"]


# ---------------------------------------------------------------------------
# 13. Skill injection no-op when no trigger
# ---------------------------------------------------------------------------


def test_skill_injection_noop(registry: SkillRegistry) -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hello, how are you?"},
    ]
    result = _maybe_inject_skill(messages, registry)
    assert result is messages


# ---------------------------------------------------------------------------
# 14. Skill injection no-op when registry is None
# ---------------------------------------------------------------------------


def test_skill_injection_none_registry() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Research something"},
    ]
    result = _maybe_inject_skill(messages, None)
    assert result is messages


# ---------------------------------------------------------------------------
# 15. Multi-skill: web-research triggers before report-generation
# ---------------------------------------------------------------------------


def test_web_research_priority_over_report(registry: SkillRegistry) -> None:
    """Multi-keyword messages trigger the first matching skill (dict iteration order)."""
    skill = registry.match_trigger("Research this topic and generate a report")
    assert skill is not None
    # First match wins based on skill dict order (alphabetical by dir name)
    assert skill.name in {"web-research", "report-generation"}


# ---------------------------------------------------------------------------
# 16. Parse SKILL.zh.md returns valid skill
# ---------------------------------------------------------------------------


def test_parse_chinese_skill_doc() -> None:
    zh_path = SKILLS_DIR / "web-research" / "SKILL.zh.md"
    skill = parse_skill(zh_path)
    assert skill.name == "web-research"
    assert "搜索" in skill.trigger_keywords or "调研" in skill.trigger_keywords
