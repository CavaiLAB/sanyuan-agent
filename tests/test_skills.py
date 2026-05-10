"""Tests for skill parsing and registry."""

from pathlib import Path

import pytest
from neuroagent.skills import SkillRegistry, load_skills_dir, parse_skill


class TestParseSkill:
    def test_parses_frontmatter_and_body(self, tmp_path: Path):
        skill_file = tmp_path / "test.skill.md"
        skill_file.write_text(
            "---\n"
            "name: bash-expert\n"
            "version: \"1.0.0\"\n"
            "description: Expert at shell scripting\n"
            "tools:\n"
            "  - bash\n"
            "  - read_file\n"
            "---\n"
            "You are a shell scripting expert.\n"
            "Use bash to solve problems efficiently.\n"
        )
        skill = parse_skill(skill_file)
        assert skill.name == "bash-expert"
        assert skill.version == "1.0.0"
        assert skill.description == "Expert at shell scripting"
        assert skill.tools == ["bash", "read_file"]
        assert "shell scripting expert" in skill.prompt_template
        assert skill.source_path == skill_file

    def test_defaults_when_frontmatter_sparse(self, tmp_path: Path):
        skill_file = tmp_path / "minimal.skill.md"
        skill_file.write_text(
            "---\n---\n"
            "Just a prompt.\n"
        )
        skill = parse_skill(skill_file)
        assert skill.name == "minimal.skill"
        assert skill.version == "0.0.0"
        assert skill.description == ""
        assert skill.tools == []
        assert skill.prompt_template == "Just a prompt."

    def test_missing_frontmatter_raises(self, tmp_path: Path):
        skill_file = tmp_path / "bad.skill.md"
        skill_file.write_text("No frontmatter here.")
        with pytest.raises(ValueError, match="frontmatter"):
            parse_skill(skill_file)


class TestSkillRegistry:
    def test_register_and_get(self):
        reg = SkillRegistry()
        skill = parse_skill(_make_skill_file("foo"))
        reg.register(skill)
        assert reg.get("foo").name == "foo"

    def test_list_returns_sorted_names(self):
        reg = SkillRegistry()
        reg.register(parse_skill(_make_skill_file("zebra")))
        reg.register(parse_skill(_make_skill_file("alpha")))
        assert reg.list() == ["alpha", "zebra"]

    def test_duplicate_name_raises_value_error(self):
        reg = SkillRegistry()
        skill1 = parse_skill(_make_skill_file("dup"))
        skill2 = parse_skill(_make_skill_file("dup"))
        reg.register(skill1)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(skill2)

    def test_get_missing_raises_key_error(self):
        reg = SkillRegistry()
        with pytest.raises(KeyError, match="not found"):
            reg.get("missing")

    def test_len_and_iter(self):
        reg = SkillRegistry()
        reg.register(parse_skill(_make_skill_file("a")))
        reg.register(parse_skill(_make_skill_file("b")))
        assert len(reg) == 2
        assert {s.name for s in reg} == {"a", "b"}


class TestLoadSkillsDir:
    def test_loads_skill_files(self, tmp_path: Path):
        (tmp_path / "a.skill.md").write_text("---\nname: skill-a\n---\nPrompt A")
        (tmp_path / "b.skill.md").write_text("---\nname: skill-b\n---\nPrompt B")
        reg = SkillRegistry()
        load_skills_dir(tmp_path, reg)
        assert reg.list() == ["skill-a", "skill-b"]

    def test_loads_subdirectory_skill_md(self, tmp_path: Path):
        sub = tmp_path / "python"
        sub.mkdir()
        (sub / "SKILL.md").write_text("---\nname: python-dev\n---\nPython prompt")
        reg = SkillRegistry()
        load_skills_dir(tmp_path, reg)
        assert reg.list() == ["python-dev"]

    def test_missing_dir_noop(self, tmp_path: Path):
        reg = SkillRegistry()
        load_skills_dir(tmp_path / "nonexistent", reg)
        assert len(reg) == 0


# -- helpers --

_counter = 0


def _make_skill_file(name: str) -> Path:
    global _counter
    import tempfile
    _counter += 1
    fd = tempfile.NamedTemporaryFile(mode="w", suffix=".skill.md", delete=False)
    fd.write(f"---\nname: {name}\n---\nPrompt for {name}.\n")
    fd.close()
    return Path(fd.name)
