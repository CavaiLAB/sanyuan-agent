"""Standalone + boundary tests for SanYuanAgent Harness."""

import ast
import subprocess
import sys
from pathlib import Path

import sanyuan_agent.cli as cli_module
from sanyuan_agent.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_version_prints_version_and_exits_zero() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_console_script_prints_version() -> None:
    """Run the real console script via subprocess."""
    result = subprocess.run(
        ["sanyuan-agent", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "sanyuan-agent" in result.stdout
    assert "0.1.0" in result.stdout


def test_desktop_run_command_propagates_protocol_exit(
    monkeypatch,
) -> None:
    monkeypatch.setattr(cli_module, "run_guarded_desktop_process", lambda: 7)
    result = runner.invoke(app, ["desktop-run"])
    assert result.exit_code == 7


def test_no_imports_from_neuroagent_app() -> None:
    """Boundary check: harness must never import neuroagent_app."""
    harness_dir = Path(__file__).parent.parent / "sanyuan_agent"
    violations: list[str] = []

    for py_file in harness_dir.rglob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "neuroagent_app" or alias.name.startswith("neuroagent_app."):
                        violations.append(f"{py_file}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module == "neuroagent_app" or (
                    node.module and node.module.startswith("neuroagent_app.")
                ):
                    violations.append(f"{py_file}: from {node.module} import ...")

    assert not violations, f"Boundary violation(s): {violations}"


def test_memory_projection_public_api_imports_in_fresh_process() -> None:
    """The memory feature imports and renders from the harness wheel alone."""
    code = """
from dataclasses import dataclass
from sanyuan_agent.memory import MemoryLevel, render_memory_markdown

@dataclass
class Record:
    id: str = "m1"
    content: str = "standalone"
    scope: str = "private"
    level: MemoryLevel = MemoryLevel.PERSONAL
    project_id: str | None = None
    session_id: str | None = None
    created_at: str = "2026-01-01T00:00:00Z"

assert render_memory_markdown([Record()]).line_by_id["m1"] > 0
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
