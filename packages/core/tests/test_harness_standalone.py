"""Standalone + boundary tests for NeuroAgent Harness."""

import ast
from pathlib import Path

from neuroagent.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_version_prints_version_and_exits_zero() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_no_imports_from_neuroagent_app() -> None:
    """Boundary check: harness must never import neuroagent_app."""
    harness_dir = Path(__file__).parent.parent / "neuroagent"
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
