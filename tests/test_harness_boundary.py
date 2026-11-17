"""Boundary test: harness package must not import app package."""

import subprocess
import sys


class TestHarnessBoundary:
    """Ensure sanyuan_agent (harness) does not depend on neuroagent_app (app)."""

    def test_harness_cannot_import_app(self):
        """Importing from sanyuan_agent must not bring in neuroagent_app."""
        # Host-side packages: the source repo's service and media layers, plus their
        # renamed counterparts (tentative).
        code = (
            "import sanyuan_agent; "
            "import sys; "
            "hosts = ('neuroagent_app', 'neuroagent_media', 'sanyuan_agent_app', 'sanyuan_agent_media'); "
            "leaked = [m for m in hosts if m in sys.modules]; "
            "assert not leaked, f'host package leaked: {leaked}'; "
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"Harness imported app. stderr: {result.stderr}"
        )

    def test_app_can_import_harness(self):
        """App importing harness is expected and must succeed."""
        code = "import neuroagent_app; import sanyuan_agent"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"App failed to import harness. stderr: {result.stderr}"
        )

    def test_harness_direct_import_app_fails(self):
        """Direct import of neuroagent_app from a clean process should work
        (this verifies the package exists), but importing sanyuan_agent must not
        trigger it."""
        code = "import neuroagent_app"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"neuroagent_app package broken. stderr: {result.stderr}"
        )
