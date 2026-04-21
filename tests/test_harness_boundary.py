"""Boundary test: harness package must not import app package."""

import subprocess
import sys


class TestHarnessBoundary:
    """Ensure neuroagent (harness) does not depend on neuroagent_app (app)."""

    def test_harness_cannot_import_app(self):
        """Importing from neuroagent must not bring in neuroagent_app."""
        code = (
            "import neuroagent; "
            "import sys; "
            "assert 'neuroagent_app' not in sys.modules, "
            "f'neuroagent_app leaked: {list(sys.modules.keys())}'; "
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
        code = "import neuroagent_app; import neuroagent"
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
        (this verifies the package exists), but importing neuroagent must not
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
