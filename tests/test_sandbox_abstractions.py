"""Verify ISandboxProvider interface completeness.

Every method on the abstract base must raise NotImplementedError
when called on a concrete subclass that does not override it.
"""

from datetime import datetime

import pytest
from sanyuan_agent.sandbox import ISandboxProvider, SandboxResult, SandboxSession


class _MinimalProvider(ISandboxProvider):
    """Stub implementation — every method raises NotImplementedError."""

    async def create_session(self):
        raise NotImplementedError

    async def run_shell(self, session_id, cmd):
        raise NotImplementedError

    async def read_file(self, session_id, path):
        raise NotImplementedError

    async def write_file(self, session_id, path, content):
        raise NotImplementedError

    async def list_dir(self, session_id, path):
        raise NotImplementedError

    async def upload_file(self, session_id, local_path, remote_path):
        raise NotImplementedError

    async def close_session(self, session_id):
        raise NotImplementedError


@pytest.mark.asyncio
class TestSandboxProviderInterface:
    """Each of the 7 methods must raise NotImplementedError."""

    async def test_create_session(self):
        with pytest.raises(NotImplementedError):
            await _MinimalProvider().create_session()

    async def test_run_shell(self):
        with pytest.raises(NotImplementedError):
            await _MinimalProvider().run_shell("sid", "echo hi")

    async def test_read_file(self):
        with pytest.raises(NotImplementedError):
            await _MinimalProvider().read_file("sid", "/tmp/a.txt")

    async def test_write_file(self):
        with pytest.raises(NotImplementedError):
            await _MinimalProvider().write_file("sid", "/tmp/a.txt", "hello")

    async def test_list_dir(self):
        with pytest.raises(NotImplementedError):
            await _MinimalProvider().list_dir("sid", "/tmp")

    async def test_upload_file(self):
        with pytest.raises(NotImplementedError):
            await _MinimalProvider().upload_file("sid", "/host/a", "/remote/a")

    async def test_close_session(self):
        with pytest.raises(NotImplementedError):
            await _MinimalProvider().close_session("sid")


class TestSandboxDataTypes:
    """Dataclass immutability and field access."""

    def test_sandbox_session_fields(self):
        s = SandboxSession(
            id="sess-1",
            created_at=datetime(2026, 4, 22, 12, 0),
            workdir="/workspace",
        )
        assert s.id == "sess-1"
        assert s.workdir == "/workspace"

    def test_sandbox_result_defaults(self):
        r = SandboxResult(success=True)
        assert r.stdout == ""
        assert r.stderr == ""
        assert r.exit_code == 0

    def test_sandbox_result_full(self):
        r = SandboxResult(
            success=False,
            stdout="out",
            stderr="err",
            exit_code=1,
        )
        assert r.stdout == "out"
        assert r.exit_code == 1
