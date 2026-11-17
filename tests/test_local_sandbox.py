"""Full lifecycle tests for LocalSandboxProvider."""

import tempfile
from pathlib import Path

import pytest
from neuroagent.sandbox.local import LocalSandboxProvider


@pytest.mark.asyncio
class TestLocalSandboxLifecycle:
    async def test_create_session_returns_handle(self):
        provider = LocalSandboxProvider()
        session = await provider.create_session()
        assert session.id
        assert Path(session.workdir).exists()

    async def test_run_shell_echo(self):
        provider = LocalSandboxProvider()
        session = await provider.create_session()
        result = await provider.run_shell(session.id, "echo hello")
        assert result.success is True
        assert result.stdout.strip() == "hello"
        assert result.stderr == ""
        assert result.exit_code == 0

    async def test_write_and_read_file(self):
        provider = LocalSandboxProvider()
        session = await provider.create_session()
        await provider.write_file(session.id, "test.txt", "hello world")
        content = await provider.read_file(session.id, "test.txt")
        assert content == "hello world"

    async def test_list_dir(self):
        provider = LocalSandboxProvider()
        session = await provider.create_session()
        await provider.write_file(session.id, "a.txt", "a")
        await provider.write_file(session.id, "b.txt", "b")
        entries = await provider.list_dir(session.id, ".")
        assert entries == ["a.txt", "b.txt"]

    async def test_upload_file(self):
        provider = LocalSandboxProvider()
        session = await provider.create_session()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".src", delete=False) as f:
            f.write("uploaded content")
            src = f.name
        try:
            await provider.upload_file(session.id, src, "uploaded.txt")
            content = await provider.read_file(session.id, "uploaded.txt")
            assert content == "uploaded content"
        finally:
            Path(src).unlink(missing_ok=True)

    async def test_close_removes_workdir(self):
        provider = LocalSandboxProvider()
        session = await provider.create_session()
        workdir = Path(session.workdir)
        await provider.close_session(session.id)
        assert not workdir.exists()

    async def test_run_shell_failure(self):
        provider = LocalSandboxProvider()
        session = await provider.create_session()
        result = await provider.run_shell(session.id, "false")
        assert result.success is False
        assert result.exit_code == 1

    async def test_write_nested_path(self):
        provider = LocalSandboxProvider()
        session = await provider.create_session()
        await provider.write_file(session.id, "sub/dir/file.txt", "deep")
        assert await provider.read_file(session.id, "sub/dir/file.txt") == "deep"

    async def test_list_nested_dir(self):
        provider = LocalSandboxProvider()
        session = await provider.create_session()
        await provider.write_file(session.id, "sub/x.txt", "x")
        entries = await provider.list_dir(session.id, "sub")
        assert entries == ["x.txt"]

    async def test_session_not_found_raises(self):
        provider = LocalSandboxProvider()
        with pytest.raises(ValueError, match="not found"):
            await provider.read_file("nonexistent-id", "foo.txt")

    async def test_run_shell_uses_shlex_no_shell_injection(self):
        provider = LocalSandboxProvider()
        session = await provider.create_session()
        # `echo` with a trailing semicolon should be treated as part of arg, not command sep
        result = await provider.run_shell(session.id, "echo hello; echo world")
        # shlex.split turns this into one arg "hello;" and another "world" for echo
        # Actually: echo "hello;" "world" → stdout = "hello; world\n"
        assert "hello;" in result.stdout
