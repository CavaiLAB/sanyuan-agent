"""Tests for sandbox tool wrappers."""

import pytest
import tempfile
from pathlib import Path

from neuroagent.sandbox.local import LocalSandboxProvider
from neuroagent.tools import build_bash_tool, build_file_ops_tools


@pytest.mark.asyncio
class TestBashTool:
    async def test_echo(self):
        provider = LocalSandboxProvider()
        session = await provider.create_session()
        bash = build_bash_tool(provider, session)
        result = await bash("echo hello")
        assert result["stdout"].strip() == "hello"
        assert result["stderr"] == ""
        assert result["exit_code"] == 0

    async def test_failure(self):
        provider = LocalSandboxProvider()
        session = await provider.create_session()
        bash = build_bash_tool(provider, session)
        result = await bash("false")
        assert result["exit_code"] == 1
        assert result["stdout"] == ""
        assert result["stderr"] == ""


@pytest.mark.asyncio
class TestFileOpsTools:
    async def test_write_read_roundtrip(self):
        provider = LocalSandboxProvider()
        session = await provider.create_session()
        tools = build_file_ops_tools(provider, session)
        await tools["write_file"]("test.txt", "hello world")
        result = await tools["read_file"]("test.txt")
        assert isinstance(result.content, str)
        assert result.content == "hello world"

    async def test_list_dir(self):
        provider = LocalSandboxProvider()
        session = await provider.create_session()
        tools = build_file_ops_tools(provider, session)
        await tools["write_file"]("a.txt", "a")
        await tools["write_file"]("b.txt", "b")
        result = await tools["list_dir"](".")
        assert result.entries == ["a.txt", "b.txt"]

    async def test_upload_file(self):
        provider = LocalSandboxProvider()
        session = await provider.create_session()
        tools = build_file_ops_tools(provider, session)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".src", delete=False) as f:
            f.write("uploaded")
            src = f.name
        try:
            await tools["upload_file"](src, "uploaded.txt")
            result = await tools["read_file"]("uploaded.txt")
            assert result.content == "uploaded"
        finally:
            Path(src).unlink(missing_ok=True)
