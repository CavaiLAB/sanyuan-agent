"""``callable_to_schema`` reflection: precise JSON types, enums, defaults, docs.

This module uses ``from __future__ import annotations`` on purpose, so every
parameter annotation below is a *string* at runtime — exactly the condition the
old reflector collapsed to ``"string"``. ``get_type_hints`` must resolve them
back to real types. These tests pin the enriched reflector's contract.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from neuroagent.agents.simple import _to_tool_def
from neuroagent.tools.registry import callable_to_schema


def _props(fn: Any) -> dict[str, Any]:
    return callable_to_schema(fn)["parameters"]["properties"]


def _required(fn: Any) -> list[str]:
    return callable_to_schema(fn)["parameters"]["required"]


class TestScalarTypes:
    def test_str_int_bool_float_resolved_through_string_annotations(self) -> None:
        def tool(a: str, b: int, c: bool, d: float) -> str:
            return ""

        props = _props(tool)
        assert props["a"] == {"type": "string"}
        assert props["b"] == {"type": "integer"}
        assert props["c"] == {"type": "boolean"}
        assert props["d"] == {"type": "number"}

    def test_unannotated_param_defaults_to_string(self) -> None:
        def tool(a) -> str:  # type: ignore[no-untyped-def]
            return ""

        assert _props(tool)["a"] == {"type": "string"}


class TestContainers:
    def test_list_of_str(self) -> None:
        def tool(items: list[str]) -> str:
            return ""

        assert _props(tool)["items"] == {"type": "array", "items": {"type": "string"}}

    def test_bare_list(self) -> None:
        def tool(items: list) -> str:
            return ""

        assert _props(tool)["items"] == {"type": "array", "items": {"type": "string"}}

    def test_dict_is_object(self) -> None:
        def tool(mapping: dict) -> str:
            return ""

        assert _props(tool)["mapping"] == {"type": "object"}


class TestOptionalAndUnion:
    def test_optional_unwraps_to_inner_type(self) -> None:
        # typing.Optional on purpose: exercises the get_origin(...) is Union arm,
        # distinct from the PEP 604 ``int | None`` (UnionType) test below.
        def tool(x: Optional[int]) -> str:  # noqa: UP045
            return ""

        assert _props(tool)["x"] == {"type": "integer"}

    def test_pep604_optional_with_none_default(self) -> None:
        def tool(x: int | None = None) -> str:
            return ""

        prop = _props(tool)["x"]
        assert prop["type"] == "integer"
        assert prop["default"] is None

    def test_heterogeneous_union_falls_back_to_string(self) -> None:
        def tool(x: int | str) -> str:
            return ""

        assert _props(tool)["x"] == {"type": "string"}


class TestLiteralEnum:
    def test_literal_strings_become_enum(self) -> None:
        def tool(mode: Literal["fast", "slow"] = "fast") -> str:
            return ""

        prop = _props(tool)["mode"]
        assert prop["enum"] == ["fast", "slow"]
        assert prop["type"] == "string"
        assert prop["default"] == "fast"

    def test_literal_ints_infer_integer_type(self) -> None:
        def tool(n: Literal[1, 2, 3]) -> str:
            return ""

        prop = _props(tool)["n"]
        assert prop["enum"] == [1, 2, 3]
        assert prop["type"] == "integer"


class TestDefaultsAndRequired:
    def test_required_lists_only_params_without_defaults(self) -> None:
        def tool(a: str, b: str = "x", c: int = 3) -> str:
            return ""

        assert _required(tool) == ["a"]
        assert _props(tool)["b"] == {"type": "string", "default": "x"}
        assert _props(tool)["c"] == {"type": "integer", "default": 3}


class TestSkips:
    def test_self_and_var_args_skipped(self) -> None:
        class Holder:
            def tool(self, a: str, *args: str, **kwargs: str) -> str:
                return ""

        assert set(_props(Holder.tool)) == {"a"}


class TestDocstringArgs:
    def test_google_args_section_populates_descriptions(self) -> None:
        def tool(query: str, limit: int = 5) -> str:
            """Search things.

            Args:
                query: The text to search for.
                limit: Max results to return.
            """
            return ""

        props = _props(tool)
        assert props["query"]["description"] == "The text to search for."
        assert props["limit"]["description"] == "Max results to return."
        assert props["limit"]["default"] == 5

    def test_no_args_section_yields_no_descriptions(self) -> None:
        def tool(query: str) -> str:
            """Just a summary, no Args block."""
            return ""

        assert "description" not in _props(tool)["query"]


class TestToToolDefParity:
    def test_to_tool_def_delegates_to_shared_reflector(self) -> None:
        def tool(a: int, b: bool = False) -> str:
            return ""

        assert _to_tool_def(tool) == {
            "type": "function",
            "function": callable_to_schema(tool),
        }

    def test_explicit_tool_schema_used_verbatim(self) -> None:
        async def proxy(**kwargs: Any) -> str:
            return ""

        schema = {"name": "x", "description": "d", "parameters": {"type": "object"}}
        proxy.__tool_schema__ = schema  # type: ignore[attr-defined]
        assert _to_tool_def(proxy) == {"type": "function", "function": schema}
