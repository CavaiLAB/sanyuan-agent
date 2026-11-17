"""Structured exceptions for agent tool execution."""

from __future__ import annotations


class ToolNotFoundError(Exception):
    """The requested tool is not registered."""

    def __init__(
        self,
        name: str,
        available: list[str],
    ) -> None:
        self.name = name
        self.available = available
        super().__init__(
            f"Tool '{name}' not found. Available: {', '.join(available)}"
        )


class ToolArgsValidationError(Exception):
    """Tool arguments failed JSON-schema validation."""

    def __init__(
        self,
        name: str,
        errors: list[str],
    ) -> None:
        self.name = name
        self.errors = errors
        super().__init__(
            f"Arguments validation failed for '{name}': "
            + "; ".join(errors)
        )


class ToolExecutionError(Exception):
    """The tool raised an exception during execution."""

    def __init__(
        self,
        name: str,
        original: Exception,
        traceback_summary: str,
    ) -> None:
        self.name = name
        self.original = original
        self.traceback_summary = traceback_summary
        super().__init__(
            f"Execution error in '{name}': {original}"
        )


class ToolCircuitBreakerError(Exception):
    """Circuit breaker tripped for this tool."""

    def __init__(
        self,
        name: str,
        fail_count: int,
        threshold: int,
    ) -> None:
        self.name = name
        self.fail_count = fail_count
        self.threshold = threshold
        super().__init__(
            f"Circuit breaker open for '{name}' "
            f"({fail_count}/{threshold} consecutive failures)"
        )
