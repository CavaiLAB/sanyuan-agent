"""Memory store protocol — default in-memory implementation."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, Protocol


class IMemoryStore(Protocol):
    """Protocol for a key/value memory store backing agent memory."""

    def save(self, key: str, value: Any) -> None:
        """Persist *value* under *key*."""
        ...

    def load(self, key: str, default: Any = None) -> Any:
        """Load value under *key*; return *default* if missing."""
        ...

    def clear(self) -> None:
        """Remove all stored data."""
        ...


class InMemoryStore(IMemoryStore):
    """Simple dict-backed memory store."""

    def __init__(self) -> None:
        self._data: MutableMapping[str, Any] = {}

    def save(self, key: str, value: Any) -> None:
        self._data[key] = value

    def load(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def clear(self) -> None:
        self._data.clear()
