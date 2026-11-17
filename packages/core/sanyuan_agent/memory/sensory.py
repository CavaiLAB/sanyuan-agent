"""Sensory memory — ring buffer of raw events."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from sanyuan_agent.memory.abc import IMemoryStore, InMemoryStore


@dataclass(frozen=True, slots=True)
class SensoryEvent:
    """A single raw sensory event."""

    kind: Literal["user", "tool_result", "observation"]
    content: str
    ts: float = field(default_factory=lambda: __import__("time").time())


class SensoryMemory:
    """Ring-buffer for recent raw sensory events."""

    _STORE_KEY = "_sensory_events"

    def __init__(
        self,
        *,
        maxlen: int = 20,
        store: IMemoryStore | None = None,
    ) -> None:
        self.maxlen = maxlen
        self._store = store or InMemoryStore()
        raw = self._store.load(self._STORE_KEY, [])
        self._events: deque[SensoryEvent] = deque(raw, maxlen=maxlen)

    def append(self, event: SensoryEvent) -> None:
        """Append an event to the buffer and persist."""
        self._events.append(event)
        self._store.save(self._STORE_KEY, list(self._events))

    def get_recent(self, n: int = 5) -> list[SensoryEvent]:
        """Return the *n* most recent events (newest first)."""
        return list(self._events)[-n:]

    def clear(self) -> None:
        """Clear the buffer and store."""
        self._events.clear()
        self._store.clear()
