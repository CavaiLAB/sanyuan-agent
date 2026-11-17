"""Split a document into overlapping character-window chunks.

Character windows are deterministic and dependency-free (no tokenizer). The
window slides by ``chunk_size - overlap`` so consecutive chunks share *overlap*
characters — enough context continuity for retrieval without a tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    """A contiguous slice of a source document."""

    index: int
    text: str
    start: int
    end: int


def chunk_text(text: str, *, chunk_size: int = 1024, overlap: int = 128) -> list[Chunk]:
    """Split *text* into overlapping chunks of at most *chunk_size* characters.

    Returns an empty list for blank text. Raises ``ValueError`` if *overlap* is
    not strictly smaller than *chunk_size* (otherwise the window can't advance).
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")
    if not text.strip():
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    start = 0
    index = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(Chunk(index=index, text=text[start:end], start=start, end=end))
        if end == n:
            break
        start += step
        index += 1
    return chunks
