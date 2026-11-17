"""Stable-source adapters selected before generic paid/browser fallbacks."""

from __future__ import annotations

from urllib.parse import urlsplit

from sanyuan_agent.tools.web_reader_models import (
    DocumentConverterClient,
    FetchDocument,
    FetchRequest,
)

_DOCUMENT_EXTENSIONS = (
    ".pdf",
    ".docx",
    ".xls",
    ".xlsx",
    ".pptx",
)
_DOCUMENT_MEDIA_TYPES = frozenset(
    {
        "application/pdf",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
)


class DocumentSourceAdapter:
    """Convert bounded untrusted documents in a dedicated sandbox image."""

    name = "document"
    third_party = False

    def __init__(self, converter: DocumentConverterClient | None) -> None:
        self._converter = converter

    @property
    def available(self) -> bool:
        return self._converter is not None

    def supports(self, request: FetchRequest, previous: FetchDocument | None) -> bool:
        path = urlsplit(request.url).path.lower()
        if path.endswith(_DOCUMENT_EXTENSIONS):
            return True
        if previous is None:
            return False
        media_type = previous.content_type.split(";", 1)[0].strip().lower()
        return previous.status == "unsupported_content" and media_type in _DOCUMENT_MEDIA_TYPES

    async def read(self, request: FetchRequest, *, deadline: float) -> FetchDocument:
        if self._converter is None:
            return FetchDocument(
                url=request.url,
                final_url=request.url,
                provider=self.name,
                status="provider_unavailable",
                error_code="document_converter_unavailable",
            )
        return await self._converter.convert(request, deadline=deadline)


__all__ = ["DocumentSourceAdapter"]
