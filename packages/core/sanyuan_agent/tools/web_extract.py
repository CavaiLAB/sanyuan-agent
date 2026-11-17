"""Pure content decoding, routing, extraction, and quality checks."""

from __future__ import annotations

import json
import re
from html import unescape
from xml.etree import ElementTree

from sanyuan_agent.tools.block_detect import detect_block
from sanyuan_agent.tools.web_models import WebStatus
from sanyuan_agent.tools.web_reader_models import FetchDocument, FetchRequest, HttpFetchResponse

_CHARSET_RE = re.compile(r"charset\s*=\s*[\"']?([^;\s\"']+)", re.I)
_TITLE_RE = re.compile(r"<title\b[^>]*>(.*?)</title\s*>", re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_LOGIN_RE = re.compile(r"\b(sign[ -]?in|log[ -]?in|authentication required)\b|登录|登入", re.I)
_CAPTCHA_RE = re.compile(r"captcha|人机验证|验证码|verify you are human", re.I)
_DOCUMENT_TYPES = frozenset(
    {
        "application/pdf",
        "application/msword",
        "application/vnd.ms-excel",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
)
_JSON_TYPES = frozenset({"application/json", "application/ld+json", "application/problem+json"})
_XML_TYPES = frozenset(
    {
        "application/atom+xml",
        "application/rss+xml",
        "application/xml",
        "text/xml",
    }
)
_HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})


def _media_type(value: str | None) -> str:
    return (value or "").split(";", 1)[0].strip().lower()


def _decode(raw: bytes, content_type: str) -> str:
    match = _CHARSET_RE.search(content_type)
    encodings = [match.group(1)] if match else []
    encodings.extend(["utf-8", "gb18030", "big5", "latin-1"])
    tried: set[str] = set()
    for encoding in encodings:
        normalized = encoding.lower()
        if normalized in tried:
            continue
        tried.add(normalized)
        try:
            return raw.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def _sniff_media_type(raw: bytes, declared: str) -> str:
    if declared:
        return declared
    prefix = raw[:512].lstrip().lower()
    if prefix.startswith(b"%pdf"):
        return "application/pdf"
    if prefix.startswith((b"{", b"[")):
        return "application/json"
    if prefix.startswith((b"<?xml", b"<rss", b"<feed")):
        return "application/xml"
    if b"<html" in prefix or b"<!doctype html" in prefix:
        return "text/html"
    return "text/plain"


def _html_title(decoded: str) -> str:
    match = _TITLE_RE.search(decoded)
    if not match:
        return ""
    return unescape(_TAG_RE.sub(" ", match.group(1))).strip()


def _extract_html(decoded: str, url: str) -> tuple[str, str]:
    from trafilatura import extract, html2txt

    title = _html_title(decoded)
    text = extract(
        decoded,
        url=url,
        fast=True,
        include_comments=False,
        include_tables=False,
        output_format="txt",
    )
    if not text or len(text.strip()) < 120:
        text = extract(
            decoded,
            url=url,
            favor_recall=True,
            include_comments=False,
            include_tables=True,
            output_format="txt",
        )
    if not text:
        text = html2txt(decoded)
    return title, (text or "").strip()


def _extract_json(decoded: str) -> tuple[str, str]:
    try:
        value = json.loads(decoded)
    except json.JSONDecodeError:
        return "JSON", decoded.strip()
    return "JSON", json.dumps(value, ensure_ascii=False, indent=2)


def _extract_xml(decoded: str) -> tuple[str, str]:
    try:
        root = ElementTree.fromstring(decoded)
    except ElementTree.ParseError:
        return "XML", decoded.strip()
    title = ""
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1].lower() == "title" and node.text:
            title = node.text.strip()
            break
    content = "\n".join(text.strip() for text in root.itertext() if text.strip())
    return title or root.tag.rsplit("}", 1)[-1], content


def _http_status(status_code: int) -> tuple[WebStatus, str | None]:
    if status_code in {401, 407}:
        return "login_required", "authentication_required"
    if status_code == 403:
        return "blocked", "http_403"
    if status_code in {408, 504}:
        return "timeout", f"http_{status_code}"
    if status_code == 429:
        return "rate_limited", "http_429"
    if status_code >= 500:
        return "provider_unavailable", f"http_{status_code}"
    if not 200 <= status_code < 300:
        return "unsupported_content", f"http_{status_code}"
    return "success", None


def extract_http_document(
    request: FetchRequest,
    response: HttpFetchResponse,
) -> FetchDocument:
    """Map one bounded HTTP response into a unified FetchDocument."""
    status, error_code = _http_status(response.status_code)
    declared = response.headers.get("content-type", "")
    media_type = _sniff_media_type(response.content, _media_type(declared))
    if status != "success":
        return FetchDocument(
            url=request.url,
            final_url=response.final_url,
            provider="http",
            status=status,
            content_type=media_type,
            status_code=response.status_code,
            bytes=len(response.content),
            retry_count=response.retry_count,
            error_code=error_code,
            block_reason=error_code if status in {"blocked", "login_required"} else None,
        )
    if media_type in _DOCUMENT_TYPES:
        return FetchDocument(
            url=request.url,
            final_url=response.final_url,
            provider="http",
            status="unsupported_content",
            content_type=media_type,
            status_code=response.status_code,
            bytes=len(response.content),
            retry_count=response.retry_count,
            error_code="document_requires_sandbox_converter",
        )

    decoded = _decode(response.content, declared)
    if media_type in _JSON_TYPES:
        title, content = _extract_json(decoded)
    elif media_type in _XML_TYPES:
        title, content = _extract_xml(decoded)
    elif media_type in _HTML_TYPES:
        title, content = _extract_html(decoded, response.final_url)
    elif media_type.startswith("text/"):
        title, content = "", decoded.strip()
    else:
        return FetchDocument(
            url=request.url,
            final_url=response.final_url,
            provider="http",
            status="unsupported_content",
            content_type=media_type,
            status_code=response.status_code,
            bytes=len(response.content),
            retry_count=response.retry_count,
            error_code="unsupported_media_type",
        )

    if _LOGIN_RE.search(f"{title}\n{content[:2000]}") and len(content) < 5000:
        return FetchDocument(
            url=request.url,
            final_url=response.final_url,
            provider="http",
            status="login_required",
            title=title,
            content_type=media_type,
            status_code=response.status_code,
            bytes=len(response.content),
            retry_count=response.retry_count,
            error_code="login_wall",
            block_reason="login_required",
        )
    blocked, block_reason = detect_block(
        url=response.final_url,
        title=title,
        content=content,
    )
    if blocked:
        block_status: WebStatus = "captcha" if _CAPTCHA_RE.search(
            f"{title}\n{content}\n{block_reason or ''}"
        ) else "blocked"
        return FetchDocument(
            url=request.url,
            final_url=response.final_url,
            provider="http",
            status=block_status,
            title=title,
            content_type=media_type,
            status_code=response.status_code,
            bytes=len(response.content),
            retry_count=response.retry_count,
            error_code="soft_block",
            block_reason=block_reason,
        )
    if len(content.strip()) < request.min_content_chars:
        return FetchDocument(
            url=request.url,
            final_url=response.final_url,
            provider="http",
            status="empty_content",
            title=title,
            content_type=media_type,
            status_code=response.status_code,
            bytes=len(response.content),
            retry_count=response.retry_count,
            error_code="content_below_quality_threshold",
        )
    return FetchDocument(
        url=request.url,
        final_url=response.final_url,
        provider="http",
        status="success",
        title=title,
        content=content,
        content_type=media_type,
        status_code=response.status_code,
        bytes=len(response.content),
        retry_count=response.retry_count,
    )


__all__ = ["extract_http_document"]
