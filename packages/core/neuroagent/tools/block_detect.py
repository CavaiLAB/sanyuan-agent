"""Conservative anti-bot / block-page detection shared by the web tools.

A fetch/navigate can return HTTP 200 yet land on a soft-block page (a redirect
to ``.../forbidden``, a captcha / 人机验证 wall, a Cloudflare interstitial).
``raise_for_status`` is happy, so without this the runtime reports a green
success for an access that was actually denied. :func:`detect_block` decides —
conservatively — whether a tool result is such a page.

Conservatism rules (kept in lockstep with ``apps/web/src/lib/toolOutcome.ts``):

* **Strong vendor phrases** are matched against the page TITLE + URL only — never
  the article body — so a page that merely *discusses* captchas isn't flagged.
* **URL patterns** are anchored to a whole path segment, so a real slug like
  ``/forbidden-fruit`` or ``/captcha-guide`` does not trip them.
* **Generic verify/deny words** are *weak*: they only count when ≥2 co-occur, or
  one co-occurs with a very short body. Block pages are tiny; an article that
  discusses anti-bot mechanisms has a long body, which is never scanned.
* **Vendor fingerprints** distinctive enough to never occur in prose
  (``px-captcha``, ``_incapsula_resource``, ``/cdn-cgi/challenge-platform`` …)
  are decisive on their own.
"""

from __future__ import annotations

import re

# Final-URL paths a site redirects to when it blocks a bot. Anchored to a whole
# path segment so legit slugs (/forbidden-fruit, /captcha-guide) don't match.
_URL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"/(forbidden|blocked|punish|captcha)(?:[/?#]|$)", re.I),
    re.compile(r"/sorry/", re.I),  # Google "unusual traffic" interstitial
    re.compile(r"/sec[-_]?check(?:[/?#]|$)", re.I),
    re.compile(r"/anti[-_]?(robot|spider|crawler)(?:[/?#]|$)", re.I),
    re.compile(r"/unusual[_-]?traffic(?:[/?#]|$)", re.I),
)

# Strong, vendor-specific block-page markers. Matched against TITLE + URL only.
_STRONG_PHRASES: tuple[str, ...] = (
    # —— 中文风控/验证页标题 ——
    "访问已被拦截",
    "访问被拦截",
    "访问已被拒绝",
    "访问被拒绝",
    "拒绝访问",
    "您的访问出现异常",
    "访问出现异常",
    "百度安全验证",  # 百度
    "腾讯防水墙",  # 腾讯
    "滑动验证",
    "滑块验证",
    # —— Cloudflare / 通用反爬标题 ——
    "just a moment",  # Cloudflare challenge title
    "attention required",  # Cloudflare block title
    "checking your browser",  # Cloudflare / DDoS-Guard
    "enable javascript and cookies to continue",
    "our systems have detected unusual traffic",  # Google's exact wording
    "i'm not a robot",  # reCAPTCHA anchor
)

# Fingerprints unique enough to be decisive wherever they appear (asset paths,
# widget ids, vendor challenge hosts — never ordinary prose).
_UNIQUE_FINGERPRINTS: tuple[str, ...] = (
    "/cdn-cgi/challenge-platform",  # Cloudflare challenge
    "g-recaptcha",  # Google reCAPTCHA widget
    "px-captcha",  # PerimeterX / HUMAN
    "/_px/",  # PerimeterX
    "_incapsula_resource",  # Imperva / Incapsula
    "captcha-delivery.com",  # DataDome captcha host
)

# Decisive fingerprint patterns (regex form).
_FINGERPRINT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"reference #\d{2}\.[0-9a-f]", re.I),  # Akamai "Reference #18.<hex>"
)

# Generic verify/deny words. WEAK: each is also common in normal prose (and some
# in URL slugs), so a hit only counts when ≥2 co-occur or one co-occurs with a
# very short body. Matched against title + url + a SHORT body only. Bare
# "forbidden"/"blocked" are intentionally absent — they live in legit slugs and
# are already covered, anchored, by _URL_PATTERNS.
_WEAK_PHRASES: tuple[str, ...] = (
    "人机验证",
    "请输入验证码",
    "请完成验证",
    "完成安全验证",
    "安全验证",
    "验证码",
    "访问频繁",
    "操作频繁",
    "access denied",
    "unusual traffic",
    "verify you are a human",
    "are not a robot",
    "are you a robot",
    "datadome",
    "perimeterx",
    "incapsula",
)

# A body shorter than this is treated as "interstitial-sized" — block pages carry
# almost no content, so a single weak hit alongside a tiny body is enough. Kept in
# lockstep with apps/web/src/lib/toolOutcome.ts.
_SHORT_BODY = 40


def detect_block(
    *, url: str = "", title: str = "", content: str = ""
) -> tuple[bool, str | None]:
    """Return ``(blocked, reason)`` for a fetched/navigated page.

    *reason* is a short human-readable Chinese phrase when blocked, else ``None``.
    """
    url = url or ""
    title = title or ""
    content = content or ""

    # 1. Redirected to a block-landing path. Anchored patterns; scan the final
    #    URL only (never the body — a slug in prose must not count).
    if url and any(p.search(url) for p in _URL_PATTERNS):
        return True, "目标站点把访问重定向到了拦截/验证页"

    strong_hay = f"{title}\n{url}".lower()

    # 2. Strong vendor phrase in the title/url.
    if any(p in strong_hay for p in _STRONG_PHRASES):
        return True, "目标站点返回了反爬/验证页(命中拦截标题)"

    # 3. Decisive vendor fingerprint anywhere.
    low_all = f"{title}\n{url}\n{content}".lower()
    if any(fp in low_all for fp in _UNIQUE_FINGERPRINTS):
        return True, "目标站点启用了人机验证(反爬服务)"
    if any(p.search(low_all) for p in _FINGERPRINT_PATTERNS):
        return True, "目标站点返回了反爬/拦截页(命中风控指纹)"

    # 4. Weak words: only the title/url plus a short body (long article bodies are
    #    never scanned). Need ≥2 distinct hits, or one alongside a tiny body.
    short = len(content.strip()) < _SHORT_BODY
    weak_hay = (f"{title}\n{url}\n{content}" if short else f"{title}\n{url}").lower()
    weak_hits = sum(1 for p in _WEAK_PHRASES if p in weak_hay)
    if weak_hits >= 2 or (weak_hits >= 1 and short):
        return True, "目标站点疑似返回了验证/拦截页"

    return False, None
