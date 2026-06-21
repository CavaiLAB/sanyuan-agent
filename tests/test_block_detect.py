"""Unit tests for the conservative anti-bot block-page detector.

Kept in lockstep with apps/web/src/lib/toolOutcome.test.ts — same positive and
(crucially) negative cases on both sides of the contract.
"""
# ruff: noqa: RUF001  — fixtures contain realistic CJK fullwidth punctuation.

from __future__ import annotations

from neuroagent.tools.block_detect import detect_block


class TestBlockedPositives:
    def test_lianjia_forbidden_redirect(self) -> None:
        blocked, reason = detect_block(
            url="https://hip.lianjia.com/forbidden?id=d944e3",
            title="访问已被拦截",
            content="访问已被拦截",
        )
        assert blocked is True
        assert reason

    def test_url_pattern_sec_check(self) -> None:
        blocked, _ = detect_block(url="https://x.com/sec-check", title="", content="")
        assert blocked is True

    def test_strong_vendor_titles(self) -> None:
        assert detect_block(title="百度安全验证")[0] is True
        assert detect_block(title="腾讯防水墙")[0] is True
        assert detect_block(title="Just a moment...")[0] is True
        assert detect_block(title="Attention Required! | Cloudflare")[0] is True

    def test_unique_fingerprints(self) -> None:
        assert detect_block(content='<div class="g-recaptcha"></div>')[0] is True
        assert detect_block(content="cdn.example/_px/xhr px-captcha")[0] is True
        assert detect_block(content="loads https://geo.captcha-delivery.com/captcha/")[0] is True
        assert detect_block(content="window._Incapsula_Resource('SWJIYLWA')")[0] is True

    def test_akamai_reference_pattern(self) -> None:
        blocked, _ = detect_block(
            title="Access Denied",
            content="You don't have permission to access this resource. Reference #18.abcd1234.1700000000.deadbeef",
        )
        assert blocked is True

    def test_weak_cooccurrence_short_body(self) -> None:
        # One weak hit alone is not enough unless the body is interstitial-sized.
        blocked, _ = detect_block(title="提示", content="请完成人机验证")
        assert blocked is True


class TestBlockedNegatives:
    def test_normal_article(self) -> None:
        blocked, reason = detect_block(
            url="https://example.com/post",
            title="北京二居室租金趋势",
            content="2024 年安贞西里两居室整租均价约 8500 元，环比小幅上涨。",
        )
        assert blocked is False
        assert reason is None

    def test_long_article_discussing_captchas(self) -> None:
        # The block words live only in a long body, which is never scanned.
        blocked, _ = detect_block(
            url="https://blog.example.com/how-captcha-works",
            title="CAPTCHA 的工作原理与发展史",
            content=(
                "本文系统介绍人机验证（captcha）、滑块验证、请输入验证码 等机制如何阻止机器人爬虫；"
                "从早期扭曲字符，到 reCAPTCHA 的行为分析，再到 DataDome、PerimeterX 等商业风控服务，"
                "并讨论访问已被拦截时网站的常见处理与对用户体验的影响。"
            ),
        )
        assert blocked is False

    def test_legit_slugs_with_block_words(self) -> None:
        assert (
            detect_block(
                url="https://recipes.example.com/forbidden-fruit-cake",
                title="禁果蛋糕食谱",
                content="材料与步骤……",
            )[0]
            is False
        )
        assert (
            detect_block(
                url="https://docs.example.com/captcha-integration-guide",
                title="集成指南",
                content="如何在你的站点集成验证。",
            )[0]
            is False
        )

    def test_empty(self) -> None:
        assert detect_block()[0] is False
        assert detect_block(url="", title="", content="")[0] is False
