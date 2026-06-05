"""Tests for working-memory compression (PRD §7.6) with an injected summarizer."""

from neuroagent.memory.working import WorkingMemory


def _fake_summarizer(text: str) -> str:
    # Deterministic stand-in for an LLM summarizer.
    return f"COMPRESSED({text.count('Step')} steps)"


class TestBackwardCompatible:
    def test_no_summarizer_never_compresses(self) -> None:
        wm = WorkingMemory()  # no summarizer → legacy ring-buffer behaviour
        for i in range(30):
            wm.add_step(plan=f"p{i}", action=f"a{i}", result=f"r{i}")
        assert wm.is_summarized is False
        summary = wm.get_summary(max_steps=5)
        assert "Summary of earlier steps" not in summary
        assert "p29" in summary  # most recent retained verbatim

    def test_empty_summary(self) -> None:
        assert WorkingMemory().get_summary() == "(no previous steps)"


class TestCompression:
    def test_compresses_when_over_step_count(self) -> None:
        wm = WorkingMemory(summarizer=_fake_summarizer, max_steps=3, max_recent=2)
        for i in range(4):  # exceeds max_steps=3
            wm.add_step(plan=f"p{i}", action=f"a{i}", result=f"r{i}")
        assert wm.is_summarized is True
        out = wm.get_summary()
        assert "Summary of earlier steps" in out
        assert "COMPRESSED" in out
        # The two most recent steps stay verbatim.
        assert "p3" in out and "p2" in out

    def test_compresses_when_over_token_budget(self) -> None:
        wm = WorkingMemory(
            summarizer=_fake_summarizer, max_steps=1000, token_budget=10, max_recent=1
        )
        wm.add_step(plan="x" * 80, action="y" * 80, result="z" * 80)
        wm.add_step(plan="next", action="act", result="res")
        assert wm.is_summarized is True

    def test_summary_accumulates_across_compressions(self) -> None:
        calls: list[str] = []

        def capturing(text: str) -> str:
            calls.append(text)
            return f"S{len(calls)}"

        wm = WorkingMemory(summarizer=capturing, max_steps=2, max_recent=1)
        for i in range(6):
            wm.add_step(plan=f"p{i}", action="a", result="r")
        # Compressed more than once, and later calls carry the prior summary.
        assert len(calls) >= 2
        assert any("Earlier summary:" in c for c in calls[1:])
