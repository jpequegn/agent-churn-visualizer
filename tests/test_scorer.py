"""Tests for all three scorer modes."""
import sys
from unittest.mock import patch

import pytest

from churn.recorder import AgentEvent
from churn.scorer import CommandScorer, HeuristicScorer, HumanScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def tool_result(output: str, tool_name: str = "bash", run_id: str = "r1") -> AgentEvent:
    return AgentEvent(
        run_id=run_id,
        event_type="tool_result",
        hypothesis_id=1,
        tool_name=tool_name,
        tool_output=output,
    )


def tool_call(name: str, args: dict | None = None, run_id: str = "r1") -> AgentEvent:
    return AgentEvent(
        run_id=run_id,
        event_type="tool_call",
        hypothesis_id=1,
        tool_name=name,
        tool_input=args or {},
    )


# ---------------------------------------------------------------------------
# HeuristicScorer
# ---------------------------------------------------------------------------


class TestHeuristicScorer:
    scorer = HeuristicScorer()

    def test_baseline_no_signals(self):
        events = [tool_result("all good")]
        score = self.scorer.score(events, [])
        assert score == pytest.approx(0.5)

    def test_zero_failures_boosts_score(self):
        events = [tool_result("0 passed, 0 failed")]
        score = self.scorer.score(events, [0.3])
        assert score > 0.5

    def test_error_keyword_with_low_prev_boosts(self):
        events = [tool_result("AssertionError: expected True")]
        score = self.scorer.score(events, [0.2])
        assert score > 0.5

    def test_error_keyword_with_high_prev_no_boost(self):
        """At high prev score, a new error shouldn't push above baseline."""
        events = [tool_result("AssertionError: expected True")]
        score = self.scorer.score(events, [0.8])
        assert score == pytest.approx(0.5)

    def test_repeated_tool_call_penalised(self):
        same_call = tool_call("bash", {"args": ["pytest"], "kwargs": {}})
        events = [same_call, same_call]
        score = self.scorer.score(events, [0.5])
        assert score < 0.5

    def test_score_clamped_to_0_1(self):
        # Multiple penalty signals shouldn't go below 0
        events = [
            tool_call("bash", {}),
            tool_call("bash", {}),
        ]
        score = self.scorer.score(events * 10, [])
        assert 0.0 <= score <= 1.0

    def test_empty_events(self):
        score = self.scorer.score([], [])
        assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# TestScorer
# ---------------------------------------------------------------------------


class TestCommandScorer:
    def _make_scorer(self, cmd: str = "pytest") -> CommandScorer:
        return CommandScorer(test_command=cmd, timeout=10)

    def test_all_passing(self):
        scorer = self._make_scorer()
        score = scorer._parse_score("5 passed in 0.1s", returncode=0)
        assert score == pytest.approx(1.0)

    def test_partial_passing(self):
        scorer = self._make_scorer()
        score = scorer._parse_score("3 passed, 1 failed in 0.2s", returncode=1)
        assert score == pytest.approx(0.75)

    def test_all_failing(self):
        scorer = self._make_scorer()
        score = scorer._parse_score("0 passed, 4 failed", returncode=1)
        assert score == pytest.approx(0.0)

    def test_no_test_output_returncode_0(self):
        scorer = self._make_scorer()
        score = scorer._parse_score("no output", returncode=0)
        assert score == pytest.approx(1.0)

    def test_no_test_output_returncode_1(self):
        scorer = self._make_scorer()
        score = scorer._parse_score("no output", returncode=1)
        assert score == pytest.approx(0.0)

    def test_errors_counted(self):
        scorer = self._make_scorer()
        score = scorer._parse_score("2 passed, 1 error in 0.3s", returncode=1)
        assert score == pytest.approx(2 / 3)

    def test_runs_real_command(self):
        scorer = CommandScorer(test_command=f"{sys.executable} -c 'import sys; sys.exit(0)'", timeout=5)
        score = scorer.score([], [])
        assert score == pytest.approx(1.0)

    def test_timeout_returns_zero(self):
        scorer = CommandScorer(test_command="sleep 10", timeout=1)
        score = scorer.score([], [])
        assert score == pytest.approx(0.0)

    def test_previous_scores_ignored(self):
        scorer = self._make_scorer()
        s1 = scorer._parse_score("3 passed", returncode=0)
        s2 = scorer._parse_score("3 passed", returncode=0)
        assert s1 == s2


# ---------------------------------------------------------------------------
# HumanScorer
# ---------------------------------------------------------------------------


class TestHumanScorer:
    scorer = HumanScorer()

    def test_valid_input_scaled(self):
        with patch("builtins.input", return_value="7"):
            score = self.scorer.score([], [])
        assert score == pytest.approx(0.7)

    def test_input_clamped_above_10(self):
        with patch("builtins.input", return_value="15"):
            score = self.scorer.score([], [])
        assert score == pytest.approx(1.0)

    def test_input_clamped_below_0(self):
        with patch("builtins.input", return_value="-3"):
            score = self.scorer.score([], [])
        assert score == pytest.approx(0.0)

    def test_eof_returns_fallback(self):
        with patch("builtins.input", side_effect=EOFError):
            score = self.scorer.score([], [])
        assert score == pytest.approx(0.5)

    def test_invalid_input_returns_fallback(self):
        with patch("builtins.input", return_value="not a number"):
            score = self.scorer.score([], [])
        assert score == pytest.approx(0.5)

    def test_zero_input(self):
        with patch("builtins.input", return_value="0"):
            score = self.scorer.score([], [])
        assert score == pytest.approx(0.0)

    def test_ten_input(self):
        with patch("builtins.input", return_value="10"):
            score = self.scorer.score([], [])
        assert score == pytest.approx(1.0)
