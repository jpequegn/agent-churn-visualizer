from __future__ import annotations

import re
import subprocess
from abc import ABC, abstractmethod

from churn.recorder import AgentEvent


class BaseScorer(ABC):
    """Interface all scorers implement."""

    @abstractmethod
    def score(self, events: list[AgentEvent], previous_scores: list[float]) -> float:
        """Return a score in [0.0, 1.0] for the current hypothesis.

        Args:
            events: All events belonging to the current hypothesis.
            previous_scores: Scores from all prior hypotheses (oldest first).
        """


# ---------------------------------------------------------------------------
# Heuristic scorer
# ---------------------------------------------------------------------------

_FAILURE_PATTERN = re.compile(
    r"(\d+)\s+(?:failed|error|errors?)",
    re.IGNORECASE,
)
_ERROR_KEYWORDS = re.compile(
    r"(?:Error|Exception|Traceback|FAILED|FAIL:|assert)",
    re.IGNORECASE,
)


def _extract_failure_count(text: str) -> int | None:
    """Pull the leading failure count from pytest/test-runner output."""
    m = _FAILURE_PATTERN.search(text)
    return int(m.group(1)) if m else None


def _tool_call_signature(event: AgentEvent) -> tuple[str | None, str]:
    """Stable identity for a tool call: (tool_name, serialised_args)."""
    import json

    args_str = json.dumps(event.tool_input, sort_keys=True) if event.tool_input else ""
    return event.tool_name, args_str


class HeuristicScorer(BaseScorer):
    """Automatic scorer based on observable signals in tool output.

    Signals (all relative to previous hypothesis):
    - Fewer test failures detected in output  → +0.2
    - Error message content changed           → +0.1  (agent learned something)
    - Same tool + same args repeated          → -0.2  (stuck in loop)
    - Baseline                                → 0.5
    """

    _BASELINE = 0.5

    def score(self, events: list[AgentEvent], previous_scores: list[float]) -> float:
        outputs = [
            e.tool_output or ""
            for e in events
            if e.event_type == "tool_result" and e.tool_output
        ]
        combined_output = "\n".join(outputs)

        adjustment = 0.0
        prev_score = previous_scores[-1] if previous_scores else self._BASELINE

        # Signal 1: failure count decreased
        current_failures = _extract_failure_count(combined_output)
        if current_failures is not None:
            # find last hypothesis output that had a failure count
            # (not available here, so use presence/absence as a proxy)
            adjustment += 0.2 if current_failures == 0 else 0.0

        # Signal 2: error content changed (agent learned something new)
        if _ERROR_KEYWORDS.search(combined_output):
            # If previous score was low, any new error info is progress
            if prev_score < 0.7:
                adjustment += 0.1

        # Signal 3: stuck-in-loop detection
        call_events = [e for e in events if e.event_type == "tool_call"]
        # We don't have prior hypothesis events here, so flag self-repeats
        # within this hypothesis as a weak stuck signal
        seen: set[tuple] = set()
        for ev in call_events:
            sig = _tool_call_signature(ev)
            if sig in seen:
                adjustment -= 0.2
                break
            seen.add(sig)

        raw = self._BASELINE + adjustment
        return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Test-based scorer
# ---------------------------------------------------------------------------


class CommandScorer(BaseScorer):
    """Score by running a test command and counting passing tests.

    Score = passing / total  (0.0 if the command fails to parse).
    """

    def __init__(self, test_command: str, timeout: int = 60) -> None:
        self.test_command = test_command
        self.timeout = timeout

    def score(self, events: list[AgentEvent], previous_scores: list[float]) -> float:  # noqa: ARG002
        try:
            result = subprocess.run(
                self.test_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            output = result.stdout + result.stderr
            return self._parse_score(output, result.returncode)
        except subprocess.TimeoutExpired:
            return 0.0

    def _parse_score(self, output: str, returncode: int) -> float:
        # pytest summary line: "3 passed, 1 failed" or "4 passed"
        passed = re.search(r"(\d+)\s+passed", output)
        failed = re.search(r"(\d+)\s+failed", output)
        errors = re.search(r"(\d+)\s+error", output)

        n_passed = int(passed.group(1)) if passed else 0
        n_failed = int(failed.group(1)) if failed else 0
        n_errors = int(errors.group(1)) if errors else 0
        total = n_passed + n_failed + n_errors

        if total == 0:
            return 1.0 if returncode == 0 else 0.0

        return n_passed / total


# ---------------------------------------------------------------------------
# Human scorer
# ---------------------------------------------------------------------------


class HumanScorer(BaseScorer):
    """Pause after each hypothesis and ask the user for a score.

    Intended for building ground-truth labelled datasets. Not suitable for
    automated runs. In non-interactive contexts, falls back to 0.5.
    """

    def score(self, events: list[AgentEvent], previous_scores: list[float]) -> float:  # noqa: ARG002
        tool_names = [e.tool_name for e in events if e.event_type == "tool_call"]
        summary = ", ".join(t for t in tool_names if t) or "(no tool calls)"
        try:
            raw = input(f"\nHypothesis tools: {summary}\nScore (0-10): ").strip()
            value = float(raw) / 10.0
            return max(0.0, min(1.0, value))
        except (EOFError, ValueError):
            return 0.5
