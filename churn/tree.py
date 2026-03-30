from __future__ import annotations

from dataclasses import dataclass

from churn.recorder import AgentEvent


@dataclass
class Hypothesis:
    id: int
    description: str  # first tool_call label in this cycle
    events: list[AgentEvent]
    score: float = 0.0
    score_delta: float = 0.0  # improvement over previous hypothesis
    token_count: int = 0
    duration_seconds: float = 0.0


@dataclass
class ExplorationTree:
    run_id: str
    task: str
    hypotheses: list[Hypothesis]
    final_score: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    duration_seconds: float = 0.0
    churn_index: float = 1.0  # total_hypotheses / depth_of_winning_path


# Cost per token (input+output combined) for a mid-tier model — caller can override
_DEFAULT_COST_PER_TOKEN = 3e-6  # $3 / 1M tokens


class TreeBuilder:
    """Build an ExplorationTree from a flat list of AgentEvents."""

    def __init__(self, task: str = "", cost_per_token: float = _DEFAULT_COST_PER_TOKEN) -> None:
        self.task = task
        self.cost_per_token = cost_per_token

    def build(self, events: list[AgentEvent]) -> ExplorationTree:
        """Group events into Hypothesis objects and compute tree-level metrics.

        The winning path depth is defined as the number of hypotheses that
        showed a strictly positive score_delta (progress toward the goal).
        A straight-line run where every hypothesis improves the score produces
        churn_index = 1.0.
        """
        if not events:
            run_id = ""
            return ExplorationTree(run_id=run_id, task=self.task, hypotheses=[])

        run_id = events[0].run_id
        hypotheses = self._group_into_hypotheses(events)
        self._compute_scores_and_deltas(hypotheses, events)
        self._compute_token_counts(hypotheses)

        total_tokens = sum(h.token_count for h in hypotheses)
        final_score = hypotheses[-1].score if hypotheses else 0.0
        run_complete = next((e for e in events if e.event_type == "run_complete"), None)
        if run_complete and run_complete.score is not None:
            final_score = run_complete.score

        churn_index = self._compute_churn_index(hypotheses)
        total_cost = total_tokens * self.cost_per_token

        return ExplorationTree(
            run_id=run_id,
            task=self.task,
            hypotheses=hypotheses,
            final_score=final_score,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            churn_index=churn_index,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _group_into_hypotheses(self, events: list[AgentEvent]) -> list[Hypothesis]:
        """Partition events by hypothesis_id, preserving order."""
        buckets: dict[int, list[AgentEvent]] = {}
        for ev in events:
            if ev.event_type == "run_complete":
                continue
            buckets.setdefault(ev.hypothesis_id, []).append(ev)

        hypotheses: list[Hypothesis] = []
        for h_id in sorted(buckets):
            evs = buckets[h_id]
            # Use the first tool_call label as the description
            first_call = next((e for e in evs if e.event_type == "tool_call"), None)
            description = first_call.tool_name or f"hypothesis {h_id}" if first_call else f"hypothesis {h_id}"
            hypotheses.append(Hypothesis(id=h_id, description=description, events=evs))

        return hypotheses

    def _compute_scores_and_deltas(
        self, hypotheses: list[Hypothesis], events: list[AgentEvent]
    ) -> None:
        """Assign score and score_delta to each hypothesis.

        Score source priority:
        1. hypothesis_end event score for this hypothesis
        2. last tool_result score for this hypothesis
        3. Carry forward the previous hypothesis score (no information)
        """
        prev_score = 0.0
        for h in hypotheses:
            # Look for an explicit score on hypothesis_end or tool_result
            score: float | None = None
            for ev in reversed(h.events):
                if ev.event_type in ("hypothesis_end", "tool_result") and ev.score is not None:
                    score = ev.score
                    break
            if score is None:
                score = prev_score
            h.score = score
            h.score_delta = score - prev_score
            prev_score = score

        # If run_complete carries a final score, push it onto the last hypothesis
        run_complete = next((e for e in events if e.event_type == "run_complete"), None)
        if run_complete and run_complete.score is not None and hypotheses:
            last = hypotheses[-1]
            delta_adj = run_complete.score - (hypotheses[-2].score if len(hypotheses) > 1 else 0.0)
            last.score = run_complete.score
            last.score_delta = delta_adj

    def _compute_token_counts(self, hypotheses: list[Hypothesis]) -> None:
        for h in hypotheses:
            h.token_count = sum(e.input_tokens + e.output_tokens for e in h.events)

    def _compute_churn_index(self, hypotheses: list[Hypothesis]) -> float:
        """churn_index = total_hypotheses / winning_path_depth.

        Winning path depth = number of hypotheses with positive score_delta.
        Falls back to len(hypotheses) if none improved (churn_index = 1.0).
        """
        if not hypotheses:
            return 1.0
        winning_depth = sum(1 for h in hypotheses if h.score_delta > 0)
        if winning_depth == 0:
            winning_depth = len(hypotheses)
        return len(hypotheses) / winning_depth
