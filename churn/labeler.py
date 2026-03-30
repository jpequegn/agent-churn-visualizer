from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from churn.tree import ExplorationTree, Hypothesis


class ChurnPattern(str, Enum):
    EFFICIENT = "efficient"          # churn_index 1.0–1.5: mostly straight to solution
    EXPLORATORY = "exploratory"      # churn_index 1.5–3.0: healthy exploration, converges
    STUCK = "stuck"                  # churn_index 3.0–5.0: loops without improvement
    CATASTROPHIC = "catastrophic"    # churn_index  >5.0: never converges


@dataclass
class Label:
    pattern: ChurnPattern
    root_cause: str          # one-sentence diagnosis
    evidence: list[str]      # supporting observations from the tree


def label(tree: ExplorationTree) -> Label:
    """Classify an ExplorationTree into a ChurnPattern with diagnosis.

    Classification is based on three signals:
    1. churn_index (primary)
    2. stuck_streak: longest run of consecutive zero-delta hypotheses
    3. final_score: did the agent ever reach a solution?
    """
    pattern = _classify_pattern(tree)
    stuck_streak = _longest_stuck_streak(tree.hypotheses)
    evidence = _collect_evidence(tree, stuck_streak)
    root_cause = _diagnose(pattern, tree, stuck_streak)
    return Label(pattern=pattern, root_cause=root_cause, evidence=evidence)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _classify_pattern(tree: ExplorationTree) -> ChurnPattern:
    idx = tree.churn_index
    if idx <= 1.5:
        return ChurnPattern.EFFICIENT
    if idx <= 3.0:
        return ChurnPattern.EXPLORATORY
    if idx <= 5.0:
        return ChurnPattern.STUCK
    return ChurnPattern.CATASTROPHIC


def _longest_stuck_streak(hypotheses: list[Hypothesis]) -> int:
    """Return the length of the longest run of zero-delta hypotheses."""
    max_streak = current = 0
    for h in hypotheses:
        if h.score_delta <= 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def _collect_evidence(tree: ExplorationTree, stuck_streak: int) -> list[str]:
    evidence: list[str] = []

    n = len(tree.hypotheses)
    if n == 0:
        return ["empty tree — no hypotheses recorded"]

    evidence.append(f"{n} hypotheses, churn_index={tree.churn_index:.2f}")
    evidence.append(f"final_score={tree.final_score:.2f}")

    # Score trajectory
    deltas = [h.score_delta for h in tree.hypotheses]
    positive = sum(1 for d in deltas if d > 0)
    zero = sum(1 for d in deltas if d == 0)
    negative = sum(1 for d in deltas if d < 0)
    evidence.append(f"score deltas: +{positive} improving, ={zero} flat, -{negative} regressing")

    if stuck_streak >= 2:
        evidence.append(f"longest stuck streak: {stuck_streak} consecutive flat/regressing hypotheses")

    # Repeated tool detection
    repeated = _repeated_tool_count(tree.hypotheses)
    if repeated:
        evidence.append(f"repeated tool calls (same tool+args): {repeated}")

    # Token distribution
    token_counts = [h.token_count for h in tree.hypotheses]
    if token_counts:
        avg_tok = sum(token_counts) / len(token_counts)
        evidence.append(f"avg tokens/hypothesis: {avg_tok:.0f}")

    return evidence


def _repeated_tool_count(hypotheses: list[Hypothesis]) -> int:
    """Count tool calls that repeat an identical (tool_name, input) from an earlier hypothesis."""
    seen: set[tuple] = set()
    repeated = 0
    for h in hypotheses:
        for ev in h.events:
            if ev.event_type != "tool_call":
                continue
            import json
            args_str = json.dumps(ev.tool_input, sort_keys=True) if ev.tool_input else ""
            sig = (ev.tool_name, args_str)
            if sig in seen:
                repeated += 1
            else:
                seen.add(sig)
    return repeated


def _diagnose(pattern: ChurnPattern, tree: ExplorationTree, stuck_streak: int) -> str:
    final = tree.final_score
    n = len(tree.hypotheses)

    if pattern == ChurnPattern.EFFICIENT:
        return "Agent found a clear path to the solution with minimal exploration."

    if pattern == ChurnPattern.EXPLORATORY:
        if final >= 0.8:
            return "Agent explored multiple approaches before converging — healthy for ambiguous tasks."
        return "Agent explored broadly but reached only a partial solution; task may need clearer acceptance criteria."

    if pattern == ChurnPattern.STUCK:
        if stuck_streak >= 3:
            return (
                f"Agent looped for {stuck_streak} consecutive hypotheses without score improvement — "
                "likely a bad tool design or missing information preventing progress."
            )
        if final < 0.3:
            return "Agent made early progress then stalled; task scope may have exceeded available tools."
        return "Agent oscillated without convergence — consider breaking the task into smaller sub-tasks."

    # CATASTROPHIC
    if final == 0.0:
        return (
            f"Agent never scored above 0 across {n} hypotheses — "
            "task is likely impossible with the current toolset, or the scoring signal is broken."
        )
    return (
        f"Agent churned {n} hypotheses without converging (final_score={final:.2f}) — "
        "severe tool-loop or fundamentally ambiguous task definition."
    )
