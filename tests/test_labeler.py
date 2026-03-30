"""Tests for the churn pattern labeler."""
import pytest

from churn.labeler import ChurnPattern, Label, label, _classify_pattern, _longest_stuck_streak
from churn.recorder import ChurnRecorder
from churn.tree import ExplorationTree, Hypothesis, TreeBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tree(scores: list[float], task: str = "test") -> ExplorationTree:
    recorder = ChurnRecorder()
    run_id = recorder.start_run(task)
    for i, score in enumerate(scores):
        recorder.step(f"tool_{i}", lambda: "ok", score=score)
        if i < len(scores) - 1:
            recorder.notify_text_response()
    recorder.end_run(run_id, final_score=scores[-1])
    return TreeBuilder(task=task).build(recorder.events)


# ---------------------------------------------------------------------------
# _classify_pattern
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index,expected", [
    (1.0, ChurnPattern.EFFICIENT),
    (1.5, ChurnPattern.EFFICIENT),
    (1.51, ChurnPattern.EXPLORATORY),
    (3.0, ChurnPattern.EXPLORATORY),
    (3.01, ChurnPattern.STUCK),
    (5.0, ChurnPattern.STUCK),
    (5.01, ChurnPattern.CATASTROPHIC),
    (10.0, ChurnPattern.CATASTROPHIC),
])
def test_classify_pattern_boundaries(index, expected):
    tree = ExplorationTree(run_id="x", task="t", hypotheses=[], churn_index=index)
    assert _classify_pattern(tree) == expected


# ---------------------------------------------------------------------------
# _longest_stuck_streak
# ---------------------------------------------------------------------------


def test_stuck_streak_all_improving():
    hs = [Hypothesis(id=i, description="", events=[], score_delta=0.5) for i in range(3)]
    assert _longest_stuck_streak(hs) == 0


def test_stuck_streak_all_flat():
    hs = [Hypothesis(id=i, description="", events=[], score_delta=0.0) for i in range(4)]
    assert _longest_stuck_streak(hs) == 4


def test_stuck_streak_mixed():
    deltas = [0.2, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.1]
    hs = [Hypothesis(id=i, description="", events=[], score_delta=d) for i, d in enumerate(deltas)]
    assert _longest_stuck_streak(hs) == 3


# ---------------------------------------------------------------------------
# label() — pattern classification
# ---------------------------------------------------------------------------


def test_label_efficient():
    tree = make_tree([0.5, 1.0])  # churn=1.0
    lbl = label(tree)
    assert lbl.pattern == ChurnPattern.EFFICIENT


def test_label_exploratory():
    tree = make_tree([0.2, 0.2, 0.2, 0.6, 0.6, 0.6, 1.0])  # churn=7/3=2.33
    lbl = label(tree)
    assert lbl.pattern == ChurnPattern.EXPLORATORY


def test_label_stuck():
    tree = make_tree([0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4])
    lbl = label(tree)
    assert lbl.pattern == ChurnPattern.STUCK


def test_label_catastrophic():
    tree = make_tree([0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    lbl = label(tree)
    assert lbl.pattern == ChurnPattern.CATASTROPHIC


# ---------------------------------------------------------------------------
# label() — return structure
# ---------------------------------------------------------------------------


def test_label_returns_label_dataclass():
    tree = make_tree([0.5, 1.0])
    lbl = label(tree)
    assert isinstance(lbl, Label)
    assert isinstance(lbl.pattern, ChurnPattern)
    assert isinstance(lbl.root_cause, str)
    assert isinstance(lbl.evidence, list)
    assert len(lbl.evidence) > 0


def test_label_evidence_includes_churn_index():
    tree = make_tree([0.5, 1.0])
    lbl = label(tree)
    combined = " ".join(lbl.evidence)
    assert "churn_index" in combined


def test_label_evidence_includes_final_score():
    tree = make_tree([0.5, 1.0])
    lbl = label(tree)
    combined = " ".join(lbl.evidence)
    assert "final_score" in combined


def test_label_stuck_mentions_streak():
    # 6 consecutive flat hypotheses
    tree = make_tree([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4])
    lbl = label(tree)
    combined = " ".join(lbl.evidence)
    assert "stuck streak" in combined


def test_label_root_cause_is_nonempty_string():
    for scores in [
        [0.5, 1.0],
        [0.2, 0.2, 0.5, 0.5, 1.0],
        [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.4],
        [0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0],
    ]:
        tree = make_tree(scores)
        lbl = label(tree)
        assert len(lbl.root_cause) > 20, f"Root cause too short for {lbl.pattern}"


def test_label_empty_tree():
    tree = ExplorationTree(run_id="x", task="empty", hypotheses=[])
    lbl = label(tree)
    assert isinstance(lbl.pattern, ChurnPattern)
    assert "empty" in lbl.evidence[0]
