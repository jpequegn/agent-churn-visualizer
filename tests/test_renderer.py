"""Tests for the terminal renderer."""
import pytest
from rich.console import Console

from churn.recorder import ChurnRecorder
from churn.renderer import _churn_label, _hypothesis_label, _score_color, render_tree
from churn.tree import ExplorationTree, Hypothesis, TreeBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tree(steps: list[tuple[str, float]], task: str = "test task") -> ExplorationTree:
    recorder = ChurnRecorder()
    run_id = recorder.start_run(task)
    for i, (label, score) in enumerate(steps):
        recorder.step(label, lambda: "ok", score=score, input_tokens=100, output_tokens=50)
        if i < len(steps) - 1:
            recorder.notify_text_response()
    recorder.end_run(run_id, final_score=steps[-1][1])
    return TreeBuilder(task=task).build(recorder.events)


def capture(tree: ExplorationTree) -> str:
    """Render tree to a string (no ANSI codes)."""
    console = Console(highlight=False, no_color=True, width=120)
    with console.capture() as cap:
        render_tree(tree, console=console)
    return cap.get()


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------


def test_score_color_positive_delta():
    assert _score_color(0.3) == "green"


def test_score_color_zero_delta():
    assert _score_color(0.0) == "yellow"


def test_score_color_negative_delta():
    assert _score_color(-0.1) == "red"


# ---------------------------------------------------------------------------
# Churn label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "index,expected",
    [
        (1.0, "efficient"),
        (1.2, "efficient"),
        (1.5, "exploratory"),
        (2.5, "exploratory"),
        (3.0, "high churn"),
        (5.0, "high churn"),
        (6.0, "lost"),
    ],
)
def test_churn_label(index, expected):
    assert _churn_label(index) == expected


# ---------------------------------------------------------------------------
# Hypothesis label content
# ---------------------------------------------------------------------------


def test_hypothesis_label_contains_score():
    h = Hypothesis(id=1, description="read_file", events=[], score=0.40, score_delta=0.40)
    text = _hypothesis_label(h).plain
    assert "0.40" in text


def test_hypothesis_label_positive_delta_prefix():
    h = Hypothesis(id=1, description="t", events=[], score=0.5, score_delta=0.5)
    text = _hypothesis_label(h).plain
    assert "(+0.50)" in text


def test_hypothesis_label_solved_badge():
    h = Hypothesis(id=2, description="t", events=[], score=1.0, score_delta=0.5)
    text = _hypothesis_label(h).plain
    assert "solved" in text


def test_hypothesis_label_no_solved_badge_below_1():
    h = Hypothesis(id=1, description="t", events=[], score=0.9, score_delta=0.3)
    text = _hypothesis_label(h).plain
    assert "solved" not in text


def test_hypothesis_label_regressed_badge():
    h = Hypothesis(id=3, description="t", events=[], score=0.2, score_delta=-0.1)
    text = _hypothesis_label(h).plain
    assert "regressed" in text


# ---------------------------------------------------------------------------
# render_tree output content
# ---------------------------------------------------------------------------


def test_render_includes_task_name():
    tree = make_tree([("read", 0.5), ("write", 1.0)], task="fix the bug")
    out = capture(tree)
    assert "fix the bug" in out


def test_render_includes_all_hypotheses():
    tree = make_tree([("h1", 0.3), ("h2", 0.6), ("h3", 1.0)])
    out = capture(tree)
    assert "H1" in out
    assert "H2" in out
    assert "H3" in out


def test_render_includes_churn_index():
    tree = make_tree([("h1", 0.5), ("h2", 1.0)])
    out = capture(tree)
    assert "Churn index" in out


def test_render_includes_token_cost():
    tree = make_tree([("h1", 1.0)])
    out = capture(tree)
    assert "tok" in out
    assert "$" in out


def test_render_churn_label_in_output():
    tree = make_tree([("h1", 0.5), ("h2", 1.0)])  # churn_index=1.0 → efficient
    out = capture(tree)
    assert "efficient" in out


def test_render_empty_tree_does_not_crash():
    tree = ExplorationTree(run_id="abc", task="empty", hypotheses=[])
    out = capture(tree)
    assert "empty" in out


def test_render_four_hypothesis_acceptance():
    """Reproduces the PLAN.md example: 4-hypothesis run produces correct output."""
    tree = make_tree(
        [
            ("read_db", 0.2),
            ("grep_check", 0.4),
            ("edit_on_conflict", 0.6),
            ("fix_sequence", 1.0),
        ],
        task="fix the duplicate key error in add_episode",
    )
    out = capture(tree)
    assert "fix the duplicate key error" in out
    assert "H1" in out
    assert "H4" in out
    assert "solved" in out
    assert "Churn index: 1.0" in out
    assert "efficient" in out
