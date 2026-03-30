"""Tests for TreeBuilder and ExplorationTree."""
from churn.recorder import AgentEvent, ChurnRecorder
from churn.tree import ExplorationTree, Hypothesis, TreeBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_events(
    *steps: tuple[str, float | None],
    task: str = "test task",
) -> tuple[str, list[AgentEvent]]:
    """Build a recorder run from (tool_label, score) pairs.

    Each pair becomes one hypothesis (text_response between each).
    Returns (run_id, events).
    """
    recorder = ChurnRecorder()
    run_id = recorder.start_run(task)
    for i, (label, score) in enumerate(steps):
        recorder.step(label, lambda: "ok", score=score)
        if i < len(steps) - 1:
            recorder.notify_text_response()
    recorder.end_run(run_id, final_score=steps[-1][1] if steps else None)
    return run_id, recorder.events


# ---------------------------------------------------------------------------
# Dataclass sanity
# ---------------------------------------------------------------------------


def test_hypothesis_defaults():
    h = Hypothesis(id=1, description="read", events=[])
    assert h.score == 0.0
    assert h.score_delta == 0.0
    assert h.token_count == 0


def test_exploration_tree_defaults():
    tree = ExplorationTree(run_id="abc", task="t", hypotheses=[])
    assert tree.final_score == 0.0
    assert tree.churn_index == 1.0
    assert tree.total_tokens == 0


# ---------------------------------------------------------------------------
# TreeBuilder.build() — basic structure
# ---------------------------------------------------------------------------


def test_empty_events_returns_empty_tree():
    tree = TreeBuilder(task="t").build([])
    assert tree.hypotheses == []
    assert tree.run_id == ""


def test_single_hypothesis_structure():
    run_id, events = make_events(("read_file", 0.5))
    tree = TreeBuilder(task="task").build(events)

    assert tree.run_id == run_id
    assert len(tree.hypotheses) == 1
    assert tree.hypotheses[0].id == 1
    assert tree.hypotheses[0].description == "read_file"


def test_two_hypotheses_from_three_events():
    """Issue acceptance: 4 events in 2 hypotheses → correct tree structure."""
    run_id, events = make_events(("read_file", 0.2), ("bash", 0.6))
    tree = TreeBuilder(task="task").build(events)

    assert len(tree.hypotheses) == 2
    assert tree.hypotheses[0].id == 1
    assert tree.hypotheses[1].id == 2


def test_four_hypotheses():
    run_id, events = make_events(
        ("read_db", 0.2),
        ("grep", 0.4),
        ("edit_file", 0.6),
        ("run_tests", 1.0),
    )
    tree = TreeBuilder(task="fix bug").build(events)
    assert len(tree.hypotheses) == 4
    assert [h.id for h in tree.hypotheses] == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Score and score_delta
# ---------------------------------------------------------------------------


def test_score_delta_computed_correctly():
    _, events = make_events(("h1", 0.2), ("h2", 0.5), ("h3", 1.0))
    tree = TreeBuilder().build(events)

    assert tree.hypotheses[0].score == pytest_approx(0.2)
    assert tree.hypotheses[0].score_delta == pytest_approx(0.2)
    assert tree.hypotheses[1].score == pytest_approx(0.5)
    assert tree.hypotheses[1].score_delta == pytest_approx(0.3)
    assert tree.hypotheses[2].score == pytest_approx(1.0)
    assert tree.hypotheses[2].score_delta == pytest_approx(0.5)


def test_score_no_improvement():
    _, events = make_events(("h1", 0.3), ("h2", 0.3))
    tree = TreeBuilder().build(events)
    assert tree.hypotheses[1].score_delta == pytest_approx(0.0)


def test_final_score_from_run_complete():
    _, events = make_events(("h1", 0.5), ("h2", None))
    # end_run was called with final_score=None; patch last event score
    run_complete = next(e for e in events if e.event_type == "run_complete")
    run_complete.score = 0.9
    tree = TreeBuilder().build(events)
    assert tree.final_score == pytest_approx(0.9)


# ---------------------------------------------------------------------------
# Churn index
# ---------------------------------------------------------------------------


def test_churn_index_straight_line():
    """Every hypothesis improves score → churn_index = 1.0."""
    _, events = make_events(("h1", 0.5), ("h2", 1.0))
    tree = TreeBuilder().build(events)
    assert tree.churn_index == pytest_approx(1.0)


def test_churn_index_one_wasted_hypothesis():
    """3 hypotheses but only 2 improved → churn_index = 1.5."""
    _, events = make_events(("h1", 0.5), ("h2", 0.5), ("h3", 1.0))
    tree = TreeBuilder().build(events)
    # winning_depth = 2 (h1 +0.5, h2 +0.0 skipped, h3 +0.5)
    assert tree.churn_index == pytest_approx(3 / 2)


def test_churn_index_acceptance_criteria():
    """From issue: 3 hypotheses on a 2-step task → churn_index = 1.5."""
    _, events = make_events(("h1", 0.5), ("h2", 0.5), ("h3", 1.0))
    tree = TreeBuilder().build(events)
    assert tree.churn_index == pytest_approx(1.5)


def test_churn_index_no_improvement_fallback():
    """If no hypothesis improves score, winning_depth = total → churn_index = 1.0."""
    _, events = make_events(("h1", 0.0), ("h2", 0.0))
    tree = TreeBuilder().build(events)
    assert tree.churn_index == pytest_approx(1.0)


# ---------------------------------------------------------------------------
# Token counts
# ---------------------------------------------------------------------------


def test_token_counts_summed_per_hypothesis():
    recorder = ChurnRecorder()
    run_id = recorder.start_run("task")
    recorder.step("t1", lambda: "ok", input_tokens=100, output_tokens=50)
    recorder.notify_text_response()
    recorder.step("t2", lambda: "ok", input_tokens=200, output_tokens=80)
    recorder.end_run(run_id)

    tree = TreeBuilder(task="task").build(recorder.events)
    assert tree.hypotheses[0].token_count == 150
    assert tree.hypotheses[1].token_count == 280
    assert tree.total_tokens == 430


def test_cost_computed_from_tokens():
    recorder = ChurnRecorder()
    run_id = recorder.start_run("task")
    recorder.step("t1", lambda: "ok", input_tokens=1_000_000, output_tokens=0)
    recorder.end_run(run_id)

    tree = TreeBuilder(task="task", cost_per_token=3e-6).build(recorder.events)
    assert tree.total_cost_usd == pytest_approx(3.0)


# ---------------------------------------------------------------------------
# run_id propagation
# ---------------------------------------------------------------------------


def test_run_id_propagated_to_tree():
    run_id, events = make_events(("h1", 1.0))
    tree = TreeBuilder(task="t").build(events)
    assert tree.run_id == run_id


# ---------------------------------------------------------------------------
# pytest.approx alias for readability
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

pytest_approx = pytest.approx
