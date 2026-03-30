"""Tests for ChurnStore — SQLite persistence."""
import pytest

from churn.recorder import ChurnRecorder
from churn.store import ChurnStore, RunSummary
from churn.tree import ExplorationTree, Hypothesis, TreeBuilder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    return ChurnStore(db_path=tmp_path / "test_runs.db")


def make_tree(task: str = "test task", n_hypotheses: int = 2, final_score: float = 1.0) -> ExplorationTree:
    recorder = ChurnRecorder()
    run_id = recorder.start_run(task)
    for i in range(n_hypotheses):
        score = (i + 1) / n_hypotheses * final_score
        recorder.step(f"tool_{i}", lambda: "ok", score=score, input_tokens=100, output_tokens=50)
        if i < n_hypotheses - 1:
            recorder.notify_text_response()
    recorder.end_run(run_id, final_score=final_score)
    return TreeBuilder(task=task).build(recorder.events)


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------


def test_store_creates_db_file(tmp_path):
    db = tmp_path / "sub" / "runs.db"
    ChurnStore(db_path=db)
    assert db.exists()


# ---------------------------------------------------------------------------
# save_run / get_run round-trip
# ---------------------------------------------------------------------------


def test_save_and_retrieve_run(store):
    tree = make_tree("fix the bug", n_hypotheses=3, final_score=1.0)
    store.save_run(tree)

    loaded = store.get_run(tree.run_id)
    assert loaded is not None
    assert loaded.run_id == tree.run_id
    assert loaded.task == tree.task
    assert loaded.final_score == pytest.approx(tree.final_score)
    assert loaded.churn_index == pytest.approx(tree.churn_index)
    assert loaded.total_tokens == tree.total_tokens


def test_hypothesis_count_preserved(store):
    tree = make_tree(n_hypotheses=4)
    store.save_run(tree)

    loaded = store.get_run(tree.run_id)
    assert len(loaded.hypotheses) == 4
    assert [h.id for h in loaded.hypotheses] == [1, 2, 3, 4]


def test_hypothesis_fields_preserved(store):
    tree = make_tree(n_hypotheses=2, final_score=0.8)
    store.save_run(tree)

    loaded = store.get_run(tree.run_id)
    h = loaded.hypotheses[0]
    assert h.description == "tool_0"
    assert h.score == pytest.approx(tree.hypotheses[0].score)
    assert h.score_delta == pytest.approx(tree.hypotheses[0].score_delta)
    assert h.token_count == tree.hypotheses[0].token_count


def test_events_preserved(store):
    tree = make_tree(n_hypotheses=2)
    store.save_run(tree)

    loaded = store.get_run(tree.run_id)
    original_event_count = sum(len(h.events) for h in tree.hypotheses)
    loaded_event_count = sum(len(h.events) for h in loaded.hypotheses)
    assert loaded_event_count == original_event_count


def test_event_fields_preserved(store):
    tree = make_tree(n_hypotheses=1)
    store.save_run(tree)

    loaded = store.get_run(tree.run_id)
    call_ev = next(e for e in loaded.hypotheses[0].events if e.event_type == "tool_call")
    assert call_ev.run_id == tree.run_id
    assert call_ev.input_tokens == 100
    result_ev = next(e for e in loaded.hypotheses[0].events if e.event_type == "tool_result")
    assert result_ev.output_tokens == 50


def test_get_run_missing_returns_none(store):
    assert store.get_run("nonexistent") is None


def test_save_run_idempotent(store):
    """Saving the same run_id twice replaces, not duplicates."""
    tree = make_tree(n_hypotheses=1)
    store.save_run(tree)
    store.save_run(tree)

    loaded = store.get_run(tree.run_id)
    assert len(loaded.hypotheses) == 1


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------


def test_list_runs_empty(store):
    assert store.list_runs() == []


def test_list_runs_returns_all_saved(store):
    for i in range(3):
        store.save_run(make_tree(task=f"task {i}"))

    summaries = store.list_runs()
    assert len(summaries) == 3


def test_list_runs_respects_last_n(store):
    for i in range(5):
        store.save_run(make_tree(task=f"task {i}"))

    summaries = store.list_runs(last_n=3)
    assert len(summaries) == 3


def test_list_runs_summary_fields(store):
    tree = make_tree(task="the task", final_score=0.9)
    store.save_run(tree)

    s = store.list_runs()[0]
    assert isinstance(s, RunSummary)
    assert s.run_id == tree.run_id
    assert s.task == "the task"
    assert s.final_score == pytest.approx(0.9)
    assert s.completed_at != ""


def test_list_runs_ordered_newest_first(store):
    """Three sequential saves — list should return newest (last saved) first."""
    ids = []
    for i in range(3):
        tree = make_tree(task=f"task {i}")
        store.save_run(tree)
        ids.append(tree.run_id)

    summaries = store.list_runs()
    # newest is last inserted
    assert summaries[0].run_id == ids[-1]
