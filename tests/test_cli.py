"""End-to-end CLI tests using Click's test runner."""
import pytest
from click.testing import CliRunner

from churn.cli import cli
from churn.recorder import ChurnRecorder
from churn.store import ChurnStore
from churn.tree import TreeBuilder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def store(tmp_path):
    return ChurnStore(db_path=tmp_path / "test.db")


def _saved_run(store: ChurnStore, task: str = "test task", steps: int = 2) -> str:
    """Record a minimal run, save it, and return the run_id."""
    recorder = ChurnRecorder()
    run_id = recorder.start_run(task)
    for i in range(steps):
        score = (i + 1) / steps
        recorder.step(f"tool_{i}", lambda: "ok", score=score, input_tokens=50, output_tokens=25)
        if i < steps - 1:
            recorder.notify_text_response()
    recorder.end_run(run_id, final_score=1.0)
    tree = TreeBuilder(task=task).build(recorder.events)
    store.save_run(tree)
    return run_id


# ---------------------------------------------------------------------------
# churn --help
# ---------------------------------------------------------------------------


def test_cli_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "inspect" in result.output
    assert "history" in result.output
    assert "compare" in result.output
    assert "stats" in result.output


# ---------------------------------------------------------------------------
# churn run
# ---------------------------------------------------------------------------


def test_run_default_scorer(runner, tmp_path, monkeypatch):
    monkeypatch.setattr("churn.cli._store", lambda: ChurnStore(db_path=tmp_path / "r.db"))
    result = runner.invoke(cli, ["run", "fix the bug", "--steps", "2"])
    assert result.exit_code == 0, result.output
    assert "fix the bug" in result.output
    assert "Churn index" in result.output
    assert "Saved run" in result.output


def test_run_creates_db_entry(runner, tmp_path, monkeypatch):
    db = tmp_path / "r.db"
    monkeypatch.setattr("churn.cli._store", lambda: ChurnStore(db_path=db))
    runner.invoke(cli, ["run", "the task", "--steps", "3"])
    store = ChurnStore(db_path=db)
    summaries = store.list_runs()
    assert len(summaries) == 1
    assert summaries[0].task == "the task"


def test_run_command_scorer_requires_test_cmd(runner, tmp_path, monkeypatch):
    monkeypatch.setattr("churn.cli._store", lambda: ChurnStore(db_path=tmp_path / "r.db"))
    result = runner.invoke(cli, ["run", "task", "--scorer", "command"])
    assert result.exit_code != 0
    assert "test-cmd" in result.output.lower() or "required" in result.output.lower()


# ---------------------------------------------------------------------------
# churn inspect
# ---------------------------------------------------------------------------


def test_inspect_existing_run(runner, tmp_path, monkeypatch, store):
    monkeypatch.setattr("churn.cli._store", lambda: store)
    run_id = _saved_run(store, task="inspect me")
    result = runner.invoke(cli, ["inspect", run_id])
    assert result.exit_code == 0, result.output
    assert "inspect me" in result.output
    assert "H1" in result.output


def test_inspect_verbose_shows_events(runner, tmp_path, monkeypatch, store):
    monkeypatch.setattr("churn.cli._store", lambda: store)
    run_id = _saved_run(store, task="verbose task")
    result = runner.invoke(cli, ["inspect", run_id, "--verbose"])
    assert result.exit_code == 0, result.output
    assert "tool_call" in result.output or "tool_result" in result.output


def test_inspect_missing_run(runner, tmp_path, monkeypatch, store):
    monkeypatch.setattr("churn.cli._store", lambda: store)
    result = runner.invoke(cli, ["inspect", "nonexistent"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# churn history
# ---------------------------------------------------------------------------


def test_history_empty(runner, tmp_path, monkeypatch, store):
    monkeypatch.setattr("churn.cli._store", lambda: store)
    result = runner.invoke(cli, ["history"])
    assert result.exit_code == 0
    assert "no runs" in result.output.lower()


def test_history_shows_runs(runner, tmp_path, monkeypatch, store):
    monkeypatch.setattr("churn.cli._store", lambda: store)
    _saved_run(store, task="task one")
    _saved_run(store, task="task two")
    result = runner.invoke(cli, ["history"])
    assert result.exit_code == 0, result.output
    assert "task one" in result.output
    assert "task two" in result.output


def test_history_respects_last_flag(runner, tmp_path, monkeypatch, store):
    monkeypatch.setattr("churn.cli._store", lambda: store)
    for i in range(5):
        _saved_run(store, task=f"task {i}")
    result = runner.invoke(cli, ["history", "--last", "2"])
    assert result.exit_code == 0
    # Data rows start with a run_id (hex); count those — should be exactly 2
    data_lines = [l for l in result.output.splitlines() if l.strip() and l.strip()[0:8].isalnum() and " " in l.strip()]
    assert len(data_lines) == 2


# ---------------------------------------------------------------------------
# churn compare
# ---------------------------------------------------------------------------


def test_compare_two_runs(runner, tmp_path, monkeypatch, store):
    monkeypatch.setattr("churn.cli._store", lambda: store)
    id1 = _saved_run(store, task="first task")
    id2 = _saved_run(store, task="second task")
    result = runner.invoke(cli, ["compare", id1, id2])
    assert result.exit_code == 0, result.output
    assert id1 in result.output
    assert id2 in result.output


def test_compare_missing_run(runner, tmp_path, monkeypatch, store):
    monkeypatch.setattr("churn.cli._store", lambda: store)
    id1 = _saved_run(store)
    result = runner.invoke(cli, ["compare", id1, "badid"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# churn stats
# ---------------------------------------------------------------------------


def test_stats_empty(runner, tmp_path, monkeypatch, store):
    monkeypatch.setattr("churn.cli._store", lambda: store)
    result = runner.invoke(cli, ["stats"])
    assert result.exit_code == 0
    assert "no runs" in result.output.lower()


def test_stats_with_runs(runner, tmp_path, monkeypatch, store):
    monkeypatch.setattr("churn.cli._store", lambda: store)
    _saved_run(store, task="task a")
    _saved_run(store, task="task b")
    result = runner.invoke(cli, ["stats"])
    assert result.exit_code == 0, result.output
    assert "Runs:" in result.output
    assert "Avg churn" in result.output
    assert "Total tokens" in result.output
