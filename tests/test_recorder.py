"""Tests for ChurnRecorder and AgentEvent."""
from churn.recorder import AgentEvent, ChurnRecorder


# ---------------------------------------------------------------------------
# AgentEvent
# ---------------------------------------------------------------------------


def test_agent_event_defaults():
    ev = AgentEvent(run_id="abc", event_type="tool_call", hypothesis_id=1)
    assert ev.tool_name is None
    assert ev.input_tokens == 0
    assert ev.score is None
    assert ev.timestamp is not None


# ---------------------------------------------------------------------------
# ChurnRecorder.start_run / end_run
# ---------------------------------------------------------------------------


def test_start_run_returns_id():
    recorder = ChurnRecorder()
    run_id = recorder.start_run("do something")
    assert isinstance(run_id, str)
    assert len(run_id) == 12


def test_start_run_resets_state():
    recorder = ChurnRecorder()
    recorder.start_run("first task")
    recorder.step("tool1", lambda: "a")
    # Start a second run — state must be clean
    run_id2 = recorder.start_run("second task")
    assert recorder._hypothesis_id == 0
    assert recorder.events == []
    assert recorder._run_id == run_id2


def test_end_run_emits_run_complete():
    recorder = ChurnRecorder()
    recorder.start_run("task")
    recorder.end_run("ignored", final_score=0.9)
    last = recorder.events[-1]
    assert last.event_type == "run_complete"
    assert last.score == 0.9


# ---------------------------------------------------------------------------
# Hypothesis boundary detection
# ---------------------------------------------------------------------------


def test_first_step_opens_hypothesis():
    recorder = ChurnRecorder()
    recorder.start_run("task")
    recorder.step("read", lambda: "data")

    types = [e.event_type for e in recorder.events]
    assert types[0] == "hypothesis_start"
    assert types[1] == "tool_call"
    assert types[2] == "tool_result"


def test_consecutive_steps_same_hypothesis():
    recorder = ChurnRecorder()
    recorder.start_run("task")
    recorder.step("read", lambda: "r1")
    recorder.step("write", lambda: "w1")

    h_ids = {e.hypothesis_id for e in recorder.events if e.event_type != "run_complete"}
    assert h_ids == {1}  # all in hypothesis 1


def test_text_response_starts_new_hypothesis():
    recorder = ChurnRecorder()
    recorder.start_run("task")

    recorder.step("read", lambda: "r1")
    recorder.notify_text_response()   # closes H1
    recorder.step("write", lambda: "w1")  # opens H2

    starts = [e for e in recorder.events if e.event_type == "hypothesis_start"]
    assert len(starts) == 2
    assert starts[0].hypothesis_id == 1
    assert starts[1].hypothesis_id == 2


def test_hypothesis_end_emitted_on_text_response():
    recorder = ChurnRecorder()
    recorder.start_run("task")
    recorder.step("read", lambda: "r")
    recorder.notify_text_response()

    types = [e.event_type for e in recorder.events]
    assert "hypothesis_end" in types


def test_no_duplicate_hypothesis_end_without_tools():
    """Calling notify_text_response before any tools emits no hypothesis_end."""
    recorder = ChurnRecorder()
    recorder.start_run("task")
    recorder.notify_text_response()  # no tools yet → no hypothesis_end

    types = [e.event_type for e in recorder.events]
    assert "hypothesis_end" not in types


# ---------------------------------------------------------------------------
# Tool output truncation
# ---------------------------------------------------------------------------


def test_tool_output_truncated():
    recorder = ChurnRecorder()
    recorder.start_run("task")
    long_output = "x" * 1000
    recorder.step("big_tool", lambda: long_output)

    result_ev = next(e for e in recorder.events if e.event_type == "tool_result")
    assert len(result_ev.tool_output) == 500


def test_tool_output_short_not_truncated():
    recorder = ChurnRecorder()
    recorder.start_run("task")
    recorder.step("small_tool", lambda: "hi")

    result_ev = next(e for e in recorder.events if e.event_type == "tool_result")
    assert result_ev.tool_output == "hi"


# ---------------------------------------------------------------------------
# Token counts and scores
# ---------------------------------------------------------------------------


def test_token_counts_recorded():
    recorder = ChurnRecorder()
    recorder.start_run("task")
    recorder.step("tool", lambda: "ok", input_tokens=42, output_tokens=17)

    call_ev = next(e for e in recorder.events if e.event_type == "tool_call")
    result_ev = next(e for e in recorder.events if e.event_type == "tool_result")
    assert call_ev.input_tokens == 42
    assert result_ev.output_tokens == 17


def test_score_attached_to_result():
    recorder = ChurnRecorder()
    recorder.start_run("task")
    recorder.step("tool", lambda: "ok", score=0.75)

    result_ev = next(e for e in recorder.events if e.event_type == "tool_result")
    assert result_ev.score == 0.75


# ---------------------------------------------------------------------------
# Passthrough: step() returns the wrapped function's value
# ---------------------------------------------------------------------------


def test_step_returns_value():
    recorder = ChurnRecorder()
    recorder.start_run("task")
    result = recorder.step("add", lambda a, b: a + b, 3, 4)
    assert result == 7


def test_step_kwargs_forwarded():
    recorder = ChurnRecorder()
    recorder.start_run("task")
    result = recorder.step("greet", lambda greeting, name: f"{greeting} {name}", greeting="hello", name="world")
    assert result == "hello world"


# ---------------------------------------------------------------------------
# run_id propagated to all events
# ---------------------------------------------------------------------------


def test_run_id_on_all_events():
    recorder = ChurnRecorder()
    run_id = recorder.start_run("task")
    recorder.step("tool", lambda: "ok")
    recorder.end_run(run_id)

    for ev in recorder.events:
        assert ev.run_id == run_id


# ---------------------------------------------------------------------------
# 3-step acceptance test (from issue)
# ---------------------------------------------------------------------------


def test_three_step_run_structure():
    """Run a 3-step task and verify event structure matches expectations."""
    recorder = ChurnRecorder()
    run_id = recorder.start_run("fix the bug")

    recorder.step("read_file", lambda p: f"contents of {p}", "db.py", input_tokens=100, output_tokens=420)
    recorder.notify_text_response()

    recorder.step("run_grep", lambda _cmd: "no match", "grep -n foo db.py", input_tokens=50, output_tokens=180)
    recorder.notify_text_response()

    recorder.step("edit_file", lambda _p, _c: "ok", "db.py", "new content", input_tokens=80, output_tokens=340)
    recorder.end_run(run_id, final_score=1.0)

    # 3 hypotheses
    starts = [e for e in recorder.events if e.event_type == "hypothesis_start"]
    assert len(starts) == 3

    # Each hypothesis has correct id
    assert [e.hypothesis_id for e in starts] == [1, 2, 3]

    # 3 tool_call + 3 tool_result pairs
    assert sum(1 for e in recorder.events if e.event_type == "tool_call") == 3
    assert sum(1 for e in recorder.events if e.event_type == "tool_result") == 3

    # hypothesis boundaries are correct
    ends = [e for e in recorder.events if e.event_type == "hypothesis_end"]
    assert len(ends) == 2  # H1 and H2 closed; H3 not explicitly closed

    # final event is run_complete with score 1.0
    assert recorder.events[-1].event_type == "run_complete"
    assert recorder.events[-1].score == 1.0
