from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


@dataclass
class AgentEvent:
    run_id: str
    event_type: str  # hypothesis_start | tool_call | tool_result | hypothesis_end | run_complete
    hypothesis_id: int
    tool_name: str | None = None
    tool_input: dict | None = None
    tool_output: str | None = None  # truncated at 500 chars
    input_tokens: int = 0
    output_tokens: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    score: float | None = None


class ChurnRecorder:
    """Transparent recording wrapper around an agent's step() calls.

    Usage::

        recorder = ChurnRecorder()
        run_id = recorder.start_run("fix the duplicate key error")

        # Drop-in replacement for nano_agent.step()
        result = recorder.step("read_db", read_file, "p3/database.py")

        recorder.end_run(run_id, final_score=1.0)
        events = recorder.events  # flush to store
    """

    _TOOL_OUTPUT_MAX = 500

    def __init__(self) -> None:
        self.events: list[AgentEvent] = []
        self._run_id: str | None = None
        self._hypothesis_id: int = 0
        self._after_text_response: bool = True  # start ready for first hypothesis

    def start_run(self, task: str) -> str:  # noqa: ARG002  # task stored externally in ExplorationTree
        """Begin a new recording session. Returns the run_id."""
        self._run_id = uuid.uuid4().hex[:12]
        self._hypothesis_id = 0
        self._after_text_response = True
        self.events = []
        return self._run_id

    def end_run(self, run_id: str, final_score: float | None = None) -> None:  # noqa: ARG002  # run_id already set via start_run
        """Close the current run and record a run_complete event."""
        self._record(
            event_type="run_complete",
            score=final_score,
        )

    def notify_text_response(self) -> None:
        """Call this after the LLM emits a text response (closes current hypothesis).

        In a real agent loop the orchestrator calls this between reasoning
        cycles so the recorder can detect hypothesis boundaries.
        """
        if not self._after_text_response:
            self._record(event_type="hypothesis_end")
        self._after_text_response = True

    def step(
        self,
        label: str,
        func: Callable[..., Any],
        *args: Any,
        input_tokens: int = 0,
        output_tokens: int = 0,
        score: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute one tool call and record input + output events.

        Args:
            label: Human-readable name for this tool call.
            func: The tool function to invoke.
            *args: Positional arguments forwarded to func.
            input_tokens: Token count for the tool input (optional).
            output_tokens: Token count for the tool output (optional).
            score: Score to attach to the tool_result event (optional).
            **kwargs: Keyword arguments forwarded to func.

        Returns:
            The return value of func(*args, **kwargs).
        """
        # New hypothesis starts on first tool call after a text response
        if self._after_text_response:
            self._hypothesis_id += 1
            self._record(event_type="hypothesis_start")
            self._after_text_response = False

        self._record(
            event_type="tool_call",
            tool_name=label,
            tool_input={"args": list(args), "kwargs": kwargs},
            input_tokens=input_tokens,
        )

        result = func(*args, **kwargs)

        raw = str(result) if result is not None else ""
        truncated = raw[: self._TOOL_OUTPUT_MAX]

        self._record(
            event_type="tool_result",
            tool_name=label,
            tool_output=truncated,
            output_tokens=output_tokens,
            score=score,
        )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record(
        self,
        event_type: str,
        tool_name: str | None = None,
        tool_input: dict | None = None,
        tool_output: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        score: float | None = None,
    ) -> None:
        self.events.append(
            AgentEvent(
                run_id=self._run_id or "",
                event_type=event_type,
                hypothesis_id=self._hypothesis_id,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=tool_output,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                score=score,
            )
        )
