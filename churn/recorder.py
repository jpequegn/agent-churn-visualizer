from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


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
    timestamp: datetime = field(default_factory=datetime.utcnow)
    score: float | None = None
