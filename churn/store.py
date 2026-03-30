from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from churn.recorder import AgentEvent
from churn.tree import ExplorationTree, Hypothesis

_DEFAULT_DB = Path.home() / ".churn" / "runs.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    task            TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'complete',
    final_score     REAL,
    churn_index     REAL,
    total_tokens    INTEGER,
    total_cost_usd  REAL,
    duration_seconds REAL,
    completed_at    TEXT
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id               INTEGER NOT NULL,
    run_id           TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    description      TEXT,
    score            REAL,
    score_delta      REAL,
    token_count      INTEGER,
    duration_seconds REAL,
    PRIMARY KEY (run_id, id)
);

CREATE TABLE IF NOT EXISTS events (
    rowid           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    hypothesis_id   INTEGER NOT NULL,
    event_type      TEXT NOT NULL,
    tool_name       TEXT,
    tool_input      TEXT,
    tool_output     TEXT,
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    score           REAL,
    timestamp       TEXT NOT NULL
);
"""


@dataclass
class RunSummary:
    run_id: str
    task: str
    final_score: float
    churn_index: float
    total_tokens: int
    total_cost_usd: float
    completed_at: str


class ChurnStore:
    """SQLite-backed persistence for ExplorationTree runs."""

    def __init__(self, db_path: Path = _DEFAULT_DB) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_run(self, tree: ExplorationTree) -> None:
        """Persist a full ExplorationTree (runs + hypotheses + events)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                    (id, task, status, final_score, churn_index, total_tokens,
                     total_cost_usd, duration_seconds, completed_at)
                VALUES (?, ?, 'complete', ?, ?, ?, ?, ?, ?)
                """,
                (
                    tree.run_id,
                    tree.task,
                    tree.final_score,
                    tree.churn_index,
                    tree.total_tokens,
                    tree.total_cost_usd,
                    tree.duration_seconds,
                    now,
                ),
            )
            for h in tree.hypotheses:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO hypotheses
                        (id, run_id, description, score, score_delta, token_count, duration_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (h.id, tree.run_id, h.description, h.score, h.score_delta, h.token_count, h.duration_seconds),
                )
                for ev in h.events:
                    conn.execute(
                        """
                        INSERT INTO events
                            (run_id, hypothesis_id, event_type, tool_name, tool_input,
                             tool_output, input_tokens, output_tokens, score, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            ev.run_id,
                            ev.hypothesis_id,
                            ev.event_type,
                            ev.tool_name,
                            json.dumps(ev.tool_input) if ev.tool_input is not None else None,
                            ev.tool_output,
                            ev.input_tokens,
                            ev.output_tokens,
                            ev.score,
                            ev.timestamp.isoformat(),
                        ),
                    )

    def get_run(self, run_id: str) -> ExplorationTree | None:
        """Reconstruct an ExplorationTree from the database."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, task, final_score, churn_index, total_tokens, total_cost_usd, duration_seconds "
                "FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                return None

            r_id, task, final_score, churn_index, total_tokens, total_cost_usd, duration_seconds = row

            h_rows = conn.execute(
                "SELECT id, description, score, score_delta, token_count, duration_seconds "
                "FROM hypotheses WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()

            ev_rows = conn.execute(
                "SELECT hypothesis_id, event_type, tool_name, tool_input, tool_output, "
                "input_tokens, output_tokens, score, timestamp "
                "FROM events WHERE run_id = ? ORDER BY rowid",
                (run_id,),
            ).fetchall()

        # Group events by hypothesis_id
        ev_by_h: dict[int, list[AgentEvent]] = {}
        for ev_row in ev_rows:
            h_id, event_type, tool_name, tool_input_raw, tool_output, in_tok, out_tok, score, ts = ev_row
            ev_by_h.setdefault(h_id, []).append(
                AgentEvent(
                    run_id=run_id,
                    event_type=event_type,
                    hypothesis_id=h_id,
                    tool_name=tool_name,
                    tool_input=json.loads(tool_input_raw) if tool_input_raw else None,
                    tool_output=tool_output,
                    input_tokens=in_tok or 0,
                    output_tokens=out_tok or 0,
                    score=score,
                    timestamp=datetime.fromisoformat(ts),
                )
            )

        hypotheses = [
            Hypothesis(
                id=h_id,
                description=desc or "",
                events=ev_by_h.get(h_id, []),
                score=h_score or 0.0,
                score_delta=h_delta or 0.0,
                token_count=h_tokens or 0,
                duration_seconds=h_dur or 0.0,
            )
            for h_id, desc, h_score, h_delta, h_tokens, h_dur in h_rows
        ]

        return ExplorationTree(
            run_id=r_id,
            task=task,
            hypotheses=hypotheses,
            final_score=final_score or 0.0,
            total_tokens=total_tokens or 0,
            total_cost_usd=total_cost_usd or 0.0,
            duration_seconds=duration_seconds or 0.0,
            churn_index=churn_index or 1.0,
        )

    def list_runs(self, last_n: int = 20) -> list[RunSummary]:
        """Return the most recent runs as lightweight summaries."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, task, final_score, churn_index, total_tokens, total_cost_usd, completed_at
                FROM runs ORDER BY completed_at DESC LIMIT ?
                """,
                (last_n,),
            ).fetchall()
        return [
            RunSummary(
                run_id=row[0],
                task=row[1],
                final_score=row[2] or 0.0,
                churn_index=row[3] or 1.0,
                total_tokens=row[4] or 0,
                total_cost_usd=row[5] or 0.0,
                completed_at=row[6] or "",
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
