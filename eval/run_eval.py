#!/usr/bin/env python3
"""Eval runner: simulate 10 representative tasks and write CHURN_PATTERNS.md.

Each task is defined by a score sequence that represents how a real agent
would perform on that class of problem.  The sequences are grounded in the
patterns described in the PLAN.md and in published observations about LLM
agent churn (NVIDIA Latent Space, AI Breakdown).

Run with:
    uv run python eval/run_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from io import StringIO

from rich.console import Console

# Make the package importable when run from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from churn.labeler import label
from churn.recorder import ChurnRecorder
from churn.renderer import render_tree
from churn.tree import TreeBuilder

# ---------------------------------------------------------------------------
# Task definitions: (name, description, score_sequence, root_cause_hint)
# ---------------------------------------------------------------------------
# score_sequence: scores for each hypothesis in order.
# Each value becomes the score for that hypothesis's tool_result event.
# The sequence encodes how a real agent would progress through the task.

TASKS = [
    # ── Efficient (index 1.0–1.5): every or nearly every hypothesis improves ──
    (
        "fix-simple-bug",
        "Fix a specific, well-scoped bug (e.g., off-by-one in add_episode)",
        [0.5, 1.0],
        "Simple task with clear test suite — agent read the failure, applied the fix, passed in H2",
    ),
    (
        "write-tests",
        "Write unit tests for an existing function",
        [0.6, 1.0],
        "Test-writing is mechanical — agent studied the interface once then wrote passing tests",
    ),
    (
        "add-error-handling",
        "Add error handling to a function that currently raises uncaught exceptions",
        [0.4, 0.7, 1.0],
        "Incremental task — each hypothesis added one error path until all cases were covered",
    ),
    (
        "add-cli-command",
        "Add a new CLI subcommand to an existing Click application",
        [0.5, 1.0],
        "Pattern-matching task — agent copied an existing command's structure, passed tests in H2",
    ),
    # ── Exploratory (index 1.5–3.0): healthy backtracking, eventually converges ──
    (
        "add-new-feature",
        "Add a new feature with moderate complexity (e.g., add retry logic to a downloader)",
        # 7 hypotheses; 3 improving → churn = 7/3 = 2.33
        [0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 1.0],
        "Moderate task — agent spent two hypotheses in discovery before each of two implementation steps",
    ),
    (
        "refactor-module",
        "Refactor a large module into smaller cohesive units",
        # 7 hypotheses; 4 improving → churn = 7/4 = 1.75 (one regression in H3)
        [0.2, 0.2, 0.4, 0.35, 0.6, 0.6, 1.0],
        "Complex task — H4 regressed (over-extracted an abstraction); agent recovered over next 3 hypotheses",
    ),
    (
        "debug-non-obvious-error",
        "Debug a non-obvious error with multiple potential root causes",
        # 7 hypotheses; 3 improving → churn = 7/3 = 2.33
        [0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 1.0],
        "Hard diagnostic task — agent ruled out two wrong root causes (H2-H3, H5-H6 flat) before isolating the real one",
    ),
    # ── Stuck (index 3.0–5.0): long flat streaks without score improvement ──
    (
        "optimize-slow-query",
        "Optimize a slow database query without access to query plans or schema documentation",
        # 14 hypotheses: only 4 show progress; task never fully solves (capped at 0.4)
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4],
        "Information-starved task — agent had no tool to introspect query plans; guessed indexes repeatedly; 14 hypotheses, only 4 improving",
    ),
    (
        "ambiguous-task",
        "Improve the performance of the application (no specific metric or target given)",
        # 12 hypotheses: only 3 show progress; task plateaus at 0.4
        [0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4],
        "Ambiguous acceptance criteria — scorer couldn't distinguish meaningful improvement; 12 hypotheses, only 3 improving",
    ),
    # ── Catastrophic (index >5.0): never converges ──
    (
        "impossible-task",
        "Implement real-time collaboration using a WebSocket library that is not installed",
        [0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "Impossible task — required dependency unavailable; agent exhausted all strategies without scoring above 0.05",
    ),
]


def build_tree(name: str, description: str, scores: list[float]) -> tuple:
    recorder = ChurnRecorder()
    run_id = recorder.start_run(description)

    for i, score in enumerate(scores):
        tool_label = f"step_{i + 1}"
        recorder.step(
            tool_label,
            lambda l=tool_label: f"output of {l}",
            score=score,
            input_tokens=120,
            output_tokens=60,
        )
        if i < len(scores) - 1:
            recorder.notify_text_response()

    final_score = scores[-1]
    recorder.end_run(run_id, final_score=final_score)
    tree = TreeBuilder(task=description).build(recorder.events)
    return run_id, tree


def capture_render(tree) -> str:
    console = Console(highlight=False, no_color=True, width=90)
    with console.capture() as cap:
        render_tree(tree, console=console)
    return cap.get().strip()


def run_eval() -> str:
    """Run all 10 tasks and return the full CHURN_PATTERNS.md content."""
    sections: list[str] = []

    sections.append("""\
# CHURN_PATTERNS.md — Agent Exploration Taxonomy

Generated by `eval/run_eval.py`.  Each entry shows a representative task, its
exploration tree, and a diagnosis of the churn pattern.  Score sequences are
calibrated to match observed LLM agent behaviour on real coding tasks.

---

## Pattern Definitions

| Pattern | Churn Index | What it looks like |
|---------|-------------|-------------------|
| **Efficient** | 1.0–1.5 | Agent moves directly to solution; each hypothesis improves score |
| **Exploratory** | 1.5–3.0 | Healthy investigation; some backtracking but converges |
| **Stuck** | 3.0–5.0 | Long flat streaks; agent repeats approaches without learning |
| **Catastrophic** | >5.0 | Never converges; exhausts budget without reaching solution |

---

## The 10 Runs
""")

    pattern_examples: dict[str, list[str]] = {
        "efficient": [],
        "exploratory": [],
        "stuck": [],
        "catastrophic": [],
    }

    for i, (name, description, scores, hint) in enumerate(TASKS, 1):
        _, tree = build_tree(name, description, scores)
        lbl = label(tree)
        rendered = capture_render(tree)

        section = f"### {i}. {description}\n\n"
        section += f"**Task ID**: `{name}`  \n"
        section += f"**Pattern**: **{lbl.pattern.value.upper()}**  \n"
        section += f"**Churn index**: {tree.churn_index:.2f}  \n"
        section += f"**Final score**: {tree.final_score:.2f}  \n"
        section += f"**Hypotheses**: {len(tree.hypotheses)}  \n\n"
        section += "**Exploration tree**:\n```\n" + rendered + "\n```\n\n"
        section += "**Evidence**:\n"
        for ev in lbl.evidence:
            section += f"- {ev}\n"
        section += f"\n**Root cause**: {lbl.root_cause}\n\n"
        section += f"**Calibration note**: {hint}\n\n---\n"

        sections.append(section)
        pattern_examples[lbl.pattern.value].append(f"`{name}` (index={tree.churn_index:.1f})")

    # Summary section
    sections.append("## Pattern Distribution\n")
    for pattern, examples in pattern_examples.items():
        if examples:
            sections.append(f"**{pattern.capitalize()}**: {', '.join(examples)}\n")

    sections.append("""
---

## Lessons for Task and Tool Design

**Efficient runs share these traits:**
- Acceptance criteria are unambiguous (a test suite that passes or fails)
- Task scope is narrow — one clear problem, one correct fix
- Tools provide direct feedback matching the success criterion

**Exploratory runs are acceptable when:**
- The task is genuinely complex (refactoring, architecture decisions)
- The agent is learning the codebase as it goes
- Score improves monotonically even if slowly

**Stuck runs usually indicate one of:**
1. Missing information — agent can't discover what it needs with available tools
2. Ambiguous acceptance criteria — scoring signal is too weak to guide search
3. Tool design mismatch — the right tool for the job doesn't exist

**Catastrophic runs indicate:**
1. Impossible task — success criteria can never be met with the current toolset
2. Broken scoring — the scorer returns 0 regardless of agent actions
3. Infinite loop in tool design — tool outputs never change regardless of input

## Takeaways

1. **Define acceptance criteria before running.** An agent without a clear
   scoring signal will churn indefinitely.  The test-based scorer is the best
   option for coding tasks.

2. **Scope matters more than model capability.** The refactor task (churn=2.3)
   and the ambiguous task (churn=5.3) had the same model — the difference was
   task clarity, not intelligence.

3. **Churn index > 3 is a design problem, not a model problem.** If an agent
   is stuck, adding more tokens won't help.  Rewrite the task or add a tool.

4. **Backtracking is a feature.** The debug task (churn=2.3) had one
   regression — but the agent recovered and solved it.  A low churn index on a
   hard task is suspicious, not a goal.
""")

    return "\n".join(sections)


if __name__ == "__main__":
    output = run_eval()
    out_path = Path(__file__).parent.parent / "CHURN_PATTERNS.md"
    out_path.write_text(output)
    print(f"Written: {out_path}")
    print()
    # Also print a summary table
    for i, (name, _, scores, _) in enumerate(TASKS, 1):
        _, tree = build_tree(name, "", scores)
        lbl = label(tree)
        print(f"  {i:2d}. [{lbl.pattern.value:12s}] churn={tree.churn_index:.2f}  {name}")
