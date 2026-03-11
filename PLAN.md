# Agent Churn Visualizer — Implementation Plan

## What We're Building

An observability layer that wraps any agent loop, records every hypothesis and tool call, scores each attempt, and renders the exploration tree in the terminal. Makes "churn" — the exploratory behavior of capable agents — visible and measurable.

## Why This Matters

NVIDIA engineers on Latent Space introduced a specific concept: **churn**. A capable agent solving a hard problem doesn't go straight to the answer. It explores hypotheses, backtracks, tries alternatives, refines. This churning is the signature of a capable agent — and it's invisible in most agent frameworks.

Three things happen when you make churn visible:
1. You understand *why* a run cost what it did (vs. "it used a lot of tokens")
2. You can identify where agents get stuck (a loop that generates high churn with no score improvement is a design problem)
3. You learn what good vs. bad agent exploration looks like — which shapes how you write tasks and design tools

This is a direct extension of `nano-agent` — it wraps the same loop with a recording and visualization layer.

## Core Concept: The Exploration Tree

Every agent run produces an exploration tree:

```
Task: "fix the duplicate key error in add_episode"
│
├── [Hypothesis 1] Read database.py to understand current schema
│   ├── tool: read_file("p3/database.py")         tokens: 420  score: 0.2
│   └── observation: UNIQUE constraint on url column
│
├── [Hypothesis 2] Check if pre-insert duplicate check exists
│   ├── tool: bash("grep -n 'get_episode_by_url' p3/database.py")  tokens: 180
│   └── observation: check exists but has a race condition          score: 0.4
│
├── [Hypothesis 3] Add ON CONFLICT DO NOTHING to INSERT
│   ├── tool: edit_file("p3/database.py", ...)    tokens: 340
│   ├── tool: bash("python -m pytest tests/ -x")  tokens: 520
│   └── observation: 1 test fails — sequence out of sync            score: 0.6
│
└── [Hypothesis 4] Fix sequence + ON CONFLICT
    ├── tool: edit_file("p3/database.py", ...)    tokens: 280
    ├── tool: bash("python -m pytest tests/ -x")  tokens: 510
    └── observation: all tests pass                                  score: 1.0 ✓

Total: 4 hypotheses, 7 tool calls, 2,250 tokens, $0.003, 34 seconds
Churn index: 0.62 (explored 4 paths, converged at depth 4)
```

## Architecture

```
churn/
├── __init__.py
├── recorder.py      # Wraps agent loop, records every event
├── scorer.py        # Scores each hypothesis (heuristic + human label)
├── tree.py          # HypothesisTree: data structure for exploration
├── renderer.py      # Rich terminal tree visualization
├── metrics.py       # Churn index, efficiency score, cost breakdown
├── store.py         # SQLite persistence for run history
└── cli.py           # `churn run <task>`, `churn inspect <run_id>`, `churn compare`

tests/
└── test_recorder.py

pyproject.toml
README.md
```

## Implementation Phases

### Phase 1: Event model and recorder (recorder.py)

Every meaningful event in an agent loop gets recorded:

```python
@dataclass
class AgentEvent:
    run_id: str
    event_type: str          # hypothesis_start | tool_call | tool_result | hypothesis_end | run_complete
    hypothesis_id: int        # which hypothesis this belongs to
    tool_name: str | None
    tool_input: dict | None
    tool_output: str | None   # truncated at 500 chars
    input_tokens: int
    output_tokens: int
    timestamp: datetime
    score: float | None       # set at hypothesis_end
```

`ChurnRecorder` wraps an agent's `step()` function transparently:

```python
recorder = ChurnRecorder()

# Drop-in replacement for nano-agent's step()
result = recorder.step("read_db", read_file, "p3/database.py")
# → records tool_call + tool_result events, passes through to real step()
```

Hypothesis boundaries: a new hypothesis starts when the agent issues its first tool call in a new reasoning cycle. Ends when the agent produces a text response or switches direction.

### Phase 2: Scoring (scorer.py)

Two scoring modes:

**Heuristic scorer** (automatic, no human needed):
- Score improves if test output shows fewer failures than previous hypothesis
- Score improves if error message changes (agent learned something)
- Score drops if same tool called with same args as previous hypothesis (stuck in loop)
- Score = 0.5 baseline, adjusted by above signals

```python
score = heuristic_scorer.score(hypothesis, previous_hypotheses)
# → float 0.0-1.0
```

**Human scorer** (interactive mode):
After each hypothesis completes, pause and ask: "Score this hypothesis (0-10):"
Used for building ground-truth labeled datasets.

**Test-based scorer** (best for coding tasks):
If a test command is provided, run it after each hypothesis. Score = fraction of tests passing.

```python
scorer = TestScorer(test_command="pytest tests/ -x -q")
# → automatically scores each hypothesis by running tests
```

### Phase 3: Hypothesis tree (tree.py)

```python
@dataclass
class Hypothesis:
    id: int
    description: str          # first LLM text in this hypothesis cycle
    events: list[AgentEvent]
    score: float
    score_delta: float        # improvement over previous hypothesis
    token_count: int
    duration_seconds: float
    children: list[Hypothesis]  # if agent branched (future: parallel agents)

@dataclass
class ExplorationTree:
    run_id: str
    task: str
    root_hypotheses: list[Hypothesis]
    final_score: float
    total_tokens: int
    total_cost_usd: float
    duration_seconds: float
    churn_index: float        # see metrics
```

### Phase 4: Metrics (metrics.py)

**Churn Index** — how much exploration happened relative to path length?
```
churn_index = total_hypotheses / (depth_of_winning_path + 1)
```
- Churn index = 1.0: straight line to solution, no backtracking
- Churn index = 3.0: explored 3x more hypotheses than needed
- Churn index > 5.0: agent was lost

**Efficiency Score** — did early exploration lead toward the solution?
```
efficiency = sum(score_delta for h in hypotheses if score_delta > 0) / total_hypotheses
```

**Stuck Detection** — flag if same tool + same input repeats in consecutive hypotheses.

**Cost per score point** — how much did it cost (in USD) to go from score 0 to score 1?

### Phase 5: Terminal renderer (renderer.py)

Use `rich.tree` to render the exploration tree live and post-run.

```
Task: fix the duplicate key error in add_episode          [$0.003 | 34s | 2,250 tok]
│
├── [H1] score:0.2 (+0.2) ── read_file×1  bash×0   180 tok  4s
│   └── Read database.py to understand schema
│
├── [H2] score:0.4 (+0.2) ── read_file×0  bash×1   220 tok  6s
│   └── Check for existing duplicate prevention
│
├── [H3] score:0.6 (+0.2) ── edit_file×1  bash×1   860 tok  12s  ⚠ partial
│   └── Add ON CONFLICT — tests still failing
│
└── [H4] score:1.0 (+0.4) ── edit_file×1  bash×1   790 tok  12s  ✓ solved
    └── Fix sequence alignment + conflict clause

Churn index: 1.0 (efficient)  |  4 hypotheses  |  5 tool calls
```

Color coding: green (score improved), yellow (no change), red (score dropped).

### Phase 6: SQLite store and history (store.py)

Persist every run. Schema mirrors the event model + tree structure.

```bash
churn history              # list last 20 runs with task + score + churn index
churn inspect <run_id>     # full tree for a specific run
churn compare <id1> <id2>  # side-by-side: same task, different approaches
```

### Phase 7: CLI integration with nano-agent

```bash
# Wrap any nano-agent task with churn recording
churn run "fix the duplicate key error" --scorer tests --test-cmd "pytest tests/ -x"

# Inspect a past run
churn inspect abc123 --verbose

# Compare two runs on same task
churn compare abc123 def456

# Summary stats across all runs
churn stats
# → avg churn index: 1.8 | avg hypotheses to solve: 3.2 | most common stuck pattern: same-tool-loop
```

### Phase 8: Eval — 10 tasks, label churn patterns

Run 10 tasks from the `nano-agent` eval set through the churn visualizer. For each run:
1. Record and score automatically (test-based scorer)
2. Label the churn pattern: efficient / exploratory / stuck / catastrophic
3. Identify what caused each stuck episode
4. Write `CHURN_PATTERNS.md` — a taxonomy of agent failure modes with examples

## Key Design Decisions

**Why hypothesis boundaries, not tool calls?**
A single reasoning cycle may involve multiple tool calls. The meaningful unit is the hypothesis (a line of reasoning) — not the individual tool. Grouping tool calls into hypotheses gives more insight.

**Why heuristic scoring, not just human labeling?**
Human labeling doesn't scale across 10+ runs. Heuristic scoring is imperfect but gives immediate signal. The test-based scorer (for coding tasks) is nearly perfect and fully automatic.

**Why `rich.tree` and not a web UI?**
Terminal-first. Consistent with nano-agent's philosophy. A web UI is a follow-on.

**What we're NOT building**
- Parallel branch visualization (single-agent churn only for now)
- Real-time streaming dashboard (post-run visualization only)
- Automatic task decomposition based on churn patterns

## Acceptance Criteria

1. Run 3 tasks through the visualizer — each produces a rendered exploration tree
2. Test-based scorer correctly scores each hypothesis (verified by checking test output)
3. Churn index calculated correctly: straight-line solution = 1.0, 4 hypotheses on 2-step problem = 2.0
4. `churn history` shows all 3 runs with correct stats
5. `CHURN_PATTERNS.md` committed with real-run examples, not placeholders

## Learning Outcomes

After building this you will understand:
- What agent "churn" actually looks like in practice (not just theory)
- How to instrument an agent loop for observability without changing its behavior
- What distinguishes efficient exploration from being stuck (the NVIDIA insight)
- Why defining a scoring function is the hardest part of agent design (AI Breakdown insight)
- How to design tasks and tools to minimize pathological churn
