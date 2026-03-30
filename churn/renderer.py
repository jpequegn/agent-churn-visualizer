from __future__ import annotations

from rich.console import Console
from rich.text import Text
from rich.tree import Tree

from churn.tree import ExplorationTree, Hypothesis

# Score thresholds for colour coding
_GREEN = "green"
_YELLOW = "yellow"
_RED = "red"


def _score_color(delta: float) -> str:
    if delta > 0:
        return _GREEN
    if delta == 0:
        return _YELLOW
    return _RED


def _churn_label(index: float) -> str:
    if index <= 1.2:
        return "efficient"
    if index <= 2.5:
        return "exploratory"
    if index <= 5.0:
        return "high churn"
    return "lost"


def render_tree(exploration: ExplorationTree, console: Console | None = None) -> None:
    """Print an ExplorationTree to the terminal using rich."""
    if console is None:
        console = Console()

    # ── Header ────────────────────────────────────────────────────────────
    cost_str = f"${exploration.total_cost_usd:.4f}"
    tok_str = f"{exploration.total_tokens:,} tok"
    header = Text()
    header.append("Task: ", style="bold")
    header.append(exploration.task, style="bold white")
    header.append(f"  [{cost_str} | {tok_str}]", style="dim")
    root = Tree(header)

    # ── Hypotheses ────────────────────────────────────────────────────────
    for h in exploration.hypotheses:
        node = root.add(_hypothesis_label(h))
        # Summary line: first tool description
        desc_line = Text(f"  {h.description}", style="dim italic")
        node.add(desc_line)

    # ── Footer ────────────────────────────────────────────────────────────
    n = len(exploration.hypotheses)
    tool_calls = sum(
        1 for h in exploration.hypotheses for e in h.events if e.event_type == "tool_call"
    )
    churn_label = _churn_label(exploration.churn_index)
    footer = Text()
    footer.append(f"Churn index: {exploration.churn_index:.1f}", style="bold")
    footer.append(f" ({churn_label})", style="dim")
    footer.append(f"  |  {n} hypothesis{'es' if n != 1 else ''}", style="dim")
    footer.append(f"  |  {tool_calls} tool call{'s' if tool_calls != 1 else ''}", style="dim")

    console.print()
    console.print(root)
    console.print(footer)
    console.print()


def _hypothesis_label(h: Hypothesis) -> Text:
    color = _score_color(h.score_delta)
    label = Text()
    label.append(f"[H{h.id}] ", style=f"bold {color}")
    label.append(f"score:{h.score:.2f} ", style=color)
    delta_sign = "+" if h.score_delta >= 0 else ""
    label.append(f"({delta_sign}{h.score_delta:.2f})", style=f"dim {color}")

    # Tool call counts
    tool_counts: dict[str, int] = {}
    for ev in h.events:
        if ev.event_type == "tool_call" and ev.tool_name:
            tool_counts[ev.tool_name] = tool_counts.get(ev.tool_name, 0) + 1

    if tool_counts:
        parts = "  ".join(f"{name}×{n}" for name, n in tool_counts.items())
        label.append(f"  ── {parts}", style="dim")

    label.append(f"  {h.token_count:,} tok", style="dim")

    # Status badge
    if h.score >= 1.0:
        label.append("  ✓ solved", style="bold green")
    elif h.score_delta < 0:
        label.append("  ✗ regressed", style="bold red")
    elif h.score_delta == 0 and h.score > 0:
        label.append("  ~ no change", style="yellow")

    return label
