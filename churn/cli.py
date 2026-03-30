from __future__ import annotations

import click
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from churn.recorder import ChurnRecorder
from churn.renderer import _churn_label, render_tree
from churn.scorer import CommandScorer, HeuristicScorer, HumanScorer
from churn.store import ChurnStore
from churn.tree import TreeBuilder

_console = Console()


def _store() -> ChurnStore:
    return ChurnStore()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _churn_color(index: float) -> str:
    if index < 1.5:
        return "green"
    if index < 3.0:
        return "yellow"
    return "red"


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """Agent churn visualizer — make exploration visible and measurable."""


# ---------------------------------------------------------------------------
# churn run
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("task")
@click.option(
    "--scorer",
    type=click.Choice(["heuristic", "command", "human"]),
    default="heuristic",
    show_default=True,
    help="Scoring mode.",
)
@click.option("--test-cmd", default=None, help="Shell command for command-based scoring.")
@click.option("--steps", default=3, show_default=True, help="Number of simulated steps (demo mode).")
def run(task: str, scorer: str, test_cmd: str | None, steps: int) -> None:
    """Record a churn run for TASK.

    In the current version this runs a simulated agent loop so you can
    see the visualizer in action without wiring up a real LLM.  Replace
    the body of _simulate_step() with your actual agent call.
    """
    if scorer == "command" and not test_cmd:
        raise click.UsageError("--test-cmd is required when --scorer=command")

    if scorer == "heuristic":
        active_scorer = HeuristicScorer()
    elif scorer == "command":
        active_scorer = CommandScorer(test_command=test_cmd)  # type: ignore[arg-type]
    else:
        active_scorer = HumanScorer()

    recorder = ChurnRecorder()
    run_id = recorder.start_run(task)
    previous_scores: list[float] = []

    _console.print(f"\n[bold]Running:[/bold] {task!r}  [dim](scorer={scorer})[/dim]\n")

    for i in range(steps):
        label = f"step_{i + 1}"
        recorder.step(label, lambda l=label: f"output of {l}", input_tokens=100, output_tokens=50)

        h_events = [e for e in recorder.events if e.hypothesis_id == recorder._hypothesis_id]
        score = active_scorer.score(h_events, previous_scores)
        previous_scores.append(score)

        recorder.notify_text_response()

    final_score = previous_scores[-1] if previous_scores else 0.0
    recorder.end_run(run_id, final_score=final_score)

    tree = TreeBuilder(task=task).build(recorder.events)
    render_tree(tree, console=_console)

    store = _store()
    store.save_run(tree)
    _console.print(f"[dim]Saved run [bold]{run_id}[/bold] to ~/.churn/runs.db[/dim]\n")


# ---------------------------------------------------------------------------
# churn inspect
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("run_id")
@click.option("--verbose", is_flag=True, help="Show tool inputs and outputs.")
def inspect(run_id: str, verbose: bool) -> None:
    """Show the full exploration tree for RUN_ID."""
    tree = _store().get_run(run_id)
    if tree is None:
        raise click.ClickException(f"Run '{run_id}' not found.")

    render_tree(tree, console=_console)

    if verbose:
        for h in tree.hypotheses:
            _console.print(f"\n[bold]H{h.id} events:[/bold]")
            for ev in h.events:
                if ev.event_type in ("tool_call", "tool_result"):
                    style = "cyan" if ev.event_type == "tool_call" else "dim"
                    _console.print(f"  [{style}]{ev.event_type}[/{style}]  {ev.tool_name}", end="")
                    if ev.tool_output:
                        _console.print(f"  → {ev.tool_output[:120]!r}")
                    else:
                        _console.print()


# ---------------------------------------------------------------------------
# churn history
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--last", default=20, show_default=True, help="Number of runs to show.")
def history(last: int) -> None:
    """List recent runs with task, score, churn index, and cost."""
    summaries = _store().list_runs(last_n=last)
    if not summaries:
        _console.print("[dim]No runs recorded yet.  Run [bold]churn run[/bold] to get started.[/dim]")
        return

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("run_id", style="dim", no_wrap=True)
    table.add_column("task")
    table.add_column("score", justify="right")
    table.add_column("churn", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("cost", justify="right")
    table.add_column("completed", style="dim")

    for s in summaries:
        churn_color = _churn_color(s.churn_index)
        table.add_row(
            s.run_id,
            s.task[:60],
            f"{s.final_score:.2f}",
            Text(f"{s.churn_index:.1f}  {_churn_label(s.churn_index)}", style=churn_color),
            f"{s.total_tokens:,}",
            f"${s.total_cost_usd:.4f}",
            s.completed_at[:19].replace("T", " "),
        )

    _console.print()
    _console.print(table)
    _console.print()


# ---------------------------------------------------------------------------
# churn compare
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("id1")
@click.argument("id2")
def compare(id1: str, id2: str) -> None:
    """Show two runs side-by-side."""
    store = _store()
    left = store.get_run(id1)
    right = store.get_run(id2)

    missing = [rid for rid, t in ((id1, left), (id2, right)) if t is None]
    if missing:
        raise click.ClickException(f"Run(s) not found: {', '.join(missing)}")

    def _capture(tree) -> str:
        c = Console(highlight=False, no_color=False, width=70)
        with c.capture() as cap:
            render_tree(tree, console=c)
        return cap.get()

    left_text = _capture(left)   # type: ignore[arg-type]
    right_text = _capture(right)  # type: ignore[arg-type]

    _console.print()
    _console.print(
        Columns(
            [
                Panel(left_text, title=f"[bold]{id1}[/bold]", width=72),
                Panel(right_text, title=f"[bold]{id2}[/bold]", width=72),
            ]
        )
    )
    _console.print()


# ---------------------------------------------------------------------------
# churn stats
# ---------------------------------------------------------------------------


@cli.command()
def stats() -> None:
    """Aggregate stats across all recorded runs."""
    summaries = _store().list_runs(last_n=1000)
    if not summaries:
        _console.print("[dim]No runs recorded yet.[/dim]")
        return

    avg_churn = sum(s.churn_index for s in summaries) / len(summaries)
    avg_score = sum(s.final_score for s in summaries) / len(summaries)
    total_tokens = sum(s.total_tokens for s in summaries)
    total_cost = sum(s.total_cost_usd for s in summaries)

    high_churn = [s for s in summaries if s.churn_index >= 3.0]

    _console.print()
    _console.print(f"[bold]Runs:[/bold]          {len(summaries)}")
    _console.print(f"[bold]Avg churn index:[/bold] {avg_churn:.2f}  ({_churn_label(avg_churn)})")
    _console.print(f"[bold]Avg final score:[/bold] {avg_score:.2f}")
    _console.print(f"[bold]Total tokens:[/bold]    {total_tokens:,}")
    _console.print(f"[bold]Total cost:[/bold]      ${total_cost:.4f}")
    if high_churn:
        _console.print(f"\n[yellow]High-churn runs (≥3.0):[/yellow] {len(high_churn)}")
        for s in high_churn[:5]:
            _console.print(f"  [dim]{s.run_id}[/dim]  {s.task[:50]}  churn={s.churn_index:.1f}")
    _console.print()
