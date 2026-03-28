import click


@click.group()
def cli() -> None:
    """Agent churn visualizer — make exploration visible and measurable."""


@cli.command()
@click.argument("task")
@click.option("--scorer", type=click.Choice(["heuristic", "test", "human"]), default="heuristic")
@click.option("--test-cmd", default=None, help="Test command for test-based scoring")
def run(task: str, scorer: str, test_cmd: str | None) -> None:
    """Run an agent task with churn recording."""
    click.echo(f"Recording churn for: {task!r}  [scorer={scorer}]")


@cli.command()
@click.argument("run_id")
@click.option("--verbose", is_flag=True)
def inspect(run_id: str, verbose: bool) -> None:
    """Inspect a recorded run by ID."""
    click.echo(f"Inspecting run: {run_id}")


@cli.command()
@click.argument("id1")
@click.argument("id2")
def compare(id1: str, id2: str) -> None:
    """Compare two runs side-by-side."""
    click.echo(f"Comparing {id1} vs {id2}")


@cli.command()
def history() -> None:
    """List the last 20 recorded runs."""
    click.echo("No runs recorded yet.")


@cli.command()
def stats() -> None:
    """Show aggregate stats across all runs."""
    click.echo("No runs recorded yet.")
