"""Smoke tests verifying the project is correctly set up."""
from click.testing import CliRunner

from churn.cli import cli
from churn.nano_agent import step


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "churn" in result.output.lower()


def test_cli_run_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0


def test_cli_history():
    runner = CliRunner()
    result = runner.invoke(cli, ["history"])
    assert result.exit_code == 0


def test_step_passthrough():
    """nano_agent.step() returns the result of the wrapped function."""
    result = step("add two numbers", lambda a, b: a + b, 2, 3)
    assert result == 5


def test_step_with_kwargs():
    result = step("greet", lambda name: f"hello {name}", name="world")  # noqa: E731
    assert result == "hello world"
