"""Minimal nano-agent step() interface.

This module mirrors the nano-agent step() signature so the churn recorder
can wrap it transparently. Keeps the churn package self-contained without
requiring a hard dependency on the nano-agent repo.
"""
from __future__ import annotations

from typing import Any, Callable


def step(label: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
    """Execute one agent tool call and return its result.

    Args:
        label: Human-readable label for this step (used by the churn recorder).
        func: The tool function to invoke.
        *args: Positional arguments forwarded to func.
        **kwargs: Keyword arguments forwarded to func.

    Returns:
        The return value of func(*args, **kwargs).
    """
    return func(*args, **kwargs)
