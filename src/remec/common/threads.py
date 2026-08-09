"""Validated configuration of NGSolve's process-global worker count."""

from __future__ import annotations


def _set_ngsolve_threads(threads: int) -> None:
    """Apply a validated thread count through the installed NGSolve API."""
    from ngsolve import SetNumThreads  # type: ignore[import-untyped]

    SetNumThreads(threads)


def configure_threads(threads: int) -> int:
    """Validate and apply the TaskManager worker count, returning the accepted count."""
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    _set_ngsolve_threads(threads)
    return threads
