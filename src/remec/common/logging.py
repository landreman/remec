"""Minimal structured event logging and timing helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any, TextIO

_RESERVED_FIELDS = frozenset({"event", "level"})
_TIMING_FIELDS = frozenset({"error", "ok", "seconds"})


class JsonEventLogger:
    """Write one deterministic JSON object per event to a text stream."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def info(self, event_name: str, **fields: Any) -> None:
        """Write an ``INFO`` event with machine-readable fields."""
        if collision := _RESERVED_FIELDS.intersection(fields):
            raise ValueError(f"reserved log field(s): {', '.join(sorted(collision))}")
        record = {"event": event_name, "level": "INFO", **fields}
        self._stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self._stream.flush()


@contextmanager
def timed(logger: JsonEventLogger, event_name: str, **fields: Any) -> Iterator[None]:
    """Log elapsed wall time plus an explicit success/failure outcome."""
    if collision := (_RESERVED_FIELDS | _TIMING_FIELDS).intersection(fields):
        raise ValueError(f"reserved timing field(s): {', '.join(sorted(collision))}")
    start = perf_counter()
    try:
        yield
    except BaseException as error:
        logger.info(
            event_name,
            **fields,
            seconds=perf_counter() - start,
            ok=False,
            error=f"{type(error).__name__}: {error}",
        )
        raise
    else:
        logger.info(event_name, **fields, seconds=perf_counter() - start, ok=True)
