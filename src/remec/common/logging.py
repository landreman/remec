"""Minimal structured event logging and timing helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any, TextIO


class JsonEventLogger:
    """Write one deterministic JSON object per event to a text stream."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def info(self, event: str, **fields: Any) -> None:
        """Write an ``INFO`` event with machine-readable fields."""
        record = {"event": event, "level": "INFO", **fields}
        self._stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self._stream.flush()


@contextmanager
def timed(logger: JsonEventLogger, event: str, **fields: Any) -> Iterator[None]:
    """Log elapsed wall time in seconds after the enclosed operation completes."""
    start = perf_counter()
    try:
        yield
    finally:
        logger.info(event, **fields, seconds=perf_counter() - start)
