"""Canonical JSON conversion for reproducible configuration records."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from typing import Any


def _json_ready(value: Any) -> Any:
    """Convert supported configuration values to JSON-compatible built-ins."""
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("configuration mapping keys must be strings")
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"unsupported configuration value: {type(value).__name__}")


def canonical_json(configuration: Any) -> str:
    """Serialize a configuration deterministically with sorted keys and compact separators."""
    return json.dumps(
        _json_ready(configuration), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def configuration_digest(configuration: Any) -> str:
    """Return the SHA-256 digest of :func:`canonical_json` configuration bytes."""
    return sha256(canonical_json(configuration).encode("utf-8")).hexdigest()
