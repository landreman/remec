"""Versioned, deterministic metadata used by future checkpoint readers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from remec.common.serialization import canonical_json
from remec.normalization import Normalization
from remec.options import RuntimeOptions

_SCHEMA_VERSION = 1


class CheckpointVersionError(ValueError):
    """Raised when metadata belongs to an unsupported checkpoint schema."""


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Portable restart metadata with an explicit, validated schema version."""

    schema_version: int
    configuration: dict[str, Any]
    state_names: tuple[str, ...]
    git_commit: str
    platform: str

    @classmethod
    def create(
        cls,
        *,
        normalization: Normalization,
        runtime: RuntimeOptions,
        state_names: tuple[str, ...],
        git_commit: str,
        platform: str,
    ) -> CheckpointMetadata:
        """Create schema-1 metadata from the common restart-critical configuration."""
        configuration = json.loads(
            canonical_json({"normalization": normalization, "runtime": runtime})
        )
        return cls(
            schema_version=_SCHEMA_VERSION,
            configuration=configuration,
            state_names=state_names,
            git_commit=git_commit,
            platform=platform,
        )

    def to_json(self) -> str:
        """Serialize this metadata deterministically for checkpoint storage."""
        return canonical_json(
            {
                "schema_version": self.schema_version,
                "configuration": self.configuration,
                "state_names": self.state_names,
                "git_commit": self.git_commit,
                "platform": self.platform,
            }
        )

    @classmethod
    def from_json(cls, serialized: str) -> CheckpointMetadata:
        """Restore metadata, rejecting all schema versions other than the supported major."""
        payload = json.loads(serialized)
        version = payload.get("schema_version")
        if version != _SCHEMA_VERSION:
            raise CheckpointVersionError(f"unsupported checkpoint schema version: {version!r}")
        return cls(
            schema_version=version,
            configuration=payload["configuration"],
            state_names=tuple(payload["state_names"]),
            git_commit=payload["git_commit"],
            platform=payload["platform"],
        )
