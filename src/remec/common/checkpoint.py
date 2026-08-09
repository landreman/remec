"""Versioned, deterministic metadata used by future checkpoint readers."""

from __future__ import annotations

import json
import os
import platform as platform_module
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from remec.common.serialization import canonical_json
from remec.normalization import Normalization
from remec.options import RuntimeOptions

_SCHEMA_VERSION = 1


class CheckpointVersionError(ValueError):
    """Raised when checkpoint metadata is invalid or has an unsupported schema."""


def _freeze_json(value: Any) -> Any:
    """Recursively freeze JSON-compatible data retained by immutable metadata."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Portable restart metadata with an explicit, validated schema version."""

    schema_version: int
    configuration: Mapping[str, Any]
    state_names: tuple[str, ...]
    git_commit: str
    platform: str
    remec_version: str
    ngsolve_version: str

    @classmethod
    def create(
        cls,
        *,
        normalization: Normalization,
        runtime: RuntimeOptions,
        state_names: tuple[str, ...],
        git_commit: str | None = None,
        platform: str | None = None,
    ) -> CheckpointMetadata:
        """Create schema-1 metadata from the common restart-critical configuration."""
        from ngsolve import __version__ as ngsolve_version  # type: ignore[import-untyped]

        from remec import __version__ as remec_version

        configuration = _freeze_json(
            json.loads(canonical_json({"normalization": normalization, "runtime": runtime}))
        )
        return cls(
            schema_version=_SCHEMA_VERSION,
            configuration=configuration,
            state_names=state_names,
            git_commit=git_commit or os.environ.get("GITHUB_SHA", "unknown"),
            platform=platform or platform_module.platform(),
            remec_version=remec_version,
            ngsolve_version=ngsolve_version,
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
                "remec_version": self.remec_version,
                "ngsolve_version": self.ngsolve_version,
            }
        )

    @classmethod
    def from_json(cls, serialized: str) -> CheckpointMetadata:
        """Restore metadata, rejecting all schema versions other than the supported major."""
        try:
            payload = json.loads(serialized)
        except json.JSONDecodeError as error:
            raise CheckpointVersionError("invalid checkpoint metadata JSON") from error
        if not isinstance(payload, dict):
            raise CheckpointVersionError("invalid checkpoint metadata: expected an object")
        version = payload.get("schema_version")
        if version != _SCHEMA_VERSION:
            raise CheckpointVersionError(f"unsupported checkpoint schema version: {version!r}")
        try:
            configuration = payload["configuration"]
            state_names = payload["state_names"]
            git_commit = payload["git_commit"]
            platform = payload["platform"]
            remec_version = payload["remec_version"]
            ngsolve_version = payload["ngsolve_version"]
        except KeyError as error:
            raise CheckpointVersionError(
                f"invalid checkpoint metadata: missing {error.args[0]!r}"
            ) from error
        if not isinstance(configuration, dict) or not isinstance(state_names, list):
            raise CheckpointVersionError("invalid checkpoint metadata field types")
        if not all(isinstance(name, str) for name in state_names):
            raise CheckpointVersionError("invalid checkpoint metadata: state_names must be strings")
        if not all(
            isinstance(item, str) for item in (git_commit, platform, remec_version, ngsolve_version)
        ):
            raise CheckpointVersionError("invalid checkpoint metadata version field types")
        return cls(
            schema_version=version,
            configuration=_freeze_json(configuration),
            state_names=tuple(state_names),
            git_commit=git_commit,
            platform=platform,
            remec_version=remec_version,
            ngsolve_version=ngsolve_version,
        )
