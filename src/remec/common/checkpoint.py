"""Versioned, deterministic metadata used by future checkpoint readers."""

from __future__ import annotations

import json
import os
import platform as platform_module
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from remec.common.serialization import canonical_json
from remec.normalization import Normalization
from remec.options import RuntimeOptions

if TYPE_CHECKING:
    from remec.profiles import TabulatedPressureProfile, TabulatedToroidalCurrentProfile

_SCHEMA_VERSION = 1


class CheckpointVersionError(ValueError):
    """Raised when checkpoint metadata is invalid or has an unsupported schema."""


@dataclass(frozen=True, slots=True)
class ConstrainedCurrentCheckpoint:
    r"""Restart-critical border state for note equations ``(M3)``--``(M3b)``.

    The one-dimensional basis is deliberately fixed to piecewise linear on the
    normalized-volume ``shell_edges``. There is one solved ``G`` coefficient per
    node, its final coefficient equals ``edge_value``, and there is one independently
    reconstructed constraint residual per shell.
    """

    shell_edges: tuple[float, ...]
    g_coefficients: tuple[float, ...]
    edge_value: float
    shell_constraint_residuals: tuple[float, ...]
    m3_relative_residual_norm: float
    m3b_relative_residual_norm: float

    def __post_init__(self) -> None:
        edges = tuple(float(value) for value in self.shell_edges)
        coefficients = tuple(float(value) for value in self.g_coefficients)
        residuals = tuple(float(value) for value in self.shell_constraint_residuals)
        if (
            len(edges) < 2
            or edges[0] != 0.0
            or edges[-1] != 1.0
            or not all(isfinite(value) for value in edges)
            or any(right <= left for left, right in pairwise(edges))
        ):
            raise CheckpointVersionError(
                "constrained-current shell edges must partition normalized-volume [0, 1]"
            )
        if len(coefficients) != len(edges) or not all(isfinite(value) for value in coefficients):
            raise CheckpointVersionError(
                "constrained-current G coefficients must match shell nodes"
            )
        if not isfinite(self.edge_value) or coefficients[-1] != self.edge_value:
            raise CheckpointVersionError(
                "final G coefficient must equal the constrained edge value"
            )
        if len(residuals) != len(edges) - 1 or not all(isfinite(value) for value in residuals):
            raise CheckpointVersionError(
                "constrained-current residuals must contain one finite value per shell"
            )
        for value in (self.m3_relative_residual_norm, self.m3b_relative_residual_norm):
            if not isfinite(value) or value < 0.0:
                raise CheckpointVersionError(
                    "constrained-current relative residuals must be finite"
                )
        object.__setattr__(self, "shell_edges", edges)
        object.__setattr__(self, "g_coefficients", coefficients)
        object.__setattr__(self, "shell_constraint_residuals", residuals)

    def to_record(self) -> dict[str, object]:
        """Return the explicit normalized-volume bordered-state record."""
        return {
            "coordinate_kind": "normalized_volume",
            "basis_kind": "piecewise_linear",
            "shell_edges": list(self.shell_edges),
            "g_coefficients": list(self.g_coefficients),
            "edge_value": self.edge_value,
            "shell_constraint_residuals": list(self.shell_constraint_residuals),
            "m3_relative_residual_norm": self.m3_relative_residual_norm,
            "m3b_relative_residual_norm": self.m3b_relative_residual_norm,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ConstrainedCurrentCheckpoint:
        """Restore only the explicit piecewise-linear normalized-volume border."""
        expected_keys = {
            "coordinate_kind",
            "basis_kind",
            "shell_edges",
            "g_coefficients",
            "edge_value",
            "shell_constraint_residuals",
            "m3_relative_residual_norm",
            "m3b_relative_residual_norm",
        }
        if set(record) != expected_keys:
            raise CheckpointVersionError("ambiguous constrained-current checkpoint record")
        if record.get("coordinate_kind") != "normalized_volume":
            raise CheckpointVersionError("constrained-current state must use normalized-volume s")
        if record.get("basis_kind") != "piecewise_linear":
            raise CheckpointVersionError("unsupported constrained-current G basis")

        def number_tuple(name: str) -> tuple[float, ...]:
            value = record.get(name)
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise CheckpointVersionError("invalid constrained-current checkpoint record")
            if any(isinstance(item, bool) for item in value):
                raise CheckpointVersionError("invalid constrained-current checkpoint record")
            return tuple(float(item) for item in value)

        try:
            shell_edges = number_tuple("shell_edges")
            g_coefficients = number_tuple("g_coefficients")
            if isinstance(record["edge_value"], bool):
                raise TypeError
            edge_value = float(record["edge_value"])  # type: ignore[arg-type]
            shell_constraint_residuals = number_tuple("shell_constraint_residuals")
            if isinstance(record["m3_relative_residual_norm"], bool) or isinstance(
                record["m3b_relative_residual_norm"], bool
            ):
                raise TypeError
            m3_relative_residual_norm = float(record["m3_relative_residual_norm"])  # type: ignore[arg-type]
            m3b_relative_residual_norm = float(record["m3b_relative_residual_norm"])  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError) as error:
            raise CheckpointVersionError("invalid constrained-current checkpoint record") from error
        return cls(
            shell_edges=shell_edges,
            g_coefficients=g_coefficients,
            edge_value=edge_value,
            shell_constraint_residuals=shell_constraint_residuals,
            m3_relative_residual_norm=m3_relative_residual_norm,
            m3b_relative_residual_norm=m3b_relative_residual_norm,
        )


def _freeze_json(value: Any) -> Any:
    """Recursively freeze JSON-compatible data retained by immutable metadata."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _default_git_commit() -> str:
    """Prefer the checked-out source revision, then CI provenance, then ``unknown``."""
    repository = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        result = None
    if result is not None and result.returncode == 0:
        commit = result.stdout.strip()
        if commit:
            return commit
    return os.environ.get("GITHUB_SHA", "unknown")


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
        state_names: tuple[str, ...] | list[str],
        pressure_profile: TabulatedPressureProfile | None = None,
        toroidal_current_profile: TabulatedToroidalCurrentProfile | None = None,
        constrained_current: ConstrainedCurrentCheckpoint | None = None,
        git_commit: str | None = None,
        platform: str | None = None,
    ) -> CheckpointMetadata:
        """Create schema-1 metadata from the common restart-critical configuration."""
        from ngsolve import __version__ as ngsolve_version  # type: ignore[import-untyped]

        from remec import __version__ as remec_version

        if not all(isinstance(name, str) for name in state_names):
            raise CheckpointVersionError("state_names must contain only strings")
        normalized_state_names = tuple(state_names)
        configuration_payload: dict[str, Any] = {
            "normalization": normalization,
            "runtime": runtime,
        }
        if (pressure_profile is None) != (toroidal_current_profile is None):
            raise CheckpointVersionError(
                "pressure and toroidal-current profiles must be checkpointed together"
            )
        if pressure_profile is not None and toroidal_current_profile is not None:
            from remec.profiles import TabulatedPressureProfile, TabulatedToroidalCurrentProfile

            if not isinstance(pressure_profile, TabulatedPressureProfile) or not isinstance(
                toroidal_current_profile, TabulatedToroidalCurrentProfile
            ):
                raise CheckpointVersionError(
                    "checkpointed profiles must be explicit tabulated normalized-volume records"
                )
            configuration_payload["profiles"] = {
                "pressure": pressure_profile.to_record(),
                "toroidal_current": toroidal_current_profile.to_record(),
            }
        if constrained_current is not None:
            if pressure_profile is None or toroidal_current_profile is None:
                raise CheckpointVersionError(
                    "constrained-current state requires normalized pressure/current profiles"
                )
            if not isinstance(constrained_current, ConstrainedCurrentCheckpoint):
                raise CheckpointVersionError(
                    "constrained_current must be a ConstrainedCurrentCheckpoint"
                )
            configuration_payload["constrained_current"] = constrained_current.to_record()
        configuration = _freeze_json(json.loads(canonical_json(configuration_payload)))
        return cls(
            schema_version=_SCHEMA_VERSION,
            configuration=configuration,
            state_names=normalized_state_names,
            git_commit=git_commit or _default_git_commit(),
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
        if not isinstance(version, int) or isinstance(version, bool) or version != _SCHEMA_VERSION:
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
        _validate_profile_configuration(configuration)
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


def _validate_profile_configuration(configuration: dict[str, Any]) -> None:
    """Reject legacy dimensional-V/F state instead of reinterpreting it on restart."""
    profiles = configuration.get("profiles")
    if profiles is None:
        if "constrained_current" in configuration:
            raise CheckpointVersionError(
                "constrained-current state requires normalized checkpoint profiles"
            )
        return
    if not isinstance(profiles, dict) or set(profiles) != {"pressure", "toroidal_current"}:
        raise CheckpointVersionError("ambiguous or legacy checkpoint profile state")
    pressure = profiles["pressure"]
    current = profiles["toroidal_current"]
    if not isinstance(pressure, dict) or not isinstance(current, dict):
        raise CheckpointVersionError("invalid checkpoint profile records")
    from remec.profiles import (
        InvalidProfileError,
        TabulatedPressureProfile,
        TabulatedToroidalCurrentProfile,
    )

    try:
        TabulatedPressureProfile.from_record(pressure)
        TabulatedToroidalCurrentProfile.from_record(current)
    except InvalidProfileError as error:
        raise CheckpointVersionError(f"invalid normalized checkpoint profile: {error}") from error
    constrained_current = configuration.get("constrained_current")
    if constrained_current is not None:
        if not isinstance(constrained_current, dict):
            raise CheckpointVersionError("invalid constrained-current checkpoint record")
        ConstrainedCurrentCheckpoint.from_record(constrained_current)
