"""Contract tests for the Phase 0.2 common utilities."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

import pytest


def test_named_block_norms_report_each_block_and_the_combined_norm() -> None:
    """Independent block norms preserve the solver diagnostic decomposition."""
    from remec.common.norms import block_l2_norms

    norms = block_l2_norms({"magnetic": [3.0, 4.0], "pressure": [12.0]})

    assert norms.blocks == {"magnetic": pytest.approx(5.0), "pressure": pytest.approx(12.0)}
    assert norms.combined == pytest.approx(13.0)


@dataclass(frozen=True)
class _ExampleConfig:
    name: str
    values: tuple[int, ...]


def test_config_serialization_is_canonical_for_mappings_and_dataclasses() -> None:
    """Equivalent input configurations have byte-identical serialized representations."""
    from remec.common.serialization import canonical_json, configuration_digest

    left = {"z": [2, 1], "a": _ExampleConfig(name="case", values=(3, 4))}
    right = {"a": _ExampleConfig(name="case", values=(3, 4)), "z": [2, 1]}

    assert canonical_json(left) == canonical_json(right)
    assert canonical_json(left) == '{"a":{"name":"case","values":[3,4]},"z":[2,1]}'
    assert configuration_digest(left) == configuration_digest(right)


def test_structured_events_and_timer_emit_machine_readable_json() -> None:
    """Logs carry event names, fields, and timing without formatting-dependent parsing."""
    from remec.common.logging import JsonEventLogger, timed

    stream = io.StringIO()
    logger = JsonEventLogger(stream)
    logger.info("solve_started", dofs=123, threads=2)
    with timed(logger, "assemble", block="chi"):
        pass

    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert records[0] == {"dofs": 123, "event": "solve_started", "level": "INFO", "threads": 2}
    assert records[1]["event"] == "assemble"
    assert records[1]["block"] == "chi"
    assert records[1]["seconds"] >= 0.0


def test_thread_configuration_validates_before_calling_ngsolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad user thread count never reaches the NGSolve global configuration."""
    from remec.common.threads import configure_threads

    calls: list[int] = []

    def set_threads(value: int) -> None:
        calls.append(value)

    monkeypatch.setattr("remec.common.threads._set_ngsolve_threads", set_threads)

    assert configure_threads(3) == 3
    assert calls == [3]
    with pytest.raises(ValueError, match="threads"):
        configure_threads(0)
    assert calls == [3]


def test_checkpoint_metadata_round_trips_normalization_and_runtime_configuration() -> None:
    """Restart metadata is versioned and preserves configuration byte-for-byte."""
    from remec import Normalization, RuntimeOptions
    from remec.common.checkpoint import CheckpointMetadata, CheckpointVersionError

    metadata = CheckpointMetadata.create(
        normalization=Normalization(reference_length=2.0, reference_field=5.0),
        runtime=RuntimeOptions(threads=3),
        state_names=("chi", "u"),
        git_commit="abc123",
        platform="test-platform",
    )

    restored = CheckpointMetadata.from_json(metadata.to_json())
    assert restored == metadata
    assert restored.configuration["normalization"]["reference_field"] == 5.0
    assert restored.configuration["runtime"]["threads"] == 3

    future = json.loads(metadata.to_json())
    future["schema_version"] = 2
    with pytest.raises(CheckpointVersionError, match="unsupported"):
        CheckpointMetadata.from_json(json.dumps(future))
