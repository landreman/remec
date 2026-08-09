"""Contract tests for the Phase 0.2 common utilities."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

import pytest


def test_named_block_norms_apply_physical_scales_before_combining() -> None:
    """Combined diagnostics use explicit per-block scales, never raw concatenated DOFs."""
    from remec.common.norms import block_l2_norms

    norms = block_l2_norms(
        {"magnetic": [3.0, 4.0], "pressure": [12.0]},
        scales={"magnetic": 1.0, "pressure": 2.0},
    )

    assert norms.blocks == {"magnetic": pytest.approx(5.0), "pressure": pytest.approx(12.0)}
    assert norms.scaled_blocks == {"magnetic": pytest.approx(5.0), "pressure": pytest.approx(6.0)}
    assert norms.combined == pytest.approx(61.0**0.5)
    with pytest.raises(ValueError, match="scales"):
        block_l2_norms({"magnetic": [1.0]}, scales={})
    with pytest.raises(ValueError, match="finite"):
        block_l2_norms({"magnetic": [float("nan")]}, scales={"magnetic": 1.0})


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
    assert (
        configuration_digest(left)
        == "1a1c5923a7c0ca511a356f69e02a6598b55166bdc160722f3c695025a36d73e9"
    )
    assert configuration_digest(left) == configuration_digest(right)
    assert configuration_digest(left) != configuration_digest({"z": [1, 2], "a": left["a"]})


@pytest.mark.parametrize(
    "configuration",
    [{1: "not-a-string-key"}, {"unsupported": {1, 2}}, {"not-a-number": float("nan")}],
)
def test_config_serialization_rejects_noncanonical_values(configuration: object) -> None:
    """Unsupported keys, values, and non-finite floats cannot enter restart metadata."""
    from remec.common.serialization import canonical_json

    with pytest.raises((TypeError, ValueError)):
        canonical_json(configuration)


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
    assert records[1]["ok"] is True
    assert records[1]["seconds"] >= 0.0

    with pytest.raises(ValueError, match="reserved"):
        logger.info("bad", event="overwritten")
    with pytest.raises(ValueError, match="reserved"), timed(logger, "bad", seconds=1.0):
        pass

    with pytest.raises(RuntimeError, match="failed"), timed(logger, "failed_work"):
        raise RuntimeError("failed")
    assert json.loads(stream.getvalue().splitlines()[-1]) == {
        "error": "RuntimeError: failed",
        "event": "failed_work",
        "level": "INFO",
        "ok": False,
        "seconds": pytest.approx(json.loads(stream.getvalue().splitlines()[-1])["seconds"]),
    }


def test_common_package_exports_all_common_utility_entry_points() -> None:
    """Callers can depend on the package-level common utility API."""
    from remec.common import JsonEventLogger, timed

    assert JsonEventLogger.__name__ == "JsonEventLogger"
    assert callable(timed)


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


def test_thread_configuration_executes_the_ngsolve_api() -> None:
    """The installed NGSolve API exposes and executes the thread setter used by REMEC."""
    import ngsolve

    from remec.common.threads import configure_threads

    assert callable(ngsolve.SetNumThreads)
    assert configure_threads(1) == 1


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
    assert restored.to_json() == metadata.to_json()
    assert restored.configuration["normalization"]["reference_field"] == 5.0
    assert restored.configuration["runtime"]["threads"] == 3
    assert restored.remec_version
    assert restored.ngsolve_version

    future = json.loads(metadata.to_json())
    future["schema_version"] = 2
    with pytest.raises(CheckpointVersionError, match="unsupported"):
        CheckpointMetadata.from_json(json.dumps(future))

    with pytest.raises(CheckpointVersionError, match="invalid"):
        CheckpointMetadata.from_json("[]")
