"""Contracts for normalized pressure and enclosed-current profile inputs."""

from __future__ import annotations

import json

import numpy as np
import pytest

from remec import (
    AnalyticPressureProfile,
    AnalyticToroidalCurrentProfile,
    InvalidProfileError,
    Normalization,
    RuntimeOptions,
    TabulatedPressureProfile,
    TabulatedToroidalCurrentProfile,
    TransplantedProfile,
)
from remec.common.checkpoint import (
    CheckpointMetadata,
    CheckpointVersionError,
    ConstrainedCurrentCheckpoint,
)
from remec.level_set import MollifiedVolumeMap, QuadratureLevelSetData


def _uniform_volume_map(total_volume: float) -> MollifiedVolumeMap:
    normalized_levels = (np.arange(4096, dtype=float) + 0.5) / 4096.0
    return MollifiedVolumeMap.build(
        QuadratureLevelSetData(
            values=normalized_levels,
            gradient_magnitudes=np.ones_like(normalized_levels),
            weights=np.full_like(normalized_levels, total_volume / len(normalized_levels)),
            element_sizes=np.full_like(normalized_levels, 1.0 / len(normalized_levels)),
        ),
        spatial_width_cells=1.0,
        levels=257,
    )


def test_pressure_profiles_require_exact_normalized_volume_coordinates() -> None:
    """M4b public pressure inputs live on exactly s in [0, 1], never dimensional V."""
    profile = TabulatedPressureProfile([0.0, 0.4, 1.0], [3.0, 1.5, 0.2])
    profile.validate(edge_value=0.2)
    assert profile.value([0.0, 0.4, 1.0]) == pytest.approx([3.0, 1.5, 0.2])

    with pytest.raises(InvalidProfileError, match=r"exactly.*\[0, 1\]"):
        TabulatedPressureProfile([0.0, 0.4, 2.0], [3.0, 1.5, 0.2])
    with pytest.raises(InvalidProfileError, match="outside s"):
        profile.value(1.01)

    analytic = AnalyticPressureProfile(
        lambda s: 2.0 - np.asarray(s, dtype=float),
        lambda s: -np.ones_like(np.asarray(s, dtype=float)),
    )
    analytic.validate(edge_value=1.0)
    with pytest.raises(InvalidProfileError, match="outside s"):
        analytic.value(-1.0e-6)


def test_current_profiles_are_cumulative_allow_reversal_and_start_at_zero() -> None:
    """M3b accepts reversed-current I0(s), while enforcing I0(0)=0."""
    profile = TabulatedToroidalCurrentProfile([0.0, 0.3, 0.7, 1.0], [0.0, 1.2, 0.8, 1.0])
    profile.validate()
    assert profile.enclosed_current([0.0, 0.3, 0.7, 1.0]) == pytest.approx([0.0, 1.2, 0.8, 1.0])
    assert profile.derivative(0.5) < 0.0

    with pytest.raises(InvalidProfileError, match=r"I_0\(0\)=0"):
        TabulatedToroidalCurrentProfile([0.0, 1.0], [0.1, 1.0])

    analytic = AnalyticToroidalCurrentProfile(
        lambda s: np.asarray(s, dtype=float) * (2.0 - np.asarray(s, dtype=float)),
        lambda s: 2.0 - 2.0 * np.asarray(s, dtype=float),
    )
    analytic.validate()
    assert analytic.enclosed_current(1.0) == pytest.approx(1.0)


def test_tabulated_profile_records_are_explicit_and_reject_legacy_coordinates() -> None:
    """Restart records never infer dimensional V versus normalized s from sample ranges."""
    pressure = TabulatedPressureProfile([0.0, 0.5, 1.0], [2.0, 1.0, 0.0])
    current = TabulatedToroidalCurrentProfile([0.0, 0.5, 1.0], [0.0, 0.7, 1.1])

    pressure_record = pressure.to_record()
    current_record = current.to_record()
    assert pressure_record["coordinate_kind"] == "normalized_volume"
    assert current_record["coordinate_kind"] == "normalized_volume"
    assert TabulatedPressureProfile.from_record(pressure_record) == pressure
    assert TabulatedToroidalCurrentProfile.from_record(current_record) == current

    ambiguous = dict(pressure_record)
    ambiguous.pop("coordinate_kind")
    with pytest.raises(InvalidProfileError, match="coordinate_kind"):
        TabulatedPressureProfile.from_record(ambiguous)
    dimensional = dict(current_record, coordinate_kind="dimensional_volume")
    with pytest.raises(InvalidProfileError, match="normalized_volume"):
        TabulatedToroidalCurrentProfile.from_record(dimensional)


def test_checkpoint_first_profile_payload_uses_normalized_contract_without_schema_bump() -> None:
    """Schema 1 first persists profiles only with explicit normalized-volume records."""
    pressure = TabulatedPressureProfile([0.0, 1.0], [1.0, 0.0])
    current = TabulatedToroidalCurrentProfile([0.0, 1.0], [0.0, 2.0])
    metadata = CheckpointMetadata.create(
        normalization=Normalization(reference_length=1.0, reference_field=1.0),
        runtime=RuntimeOptions(),
        state_names=("chi",),
        pressure_profile=pressure,
        toroidal_current_profile=current,
        git_commit="abc123",
        platform="test-platform",
    )

    assert metadata.schema_version == 1
    assert metadata.configuration["profiles"]["pressure"]["coordinate_kind"] == (
        "normalized_volume"
    )
    assert metadata.configuration["profiles"]["toroidal_current"]["values"][-1] == 2.0
    assert CheckpointMetadata.from_json(metadata.to_json()) == metadata

    legacy = json.loads(metadata.to_json())
    legacy["configuration"]["profiles"] = {
        "prescribed_current_profile": {"identifier": "old-F-of-p"}
    }
    with pytest.raises(CheckpointVersionError, match="legacy|ambiguous"):
        CheckpointMetadata.from_json(json.dumps(legacy))


def test_checkpoint_persists_the_solved_unknown_g_border_without_schema_bump() -> None:
    r"""Restart state stores ``I_0``, shell basis, solved ``G``, and every (M3b) row."""
    pressure = TabulatedPressureProfile([0.0, 1.0], [1.0, 0.1])
    current = TabulatedToroidalCurrentProfile([0.0, 0.5, 1.0], [0.0, 0.4, 0.7])
    constrained = ConstrainedCurrentCheckpoint(
        shell_edges=(0.0, 0.5, 1.0),
        g_coefficients=(0.15, 0.25, 0.0),
        edge_value=0.0,
        shell_constraint_residuals=(2.0e-14, -3.0e-14),
        m3_relative_residual_norm=4.0e-15,
        m3b_relative_residual_norm=3.6e-14,
    )
    metadata = CheckpointMetadata.create(
        normalization=Normalization(reference_length=1.0, reference_field=2.0),
        runtime=RuntimeOptions(regularization_gradient="full"),
        state_names=("chi", "utilde", "g_coefficients"),
        pressure_profile=pressure,
        toroidal_current_profile=current,
        constrained_current=constrained,
        git_commit="abc123",
        platform="test-platform",
    )

    record = metadata.configuration["constrained_current"]
    assert metadata.schema_version == 1
    assert record["coordinate_kind"] == "normalized_volume"
    assert record["basis_kind"] == "piecewise_linear"
    assert record["shell_edges"] == (0.0, 0.5, 1.0)
    assert record["g_coefficients"][-1] == record["edge_value"] == 0.0
    assert record["shell_constraint_residuals"] == pytest.approx((2.0e-14, -3.0e-14))
    assert CheckpointMetadata.from_json(metadata.to_json()) == metadata

    invalid = json.loads(metadata.to_json())
    invalid["configuration"]["constrained_current"]["coordinate_kind"] = "dimensional_volume"
    with pytest.raises(CheckpointVersionError, match="normalized-volume"):
        CheckpointMetadata.from_json(json.dumps(invalid))


def test_pressure_and_current_semantics_do_not_change_with_domain_volume() -> None:
    """One shared s=V_chi/V_omega field makes profile meaning volume-scale invariant."""
    pressure = TabulatedPressureProfile([0.0, 0.5, 1.0], [2.0, 1.5, 0.25])
    current = TabulatedToroidalCurrentProfile([0.0, 0.5, 1.0], [0.0, 0.8, 1.0])
    unit_map = _uniform_volume_map(1.0)
    scaled_map = _uniform_volume_map(7.5)
    levels = np.linspace(0.0, 1.0, 41)

    unit_transplant = TransplantedProfile(unit_map, pressure)
    scaled_transplant = TransplantedProfile(scaled_map, pressure)
    assert scaled_map.evaluate_volume_coordinate(levels) == pytest.approx(
        unit_map.evaluate_volume_coordinate(levels), abs=2.0e-12
    )
    assert scaled_transplant.pressure(levels) == pytest.approx(
        unit_transplant.pressure(levels), abs=2.0e-12
    )
    s = np.asarray(scaled_map.evaluate_volume_coordinate(levels), dtype=float)
    assert current.enclosed_current(s[[0, 20, -1]]) == pytest.approx([1.0, 0.8, 0.0])
