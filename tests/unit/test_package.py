"""Phase 0 packaging contract tests."""

import importlib

import pytest


def test_public_package_and_ngsolve_are_importable() -> None:
    """The base install exposes remec and its binary FEM dependency."""
    remec = importlib.import_module("remec")
    ngsolve = importlib.import_module("ngsolve")

    assert remec.__version__
    assert ngsolve.__version__


def test_normalization_derives_documented_scales() -> None:
    """The Phase 0 normalization record exposes the §6 reference scales."""
    from remec import Normalization

    normalization = Normalization(reference_length=2.0, reference_field=3.0, mu0=0.5)

    assert normalization.pressure_scale == 18.0
    assert normalization.current_density_scale == 3.0
    assert normalization.vector_potential_scale == 6.0
    assert normalization.u_scale == 1.0
    assert normalization.du_scale == 6.0

    with pytest.raises(ValueError, match="reference_length"):
        Normalization(reference_length=0.0, reference_field=3.0)


def test_runtime_options_have_a_conservative_layer_resolution_default() -> None:
    """The initial options record carries the documented six-cell default."""
    from remec import RuntimeOptions

    assert RuntimeOptions().min_layer_cells == 6

    with pytest.raises(ValueError, match="min_layer_cells"):
        RuntimeOptions(min_layer_cells=0)
