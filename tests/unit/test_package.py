"""Phase 0 packaging contract tests."""

import importlib
from importlib.metadata import version
from pathlib import Path
from subprocess import run

import pytest


def test_public_package_and_ngsolve_are_importable() -> None:
    """The base install exposes remec and its binary FEM dependency."""
    remec = importlib.import_module("remec")
    ngsolve = importlib.import_module("ngsolve")

    assert remec.__version__ == version("remec")
    assert ngsolve.__version__
    assert Path(remec.__file__).with_name("py.typed").is_file()


def test_normalization_derives_documented_scales() -> None:
    """The Phase 0 normalization record exposes the §6 reference scales."""
    from remec import Normalization

    normalization = Normalization(
        reference_length=2.0,
        reference_field=3.0,
        chi_transport_scale=7.0,
        chi_source_scale=11.0,
        mu0=5.0,
    )

    assert normalization.pressure_scale == pytest.approx(1.8)
    assert normalization.current_density_scale == pytest.approx(0.3)
    assert normalization.vector_potential_scale == pytest.approx(6.0)
    assert normalization.u_scale == pytest.approx(0.1)
    assert normalization.du_scale == pytest.approx(6.0)
    assert normalization.chi_transport_scale == pytest.approx(7.0)
    assert normalization.chi_source_scale == pytest.approx(11.0)

    with pytest.raises(ValueError, match="reference_length"):
        Normalization(reference_length=0.0, reference_field=3.0)


def test_runtime_options_have_a_conservative_layer_resolution_default() -> None:
    """The initial options record carries the documented six-cell default."""
    from remec import RuntimeOptions

    assert RuntimeOptions().min_layer_cells == 6

    with pytest.raises(ValueError, match="min_layer_cells"):
        RuntimeOptions(min_layer_cells=0)


def test_make_default_lists_recipes() -> None:
    """A bare make invocation advertises the documented development gates."""
    result = run(
        ["make", "--no-print-directory"],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "check" in result.stdout
    assert "wheel-smoke" in result.stdout


def test_ci_uses_pip_and_covers_the_supported_python_floor() -> None:
    """CI installs project extras with pip on every supported Python version."""
    workflow = (Path(__file__).parents[2] / ".github/workflows/ci.yml").read_text()

    assert 'python: ["3.10", "3.11", "3.12"]' in workflow
    assert 'python -m pip install -e ".[dev,cutcell]"' in workflow
    assert "setup-uv" not in workflow
    assert "uv " not in workflow
