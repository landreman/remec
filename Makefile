# List available recipes when you type `make`
.PHONY: default test test-full test-exhaustive lint check smoke wheel-smoke

default:
	@grep -E '^[A-Za-z][A-Za-z0-9_-]*:' $(MAKEFILE_LIST) | cut -d: -f1

# PR-CI subset: fast tests only. Budget: < 2 min on the reference laptop (DESIGN.md
# §22.1). Both slower tiers are excluded. The durations report is the early warning.
test:
	python -m pytest -m "not slow and not exhaustive" --durations=15

# Complete developer-runnable suite, including slow but excluding remote-only
# exhaustive tests. Budget: < 5 min (DESIGN.md §22.1).
test-full:
	python -m pytest -m "not exhaustive" --durations=25

# Complete remote verification, including exhaustive 3D ladders and benchmarks.
# This has no normative laptop wall-clock cap; use the scheduled/manual workflow.
test-exhaustive:
	python -m pytest --durations=40

# Formatting, linting, types
lint:
	ruff format --check src tests
	ruff check src tests
	mypy src/remec

# The gate. This is what "done" means.
check: lint test

# Clean-environment import check; CI separately exercises editable install plus pytest.
smoke:
	rm -rf /tmp/remec-smoke
	python -m venv /tmp/remec-smoke
	/tmp/remec-smoke/bin/python -m pip install -q .
	/tmp/remec-smoke/bin/python -c "import remec, ngsolve; print(remec.__version__, ngsolve.__version__)"

# Wheel build and clean-install check required by PR CI.
wheel-smoke:
	rm -rf /tmp/remec-wheel /tmp/remec-dist
	python -m build --wheel --outdir /tmp/remec-dist
	python -m venv /tmp/remec-wheel
	/tmp/remec-wheel/bin/python -m pip install -q /tmp/remec-dist/*.whl
	/tmp/remec-wheel/bin/python -c "import remec, ngsolve; from pathlib import Path; assert Path(remec.__file__).with_name('py.typed').is_file(); print(remec.__version__, ngsolve.__version__)"
