# List available recipes when you type `make`
.PHONY: default test test-full test-exhaustive lint check smoke wheel-smoke

# pytest-xdist worker count. Overrides the `-n 3` default in pyproject.toml, which the
# command line wins over. The two bounded tiers are CPU-bound on the reference laptop.
# The exhaustive tier is memory-bound instead: the Section 21 direct solver holds a full
# 3D factorization per worker, so it defaults to fewer workers. Raise or lower either on
# the command line, e.g. `make test-exhaustive EXHAUSTIVE_WORKERS=1`.
PYTEST_WORKERS ?= 3
EXHAUSTIVE_WORKERS ?= 2

# Extra pytest arguments, e.g. `make test-exhaustive PYTEST_ARGS="-k reiman"`.
PYTEST_ARGS ?=

default:
	@grep -E '^[A-Za-z][A-Za-z0-9_-]*:' $(MAKEFILE_LIST) | cut -d: -f1

# PR-CI subset: fast tests only. Budget: < 2 min on the reference laptop (DESIGN.md
# §22.1). Both slower tiers are excluded. The durations report is the early warning.
test:
	python -m pytest -n $(PYTEST_WORKERS) -m "not slow and not exhaustive" \
		--durations=15 $(PYTEST_ARGS)

# Complete developer-runnable suite, including slow but excluding remote-only
# exhaustive tests. Budget: < 5 min (DESIGN.md §22.1).
test-full:
	python -m pytest -n $(PYTEST_WORKERS) -m "not exhaustive" --durations=25 $(PYTEST_ARGS)

# Complete remote verification, including exhaustive 3D ladders and benchmarks.
# This has no normative laptop wall-clock cap; use .github/workflows/exhaustive.yml.
test-exhaustive:
	python -m pytest -n $(EXHAUSTIVE_WORKERS) --durations=40 $(PYTEST_ARGS)

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
