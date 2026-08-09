# List available recipes when you type `make`
default:
	@make --list

# PR-CI subset: fast tests only
test:
	python -m pytest -q -m "not slow"

# Everything, including nightly-marked tests
test-full:
	python -m pytest -q

# Formatting, linting, types
lint:
	ruff format --check src tests
	ruff check src tests
	mypy src/remec

# The gate. This is what "done" means.
check: lint test

# Clean-environment install check (Phase 0 acceptance criterion)
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
	/tmp/remec-wheel/bin/python -c "import remec, ngsolve; print(remec.__version__, ngsolve.__version__)"
