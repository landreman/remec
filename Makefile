# List available recipes when you type `make`
default:
    @make --list

# PR-CI subset: fast tests only
test:
    pytest -q -m "not slow"

# Everything, including nightly-marked tests
test-full:
    pytest -q

# Formatting, linting, types
lint:
    ruff format --check .
    ruff check .
    mypy src/remec

# The gate. This is what "done" means.
check: lint test

# Clean-environment install check (Phase 0 acceptance criterion)
smoke:
    rm -rf /tmp/remec-smoke
    uv venv /tmp/remec-smoke
    /tmp/remec-smoke/bin/python -m pip install -q .
    /tmp/remec-smoke/bin/python -c "import remec, ngsolve; print(remec.__version__, ngsolve.__version__)"
