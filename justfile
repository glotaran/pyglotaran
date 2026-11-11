# Set shell for Windows compatibility
set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

# Default recipe (show available recipes)
default:
    @just --list

# ============================================================================
# Hidden helpers (cross-platform using Python)
# ============================================================================

# Remove directory or file recursively (cross-platform)
[private]
_rm-rf +paths:
    @uv run python -c "import shutil, pathlib; [shutil.rmtree(pathlib.Path(p), ignore_errors=True) if pathlib.Path(p).exists() else None for p in '{{paths}}'.split()]"

# Remove file (cross-platform)
[private]
_rm-f path:
    @uv run python -c "import pathlib; p = pathlib.Path('{{path}}'); p.unlink(missing_ok=True)"

# Find and remove files by pattern (cross-platform)
[private]
_find-rm +patterns:
    @uv run python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink() for pattern in '{{patterns}}'.split() for p in pathlib.Path('.').rglob(pattern)]"

# List directory contents (cross-platform)
[private]
_ls-l path:
    @uv run python -c "import pathlib; [print(f'{p.stat().st_size:>10} {p.name}') for p in sorted(pathlib.Path('{{path}}').iterdir())]"

# Open file or URL in default browser/application (cross-platform)
[private]
_open-browser path:
    @uv run python -c "import webbrowser, pathlib; webbrowser.open('file://' + str(pathlib.Path('{{path}}').resolve()))"

# ============================================================================
# Cleaning
# ============================================================================

# Remove all build, test, coverage and Python artifacts
clean: clean-build clean-pyc clean-test
    just docs/clean

# Remove build artifacts
clean-build:
    @echo "Removing build artifacts..."
    @just _rm-rf build dist .eggs
    @just _find-rm "*.egg-info" "*.egg"

# Remove Python file artifacts
clean-pyc:
    @echo "Removing Python file artifacts..."
    @just _find-rm "*.pyc" "*.pyo" "*~" "__pycache__"

# Remove test and coverage artifacts
clean-test:
    @echo "Removing test and coverage artifacts..."
    @just _rm-rf htmlcov .pytest_cache
    @just _rm-f .coverage

# ============================================================================
# Code Quality
# ============================================================================

# Check style and linting with pre-commit
lint:
    @uv run python -c "import shutil, sys; sys.exit(0) if shutil.which('pre-commit') else (print('Error: pre-commit is not installed.\\nInstall it with: uv tool install pre-commit --with pre-commit-uv --force-reinstall --force', file=sys.stderr) or sys.exit(1))"
    pre-commit run -a

# Run tests with pytest
test:
    uv run pytest tests

# Run the documentation notebooks as tests
test-notebooks:
    uv run python docs/remove_notebook_written_data.py
    uv run pytest --nbval docs/source/notebooks
    uv run python docs/remove_notebook_written_data.py

# Check code coverage and open HTML report
cov:
    uv run pytest --cov=glotaran --cov-report html tests
    @just _open-browser htmlcov/index.html

# ============================================================================
# Documentation
# ============================================================================

# Generate Sphinx HTML documentation and open in browser
docs:
    just docs/clean
    just docs/html
    just _open-browser docs/build/html/index.html

# ============================================================================
# Distribution
# ============================================================================

# Build source and wheel package
dist: clean-build
    uv build
    @just _ls-l dist

# Install the package and its dependencies
install:
    uv sync
