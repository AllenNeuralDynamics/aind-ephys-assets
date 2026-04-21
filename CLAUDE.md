# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run all checks (formatting, linting, type checking, tests)
./scripts/run_linters_and_checks.sh -c

# Run tests
uv run pytest

# Run a single test file
uv run pytest tests/test_example.py

# Run a single test by name
uv run pytest -k "test_name"

# Formatting
uv run ruff format

# Linting
uv run ruff check
uv run ruff check --fix

# Type checking
uv run mypy

# Spell checking
uv run codespell --check-filenames

```

Always use `uv run` to execute commands, `uv add` to add dependencies, and `uv sync` to set up the environment. Never use bare `pip` or `python`.

## Architecture

This is a Python package using a `src/` layout. Source code lives in `src/aind_ephys_assets/`, tests in `tests/`.

- Build system: hatchling
- Formatting/linting: ruff (line length 120, numpy docstring convention)
- Testing: pytest with coverage reporting
- Type checking: mypy (strict mode)
- Versioning: commitizen (semantic versioning via conventional commits)
- CLI entry points: defined in `[project.scripts]` in pyproject.toml, implemented in `src/aind_ephys_assets/scripts/`

## Scope

AIND ephys asset **layout knowledge** — stream discovery, path resolution, name
parsing — shared across capsules that consume AIND raw + sorted assets. Pure
filesystem / S3 structure reasoning.

**Non-goals**: not a SpikeInterface wrapper (we read SI-format zarrs directly,
no `si.load` / object-graph construction), not a mipmap helper (consumed by
`ecephys-mipmap-zarr`, not vice versa), not a metadata client (no
`aind-data-schema` / `aind-data-access-api`).

**Source capsules** being de-duplicated:
- `~/Documents/Code/ecephys-mipmap-builder-capsule/code/run_capsule.py`
- `~/Documents/Code/ccg-analysis-monorepo/pl-oversplitting-analysis-capsule/code/data_discovery.py`

## AIND layout gotchas (canonical reference)

- Raw assets may be passed at the root or already descended into
  `ecephys/ecephys_compressed/` — `resolve_raw_root` must handle both.
- Sort assets may be passed at their root (contains `postprocessed/`) or one
  level above (pipeline-mounted layouts nest under `<asset_name>/`) —
  `resolve_sort_root` descends one level before giving up.
- Analyzer zarrs live at `postprocessed/<name>_recording<N>.zarr`; the matching
  raw zarr is `ecephys_compressed/<name>.zarr` (no `_recording<N>` suffix).
- Stream names like `experiment1_Record Node 111#Neuropix-PXI-100.50213-1`
  contain spaces, `#`, and `.` — sanitize before using as a filesystem basename.
- CO pipeline workers stage mounts via `/tmp` symlinks — use
  `os.walk(followlinks=True)`, not `Path.rglob`, when traversing `/data`.
