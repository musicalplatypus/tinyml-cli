---
phase: 01
plan: 01
status: COMPLETE
completed: 2026-07-06
---

# Summary: Core CLI Structure and Essential Commands

## What Was Built

Delivered the full argparse-based CLI skeleton and subprocess dispatch pattern:

- **`mmcli/cli.py`** — main entry point with `train`, `compile`, `run`, `info`, `init`, `analyze`, `recommend`, `deploy`, `compare`, `diagnose`, `shell` subcommands; `_add_common_args()` for shared flags; `_validate_args()` for pre-dispatch validation
- **`mmcli/builder.py`** — translates CLI args into the YAML config dict consumed by tinyml-modelmaker; handles common/training/compilation/dataset sections
- **`mmcli/__init__.py`** — package init with `__version__` and `COMPATIBLE_MODELMAKER`
- **`mmcli/__main__.py`** — `python -m mmcli` entry point
- **`pyproject.toml`** — project metadata, `mmcli = "mmcli.cli:main"` console script, pytest marker config
- **`build_macos.sh`** — PyInstaller build script for macOS ARM64 standalone binary

## Key Architecture Decision

All tinyml-modelmaker work is dispatched via `subprocess.run(shell=False)` to `MMCLI_PYTHON` (a separate venv). This isolates heavy ML dependencies (TensorFlow, ONNX, NumPy) from the CLI binary.

## Acceptance Criteria — All Met

- `mmcli --help` shows all subcommands ✓
- `mmcli train` dispatches to MMCLI_PYTHON subprocess ✓
- `mmcli run` invokes train + compile in sequence ✓
- YAML config override (`--config`) works ✓
- Build script produces standalone binary on macOS ARM64 ✓
