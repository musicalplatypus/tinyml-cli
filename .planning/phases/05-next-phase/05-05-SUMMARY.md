---
phase: 05
plan: 05-05
type: feat
status: complete
date_completed: "2026-07-08"
---

# Summary 05-05: Troubleshooting Assistant (diagnose)

## Outcome

Complete. `mmcli diagnose` runs real environment checks and reports findings with severity levels, fix suggestions, and an exit code of 1 on critical failures.

## What Was Delivered

- `mmcli/diagnose.py` — `DiagnosticIssue`, `DiagnosticResult`, check functions, formatter (274 lines)
- `mmcli diagnose` and `mmcli diagnose --full` subcommands wired in `cli.py`
- `mmcli diagnose --error "<message>"` for targeted fix suggestions
- `tests/test_diagnose.py` — dedicated test file

## Checks Implemented

| Check | Severity | Notes |
|-------|----------|-------|
| Python version ≥ 3.10 | critical | |
| MMCLI_PYTHON set | warning | |
| MMCLI_MODELZOO_PATH set | warning | |
| tinyml_modelmaker importable | critical | |
| Current directory accessible | info | `--full` only |
| Disk space > 100 MB | info | `--full` only |
