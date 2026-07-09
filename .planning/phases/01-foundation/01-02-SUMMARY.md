---
phase: 01
plan: 02
status: COMPLETE
completed: 2026-07-08
note: Restoration completed in this session after Phase 5 regression (eb0a1bd)
---

# Summary: Security Hardening (Input Validation & Subprocess Isolation)

## What Was Built

Security helper functions in `mmcli/cli.py` using pathlib-based semantic checks:

- **`_is_safe_path(path, base_dir=None)`** — resolves path via `pathlib.Path.resolve()`, checks `is_relative_to(cwd)` or `is_relative_to(tempdir)`. Immune to dot-collapse bypasses (`a....b`), encoded traversal (`%2e%2e`), and Unicode path breakage that affected the original string-based implementation.
- **`_sanitize_input(input_str, max_length=1024)`** — raises `ValueError` on length exceeded; does NOT strip characters. Shell injection is prevented by `shell=False` in all subprocess calls, not by character stripping.
- **`_validate_args()` wiring** — both functions applied at the top of `_validate_args()`: length cap on `module`/`task`/`device`/`model` string flags; path traversal guard on relative `--config`/`--onnx`/`--project` paths; absolute user-specified paths accepted without traversal check.

## Regression History

Both functions were silently removed in Phase 5 commit `eb0a1bd` (564-line cli.py rewrite). Restored with improved semantics in commit `e2ec9ae` (2026-07-08). Character-stripping behavior in the restored `_sanitize_input` was caught by peer review and fixed in `7995ddf`.

## Acceptance Criteria — All Met

- `_is_safe_path("../../etc/passwd")` returns False ✓
- `_sanitize_input("x" * 1025)` raises ValueError ✓
- `_sanitize_input("; rm -rf /")` returns `"; rm -rf /"` unchanged ✓
- All subprocess calls use `shell=False` ✓
- `pytest tests/test_security_fixes.py -v` — 12 passed ✓
