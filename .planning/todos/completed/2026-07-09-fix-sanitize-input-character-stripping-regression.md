---
created: 2026-07-09T03:46:52.963Z
title: Fix _sanitize_input character stripping security regression
area: security
files:
  - mmcli/cli.py:85-102
  - tests/test_regression.py:43-49
  - test_security_fixes.py:36-40
---

## Problem

`_sanitize_input` in `mmcli/cli.py` (lines 85–102) still contains the old
character-stripping implementation from before Phase 5 — it removes `;`, `$`,
`` ` ``, `|`, `&`, `>`, `<`, `(`, `)`, `{`, `}` from the input string. This
is the bypassable, lossy design the cross-AI review (Phase 3) identified as a
regression.

The session that ran `fix(security): restore _is_safe_path and _sanitize_input
removed in Phase 5` (commit e2ec9ae) documented and intended the correct
raises-only behavior, and the tests assert it — but the committed function body
was never actually replaced. One test fails:

```
FAILED tests/test_regression.py::TestSecurityRegression::test_sanitize_input_does_not_strip_chars
FAILED test_security_fixes.py::test_sanitize_input_does_not_strip_chars
# actual: ' rm -rf /'  (semicolon stripped)
```

Side issue: `_is_safe_path` (lines 48–82) creates `normalized_path` via
`path.replace('\\\\', '/')` but then passes `path` (not `normalized_path`) to
`Path()` in both branches. The `if '..' in normalized_path / else` branching
is identical in both arms — dead code that creates false impression of
Windows-path normalization.

Shell injection prevention is already handled by `shell=False` in all
subprocess calls — char-stripping is not needed and should not be present.

## Solution

Replace `_sanitize_input` body (mmcli/cli.py:85–102) with:

```python
def _sanitize_input(input_str: str, max_length: int = 1024) -> str:
    """Enforce a length cap. Raises ValueError if exceeded. Does NOT strip chars."""
    if not isinstance(input_str, str):
        input_str = str(input_str)
    if len(input_str) > max_length:
        raise ValueError(f"Input exceeds maximum length of {max_length}")
    return input_str
```

Simplify `_is_safe_path` by removing the dead `normalized_path` / `if '..'`
branching — keep a single `try` block using `Path(path).resolve()`.

Verify both tests pass after fix:
```bash
python3 -m pytest tests/test_regression.py::TestSecurityRegression -v
python3 -m pytest test_security_fixes.py -v
```
