---
created: 2026-07-09T12:55:08.142Z
title: Fix yaml.py stub safe_load to accept string input
area: general
files:
  - yaml.py
  - tests/test_e2e.py
---

## Problem

The project has a root-level `yaml.py` that shadows PyYAML. Its `safe_load` uses `json.load(stream)` which requires a file-like object, but callers pass plain strings. When given a string, `json.load` raises → the except clause returns `{}` → downstream code hits `KeyError: 'training'`.

Three `TestDryRunE2E` tests in `tests/test_e2e.py` fail with this pattern. The stub's `safe_load`:

```python
def safe_load(stream: TextIO) -> Any:
    try:
        return json.load(stream)  # BUG: needs file-like, not str
    except Exception:
        try:
            stream.seek(0)
        except Exception:
            pass
        return {}
```

## Solution

Fix `yaml.py:safe_load` to handle both strings and file-like objects:

```python
def safe_load(stream):
    try:
        if isinstance(stream, str):
            return json.loads(stream)
        return json.load(stream)
    except Exception:
        try:
            stream.seek(0)
        except Exception:
            pass
        return {}
```

This is the minimal fix: `json.loads` accepts strings, `json.load` accepts file-like objects. The seek fallback stays for file-like callers that may be reused.
