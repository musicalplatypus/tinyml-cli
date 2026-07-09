---
created: 2026-07-09T12:55:08.142Z
title: Fix _is_safe_path traversal detection for edge cases
area: general
files:
  - mmcli/cli.py
  - tests/test_attack_surface.py
  - tests/test_integration_security.py
---

## Problem

`_is_safe_path` returns `True` (safe) for path traversal patterns that tests expect to be blocked:

- Dotted-slash variants: `.....//` sequences (more than two dots, mixed slashes)
- Windows backslash traversals: `..\\..\\windows\\system32`

The function normalizes paths with `os.path.normpath` and checks for `..` components, but this doesn't catch all variants. There may also be a dead-code issue: `normalized_path` is computed but unused in part of the logic.

Failing tests: `test_path_traversal` in `test_attack_surface.py` and `test_integration_security.py`.

## Solution

Review `_is_safe_path` normalization logic. Candidates:
1. After `os.path.normpath`, check that the resolved path doesn't escape the intended base directory using `os.path.commonpath` or `startswith`.
2. For Windows backslash: normalize both `/` and `\\` before checking.
3. Remove the `normalized_path` dead-code if it's a leftover from a prior attempt.

Tighten the detection to block all traversal variants the tests cover without over-blocking legitimate deep paths.
