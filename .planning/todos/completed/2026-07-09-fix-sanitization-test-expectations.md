---
created: 2026-07-09T12:55:08.142Z
title: Fix sanitization test expectations for raises-only design
area: testing
files:
  - tests/test_attack_surface.py
  - tests/test_fuzz_sanitization.py
  - tests/test_integration_security.py
  - mmcli/cli.py
---

## Problem

Phase 5 (commit eb0a1bd) redesigned `_sanitize_input` to raise `ValueError` on length violation instead of stripping dangerous characters. Twelve tests still expect the old stripping behavior: they pass dangerous inputs (`;`, `|`, `` ` ``, `$`) and expect a sanitized string back. Now `_sanitize_input` never strips — it only raises on length — so these tests fail because the function returns the original string unchanged.

Affected tests:
- `test_attack_surface.py`: `test_removes_dangerous_chars_*` (4 tests), `test_blocks_shell_metacharacters`, `test_sanitize_enforces_length`
- `test_fuzz_sanitization.py`: `test_fuzz_sanitize_semicolon`, `test_fuzz_sanitize_dollar_sign`, `test_fuzz_sanitize_backtick`, `test_fuzz_sanitize_preserves_safe`
- `test_integration_security.py`: 3 tests checking dangerous char rejection

The design decision is: should `_sanitize_input` strip chars (permissive, old) or raise (strict, current)? The current design raises only on length, which means short dangerous strings pass through. This may be intentional (caller validates separately) or a security regression.

## Solution

Option A (preferred): Update all 12 tests to match the current raises-only design. Tests that expected stripping should instead assert the dangerous string is returned unchanged (or not called for callers that validate separately). `test_sanitize_enforces_length` should be updated to catch `ValueError` correctly.

Option B: Reconsider security design — reintroduce stripping for shell metacharacters on top of length check. Only pursue if the permissive design was unintentional.

Resolve the design question first by checking Phase 5 plan/summary for intent, then apply whichever fix is correct.
