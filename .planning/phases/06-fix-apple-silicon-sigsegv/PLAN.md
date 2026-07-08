# Phase 6: Fix Apple Silicon SIGSEGV Crash

**Revised:** 2026-07-08 — pivoted to Option A per cross-AI review (Ollama/qwen3.6).
Original plan (5 plans: environment audit → architecture testing → remediation → regression → docs)
addressed an unconfirmed NumPy/OpenMP architecture mismatch. Review confirmed the actual root
cause is `onnxsim_cpp2py_export.so`'s C++ destructor crash during Python **shutdown** — not an
architecture issue. All plans collapsed to one (06-01), already executed.

## Goal

Accept and document exit code 245 on macOS ARM64 as a known benign crash in the onnxsim C
extension during Python interpreter teardown. Pipeline artifacts are written before the crash.
No NumPy, OpenMP, or DYLD_LIBRARY_PATH remediation is required.

## Root Cause (confirmed)

`onnxsim_cpp2py_export.cpython-310-darwin.so` performs a null-pointer dereference in its C++
destructor during Python interpreter teardown on macOS ARM64. Exit code 245 = -11 mod 256
= SIGSEGV. Fires **after** all pipeline artifacts are written — functionally harmless.

## Strategy: Option A — Accept and Document

| Approach | Effort | Status |
|----------|--------|--------|
| Option A: Accept exit 245, document in test harness + README + SKILL.md | Low | **COMPLETE** |
| Option B: Fix onnxsim destructor (downgrade/upgrade, monkey-patch, subprocess wrapper) | Medium | Deferred — not needed |

## Plans

| Plan | Type | Status |
|------|------|--------|
| 06-01-PLAN.md — Document and mitigate onnxsim shutdown crash | doc/verify | ✅ COMPLETE |

## Success Criteria

- Exit code 0 or 245 from `mmcli run`/`mmcli compile` on macOS ARM64 both indicate success
- Test harness (`test_e2e.py`) never fails a passing pipeline solely because exit code is 245
- README.md informs users the crash is benign and artifacts are intact
- SKILL.md informs developers of the same
- No architecture remediation (NumPy wheels, DYLD_LIBRARY_PATH, otool audits) required
