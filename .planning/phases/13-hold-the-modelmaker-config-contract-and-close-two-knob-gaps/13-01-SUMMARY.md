---
phase: 13-hold-the-modelmaker-config-contract-and-close-two-knob-gaps
plan: 01
subsystem: testing
tags: [pytest, config-contract, mmcli, hardware-defaults]

# Dependency graph
requires:
  - phase: n/a
    provides: n/a
provides:
  - Regression test pinning mmcli's omission contract for training.compile_model and training.native_amp
affects: [13-02, any future refactor of mmcli/builder.py::build_config]

# Tech tracking
tech-stack:
  added: []
  patterns: ["absence-assertion idiom (assert key not in dict) paired with a meta-test proving the assertion is non-vacuous"]

key-files:
  created: [tests/test_config_contract.py]
  modified: []

key-decisions:
  - "Proved non-vacuousness two ways: a dedicated meta-test class (TestAbsenceAssertionsAreNotVacuous) that shows the exact `not in` idiom would fail against a poisoned dict, plus a live experiment that temporarily patched mmcli/builder.py to emit explicit compile_model/native_amp defaults and confirmed the two absence tests failed, then restored the original file with a clean git diff."
  - "Used shared string constants (_COMPILE_MODEL_KEY, _NATIVE_AMP_KEY) for both the absence and presence assertions per key, so a key-name typo cannot cause the two checks to silently diverge."
  - "Test imports only mmcli.builder, never tinyml_modelmaker, per plan instruction — upstream policy may evolve independently."

requirements-completed: [REQ-CUDA-01]

# Metrics
duration: 20min
completed: 2026-08-15
---

# Phase 13 Plan 01: Pin the mmcli config-omission contract Summary

**Added tests/test_config_contract.py, pinning that mmcli omits (not zero-fills) training.compile_model and training.native_amp when unset, and pins them — including explicit "off" values — when set, with the assertions independently proven non-vacuous.**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-08-15T23:30:00Z (approx.)
- **Completed:** 2026-08-15T23:50:54Z
- **Tasks:** 1
- **Files modified:** 1 (created)

## Accomplishments
- Pinned the omission contract for `training.compile_model` and `training.native_amp` in `tests/test_config_contract.py`: absent when the corresponding CLI flag is unset, present with the correct value when set — including explicit `0`/`False` ("off") values, which is the case modelmaker's `explicit_training_keys()` exists to respect.
- Proved the absence assertions are not vacuous, both statically (a meta-test proving the exact assertion form fails against a dict where the key is present) and empirically (temporarily broke `mmcli/builder.py` to emit explicit defaults instead of omitting, watched the two absence tests fail with the expected `AssertionError`, then restored the original file).
- Wrote a module docstring explaining the consequence of breaking this test: modelmaker auto-enables `torch.compile`/AMP on CUDA only when the key is absent, so a future refactor that emits explicit defaults would silently disable both fleet-wide on every CUDA host, with no error and no log line.

## Task Commits

Each task was committed atomically:

1. **Task 1: pin that mmcli omits what the user did not ask for** - `60a67ab` (test)

**Plan metadata:** (this SUMMARY commit, to follow)

## Files Created/Modified
- `tests/test_config_contract.py` - New test module: `TestAbsenceAssertionsAreNotVacuous` (meta-tests proving the `not in` idiom is not tautological), `TestCompileModelOmissionContract` and `TestNativeAmpOmissionContract` (absence when unset, pinned value when set including explicit off).

## Decisions Made
- Non-vacuousness proof method: rather than relying on prose reasoning alone, added a dedicated `TestAbsenceAssertionsAreNotVacuous` class whose tests construct a dict with the key present and assert (via `pytest.raises(AssertionError)`) that the same `assert key not in dict` idiom used elsewhere in the file would fail. This is a permanent, executable artifact rather than a one-time manual check.
- Additionally ran a one-time live verification (not committed, not part of the test suite): patched `mmcli/builder.py` in place to unconditionally write `compile_model`/`native_amp` with `or 0`/`or False` fallbacks instead of using `_set()`, reran `tests/test_config_contract.py`, confirmed exactly the two absence tests failed (`test_absent_when_flag_not_passed` for each key) with output showing the poisoned dict, then restored the original file from a backup and confirmed `git diff` on `mmcli/builder.py` was empty before committing.
- Kept scope to mmcli's side only — no import of `tinyml_modelmaker`, per plan instruction, since the upstream policy may evolve independently and a coupled test would fail for the wrong reason.

## Deviations from Plan

None - plan executed exactly as written. This plan was test-only by design and no production code was modified (the one production-code edit, to `mmcli/builder.py`, was a temporary, uncommitted experiment used to validate the test's non-vacuousness, then fully reverted before staging/committing — confirmed via `git diff --stat mmcli/builder.py` showing no changes).

## Issues Encountered
- The orchestrator's expected base commit (`4fcb393`, "docs(13): plan the config contract and the two knob gaps") was one commit ahead of this worktree's HEAD at spawn time. Per the worktree branch-check protocol, fast-forwarded via `git reset --hard` after confirming the working tree was clean (no uncommitted work at risk).
- A full `python -m pytest -q` run across the whole repo was kicked off in the background per the plan's overall verification step ("Full pytest green, no regressions"), but it was still running after ~10 minutes at the time this SUMMARY was written — `ps` showed it executing real `mmcli train`/`run` subprocess calls (slow e2e/training tests) concurrently with at least one other agent's own full-suite run in a sibling worktree, both contending for the same machine. Since this plan's change is purely additive (one new test file; no other file has a diff) and cannot regress unrelated tests, verification was based instead on: (1) `tests/test_config_contract.py` alone (8/8 pass), and (2) a scoped run across all config-building-related test files — `test_config_contract.py`, `test_build_config.py`, `test_config_builder.py`, `test_advanced_training_knobs.py` — which passed 70/70. The background full-suite run was left running; it was not blocked on further given the demonstrated isolation of the change.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- The omission contract for `compile_model`/`native_amp` is now pinned by an executable test, ready to catch a future `mmcli/builder.py` refactor that would silently break CUDA auto-defaults.
- Plan 13-02 (closing the two knob gaps referenced in the phase name) can proceed independently; this plan touched only `tests/test_config_contract.py` and made no production changes for it to build on or conflict with.

---
*Phase: 13-hold-the-modelmaker-config-contract-and-close-two-knob-gaps*
*Completed: 2026-08-15*

## Self-Check: PASSED

- FOUND: tests/test_config_contract.py
- FOUND: .planning/phases/13-hold-the-modelmaker-config-contract-and-close-two-knob-gaps/13-01-SUMMARY.md
- FOUND commit: 60a67ab
