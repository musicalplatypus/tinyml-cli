---
phase: 13-hold-the-modelmaker-config-contract-and-close-two-knob-gaps
plan: 04
subsystem: testing
tags: [pytest, preset-selection, timeseries, findings]

# Dependency graph
requires:
  - phase: 13-hold-the-modelmaker-config-contract-and-close-two-knob-gaps
    provides: channel-aware feature-extraction preset selection (a7804ca, plans 13-01/13-02); F-2 upstream-gap assertion pattern established in test_tier4_cli.py (13-03)
provides:
  - test_cli_integration.py::TestDryRun now pins the correct upstream-gap message for regression, forecasting, and anomalydetection (it was never in 13-03's scope, per plan)
  - test_tier4_cli.py::TestDryRunCrossDevice now pins the F-9 channel-mismatch message for the remaining 3 generic_timeseries_regression cases
  - F-9 recorded in FINDINGS-training-matrix.md, with a cross-cutting "3 of 4 generic timeseries task types cannot auto-select a preset" observation
affects: [testing, preset-selection, findings-training-matrix]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two distinct upstream-gap failure paths from choose_preset get two distinct assertions: zero presets (F-2, 'has no feature-extraction presets available ... gap in the upstream preset catalog') vs. presets exist but none channel-matches (F-9, 'No usable feature-extraction preset for task ... matches the N input channel(s)'). Never conflate them or assert a bare non-zero exit."

key-files:
  created: []
  modified:
    - tests/test_cli_integration.py
    - tests/test_tier4_cli.py
    - .planning/FINDINGS-training-matrix.md

key-decisions:
  - "test_cli_integration.py::TestDryRun was out of 13-03's scoped file set entirely (a planning gap in 13-03, not an execution one) — it needed the same regression/forecasting/anomalydetection treatment as test_tier4_cli.py, applied here for the first time."
  - "Verified each error message by running the CLI directly against the exact fixture shape each test uses, rather than inferring from the source: forecasting and anomalydetection both raise choose_preset's 'if not presets' branch (F-2); regression raises the 'if not usable' branch with 2 detected input channels (F-9), never the 11 the catalog's one usable preset requires."
  - "Recorded regression's gap as new finding F-9 rather than extending F-2 or F-5 — F-2 is zero presets, F-5 (as corrected) was classification's missing-default problem already fixed by channel-aware selection; F-9 is presets existing but none fitting any realistic dataset shape, which is a different, currently-open catalog gap."
  - "Added a cross-cutting table to FINDINGS-training-matrix.md showing only generic_timeseries_classification can currently auto-select a preset; regression/forecasting/anomalydetection each block on their own upstream catalog gap (F-9/F-2/F-2 respectively)."

requirements-completed: [REQ-CUDA-01]

# Metrics
duration: ~20min
completed: 2026-08-15
---

# Phase 13 Plan 04: Gap Closure — Finish the Preset-Gap Test Fixes Summary

**Pinned the correct upstream-gap error (F-2 zero-presets vs. F-9 channel-mismatch) for all 6 remaining dry-run failures across `test_cli_integration.py` and `test_tier4_cli.py`, and recorded the previously-untracked regression gap as finding F-9 — no changes to `mmcli/`.**

## Performance

- **Duration:** ~20 min
- **Completed:** 2026-08-15T~21:10Z (commit timestamps; local execution ran into 2026-08-16 UTC)
- **Tasks:** 2/2 completed
- **Files modified:** 3

## Accomplishments
- `tests/test_cli_integration.py::TestDryRun::test_dry_run_generates_config` — the file 13-03's plan never scoped — now special-cases `generic_timeseries_forecasting`/`generic_timeseries_anomalydetection` (F-2, zero presets) and `generic_timeseries_regression` (F-9, channel mismatch) with their own specific error-text assertions instead of an impossible `rc == 0`. `classification` is untouched.
- `tests/test_tier4_cli.py::TestDryRunCrossDevice::test_dry_run_valid_config` — already special-cased forecasting for F-2 (13-03) — now also special-cases the 3 remaining `generic_timeseries_regression` × device cases for F-9, using the same pattern.
- `generic_timeseries_regression`'s preset gap is now tracked as **F-9** in `FINDINGS-training-matrix.md`, distinguished from F-2 (zero presets) and F-5 (classification's already-fixed missing-default problem), plus a cross-cutting note that only classification of the 4 generic timeseries task types can currently auto-select a preset.

## Task Commits

Each task was committed atomically:

1. **Task 1: pin the right error for each task type, in the file 13-03 missed** - `a7d4c4c` (test)
2. **Task 2: record the regression gap as F-9** - `e6e6a74` (docs)

**Plan metadata:** (this commit, `13-04-SUMMARY.md`)

## Files Created/Modified
- `tests/test_cli_integration.py` - `TestDryRun::test_dry_run_generates_config` special-cases forecasting/anomalydetection (assert the F-2 "no feature-extraction presets available ... gap in the upstream preset catalog" text) and regression (assert the F-9 "No usable feature-extraction preset for task ... matches the N input channel(s)" text). Docstring explains all three branches and cites both findings.
- `tests/test_tier4_cli.py` - `TestDryRunCrossDevice::test_dry_run_valid_config` adds the same F-9 regression branch alongside the existing F-2 forecasting branch from 13-03, following its exact assertion style (specific text, not bare `rc != 0`).
- `.planning/FINDINGS-training-matrix.md` - new `## F-9` section (2 presets, 1 usable, requires 11 channels; the other preset matches realistic channel counts but declares no `feat_ext_transform`) plus a cross-cutting "3 of 4 generic timeseries task types cannot auto-select a preset" table; `STATUS OF FIXES` table gets an F-9 row (`open, upstream`).

## Decisions Made
- Task 1: verified every error message by actually invoking `mmcli --dry-run train` against each test's exact fixture (data shape and all), rather than inferring wording from reading `preset_selection.py`. Confirmed forecasting and anomalydetection both hit the zero-presets branch (identical text, F-2); regression hits the channel-mismatch branch with **2** detected channels (not 11), since both test fixtures use the same 3-raw-column CSV that becomes 2 input channels once the trailing target column is subtracted.
- Task 1: `test_cli_integration.py` was flagged in the plan as a planning gap in 13-03 (never scoped), not an execution error — fixed here as instructed, following `test_tier4_cli.py`'s established pattern rather than inventing a new one.
- Task 2: placed F-9 after F-8 (chronologically last) rather than inline near F-5/F-6, matching the file's existing pattern of appending new findings at the point they were confirmed rather than reordering by topic. Added the cross-cutting 3-of-4 observation as its own subsection within F-9 since it synthesizes F-2 and F-9 together and the plan asked for it to be captured, not just the individual entry.

## Deviations from Plan

None - plan executed exactly as written. No changes to `mmcli/`; both target test files verified green (excluding the pre-existing worktree artifact noted below).

## Issues Encountered

**`TestInitDatasetExtractReal` (10 failures) is a worktree environment artifact, not a code or test defect** — confirmed identically to 13-03's finding. `mmcli/example_datasets/*.zip` files are gitignored and do not propagate into a fresh git worktree (untracked files are not shared between the primary checkout and linked worktrees). These are unrelated to preset selection and expected to pass in the orchestrator's full-suite run against the real checkout. No action taken, consistent with the plan's explicit note about this class of failure.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `tests/test_cli_integration.py` and `tests/test_tier4_cli.py` are fully green except for the 10 `TestInitDatasetExtractReal` cases (worktree-only artifact, expected to pass at merge). Verified: 45 passed, 20 deselected (`TestInitDatasetExtractReal`), 0 failures across both files together.
- No changes to `mmcli/` — confirmed via `git diff --stat` against this plan's base commit returning empty for the `mmcli/` path. The channel-aware selector's behavior is unchanged; both failure paths it produces (F-2 zero-presets, F-9 channel-mismatch) are now correctly asserted rather than expecting an impossible success.
- All 6 previously-remaining failures from the 13-hold-the-modelmaker-config-contract-and-close-two-knob-gaps phase's gap-closure chain (13-03 → 13-04) are now resolved as test-suite fixes; the underlying upstream catalog gaps (F-2, F-9) remain open and are documented for a future upstream fix, not this fork.

## Self-Check: PASSED

- FOUND: tests/test_cli_integration.py
- FOUND: tests/test_tier4_cli.py
- FOUND: .planning/FINDINGS-training-matrix.md
- FOUND: a7d4c4c (git log)
- FOUND: e6e6a74 (git log)

---
*Phase: 13-hold-the-modelmaker-config-contract-and-close-two-knob-gaps*
*Completed: 2026-08-15*
