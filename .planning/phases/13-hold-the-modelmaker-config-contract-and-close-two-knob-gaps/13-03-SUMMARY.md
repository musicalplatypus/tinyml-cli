---
phase: 13-hold-the-modelmaker-config-contract-and-close-two-knob-gaps
plan: 03
subsystem: testing
tags: [pytest, preset-selection, timeseries, fixtures]

# Dependency graph
requires:
  - phase: 13-hold-the-modelmaker-config-contract-and-close-two-knob-gaps
    provides: channel-aware feature-extraction preset selection (a7804ca, plans 13-01/13-02)
provides:
  - Corrected test_workflow.py classification fixture (single unlabelled column, matches real data)
  - test_tier4_cli.py forecasting dry-run cases pin the F-2 upstream-gap error instead of asserting an impossible success
affects: [testing, preset-selection, findings-training-matrix]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "When a task type has zero (or effectively zero usable) upstream presets, assert the specific error message rather than a bare non-zero exit or a false rc==0."

key-files:
  created: []
  modified:
    - tests/test_workflow.py
    - tests/test_tier4_cli.py

key-decisions:
  - "Fixed test_workflow.py::test_dry_run_train fixture to a single unlabelled data column (no header) — the real shape of classification data, verified against a dataset from a project that trained successfully — rather than relaxing the channel-aware preset selector."
  - "Pinned the F-2 upstream-gap error text for generic_timeseries_forecasting dry-run cases in TestDryRunCrossDevice instead of asserting rc == 0, since that task has zero feature-extraction presets and can never succeed."
  - "Left generic_timeseries_regression cases in TestDryRunCrossDevice untouched and failing, per the plan's explicit instruction — it genuinely has presets (unlike forecasting), so this is a new, undiagnosed finding rather than something this test-only plan is scoped to fix."

patterns-established:
  - "Pin upstream-catalog gaps with an assertion on the specific error text, citing the finding ID (F-2) and FINDINGS-training-matrix.md, so a future catalog fix surfaces as an obviously-outdated test rather than a silent pass."

requirements-completed: [REQ-CUDA-01]

# Metrics
duration: ~25min
completed: 2026-08-15
---

# Phase 13 Plan 03: Gap Closure — Test Fixes for Pre-Existing Preset-Selection Failures Summary

**Fixed the classification dry-run fixture to look like real single-column data, and pinned the F-2 upstream forecasting-preset gap as an expected, specific error instead of an impossible `rc == 0` — no production code touched.**

## Performance

- **Duration:** ~25 min
- **Completed:** 2026-08-15T~00:43Z
- **Tasks:** 2/2 completed
- **Files modified:** 2

## Accomplishments
- `tests/test_workflow.py::test_dry_run_train` now uses a single unlabelled data column (matching real classification data shape) and passes because the input is valid — not because the channel-aware preset selector was weakened.
- `tests/test_tier4_cli.py::TestDryRunCrossDevice::test_dry_run_valid_config` now asserts the specific F-2 upstream-gap error text for the 3 `generic_timeseries_forecasting` × device cases, citing `.planning/FINDINGS-training-matrix.md` so a future catalog fix is caught rather than silently passing.
- Discovered and documented (did not fix) a third, previously undiagnosed failure class: `generic_timeseries_regression` cases in the same parametrized test also fail, for a different reason than forecasting — see "New Finding" below.

## Task Commits

Each task was committed atomically:

1. **Task 1: make the classification fixture look like real data** - `9f5ca46` (test)
2. **Task 2: forecasting must assert the truth, not a success it cannot have** - `7ba9158` (test)

**Plan metadata:** (this commit, `13-03-SUMMARY.md`)

## Files Created/Modified
- `tests/test_workflow.py` - `test_dry_run_train` fixture changed from a two-column `feature,label` CSV to a single unlabelled data column, with a comment recording why the shape matters and citing the real dataset it was verified against.
- `tests/test_tier4_cli.py` - `TestDryRunCrossDevice::test_dry_run_valid_config` special-cases `generic_timeseries_forecasting`: asserts `rc != 0` plus the specific "no feature-extraction presets available" / "gap in the upstream preset catalog" error text, citing F-2. Classification and regression cases are unchanged.

## Decisions Made
- Task 1: Corrected the fixture rather than relaxing `assert rc == 0` — the selector's rejection of a 2-channel dataset for a task with no 2-channel preset is correct; the fixture, not the code, was wrong.
- Task 2: Pinned the *specific* upstream-gap error text (not a bare non-zero exit) for forecasting, so the test would fail loudly on a crash, typo, or unrelated regression, while still catching the moment upstream ships forecasting presets.
- Task 2 scope boundary: per the plan's explicit instruction ("do not apply this to other task types... that genuinely have presets — only forecasting is affected"), left `generic_timeseries_regression` cases untouched even though 3 of them still fail after this plan. See "New Finding" below.

## Deviations from Plan

None that required auto-fixing — no Rule 1/2/3 fixes were needed and `mmcli/` was not touched, as required. The one notable departure from the plan is documented as a **New Finding**, not a deviation, because it was explicitly called out as in-scope investigation ("check which parametrised cases actually fail... only some are forecasting") but out-of-scope to *fix*.

## New Finding (not fixed — needs a follow-up plan)

**`TestDryRunCrossDevice` also fails for `generic_timeseries_regression` on all 3 representative devices, for a reason distinct from F-2.**

Unlike forecasting (zero presets total), `generic_timeseries_regression` has two presets in the catalog:
- `Custom_Default` (`variables=1`) — declares no `feat_ext_transform`, so `choose_preset` correctly treats it as unusable (same rule documented in `mmcli/preset_selection.py`'s `_FRAME_STRUCTURE_KEYS` comment).
- `Generic_8Input_ABS_8Feature_1Frame` (`variables=11`) — usable, but requires an 11-input-channel dataset.

The shared `TestDryRunCrossDevice` fixture (`"1,2,3\n4,5,6\n"`, 3 raw columns) yields 2 detected input channels for regression once the trailing target column is subtracted (`TASKS_WITH_TRAILING_TARGET`), which matches neither preset. I manually confirmed this is not merely an unrealistic-fixture problem in the Task 1 sense: `mmcli/preset_selection.py`'s own docstring records the *real* shipped regression dataset shape as `x,y` (1 input + 1 target), which detects as 1 channel — and that also has no usable preset (`Custom_Default` is the only 1-channel match, and it's unusable). So under the current catalog, **no realistic regression dataset shape has a usable feature-extraction preset**; the only preset that would pass requires an 11-channel input no real dataset here would plausibly have. This is structurally similar to `F-5` ("the default FE preset is not viable for at least some classification models") but for regression, and is not currently tracked as its own finding.

Per this plan's explicit scope ("Do not apply this to other task types... only forecasting is affected... genuinely have presets... left alone" and "a production change here would be a finding to report, not a fix to make"), I did not modify `mmcli/`, the regression fixture, or the regression assertions. As a result:

- `tests/test_tier4_cli.py::TestDryRunCrossDevice::test_dry_run_valid_config[timeseries-generic_timeseries_regression-F28P55]` — still fails
- `tests/test_tier4_cli.py::TestDryRunCrossDevice::test_dry_run_valid_config[timeseries-generic_timeseries_regression-MSPM0G3507]` — still fails
- `tests/test_tier4_cli.py::TestDryRunCrossDevice::test_dry_run_valid_config[timeseries-generic_timeseries_regression-AM263]` — still fails

**Recommendation:** file this as a new finding in `.planning/FINDINGS-training-matrix.md` (candidate F-9) and let a follow-up plan decide, analogous to F-5's unresolved decision, whether to (a) change the regression fixture to an 11-channel shape purely for test purposes (misleading, since no real dataset looks like that), (b) add a usable low-channel preset to the catalog, or (c) pin the regression cases the same way as forecasting once it's confirmed no realistic shape can ever pass.

## Parametrized Cases Changed (Task 2)

- **Cases changed:** 3 — `generic_timeseries_forecasting` × `{F28P55, MSPM0G3507, AM263}` in `TestDryRunCrossDevice::test_dry_run_valid_config`.
- **Task types covered:** `generic_timeseries_forecasting` only. `generic_timeseries_classification` (already passing) and `generic_timeseries_regression` (still failing, see New Finding above) were left untouched.

## Issues Encountered

**`TestInitDatasetExtractReal` (10 failures) is a worktree environment artifact, not a code or test defect.** `mmcli/example_datasets/*.zip` files are gitignored (`.gitignore:10`) and exist only as untracked local files in the main checkout; git worktrees do not share untracked files with the primary working tree, so this fresh worktree has none of them (verified: only `generic_audio_classification.zip`, the one exception not covered by the glob ignore, is present). These 10 failures are unrelated to preset selection or this plan's scope and are expected to pass in the orchestrator's full-suite run against the real checkout. No action taken.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `tests/test_workflow.py` is fully green (4/4).
- `tests/test_tier4_cli.py` is green except for the 3 `generic_timeseries_regression` `TestDryRunCrossDevice` cases (New Finding above, needs its own follow-up plan) and the 10 `TestInitDatasetExtractReal` cases (worktree-only artifact, expected to pass at merge).
- No changes to `mmcli/` — the channel-aware selector's behavior is unchanged and confirmed correct in both cases this plan addressed.

---
*Phase: 13-hold-the-modelmaker-config-contract-and-close-two-knob-gaps*
*Completed: 2026-08-15*
