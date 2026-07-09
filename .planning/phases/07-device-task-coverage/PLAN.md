# Phase 7: Device & Task Coverage

**Milestone:** v1.2  
**Depends on:** Phase 6 (complete)

## Goal

Close the two HIGH-priority discoverability gaps found in the gap analysis:

1. **`F28E12` device is missing** from mmcli's `TARGET_DEVICES` list despite being valid in
   `tinyml_modelmaker.ai_modules.timeseries.constants.TARGET_DEVICES`. Users who specify
   this C2000 device get a validation error from mmcli instead of a useful pipeline run.

2. **`audio_classification` task type is invisible.** The `audio` module is listed in
   `MODULES` and works end-to-end, but no `TASK_TYPES_AUDIO` constant exists in `cli.py`,
   the `--task` help text omits all audio tasks, and `mmcli info -m audio` produces no
   task-type enumeration. Users have no CLI-discoverable path to audio classification.

Both gaps are in `mmcli/cli.py` constants and help text — no backend changes needed.

## Mode: TDD

Wave ordering follows RED → GREEN discipline: tests written first (failing), then implementation.

## Plans

| Plan | Wave | Type | Status |
|------|------|------|--------|
| 07-01-PLAN.md — TDD: write failing tests → add F28E12 + TASK_TYPES_AUDIO (RED→GREEN) | 1 | tdd | PENDING |
| 07-02-PLAN.md — Regression guard: full suite verification | 2 | execute | PENDING |

## Success Criteria

- `mmcli train -m timeseries -t arc_fault -d F28E12 --help` exits 0 (device accepted)
- `mmcli info -m audio` lists `audio_classification` as a supported task
- `mmcli train --help` help text includes `audio_classification` under audio tasks
- `pytest tests/test_device_task_coverage.py` — all tests pass
- No regressions in existing device or task validation tests
