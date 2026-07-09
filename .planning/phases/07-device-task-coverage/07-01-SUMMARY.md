---
plan: 07-01
phase: 07-device-task-coverage
status: complete
commits:
  - ad10a16
  - 165097f
key-files:
  created:
    - tests/test_device_task_coverage.py
  modified:
    - mmcli/cli.py
tdd:
  red: ad10a16
  green: 165097f
---

# Summary: 07-01 — TDD Device & Task Coverage

## What was built

Closed two HIGH-priority discoverability gaps in `mmcli/cli.py`:

1. **F28E12 device** — added to `TARGET_DEVICES` list between `"F2837"` and `"F28P55"`. Also reflected in the `--device` help text (C2000 section).
2. **`TASK_TYPES_AUDIO` constant** — added `TASK_TYPES_AUDIO = ["audio_classification"]` after `TASK_TYPES_VISION`. Added Audio tasks section to `--task` help text in `_add_common_args`.

## TDD Gate

- **RED** (`test(07-01)` @ `ad10a16`): 12-test file created; 9 tests failed before implementation
- **GREEN** (`feat(07-01)` @ `165097f`): all 12 tests pass after implementation

## Deviations

- `test_deploy_help_lists_f28e12` was updated in the GREEN commit: the original test incorrectly used `mmcli deploy --help` (a sub-dispatcher that shows subcommands only). Corrected to `mmcli deploy check-sdk --help` which does show `choices=TARGET_DEVICES` with F28E12.

## Self-Check: PASSED

- `"F28E12" in TARGET_DEVICES` ✓ (position between F2837 and F28P55 ✓)
- `TASK_TYPES_AUDIO = ["audio_classification"]` ✓
- `--task` help text includes "Audio tasks:\n  audio_classification" ✓
- `--device` help text includes "F28E12" on C2000 line ✓
- `NAS_SUPPORTED_TASKS` not modified ✓
- `info.py` not modified ✓
- All 12 tests pass ✓
