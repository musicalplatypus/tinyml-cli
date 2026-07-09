---
status: complete
phase: 07-device-task-coverage
source:
  - .planning/phases/07-device-task-coverage/07-01-SUMMARY.md
  - .planning/phases/07-device-task-coverage/07-02-SUMMARY.md
started: 2026-07-09T00:00:00Z
updated: 2026-07-09T00:00:01Z
---

## Current Test

[testing complete]

## Tests

### 1. F28E12 in train --device help
expected: Running `mmcli train --help` shows F28E12 in the --device section under C2000 targets.
result: pass

### 2. audio_classification in train --task help
expected: Running `mmcli train --help` shows "audio_classification" listed under the --task option (under an "Audio tasks" heading or similar).
result: issue
reported: "it appears under module, but not under task"
severity: minor

### 3. F28E12 in deploy subcommand help
expected: Running `mmcli deploy check-sdk --help` shows F28E12 among the valid device choices for --device.
result: pass

### 4. F28E12 accepted by train command
expected: Running `mmcli train -m timeseries -t motor_fault -d F28E12 --model CLS_1k_NPU` does NOT error with "invalid choice: 'F28E12'". It may fail for other reasons (missing project, modelmaker not installed) but not because F28E12 is unrecognized.
result: pass

## Summary

total: 4
passed: 3
issues: 1
pending: 0
skipped: 0

## Gaps

- truth: "--task help should clearly show 'Audio tasks: audio_classification' as a distinct section, not buried in wrapped line continuation from Vision tasks"
  status: failed
  reason: "User reported: 'it appears under module, but not under task' — argparse word-wrap splits 'Audio' onto end of Vision line; 'tasks: audio_classification' starts next line without label, making Audio tasks section visually invisible"
  severity: minor
  test: 2
  artifacts: [mmcli/cli.py]
  missing: []
