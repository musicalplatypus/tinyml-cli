---
status: complete
phase: 09-advanced-training-knobs
source:
  - .planning/phases/09-advanced-training-knobs/09-01-SUMMARY.md
  - .planning/phases/09-advanced-training-knobs/09-02-SUMMARY.md
started: 2026-07-09T00:00:00Z
updated: 2026-07-09T00:00:00Z
---

## Current Test

[testing complete]

## Tests

### 1. --nn-feature-extraction and --gof-test in train --help
expected: `mmcli train --help` lists both `--nn-feature-extraction` and `--gof-test` as training options.
result: pass

### 2. --nn-feature-extraction and --gof-test absent from compile --help
expected: `mmcli compile --help` does NOT mention either flag (compile skips training args).
result: pass

### 3. --quantization-mode in train --help with qat/ptq
expected: `mmcli train --help` shows `--quantization-mode MODE` with qat and ptq mentioned in the help text.
result: pass

### 4. Invalid --quantization-mode rejected
expected: Running `mmcli train -m timeseries -t motor_fault -d F28P55 --model CLS_1k_NPU --quantization-mode invalid` exits non-zero with "invalid choice" in stderr.
result: pass

### 5. Valid --quantization-mode accepted
expected: Running `mmcli train -m timeseries -t motor_fault -d F28P55 --model CLS_1k_NPU --quantization-mode qat` fails with "Project directory not found" (or similar), NOT "invalid choice" — confirming qat is a recognized value.
result: pass

### 6. --nn-feature-extraction accepted by train
expected: Running `mmcli train -m timeseries -t motor_fault -d F28P55 --model CLS_1k_NPU --nn-feature-extraction` fails with "Project directory not found", NOT "unrecognized argument".
result: pass

## Summary

total: 6
passed: 6
issues: 0
pending: 0
skipped: 0

## Gaps

[none yet]
