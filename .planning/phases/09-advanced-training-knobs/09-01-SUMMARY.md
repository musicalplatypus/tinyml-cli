---
plan: 09-01
phase: 09-advanced-training-knobs
status: complete
commits:
  - a5120f1
key-files:
  modified:
    - mmcli/cli.py
    - mmcli/builder.py
---

# Summary: 09-01 — NN Feature Extraction and GOF Test Flags

## What was built

1. **`--nn-feature-extraction` flag** in `mmcli/cli.py` (`_add_training_args`): inserted after `--dataset-preset`, before `--epochs`. `action="store_true"`, `default=False`. Shared via `_add_training_args` → present in train/run, absent from compile.

2. **`--gof-test` flag** in `mmcli/cli.py`: same position, same mechanics.

3. **Builder wiring** in `mmcli/builder.py`: after `dataset_name` `_set`, conditional injection:
   - `if nn_feature_extraction: config["data_processing_feature_extraction"]["nn_for_feature_extraction"] = True`
   - `if gof_test: config["data_processing_feature_extraction"]["gof_test"] = True`
   - Uses absent-key semantics — keys only appear when flag is True.

## Acceptance Criteria

- Both flags in train/run help ✓
- Neither key in config when flags omitted ✓
- Both keys set when flags provided ✓
- Flags are independent ✓

## Self-Check: PASSED
