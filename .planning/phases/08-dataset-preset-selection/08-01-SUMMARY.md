---
plan: 08-01
phase: 08-dataset-preset-selection
status: complete
commits:
  - 7024ff5
key-files:
  modified:
    - mmcli/cli.py
    - mmcli/builder.py
---

# Summary: 08-01 — --dataset-preset Flag + Builder Wiring

## What was built

1. **`--dataset-preset PRESET` flag** added to `_add_training_args` in `mmcli/cli.py`, immediately after the `--feature-extraction` block. Present in `train` and `run` subcommands; absent from `compile`.

2. **Builder wiring** in `mmcli/builder.py`: after the `feature_extraction_name` `_set` call (which follows the project_dir block), added:
   ```python
   _set(config, "dataset", "dataset_name", getattr(args, "dataset_preset", None))
   ```
   Placement after the project_dir block is critical: `_set` skips when value is None, so omitting `--dataset-preset` preserves the `basename(project_dir)` value; providing a preset overrides it.

## Acceptance Criteria

- `--dataset-preset` present in `train --help` ✓
- `--dataset-preset` present in `run --help` ✓
- `--dataset-preset` absent from `compile --help` ✓
- `preset=None` → `dataset_name == basename(project)` ("default") ✓
- `preset="motor_fault_sample"` → `dataset_name == "motor_fault_sample"` ✓
- `preset` overrides basename when project is "my_project" ✓

## Self-Check: PASSED
