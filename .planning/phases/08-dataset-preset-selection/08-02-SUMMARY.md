---
plan: 08-02
phase: 08-dataset-preset-selection
status: complete
commits:
  - ac8ca58
key-files:
  modified:
    - mmcli/info.py
  created:
    - tests/test_dataset_preset.py
---

# Summary: 08-02 — mmcli info Dataset Preset Listing + Tests

## What was built

1. **`_QUERY_SCRIPT` extension** in `mmcli/info.py`: after `result["fe_presets"] = fe_presets`, added a `# --- Dataset presets ---` block using `training.ModelRunner.init_params()` and `get_dataset_preset_descriptions()`. Wrapped in `try/except` so missing method gracefully falls back to `[]`.

2. **`_print_task_details` extension**: after the FE Presets section, added a Dataset Presets section that prints `"Dataset Presets --dataset-preset (N available):"` followed by sorted preset names. Skips section when empty.

3. **`tests/test_dataset_preset.py`**: 7 tests covering:
   - `--dataset-preset` present in train/run help
   - `--dataset-preset` absent from compile help
   - `preset=None` → `dataset_name == basename(project)` ("default")
   - `preset="motor_fault_sample"` → `dataset_name == "motor_fault_sample"`
   - `preset="default"` (explicit) is harmless
   - preset overrides project basename when both set

## Acceptance Criteria

- All 7 tests pass ✓
- `info.py` imports cleanly ✓
- JSON output via `format_json` auto-includes `dataset_presets` key (no change needed) ✓
- When presets empty/not available, no "Dataset Presets" clutter in text output ✓

## Self-Check: PASSED
