---
phase: 05
plan: 05-03
type: feat
status: complete
date_completed: "2026-07-08"
---

# Summary 05-03: Model Comparison Command

## Outcome

Complete. `mmcli compare` queries real modelmaker registry data via MMCLI_PYTHON subprocess (same pattern as `mmcli info`) and produces an accurate side-by-side comparison of model availability across task types.

## What Was Delivered

- `mmcli/compare.py` — rewritten to query real modelmaker data (no mocks)
  - `_query_models()` — subprocess query using MMCLI_PYTHON
  - `compare_models(module_type, task_types, device)` — aggregates per-task model counts, quantization types, supported device counts
  - `format_comparison()` — summary table + per-task model list
- `mmcli/cli.py` — updated call site: `models=` → `task_types=`, `--all-models` uses full task type strings

## Example Output

```
mmcli compare -m timeseries \
    --model1 generic_timeseries_classification \
    --model2 generic_timeseries_regression
```

Shows 24 classification models vs 11 regression models, real device support counts, quantization types — all from modelmaker registry.

## Fix Applied

Prior implementation returned hardcoded mock values (`accuracy: 0.95`, `size_kb: 128`) for every model, making all comparisons meaningless. Replaced with real subprocess query.
