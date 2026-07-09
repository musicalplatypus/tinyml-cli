# Phase 8: Dataset Preset Selection

**Milestone:** v1.2  
**Depends on:** Phase 7

## Goal

Expose `dataset_name` (the modelmaker dataset preset selector) through the CLI.

The modelmaker's `run_tinyml_modelmaker.py` reads `config['dataset']['dataset_name']` and
resolves it via `ModelRunner.get_dataset_preset_descriptions()`. mmcli's builder hardcodes
this to `"default"` with no user override. Users who want a named preset (e.g.
`motor_fault_sample`) must write a full YAML config — there is no CLI flag.

This phase adds:
- `--dataset-preset` flag on `train`, `run` (and optionally `compile`) subcommands
- Builder wiring: `_set(config, "dataset", "dataset_name", args.dataset_preset)`
- `mmcli info` extension: when a task type is provided, also print available dataset presets
  via the `get_dataset_preset_descriptions()` registry call

## Plans

| Plan | Type | Status |
|------|------|--------|
| 08-01-PLAN.md — Add --dataset-preset flag + builder wiring | feat | PENDING |
| 08-02-PLAN.md — Extend mmcli info with dataset preset listing + tests | feat+tdd | PENDING |

## Success Criteria

- `mmcli train -m timeseries -t motor_fault -d F28P55 -n CLS_1k_NPU --dataset-preset motor_fault_sample` wires `dataset_name` into the config correctly
- `mmcli info -m timeseries -t motor_fault` lists available dataset preset names
- `--dataset-preset default` produces identical output to omitting the flag
- Existing train/run workflows with no `--dataset-preset` flag are unaffected
- `pytest tests/test_dataset_preset.py` — all tests pass
