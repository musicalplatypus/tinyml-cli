---
plan: 09-02
phase: 09-advanced-training-knobs
status: complete
commits:
  - 1e0a214
key-files:
  modified:
    - mmcli/cli.py
    - mmcli/builder.py
  created:
    - tests/test_advanced_training_knobs.py
---

# Summary: 09-02 — Quantization Mode Flag and Tests

## What was built

1. **`--quantization-mode MODE` flag** in `mmcli/cli.py`: inserted after `--autoquant-tolerance-anomaly`, before `# Performance optimization flags` comment. `choices=["qat", "ptq"]`, `default=None`, `metavar="MODE"`. argparse rejects invalid choices automatically.

2. **Builder wiring** in `mmcli/builder.py`: after `autoquant_tolerance_anomaly` `_set`:
   ```python
   quant_mode = getattr(args, "quantization_mode", None)
   if quant_mode is not None:
       config["training"]["quantization_mode"] = quant_mode
   ```
   Absent-key semantics — key only appears when mode is explicitly set.

3. **`tests/test_advanced_training_knobs.py`**: 13 tests across `TestNNFeatureExtractionFlag`, `TestGofTestFlag`, `TestQuantizationModeFlag`. All 13 pass.

## Acceptance Criteria

- `--quantization-mode` in train help ✓
- `quantization_mode` key absent when not set ✓
- ptq and qat both propagate ✓
- Invalid mode rejected by argparse (exit != 0, "invalid choice" in stderr) ✓
- All 13 Phase 9 tests pass ✓
- All 32 Phase 7+8+9 tests pass ✓

## Self-Check: PASSED
