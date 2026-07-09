# Phase 9: Advanced Training Knobs

**Milestone:** v1.2  
**Depends on:** Phase 8

## Goal

Expose three MEDIUM-priority modelmaker capabilities that are currently config-file-only:

1. **`--nn-feature-extraction`** — enables neural-network-based feature extraction instead
   of the default classical signal-processing pipeline. Corresponds to
   `data_processing_feature_extraction.nn_for_feature_extraction` in timeseries params.
   Primarily useful when classical FE doesn't generalise and the user wants to learn the
   feature transform end-to-end.

2. **`--gof-test`** — runs a goodness-of-fit statistical test during training. Corresponds
   to `data_processing_feature_extraction.gof_test` in timeseries params. Niche but
   useful for domain validation before committing to a long training run.

3. **QAT vs PTQ mode selection** — `torchmodelopt/quantization/` provides both
   `GenericTinyMLQATFxModule` (quantization-aware training) and
   `GenericTinyMLPTQFxModule` (post-training quantization). Currently `--quantization
   QUANTIZATION_TINPU` picks a path without user control. A `--quantization-mode qat|ptq`
   flag (or a new `QUANTIZATION_TINPU_QAT` / `QUANTIZATION_TINPU_PTQ` choice on the
   existing `--quantization` flag) would expose this distinction.

## Plans

| Plan | Type | Status |
|------|------|--------|
| 09-01-PLAN.md — Add --nn-feature-extraction + --gof-test flags + builder wiring | feat | PENDING |
| 09-02-PLAN.md — QAT vs PTQ mode flag + tests for all three knobs | feat+tdd | PENDING |

## Success Criteria

- `mmcli train ... --nn-feature-extraction` sets `nn_for_feature_extraction=True` in the config
- `mmcli train ... --gof-test` sets `gof_test=True` in the config
- `mmcli train ... --quantization QUANTIZATION_TINPU --quantization-mode ptq` selects PTQ path
- Omitting all three flags preserves existing default behaviour (no regression)
- `pytest tests/test_advanced_training_knobs.py` — all tests pass
- `mmcli train --help` documents all three new flags with one-line descriptions
