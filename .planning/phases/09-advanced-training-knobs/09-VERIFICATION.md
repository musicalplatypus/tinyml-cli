---
status: passed
phase: 09-advanced-training-knobs
verified: 2026-07-09
---

# Verification: Phase 9 — Advanced Training Knobs

## Goal Assessment

**Goal:** Add `--nn-feature-extraction`, `--gof-test`, and `--quantization-mode` flags to expose advanced training configuration without YAML editing.

**Verdict: ACHIEVED** ✓

## Must-Have Checks

| Check | Status | Evidence |
|-------|--------|---------|
| `--nn-feature-extraction` in `train --help` | ✓ PASS | pytest + manual grep |
| `--nn-feature-extraction` in `run --help` | ✓ PASS | pytest |
| `nn_for_feature_extraction` absent when flag omitted | ✓ PASS | `test_false_by_default_key_absent` |
| `nn_for_feature_extraction = True` when flag set | ✓ PASS | `test_true_when_flag_set` |
| `--gof-test` in `train --help` | ✓ PASS | pytest |
| `gof_test` absent when flag omitted | ✓ PASS | `test_false_by_default_key_absent` |
| `gof_test = True` when flag set | ✓ PASS | `test_true_when_flag_set` |
| `--nn-feature-extraction` and `--gof-test` are independent | ✓ PASS | `test_independent_of_nn_fe` |
| `--quantization-mode` in `train --help` | ✓ PASS | pytest |
| `quantization_mode` absent when not set | ✓ PASS | `test_none_by_default_key_absent` |
| `quantization_mode = "ptq"` propagates | ✓ PASS | `test_ptq_mode_propagates` |
| `quantization_mode = "qat"` propagates | ✓ PASS | `test_qat_mode_propagates` |
| Invalid mode rejected by argparse | ✓ PASS | `test_invalid_mode_rejected` |
| All 13 Phase 9 tests pass | ✓ PASS | `pytest tests/test_advanced_training_knobs.py` 13/13 |
| No regressions in Phase 7+8 tests | ✓ PASS | 32/32 combined |

## Key Correctness Properties

- **Absent-key semantics** throughout: `bool` flags use conditional injection (`if flag: config[...] = True`), `quantization_mode` uses `if quant_mode is not None`. Keys are never present as `False`/`None` — modelmaker defaults apply when absent.
- `config["training"]` always present from `_SKELETON`; `getattr(..., None)` makes builder safe for compile subcommand (no training args).
- `choices=["qat", "ptq"]` with `metavar="MODE"` means argparse shows `MODE` in usage line but hard-rejects invalid values.
