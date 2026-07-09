# Phase 9 Research: Advanced Training Knobs

**Date:** 2026-07-09  
**Phase:** 09 — Add `--nn-feature-extraction`, `--gof-test`, and `--quantization-mode` CLI flags

---

## Finding 1: nn-feature-extraction and gof-test insertion point in cli.py

`_add_training_args` is defined at `mmcli/cli.py:345`. The `--feature-extraction` arg is at
lines 358–376. Phase 8 adds `--dataset-preset` immediately after `--feature-extraction`.

For Phase 9: `--nn-feature-extraction` and `--gof-test` should be added after the
`--dataset-preset` block (Phase 8's addition), continuing the same feature-extraction group.

This keeps all FE-related arguments adjacent:
```
--feature-extraction       (existing)
--dataset-preset           (Phase 8)
--nn-feature-extraction    (Phase 9)
--gof-test                 (Phase 9)
```

**Dependency note**: Plan 09-01 `depends_on: []` but it logically depends on Phase 8
completing so `--dataset-preset` exists in `_add_training_args`. The plan says
"after `--dataset-preset`" — executor must verify Phase 8 executed first.

---

## Finding 2: nn_for_feature_extraction and gof_test wiring in builder.py

The feature extraction section in `build_config` (lines 173–178):
```python
_set(config, "data_processing_feature_extraction", "feature_extraction_name", ...)
```

Phase 8 adds immediately after:
```python
_set(config, "dataset", "dataset_name", getattr(args, "dataset_preset", None))
```

Phase 9 adds immediately after the dataset_preset line:
```python
if getattr(args, "nn_feature_extraction", False):
    config.setdefault("data_processing_feature_extraction", {})
    config["data_processing_feature_extraction"]["nn_for_feature_extraction"] = True

if getattr(args, "gof_test", False):
    config.setdefault("data_processing_feature_extraction", {})
    config["data_processing_feature_extraction"]["gof_test"] = True
```

`getattr(..., False)` is critical — these flags won't exist in the args namespace for
subcommands that don't call `_add_training_args` (compile, init, info). The conditional
skip on `False` also avoids injecting keys with `False` values into the config (modelmaker
defaults to False; injecting False would be redundant and potentially confusing).

---

## Finding 3: --quantization-mode insertion point in cli.py (CORRECTED)

The quantization block in `_add_training_args` (lines 423–481):
```
--quantization              choices=QUANTIZATION_OPTIONS (line 423)
  aq = mutex group
  --auto-quantization       (line 435)
  --no-auto-quantization    (line 445)
--autoquant-tolerance-classification  (line 449)
--autoquant-tolerance-regression      (line 456)
--autoquant-tolerance-forecasting     (line 464)
--autoquant-tolerance-anomaly         (line 472, last autoquant arg)
# Performance optimization flags (advanced)  ← line 483 comment
```

**The existing plan 09-02 says "after the mutex group" — this is BETWEEN the mutex group
and the tolerance args (lines 448–449). This is incorrect.**

**Correct insertion point**: After `--autoquant-tolerance-anomaly` (line 472–481) and before
the `# Performance optimization flags (advanced)` comment (line 483). This keeps all
auto-quantization-related flags together without splitting them.

Use `group.add_argument` (same training options group), not `aq` (the mutex group).

```python
group.add_argument(
    "--quantization-mode",
    dest="quantization_mode",
    choices=["qat", "ptq"],
    default=None,
    metavar="MODE",
    help=(
        "Quantization training mode (requires --quantization QUANTIZATION_TINPU).\n"
        "  qat  Quantization-aware training — learns quantization during training (default)\n"
        "  ptq  Post-training quantization — quantizes after training completes\n"
        "QAT generally produces higher accuracy; PTQ is faster."
    ),
)
```

---

## Finding 4: quantization_mode wiring in builder.py (insertion point)

The quantization wiring block (lines 186–200):
```python
quant = getattr(args, "quantization", None)
_set(config, "training", "quantization", ...)
_set(config, "training", "auto_quantization", ...)
_set(config, "training", "autoquant_tolerance_classification", ...)
_set(config, "training", "autoquant_tolerance_regression", ...)
_set(config, "training", "autoquant_tolerance_forecasting", ...)
_set(config, "training", "autoquant_tolerance_anomaly", ...)
# ← INSERT HERE (after all autoquant tolerance lines)
_set(config, "training", "compile_model", ...)   # ← Performance section starts
_set(config, "training", "native_amp", ...)
```

Insert `--quantization-mode` wiring AFTER the last `autoquant_tolerance_anomaly` line and
BEFORE `_set(config, "training", "compile_model", ...)`:

```python
quant_mode = getattr(args, "quantization_mode", None)
if quant_mode is not None:
    config["training"]["quantization_mode"] = quant_mode
```

`config.setdefault("training", {})` is not needed — `config["training"]` is always
present from `_SKELETON`. The `_set` helper would work too but the plan uses direct
assignment since `quant_mode` is checked for `None` explicitly.

---

## Finding 5: quantization_mode in modelmaker

`quantization_mode` does NOT appear in the mock modelmaker (`/private/tmp/tinyml_modelmaker_mock`).
This is expected — the mock is minimal. The field is documented in the gap analysis as
coming from `tinyml_torchmodelopt/quantization/__init__.py` which exports both
`GenericTinyMLQATFxModule` and `GenericTinyMLPTQFxModule`.

Whether modelmaker's training config accepts `quantization_mode` as a key is unverifiable
without the real package. The builder should write it; if modelmaker ignores it, no harm
is done. The key is user-visible discoverability.

---

## Finding 6: test_advanced_training_knobs.py — _make_args defaults

The test file in plan 09-02 uses `_make_args(**kwargs)` with a defaults dict. After Phases 8
and 9 execute, `build_config` will try to access:
- `args.dataset_preset` (Phase 8) → must be in defaults (`dataset_preset=None`)
- `args.nn_feature_extraction` (Phase 9) → must be in defaults (`nn_feature_extraction=False`)
- `args.gof_test` (Phase 9) → must be in defaults (`gof_test=False`)
- `args.quantization_mode` (Phase 9) → must be in defaults (`quantization_mode=None`)

The existing plan 09-02's `_make_args` already includes `nn_feature_extraction=False`,
`gof_test=False`, and `quantization_mode=None`. It also includes `dataset_preset=None`.
These are correct.

---

## Plan corrections needed in existing plans

**09-02 Task 1**: The `--quantization-mode` insertion point is described as "after the
auto-quantization/no-auto-quantization mutually exclusive group". The CORRECT insertion point
is AFTER all `--autoquant-tolerance-*` args (after line 481), before `# Performance
optimization flags (advanced)` (line 483). Update `<read_first>` and `<action>` accordingly.

**09-02 Task 2 builder.py**: The plan says "after the block that wires `--quantization` into
the config". Clarify: insert after the last `autoquant_tolerance_anomaly` line (not right
after the `_set(config, "training", "quantization", ...)` line). Use lines 186–201 range.

**09-01 implicit Phase 8 dependency**: Plan says "after `--dataset-preset` block" — executor
must have completed Phase 8 first. The depends_on metadata says `[]` but ROADMAP says
"Depends on: Phase 8". Clarify in plan that `--dataset-preset` is a Phase 8 addition.

---

## No changes needed to

- `NAS_SUPPORTED_TASKS` (not a NAS feature)
- `mmcli/info.py` (no discoverability changes needed for these knobs — advanced users)
- `mmcli/cli.py` MODULES or QUANTIZATION_OPTIONS constants
- Any module other than `mmcli/cli.py` and `mmcli/builder.py`
