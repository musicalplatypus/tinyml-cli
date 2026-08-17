# Defect — the `arc_fault` task advertises four models, none of which exist

**Found:** 2026-08-14, training an `arc_fault` project from PlatypusStudio.
**Confirmed:** 2026-08-15, against `tinyml-tensorlab` at `6f2c11c` (latest modelzoo change).
**Component:** `tinyml-modelmaker` (advertises) / `tinyml-modelzoo` (registry).
**Severity:** the entire `arc_fault` task type is unusable. Not a degradation — no run can succeed.

## Summary

`tinyml-modelmaker`'s `arc_fault` task advertises four models:

| Advertised name | `model_training_id` |
|---|---|
| `ArcFault_model_1400_t` | `CNN_AF_3L_1400` |
| `ArcFault_model_700_t` | `CNN_AF_3L_700` |
| `ArcFault_model_300_t` | `CNN_AF_3L_300` |
| `ArcFault_model_200_t` | `CNN_AF_3L_200` |

**None exists in `tinyml-modelzoo`'s registry.** The registry holds 63 entries and **zero** contain
`_AF_`, so the whole arc-fault model family is missing rather than individual entries.

Any `arc_fault` run therefore fails at model construction:

```
ValueError: Model 'CNN_AF_3L_1400' not found in registry or specified model_spec.
```

## Why it is worse than a plain missing model

The failure lands **after** feature extraction and dataset loading have already succeeded. From a
real run:

```
INFO: root.load_data : Train Data: target count: 2654 : Split Up: arc(0): 1718 ; non_arc(1): 936
INFO: root.load_data : Val Data:   target count: 1344 : Split Up: arc(0): 945 ;  non_arc(1): 399
INFO: root.main      : Creating model
Traceback (most recent call last):
  ...
  File ".../tinyml_modelzoo/models/__init__.py", line 210, in get_model
    raise ValueError(f"Model '{model_name}' not found in registry or specified model_spec. ...")
ValueError: Model 'CNN_AF_3L_1400' not found in registry or specified model_spec.
```

So the user pays the full feature-extraction cost — minutes, on a real dataset — before learning
that the model they selected from the task's own advertised list cannot be built. Selection-time
validation would fail in milliseconds.

## Reproduction

```bash
mmcli train -m timeseries -t arc_fault -n ArcFault_model_1400_t -d F28P55 -i <arc_fault_project> --epochs 1
```

Or open any `arc_fault` project in PlatypusStudio and train. Reproduced on two separate projects
(`~/Documents/PlatypusStudio Projects/arc_1`, `~/Documents/edgeai/arc_fault`).

## Verification, and one method correction worth repeating

Confirmed with a **control**, because the first check was invalid:

```python
MZ.get_model("CNN_AF_3L_1400",           variables=1, num_classes=2)
#   -> ValueError: not found in registry          <- fails AT the lookup

MZ.get_model("CNN_TS_GEN_BASE_1P2K_NPU", variables=1, num_classes=2)
#   -> AssertionError: input_features must be provided  <- got PAST the lookup
```

A known-good model fails at a *later* stage (argument validation), proving its registry lookup
succeeded. The arc_fault model fails at the lookup itself. That contrast is the evidence.

**The invalid first attempt:** calling `MZ.get_model("CNN_AF_3L_1400")` with no further arguments
raises `TypeError: missing 2 required positional arguments`, **not** the registry `ValueError`. A
check that catches any exception therefore reports "absent" for every model, present or not. The
conclusion happened to be right, but the test could not have distinguished. Anyone re-verifying
this should pass the required arguments and assert on the **exception type**, not on pass/fail.

## Suggested resolution

Either is acceptable; the second is strictly better than the status quo even if the first is
planned:

1. **Add the `CNN_AF_3L_*` models to the modelzoo registry.**
2. **Stop advertising models that cannot be constructed.** Advertising four models and having none
   is worse than advertising none: it presents a menu whose every entry fails, and it fails late.
   A startup-time or selection-time consistency check between what a task advertises and what the
   registry holds would surface this class of defect for every task at once.

## Related findings from the same investigation

Two further upstream defects, distinct from this one, found while getting a PlatypusStudio project
to train:

- **NAS search cannot run.** `tinyml-tinyverse`'s
  `references/timeseries_classification/train.py:259` calls `models.get_model()` **unconditionally,
  even under NAS mode**, so mmcli's synthetic `NAS_<size>` placeholder hits a registry lookup that
  was never meant to receive it. Structurally the same failure as this one — a registry lookup that
  should not happen.
- **`mmcli info` prints an import error and exits 0**, so a module that fails to import is
  indistinguishable from a module with no models.

A separate review of the CUDA auto-defaults policy is in
`ANALYSIS-cuda-auto-defaults.md` alongside this file.

---

## Update 2026-08-16 — scope is wider, and the resolution may have changed

**Scope.** Re-checked by construction against the registry: this is not only `arc_fault`.

| task type | advertised | absent from registry |
|---|---|---|
| `arc_fault` | 4 | **4** (`CNN_AF_3L_*`) |
| `motor_fault` | 3 | **3** (`CNN_MF_*`) |
| `blower_imbalance` | 3 | **3** (`CNN_MF_*` — the same models as motor_fault) |
| `pir_detection` | 1 | 0 |
| `ecg_classification` | 1 | 0 |

Ten model entries across three task types. `blower_imbalance` was not previously identified and
fails collaterally, because it maps to motor_fault's models.

**Resolution.** Martin's understanding, stated 2026-08-16 as belief rather than established fact:
**the arc-fault and motor-fault models are TI proprietary and will not be published.**

If that holds, resolution 1 in this report ("add the models to the registry") is unavailable, and
resolution 2 ("stop advertising models that cannot be constructed") is the only one left. That is
worth confirming before acting, because it is the difference between a gap that will close and one
that never will — and the two call for opposite responses.

It also changes who can fix it. A permanent absence is not a bug awaiting an upstream fix, so
filtering these task types at the `mmcli info` boundary — so consumers never offer a task that
cannot succeed — stops being "working around an upstream defect" and becomes the correct behaviour.
Tracked as **REQ-UP-02** in `.planning/ROADMAP.md` Phase 15.
