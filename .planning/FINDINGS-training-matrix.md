# Findings — training matrix sweep

Companion to `TEST-PLAN-training-matrix.md`. Records defects found while running the
model × task sweep (`--epochs 1`, `--training-device cpu`).

**Sweep status at time of writing: in progress.** Final results will be appended when it
completes. Nothing here is inferred from the runner alone — every finding below was reproduced
by hand.

---

## F-1 — `mmcli train` requires `annotations/`, which modelmaker generates itself

**Severity: real, blocking for at least one task. Affects 12 of 75 combinations.**

`mmcli/cli.py:2001` rejects any project whose `dataset/` lacks an `annotations/` subdirectory:

```
Dataset missing 'annotations/' subdirectory: <project>/dataset
```

The check is **unconditional** — no task branching. It is mmcli's own precondition; nothing in
`tinyml-modelmaker` or `tinyml-tinyverse` requires it.

### Why it is wrong

`tinyml_modelmaker/ai_modules/common/datasets/__init__.py` treats `annotations/` as an **output**:

- When split-list files are absent it sets `need_to_create_splits = True`, then calls
  `remove_if_exists(annotations_dir)` followed by `os.makedirs(annotations_dir, exist_ok=True)` —
  it deletes and regenerates the directory. Demanding it up front inverts the contract.
- `annotation_dir='annotations'` is a **default output name** in
  `ai_modules/timeseries/params.py:74`, alongside `annotation_prefix='instances'`.
- For `TASK_CATEGORY_TS_ANOMALYDETECTION` the loader globs `<data_dir>/Normal` and
  `<data_dir>/Anomaly` and builds the splits from them — deliberately training on Normal only and
  appending all Anomaly files to the test list, because anomaly data must not be trained on.

The shipped `generic_timeseries_anomalydetection.zip` contains exactly `classes/Normal` and
`classes/Anomaly` and no `annotations/`. **That layout is correct.**

### Proof by experiment

Added an **empty** `annotations/` directory — contributing no data, purely to satisfy the gate.
Training then proceeded past the check and failed much further downstream (at feature
extraction). modelmaker generated the real annotations itself, in its own workspace:

```
<project>/run/dataset/annotations/instances_train_list.txt
<project>/run/dataset/annotations/instances_val_list.txt
<project>/run/dataset/annotations/instances_test_list.txt
  -> Normal/normal_0004.csv
  -> Normal/normal_0010.csv
```

So the directory was never required as input. mmcli blocked a supported layout.

### Correction to an earlier claim
This was first reported as "the anomaly-detection dataset ships without `annotations/`" — i.e. a
dataset-content defect. **That was wrong and is withdrawn.** The dataset is correct; the
validation is not.

### Consequence for the sweep
The 12 `generic_timeseries_anomalydetection` combinations recorded `CONFIG-INVALID` are artifacts
of this bug, not real outcomes. They were blocked before modelmaker ran. **Their true status is
unknown** and they must be re-run once the check is relaxed.

### Suggested fix
Drop the `annotations/` requirement, or downgrade it to a warning. `mmcli/cli.py:2001-2003`.
Note the sibling check for a data subdirectory (`classes/` etc.) at `:2004-2008` is legitimate
and should stay.

---

## F-2 — anomaly detection has no usable feature-extraction preset

**Severity: real, open. Blocks all 12 anomaly-detection combinations independently of F-1.**

With F-1 worked around, training still fails:

```
Exception: Not enough dimensions present. Extract more features
  tinyml_tinyverse/common/datasets/timeseries_dataset.py:797 _rearrange_dims
```

This is the same failure class the default FE preset produces elsewhere — but for classification
a catalog preset rescues it (see F-4). For anomaly detection, `mmcli info -m timeseries -t
generic_timeseries_anomalydetection` lists **zero** feature-extraction presets, so there is
nothing to fall back to from the catalog.

`generic_timeseries_forecasting` also reports 0 presets and should be checked for the same
problem once the sweep reaches it.

Open question: is the correct fix a catalog gap (presets exist but are not exposed for this
task), or does anomaly detection need different default FE parameters?

---

## F-3 — the catalog advertises models absent from the installed modelzoo

**Severity: real. Affects the 4 `arc_fault` combinations; scope beyond that not yet known.**

`mmcli info -m timeseries -t arc_fault` lists `ArcFault_model_200_t`. Training it fails:

```
ValueError: Model 'CNN_AF_3L_200' not found in registry or specified model_spec.
  tinyml_modelzoo/models/__init__.py:188
```

The catalog name resolves to an internal model id that does not exist in the installed
`tinyml_modelzoo` registry. The error lists ~68 models that DO exist (`CNN_TS_GEN_BASE_*`,
`AE_CNN_TS_GEN_BASE_*`, `REG_TS_GEN_BASE_*`, …), none of them `CNN_AF_3L_*`.

So `mmcli info` will offer a user a model that cannot be trained. Whether this is a modelzoo
packaging gap or a stale catalog entry is not yet determined.

---

## F-4 — `--dry-run` does not catch missing models

**Severity: minor, but it undermines the cheap pre-flight tier.**

All 4 `arc_fault` combinations **passed** `mmcli --dry-run train`, then failed at real training
with F-3's registry error. Dry-run validates argument and config resolution but does not check
that the resolved model exists in the modelzoo. A pre-flight that misses this class of failure
cannot be used to screen a matrix before committing hours to it.

---

## F-5 — the default FE preset is not viable for at least some classification models

**Severity: real, user-facing on the documented happy path.**

`generic_timeseries_classification` / `CLS_100_NPU` with the **default** preset fails in 9 s with
`Not enough dimensions present`; with `Generic_256Input_RAW_256Feature_1Frame` it succeeds in
364 s producing both ONNX artifacts. The sweep confirms the pattern — `CLS_1.2k_NPU` records
`PASS-WITH-PRESET` at 382.5 s.

This means a user following the documented `init` → `train` flow with default settings hits an
error, on a combination that works fine once a preset is named. The sweep records
`PASS` and `PASS-WITH-PRESET` as distinct outcomes precisely so the size of this gap is
measurable rather than hidden.

---

# FINAL RESULTS — sweep complete (75/75)

`--epochs 1 --training-device cpu`. Raw data: `training-matrix-results.{ndjson,csv}`.
Total successful-run compute: 2.71 h across 24 passing combinations.

| Outcome | Count | Tasks |
|---|---:|---|
| **PASS-WITH-PRESET** | **24** | generic_timeseries_classification (24/24) |
| CONFIG-INVALID | 16 | anomalydetection (12), image_classification (3), audio_classification (1) |
| FE-MISMATCH | 12 | generic_timeseries_forecasting (12/12) |
| ARTIFACT-MISSING | 11 | generic_timeseries_regression (11/11) |
| TRAIN-FAIL | 9 | arc_fault (4), motor_fault (3), ecg_classification (1), pir_detection (1) |
| BLOCKED | 3 | blower_imbalance |

**Outcomes cluster perfectly by task — not one task has a mixed result.** Every failure is a
task-level defect; no individual model failed on its own merits. Of 11 tasks, exactly **one**
trains end to end.

## Working combinations

**24 of 24 `generic_timeseries_classification` models pass**, every one with
`--feature-extraction Generic_256Input_RAW_256Feature_1Frame`, 340–678 s (median 359 s), both
ONNX artifacts present in all 24. **Zero combinations pass with the default preset** — see F-5.

Models: CLS_100/500/1k/1.2k/1.5k/1.9k/2k/2.8k/3.1k/3.9k/4k/4.2k/5k/6k/8k/13k/20k/40k/55k_NPU,
CLS_ResAdd_3k, CLS_ResCat_3k, ElectricalFault_model_40k_t, GearboxFault_model_1.2k_t,
GearboxFault_model_1.5k_t.

## F-6 — `mmcli train` exits 0 while producing nothing (regression)

**Severity: highest of the sweep.** All 11 `generic_timeseries_regression` combinations exit **0**
in ~7.5 s having written **no artifacts**, while logging:

```
File: <proj>/run/dataset/files/file_4.csv.
Error message: index 2 is out of bounds for axis 0 with size 2
```

Reproduced by hand twice, deterministically, with the task's own catalog preset
`Generic_8Input_ABS_8Feature_1Frame`. Any script, CI job or GUI that trusts the exit code will
record these as successful training runs. This is the "green but broken" class.

**The models themselves are fine.** The same `REGR_1k` trains successfully — exit 0, both
artifacts, ~6 min — when given `Generic_256Input_RAW_256Feature_1Frame`, a preset the regression
catalog does **not** list. So the defect is the catalog preset plus the swallowed error, not the
model.

*Correction:* this was first attributed to the default preset. Wrong — with the default preset
regression exits **1** (verified three times). The exit-0 path is specific to
`Generic_8Input_ABS_8Feature_1Frame`.

## F-1 is wider than first reported — 16 combinations, not 12

The unconditional `annotations/` precondition (`mmcli/cli.py:2001`) also blocks
**image_classification (3)** and **audio_classification (1)**, not just anomaly detection.
16 of 75 cells are "not tested" rather than "failed" — mmcli refused before modelmaker ran.

## F-3 is wider than first reported — 7 combinations

Catalog entries missing from the installed `tinyml_modelzoo` registry now include `motor_fault`
as well as `arc_fault`: `CNN_AF_3L_{200,300,700,1400}` and `CNN_MF_{1L,2L,3L}`.

## Two further single-combination failures

- `ecg_classification` / ECG_model_1 → `KeyError: 'ecg_classification'`
- `pir_detection` / PIR_model_1 → `AssertionError: min tensor([nan × 8]) should be less than max
  tensor([nan × 8])` — all-NaN feature tensor

## Honest limits of this sweep

- A PASS means **one epoch completed and both ONNX files were written**. It says nothing about
  whether the model learned anything. No accuracy threshold was applied.
- The 16 CONFIG-INVALID combinations have **unknown** true status.
- Only one FE preset was tried per task. A different preset may rescue forecasting or regression;
  the full model×task×preset space (456+ on classification alone) was not explored.
- The 12 forecasting and 11 regression failures share the same underlying
  `Not enough dimensions present` / feature-shape family as F-5, so a single upstream FE fix may
  move many cells at once.

---

# SWEEP v2 — F-1 worked around, complete 75/75

Same config as v1 (CPU, auto-quantization on, `--epochs 1`) plus one change: an **empty
`dataset/annotations/` directory is created after `init`** to lift the F-1 gate. That directory
contributes no data — modelmaker deletes and regenerates it — so it only stops mmcli refusing
before modelmaker runs. Raw data: `training-matrix-results-v2.{ndjson,csv}`. 8.4 h total compute.

Two knobs were measured and deliberately NOT changed:
- `--no-auto-quantization` saves 8 s of 355 s (2%). Not the bottleneck; disabling it would have
  made v2 non-comparable to v1.
- MPS: see F-8. Not used for the sweep.

| Outcome | v2 | v1 |
|---|---:|---:|
| PASS-WITH-PRESET | 24 | 24 |
| **PASS (default preset)** | **2** | 0 |
| FE-MISMATCH | 24 | 12 |
| ARTIFACT-MISSING | 11 | 11 |
| TRAIN-FAIL | 10 | 9 |
| BLOCKED | 3 | 3 |
| TIMEOUT | 1 | 0 |
| CONFIG-INVALID | **0** | 16 |

**26 of 75 combinations train successfully** (24 with an explicit preset, 2 with the default).

## v1 → v2 changes — all from lifting F-1

| Task | v1 | v2 |
|---|---|---|
| image_classification | CONFIG-INVALID ×3 | **PASS ×2**, TIMEOUT ×1 |
| generic_timeseries_anomalydetection | CONFIG-INVALID ×12 | FE-MISMATCH ×12 |
| audio_classification | CONFIG-INVALID ×1 | TRAIN-FAIL ×1 |

Everything else reproduced identically, which is the expected control: nothing but the
annotations gate changed.

**F-1 was concealing working configurations.** `Lenet5` and `MobileNetV2_58k_NPU` both train
successfully and were blocked by an incorrect precondition, not by any real defect. That raises
F-1's severity: it does not merely obscure results, it prevents supported work.

## F-7 — image-classification models are 25–85× slower than everything else

| Model | Outcome | Wall |
|---|---|---:|
| Lenet5 | PASS | 352 s |
| MobileNetV2_58k_NPU | PASS | **8,613 s (2.4 h)** |
| MobileNetV1_58k_NPU | TIMEOUT | >10,800 s (3 h, killed) |

Every non-image combination in the matrix finishes in ~350 s. The MobileNets are 25× and >30×
that. `MobileNetV1_58k_NPU` did not complete within a three-hour budget and its true status
remains **unknown** — it is recorded TIMEOUT, not FAIL.

Whether this is expected for MNIST-scale image training on CPU, or a pathology, is not
established here. It is the one result in the matrix that a longer budget could still change.

## F-5 refined — the default preset works exactly once

`Lenet5` (`fe_preset: null`) is the **only** combination of 75 that trains on the default
feature-extraction preset. All 24 classification successes required
`Generic_256Input_RAW_256Feature_1Frame`. The default-preset problem is therefore near-universal
rather than a classification quirk.

## Still true after v2

- Anomaly detection (12) fails at feature extraction with **zero catalog presets** to fall back
  on — F-2 confirmed by measurement, not inference. Fixing F-1 did not rescue it, as predicted.
- Regression (11) still exits **0** with no artifacts — F-6, the silent failure, unchanged.
- arc_fault (4) and motor_fault (3) still fail on missing modelzoo entries — F-3.
- blower_imbalance (3) still has no dataset.

## Runner defect, disclosed

v2 initially aborted at 73/75: `TypeError: can't concat str to bytes` in the sweep runner's own
timeout handler (`TimeoutExpired.stdout` returns bytes even under `text=True`). This was a defect
in the harness, not in mmcli. Fixed (bytes-safe) and the last two combinations were re-run with a
180-minute budget. No earlier result was affected.

## F-8 — MPS: 3.8× faster on large models, but unusable with auto-quantization

Benchmarked on `CLS_55k_NPU`, the largest generic timeseries model, 50 epochs, CPU vs MPS run
sequentially on an otherwise idle machine.

Steady state (epochs 1-49, after one-off dataset prep):

| | compute/batch | samples/s | time/epoch |
|---|---:|---:|---:|
| CPU | 0.0407 s | 1,572-1,650 | 2 s |
| MPS | **0.0108 s** | **5,909-6,916** | **<1 s** |

Float-training total: CPU 175 s vs MPS 76 s.

**There is a crossover with model size.** On `CLS_100_NPU` (~100 params) MPS was ~20× *slower*
(276-776 samples/s vs 5,543-6,407). On `CLS_55k_NPU` (~55k params) it is ~3.8× *faster*. Tiny
models never amortise GPU dispatch and host↔device copy overhead; larger ones do.

**But the MPS run FAILED** — exit 1, one artifact of two:

```
torchmodelopt/quantization/base/fx/auto_quantization.py:82 compute_hessian_eigenvalues
  -> power_iteration -> compute_hessian_vector_product
RuntimeError: max_pool2d with `return_indices=False` is not infinitely differentiable.
```

Hessian-based auto-quantization needs second-order derivatives; double-backward through
`max_pool2d` is unsupported on this path. CPU reached the same code and survived. **This is a
distinct defect from the known MPS float64 item.** The wall times (MPS 234 s vs CPU 565 s) are
NOT comparable — MPS "finished" sooner only because it crashed before quantization, evaluation
and the second export.

Consequence: the 3.8× training speedup is currently unreachable with the default configuration.
Untested follow-up: MPS with `--no-auto-quantization` should complete and would give the first
genuinely comparable wall-time figure.

## Epoch scaling (answers "would more epochs change this")

Dataset preparation is a **one-off**, not per-epoch:

- Epoch 0: 18-20 s, of which ~15 s is data loading
- Epochs 1-49: **2 s** (CPU), **<1 s** (MPS)

So single-epoch measurements are dominated by data loading and are a poor basis for judging the
training device — which is exactly why the 1-epoch comparison misled. Marginal epochs are cheap
and increasingly compute-bound, so the device ratio matters more as epochs increase, not less.

---

# FIXES APPLIED

## F-1 — FIXED (tinyml-cli `bec6f87`)

Removed the `dataset/annotations/` requirement from `mmcli/cli.py`. The sibling
data-subdirectory check is retained. Regression tests added in
`tests/test_datasets_cli.py::TestAnnotationsDirIsNotRequired` — they drive the real CLI in a
subprocess, and were **mutation-tested**: reintroducing the requirement turns 3 of the 4 red,
restoring turns them green.

Verified after the fix: anomaly detection, image classification and audio classification all
pass the gate with their shipped `classes/`-only layouts and **no workaround**. 113 tests across
the four surrounding suites still pass.

## F-6 — FIXED (tinyml-tensorlab `f484ddf`)

Root cause: `timeseries_dataset.py` called bare **`exit()`** in its `IndexError` handler.
`exit()` with no argument raises `SystemExit(None)`, which the interpreter reports as **status
0** — so a hard dataset failure was indistinguishable from success to anything checking the exit
code. Confirmed directly: `python3 -c "exit()"` → `$?=0`.

Now `exit(1)`. The `KeyboardInterrupt` handler had the identical defect (a cancelled run
reported success) and is now `exit(130)`. Both occurrences fixed.

Verified: the exact failing case (regression + its own catalog preset) went from **exit 0 / 0
artifacts** to **exit 1 / 0 artifacts**, deterministically over two runs. Controls confirm
success paths are unaffected — regression and classification with a working preset still exit 0
with both ONNX artifacts.

Note this does not make regression *train*; it makes its failure **visible**. The underlying FE
problem is F-5 below.

## F-5 — ROOT-CAUSED, not fixed (needs a decision)

**The task's default feature-extraction preset expects 3 input channels; the shipped example
dataset has 1.**

`tinyml_modelmaker/ai_modules/timeseries/constants.py:1369` sets the default for
`generic_timeseries_classification` to:

```
Generic_256Input_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1,  variables=3
```

The shipped `generic_timeseries_classification` dataset is **single-column**: sample CSVs contain
one value per row (verified: `head -1 saw10.csv | awk -F, '{print NF}'` → `1`). A 1-channel input
through a 3-channel preset yields a 2-D tensor, and `_rearrange_dims`
(`timeseries_dataset.py:791-797`) requires 3-D:

```python
if x_temp.ndim == 2:
    raise Exception("Not enough dimensions present. Extract more features")
```

Which is exactly the observed error. The preset that works,
`Generic_256Input_RAW_256Feature_1Frame`, carries no channel qualifier.

**A refuted hypothesis, recorded so it is not retried:** mmcli's `builder.py:82` hardcodes
`"feature_extraction_name": "default"` in its BASE_CONFIG, and `default` is not a real preset
name. That looked like the cause. Setting it to `None` — so modelmaker would fall back to its own
per-task default — **did not fix it**, because that per-task default is itself the 3-channel
preset. The hardcoded `"default"` string is still questionable and worth tidying, but it is not
the root cause.

**Why the mismatch likely exists:** `constants.py:1368` points at
`https://software-dl.ti.com/.../generic_timeseries_classification.zip` — the original upstream
dataset, which the defaults were written for. This project mirrors its own copy (phase 10, after
the upstream CDN moved). If the mirrored zip is not channel-identical to the one the defaults
assume, every default-preset run mismatches. **This should be checked before changing any
default.**

**The decision required** (not made here, as each option changes behaviour for all users of this
fork):
1. Change the per-task default FE preset to a 1-channel preset matching the shipped dataset.
2. Ship a 3-channel example dataset matching the existing default.
3. Have mmcli detect the input channel count and select a compatible preset.

Option 3 is the most robust and the only one that fixes it for user-supplied datasets too, but it
is the largest change.

## F-2 — unchanged

Anomaly detection still exposes **zero** catalog FE presets, so it has no fallback at all. Its
12 combinations remain blocked. Whether this is the same channel-mismatch family as F-5 or a
distinct catalog gap is not established.
