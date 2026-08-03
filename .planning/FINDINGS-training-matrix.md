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
