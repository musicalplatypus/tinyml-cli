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
