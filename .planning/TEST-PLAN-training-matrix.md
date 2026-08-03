# Test Plan — successful training for every model × task

**Goal:** establish, for each available model on each available task type, whether `mmcli train`
completes successfully and produces usable artifacts.

**Status:** plan only. Nothing in the matrix has been run except the three calibration runs
recorded under "Measured cost basis", which are real.

All figures below were measured on this machine on 2026-08-03, not estimated.

---

## 1. The matrix

Enumerated from `mmcli info` (run from source with `MMCLI_PYTHON=$HOME/.venv-tinyml/bin/python`;
the shipped `dist/mmcli` binary **cannot** run `info` — it excludes `tinyml_modelmaker` by
design, and errors with `Cannot import tinyml_modelmaker`).

| Task | Module | Models | FE presets | Dataset |
|---|---|---:|---:|---|
| arc_fault | timeseries | 4 | 8 | arc_fault_classification |
| blower_imbalance | timeseries | 3 | 9 | **NONE** |
| ecg_classification | timeseries | 1 | 1 | ecg_classification |
| generic_timeseries_anomalydetection | timeseries | 12 | 0 | generic_timeseries_anomalydetection |
| generic_timeseries_classification | timeseries | 24 | 19 | generic_timeseries_classification, ecg_classification |
| generic_timeseries_forecasting | timeseries | 12 | 0 | generic_timeseries_forecasting |
| generic_timeseries_regression | timeseries | 11 | 2 | generic_timeseries_regression |
| motor_fault | timeseries | 3 | 9 | fan_blade_fault |
| pir_detection | timeseries | 1 | 2 | pir_detection |
| audio_classification | audio | 1 | 1 | generic_audio_classification |
| image_classification | vision | 3 | 4 | mnist_image_classification |

**75 model×task combinations. 72 trainable. 3 blocked** — `blower_imbalance` has no dataset in
`DATASET_REGISTRY`, so its 3 models cannot be trained from a stock install. Those 3 must be
reported **blocked**, never **failed**; conflating the two would manufacture a bug report against
working code.

---

## 2. Measured cost basis

Three real runs, `--epochs 1 --training-device cpu`:

| Run | Outcome | Wall time |
|---|---|---|
| `generic_timeseries_classification` / `CLS_100_NPU`, **default** FE preset | **FAILED** — `Exception: Not enough dimensions present. Extract more features` | 9 s |
| same, FE preset `Generic_256Input_RAW_256Feature_1Frame` | **SUCCEEDED** — `model.onnx` in `run/training_base/` and `run/training_quantization/` | **364 s** |
| `mmcli info` enumeration (all 11 tasks) | n/a | ~3 min total |

**Extrapolation, stated as an extrapolation:** 72 × ~364 s ≈ **7.3 hours** at Tier 1, and that is
a *floor*. The calibration run used one of the smallest models on a 2.5 MB dataset. `fan_blade_fault`
(54 MB) and `mnist_image_classification` (44.8 MB) will be materially slower, as will the larger
models (`CLS_55k_NPU`, `CLS_40k_NPU`). Budget a full Tier 1 sweep as an overnight job, not an
interactive one.

---

## 3. The third axis nobody planned for: feature-extraction preset

The calibration runs prove the **default FE preset is not universally viable** — the same model
and task fails with the default and succeeds with an explicit preset. So a model×task matrix
alone cannot answer "does training succeed"; the answer depends on the FE preset.

The full model×task×preset space is far larger than 75 (24 models × 19 presets on
`generic_timeseries_classification` alone = 456). **Do not attempt it.** Instead:

- **Pin one known-good FE preset per task** and record it in the results as a first-class field.
  A pass means "passes with preset P", not "passes".
- Treat "default preset fails for this task" as a **finding in its own right** (Tier 0 below),
  because that is what a new user hits first.
- Note that 2 tasks report **0** presets (`generic_timeseries_anomalydetection`,
  `generic_timeseries_forecasting`). Determine before the sweep whether `--feature-extraction`
  is inapplicable there or the catalog is incomplete — that distinction changes whether a
  failure is a bug or a misuse.

---

## 4. Tiers

Run in order. Do not start a tier until the previous one is clean or its failures are triaged.

### Tier 0 — config resolution (all 75, minutes)
```
mmcli --dry-run train -m <module> -t <task> -d <device> -n <model> -i <project>
```
Validates catalog resolution, device compatibility and config construction without training.
Catches the cheap class of failure — bad model name, unsupported device, missing preset — before
spending 6 minutes to learn the same thing. Also run each task once with **no** `--feature-extraction`
to record which tasks fail on their default preset.

### Tier 1 — one-epoch smoke (72, ~7.3 h floor)
```
mmcli train -m <module> -t <task> -d <device> -n <model> -i <project> \
  --epochs 1 --training-device cpu --feature-extraction <pinned preset> --run-name <combo-id>
```
Proves the model builds, data loads, forward/backward runs, quantization runs, and artifacts are
written. This is "successful training" in the minimal, useful sense. **This tier is the deliverable.**

### Tier 2 — convergence spot-check (representative subset, hours)
Full default epochs for one small + one large model per task. Tier 1 proves it *runs*; Tier 2 is
the only tier that says anything about whether it *learns*. Keep the subset small and explicit.

---

## 5. Per-combination procedure

1. **Fresh project dir per combination.** `mmcli` writes into `<project>/run/` and overwrites it;
   there are no isolated run directories. Reusing one project across combinations will silently
   destroy the previous run's artifacts. Use `mktemp -d` per combo, or `--run-name`, or both.
2. **Choose a device the model actually supports.** Device support is per model, not per task
   (`mmcli info -m <mod> -t <task>` lists supported devices per model). Hardcoding one device
   across the sweep will produce failures that are device-mismatch, not training defects.
3. Materialise the dataset: `mmcli init --dataset <name> -t <task> -p <project>`.
4. Run the tier command; capture stdout+stderr to `<results>/<combo-id>.log`; record wall time.
5. Evaluate against the pass criteria below.

---

## 6. Pass criteria

A combination **PASSES** only if all hold:

- exit code 0
- `<project>/run/training_base/model.onnx` exists and is non-empty
- `<project>/run/training_quantization/model.onnx` exists and is non-empty
- `status.json` present in the run tree

Note the artifact paths are the **live run tree**, not the `.platypus/runs/` archive. Per phase
10's `deferred-items.md` (D-A), a completed run archives metadata only — `metrics: {}`,
`artifacts: {}`, no `run.log`. **Asserting on the archive would produce false failures.** This is
a known open defect owned by Phase 11 (REQ-RUN-01), not something this sweep should re-litigate.

### Outcome taxonomy — keep these distinct
| Outcome | Meaning |
|---|---|
| PASS | all criteria met |
| BLOCKED | no dataset exists for the task (the 3 `blower_imbalance` combos) |
| CONFIG-INVALID | Tier 0 dry-run failed — never reached training |
| FE-MISMATCH | failed with `Not enough dimensions present` or similar preset-shape error |
| TRAIN-FAIL | training itself errored |
| ARTIFACT-MISSING | exit 0 but a required artifact absent — the most interesting failure |

`ARTIFACT-MISSING` deserves special attention: it is the "green but broken" class this project
has already been bitten by twice (a vacuous zip-slip test, a size-ceiling guard that could not
fail). An exit code is not evidence that training produced anything.

---

## 7. Environment prerequisites

- `export MMCLI_PYTHON=$HOME/.venv-tinyml/bin/python` — required; several paths misbehave without it.
- Run `mmcli` **from source** (`python -m mmcli`), not `dist/mmcli`. The shipped binary excludes
  `tinyml_modelmaker` and cannot train or run `info`.
- All 10 datasets local before starting (9 cached + 1 bundled) so the sweep is not
  network-dependent. Verify with `mmcli datasets list`.
- **Use `--training-device cpu`.** An MPS float64 crash from a `tinyml_tinyverse` update is an
  open item in the root `.planning/todos/`. Running the sweep on MPS would conflate an upstream
  regression with per-model failures. If MPS coverage is wanted, run it as a **separate axis**
  after the CPU sweep is green, so the two are distinguishable.
- Disk: each combination leaves a project tree with two ONNX models plus intermediates. Budget
  for 72 of them, or delete on PASS and keep only failures.

---

## 8. Results schema

One row per combination, machine-readable (`results.csv` or NDJSON):

```
combo_id, module, task, model, device, fe_preset, tier,
outcome, exit_code, wall_seconds,
artifact_training_base, artifact_training_quantization, status_json,
log_path, error_signature
```

`error_signature` = first exception line, normalised — this is what makes 72 failures
triageable by grouping rather than by reading 72 logs.

---

## 9. Deliverable

1. `results.csv` — every combination with a definitive outcome.
2. A summary table: PASS / BLOCKED / CONFIG-INVALID / FE-MISMATCH / TRAIN-FAIL / ARTIFACT-MISSING.
3. Failure groups by `error_signature`, most frequent first.
4. An explicit statement of **which FE preset each task was pinned to** — without it, "72/72 pass"
   is not a reproducible claim.
5. Anything not run, named as not run. A combination that was skipped is not a combination that
   passed.

---

## 10. Open questions to settle before the sweep

1. Which FE preset should each task be pinned to? The calibration run shows the default is not
   always viable, so this must be decided deliberately, not inherited.
2. Are `generic_timeseries_anomalydetection` and `generic_timeseries_forecasting` genuinely
   preset-free (0 listed), or is that a catalog gap?
3. Is `blower_imbalance` expected to ship without a dataset, or is that itself the bug? It is the
   only task of 11 with no data.
4. Should Tier 1 use `--epochs 1` or a slightly larger value? One epoch proves the machinery runs
   but will not surface instabilities that appear at epoch 2+.
