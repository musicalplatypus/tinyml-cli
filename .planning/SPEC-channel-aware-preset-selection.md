# SPEC (draft, for discussion) — channel-aware feature-extraction preset selection

**Status: proposal. Nothing implemented.** Written to be argued with; the open questions at the
end are real, not rhetorical.

Addresses F-5 (and part of F-6's visible symptom) from `FINDINGS-training-matrix.md`.

---

## 1. The problem, restated precisely

`mmcli train` without `--feature-extraction` fails for nearly every timeseries task because the
feature-extraction preset in play expects a different number of input channels than the dataset
has.

Two measured instances:

| Task | Dataset | Preset used | Preset expects | Result |
|---|---|---|---|---|
| generic_timeseries_classification | 1 column | `Generic_256Input_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1` | `variables=3` | `Not enough dimensions present` |
| generic_timeseries_regression | 2 columns (`x,y`) | `Generic_8Input_ABS_8Feature_1Frame` | `variables=11` | `index 2 is out of bounds for axis 0 with size 2` |

The second error names the mismatch outright: `size 2` is the dataset's column count.

Downstream, `_rearrange_dims` (`timeseries_dataset.py:791-797`) requires a 3-D tensor; a
channel-count mismatch yields 2-D and it raises.

## 2. Why this is implementable (the key enabling fact)

Every preset in `FEATURE_EXTRACTION_PRESET_DESCRIPTIONS`
(`tinyml_modelmaker/ai_modules/timeseries/constants.py:1214`) carries **machine-readable**
metadata. Verified across all 40 presets — there are no gaps:

- `data_processing_feature_extraction.variables` → the channel count it expects
- `common.task_type` → the task(s) it applies to (a str or a list)

Measured distribution:

| | count |
|---|---:|
| total presets | 40 |
| with `variables` | **40 (100%)** |
| with `task_type` | **40 (100%)** |
| `variables=1` | 28 |
| `variables=3` | 11 |
| `variables=11` | 1 |

Per task:

| Task | presets | channel counts available |
|---|---:|---|
| generic_timeseries_classification | 19 | 17× 1-ch, 2× 3-ch |
| generic_timeseries_regression | 2 | 1× 1-ch, 1× 11-ch |
| **generic_timeseries_anomalydetection** | **0** | — |
| **generic_timeseries_forecasting** | **0** | — |

So selection can be done on facts, not by parsing preset names.

## 3. Scope — what this does and does not fix

**This is the most important part of the spec. Option 3 cannot fix half the failures.**

| Task | Combos | Would option 3 help? |
|---|---:|---|
| generic_timeseries_classification | 24 | **Yes** — 17 one-channel presets available; currently pass only with an explicit preset |
| generic_timeseries_regression | 11 | **Probably** — would select `Custom_Default` (1-ch) instead of the 11-ch preset. Needs verification |
| generic_timeseries_anomalydetection | 12 | **No** — zero presets exist to select from (F-2) |
| generic_timeseries_forecasting | 12 | **No** — zero presets exist (F-2) |
| arc_fault, motor_fault | 7 | No — models absent from the modelzoo (F-3) |
| ecg, pir, audio | 3 | No — distinct causes |
| blower_imbalance | 3 | No — no dataset |
| image_classification | 3 | Not applicable — vision path, already works |

**Net effect:** ~35 combinations touched. Of those, 24 already pass when the user knows the magic
preset name. The genuine new passes are regression's 11, *if* `Custom_Default` works there.

**So the honest value of option 3 is UX, not pass-count**: it makes the documented `init` → `train`
happy path work without the user having to discover a preset name by trial and error. That is
worth doing — but it should not be sold as "fixes the matrix". Anomaly detection and forecasting
(24 of 49 failures) need presets to exist upstream first, which is a separate piece of work.

## 4. Proposed behaviour

### 4.1 Trigger
Only when the user did **not** pass `--feature-extraction`. An explicit preset always wins,
unchanged, with no validation-by-refusal (see Q4).

### 4.2 Detect the dataset's channel count
Inspect a sample of data files under `<project>/dataset/<data_subdir>/`:

- **CSV**: number of columns in the first data row. Detect and skip a header row (regression's
  dataset has `x,y`; classification's has none).
- **`.npy`**: last dimension of the loaded array shape.
- **`.pkl`**: shape of the contained frame/array.

Sample **N files across different class directories** (proposal: N=3), not just the first, and
require agreement. Disagreement is a real condition — a dataset with inconsistent widths — and
should be reported, not averaged away.

### 4.3 Select a preset
From `FEATURE_EXTRACTION_PRESET_DESCRIPTIONS`, take presets where:
1. `task_type` includes the requested task, **and**
2. `variables` equals the detected channel count

Then pick deterministically (see Q2 for the tie-break, which is unresolved).

### 4.4 Report the choice
Log at INFO, e.g.:

```
Detected 1 input channel in <dataset>; selected feature-extraction preset
'Custom_Default' (1 of 17 compatible). Override with --feature-extraction.
```

Silent magic is worse than no magic. The user must be able to see what was chosen and why, and
reproduce it explicitly.

### 4.5 When nothing matches — fail, do not guess
If no preset matches the detected channel count, **exit non-zero** with the detected count, the
task, and the available presets with their channel counts. Falling back to an arbitrary preset
would reproduce exactly the current failure mode one layer deeper, and this project has already
been bitten twice by silent-wrong-thing behaviour (F-6; phase 10's vacuous guards).

For anomaly detection and forecasting — zero presets — the message should say plainly that the
task has no feature-extraction presets available, rather than implying the user chose badly.

## 5. Where it lives

`mmcli/builder.py`, at the point that currently hardcodes
`"feature_extraction_name": "default"` (line 82) — which is itself wrong and should go regardless
(see F-5 notes; setting it to `None` alone does not fix anything, but the literal string is not a
real preset name).

Selection needs `tinyml_modelmaker` importable. mmcli already shells out to `MMCLI_PYTHON` for
exactly this reason, so the lookup must run **in that subprocess**, not in mmcli's own
interpreter — the packaged binary deliberately excludes modelmaker.

## 6. Testing requirements

Per this project's blocking anti-patterns:

- Drive the **real CLI** in a subprocess, not `builder.py` internals.
- **Mutation-test** every guard: break the selector (e.g. force it to pick a 3-channel preset for
  1-channel data) and confirm the test goes red.
- Cover: 1-channel classification, 2-column regression, no-match (must exit non-zero),
  explicit `--feature-extraction` override (selector must not run), and a task with zero presets.
- End-to-end: `init` → `train` with **no** preset flag must produce both ONNX artifacts for at
  least one classification and one regression model.

---

## Open questions for discussion

**Q1 — What is a "channel" for regression?** The regression dataset is `x,y`: 2 columns, but
almost certainly 1 input variable and 1 target. Naive column counting gives 2, and there is no
2-channel preset — so selection would fail on a dataset that should work. Detection probably has
to be task-aware (subtract the target column for regression/forecasting), which means encoding
knowledge of the data contract per task category. Is that acceptable, or is there an existing
declaration of target columns I have not found?

**Q2 — Which preset when several match?** Classification has 17 one-channel candidates. Options:
(a) a hardcoded per-task preference list; (b) first in declaration order; (c) prefer
`Custom_Default` when compatible; (d) prefer the one the task's own `constants.py` default names
if it happens to match the channel count, else fall back. (a) is explicit but becomes a
maintenance burden; (b) is arbitrary and could change silently when upstream reorders. I lean (c)
then (a), but this is a judgement call about defaults and it is yours.

**Q3 — Should this also set `variables` in the config?** The task defaults set `variables=3`
alongside the 3-channel preset. If we select a 1-channel preset but leave `variables=3` from the
task default, we may just move the mismatch. Likely we must set both — needs verification.

**Q4 — Should an explicit `--feature-extraction` be validated against the detected channels?**
Warning on a mismatch would have saved this entire investigation. But it risks false positives on
datasets whose channel count we detect wrongly, and it could block a user who knows better.
Proposal: warn, never block.

**Q5 — Does upstream want this instead?** The mismatch arguably belongs in modelmaker, which owns
both the presets and the task defaults. Putting it in mmcli fixes mmcli users only — PlatypusStudio
included, since it drives mmcli — but leaves anyone using modelmaker directly broken. Fixing it in
`tinyml-tensorlab` (your fork) would cover both. Where do you want it?

**Q6 — Is the mirrored dataset even the right one?** `constants.py:1368` points at the original
upstream `generic_timeseries_classification.zip`; this project mirrors its own copy since phase 10.
If the mirrored zip is not channel-identical to the one the defaults were written for, the truer
fix might be to the dataset, and option 3 would be papering over a mirroring error. Worth
checking before building anything.
