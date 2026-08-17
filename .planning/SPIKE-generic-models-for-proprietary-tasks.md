# Spike — can generic classifiers replace the proprietary fault-detection models?

**Run:** 2026-08-16. **Result: mechanism works, quality does not. Not a drop-in replacement.**
**Tracked against:** REQ-UP-02, `.planning/ROADMAP.md` Phase 15.

## Why this was run

Martin confirmed the `arc_fault` and `motor_fault` models are TI proprietary and **will not be
published**, and proposed two directions: offer the generic classification models for those task
types, or support NAS only. This spike tested the first.

Three task types are affected — 10 model entries absent from the 63-entry registry:
`arc_fault` (4), `motor_fault` (3), `blower_imbalance` (3, sharing motor_fault's `CNN_MF_*`).

## What works

**The substitution is structurally possible.** `mmcli train -t arc_fault -n CLS_1.2k_NPU` runs end
to end and exits 0, producing a quantised model and `model_aux.h`.

It works because `get_model_description(name)` is **not** task-filtered — it resolves `CLS_1.2k_NPU`
→ `CNN_TS_GEN_BASE_1P2K_NPU` regardless of the configured `task_type`. Only
`get_model_descriptions(task_type=…)`, which builds the *menu*, filters. So the registry lookup that
kills a normal arc_fault run is bypassed entirely.

The task's own feature extraction is preserved: the generated config keeps
`task_type: arc_fault` and selects `ArcFault_1024Input_256Feature_1Frame_Full_Bandwidth`.

## What does not work

The model never learns to discriminate, and **more training made it worse**:

| | 1 epoch | 10 epochs |
|---|---|---|
| Accuracy | 29.69% | 42.56% |
| F1 | 0.297 | 0.426 |
| **AUC ROC** | **0.974** | **0.542** |

Final confusion matrix at 10 epochs — every input predicted `non_arc`:

```
                       | Predicted as: arc | Predicted as: non_arc
 Ground Truth: arc     |         0         |         433
 Ground Truth: non_arc |         0         |         285
```

**The prediction going in was that accuracy would rise to meet the 0.974 AUC — a threshold problem
that more epochs would fix. That was wrong.** AUC collapsed to near-random instead, so separability
was lost rather than gained. This is not a training-length issue.

Two details that argue for a real architecture/feature mismatch rather than plain class imbalance:

- It collapses onto the **minority** training class (`non_arc`, 936 of 2654). Degenerate collapse
  normally lands on the majority.
- Accuracy rose while AUC fell — the model got better at guessing the base rate and worse at
  separating the classes.

**Not investigated** (deliberately — this was a 20-minute spike and the next step is a design
decision, not more runs): whether `input_features` is passed correctly for a 256-feature/1-frame
arc preset, whether the learning rate (0.002) suits it, and what `CNN_AF_3L_*` did differently.

## Reproduction

```bash
export MMCLI_PYTHON=/Users/martin/.venv-tinyml/bin/python
./dist/mmcli train -i "~/Documents/PlatypusStudio Projects/arc_1" \
  -m timeseries -t arc_fault -n CLS_1.2k_NPU -d F28P55 --epochs 10
```

Exits 0. Inspect `FloatTrain.BestEpoch` metrics and the final confusion matrix — **the exit code
says nothing**. That is the sixth time in this session that exit 0 accompanied a non-result;
it nearly got reported here as a success too.

## What this means for REQ-UP-02

Three options, none free:

1. **Generic models + tuning.** Mechanism proven, quality unsolved. Someone has to work out why a
   generic CNN cannot separate features a proprietary one presumably handled. Cheapest *if* the
   cause is trivial (an `input_features` or LR mismatch); open-ended if not.
2. **NAS only.** This spike is an argument *for* it: the failure is precisely "no known architecture
   fits these features", which is what architecture search exists to solve. Blocked by **REQ-UP-03**
   (NAS cannot run at all — `train.py:259` does an unconditional registry lookup), and slow per run.
3. **Stop advertising the three task types.** Honest and cheap; removes a menu whose every entry
   fails. Delivers no fault detection.

**This spike raised the value of fixing REQ-UP-03.** Before it, NAS looked like the expensive path
and generic models like the cheap one. Generic models are now known *not* to be cheap, and NAS is
the option whose premise matches the actual problem.
