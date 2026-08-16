---
phase: 13-hold-the-modelmaker-config-contract-and-close-two-knob-gaps
plan: 02
subsystem: cli
tags: [argparse, modelmaker, quantization, torch-compile, config-contract]

# Dependency graph
requires:
  - phase: 13-01
    provides: the omission-rule convention for CUDA-auto-default knobs (compile_model, native_amp) that this plan's new knob must not become the exception to
provides:
  - "--quant-train-only flag exposing modelmaker's training.run_quant_train_only knob, with the NO_QUANTIZATION precondition enforced in mmcli at parse time"
  - "Confirmed (via real --dry-run config generation) that --compile-model is module-agnostic and works for vision and audio, not just timeseries"
affects: [future-modelmaker-knob-exposure, radar-module-integration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Cross-flag preconditions are enforced in _validate_args() via the errors.append() list, printed together, exit(2) — not left for the modelmaker subprocess to raise"
    - "New boolean training knobs use action='store_true', default=None so _set() omits them from the config when unset (13-01's omission rule, carried forward)"

key-files:
  created: []
  modified:
    - mmcli/cli.py
    - mmcli/builder.py
    - tests/test_advanced_training_knobs.py

key-decisions:
  - "Precondition check treats --quantization omitted (None) the same as explicit NO_QUANTIZATION, because modelmaker's own default (TinyMLQuantizationVersion.NO_QUANTIZATION) is what training.quantization resolves to when the key is absent — omission is not a way to dodge the raise"
  - "--compile-model verification scoped to real --dry-run config generation per module (timeseries baseline + vision + audio), per the plan's explicit preference for a real check over a source-reading one"
  - "Radar deferred to Phase 12, not tested: mmcli has no radar task types/models registered yet (Phase 12 point-cloud classification support hasn't landed), so there is nothing to run a radar --dry-run against"

patterns-established:
  - "Boolean training knobs with a hard modelmaker-side precondition (raises ValueError on an incompatible flag combination) are validated in mmcli's _validate_args() before the subprocess starts, naming both flags in the error message"

requirements-completed: [REQ-QUANT-01, REQ-COMPILE-01]

# Metrics
duration: 45min
completed: 2026-08-15
---

# Phase 13 Plan 02: Quant-train-only knob and compile-model cross-module verification Summary

**Exposed `training.run_quant_train_only` via `--quant-train-only` with its `NO_QUANTIZATION` precondition enforced at mmcli parse time, and confirmed `--compile-model` is module-agnostic across timeseries, vision, and audio via real `--dry-run` config generation.**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-08-15 (session start)
- **Completed:** 2026-08-15
- **Tasks:** 2 completed
- **Files modified:** 3 (`mmcli/cli.py`, `mmcli/builder.py`, `tests/test_advanced_training_knobs.py`)

## Accomplishments

- `mmcli train`/`run` now expose `--quant-train-only`, which skips the float training pass and runs only the quantisation training pass — previously unreachable from the CLI despite existing in all four modelmaker training bases (timeseries, vision, audio, radar).
- The precondition modelmaker enforces deep in its training loop (`raise ValueError` when `quantization == NO_QUANTIZATION` and `run_quant_train_only` is set) is now caught in mmcli's `_validate_args()` before the subprocess starts, with a message naming both `--quant-train-only` and `--quantization`.
- Confirmed via real `--dry-run` config generation (not just source-reading) that `--compile-model` is module-agnostic: `training.compile_model` is emitted identically for `timeseries`/`motor_fault`/`CLS_1k_NPU`, `vision`/`image_classification`/`Lenet5`, and `audio`/`audio_classification`/`DSCNN_NPU`. Cross-checked against modelmaker's own `params.py` for vision and audio, both of which declare `compile_model=0` as an accepted key.

## Task Commits

Each task was committed atomically:

1. **Task 1: expose run_quant_train_only, with its precondition enforced here (REQ-QUANT-01)** - `33833a5` (feat)
2. **Task 2: verify --compile-model across modules (REQ-COMPILE-01)** - `cfda9ee` (test)

**Plan metadata:** (this commit, made after this file)

## Files Created/Modified

- `mmcli/cli.py` — Added `--quant-train-only` (`dest=run_quant_train_only`, `action="store_true"`, `default=None`) to the training options group; added the `NO_QUANTIZATION` precondition check to `_validate_args()`, treating an omitted `--quantization` the same as an explicit `NO_QUANTIZATION` since that is modelmaker's own default.
- `mmcli/builder.py` — Wired `training.run_quant_train_only` via `_set()` so the key is absent from the generated config when the flag is not passed (omission rule, matching 13-01's `compile_model`/`native_amp` convention).
- `tests/test_advanced_training_knobs.py` — Added `TestQuantTrainOnlyFlag` (help text present, key omitted by default, key set when passed, both rejection paths — explicit `NO_QUANTIZATION` and omitted `--quantization` — and a real `--dry-run` acceptance case) and `TestCompileModelCrossModule` (real `--dry-run` config generation confirming `training.compile_model` is emitted for timeseries, vision, and audio, with radar's deferral documented in the class docstring).

## Decisions Made

- Enforced the `run_quant_train_only`/`NO_QUANTIZATION` precondition by extending the existing `errors.append()` pattern in `_validate_args()` rather than introducing a new validation mechanism — this repo already has one convention for "reject at parse time with a clear message" and the plan explicitly asked to follow it rather than invent one.
- Treated `--quantization` omitted (`None`) as equivalent to explicit `NO_QUANTIZATION` for the precondition check. Read from modelmaker's `params.py` (`quantization=TinyMLQuantizationVersion.NO_QUANTIZATION` default in all four modules): when mmcli emits no `quantization` key, modelmaker's own default resolves to `NO_QUANTIZATION`, so the raise would still fire. Rejecting only the explicit string would have left a gap where `--quant-train-only` alone (no `--quantization` flag at all) sailed through mmcli's validation and hit the same traceback the fix was meant to prevent.
- For `--compile-model` cross-module verification, used real `--dry-run` config generation instead of relying solely on source-reading, per the plan's explicit instruction ("prefer a real check over a source-reading one"). Source-reading (`vision/params.py`, `audio/params.py`) was used only as corroborating evidence that both modules declare `compile_model` as an accepted key, not as the primary verification method.

## Deviations from Plan

None - plan executed as written. The plan itself anticipated the possibility that a module might reject the `compile_model` key ("if a module turns out NOT to accept the key, that is a finding") — this did not happen; all three tested modules accept and emit the key identically.

## Issues Encountered

- The full `pytest` suite takes over 11 minutes on this machine (confirmed twice, independently, by two concurrent agent runs). Per explicit coordinator instruction, this plan's tests were scoped to `tests/test_advanced_training_knobs.py` (22 tests, ~38s) plus `tests/test_build_config.py` (36 tests, <1s) as additional confidence on `builder.py` changes. The full suite was not run as part of this plan's execution; running it is deferred to the coordinator at merge time.

## Radar Deferral (REQ-COMPILE-01 scope note)

Radar was explicitly out of scope for the `--compile-model` cross-module verification. `compile_model` was wired into modelmaker's radar training base (`9a5facc`) alongside vision and audio, but mmcli itself has no radar module, task types, or models registered — that support depends on Phase 12 (radar point-cloud classification), which has not run. There is no `--dry-run` invocation that could exercise a radar path today. This is a stated deferral, not a silent omission: Phase 12, once implemented, should add a radar case to `TestCompileModelCrossModule` alongside the existing timeseries/vision/audio cases.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `--quant-train-only` and `--compile-model` (for vision/audio) are both usable end-to-end from mmcli today; no further wiring needed in this repo.
- When Phase 12 lands radar support in mmcli (task types, models, `TASK_TYPES_RADAR` etc.), add a radar case to `TestCompileModelCrossModule` in `tests/test_advanced_training_knobs.py` to close the deferred coverage gap noted above.

---
*Phase: 13-hold-the-modelmaker-config-contract-and-close-two-knob-gaps*
*Completed: 2026-08-15*

## Self-Check: PASSED

- FOUND: `.planning/phases/13-hold-the-modelmaker-config-contract-and-close-two-knob-gaps/13-02-SUMMARY.md`
- FOUND: `33833a5` (Task 1 commit)
- FOUND: `cfda9ee` (Task 2 commit)
