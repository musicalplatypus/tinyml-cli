---
phase: 14-mmcli-cold-start-stop-paying-seconds-before-doing-any-work
plan: 01
subsystem: cli
tags: [argparse, performance, cold-start, pyinstaller, macos]

# Dependency graph
requires: []
provides:
  - _detect_training_device() memoised (process-lifetime cache) and lazy (0-1 calls per invocation instead of 3)
  - --training-device argparse default deferred to point-of-use, resolved identically to the pre-fix eager default
  - tests/test_cold_start.py call-count regression test
  - Phase 14 ROADMAP.md before/after measurements (REQ-COLD-04)
affects: [14-02, 14-03]  # any follow-up plan touching REQ-COLD-03 (onefile unpacking) or mmcli/cli.py argparse setup

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Lazy argparse defaults: sentinel None + resolve-at-point-of-use instead of eager default= computation, to keep argparse tree construction (which always runs, for every subcommand) free of expensive side effects"
    - "Process-lifetime memoisation via a module-level `None`-sentinel cache variable for machine-constant probes"

key-files:
  created:
    - tests/test_cold_start.py
  modified:
    - mmcli/cli.py
    - .planning/ROADMAP.md

key-decisions:
  - "--training-device help text no longer shows a concrete 'Default (detected): mps' value; it now says detection happens at training time. Showing the real value would require running the ~0.9s probe just to build --help text for train/run, which happens on every invocation regardless of subcommand (argparse builds all subparsers eagerly)."
  - "Top-level `mmcli --help` is the one path allowed to still show the concrete detected device (its description text includes 'Detected training backend: X'); detection is triggered lazily by scanning argv for -h/--help before the subcommand token, not called for --version or any subcommand --help."
  - "Resolution of the None sentinel happens once in main(), immediately after parse_args(), guarded by hasattr(args, 'training_device') so it only fires for train/run (the only subparsers that declare the flag) and never for --version/--help (which exit during parsing, before this line is reached)."

requirements-completed: [REQ-COLD-01, REQ-COLD-02, REQ-COLD-04]

# Metrics
duration: ~25min
completed: 2026-08-15
---

# Phase 14 Plan 01: mmcli cold-start device-detection fix Summary

**Memoised and made lazy the 3x-per-invocation `_detect_training_device()` probe, cutting `mmcli --version` from 2.75s to 0.04s and the onefile binary's `--version` from 6.58s to 3.93s, with a call-count regression test and real measurements recorded in ROADMAP.md.**

## Performance

- **Duration:** ~25 min
- **Tasks:** 3/3 completed
- **Files modified:** 2 (mmcli/cli.py, .planning/ROADMAP.md); 1 created (tests/test_cold_start.py)

## Accomplishments

- `_detect_training_device()` is now memoised for the process lifetime (`_TRAINING_DEVICE_CACHE`), so any repeated call is free.
- `--training-device`'s argparse default changed from an eagerly-computed concrete value to `None` (deferred), resolved in `main()` right before `build_config()` consumes it — only for `train`/`run`, never for `--version`/`--help`/other subcommands.
- The top-level `--help` description's "Detected training backend: X" line only triggers detection when top-level help is actually about to be rendered (argv scanned for `-h`/`--help` appearing before the subcommand token).
- Added `tests/test_cold_start.py`: 3 tests asserting call counts (not wall time) via a counting wrapper around `_detect_training_device()`, invoking `cli.main()` with patched `sys.argv` and catching the `SystemExit` that `--version`/`--help` raise.
- Recorded real before/after measurements in `.planning/ROADMAP.md`'s Phase 14 section, including a fresh onefile-binary rebuild (not timing a stale artifact).

## Task Commits

1. **Task 1: detect at most once, and only when the answer is needed** - `54ec48d` (fix)
2. **Task 2: pin the win so it cannot silently regress** - `fdef56c` (test)
3. **Task 3: record before/after from measurement (REQ-COLD-04)** - `4c21888` (docs)

_No separate plan-metadata commit was made per this plan's execution instructions (STATE.md updates and the standard final-commit step were explicitly out of scope for this run; the Task 3 commit already carries the ROADMAP.md update)._

## Files Created/Modified

- `mmcli/cli.py` — `_detect_training_device()` memoised; `_add_training_args()` no longer calls it eagerly (`--training-device` defaults to `None`); `main()` resolves the sentinel once, only for train/run, and only computes the top-level `--help` description's detected-device text when top-level help is actually being shown.
- `tests/test_cold_start.py` — new file, 3 tests pinning call-count behavior for `--version`, `train --help`, and `--help`.
- `.planning/ROADMAP.md` — Phase 14 section only: before/after measurement table, a new "After (REQ-COLD-04, 14-01)" narrative paragraph, and per-requirement "Done"/"Still open" status notes for REQ-COLD-01/02/03/04.

## Decisions Made

- **Help text tradeoff (Task 1's explicit ask):** chose to stop showing a concrete "Default (detected): mps" value in `train --help`/`run --help`, replacing it with "Default: auto-detected for this machine when training starts (not computed for --help/--version)." Rationale: those subparsers are built on every single invocation (argparse builds the full tree regardless of which subcommand runs), so keeping the concrete value there would mean paying the ~0.9s probe cost universally again — the exact problem this plan fixes. Top-level `mmcli --help` (and `mmcli help`, which already called detection on-demand before this plan) are the two places that still show a real detected value, because only those two are on a genuinely help-requesting path where the extra ~0.9s is expected and bounded.
- **Where to resolve the sentinel:** placed immediately after `parser.parse_args()` in `main()`, gated on `hasattr(args, "training_device")`, rather than inside `mmcli/builder.py` at the point of consumption. This avoids introducing a new import from `builder.py` back into `cli.py` (which already imports `from mmcli.builder import build_config, ...`, so the reverse import would be circular) and keeps the "when do we pay for the probe" policy in one place (`cli.py`) rather than splitting it across two modules.
- **Preserved existing dead-code quirk without touching it (not a decision I made, a pre-existing behavior I deliberately left alone):** the Darwin branch of `_detect_training_device()` already returned `"mps"` on every code path (whether or not `"Metal"`/`"Apple"` appeared in `system_profiler`'s output, and on exception) — the conditional check has no effect on the return value in the original code. This is unrelated to cold-start and out of this task's scope ("do not change what is detected"), so it was preserved exactly, just restructured to fit the single-return/cache-then-return pattern. Documented inline with a comment pointing back to this observation so a future reader doesn't mistake it for new dead code.

## Deviations from Plan

None — plan executed exactly as written. The plan's own three tasks fully specified the memoisation, the laziness split across three call sites, the help-text tradeoff decision, the call-count test, and the before/after measurement; no additional bugs, missing functionality, or blocking issues were found outside that scope.

## Non-Vacuousness Verification (Task 2)

Verified `tests/test_cold_start.py` is not vacuous by:
1. Backing up the fixed `mmcli/cli.py` to the scratchpad.
2. Extracting the pre-fix version via `git show HEAD:mmcli/cli.py` (run *before* Task 1's commit existed, so `HEAD` at that point was the plan-only commit `602b8c6`, i.e. genuinely pre-fix).
3. Overwriting `mmcli/cli.py` with that pre-fix content and running `pytest tests/test_cold_start.py -v`.
4. All 3 tests **failed**, each reporting `actual_calls == 3` — matching the plan objective's measured figure exactly (`_detect_training_device() called 3x`).
5. Restored the fixed `mmcli/cli.py` from the scratchpad backup; re-ran the 3 tests, all passed.

No git history was altered to do this (plain file copy/overwrite of the working tree, using `git show` read-only to fetch historical content) — at no point was `git checkout`, `git reset`, or `git stash` used.

## Verification Performed

- `--version` timing: 2.75s → 0.04s (source), 6.58s → 3.93s (onefile binary, freshly rebuilt via `build_macos.sh` with the venv on `PATH`).
- `mmcli info -m timeseries -t generic_timeseries_classification --format json`: onefile 8–9s → 6.81s (3-run median); source-run figure newly recorded at 2.78s (not separately measured before this plan).
- `--dry-run train` device selection identical before/after for three cases: no `--training-device` flag (→ `training_device: mps`, `num_gpus: 1`), explicit `--training-device cpu` (→ `training_device: cpu`, `num_gpus: 0`), explicit `--training-device auto` (→ omitted from config, letting `tinyml_modelmaker` decide, unchanged from before).
- Scoped suite (`pytest tests/ -q -k "not Real"`): 694 passed, 7 skipped, 21 deselected, 0 failed (446s).
- `tests/test_cold_start.py` alone: 3 passed.
- Full suite (`pytest tests/`, all ~722+3 tests including the "Real" subprocess-spawning ones) was **not** run in this session per the environment instructions (~14 min, orchestrator's job at merge) — the scoped run above (694 passed, 7 skipped, 21 deselected, 0 failed) is the substitute used during execution. The orchestrator's full-suite run at merge is the authoritative check against the 722-passed/0-failures baseline plus these 3 new tests.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- REQ-COLD-01, REQ-COLD-02, REQ-COLD-04 are complete (see ROADMAP.md Phase 14 section for the requirement-by-requirement status notes added in this plan).
- REQ-COLD-03 (evaluate moving off PyInstaller `--onefile`) remains open and untouched by this plan — the onefile binary's remaining ~3.8s of `--version` time is unpacking overhead, now cleanly isolated from detection overhead by this fix, ready for whoever scopes that follow-up plan.
- `mmcli/cli.py`'s lazy-default pattern (`default=None` + resolve-at-point-of-use, gated by `hasattr`) is now an established pattern in this file; any future argparse default that requires non-trivial computation should follow the same shape rather than reintroducing an eager `default=<expensive_call()>`.

---
*Phase: 14-mmcli-cold-start-stop-paying-seconds-before-doing-any-work*
*Completed: 2026-08-15*

## Self-Check: PASSED

- FOUND: `mmcli/cli.py`
- FOUND: `tests/test_cold_start.py`
- FOUND: `.planning/ROADMAP.md`
- FOUND: `.planning/phases/14-mmcli-cold-start-stop-paying-seconds-before-doing-any-work/14-01-SUMMARY.md`
- FOUND commit `54ec48d` (Task 1)
- FOUND commit `fdef56c` (Task 2)
- FOUND commit `4c21888` (Task 3)
