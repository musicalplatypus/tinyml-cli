---
phase: 10-dataset-distribution-and-binary-size
plan: 08
subsystem: ci
tags: [github-actions, pytest, pyinstaller, ci-gate, release-safety]

requires:
  - phase: 10-01
    provides: "scripts/pyinstaller_excludes.txt, scripts/binary_size_ceiling.txt, tests/test_build_config.py's exclude/ceiling guards"
  - phase: 10-02
    provides: "tests/test_datasets_download.py's MMCLI_DATASETS air-gap / fetch_dataset guards"
  - phase: 10-03
    provides: "the BUNDLED_DATASETS allowlist, the GitHub release mirror datasets-01_03_00, and the lowered/revised size ceiling this plan gates on"
  - phase: 10-06
    provides: "tests/test_datasets_cli.py's datasets list --format json contract and D-5 auto-fetch policy guards"
  - phase: 10-10
    provides: "TestPackageDataBundlesOnlyTheOneLocalDataset in tests/test_build_config.py, and the reasoning for why a wheel-size CI gate would pass vacuously"
provides:
  - "Both CI workflows (test-cli.yml, release.yml) collect all six phase-10 test files (161 tests total), not just the original two"
  - "tests/test_ci_workflows.py — the drift guard between the two workflows' pytest invocations"
  - "Per-artifact release-build gates: binary size ceiling (scripts/binary_size_ceiling.txt), non-empty dataset bundle probe, and a loose startup regression gate, all before Upload artifact"
  - "A mirror-healthcheck job verifying the datasets-01_03_00 release is reachable and correctly tagged, with no payload download, gating the build job"
affects: []

tech-stack:
  added: []
  patterns:
    - "regex-over-raw-text parsing of workflow YAML (no PyYAML dependency added) to assert the literal shell command a workflow runs, matching tests/test_build_config.py's existing convention"
    - "release-build gates read config from files (scripts/binary_size_ceiling.txt) at runtime rather than duplicating a literal in the workflow YAML"
    - "gh release view --json for reachability/metadata checks, never a payload download, when a CI checkout structurally cannot hold the full dataset set"

key-files:
  created:
    - tests/test_ci_workflows.py
  modified:
    - .github/workflows/test-cli.yml
    - .github/workflows/release.yml

key-decisions:
  - "Split Task 2's literal per-artifact gates (size ceiling, bundle probe) into their own commit, separate from the context-driven startup gate and mirror-healthcheck additions, so the plan's original scope and the context's additive scope are each independently reviewable and revertible."
  - "Startup regression gate uses bash's SECONDS builtin (not `date +%s.%N`/awk), matching the project's established wc -c-over-stat convention: one portable mechanism across all three runners, no platform-specific timing tool."
  - "Mirror healthcheck derives the expected dataset filename set from mmcli.datasets.DATASET_REGISTRY (installed via a lightweight `pip install -e .` — only PyYAML+tqdm, no training engine) rather than hardcoding a fourth copy of the nine filenames already living in tests/test_build_config.py."

requirements-completed: [REQ-SIZE-01, REQ-SIZE-02]

metrics:
  duration: "~55 min"
  completed: "2026-07-31"
---

# Phase 10 Plan 08: Wire phase-10 regression guards into CI + release-build gates Summary

**Every guard this phase built (test_build_config.py's 36 assertions, test_datasets_download.py's 50, test_datasets_cli.py's 24) was previously collected by neither CI workflow; both now run all six phase-10 test files (161 tests), and the release build additionally gates on the size ceiling, a non-empty dataset bundle, a loose startup regression bound, and — per a context update gathered after this plan was written — a payload-free mirror-release healthcheck.**

## Performance

- **Started:** 2026-07-31T15:44:00Z (approx, first file read)
- **Completed:** 2026-07-31T16:05:02Z
- **Duration:** ~55 min
- **Tasks:** 2/2 plan tasks complete, plus 2 context-driven additions (D-01, D-02)
- **Files modified:** 3 (2 workflows modified, 1 test file created)

## Accomplishments

- Both `.github/workflows/test-cli.yml` and `.github/workflows/release.yml` now run an
  identical pytest invocation naming all six phase-10 test files
  (`test_cli_integration.py`, `test_tier4_cli.py`, `test_build_config.py`,
  `test_datasets_download.py`, `test_datasets_cli.py`, `test_ci_workflows.py`), with the
  original `-k "not TestInitDatasetExtractReal"` deselection intact.
- `tests/test_ci_workflows.py` (new, 6 tests): asserts both workflows carry exactly one
  pytest invocation, that invocation names an identical set of test files in both
  workflows (the drift guard), that set is a superset of the required six, every named
  path exists on disk, and the deselection string survives verbatim.
- `release.yml`'s `build` job gained three gates before "Upload artifact": a per-artifact
  size ceiling read from `scripts/binary_size_ceiling.txt` (never a YAML literal), a
  runtime probe (`datasets path generic_audio_classification`) that the staged
  `--add-data` bundle actually landed in the binary, and a loose (25s) startup
  regression gate.
- A new `mirror-healthcheck` job verifies the `datasets-01_03_00` release is reachable
  and correctly tagged via `gh release view --json tagName,assets`, checking all nine
  mirrored dataset names are present with non-zero size — no payload downloaded. The
  `build` job now depends on both `test` and `mirror-healthcheck`.

## Task Commits

1. **Task 1: Wire the phase-10 test files into both workflows** — `a0e919c` (feat)
2. **Task 2: Gate the release build on the size ceiling and a non-empty dataset bundle**
   — `c12c481` (feat)
3. **Context-driven addition (D-01/D-02): mirror healthcheck job + startup regression
   gate** — `0c3f650` (feat)

No separate plan-metadata commit yet; this SUMMARY and STATE.md updates follow in the
final commit per the execution protocol.

## Context read first (postdates the plan)

`10-CONTEXT.md` was gathered after `10-08-PLAN.md` was written and is authoritative
where it touches this plan. Its net effect was additive, confirmed against the plan text
during execution:

- **D-01 (mirror healthcheck, new job):** implemented as `mirror-healthcheck` in
  `release.yml`, gating `build`. Downloads no payload — verified for real (see
  Verification below) both on the success path (9/9 assets present) and the failure path
  (a nonexistent tag returns non-zero with `release not found` in stderr).
- **D-02 (startup gate at 20-30s, not the literal 8s):** implemented at **25s**,
  justified as roughly 3x the ~6-6.6s measured on the maintainer's laptop (per
  `unplanned-work.md` §1 and `ROADMAP.md`'s REQ-SIZE-01 revision table) — wide enough to
  absorb hosted-runner variance (macOS runners worst) while still catching a catastrophic
  regression like a re-bundled training engine, which per `10-03-SUMMARY.md`'s own
  measurements (~31.8 MB unbundled binary before the PIL/cryptography exclusion) would
  push real elapsed time far past 25s, not merely brush against it.
- **D-03 / the "one warning" (no wheel-size CI gate):** not added, per `10-10-SUMMARY.md`'s
  own finding that CI's checkout (holding one of ten dataset zips) would make such a gate
  pass vacuously; the real guard for the wheel/sdist channel is
  `TestPackageDataBundlesOnlyTheOneLocalDataset` in `tests/test_build_config.py`, already
  wired into CI by Task 1.
- **No datasets seeded into CI** — confirmed neither workflow adds a step to fetch any of
  the nine mirrored zips; the mirror healthcheck reads only release *metadata*.
- **No test deselected to make this plan pass** — the `-k` expression is unchanged from
  before this plan (verified: `tests/test_ci_workflows.py::test_deselection_is_still_intact_in_both_workflows`
  and a full local run below).

## Files Created/Modified

- `tests/test_ci_workflows.py` — new. Regex-over-raw-text parser (no PyYAML dependency,
  matching `tests/test_build_config.py`'s own stated reason for avoiding `tomllib`/`tomli`
  elsewhere) asserting workflow drift cannot occur silently.
- `.github/workflows/test-cli.yml` — Tier 4 step extended from 2 to 6 named test files.
- `.github/workflows/release.yml` — "Run tests" step extended identically; `build` job
  gained three gate steps (size, bundle, startup) before "Upload artifact"; new
  `mirror-healthcheck` job added, and `build`'s `needs:` extended to `[test,
  mirror-healthcheck]`.

## Verification Performed

**Local pytest — per-file selected-test counts** (Task 1's third verify block, `-k "not
TestInitDatasetExtractReal"`):

| File | Selected |
|---|---|
| `test_build_config.py` | 36 |
| `test_datasets_download.py` | 50 |
| `test_datasets_cli.py` | 24 |
| `test_ci_workflows.py` | 6 |

**Full six-file invocation, exactly as both workflows now run it** (with
`MMCLI_PYTHON=$HOME/.venv-tinyml/bin/python`, required per project convention or
info/device tests fail for unrelated environment reasons):

```
161 passed, 20 deselected, 6 warnings in 214.80s
```

161 = 155 (pre-existing baseline across all five originally-scoped files) + 6 (new
`test_ci_workflows.py`). No test was deselected to reach this result.

**Drift-guard mutation test:** temporarily removed `tests/test_datasets_cli.py` from
`test-cli.yml`'s invocation — `tests/test_ci_workflows.py` failed exactly the two
expected assertions (`test_workflows_name_the_same_set_of_test_files`,
`test_named_set_is_a_superset_of_the_required_files`); restored, confirmed the restored
file is identical to the version staged for commit, suite green again.

**Release-build gates — rehearsed locally against the real macOS binary** (built by
10-03/10-01's `bash build_macos.sh`, present in the working tree at 25,256,016 bytes,
matching the ceiling-revision measurement in `unplanned-work.md` §1):

```
size=25256016 ceiling=27262976   -> SIZE GATE: PASS
datasets path generic_audio_classification -> resolves inside the PyInstaller extraction dir -> BUNDLE GATE: PASS
startup elapsed=5-6s ceiling=25s -> STARTUP GATE: PASS
```

Inspection confirms the ceiling appears nowhere in `release.yml` as a literal (`grep -E
"15728640|27262976"` over the file returns nothing) — raising
`scripts/binary_size_ceiling.txt` remains the only way to loosen the gate.

**Mirror healthcheck — rehearsed for real against the live GitHub release** (the exact
Python extracted from the workflow's `run:` block via a YAML round-trip, executed
locally):

```
OK: mirror release 'datasets-01_03_00' has all 9 expected assets, all non-zero size (no payload downloaded).
  arc_fault_classification.zip: 13290076 bytes
  ecg_classification.zip: 4651662 bytes
  fan_blade_fault.zip: 56595859 bytes
  generic_timeseries_anomalydetection.zip: 4242845 bytes
  generic_timeseries_classification.zip: 2579940 bytes
  generic_timeseries_forecasting.zip: 71053 bytes
  generic_timeseries_regression.zip: 906660 bytes
  mnist_image_classification.zip: 46993516 bytes
  pir_detection.zip: 1579936 bytes
```

Failure path also rehearsed: `gh release view datasets-99_99_99 --repo
musicalplatypus/tinyml-cli --json tagName,assets` returns exit code 1 with `release not
found` — confirming the healthcheck's `sys.exit(1)` branch triggers on a missing/mis-tagged
release, not just theoretically.

**Not executed: the GitHub Actions workflows themselves.** This agent cannot run GitHub
Actions. `yaml.safe_load()` confirmed both workflow files are syntactically valid YAML
and that `build`'s `needs:` correctly lists `[test, mirror-healthcheck]`; every gate's
*logic* was rehearsed locally against real artifacts (the macOS binary, the live GitHub
release) rather than merely asserted by inspection, but the actual CI run — matrix
scheduling, Windows Git Bash behavior, `GH_TOKEN` propagation via `secrets.GITHUB_TOKEN`
— has not been observed. State this plainly rather than implying a green Actions run.

## Decisions Made

See `key-decisions` in the frontmatter. In addition: the two context-driven gates were
committed separately from the plan's literal Task 2 gates (three commits total instead
of two) so that the plan's original scope and the context's additive scope are each
independently reviewable — a deviation from "one commit per plan task," made deliberately
rather than folding unplanned scope into an existing task's commit.

## Deviations from Plan

### Context-driven additions (not deviations from 10-08-PLAN.md's written tasks, but scope added by 10-CONTEXT.md, gathered after the plan)

**1. [10-CONTEXT.md D-01] Added `mirror-healthcheck` job to `release.yml`**
- **Reason:** CI's checkout holds only 1 of 10 dataset zips; an artifact-level dataset
  assertion would pass vacuously. The healthcheck instead verifies the release's own
  asset metadata is present and correctly tagged, catching the realistic failure (a
  deleted or mis-tagged release) without downloading ~131 MB per run.
- **Files modified:** `.github/workflows/release.yml`
- **Commit:** `0c3f650`

**2. [10-CONTEXT.md D-02] Added a startup regression gate to the `build` job**
- **Reason:** REQ-SIZE-01's 8s bound was measured on a developer laptop; hosted runners
  are slower and load-variable. A 25s bound (justified above) catches catastrophic
  regressions without failing on ordinary runner variance.
- **Files modified:** `.github/workflows/release.yml`
- **Commit:** `0c3f650`

No Rule 1/2/3 auto-fixes were needed beyond these two explicitly-scoped context
additions — the plan's own two tasks were implemented as written, verified, and
committed without requiring a bug fix or missing-functionality patch.

## Known Stubs

None. No placeholder values or unwired data paths were introduced.

## Threat Flags

None beyond what the plan's own `<threat_model>` already anticipated. The
`mirror-healthcheck` job introduces a new outbound call to the GitHub API
(`gh release view`), scoped to read-only release metadata on a public repository, using
the standard `secrets.GITHUB_TOKEN` already available to every Actions run in this
repository — no new credential, no new write scope, no payload transfer.

## Self-Check

- FOUND: `tests/test_ci_workflows.py`
- FOUND: commit `a0e919c` (`git log --oneline --all | grep a0e919c`)
- FOUND: commit `c12c481`
- FOUND: commit `0c3f650`
- `pytest tests/test_ci_workflows.py -q`: 6 passed
- `pytest tests/test_build_config.py tests/test_ci_workflows.py -q`: 42 passed
- Full six-file invocation with `MMCLI_PYTHON` set: 161 passed, 20 deselected
- `yaml.safe_load()` on both workflow files: valid; `build.needs == ["test",
  "mirror-healthcheck"]`
- Mirror healthcheck script run for real against the live mirror: 9/9 assets present,
  non-zero size, zero payload bytes downloaded
- Release gates run for real against the actual macOS `dist/mmcli` (25,256,016 bytes):
  all three gates pass
