---
phase: 10-dataset-distribution-and-binary-size
plan: 01
subsystem: build/packaging
tags: [pyinstaller, build-scripts, binary-size, regression-guard]
requires: []
provides:
  - scripts/pyinstaller_excludes.txt (single-source PyInstaller exclude list)
  - scripts/binary_size_ceiling.txt (single-source CI size ceiling)
  - tests/test_build_config.py (source-level regression guard)
affects:
  - 10-03 (lowers the ceiling to 15728640 and extends test_build_config.py with --add-data assertions)
  - 10-08 (CI wiring reads the same ceiling file; wires this test into both workflows)
tech-stack:
  added: []
  patterns:
    - "shared config file read by multiple build scripts (bash `while read`, PowerShell `Get-Content` + array splat) instead of duplicated inline lists"
    - "source-level regression test asserting build-script text/behavior rather than building the artifact"
key-files:
  created:
    - scripts/pyinstaller_excludes.txt
    - scripts/binary_size_ceiling.txt
    - tests/test_build_config.py
  modified:
    - build_macos.sh
    - build_linux.sh
    - build_windows.ps1
decisions:
  - "Task 1 required no new code change: the macOS EXCLUDES array from 143dd7e was already correct and matched the plan's list verbatim, so it was reconciled (read from the shared file in Task 2) rather than duplicated."
  - "Corrected the stale '~10 MB' comment in all three scripts (never true, never enforced) — required by the phase-level checker finding B-1 resolution table even though Task 1's action text didn't call it out explicitly."
  - "pwsh is not installed on this machine; build_windows.ps1 was verified by source-level assertion only (tests/test_build_config.py), not parsed by the PowerShell language parser. Not installed per the plan's own guidance to report absence rather than imply the script was checked."
metrics:
  duration: "~50 min"
  completed: "2026-07-22"
---

# Phase 10 Plan 01: PyInstaller exclusions across all three builds + single-source size ceiling Summary

Extracted the PyInstaller training-engine exclusion list to one shared file consumed by all
three release build scripts (macOS, Linux, Windows), and added a source-level pytest guard
that fails if any script silently drops the exclusions, hardcodes a diverging copy, or the
CI size ceiling is loosened.

## What was built

**Task 1 — macOS exclusions (already committed, reconciled not duplicated).**
`build_macos.sh` already carried the correct `EXCLUDES=()` array from commit `143dd7e`
(pre-dating this session), matching the plan's thirteen-module list exactly. No new commit
was needed for Task 1 in isolation; its array was superseded in Task 2 by a read of the new
shared file, per the plan's explicit instruction to reconcile rather than append a second
copy.

**Task 2 — shared exclude list, all three scripts.**
- `scripts/pyinstaller_excludes.txt`: thirteen module names, one per line, no comments —
  `torch`, `torchvision`, `torchaudio`, `tinyml_modelmaker`, `tinyml_tinyverse`,
  `tinyml_torchmodelopt`, `tinyml_modelzoo`, `tvm`, `matplotlib`, `scipy`, `sklearn`, `onnx`,
  `onnxruntime`.
- `build_macos.sh`: replaced the inline array with a `while IFS= read -r m` loop over the
  shared file.
- `build_linux.sh`: had no `--exclude-module` at all before this plan (only
  `--hidden-import mmcli{,.builder,.cli}`); added the same read-and-expand pattern. Left
  `--paths`, `--collect-submodules` and `--add-data` untouched — Linux never had any of
  those either, and that gap is 10-03's problem, not this plan's.
- `build_windows.ps1`: `Get-Content` piped through `ForEach-Object { '--exclude-module', $_ }`
  into `$ExcludeArgs`, then splatted into the pyinstaller call with `@ExcludeArgs`. Per the
  plan's explicit warning, the array is **not** interpolated as `"$ExcludeArgs"` into the
  backtick-continued command line — that stringifies an array into one space-joined,
  malformed flag. Path resolved via nested `Join-Path` calls, not string concatenation.
- Corrected the stale "the binary is lightweight (~10 MB) because tinyml_modelmaker is NOT
  bundled" comment in all three scripts (line 3-4), which was never true and never enforced —
  this was flagged in the phase-level PLAN.md's checker-findings table (B-1) as part of
  10-01's scope even though Task 1/2's action text didn't spell it out.

**Task 3 — size ceiling + regression guard.**
- `scripts/binary_size_ceiling.txt`: single line, `152043520` (145 MiB), the interim ceiling
  while the dataset payload is still bundled. 10-03 lowers this to `15728640` (15 MiB,
  REQ-SIZE-01) once datasets are unbundled.
- `tests/test_build_config.py` (15 tests): asserts the shared list has exactly the thirteen
  expected entries and never lists `numpy`/`pandas`; parametrises over all three build
  scripts to assert each either reads the shared file (checked by presence of both the
  `pyinstaller_excludes` reference and an actual `--exclude-module`-generating pattern) or
  — documented fallback — hardcodes an equal literal copy; asserts each script references
  `pyinstaller_excludes` by name so a silent revert to a hardcoded array is caught even
  before it goes stale; asserts the ceiling parses as a positive int and is one of the two
  sanctioned values (`152043520` or `15728640`).

## Verification performed

- `pytest tests/test_build_config.py -q` — **15 passed**.
- Deliberately removed `torch` from `scripts/pyinstaller_excludes.txt` → suite failed
  (`test_excludes_file_has_thirteen_expected_entries`) → restored → suite green again.
- Deliberately reverted `build_windows.ps1`'s `Get-Content` read to a bare
  `$ExcludeArgs = @()` (simulating a silent revert) → suite failed (2 tests, including the
  "references shared file" guard) → restored → confirmed the restored file is byte-identical
  to the pre-break version.
- Deliberately set `scripts/binary_size_ceiling.txt` to `45000000` (an unsanctioned value) →
  `test_ceiling_is_a_sanctioned_value` failed → restored.
- Ran `bash build_macos.sh` for real (macOS arm64, using `~/.venv-tinyml/bin/python` on
  `PATH` and as `MMCLI_PYTHON`): build succeeded, reported **"mmcli modules bundled: 17"**,
  produced `dist/mmcli` at **145,388,496 bytes (138.6 MB)** — under the 152,043,520-byte
  ceiling and matching `10-RESEARCH.md`'s measured 138.7 MB.
- Ran the produced binary: `--version` → `mmcli 1.1.2`; `init --list` → lists all 10
  datasets; `info -m timeseries` → lists task types (subprocess path, unaffected by
  exclusions); `analyze -i <extracted generic_audio_classification>` → produced a full
  class-distribution report, confirming `numpy`/`pandas` are still bundled and functional;
  `diagnose` → ran all 8 checks, correctly reporting `tinyml_modelmaker` as not importable
  in-process (the expected, pre-existing behavior of the guarded probe now that it's
  excluded — this is the exact no-op the `except ImportError` branch was already written
  to handle, not a regression).
- `bash -n build_linux.sh && bash -n build_macos.sh` → **BASH SYNTAX OK**.
- `pwsh` is **not installed** on this machine (`command -v pwsh` failed). Per the plan's own
  instruction ("if it is absent, say so in the summary rather than implying the script was
  checked"), `build_windows.ps1` was **not** parsed by the PowerShell language parser in this
  session — it is verified only at the source-assertion level by
  `tests/test_build_config.py`. The real Windows gate remains the CI build job in
  `.github/workflows/release.yml`, extended by 10-08.
- `bash/dist/build` artifacts from the real build were left in place only long enough to
  measure and functionally test; both `dist/` and `build/` are gitignored and `git status`
  confirms no untracked build output remains.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - missing correctness/documentation accuracy] Corrected the stale "~10 MB"
comment in all three build scripts**
- **Found during:** Task 2
- **Issue:** All three scripts claimed "the binary is lightweight (~10 MB) because
  tinyml_modelmaker is NOT bundled" — a claim that was never true (measured 138.7 MB even
  after exclusions, because the dataset payload is still bundled) and, before this plan,
  never enforced at all. The phase-level `PLAN.md` checker-findings table (B-1) explicitly
  attributes this correction to 10-01 Tasks 2-3, even though the individual task action text
  in `10-01-PLAN.md` didn't call it out as a discrete step.
- **Fix:** Replaced the comment in all three scripts with an accurate description: the
  training engine is excluded via the shared list, mmcli reaches it through the
  `MMCLI_PYTHON` subprocess, and the bundled datasets remain the largest component until a
  later phase unbundles them.
- **Files modified:** `build_macos.sh`, `build_linux.sh`, `build_windows.ps1`
- **Commit:** `4704b57`

No other deviations. All three tasks' `<done>` and `<acceptance_criteria>` conditions were
met as specified; Task 1's macOS array pre-existed and was reconciled per the plan's own
explicit contingency for that case.

## Known Stubs

None. No placeholder values, empty stubs, or unwired data paths were introduced.

## Threat Flags

None. This plan's changes are confined to build scripts and a source-level test; no new
network endpoints, auth paths, or trust-boundary-crossing file access was introduced. The
`<threat_model>` T-10-01-01 through T-10-01-04 mitigations are all implemented as specified
(parametrised test over all three scripts; single-source ceiling; single shared list;
Windows asserted at source level in the absence of a local Windows host).

## Self-Check: PASSED

- FOUND: scripts/pyinstaller_excludes.txt
- FOUND: scripts/binary_size_ceiling.txt
- FOUND: tests/test_build_config.py
- FOUND: build_macos.sh (modified)
- FOUND: build_linux.sh (modified)
- FOUND: build_windows.ps1 (modified)
- FOUND commit 4704b57
- FOUND commit 41b8ec1
- pytest tests/test_build_config.py -q: 15 passed
- bash build_macos.sh: real build succeeded, 145,388,496 bytes, 17 mmcli modules bundled, under the 152,043,520-byte ceiling
