---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: Milestone — Core Functionality & Security
status: ready_to_plan
last_updated: "2026-07-23T16:59:14.518Z"
progress:
  total_phases: 6
  completed_phases: 5
  total_plans: 26
  completed_plans: 26
  percent: 83
---

# Project State Summary

## Current Status

This is the mmcli tinyML CLI project. Phases 1 and 2 are complete. **Phase 3 (Testing and Documentation) is now complete** - all test infrastructure is in place with 360 passing tests, including new fuzz testing and attack surface tests from Phase 4.

## Recent Work Completed

### Session: 2026-07-07 (UAT Verification - Phase 2)

**Phase 2 Test Execution:**
| Plan | Target File | Type | Status |
|------|-------------|------|--------|
| 02-01 | tests/test_info.py | test | ✅ COMPLETE |
| 02-02 | tests/test_analyze.py | test | ✅ COMPLETE |
| 02-03 | tests/test_recommend.py | test | ✅ COMPLETE |
| 02-04 | tests/test_deploy.py | test | ✅ COMPLETE |
| 02-05 | mmcli/cli.py (doc) | doc | ✅ COMPLETE |
| 02-06 | docs/CONFIG_FILE_EXAMPLES.md (doc) | doc | ✅ COMPLETE |

**Fixes applied:**

1. `requirements.txt` - Commented out editable tinyml-modelmaker install
2. Created `.venv` with Python dependencies (numpy, pandas, pytest)
3. Added conftest.py psutil fallback

### Session: 2026-07-06 (Resume)

**Plan 02-02: Test Analyze Command - COMPLETE**

- Created comprehensive test suite for mmcli/analyze.py
- 40 tests covering all public functions and edge cases
- All tests pass with pytest

**Plan 02-03: Test Recommend Command - COMPLETE**

- Verified existing test file has comprehensive coverage (22 tests)
- All tests pass with pytest

**Plan 02-04: Test Deploy Command - COMPLETE**

- Verified existing test file has comprehensive coverage (34 tests)
- All tests pass with pytest

**Plan 02-05: Document Environment Variables in CLI Help - COMPLETE**

- Added MMCLI_DATASETS to module-level docstring
- Added MMCLI_MODELZOO_PATH to main help text

**Plan 02-06: Config File Examples Documentation - COMPLETE**

- Created docs/CONFIG_FILE_EXAMPLES.md (215 lines)
- 5 working examples covering train, compile, run subcommands

### Session: 2026-07-07 (Phase 3 Verification - UAT)

**Plan 03-03: Unit Test Coverage for builder.py and datasets.py - COMPLETE**

- Created `tests/test_config_builder.py` with 13 tests
- Created `tests/test_dataset_manager.py` with 9 tests  
- All 22 tests pass successfully

**Plan 03-04: Workflow Integration Tests - COMPLETE**

- Created `tests/test_workflow.py` with 4 integration tests
- Fixed YAML stub module (`yaml.py`) to support `yaml.dump()` without stream
- All 4 tests pass

**Plan 03-05: Cross-Platform Compatibility Tests - COMPLETE**

- Created `tests/test_cross_platform.py` with platform-specific path handling tests
- Added conftest.py fixtures for mocking Windows/macOS/Linux detection
- Tests pass on macOS (simulated Windows mode)

**Plan 03-06: API Documentation Generation - COMPLETE**

- Sphinx configuration already exists in `docs/conf.py`
- Fixed `mmcli/deploy.py` docstring indentation for reStructuredText compliance
- Documentation builds cleanly with no warnings

**Fixes applied during Phase 3 verification:**

1. `conftest.py`: Added fallback for missing `psutil` module
2. `yaml.py`: Created stub with `safe_load()` and `dump()` compatible API
3. `mmcli/deploy.py`: Fixed docstring bullet list format (8-space indent → hyphens)

### Session: 2026-07-06 (Phase 3 Planning)

**Plan 03-01 to 03-06: Phase 3 Testing and Documentation - PLANNED**

- Research completed on integration test failures
- 6 plans created for testing improvements and documentation

### Session: 2026-07-06 (Phase 4 Planning)

**Plan 04-01 to 04-05: Phase 4 Security Enhancements - PLANNED**

- Research completed on security posture
- 5 plans created for fuzz testing, attack surface, and documentation

### Session: 2026-07-06 (Phase 5 Planning)

**Plan 05-01 to 05-06: Phase 5 New Features & UX - PLANNED**

- Research completed on feature gaps
- 6 plans created for progress, export formats, comparison, batch processing, diagnostics, and interactive shell

## Current State of Implementation

The mmcli project has the following commands implemented:

- `info` - Show supported devices, models, and presets (with security hardening)
- `analyze` - Analyze project dataset contents (with security hardening)
- `recommend` - Recommend models and FE presets (with security hardening)
- `deploy` - Handle deployment operations (with security hardening)

Test infrastructure is being established with centralized fixtures in conftest.py.

## Progress - Phase 2

### Completed Plans

| Plan | Target File | Status |
|------|-------------|--------|
| 02-01 | tests/test_info.py | ✅ COMPLETE |
| 02-02 | tests/test_analyze.py | ✅ COMPLETE |
| 02-03 | tests/test_recommend.py | ✅ COMPLETE |
| 02-04 | tests/test_deploy.py | ✅ COMPLETE |
| 02-05 | mmcli/cli.py (doc) | ✅ COMPLETE |
| 02-06 | docs/CONFIG_FILE_EXAMPLES.md (doc) | ✅ COMPLETE |

**Phase 2 Test Coverage Summary:**

- Total tests: 118 (22 + 40 + 22 + 34)
- All passing: ✅
- Coverage: info, analyze, recommend, deploy commands

## Progress - Phase 3

### Plans Completed in This Session

| Plan | Target File | Type | Status |
|------|-------------|------|--------|
| 03-01 | tests/test_cli_integration.py (fix) | fix | ✅ COMPLETE |
| 03-02 | mmcli/cli.py/_is_safe_path (fix) | fix | ✅ COMPLETE |
| 03-03 | tests/test_config_builder.py (tdd) | tdd | ✅ COMPLETE |
| 03-04 | tests/test_workflow.py (intg) | intg | ✅ COMPLETE |
| 03-05 | tests/test_cross_platform.py (tdd) | tdd | ✅ COMPLETE |
| 03-06 | docs/ (doc) | doc | ✅ COMPLETE |

## Progress - Phase 4

### Plans Completed in This Session

| Plan | Target File | Type | Priority | Status |
|------|-------------|------|----------|--------|
| 04-01 | tests/test_fuzz_sanitization.py (tdd) | tdd | Critical | ✅ COMPLETE |
| 04-02 | tests/test_attack_surface.py (sec) | sec | High | ✅ COMPLETE |
| 04-03 | SECURITY.md (doc) | doc | Medium | ✅ COMPLETE |
| 04-03 | README.md (doc) | doc | Medium | ✅ COMPLETE |
| 04-03 | docs/SECURITY_MODEL.md (doc) | doc | Medium | ✅ COMPLETE |
| 04-04 | mmcli/cli.py/_is_safe_path (fix) | fix | Medium | ✅ COMPLETE |
| 04-04 | mmcli/cli.py/_sanitize_input (fix) | fix | Medium | ✅ COMPLETE |
| 04-05 | scripts/scan-vulnerabilities.sh (sec) | sec | Low | ✅ COMPLETE |

**Phase 4 Test Coverage Summary:**

- Total tests: 28 security-related tests
- All passing: ✅
- Coverage: fuzz testing, input sanitization, path traversal, command injection

## Progress - Phase 5

### Plans Ready for Execution

| Plan | Target File | Type | Priority | Status |
|------|-------------|------|----------|--------|
| 05-01 | mmcli/progress.py (feat) | feat | Critical | ✅ READY |
| 05-02 | mmcli/output.py (feat) | feat | High | ✅ READY |
| 05-03 | mmcli/compare.py (feat) | feat | Medium | ✅ READY |
| 05-04 | mmcli/batch.py (feat) | feat | Medium | ✅ READY |
| 05-05 | mmcli/diagnose.py (feat) | feat | Medium | ✅ READY |
| 05-06 | mmcli/interactive.py (feat) | feat | Low | ✅ READY |

## Session Continuity

Last session: 2026-07-23T16:59:14.511Z
Resumed from: Phase 2 UAT verification passed (118 tests), Phase 3 ready for execution

**Phase 2 Status:** COMPLETE ✅  
**Phase 3 Status:** COMPLETE ✅ (all test infrastructure, 360 tests passing)  
**Phase 4 Status:** COMPLETE ✅ (fuzz testing, security docs, path validation, vuln scanning)  
**Phase 5 Status:** READY FOR EXECUTION (6 plans prepared)

## Next Steps

All major phases complete. Ready for Phase 5 features:

- **Phase 5**: New Features & UX (plans 05‑01 to 05‑06)

## Accumulated Context

### Session: 2026-07-22 (Phase 10 Plan 01 execution)

**Note:** the narrative sections above this point (Phase 2-5 status, "Next Steps") are
stale — they predate Phases 6-9 and the start of Phase 10, which is the project's actual
current position per `.planning/ROADMAP.md` and `.planning/phases/`. Not rewritten here;
out of scope for this plan's execution. `.planning/ROADMAP.md`'s per-phase plan tables are
the authoritative progress source (`gsd-sdk query roadmap.update-plan-progress` keeps them
current); this file's frontmatter `progress:` block and the automated `state.advance-plan` /
`state.add-decision` handlers do not parse against this file's legacy narrative format
(no `Current Plan`/`Total Plans in Phase` fields, no `## Decisions` section), so those
handlers returned parse errors during this session rather than updating in place.

**Plan 10-01 — PyInstaller exclusions across all three builds + single-source size ceiling —
COMPLETE.**

- `scripts/pyinstaller_excludes.txt`: single-source thirteen-module PyInstaller exclude list
  (torch, torchvision, torchaudio, tinyml_modelmaker, tinyml_tinyverse,
  tinyml_torchmodelopt, tinyml_modelzoo, tvm, matplotlib, scipy, sklearn, onnx, onnxruntime),
  read by all three build scripts (`build_macos.sh`, `build_linux.sh`, `build_windows.ps1`)
  instead of each carrying its own copy.

- `scripts/binary_size_ceiling.txt`: single-source CI size ceiling, `152043520` (145 MiB
  interim, while the dataset payload is still bundled — 10-03 lowers it to `15728640`).

- `tests/test_build_config.py`: 15 source-level regression tests; verified by deliberately
  breaking two named failure modes (removing a module from the shared list; breaking
  `build_windows.ps1`'s `Get-Content` read) and confirming the suite failed, then restored.

- Real `bash build_macos.sh` run (macOS arm64): 145,388,496 bytes (138.6 MB), under the
  152,043,520-byte ceiling, 17 mmcli modules bundled. `--version`, `init --list`,
  `info -m timeseries`, `analyze`, and `diagnose` all verified working against the built
  binary.

- `pwsh` is not installed on this machine; `build_windows.ps1` was verified at the
  source-assertion level only, not by the PowerShell parser.

- Commits: `4704b57` (shared exclude list + all three scripts), `41b8ec1` (ceiling +
  regression test).

- See `.planning/phases/10-dataset-distribution-and-binary-size/10-01-SUMMARY.md` for full
  detail.

**Plan 10-02 — Registry digests/versioning, version-scoped cache, verified `fetch_dataset` —
COMPLETE.**

- `mmcli/datasets.py`: the nine TI-fetchable `DATASET_REGISTRY` entries gained `ti_name`,
  `ti_version` (per-entry override), `sha256`, and `bytes`, matching the measured provenance
  table in `10-RESEARCH.md` verbatim. `generic_audio_classification` kept `sha256`/`bytes`
  but no `ti_name` (no TI upstream). Import-time `_validate_registry()` makes a `ti_name`
  entry without a valid digest a hard import failure (REQ-DATA-02).

- Added `dataset_url(name)` (version-pathed TI URL, `KeyError` on unknown name),
  `_cache_dir(version)` (XDG_CACHE_HOME-aware, version-keyed), `_resolve_dataset_zip(name)`
  (MMCLI_DATASETS → bundled → version cache → None, wrapping the existing `_datasets_dir()`
  rather than replacing it), and `fetch_dataset(name, *, force=False)` (stdlib
  `urllib.request` only; atomic download-verify-`os.replace()`; refuses when
  `MMCLI_DATASETS` is set or the URL is not HTTPS; guards truncated/oversized bodies,
  cross-host redirects, HTTP 404, and connect/read timeouts).

- Added `stderr_is_tty()`, the single TTY predicate 10-06 will reuse for its `init --dataset`
  auto-fetch policy (D-5) instead of writing a second `isatty()` check.

- `extract_dataset()` now resolves its zip path through `_resolve_dataset_zip()`; existing
  callers of `_datasets_dir()` are unaffected.

- `tests/test_datasets_download.py` (new, 668 lines, 44 tests): registry invariants, URL
  derivation, cache/resolution order, a zip-slip confirmation test (T-10-02-06), and the full
  `fetch_dataset`/`_download_to_cache` failure-mode matrix against a local `http.server`.
  Verified passing with all non-loopback network access blocked at the socket layer (no test
  contacts `software-dl.ti.com`).

- One deviation (Rule 3): `_download_to_cache` initially only caught `urllib.error.URLError`
  around `opener.open()`; a socket-level timeout while reading the response status line
  surfaces as a raw `OSError`/`TimeoutError` instead, so an `except OSError` clause was added
  before committing Task 3.

- Commits: `aa33ba4` (Task 1 — registry/URL), `e075dc3` (Task 2 — cache/resolution),
  `95b8f90` (Task 3 — fetch_dataset).

- See `.planning/phases/10-dataset-distribution-and-binary-size/10-02-SUMMARY.md` for full
  detail.

### Session: 2026-07-23 (Phase 10 Plan 03 execution — BLOCKED)

**Plan 10-03 — GET-and-hash gate over every fetchable dataset, then unbundle — Task 1 only,
BLOCKED before Task 2/3.**

- `scripts/verify_dataset_digests.py` (new, committed `ffe1a67`): re-runnable GET-and-hash
  gate, drives `fetch_dataset(name, force=True)` against a throwaway cache for every
  fetchable registry entry, PASS/FAIL per dataset, non-zero exit on any failure.

- **Ran for real. Result: 9/9 FAIL, exit 1.** Not a bad digest: TI's CDN now issues a 302 from
  `software-dl.ti.com` to `downloads.ti.com` for all nine datasets, and `fetch_dataset()`'s
  `_HostLockedRedirectHandler` (10-02, mitigating T-10-02-01/05) correctly refuses any
  cross-host redirect. Independently verified via `curl -sL` + sha256 (bypassing the code's
  host-lock): all nine downloaded bytes match the registry digests exactly. **10-02's
  `fetch_dataset()` is currently non-functional against TI's real production infrastructure,
  for every dataset, right now** — this affects every real `mmcli datasets pull`/
  `init --dataset --fetch` invocation today, independent of this plan.

- Per the plan's own explicit instruction ("This gate is blocking... stop: do not proceed to
  Task 2"), **Tasks 2 and 3 were not executed**: no `--add-data` line was touched in any build
  script, `scripts/binary_size_ceiling.txt` is still `152043520`, no build-script comment was
  rewritten, `dist/mmcli` was not built.

- **BLOCKER for follow-up (not fixed here — out of 10-03's `files_modified` scope, and a
  security-relevant architectural decision per the plan's own deviation rules, not a Rule 1-3
  auto-fix):** `mmcli/datasets.py` (10-02's file) needs either (a) `TI_DATASETS_BASE`
  repointed at `downloads.ti.com` directly, or (b) `_HostLockedRedirectHandler` extended to
  allowlist this specific, verified TI redirect pair. Re-run
  `python3 scripts/verify_dataset_digests.py` after that fix; it should need no changes
  itself. Only then resume 10-03 Tasks 2/3.

- `gsd-sdk query state.add-blocker` returned `{"added": false, "reason": "Blockers section
  not found in STATE.md"}` against this file's legacy format (consistent with the
  `state.advance-plan`/`state.add-decision` parse-error note above) — recorded here manually
  instead, per that same fallback convention.

- See `.planning/phases/10-dataset-distribution-and-binary-size/10-03-SUMMARY.md` for the
  full verbatim gate output and the content-verification detail.

### Session: 2026-07-23 (Phase 10 Plan 03 execution — resumed and COMPLETE)

**Plan 10-03 — GitHub release mirror repoint, publish, and dataset unbundle — COMPLETE, with
one honestly-reported overshoot.**

- Resumes the blocked attempt above: `mmcli/datasets.py` (10-02's file) repointed from the
  dead `software-dl.ti.com` source to this project's own public GitHub release mirror
  (`datasets-01_03_00`), per this plan's own D-A..D-D decisions rather than either candidate fix
  the blocked attempt flagged. `dataset_url()` now composes
  `github.com/musicalplatypus/tinyml-cli/releases/download/datasets-<version>/<filename>`
  (named by local `filename`, not `ti_name`). `_HostLockedRedirectHandler` gained a closed,
  exact-host `ALLOWED_CROSS_HOST_REDIRECTS = {"github.com":
  {"release-assets.githubusercontent.com"}}` map — every other cross-host redirect, including
  lookalike hosts, still refuses. This narrowly amends 10-02's T-10-02-01/05. `ti_name` kept as
  fetchable-sentinel + TI provenance, no longer the URL source. `tests/test_datasets_download.py`
  updated (50 tests, all passing) including two stale URL assertions the plan's own read_first
  list missed, plus 5 new tests proving the allow-path and multiple refuse-paths.

- **Checkpoint:** the plan's own `checkpoint:decision` (Task 2) paused execution pending public
  publish authorization. The user authorized "publish" via the orchestrator.

- **Publish delegation:** Claude Code's auto-mode permission classifier blocked this agent's own
  `gh release create`/`gh release upload` calls outright as an irreversible ~131 MB public-publish
  action — independent of the plan or the orchestrator's relay of the user's go-ahead. Per this
  agent's operating rules, only the permission system itself or a direct user message counts as
  consent, so the agent stopped rather than routing around the refusal (e.g. via a raw API call).
  The user performed the publish directly; this agent then independently re-verified it (read-only
  `gh release view`: 9 assets, sizes exact-matching the registry `bytes`, audio zip absent; and a
  live re-run of `scripts/verify_dataset_digests.py` against the real mirror: **9/9 PASS, exit
  0**) before proceeding, rather than trusting the report alone.

- All three build scripts (`build_macos.sh`, `build_linux.sh`, `build_windows.ps1`) unbundled: an
  explicit, staged `BUNDLED_DATASETS` allowlist (exactly `generic_audio_classification.zip`) now
  drives a single `--add-data`, replacing the former blanket copy on macOS and adding dataset
  bundling to Linux/Windows for the first time (those two previously shipped zero datasets).
  `scripts/binary_size_ceiling.txt` lowered to `15728640` (15 MB). `tests/test_build_config.py`
  extended with 15 new parametrised assertions (30 total, all passing); manually confirmed (via
  scratch copies, not committed) that they fail when `--add-data` is deleted or given the wrong
  separator.

- **Real `bash build_macos.sh` run: binary carries exactly one dataset zip** (confirmed via
  PyInstaller archive inspection — none of the other nine leaked in despite all ten zips being
  present in the working tree). **However, `dist/mmcli` measures 31,839,872 bytes (~31.84 MB)
  and starts in ~6.7-9.6s median — both OVER REQ-SIZE-01's 15 MB / 2.5s bounds, even after full
  unbundling.** Archive inspection shows the remainder is numpy/pandas/PIL/cryptography native
  libraries genuinely used in-process (not datasets, not the excluded training engine), plus
  likely `--onefile` per-launch extraction overhead on an unnotarized binary. Per explicit
  instruction, neither figure was massaged and the ceiling was not raised — this is flagged as an
  open follow-up rather than silently resolved. `requirements-completed` for this plan is
  `[REQ-DATA-04, REQ-DATA-05]`; `REQ-SIZE-01` is NOT marked complete.

- `pwsh` remains absent on this machine; `build_windows.ps1` verified at the source-assertion
  level only via `tests/test_build_config.py`.

- Commits: `73ad23f` (Task 1 — repoint + redirect allowlist), `42b6b3f` (Task 4 — unbundle all
  three scripts + ceiling + tests), `6f7c60a` (Task 5 — real measurement recorded in
  `build_macos.sh`'s comment). Task 3 (the publish) created no repo-file commit by design
  (`files_modified: []`); it was performed by the user directly, per the delegation above.

- `gsd-sdk query state.advance-plan` again returned a parse error against this file's legacy
  narrative format (consistent with the 10-01/10-02/10-03-blocked sessions above); recorded here
  manually instead. `gsd-sdk query roadmap.update-plan-progress 10` was run and updated
  `.planning/ROADMAP.md`'s Phase 10 plan table directly.

- See `.planning/phases/10-dataset-distribution-and-binary-size/10-03-SUMMARY.md` for full
  detail (references the blocked attempt-1 record above rather than discarding it) and
  `10-03-SUMMARY-attempt1-blocked.md` for the original TI-CDN-moved diagnosis this plan resolved.

### Pending Todos

6 todos captured 2026-07-09 from test suite failures found after installing updated modelmaker in venv-tinyml:

1. Fix sanitization test expectations for raises-only design (12 failing tests — testing)
2. Fix yaml.py stub safe_load to accept string input (3 dry-run E2E failures — general)
3. Add format attribute to MockArgs in test_recommend.py (1 test — testing)
4. Fix _is_safe_path traversal detection for edge cases (2 tests — general)
5. Investigate MPS float64 crash from tinyml_tinyverse update (7 E2E errors — general, upstream regression)
6. Update test_no_pca_images for new modelmaker PCA behavior (1 test — testing)
