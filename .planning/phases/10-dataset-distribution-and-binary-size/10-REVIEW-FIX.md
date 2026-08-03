---
phase: 10-dataset-distribution-and-binary-size
fixed_at: 2026-08-03T00:25:15Z
review_path: .planning/phases/10-dataset-distribution-and-binary-size/10-REVIEW.md
iteration: 1
findings_in_scope: 16
fixed: 16
skipped: 0
status: all_fixed
---

# Phase 10: Code Review Fix Report

**Fixed at:** 2026-08-03T00:25:15Z
**Source review:** .planning/phases/10-dataset-distribution-and-binary-size/10-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 16 (2 Critical + 14 Warning; Info findings IN-01..IN-07 are out of scope per `fix_scope: critical_warning`)
- Fixed: 16
- Skipped: 0

## Mandatory mutation-testing discipline (CR-01, CR-02)

Both CRITICAL findings were instances of "a test that simulates a code path does not test
it." Both were mutation-tested per the explicit dispatch instructions, with real (not
assumed) red/green transitions recorded below, and both left the mutated file byte-identical
to its pre-mutation state afterward (`git diff` empty before each commit).

- **CR-01**: mutated `scripts/binary_size_ceiling.txt` to the retired `152043520` (145 MiB)
  value — `TestBinarySizeCeiling::test_ceiling_is_the_sanctioned_value` went red
  (`AssertionError: 152043520 != the sanctioned ceiling 27262976`). Restored to `27262976`,
  confirmed green, confirmed `git diff scripts/binary_size_ceiling.txt` was empty before
  committing.
- **CR-02**: mutated `extract_dataset()` to a naive, unsanitised member-by-member write
  (simulating a vulnerable extractor with zero containment) — the fixed test went red
  (`AssertionError: zip-slip member escaped to <exact predicted path>`), catching the escape
  at precisely the filesystem location the math in the finding predicts. Restored the real
  containment guard, confirmed green.

Every Warning-severity test-guard fix below also received a real red/green mutation-test
cycle (not just a passing run against the fixed code); each is documented individually.

## Fixed Issues

### CR-01: Binary size gate is bypassable — the retired 145 MiB ceiling was still sanctioned

**Files modified:** `tests/test_build_config.py`
**Commit:** `770bf84`
**Applied fix:** Collapsed `SANCTIONED_CEILINGS = (152043520, 15728640, 27262976)` to a
single `CEILING = 27262976` constant; the two retired values now live only in a comment,
never in an assertion set. `test_ceiling_is_a_sanctioned_value` (membership check) became
`test_ceiling_is_the_sanctioned_value` (equality check). Mutation-tested as described above.

### CR-02: The zip-slip regression test asserted on a path a successful escape can never create

**Files modified:** `mmcli/datasets.py`, `tests/test_datasets_download.py`
**Commit:** `306310a`
**Applied fix:** Two parts.
1. Fixed the test's marker path from `tmp_path / "tmp" / "..."` (one level inside `tmp_path`,
   written by neither safe nor unsafe extraction) to the escape's real destination —
   `(project_path / "dataset" / "../../../../tmp").resolve() / "evil_zip_slip_marker.txt"` —
   confirmed by direct computation to match what an actual escape would produce. Added a
   positive containment assertion (every extracted file must resolve under the project
   directory).
2. Added an explicit containment guard to `extract_dataset()` in `mmcli/datasets.py`. The
   review's own suggested implementation (pre-computing the target via
   `os.path.realpath(os.path.join(dataset_dir, member))`) was adapted after direct testing
   showed it disagrees with zipfile's actual arcname sanitisation (zipfile strips leading
   `../` path *components*, it does not resolve them via the filesystem) — the naive
   pre-check produced a false positive, rejecting a member zipfile itself already handles
   safely. The implemented guard instead extracts member-by-member via `zf.extract()` and
   validates its own returned path post-write, avoiding any need to reimplement zipfile's
   internal path-sanitisation logic.

### WR-01: `init --dataset` downloaded up to 56 MB before validating destination/task

**Files modified:** `mmcli/cli.py`, `tests/test_datasets_cli.py`
**Commit:** `7912234`
**Applied fix:** Hoisted the task-compatibility and "project directory already exists"
checks ahead of the D-5 auto-fetch policy call in `main()`'s `init` branch. Added 2 new
tests (`test_incompatible_task_rejected_before_any_download_attempt`,
`test_existing_project_dir_rejected_before_any_download_attempt`), both using
`_forbid_download` to fail loudly if a download is attempted. Mutation-tested: with the fix
reverted, both new tests fail with `assert 1 == 2` (the forbidden-download path fires
instead of the expected argument-error exit code), confirming the ordering bug is real and
caught.

Also discovered and fixed during this change: a pre-existing `UnboundLocalError` bug — a
local `import ... os` later in the same `main()` function makes `os` function-local for the
whole function per Python scoping rules, so referencing `os.path.exists` earlier (my new
code) raised `UnboundLocalError` until a local `import os` was added at the point of first
use.

### WR-02: `release_preflight.py`'s mirror check was a verbatim copy of `release.yml`'s, with no drift guard

**Files modified:** `scripts/release_preflight.py`, `docs/RELEASING.md`, `tests/test_ci_workflows.py`
**Commit:** `c1b14dc`
**Applied fix:** Implemented the review's fallback option (add a drift-guard test) rather
than the refactor-into-shared-module option, since the latter requires modifying and
re-verifying a live GitHub Actions workflow file end-to-end, which cannot be executed
locally. Added `test_mirror_check_gh_argv_matches_between_script_and_workflow` and
`test_mirror_check_fatal_messages_match_between_script_and_workflow` to
`tests/test_ci_workflows.py`, extracting the `gh release view` argv and the four `FATAL:`
message templates from both files via regex and asserting equality. Corrected the false
"reused rather than reimplemented" claim in both the script docstring and
`docs/RELEASING.md` §5 to describe what's actually true: a duplicated implementation kept
in lockstep by the new drift-guard tests. Mutation-tested: renamed one FATAL message word in
`release.yml` — the drift-guard test caught it precisely; restored, confirmed green.

### WR-03 / WR-04: `release_preflight.py` crashed on missing `gh`/CWD-relative path; zero test coverage

**Files modified:** `scripts/release_preflight.py`, `tests/test_release_scripts.py` (new file), `tests/test_ci_workflows.py`
**Commit:** `9902c96`
**Applied fix:** Combined WR-03 and WR-04 into one commit since WR-04's new tests are the
verification evidence for WR-03's fix. `check_mirror_tag_and_assets()` now catches
`FileNotFoundError` from the `gh` subprocess call and `ImportError` from
`mmcli.datasets`, printing a `FATAL:` line instead of letting a traceback escape.
`check_digests()` now resolves `scripts/verify_dataset_digests.py` via
`REPO_ROOT = Path(__file__).resolve().parent.parent` instead of a CWD-relative path, and
checks the file exists before invoking it. Created `tests/test_release_scripts.py` (17
tests) covering both scripts' decision logic with `gh`/subprocess/`fetch_dataset` stubbed:
missing-`gh` failing closed, wrong tagName, missing/zero-size asset, non-zero digest exit,
missing digest script, CWD-independence, and `verify_dataset_digests.py`'s pass/fail/unknown
`--only` paths. This file is **not** one of the six CI-collected files (that gap is IN-06,
an Info finding, out of scope) — run separately, 17/17 pass. Mutation-tested: reverted the
WR-03 code change and re-ran the new tests — 3 of them (the `gh`-missing and
`REPO_ROOT`-dependent ones) correctly went red (`FileNotFoundError` propagating raw, and
`AttributeError: no attribute 'REPO_ROOT'`); restored, confirmed green.

### WR-05: The redirect handler locked the host but not the scheme

**Files modified:** `mmcli/datasets.py`, `tests/test_datasets_download.py`
**Commit:** `4622be8`
**Applied fix:** Adapted the review's suggested "always require https" check after direct
testing showed it breaks a legitimate existing test
(`test_cross_host_redirect_refused`, which deliberately uses `http://127.0.0.1:<port>`
against a local test server, since `fetch_dataset()` — not this low-level handler — is what
enforces HTTPS-only on the initial URL in production). The implemented fix instead refuses
any scheme **downgrade** relative to the original request's scheme, and refuses `ftp`
unconditionally regardless of the original scheme: `https → http/ftp` is always refused;
`http → http` (only reachable via direct low-level test calls) is tolerated, since it is not
a downgrade. Added 3 new tests: same-host http downgrade refused, same-host ftp downgrade
refused, and the allowlisted cross-host target refused over http (the scheme guard runs
before the host allowlist). Mutation-tested: reverted the fix — all 3 new tests failed with
`DID NOT RAISE RuntimeError`; restored, confirmed green.

### WR-06: Test fixtures wrote synthetic zips into the real package source directory

**Files modified:** `tests/test_datasets_cli.py`, `tests/test_datasets_download.py`
**Commit:** `fc08930`
**Applied fix:** This bug reproduced live, twice, during this fix session — stray fake zips
accumulated in `mmcli/example_datasets/` mid-session and caused real, confusing test
failures in *other*, unrelated tests (`sha256 mismatch` errors) exactly as the finding
predicts. Both autouse fixtures (`test_datasets_cli.py`'s `dataset_zips_present` and
`test_datasets_download.py`'s `_dataset_zips_present`) now stage real-copied-or-synthesised
stand-in zips in a `tmp_path_factory`-managed directory and redirect `_datasets_dir()`
there, never writing into the real package tree. Preserved `_datasets_dir()`'s real
`MMCLI_DATASETS`-first priority in the redirect (an earlier version of this fix broke
`test_mmcli_datasets_set_absent_refuses_even_with_fetch`, since a naive "always return the
staged dir" lambda silently defeated MMCLI_DATASETS overrides — caught before commit).
Updated several test bodies that independently hardcoded the real package path
(`_REAL_BUNDLED_DIR`) to instead reference the staged directory (via the fixture's return
value where `hide_bundled` is also in play, since that fixture separately overrides
`_datasets_dir()`). Verified the fix by running the full `test_datasets_download.py` +
`test_datasets_cli.py` suite twice in a row and confirming `mmcli/example_datasets/`
contained only the one real tracked file (`generic_audio_classification.zip`) both times —
no accumulation.

### WR-07: `TestRegistryInvariants` opted out of the stand-in fixture by class-name string

**Files modified:** `tests/test_datasets_download.py`, `pyproject.toml`
**Commit:** `f808eeb`
**Applied fix:** Replaced the `request.cls.__name__ == "TestRegistryInvariants"` string
comparison with `request.node.get_closest_marker("no_dataset_standins")`, and marked the
class with `@pytest.mark.no_dataset_standins`. Registered the marker in
`pyproject.toml`'s `[tool.pytest.ini_options]` (this project's `pytest.ini` has an inert
`[tool:pytest]` header rather than `[pytest]`, so pytest actually reads config from neither
file for `markers`/`--strict-markers` purposes — confirmed via `configfile: pytest.ini
(WARNING: ignoring pytest config in pyproject.toml!)` in test output — so this marker
produces the same harmless `PytestUnknownMarkWarning` the pre-existing `cli`/`e2e` markers
already do; not a regression). Mutation-tested: removed the marker from the class —
`test_fan_blade_fault_measured_values` (which asserts exact hardcoded sha256/byte values)
went red (`assert 334 == 56595859`, the stand-in's byte count instead of the real one);
restored, confirmed green.

### WR-08: `fetch_dataset`'s stale-cache branch was untested; unguarded `os.unlink`

**Files modified:** `mmcli/datasets.py`, `tests/test_datasets_download.py`
**Commit:** `4cf7c2b`
**Applied fix:** Added `test_fetch_dataset_stale_cache_entry_is_unlinked_and_redownloaded`,
exercising `fetch_dataset()` itself (not `_download_to_cache` directly, which the existing
`test_corrupted_cache_entry_is_redownloaded` does — testing the helper, not the branch it
claims to cover). Wrapped the `os.unlink(dest_path)` call in try/except, raising
`RuntimeError` with the dest path and original exception on failure — matching the
docstring's promise and what `_handle_datasets_pull`/`_do_init_fetch`'s exception handlers
actually catch. Added `test_fetch_dataset_stale_cache_unlink_failure_raises_runtimeerror`.
Mutation-tested: reverted the guard — the OSError test failed with a raw
`OSError: Permission denied (simulated)` propagating instead of `RuntimeError`; restored,
confirmed green.

### WR-09: A subprocess guard test never checked the exit code

**Files modified:** `tests/test_datasets_cli.py`
**Commit:** `d7b2a7a`
**Applied fix:** `test_datasets_path_does_not_create_the_cache_directory` now asserts
`proc.returncode in (0, 1)` (0 = found locally, 1 = "not available locally" — both are
legitimate outcomes depending on whether real dataset zips happen to be present on the dev
machine) and `"Traceback" not in proc.stderr`. Mutation-tested: inserted an unconditional
`raise RuntimeError(...)` at the top of `_handle_datasets_path` — the test correctly failed
on the traceback check; restored, confirmed clean (`git diff mmcli/cli.py` empty).

### WR-10: The exclude-list guard early-returned on a bare substring match

**Files modified:** `tests/test_build_config.py`
**Commit:** `8c348d7`
**Applied fix:** Adapted the review's suggested regex after direct testing exposed two real
bugs in it: (1) `re.search` from the bare word "pyinstaller" matched an earlier, unrelated
`pip install pyinstaller` prerequisite-check occurrence in every script, letting the 800-char
window spuriously reach an unrelated `$ExcludeArgs = ...` *assignment* line rather than the
actual invocation; (2) the suggested pattern used PowerShell's `$ExcludeArgs` variable-read
syntax, but the actual script uses `@ExcludeArgs` (the splat operator) — these silently
false-positive-passed even with the exclude flags fully removed from the Windows script.
Fixed by anchoring on the actual multi-line invocation specifically (`pyinstaller` followed
immediately by a line-continuation character) and matching the correct splat syntax.
Mutation-tested against all three build scripts simultaneously (removed
`"${EXCLUDE_ARGS[@]}"`/`@ExcludeArgs` from each pyinstaller call) — all 3 parametrised test
cases went red with the exact predicted message; restored all three, confirmed
`git diff build_macos.sh build_linux.sh build_windows.ps1` empty before committing.

### WR-11: `MMCLI_AUTO_FETCH` had no test coverage and silently ignored most values

**Files modified:** `mmcli/cli.py`, `tests/test_datasets_cli.py`
**Commit:** `36be051`
**Applied fix:** `_resolve_explicit_fetch` now normalises (`.strip().lower()`) and accepts
`1/true/yes/on` and `0/false/no/off`; any other value prints a `WARNING:` to stderr and
falls back to the TTY rule (previously silent). Added `TestResolveExplicitFetch` (22 tests):
recognised true/false spellings (including case and whitespace variants), unrecognised
values producing the warning, and CLI-flag-beats-env-var precedence in both directions.
Mutation-tested: reverted the fix — 16 of 22 new tests correctly failed (the previously
unrecognised spellings resolved to `None` instead of the expected bool, and no warning was
printed); restored, confirmed green.

### WR-12: `_validate_args` never ran for `init`, bypassing the path-traversal guard

**Files modified:** `mmcli/cli.py`, `tests/test_datasets_cli.py`
**Commit:** `15a17e8`
**Applied fix:** Chose the review's minimal option (apply the guard directly in the `init`
branch) over moving the dispatch after `_validate_args(args)`, since the latter's other
checks (module/task/device/model "required" enforcement, NAS validation) assume a
`train`/`compile`-shaped argument set and would spuriously reject `init`/`datasets` — a
larger, riskier change for a security-inconsequential-but-inconsistent gap (the review
itself notes "no privilege boundary is crossed"). Added the same `_is_safe_path` check
`_validate_args` applies to `--project` elsewhere, plus `_sanitize_input` on `--task` (which
is otherwise embedded unsanitised in the printed "Next steps" command). Added
`test_project_path_traversal_rejected_before_any_download_attempt`. Mutation-tested:
reverted the fix — the new test failed with `assert 1 == 2` (the forbidden-download path
fired, proving the traversal guard was bypassed and a fetch was attempted); restored,
confirmed green.

### WR-13: A failed extraction left a half-created project directory; narrow except clause

**Files modified:** `mmcli/datasets.py`, `tests/test_datasets_download.py`
**Commit:** `9fea0d8`
**Applied fix:** Widened `except zipfile.BadZipFile` to
`except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, RuntimeError)`, and added
`shutil.rmtree(project_path, ignore_errors=True)` cleanup on every failure path, including
the zip-slip containment-escape branch added in CR-02 (for full consistency, since the same
"half-created directory blocks retry" problem applies there too). Added
`test_bad_zip_file_removes_the_half_created_project_directory` and
`test_oserror_mid_extract_is_caught_and_cleans_up`. Mutation-tested: reverted the fix — both
tests failed (the BadZipFile case left the directory behind; the OSError case propagated
raw instead of being caught by `SystemExit`); restored, confirmed green.

### WR-14: Windows test failures did not block the release build

**Files modified:** `.github/workflows/release.yml`, `tests/test_ci_workflows.py`
**Commit:** `5bcfec7`
**Applied fix:** Chose the review's Option 1 (drop `continue-on-error` from `release.yml`'s
`test` job specifically) over Option 2 (deselect specific known-flaky tests), since I have
no Windows runner to identify which tests are genuinely flaky versus load-bearing, and the
mechanical fix — a release build must not proceed on any red matrix leg — is unambiguous and
independently correct regardless of that judgment call. `test-cli.yml` (PR iteration, where
tolerating known Windows flakiness may be a deliberate, separate call) is intentionally
untouched. Added `test_release_workflow_does_not_tolerate_windows_test_failures`, asserting
`continue-on-error` does not appear in the (comment-stripped) workflow text. Mutation-tested:
restored the old `continue-on-error` line — the new test correctly failed; removed it again,
confirmed `git diff .github/workflows/release.yml` matched exactly the intended diff before
committing.

## Skipped Issues

None — all 16 in-scope findings (CR-01, CR-02, WR-01 through WR-14) were fixed.

## Out of scope (not attempted)

Info findings IN-01 through IN-07 are excluded per `fix_scope: critical_warning` and were
not attempted.

## Final verification

Ran the full six-file phase-10 suite exactly as both CI workflows do:

```
MMCLI_PYTHON=$HOME/.venv-tinyml/bin/python $HOME/.venv-tinyml/bin/python -m pytest \
  tests/test_cli_integration.py tests/test_tier4_cli.py tests/test_build_config.py \
  tests/test_datasets_download.py tests/test_datasets_cli.py tests/test_ci_workflows.py \
  -q --tb=short -k "not TestInitDatasetExtractReal"
```

Real output: **212 passed, 2 skipped, 20 deselected, 7 warnings in 107.11s**.

Baseline before any fixes in this session: 178 passed, 20 deselected (per dispatch
instructions). This run added 36 new tests across the fixes above (25 in
`test_datasets_cli.py`: WR-01 +2, WR-11 +22, WR-12 +1; 7 in `test_datasets_download.py`:
WR-05 +3, WR-08 +2, WR-13 +2; 4 in `test_ci_workflows.py`: WR-02 +3, WR-14 +1) —
`178 + 36 = 214` total, confirmed via `--collect-only` (`214/234 tests collected, 20
deselected`). Of the 214, 2 are skipped: the pre-existing `_needs_real_zips`-marked
subprocess tests, which correctly skip because this checkout's `mmcli/example_datasets/`
contains only the one real tracked zip (`generic_audio_classification.zip`) — unrelated to
any fix in this session, and the WR-06 fix is specifically what keeps this skip condition
stable across repeated runs (previously it flickered based on fixture-teardown leakage).
`178 + 36 - 2 = 212` passed. Nothing failed.

The new `tests/test_release_scripts.py` (WR-03/WR-04, 17 tests) is not one of the six
CI-collected files — that gap is IN-06 (Info, out of scope). Run separately: **17 passed**.

`git status --short` in the fix worktree: clean (no stray mutation-test edits) before every
commit and at the end of this session.

---

_Fixed: 2026-08-03T00:25:15Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
