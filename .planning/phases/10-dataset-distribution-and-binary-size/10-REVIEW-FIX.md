---
phase: 10-dataset-distribution-and-binary-size
fixed_at: 2026-08-03T04:49:03Z
review_path: .planning/phases/10-dataset-distribution-and-binary-size/10-REVIEW.md
iteration: 2
findings_in_scope: 23
fixed: 23
skipped: 0
status: all_fixed
---

# Phase 10: Code Review Fix Report

**Fixed at:** 2026-08-03T04:49:03Z
**Source review:** .planning/phases/10-dataset-distribution-and-binary-size/10-REVIEW.md
**Iteration:** 2 (cumulative — iteration 1 fixed CR-01/CR-02/WR-01..WR-14; iteration 2 fixed
the 7 Info findings IN-01..IN-07)

**Summary:**
- Findings in scope: 23 (2 Critical + 14 Warning + 7 Info)
- Fixed: 23
- Skipped: 0

This report is cumulative across both fix iterations. Iteration 1's findings and their
commits (`770bf84`..`5bcfec7`) are preserved below verbatim from the original report;
iteration 2 added the "Info findings (iteration 2)" section and its own final verification.

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
Every Info-severity fix in iteration 2 (below) received the same treatment.

## Fixed Issues (iteration 1: CR-01, CR-02, WR-01..WR-14)

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
`--only` paths. This file is **not** one of the six CI-collected files at the time of
iteration 1 (that gap was IN-06, then Info/out-of-scope) — run separately, 17/17 pass.
Mutation-tested: reverted the WR-03 code change and re-ran the new tests — 3 of them (the
`gh`-missing and `REPO_ROOT`-dependent ones) correctly went red (`FileNotFoundError`
propagating raw, and `AttributeError: no attribute 'REPO_ROOT'`); restored, confirmed green.

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

## Fixed Issues (iteration 2: IN-01..IN-07)

### IN-01: `verify_dataset_digests.py --only <bundled-only>` reported a registry-wide message

**Files modified:** `scripts/verify_dataset_digests.py`, `tests/test_release_scripts.py`
**Commit:** `529ec8f`
**Applied fix:** When `checked == 0`, distinguish `args.only is not None` (a real registry
entry with no mirror asset, e.g. `generic_audio_classification`) from the true whole-registry
case: prints `"'{name}' has no mirror asset and is not fetchable (bundled-only)."` and returns
`2` (an argument-shape problem, consistent with the existing unknown-name case) instead of the
generic "No fetchable datasets found in DATASET_REGISTRY." + exit `1`. Added
`test_only_bundled_only_name_reports_specific_message_not_registry_wide`. Mutation-tested:
reverted the source fix — the new test failed (`assert 1 == 2`, old registry-wide message
present in stderr); restored, confirmed green.

### IN-02: `--skip-digests` returned exit 0 for a partial gate; static `[1/2]` labels

**Files modified:** `scripts/release_preflight.py`, `tests/test_release_scripts.py`, `docs/RELEASING.md`
**Commit:** `c950c7f`
**Applied fix:** Introduced `PARTIAL_EXIT_CODE = 3`, returned by `main()` for a
`--skip-digests` run whose mirror check passed — deliberately not `0`, so a wrapper checking
only `$?` cannot mistake a skipped ~131 MB digest gate for a full pass. `1` is preserved for
an actual check failure. `check_mirror_tag_and_assets()` gained a `total_steps` kwarg
(default `2`); `main()` passes `1` under `--skip-digests` so the `"[1/N]"` progress label
(and the `"PREFLIGHT FAILED at step 1/N"` message) reflect that only one step will run in
that invocation, instead of an unconditional `"[1/2]"`. Documented the new exit codes in the
script's own docstring and `docs/RELEASING.md` §5. Updated the one pre-existing test that
encoded the old (buggy) "exits 0" behavior; added 4 new tests (partial-exit-code value,
`total_steps=1` under `--skip-digests`, `total_steps=2` without it, and the resulting
`"[1/1]"` vs `"[1/2]"` label text). Mutation-tested: reverted the source fix, confirmed all 4
new tests failed (`TypeError` on the unexpected `total_steps` kwarg / `AttributeError: no
attribute 'PARTIAL_EXIT_CODE'` / `assert None == 1`); restored, confirmed green.

### IN-03: `gh` JSON parsing was unguarded in both `release_preflight.py` and `release.yml`

**Files modified:** `scripts/release_preflight.py`, `.github/workflows/release.yml`, `tests/test_release_scripts.py`
**Commit:** `32af3e3`
**Applied fix:** Wrapped `json.loads(result.stdout)` (plus the `.get("tagName")` access, which
can raise `AttributeError` on non-object JSON) in `try/except (ValueError, TypeError,
AttributeError)`, and the `{a["name"]: a.get("size", 0) for a in data.get("assets", [])}`
comprehension in `try/except (TypeError, KeyError)` — both printing a `FATAL:` line with the
parse error and the first 200 characters of raw `gh` output. Applied the byte-identical change
to `release.yml`'s embedded mirror-healthcheck script (using `sys.exit(1)` instead of `return
False`) so `tests/test_ci_workflows.py`'s WR-02 drift-guard tests (which compare the `gh` argv
and `FATAL:` message wording between the two files) stay green — verified directly, not
assumed. Added 4 regression tests: non-JSON stdout, non-object JSON (a bare list), `"assets":
null`, and an asset object missing `"name"`. Mutation-tested: reverted both source files,
confirmed all 4 new tests failed with the raw `JSONDecodeError`/`AttributeError`/
`TypeError`/`KeyError` propagating; restored, confirmed green (drift-guard tests included).

### IN-04: A cache hit for an entry with no `sha256` was reported as a digest mismatch

**Files modified:** `mmcli/datasets.py`, `tests/test_datasets_download.py`
**Commit:** `4655cd9`
**Applied fix:** `_resolve_dataset_zip`'s cache-hit branch now distinguishes a falsy
`expected` (no `sha256` recorded — legal for a non-`ti_name` local entry, since
`_validate_registry` only requires `sha256` for `ti_name` entries) from an actual digest
mismatch, with its own message stating there was nothing to verify against, and drops the
`--force` suggestion (which fails for an entry with no fetchable URL). Added
`test_cache_hit_with_no_sha256_reported_as_unverifiable_not_mismatch`, asserting the new
wording is present and the old "does not match"/`--force` wording is absent. Mutation-tested:
reverted the source fix, confirmed the new test failed (old "does not match"/`--force`
wording present in stderr); restored, confirmed green (58 passed in
`test_datasets_download.py`).

### IN-05: Direct `os.environ` mutation and a `finally` depending on an import inside `try`

**Files modified:** `tests/test_datasets_download.py`
**Commit:** `5621d69`
**Applied fix:** `TestZipSlipProtection::test_parent_traversal_member_stays_inside_project`
replaced `import os as _os` (performed inside the `try`, with the `finally` depending on that
name existing) and raw `os.environ["MMCLI_DATASETS"] = ...` / `DATASET_REGISTRY[...] = ...`
mutation with `monkeypatch.setenv(...)` / `monkeypatch.setitem(...)`, both auto-restored on
teardown including on assertion failure, and dropped the try/finally entirely. `os` is
already imported at module scope, so the shadow import was purely redundant. No change to
what the test asserts. Verified with a full run of `test_datasets_download.py` (57 passed) and
the specific test in isolation.

### IN-06: Only 6 of ~38 test files ran in either CI workflow

**Files modified:** `.github/workflows/test-cli.yml`, `.github/workflows/release.yml`, `tests/test_ci_workflows.py`
**Commit:** `8682fe3`
**Applied fix:** Per the dispatch's mandatory procedure for this finding, ran the full suite
for real *before* touching anything:
`MMCLI_PYTHON=$HOME/.venv-tinyml/bin/python $HOME/.venv-tinyml/bin/python -m pytest tests/ -q
--tb=short -k "not TestInitDatasetExtractReal"` → **644 passed, 2 skipped, 20 deselected,
11 warnings in 698.16s**, fully green. Since fully green, widened CI scope: both workflows'
`Run tests` step now runs
`python -m pytest tests/ -v --tb=short -k "not TestInitDatasetExtractReal"` (collection by
default) instead of naming six files explicitly, so `test_security.py`,
`test_fuzz_path_validation.py`, `test_attack_surface.py`, `test_integration_security.py`, and
every other previously-uncollected file are now in CI on every push/PR/release. Rewrote
`tests/test_ci_workflows.py`'s drift guard to match: instead of asserting the named-file set
is a superset of six required files, it now asserts (a) the invocation's first argument is
the bare `tests/` directory, (b) no individual `tests/test_*.py` file is hardcoded any more
(guards against silently narrowing back down), and (c) the two workflows' invocations are
identical after whitespace normalisation. `REQUIRED_TEST_FILES` is kept as a lighter-weight
residual guard (existence-on-disk only) against those six specific files being silently
deleted/renamed, since with `tests/` collected wholesale a missing file produces no pytest
error at all — just fewer tests run.

Re-ran the full suite for real *after* the change, from a clean git status:
**644 passed, 2 skipped, 20 deselected, 10 warnings in 675.00s** — identical pass/skip/
deselect counts to the pre-change baseline (one fewer duplicate warning line), confirming the
widened scope surfaces no new failures from files that were never exercised in CI before.
Both counts were measured directly this session, not assumed from the review or from
iteration 1.

Mutation-tested the guard itself: reverted both workflow files to the old six-file,
backslash-continued form, confirmed
`test_pytest_invocation_collects_the_whole_tests_directory` failed (`first_arg` was the line-
continuation backslash `'\\'`, not `'tests/'`); the other new/kept tests in the file correctly
stayed green under the old form (they check different properties), confirming the new test is
the one actually carrying the regression signal. Restored, confirmed green (10 passed in
`test_ci_workflows.py`).

### IN-07: `_download_to_cache`'s cleanup handler could mask the original exception

**Files modified:** `mmcli/datasets.py`, `tests/test_datasets_download.py`
**Commit:** `629856a`
**Applied fix:** The `except BaseException: os.unlink(tmp_path); raise` cleanup handler's own
`os.unlink` call is now wrapped in its own `try/except OSError: pass`, so a failure to remove
the temp file (a race with another process, a read-only mount) can never replace the real
error (checksum mismatch, oversize body, truncated download, etc.) the user needs to see —
cleanup is genuinely best-effort. Added
`test_cleanup_unlink_failure_does_not_mask_the_original_error`, which patches `os.unlink` to
raise for `.part` temp files specifically (real files unaffected) during a real checksum-
mismatch download against the local `http.server` fixture. Mutation-tested: reverted the
source fix, confirmed the new test failed with the raw simulated `OSError` propagating in
place of `RuntimeError('Checksum mismatch...')` (visible in the traceback as "During handling
of the above exception, another exception occurred"); restored, confirmed green (59 passed in
`test_datasets_download.py`).

## Skipped Issues

None — all 23 in-scope findings (CR-01, CR-02, WR-01 through WR-14, IN-01 through IN-07) were
fixed across both iterations.

## Final verification (iteration 2)

All commands below were run for real this session against the fix worktree; none of the
figures are assumed or carried over unverified from the review or from iteration 1.

**Full suite** (before and after the IN-06 CI-scope widening — see IN-06 above for the
before/after breakdown):
```
MMCLI_PYTHON=$HOME/.venv-tinyml/bin/python $HOME/.venv-tinyml/bin/python -m pytest tests/ -q \
  --tb=short -k "not TestInitDatasetExtractReal"
```
Before: **644 passed, 2 skipped, 20 deselected, 11 warnings in 698.16s**.
After: **644 passed, 2 skipped, 20 deselected, 10 warnings in 675.00s**.

**Six-file CI suite, exactly as both workflows ran it before IN-06** (kept as a targeted
smoke check even though both workflows now collect all of `tests/`):
```
MMCLI_PYTHON=$HOME/.venv-tinyml/bin/python $HOME/.venv-tinyml/bin/python -m pytest \
  tests/test_cli_integration.py tests/test_tier4_cli.py tests/test_build_config.py \
  tests/test_datasets_download.py tests/test_datasets_cli.py tests/test_ci_workflows.py \
  -q --tb=short -k "not TestInitDatasetExtractReal"
```
Real output: **214 passed, 2 skipped, 20 deselected, 7 warnings in 264.22s**.

Iteration-1 baseline for this same six-file set was **214 total** (212 passed + 2 skipped).
This run shows **216 total** (214 passed + 2 skipped) — a difference of exactly **+2**,
confirmed via `--collect-only`: `216/236 tests collected (20 deselected)` versus iteration
1's recorded `214/234`. The +2 matches the two new regression tests added within these six
files this session: IN-04's `test_cache_hit_with_no_sha256_reported_as_unverifiable_not_mismatch`
(in `test_datasets_download.py`) and IN-07's
`test_cleanup_unlink_failure_does_not_mask_the_original_error` (also
`test_datasets_download.py`). IN-06's own edits to `test_ci_workflows.py` removed 3 old tests
and added 3 new ones (net 0). IN-01/IN-02/IN-03's new tests live in
`tests/test_release_scripts.py`, which is not one of these six files. Nothing failed.

**Real release preflight** (full ~131 MB digest gate against the live mirror, per the
dispatch's requirement since IN-02/IN-03 touch this script directly):
```
$HOME/.venv-tinyml/bin/python scripts/release_preflight.py
```
Real output:
```
[1/2] Checking mirror release 'datasets-01_03_00' in musicalplatypus/tinyml-cli ...
OK: mirror release 'datasets-01_03_00' has all 9 expected assets, all non-zero size (no payload downloaded).
[2/2] Running scripts/verify_dataset_digests.py (full digest gate) ...
... (9/9 PASS lines, one per fetchable dataset) ...
All 9 fetchable dataset(s) PASSED.

PREFLIGHT PASSED: mirror tag/assets OK, all fetchable digests verified. Safe to build.
```
Exit code: **0**. 9/9 digests PASS against the live mirror, unchanged from before IN-01/
IN-02/IN-03 — confirming those fixes did not alter observable behaviour on the success path.

**Mutation-residue check** (per the dispatch's zip-slip-marker warning): confirmed
`/tmp/evil_zip_slip_marker.txt` does not exist and no `*evil_zip_slip*` files were left under
`/tmp` before running the final suite.

**`git status --short`** in the fix worktree: clean before every commit and at the end of
this session (verified directly, not assumed).

All 7 commits from this iteration verified to resolve as real commits
(`git cat-file -t <sha>` → `commit` for each): `5621d69`, `529ec8f`, `4655cd9`, `629856a`,
`32af3e3`, `c950c7f`, `8682fe3`.

---

_Fixed: 2026-08-03T04:49:03Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 2_
