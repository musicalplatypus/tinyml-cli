---
phase: 10-dataset-distribution-and-binary-size
verified: 2026-08-02T23:15:00Z
status: gaps_found
score: 9/13 must-haves verified (1 failed/blocker, 1 partial, 2 unverifiable from this repo)
overrides_applied: 0
gaps:
  - truth: "The binary size ceiling enforced by CI lives in exactly one place, so it cannot drift from the requirement (10-01 must-have); a release build that exceeds scripts/binary_size_ceiling.txt fails before the artifact is uploaded (10-08 must-have, REQ-SIZE-01/REQ-SIZE-02)"
    status: failed
    reason: >
      tests/test_build_config.py:39 defines
      SANCTIONED_CEILINGS = (152043520, 15728640, 27262976), and
      test_ceiling_is_a_sanctioned_value (line ~491) asserts only membership in
      that tuple, not equality with the single current value. 152043520 (145
      MiB) is the retired interim ceiling from when datasets were still
      bundled; 15728640 (15 MiB) is the retired REQ-SIZE-01-original value.
      Independently reproduced during this verification: writing 152043520
      into scripts/binary_size_ceiling.txt and running
      `pytest tests/test_build_config.py -k ceiling` still reports "2 passed,
      0 failed". .github/workflows/release.yml:215-233's size gate reads that
      same file at runtime with no independent bound
      (`CEILING=$(cat scripts/binary_size_ceiling.txt)`), so a binary up to
      5.6x the REQ-SIZE-01 limit — effectively the pre-exclusion,
      training-engine-bundled artifact — would ship with a fully green
      pipeline. The file was restored to 27262976 after the test; `git status`
      confirms no residual change.
    artifacts:
      - path: "tests/test_build_config.py"
        issue: "SANCTIONED_CEILINGS tuple sanctions two retired, oversized ceiling values instead of pinning the single current one"
      - path: "scripts/binary_size_ceiling.txt"
        issue: "Not independently bounded anywhere; its value's correctness rests entirely on the tuple above"
      - path: ".github/workflows/release.yml"
        issue: "Size gate (lines 215-233) trusts the file with no second source of truth"
    missing:
      - "Collapse SANCTIONED_CEILINGS to a single CEILING = 27262976 constant; move the retired values into a comment (not an assertion set), per 10-REVIEW.md CR-01's suggested fix"
      - "Re-run the mutation (write 152043520 into the ceiling file, confirm the test now fails) before closing this gap"
  - truth: "extract_dataset() cannot be used to write outside the target project directory via a malicious zip member path (T-10-02-06's stated threat model; implicitly part of REQ-DATA-02's data-integrity guarantee)"
    status: partial
    reason: >
      tests/test_datasets_download.py:423-459 (TestZipSlipProtection) computes
      the malicious member as "../../../../tmp/evil_zip_slip_marker.txt" and
      extracts into tmp_path/proj/dataset/, but asserts on
      `tmp_path / "tmp" / "evil_zip_slip_marker.txt"` — one level *inside*
      tmp_path, a path neither safe extraction nor a real escape would ever
      write to (a real escape from tmp_path/proj/dataset/ with four `..`
      segments lands at tmp_path's grandparent, not tmp_path/tmp). The
      assertion is true unconditionally, so the test would pass identically
      against an extract_dataset with zero path-traversal protection.
      mmcli/datasets.py's extract_dataset uses a bare `zf.extractall(dataset_dir)`
      with no explicit member-path guard of its own — the safety property is
      entirely delegated to zipfile's own member-path sanitisation (correct
      in current CPython, not documented as a stability guarantee). This path
      is reachable with attacker-influenced zip content via MMCLI_DATASETS,
      which is deliberately not digest-verified. Confirmed by direct code
      reading during this verification; not re-run as a live exploit (out of
      scope for a verification pass — mutation-testing an actual traversal
      was left to the review, which already performed the path-arithmetic
      check).
    artifacts:
      - path: "tests/test_datasets_download.py"
        issue: "Zip-slip regression test asserts on an unreachable path; proves nothing about extract_dataset's real behavior"
      - path: "mmcli/datasets.py"
        issue: "extract_dataset (line ~846) has no explicit member-path containment guard; relies entirely on zipfile's undocumented sanitisation"
    missing:
      - "Fix the assertion to check the real escape destination and add a positive is_relative_to() check over every extracted file, per 10-REVIEW.md CR-02's suggested fix"
      - "Add an explicit realpath-containment guard inside extract_dataset so the property is enforced by mmcli rather than inherited from zipfile"
human_verification:
  - test: "PlatypusStudio New Project sheet: selecting an unbundled dataset shows size + explicit download action, Create stays disabled until local, progress/cancel work, offline vs checksum failures produce different messages (REQ-UX-01)"
    expected: "Matches the 6 must-have truths in 10-04-PLAN.md; per 10-CONTEXT.md D-09 this was driven against the real app for the 10-04 checkpoint, but that verification explicitly could not be reproduced from this repo"
    why_human: "Cross-repo: implemented in ../PlatypusStudio (Sources/PlatypusStudio/NewProjectSheet.swift, Sources/MMCLIKit/DatasetCatalog.swift), a separate git repository with its own SwiftUI executable target that has zero automated test coverage (Package.swift declares one test target, MMCLIKitTests, against MMCLIKit only — no test imports SwiftUI). A tinyml-cli-only verifier cannot exercise the UI."
  - test: "PlatypusStudio dataset library: reachable from workspace toolbar at any time, lists all datasets with size/local state, download/remove work with correct affordances and messages, removal never touches packaged or MMCLI_DATASETS directories, out-of-space produces a legible error, and Cancel during a bulk download actually drops the remaining queue (REQ-UX-02)"
    expected: "Matches the 8 must-have truths in 10-09-PLAN.md"
    why_human: "Cross-repo, same SwiftUI-has-no-tests constraint as above. Additionally, the phase's own record (.continue-here.md, 10-09-SUMMARY.md) states the cancel-drops-queue behavior was specifically left unverified — downloads complete faster than the driving automation could click Cancel — so even the phase's own human-in-the-loop pass did not close this one out."
---

# Phase 10: Dataset Distribution and Binary Size — Verification Report

**Phase Goal (revised facts, not the stale ROADMAP prose):** Ship `dist/mmcli` at ≤26 MiB
(27,262,976 bytes) and <8 s startup (REQ-SIZE-01 as revised 2026-07-31), fetching the nine
non-bundled example datasets on demand from this project's own GitHub release mirror
(`datasets-01_03_00` on `musicalplatypus/tinyml-cli`) rather than from TI, so a dataset can be
release-specific and updated without rebuilding the binary.

**Verified:** 2026-08-02T23:15:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `dist/mmcli` measures ≤27,262,976 B and starts <8 s (3-run median) | ✓ VERIFIED | Measured directly: `dist/mmcli` is 25,258,768 bytes (2,004,208 B headroom); 3 fresh `--version` runs timed 7.32 s / 6.27 s / 6.20 s (median 6.27 s, within the 8 s bound) |
| 2 | The size ceiling enforced by CI cannot silently drift to an unsafe value | ✗ FAILED | `SANCTIONED_CEILINGS` in `tests/test_build_config.py` sanctions two retired oversized values; mutation-confirmed the ceiling test cannot fail in the direction that matters (see Gaps) |
| 3 | PyInstaller excludes the training engine from all three published artifacts, guarded by CI | ✓ VERIFIED | Exclude list (`scripts/pyinstaller_excludes.txt`) consumed by `build_macos.sh:70`, `build_linux.sh:47`, `build_windows.ps1:48`; `tests/test_build_config.py` wired into both `.github/workflows/test-cli.yml` and `release.yml`; CI run `30767443908` on `7eeae8b` green on ubuntu/macos (per `.continue-here.md`); `build` job's per-platform size/bundle gates run on all 3 OSes unconditionally (`continue-on-error` in `release.yml` is scoped to the `test` job only, confirmed by reading the workflow) |
| 4 | Wheel and sdist ship only the locally-authored dataset, not the nine mirrored ones | ✓ VERIFIED | `pyproject.toml`: `include-package-data = false` + `[tool.setuptools.package-data]` allowlist of `generic_audio_classification.zip` only; `tests/test_build_config.py` guards for the allowlist and `MANIFEST.in`; full test run green (198 passed, 0 failed) |
| 5 | Dataset resolution order is `MMCLI_DATASETS` → bundled → cache → download | ✓ VERIFIED | `mmcli/datasets.py:_resolve_dataset_zip` implements exactly that order (lines ~358-393); exercised by `tests/test_datasets_download.py`, all passing |
| 6 | A fetchable registry entry with no valid sha256/bytes is a hard error at import | ✓ VERIFIED | `mmcli/datasets.py:_validate_registry` (lines 259-287) raises `ValueError` naming the offending entry; digest is re-verified before every `os.replace` in `fetch_dataset` |
| 7 | A malicious zip member cannot write outside the project directory | ? PARTIAL | Underlying `zipfile` sanitisation likely holds (Python ≥3.6 default behavior), but the only test claiming to prove it (`TestZipSlipProtection`) asserts on a path neither a safe extraction nor a real escape would ever touch — proves nothing either way. `extract_dataset` has no explicit containment guard of its own. See Gaps. |
| 8 | `MMCLI_DATASETS` disables all network fetching | ✓ VERIFIED | `fetch_dataset` checks `os.environ.get("MMCLI_DATASETS")` and refuses (raises) before any URL is composed (`datasets.py:668-683`) |
| 9 | All 10 datasets remain obtainable offline via `MMCLI_DATASETS` | ✓ VERIFIED | Same resolution path as truth 5 applies uniformly to all registry entries including the bundled-only `generic_audio_classification`; 10-05's offline recipe is documented as executed end-to-end in `10-05-SUMMARY.md`. Not independently re-executed in this verification pass (time cost of assembling 10 zips); confidence rests on code-path uniformity + passing regression tests, not a fresh run. |
| 10 | Datasets are fetched from this project's own versioned GitHub release mirror, cache keyed by version, with a per-dataset override | ✓ VERIFIED | `DATASETS_MIRROR_BASE = "https://github.com/musicalplatypus/tinyml-cli/releases/download"`; `version = meta.get("ti_version") or DATASETS_DEFAULT_VERSION` at 4 call sites; **live-executed** `scripts/release_preflight.py` during this verification: mirror tag/assets OK, all 9 fetchable digests verified against the real mirror, exit 0 |
| 11 | PlatypusStudio blocks example-project creation until the dataset is locally present, with an explicit download step (REQ-UX-01) | ? UNVERIFIABLE FROM HERE | Cross-repo (`../PlatypusStudio`). Artifacts exist (`DatasetCatalog.swift`, `NewProjectSheet.swift`, `DatasetCatalogTests.swift`); no automated coverage of the SwiftUI surface itself. Routed to human verification below. |
| 12 | PlatypusStudio offers a standalone dataset library (download/remove, independent of project creation) (REQ-UX-02) | ? UNVERIFIABLE FROM HERE | Cross-repo. Artifacts exist (`DatasetLibraryView.swift`, extended `DatasetCatalog.swift`, `mmcli datasets remove`). `cache_bytes` field confirmed present in live `mmcli datasets list --format json` output (D-10). Cancel-drops-queue explicitly left unverified by the phase itself. Routed to human verification below. |
| 13 | No README/help-text statement about dataset location is false; `docs/RELEASING.md` states the release dataset obligations (REQ-DOC-01) | ✓ VERIFIED | `docs/RELEASING.md` exists (242 lines, mentions `DATASETS_DEFAULT_VERSION`); `mmcli datasets --help` / `mmcli init --help` no longer say "from TI" (confirmed by live `--help` grep, empty match); README's one "not from TI" mention is explanatory of the mirror move, not a false claim |

**Score:** 9/13 truths verified, 1 failed (blocker), 1 partial, 2 unverifiable from this repository (routed to human verification)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/pyinstaller_excludes.txt` | Single source of truth for excludes | ✓ VERIFIED | Read by all 3 build scripts |
| `scripts/binary_size_ceiling.txt` | Single source of truth for CI ceiling | ⚠️ EXISTS BUT UNDER-GUARDED | Correct value today (`27262976`); nothing prevents a regression to a sanctioned-but-retired value — see Gap 1 |
| `tests/test_build_config.py` | Source-level regression guard | ⚠️ PARTIAL | 36+ guards exist and mostly work (green in full test run); the ceiling guard specifically is vacuous in the direction that matters (CR-01) |
| `mmcli/datasets.py` | Registry, mirror URL composition, verified `fetch_dataset`, `extract_dataset` | ✓ VERIFIED (download path) / ⚠️ UNGUARDED (extraction containment) | Digest verification is real and mandatory; zip-slip containment is delegated to `zipfile` with no explicit guard and an unverifying test |
| `tests/test_datasets_download.py` | Registry invariants, resolution order, fetch failure modes, zip-slip | ✓ MOSTLY VERIFIED | Full suite green; the one zip-slip test present is vacuous (CR-02) |
| `docs/RELEASING.md` | Release-time dataset obligations + ordered checklist + preflight | ✓ VERIFIED | Exists, 242 lines; `scripts/release_preflight.py` executed live during this verification, exit 0, 9/9 digests PASS |
| `scripts/release_preflight.py` | Scripted D-05 preflight | ✓ VERIFIED (functionally) / ℹ️ untested itself | Ran successfully against the live mirror; review notes (WR-04) it has zero unit-test coverage of its own decision logic |
| `.github/workflows/release.yml`, `test-cli.yml` | CI wiring of all phase-10 guards | ✓ MOSTLY WIRED | Six-file pytest invocation present in both; `mirror-healthcheck` job present; per-platform size/bundle/startup gates in `build` job unconditional on all 3 OSes; `test` job has Windows `continue-on-error` (narrower issue than the ceiling gap — does not affect the `build` job's own gates) |
| `../PlatypusStudio/Sources/PlatypusStudio/NewProjectSheet.swift`, `DatasetLibraryView.swift` | Download affordance, dataset library | ✓ EXISTS (unverified behaviorally from here) | Files present, non-trivial size, committed; cannot exercise SwiftUI behavior from this repo |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `mmcli.datasets.dataset_url` | `github.com/musicalplatypus/tinyml-cli/releases/download` | f-string over `DATASETS_MIRROR_BASE` | ✓ WIRED | Confirmed by live preflight run — real GETs against the real mirror, 9/9 PASS |
| `mmcli.datasets._HostLockedRedirectHandler` | `release-assets.githubusercontent.com` | explicit host-pair allowlist | ✓ WIRED | Present at `datasets.py:454`; scheme is NOT locked (WR-05, warning not blocker — digest verification catches content substitution even on a downgraded transport) |
| `release.yml` build job | `scripts/binary_size_ceiling.txt` | `CEILING=$(cat ...)` at runtime | ⚠️ WIRED BUT UNDER-GUARDED | The comparison itself is correct; the file's correctness has no independent backstop (Gap 1) |
| `tests/test_build_config.py` | `scripts/binary_size_ceiling.txt` | `SANCTIONED_CEILINGS` membership check | ✗ NOT DISCRIMINATING | Membership in a 3-value tuple, not equality with the current value — mutation-confirmed to pass a retired 145 MiB value |
| `scripts/release_preflight.py` | `scripts/verify_dataset_digests.py` | subprocess call | ✓ WIRED | Both steps ran and reported correctly in the live execution |
| `mmcli/cli.py datasets pull` | `mmcli.datasets.fetch_dataset` | direct call | ✓ WIRED | `mmcli datasets list --format json` produces the documented JSON contract live (`name/version/state/bytes/cache_bytes/...`) |
| `DatasetCatalog.swift` (PlatypusStudio) | `mmcli datasets list --format json` | `ProcessRunner` | ? UNVERIFIABLE FROM HERE | JSON shape confirmed correct from the mmcli side; cannot confirm the Swift decode path without running the app |

### Requirements Coverage

| Requirement | Claimed by | Status | Evidence |
|---|---|---|---|
| REQ-SIZE-01 | 10-01, 10-03 | ⚠️ PARTIAL | Numeric bound genuinely met today (measured); the CI guard protecting that bound against regression is vacuous (Gap 1) |
| REQ-SIZE-02 | 10-01, 10-08 | ⚠️ PARTIAL | Exclusions are real and wired into CI; the ceiling-drift gap above is shared infrastructure with REQ-SIZE-01, so the same weakness applies — a training-engine-bundled binary at the retired 145 MiB ceiling would also pass |
| REQ-SIZE-03 | 10-10 | ✓ SATISFIED | package-data allowlist + `include-package-data = false`, tests green |
| REQ-DATA-01 | 10-02, 10-06 | ✓ SATISFIED | Resolution order confirmed in code and tests |
| REQ-DATA-02 | 10-02 | ⚠️ PARTIAL | Digest enforcement at import/download is solid and independently confirmed; the extraction-containment half (zip-slip) is untested in a way that proves anything (Gap 2) |
| REQ-DATA-03 | 10-02 | ✓ SATISFIED | `MMCLI_DATASETS` refusal checked before URL composition |
| REQ-DATA-04 | 10-03, 10-05 | ✓ SATISFIED (not re-executed) | Uniform resolution path; 10-05's recipe documented as executed once, not repeated here |
| REQ-DATA-05 | 10-02, 10-03, 10-07 | ✓ SATISFIED | Live mirror fetch + digest verification confirmed by direct execution during this verification |
| REQ-UX-01 | 10-04, 10-06 | ? NEEDS HUMAN | Cross-repo; no verifier available from tinyml-cli. Phase's own record says checkpoint-driven against the real app, but that pass is not reproducible from here and its own SwiftUI target has zero automated coverage. |
| REQ-UX-02 | 10-09 | ? NEEDS HUMAN | Cross-repo, same constraint; cancel-drops-queue explicitly left unverified by the phase itself |
| REQ-DOC-01 | 10-05, 10-07 | ✓ SATISFIED | `docs/RELEASING.md` exists and substantive; stale "from TI" CLI help corrected; verified live |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `tests/test_build_config.py` | 39, 491-494 | Guard sanctions retired values instead of pinning the current one | 🛑 BLOCKER | Ceiling can silently regress to 5.6x the requirement with a fully green pipeline (CR-01, independently confirmed by mutation) |
| `tests/test_datasets_download.py` | 423-459 | Regression test asserts on an unreachable path | ⚠️ WARNING | Zip-slip containment for `extract_dataset` is effectively untested; underlying `zipfile` protection is plausible but undocumented as guaranteed (CR-02) |
| `.github/workflows/release.yml` | 124 | `continue-on-error` on the Windows `test` job | ⚠️ WARNING | Regression-test failures on Windows (including phase-10's own guards) don't block a release; the `build` job's runtime size/bundle gates are unaffected by this and still run unconditionally on all 3 platforms |
| `mmcli/datasets.py` | 475-487 | Redirect handler locks host, not scheme | ℹ️ INFO | Same-host `http://`/`ftp://` redirects would be followed; bounded impact because sha256 verification still runs on the resulting bytes (WR-05) |
| `mmcli/cli.py` | 2156-2171 | `init --dataset` fetches (up to 56 MB) before validating destination/task compatibility | ℹ️ INFO | Both validations are microsecond-cheap and currently run after the download (WR-01) |
| No `TBD`/`FIXME`/`XXX` markers found | — | — | — | Scanned `mmcli/datasets.py`, `mmcli/cli.py`, `scripts/release_preflight.py`, `scripts/verify_dataset_digests.py`, `docs/RELEASING.md` — clean |

No debt-marker gate violations found in the files scanned.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Binary size within ceiling | `stat -f%z dist/mmcli` | 25,258,768 (ceiling 27,262,976) | ✓ PASS |
| Startup under bound | 3x `time ./dist/mmcli --version` | 7.32 s / 6.27 s / 6.20 s (median 6.27 s, bound 8 s) | ✓ PASS |
| Ceiling test is discriminating (mutation) | mutate `binary_size_ceiling.txt` to `152043520`, run `pytest tests/test_build_config.py -k ceiling` | "2 passed" — unchanged from the correct-value run | ✗ FAIL (this is the evidence for Gap 1) |
| Live mirror + digest gate | `python scripts/release_preflight.py` | `PREFLIGHT PASSED: mirror tag/assets OK, all fetchable digests verified. Safe to build.` exit 0 | ✓ PASS |
| `datasets list --format json` contract | `python -m mmcli datasets list --format json` | Valid JSON, all documented fields present including `cache_bytes` (D-10) | ✓ PASS |
| No stale "from TI" help text | `mmcli datasets --help`, `mmcli init --help` \| grep -i "from TI"` | No matches | ✓ PASS |
| Full phase-10 test suite | `MMCLI_PYTHON=~/.venv-tinyml/bin/python pytest tests/test_cli_integration.py tests/test_tier4_cli.py tests/test_build_config.py tests/test_datasets_download.py tests/test_datasets_cli.py tests/test_ci_workflows.py -q` | 198 passed, 0 failed, 6 warnings, 264.6 s | ✓ PASS |

Note: the file was restored to `27262976` immediately after the mutation check; `git status --short` and `git diff --stat` both confirm zero residual changes in the working tree.

### Human Verification Required

See `human_verification` in frontmatter — both items are REQ-UX-01 and REQ-UX-02, cross-repo in `../PlatypusStudio`. This verifier operates against `tinyml-cli` and cannot drive a SwiftUI app; the phase's own `.continue-here.md` states plainly: "cross-repo requirements (REQ-UX-01/02) have no verifier, since a phase verifier reading `tinyml-cli` cannot inspect `PlatypusStudio`." That is accepted here rather than resolved — these are reported as unverifiable-from-here, not passed or failed.

### Gaps Summary

Two gaps block a clean "passed" verdict, both concentrated in the **verification layer** the code
review already identified — not in the download/extraction implementation itself, which review
and this pass agree is solid (mandatory digest verification, atomic replace, re-hash on cache
hit, MMCLI_DATASETS refusal before URL composition all hold up).

1. **Blocker — the binary-size ceiling gate cannot fail in the direction that matters.**
   `SANCTIONED_CEILINGS` in `tests/test_build_config.py` still sanctions the retired 145 MiB and
   15 MiB ceilings, and the CI size gate in `release.yml` trusts `scripts/binary_size_ceiling.txt`
   with no independent bound. I reproduced this myself: writing `152043520` into the ceiling file
   and running `pytest tests/test_build_config.py -k ceiling` still reports 2 passed. Today's
   binary is well within bound (25,258,768 / 27,262,976 bytes measured live), so REQ-SIZE-01's
   *literal number* is currently true — but the mechanism meant to keep it true going forward is
   not a guard, it is a formality. This is exactly the failure class the phase's own
   `.continue-here.md` names as blocking ("a test that simulates a code path does not test it"),
   and it was caught by the phase's own code review (CR-01), not invented here.

2. **Partial — the zip-slip regression test proves nothing.** The one test claiming to cover
   path-traversal containment for `extract_dataset` asserts on a filesystem location that neither
   a safe extraction nor a real escape would ever write to. `extract_dataset` itself has no
   explicit containment guard — the property is entirely inherited from `zipfile`'s own member-path
   sanitisation, which is correct in current CPython but not a documented stability guarantee.
   This is reachable with attacker-influenced content via `MMCLI_DATASETS`, which is deliberately
   not digest-verified. No live exploit was attempted during this verification (that would exceed
   a verification pass's scope); the gap is that the test cannot currently detect one either way.

Neither gap is deferred to a later phase — Phase 11 (PlatypusStudio run archive/training views) is
unrelated in subject matter, so Step 9b's deferred-item check found no match. Both gaps have
concrete, already-drafted fixes in `10-REVIEW.md` (CR-01, CR-02) ready for a closure plan.

Everything else checked — the download/verify/cache mechanics, the mirror repoint, the wheel/sdist
narrowing, the CLI contract, and the documentation — is substantively implemented, wired, and (for
everything reachable from this repository) live-tested rather than taken on the SUMMARY's word.

---

_Verified: 2026-08-02T23:15:00Z_
_Verifier: Claude (gsd-verifier)_
