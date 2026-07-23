---
phase: 10-dataset-distribution-and-binary-size
plan: 03
subsystem: dataset-distribution
tags: [pyinstaller, build-scripts, binary-size, dataset-verification, github-releases, redirect-security]

requires:
  - phase: 10-02
    provides: "fetch_dataset(), dataset_url(), DATASET_REGISTRY digests, _HostLockedRedirectHandler — the fetch/security machinery this plan repoints and relaxes"
  - phase: 10-03-attempt1
    provides: "scripts/verify_dataset_digests.py and the curl-verified proof that the registry sha256 digests are correct — see 10-03-SUMMARY-attempt1-blocked.md"
provides:
  - "mmcli/datasets.py repointed from the dead software-dl.ti.com source to this project's own public GitHub release mirror (datasets-01_03_00), with a narrow exact-host redirect allowlist"
  - "Public release datasets-01_03_00 on musicalplatypus/tinyml-cli carrying the nine mirrored dataset zips, content-verified end to end through the real fetch_dataset() path"
  - "All three build scripts (macOS/Linux/Windows) unbundled to ship only generic_audio_classification.zip via an explicit, source-asserted BUNDLED_DATASETS allowlist"
  - "Lowered binary size ceiling (15,728,640 bytes) and an honest, real macOS build measurement against it"
affects: [10-04, 10-05, 10-07, 10-08]

tech-stack:
  added: []
  patterns:
    - "Closed, exact-host redirect allowlist (ALLOWED_CROSS_HOST_REDIRECTS) as a narrow, auditable relaxation of a host-locked redirect handler, instead of a broader trust rule"
    - "Explicit staged bundling allowlist (BUNDLED_DATASETS -> temp dir -> --add-data) so what a build ships is a property of the build script, not of the developer's working tree"

key-files:
  created: []
  modified:
    - mmcli/datasets.py
    - tests/test_datasets_download.py
    - scripts/verify_dataset_digests.py
    - build_macos.sh
    - build_linux.sh
    - build_windows.ps1
    - scripts/binary_size_ceiling.txt
    - tests/test_build_config.py

key-decisions:
  - "Mirror URL/asset naming (D-A): assets are named by the registry entry's local `filename`, not `ti_name`; dataset_url() composes github.com/musicalplatypus/tinyml-cli/releases/download/datasets-<version>/<filename>."
  - "Versioning (D-B): DATASETS_DEFAULT_VERSION stays 01_03_00 and still keys the on-disk cache path, but now labels the mirror release tag, not a TI engine version."
  - "Redirect relaxation (D-C, amends 10-02 T-10-02-01/05): a closed, exact-host ALLOWED_CROSS_HOST_REDIRECTS map permits only github.com -> release-assets.githubusercontent.com; every other cross-host redirect, including suffix/prefix lookalikes, still raises. sha256 verification stays mandatory."
  - "ti_name kept (D-D): no longer the URL source, but retained as the fetchable-sentinel and TI-provenance record."
  - "Publish (Task 3) was performed by the user/orchestrator directly, not by this agent: Claude Code's auto-mode permission classifier blocked `gh release create`/`gh release upload` for this agent as an irreversible, ~131 MB public-publish action. The agent independently re-verified the published release (read-only `gh release view`) and independently re-ran `scripts/verify_dataset_digests.py` against the live mirror before proceeding to unbundle, rather than trusting the report alone."
  - "Binary size (31.84 MB) and startup time (~6.7-9.6s median) both measured honestly on the real macOS build and both exceed REQ-SIZE-01's 15 MB / 2.5s bounds even after full unbundling. Per explicit instruction, the ceiling was NOT raised and the measurement was NOT massaged to pass — this is flagged as an open follow-up, not silently fixed."

requirements-completed: [REQ-DATA-04, REQ-DATA-05]

duration: ~2h (spans a checkpoint pause for the Task 2 public-publish authorization, confirmed via the orchestrator/user, plus a permission-system block on Task 3's gh commands worked around by the user performing the publish directly)
completed: 2026-07-23
---

# Phase 10 Plan 03: GitHub release mirror repoint, publish, and dataset unbundle Summary

**Repointed `mmcli`'s dead TI dataset fetch to this project's own public GitHub release mirror (`datasets-01_03_00`), published and content-verified all nine dataset zips against the real fetch path, then unbundled all three build scripts down to a single locally-authored dataset — but the resulting macOS binary (31.84 MB, ~7-10s startup) still overshoots REQ-SIZE-01's 15 MB / 2.5s bounds even after unbundling, which is reported here honestly rather than fixed by loosening the ceiling.**

This plan resumes and supersedes the blocked first attempt recorded in
[10-03-SUMMARY-attempt1-blocked.md](./10-03-SUMMARY-attempt1-blocked.md): that attempt built
`scripts/verify_dataset_digests.py`, ran the real gate against `software-dl.ti.com`, got 9/9
`FAIL` (TI's CDN now 302s to `downloads.ti.com`, which `fetch_dataset()`'s security-hardened
redirect handler correctly refused), and independently confirmed via `curl -sL` + sha256 that
all nine registry digests were correct — the failure was TI's infrastructure moving, not a bad
digest or a tampering event. That evidence is preserved as-is in the attempt-1 file, not
discarded, and is the reason this plan mirrors to GitHub rather than re-pointing at TI again.

## Performance

- **Tasks:** 5/5 complete across two agent sessions (a `checkpoint:decision` pause between
  Task 1 and Task 3 for explicit publish authorization)
- **Files modified:** 8

## Accomplishments

- `mmcli/datasets.py` repointed: `dataset_url()` composes
  `https://github.com/musicalplatypus/tinyml-cli/releases/download/datasets-<version>/<filename>`,
  named by the entry's local `filename` (D-A), not `ti_name`. `TI_DATASETS_BASE` removed.
- `_HostLockedRedirectHandler` relaxed with a closed, exact-host
  `ALLOWED_CROSS_HOST_REDIRECTS = {"github.com": frozenset({"release-assets.githubusercontent.com"})}`
  map (D-C); every other cross-host redirect, including lookalike-suffix/prefix hosts, still
  raises `RuntimeError`. This narrowly amends 10-02's threat mitigation T-10-02-01/05.
- Public release `datasets-01_03_00` published on `musicalplatypus/tinyml-cli`
  (https://github.com/musicalplatypus/tinyml-cli/releases/tag/datasets-01_03_00), carrying
  exactly the nine mirrored zips, each server-side asset size matching the registry `bytes`
  field exactly, `generic_audio_classification.zip` correctly absent (D-2).
- `scripts/verify_dataset_digests.py` run for real against the live mirror:
  **all 9 fetchable datasets PASSED, exit 0** — the `github.com` ->
  `release-assets.githubusercontent.com` redirect was followed and every sha256 matched.
- All three build scripts (`build_macos.sh`, `build_linux.sh`, `build_windows.ps1`) now stage
  an explicit `BUNDLED_DATASETS` allowlist (exactly `generic_audio_classification.zip`) into a
  fresh temp directory and `--add-data` that directory — `build_linux.sh` and
  `build_windows.ps1` previously shipped zero datasets at all, so this also fixes REQ-DATA-04
  on those two platforms.
- `scripts/binary_size_ceiling.txt` lowered to `15728640` (15 MB, REQ-SIZE-01).
- Real `bash build_macos.sh` run: binary carries exactly one dataset zip
  (`generic_audio_classification.zip`, verified via PyInstaller archive inspection — none of
  the other nine leaked in despite all ten being present in the developer's working tree), all
  smoke tests pass, but the binary is 31,839,872 bytes and starts in ~6.7-9.6s — both over
  REQ-SIZE-01's bounds, reported honestly below.

## Task Commits

1. **Task 1: Repoint fetch machinery to the GitHub release mirror and narrow the redirect
   rule** — `73ad23f` (feat)
2. **Task 2: Authorize publishing the nine dataset zips as a public release** —
   `checkpoint:decision`, no commit (decision only). Authorized by the user via the
   orchestrator; see "Task 3 delegation" below for why the agent did not create the release
   itself.
3. **Task 3: Publish the mirror release, upload the nine assets, and verify the whole mirror
   end to end** — no repo-file commit (this task creates GitHub-hosted release/assets, not
   repo files, per its own `files_modified: []`). Performed by the user directly (see below);
   independently re-verified by this agent via read-only `gh release view` and a live re-run
   of `scripts/verify_dataset_digests.py`.
4. **Task 4: Unbundle all three build scripts, lower the size ceiling, extend the build-config
   test, and correct the stale comments** — `42b6b3f` (feat)
5. **Task 5: Real macOS build under the new ceiling, with startup, smoke, offline and
   non-TTY-message verification** — `6f7c60a` (docs — records the real measurement in
   `build_macos.sh`'s comment; no build-script logic changed by this commit)

## Task 3 delegation (why the agent did not run `gh release create`/`gh release upload`)

Claude Code's auto-mode permission classifier blocked this agent's `gh release create` call
outright (`"Blocked by classifier"`), independent of any plan-logic decision. Per this agent's
operating rules, a relayed "the user said go ahead" from the orchestrator is not the user's own
direct consent, and the permission system itself had just independently refused the action — so
the agent stopped rather than finding an alternate tool path (`curl` against the GitHub API,
etc.) to route around that refusal. The user performed the publish directly. Before proceeding
to Task 4's unbundle, this agent independently re-confirmed the ground truth rather than taking
the report on faith:

```
$ gh release view datasets-01_03_00 --repo musicalplatypus/tinyml-cli --json name,tagName,isDraft,url,assets
name: Example datasets (engine version 01_03_00)
tag: datasets-01_03_00
draft: False
url: https://github.com/musicalplatypus/tinyml-cli/releases/tag/datasets-01_03_00
arc_fault_classification.zip 13290076
ecg_classification.zip 4651662
fan_blade_fault.zip 56595859
generic_timeseries_anomalydetection.zip 4242845
generic_timeseries_classification.zip 2579940
generic_timeseries_forecasting.zip 71053
generic_timeseries_regression.zip 906660
mnist_image_classification.zip 46993516
pir_detection.zip 1579936
asset count: 9
```

Every size matches the corresponding `DATASET_REGISTRY[...]["bytes"]` value exactly (confirmed
against the local zips too); `generic_audio_classification.zip` is correctly absent.

```
$ MMCLI_PYTHON="$HOME/.venv-tinyml/bin/python" PYTHONPATH="$PWD" ~/.venv-tinyml/bin/python scripts/verify_dataset_digests.py
arc_fault_classification: ...: PASS
ecg_classification: ...: PASS
fan_blade_fault: ...: PASS
generic_timeseries_anomalydetection: ...: PASS
generic_timeseries_classification: ...: PASS
generic_timeseries_forecasting: ...: PASS
generic_timeseries_regression: ...: PASS
mnist_image_classification: ...: PASS
pir_detection: ...: PASS

All 9 fetchable dataset(s) PASSED.
```
Exit code: `0`. This is the same `fetch_dataset(force=True)` code path a real
`mmcli datasets pull` invocation uses, run against the live, public mirror — not an
approximation. The `github.com` -> `release-assets.githubusercontent.com` redirect (the sole
entry in `ALLOWED_CROSS_HOST_REDIRECTS`) was exercised for real on all nine downloads; no other
signed-asset host was observed, so no extension to the allowlist was needed (the plan's
contingency for a different host did not trigger).

## Real macOS build measurement (Task 5) — honest overshoot, not fixed here

```
bash build_macos.sh   # ~/.venv-tinyml, arm64, ~16s build
```
succeeded: 17 mmcli modules bundled, PyInstaller reports build complete.

**Size:** `stat -f%z dist/mmcli` = **31,839,872 bytes (~31.84 MB / 30.36 MiB)** — over the
15,728,640-byte (15 MB) ceiling in `scripts/binary_size_ceiling.txt`, even after the unbundle.
Archive inspection (`PyInstaller.archive.readers.CArchiveReader`) shows the dataset payload is
no longer the weight driver — the binary carries exactly one dataset zip
(`mmcli/example_datasets/generic_audio_classification.zip`, 15,777 compressed bytes) — the
remainder is native library weight that survives `scripts/pyinstaller_excludes.txt`'s
exclusions because these modules are genuinely used in-process:

| Component | Approx. size | Why it's present |
|---|---|---|
| `PYZ.pyz` (zipped Python stdlib + mmcli) | 10.1 MB | Base interpreter payload |
| `cryptography` (`_rust.abi3.so` + `libcrypto`) | 5.7 MB | Pulled in by the requests/urllib3 stack |
| `pandas` (multiple `_libs/*.so`) | ~4 MB | `analyze.py` genuinely uses it |
| `numpy` | ~1.9 MB | `analyze.py` genuinely uses it |
| `PIL` (`.dylibs`: harfbuzz, freetype, jpeg, webp, tiff, openjp2, lcms2) | ~1.9 MB | Image-format libraries pulled in by a dependency |
| `Python.framework`, `libsqlite3`, misc stdlib `.so` | ~2.2 MB | Interpreter runtime |

None of these are datasets and none are excludable without a functional regression (they are
used in-process, unlike the excluded torch/TVM/tinyml_modelmaker engine, which is only ever
invoked out-of-process via `MMCLI_PYTHON`). **REQ-SIZE-01's 15 MB figure appears to have assumed
that removing the training engine plus the datasets would be sufficient; it was not** — the
remaining Python/numpy/pandas/PIL/cryptography floor is roughly double that. This was not
resolved in this session: doing so would mean either excluding `pandas`/`PIL`/`cryptography`
(a functional regression to a working code path, Rule 4 territory) or switching to `--onedir`
mode (an architectural build-mode change, also Rule 4). Per explicit instruction, the ceiling
was **not** raised to make this pass and the discrepancy is flagged here for follow-up instead.

**Startup:** 3-run median, warm and cold, consistently **~6.6-9.6 seconds** — over the 2.5s
REQ-SIZE-01 bound. In every run, wall-clock (`real`) time is far larger than `user`+`sys` time
combined (e.g. `real 7.62 / user 0.41 / sys 2.16`), which is not CPU work — it is consistent
with PyInstaller `--onefile` mode's per-launch extract-to-a-temp-directory overhead on an
ad-hoc-signed, unnotarized binary (`spctl -a -v dist/mmcli` reports `rejected`; no quarantine
xattr is present since the binary was built locally, so execution still succeeds, but Gatekeeper
assessment on the freshly-extracted native `.so`/`.dylib` files may still be adding overhead on
each run). This is the first time REQ-SIZE-01's startup bound has actually been measured (10-01
measured only binary size); the result is reported honestly and was not massaged.

**Smoke tests, all passing:**
- `--version` → `mmcli 1.1.2`
- `init --list` → shows all 10 datasets with correct task types/descriptions
- `datasets path generic_audio_classification` (no network, `MMCLI_DATASETS` unset) → resolves
  to the bundled zip inside the PyInstaller extraction dir, exit 0
- D-5 non-TTY check: `XDG_CACHE_HOME=<empty tmp dir> ./dist/mmcli datasets path fan_blade_fault < /dev/null` →
  exit 1, stderr prints exactly
  `ERROR: 'fan_blade_fault' is not available locally.` /
  `` Run `mmcli datasets pull fan_blade_fault` to fetch it. `` — no download attempt, no
  traceback
- `MMCLI_DATASETS=mmcli/example_datasets ./dist/mmcli datasets list --format json` → all ten
  datasets report state `bundled` (REQ-DATA-04's offline escape hatch), none report `NOT
  OFFLINE`
- Archive inspection confirms the binary embeds exactly `generic_audio_classification.zip` and
  none of the other nine, despite all ten zips being present in the developer's working tree at
  build time — the staged-allowlist design (Task 4) held

`pwsh` is absent on this host; `build_windows.ps1` was verified at the source-assertion level
only (`tests/test_build_config.py`, 30/30 passing, including new assertions specific to the
Windows `;` separator). No Windows or Linux build was performed or claimed.

## Files Created/Modified

- `mmcli/datasets.py` — repointed `dataset_url()` to the GitHub release mirror; added
  `DATASETS_MIRROR_BASE`/`DATASETS_MIRROR_TAG_PREFIX`; added
  `ALLOWED_CROSS_HOST_REDIRECTS` and relaxed `_HostLockedRedirectHandler`; updated docstrings
  (module, `_cache_dir`, `_validate_registry`, `dataset_url`, `fetch_dataset`) to describe the
  mirror instead of TI; removed `TI_DATASETS_BASE`.
- `tests/test_datasets_download.py` — rewrote all stale `software-dl.ti.com` URL-form
  assertions to the mirror shape (including two the plan's own read_first list missed:
  `test_url_uses_default_version`, `test_per_entry_ti_version_override` — rewritten, not
  deleted); added `TestAllowedCrossHostRedirect` (5 tests: allowed pair followed; arbitrary,
  suffix-lookalike, prefix-lookalike, and wrong-origin-host redirects all still refused).
- `scripts/verify_dataset_digests.py` — docstring refresh only (mirror + ~131 MB instead of
  `software-dl.ti.com` + ~125 MB); fetch logic unchanged (source-agnostic).
- `build_macos.sh` / `build_linux.sh` / `build_windows.ps1` — staged `BUNDLED_DATASETS`
  allowlist (`generic_audio_classification.zip` only) replacing the former blanket
  `--add-data` (macOS) or adding one for the first time (Linux/Windows); corrected the stale
  "bundled example datasets are still the largest remaining component" comment; `build_macos.sh`
  additionally carries the real measured-size/startup figures from Task 5.
- `scripts/binary_size_ceiling.txt` — `152043520` → `15728640`.
- `tests/test_build_config.py` — new `TestBuildScriptsBundleOnlyTheOneLocalDataset` class (5
  parametrised tests x 3 scripts): exactly one `--add-data`, platform-correct separator and
  destination, staging-variable source resolved via the `BUNDLED_DATASETS` allowlist, and no
  script ever names any of the nine mirrored zips.

## Decisions Made

See `key-decisions` in the frontmatter above (D-A through D-D per 10-03-PLAN.md, plus the
Task 3 delegation and the honest-overshoot decision for Task 5).

## Deviations from Plan

### Auto-fixed / process deviations

**1. [Checker finding M1 — pre-existing plan gap, fixed in Task 1] Two additional stale URL
assertions the plan's own read_first list missed.**
- **Found during:** Task 1
- **Issue:** `test_url_uses_default_version` and `test_per_entry_ti_version_override` also
  asserted the old `/<version>/datasets/` TI URL form and would have gone red after the
  repoint; the plan's `<read_first>` only named `:217-219`, `:252`, and the redirect test.
- **Fix:** Rewrote both (not deleted) to assert the mirror URL shape; the ti_version-override
  test still guards that override, now against `.../datasets-01_04_00/<filename>`.
- **Files modified:** `tests/test_datasets_download.py`
- **Committed in:** `73ad23f`

**2. [Rule 4 — architectural, resolved by the permission system, not by this agent] Task 3's
`gh release create`/`upload` blocked by the auto-mode classifier.**
- **Found during:** Task 3 (first attempt, this session)
- **Issue:** The classifier refused the agent's `gh release create` call for this
  ~131 MB public-publish action.
- **Resolution:** Per operating rules, an orchestrator relay of "the user said go ahead" is
  not itself consent, and the permission system's own refusal is authoritative — so the agent
  stopped and reported rather than attempting to route around it. The user performed the
  publish directly; the agent then independently re-verified the outcome (read-only
  `gh release view` + a live re-run of the digest gate) before proceeding, rather than trusting
  the report alone.
- **Files modified:** none (no repo files touch Task 3 by design — `files_modified: []`)

**3. [Reported, not auto-fixed — correctly outside Rule 1-3, this is Rule 4 territory] Binary
size and startup time both exceed REQ-SIZE-01 even after full unbundling.**
- **Found during:** Task 5
- **Issue:** `dist/mmcli` measures 31,839,872 bytes (~31.84 MB, over the 15 MB ceiling) and
  starts in ~6.7-9.6s (over the 2.5s bound). The dataset payload is confirmed gone (archive
  inspection shows only the one bundled dataset); the remainder is numpy/pandas/PIL/cryptography
  native libraries genuinely used in-process, plus what looks like PyInstaller `--onefile`
  per-launch extraction overhead on an unnotarized binary.
- **Not fixed here:** Excluding pandas/PIL/cryptography would break `analyze.py` and the
  requests/urllib3 stack (a functional regression); switching to `--onedir` is a build-mode
  change with distribution implications. Both are architectural decisions (Rule 4), not
  auto-fixable within this plan's scope, and the explicit instruction for this task was to
  report honestly rather than adjust the ceiling.
- **Files modified:** `build_macos.sh` (comment records the measurement; ceiling and
  PyInstaller invocation both left unchanged)

---

**Total deviations:** 1 test-coverage auto-fix (Rule 1-class), 1 permission-system-mediated
process deviation (not an auto-fix at all — a hard external stop, correctly not routed around),
1 honestly-reported unresolved overshoot (explicitly not auto-fixed, flagged for follow-up).
**Impact on plan:** No scope creep; the two non-trivial items (Task 3 delegation, size/startup
overshoot) were both handled by stopping and reporting rather than taking an unreviewed
architectural shortcut.

## Issues Encountered

- The permission-system block on `gh release create` (see deviation 2 above) required a
  mid-plan pause and explicit hand-off to the user/orchestrator — resolved by the user
  performing the publish and the agent independently re-verifying before continuing.
- The 15 MB / 2.5s bounds in REQ-SIZE-01 are not achievable with the current dependency set
  (numpy, pandas, PIL, cryptography all genuinely in use) under PyInstaller `--onefile` mode —
  see "Real macOS build measurement" above for the full breakdown. This needs a follow-up
  decision (accept a higher ceiling, trim dependencies, or change build mode) before REQ-SIZE-01
  can be marked satisfied.

## Next Phase Readiness

**Ready for 10-04/10-05/10-07 with one caveat.** The mirror is live, public, and content-verified
(REQ-DATA-05); all ten datasets remain offline-obtainable via `MMCLI_DATASETS` (REQ-DATA-04);
the D-5 non-TTY policy holds on the real unbundled binary. **REQ-SIZE-01 is NOT satisfied** —
the real macOS binary is roughly double the size ceiling and roughly 3x the startup bound, for
reasons unrelated to datasets (native-library weight from in-process dependencies, plus likely
`--onefile` extraction overhead). This should be surfaced to 10-08 (CI release job, which will
otherwise fail its own per-artifact size gate against `scripts/binary_size_ceiling.txt`) and to
whoever owns REQ-SIZE-01 for a decision: raise the ceiling to reflect reality, trim
numpy/pandas/PIL/cryptography usage, or move off `--onefile` mode. `requirements-completed`
above intentionally lists only `REQ-DATA-04`/`REQ-DATA-05`, not `REQ-SIZE-01`.

## Self-Check: PASSED

- FOUND: `mmcli/datasets.py` contains `release-assets.githubusercontent.com` and
  `DATASETS_MIRROR_BASE`
- FOUND: commit `73ad23f` (`git log --oneline --all | grep 73ad23f`)
- FOUND: commit `42b6b3f`
- FOUND: commit `6f7c60a`
- FOUND: `scripts/binary_size_ceiling.txt` reads `15728640`
- FOUND: `build_macos.sh`, `build_linux.sh`, `build_windows.ps1` all contain
  `BUNDLED_DATASETS`/`$BundledDatasets` and no longer contain the stale "largest remaining
  component" comment
- Ran `pytest tests/test_datasets_download.py -q`: 50 passed
- Ran `pytest tests/test_build_config.py -q`: 30 passed
- Ran `bash build_macos.sh` for real: succeeded, `dist/mmcli` = 31,839,872 bytes, archive
  inspection confirms exactly one bundled dataset zip
- Independently re-verified (read-only `gh release view` + live `verify_dataset_digests.py`
  re-run) the Task 3 publish performed by the user: 9 assets, sizes exact, audio zip absent,
  9/9 digest PASS, exit 0

## Threat Flags

None new. `T-10-03-07`'s mitigation (the closed `ALLOWED_CROSS_HOST_REDIRECTS` map) is
implemented and tested exactly as the threat register specifies — both the allow-path and the
refuse-path (including lookalike hosts) are covered by passing tests.

---
*Phase: 10-dataset-distribution-and-binary-size*
*Completed: 2026-07-23*
