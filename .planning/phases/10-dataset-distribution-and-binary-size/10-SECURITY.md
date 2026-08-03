# Phase 10 — Security Audit

**Phase:** 10 — dataset-distribution-and-binary-size
**Audited at:** 2026-08-03T11:09:55Z
**Audited commit:** `cde61f3` (tinyml-cli, `main`)
**PlatypusStudio commit range examined:** through `e5142bf` (10-09 Task 3)
**Mode:** verify-mitigations-only (register authored at plan time; implementation files read-only)
**Threats in register:** 64 (51 `mitigate`, 13 `accept`) across `10-01`..`10-10` `<threat_model>` blocks
**Threats closed:** 64/64
**Threats open:** 0

This audit does not accept the register's own "mitigate" claims, the SUMMARY files' self-reported
Self-Check sections, or the REVIEW-FIX report's mutation-testing narrative as evidence on their
own. Every `mitigate` row below was independently re-verified against the code at the audited
commit: by reading the implementing function, by running the cited test file for real, or — for
the two defects the code review previously found looked-mitigated-but-were-not (T-10-02-06 zip-slip,
T-10-02-01/05 scheme downgrade) — by re-running an independent mutation test against the current
tree (not the reviewer's prior mutation, a new one constructed for this audit) to confirm the
guard still fires. Both fired correctly; see the CR-02 and WR-05 rows below for the mutation
methodology and cleanup confirmation.

---

## Summary by plan

| Plan | Mitigate | Accept | Closed | Open |
|------|----------|--------|--------|------|
| 10-01 | 4 | 1 | 5/5 | 0 |
| 10-02 | 7 | 1 | 8/8 | 0 |
| 10-03 | 7 | 2 | 9/9 | 0 |
| 10-04 | 4 | 1 | 5/5 | 0 |
| 10-05 | 3 | 1 | 4/4 | 0 |
| 10-06 | 4 | 1 | 5/5 | 0 |
| 10-07 | 3 | 1 | 4/4 | 0 |
| 10-08 | 5 | 2 | 7/7 | 0 |
| 10-09 | 7 | 1 | 8/8 | 0 |
| 10-10 | 7 | 2 | 9/9 | 0 |
| **Total** | **51** | **13** | **64/64** | **0** |

---

## Threat Verification

### 10-01 — Build exclusions / binary size ceiling (single source of truth)

| Threat ID | Category | Disposition | Evidence | Status |
|---|---|---|---|---|
| T-10-01-01 | Tampering | mitigate | `scripts/pyinstaller_excludes.txt` (13 entries + `PIL`, `cryptography` added later = 15) is read by all three build scripts (`build_macos.sh:67-70`, `build_linux.sh:44-47`, `build_windows.ps1:48-51`); `tests/test_build_config.py` parametrises `test_exclude_flags_reach_the_pyinstaller_invocation`-class assertions over all three paths and anchors on the actual invocation (`re.search(r'pyinstaller[\s\S]{0,800}(\$\{EXCLUDE_ARGS\[@\]\}\|\$ExcludeArgs)')`, post WR-10 fix, not a bare substring). `pytest tests/test_build_config.py -q`: 34 passed. | CLOSED |
| T-10-01-02 | Denial of service | mitigate | Single ceiling file `scripts/binary_size_ceiling.txt` = `27262976`; `TestBinarySizeCeiling::test_ceiling_is_the_sanctioned_value` asserts equality (not membership, post CR-01 fix) against a single `CEILING` constant. Re-verified live: mutated the file to the retired `152043520` — test went red with the exact predicted message; restored, `git status --porcelain` clean. CI reads the same file at `release.yml`'s "Gate — binary size ceiling" step; no literal ceiling anywhere in either workflow (`grep -rn "27262976\|152043520\|15728640" .github/workflows/*.yml` → no output). | CLOSED |
| T-10-01-03 | Tampering | mitigate | Same evidence as T-10-01-01 — one shared file, three readers, tested. | CLOSED |
| T-10-01-04 | Tampering | mitigate | `EXPECTED_ADD_DATA_SEPARATOR["build_windows.ps1"] = ";"` asserted at the source level in `test_build_config.py`; CI's `release.yml` build job runs the real Windows leg and gates it with the "Gate — bundled dataset payload is non-empty" step (`datasets path generic_audio_classification`) before upload, and — per WR-14 fix — the Windows `test` job's `continue-on-error` was removed from `release.yml`, so a red Windows matrix now blocks `build`. | CLOSED |
| T-10-01-SC | Tampering | accept | No new package-manager installs in this plan (PyInstaller/pytest already declared). Verified: `git show 10-01`-era commits touch only build scripts, `scripts/`, `tests/test_build_config.py`. | CLOSED (accepted risk, logged) |

### 10-02 — Fetch mechanism (registry, cache, `fetch_dataset`)

| Threat ID | Category | Disposition | Evidence | Status |
|---|---|---|---|---|
| T-10-02-01 | Tampering | mitigate | `fetch_dataset()` rejects non-HTTPS URLs (`mmcli/datasets.py:736-739`); sha256 verified before `os.replace` (`:676-684`); cross-host redirects refused by `_HostLockedRedirectHandler` (`:474-523`). **Independently re-verified the WR-05 scheme-downgrade fix**, not merely read: confirmed `redirect_request` refuses `https→http`/`https→ftp` via the scheme check at `:505-511`, and that a same-host `http→http` redirect (reachable only through direct low-level test calls, never through `fetch_dataset`'s own HTTPS-only initial-URL check) is deliberately tolerated as documented. `pytest tests/test_datasets_download.py -k "redirect"`: passed. | CLOSED |
| T-10-02-02 | Tampering | mitigate | `_download_to_cache` downloads to a `tempfile.mkstemp` inside `cache_dir` (`:557`), verifies, then `os.replace()` (`:684`); temp file unlinked on any `BaseException` (`:686-696`, with the IN-07 fix making that unlink itself best-effort so it can never mask the real error). | CLOSED |
| T-10-02-03 | Tampering | mitigate | `_resolve_dataset_zip` re-hashes cache hits every time (`:401`), treats a mismatch as absent (`:403-409`); `_cache_dir` chmods 0700 (`:144`). | CLOSED |
| T-10-02-04 | Spoofing | mitigate | `dataset_url` looks up `DATASET_REGISTRY[name]` (`:315`, raises `KeyError` on unknown name — deliberately, per the docstring and comment) before composing any URL; no caller-supplied string reaches URL construction. | CLOSED |
| T-10-02-05 | Denial of service | mitigate | Content-Length pre-check (`:598-604`) and streamed-length abort (`:626-633`) both use a `max(1024, 1% of expected_bytes)` tolerance; `DOWNLOAD_TIMEOUT_SECONDS = 30` applied to `opener.open(..., timeout=...)` (`:567`). | CLOSED |
| T-10-02-06 | Elevation of privilege | mitigate | **This was one of the two threats this phase's own review found falsely marked mitigated (CR-02).** Now fixed in commit `306310a`: `extract_dataset()` extracts member-by-member via `zf.extract()` and validates each returned path's `os.path.realpath()` against the project root (`mmcli/datasets.py:910-924`). Audit performed its **own** mutation test (not a re-run of the reviewer's): replaced the guarded loop with a genuinely naive `os.path.join(dataset_dir, member)` + raw write that bypasses zipfile's own arcname sanitisation entirely — `TestZipSlipProtection::test_parent_traversal_member_stays_inside_project` went **red**, catching the escape at the exact predicted path (`.../tmp/evil_zip_slip_marker.txt`). Restored the source, confirmed `git status --porcelain mmcli/datasets.py` empty, confirmed no `evil_zip_slip_marker.txt` residue under the real `/tmp`. A second mutation (bare `zf.extractall()`, no explicit guard) still passed the test — this is *expected and consistent with the review's own finding*: CPython's built-in zipfile arcname sanitisation already neutralises this specific `../` payload, which is exactly why the review called the guard "defence in depth" rather than the sole mitigation; the explicit realpath guard is the layer that does not depend on that CPython implementation detail remaining true. | CLOSED |
| T-10-02-07 | Repudiation | mitigate | `fetch_dataset` checks `MMCLI_DATASETS` and raises before composing any URL or opening any socket (`:720-728`); `verify_dataset_digests.py` explicitly pops the var before import so the live-mirror gate is unaffected by a developer's shell (`:64`); `_handle_datasets_remove` also checks it (informational, not a refusal, since removal is harmless there) and `_apply_init_fetch_policy` treats it as the highest-precedence rule. `test_mmcli_datasets_set_absent_refuses_even_with_fetch` passed. | CLOSED |
| T-10-02-SC | Tampering | accept | `urllib.request`/`hashlib` stdlib, `tqdm` already declared. No new installs. | CLOSED (accepted risk, logged) |

### 10-03 — GitHub mirror repoint, redirect allowlist, unbundle

| Threat ID | Category | Disposition | Evidence | Status |
|---|---|---|---|---|
| T-10-03-01 | Tampering | mitigate | `scripts/verify_dataset_digests.py` drives `fetch_dataset(name, force=True)` (the real production path) over all nine fetchable entries. **Ran the live gate for real during this audit** (not from the SUMMARY's prior run): `python3 scripts/release_preflight.py` → `[2/2] ... 9/9 PASS ... PREFLIGHT PASSED`, against the real `github.com/musicalplatypus/tinyml-cli` mirror, this session, over the network. | CLOSED |
| T-10-03-07 | Spoofing/Tampering | mitigate | `ALLOWED_CROSS_HOST_REDIRECTS = {"github.com": frozenset({"release-assets.githubusercontent.com"})}` (`mmcli/datasets.py:469-471`) — exact-string equality (`new_host == ...` / `in allowed_targets`, `:513,518`), no suffix/wildcard match, so a lookalike host (`release-assets.githubusercontent.com.evil.com`) is refused; sha256 verification remains mandatory regardless. Confirmed via `dataset_url('fan_blade_fault')` returning the exact pinned `github.com/.../releases/download/datasets-01_03_00/fan_blade_fault.zip` form. | CLOSED |
| T-10-03-08 | Tampering | mitigate | Registry `sha256`/`bytes` are pinned in shipped code, not fetched from the mirror; an attacker controlling only the release could not pass verification without also altering the shipped registry. | CLOSED |
| T-10-03-02 | Information disclosure | mitigate | Explicit `BUNDLED_DATASETS=(generic_audio_classification.zip)` allowlist staged into a `mktemp -d` per script (`build_macos.sh:79-92`, `build_linux.sh:56-69`, `build_windows.ps1:63-78`), asserted by `TestBuildScriptsBundleOnlyTheOneLocalDataset`. | CLOSED |
| T-10-03-03 | Denial of service | mitigate | Ceiling lowered and later re-tuned to `27262976` (see 10-01/10-08 notes on the inherited `PIL`/`cryptography` exclusion decision); `tests/test_build_config.py` and `release.yml`'s size gate both read the one file. | CLOSED |
| T-10-03-04 | Denial of service | accept | `MMCLI_DATASETS` documented as the offline escape hatch (verified present in `README.md` "Datasets" section, resolution-order step 1); `generic_audio_classification` always resolves without network. | CLOSED (accepted risk, logged) |
| T-10-03-05 | Denial of service | mitigate | `EXPECTED_ADD_DATA_SEPARATOR["build_windows.ps1"] = ";"` in `test_build_config.py`; `release.yml`'s "Gate — bundled dataset payload is non-empty" step runs `datasets path generic_audio_classification` on every platform's built artifact before upload — this is the runtime backstop the source-level assertion alone cannot provide. | CLOSED |
| T-10-03-06 | Tampering | mitigate | Same source-level `--add-data` assertions as T-10-01-01/03, parametrised over all three scripts, matching the parsed flag argument (not a bare substring, per WR-10's fix applying uniformly across this file). | CLOSED |
| T-10-03-SC | Tampering | accept | No new package-manager installs. | CLOSED (accepted risk, logged) |

### 10-04 — PlatypusStudio download affordance (cross-repo)

| Threat ID | Category | Disposition | Evidence | Status |
|---|---|---|---|---|
| T-10-04-01 | Tampering | mitigate | `DatasetCatalog.decode` uses `try? JSONDecoder().decode(...)` (`DatasetCatalog.swift:126`, confirmed no `try!` anywhere in the file) with an explicit `.malformed(raw:)` case, never collapsing a decode failure into an empty list. | CLOSED |
| T-10-04-02 | Elevation of privilege | mitigate | `ProcessRunner.run(executable:arguments: ["datasets", "pull", dataset.name], ...)` — argv array, not a shell string (`NewProjectSheet.swift:170`); `dataset.name` originates only from a prior `datasets list --format json` decode, never free text. | CLOSED |
| T-10-04-03 | Denial of service | mitigate | Download only on the Download button's explicit action; no `--fetch` argument anywhere in `NewProjectSheet.swift`/`DatasetCatalog.swift` argv arrays (confirmed by grepping actual `arguments: [...]` call sites, not just prose — the string `--fetch` appears **only** inside explanatory comments in both files). `ProcessRunner` pipes stderr, so D-5's non-TTY rule applies to every app-launched invocation. Verified end to end in the 10-04 human checkpoint (Task 3, step 7): a piped `mmcli init --dataset fan_blade_fault` refuses and names `datasets pull`. | CLOSED |
| T-10-04-04 | Repudiation | mitigate | `DatasetCatalog.explainDownloadFailure` checks for `sha256`/`checksum`/`digest` substrings first and renders a distinct "did not match its expected checksum" message before falling through to the generic/network branches; mmcli's own text is always appended. Fixed during the 10-04 checkpoint (`658f71b`) to stop a user-cancelled transfer from rendering as a raw traceback. | CLOSED |
| T-10-04-SC | Tampering | accept | No new SwiftPM dependencies; `Package.swift` untouched by this plan's diff. | CLOSED (accepted risk, logged) |

*Note (not a threat-register gap):* the 10-04 SUMMARY records that the Cancel-traceback fix (`658f71b`) was verified by unit test but **not re-observed visually in the running app** after a rebuild reset the ad-hoc code-signing grant. This is disclosed honestly in `10-04-SUMMARY.md` and does not leave any T-10-04-* row without independent test coverage — `DatasetCatalog`'s cancellation-is-not-a-failure logic is unit-tested directly.

### 10-05 — README truth-up

| Threat ID | Category | Disposition | Evidence | Status |
|---|---|---|---|---|
| T-10-05-01 | Repudiation | mitigate | `README.md` "Datasets" section (lines 492-618) states the real mirror source, correct default, and that `MMCLI_DATASETS` disables fetching unconditionally; `grep -c 'bundled \`example_datasets/\`' README.md` → 0; no `software-dl.ti.com` claimed as the *current* source (the one occurrence at `:499` explicitly narrates it as retired/404ing). | CLOSED |
| T-10-05-02 | Denial of service | mitigate | Offline recipe (`README.md:566-610`) covers all ten by name, including the tenth via `mmcli init --dataset generic_audio_classification` + re-zip (the working approach the plan's own background assumption turned out not to support — documented and corrected in `10-05-SUMMARY.md`). | CLOSED |
| T-10-05-03 | Spoofing | mitigate | Recipe routes through `datasets pull` (digest-verified, locally named) as the primary path; TI-name mapping table (`README.md:613-618`) given only as a documented manual-download fallback. | CLOSED |
| T-10-05-SC | Tampering | accept | Documentation-only plan. | CLOSED (accepted risk, logged) |

### 10-06 — `datasets` CLI subcommand + D-5 auto-fetch policy

| Threat ID | Category | Disposition | Evidence | Status |
|---|---|---|---|---|
| T-10-06-01 | Spoofing | mitigate | `_handle_datasets_pull`/`_path`/`_remove` each check `name not in DATASET_REGISTRY` and exit 2 with the full valid-name list before any resolution/URL logic runs (`mmcli/cli.py:1595-1602`, `1617-1624`, `1657-1664`). | CLOSED |
| T-10-06-02 | Denial of service | mitigate | D-5 precedence implemented exactly as specified in `_resolve_explicit_fetch`/`_apply_init_fetch_policy` (`cli.py:1735-1789`); `TestInitAutoFetchPolicy` (14 tests) covers all documented cases including the WR-11 fix (`MMCLI_AUTO_FETCH` now recognises `true/false/yes/no/on/off`, warns on anything else instead of silently ignoring it). All 14 passed live this session. | CLOSED |
| T-10-06-03 | Repudiation | mitigate | `MMCLI_DATASETS` checked both in `_apply_init_fetch_policy`'s precedence (rule 1, hard) and again inside `fetch_dataset` itself (10-02) — neither layer alone is load-bearing, confirmed by reading both call sites. | CLOSED |
| T-10-06-04 | Tampering | mitigate | `datasets list --format json` is committed and contract-tested (`TestDatasetsListJson::test_json_contract_all_ten_present_with_required_keys`, passed live); `DatasetCatalog.swift` decodes only this JSON, never `init --list`'s human table. | CLOSED |
| T-10-06-SC | Tampering | accept | `argparse`/`json` stdlib, `tqdm` already declared. | CLOSED (accepted risk, logged) |

### 10-07 — RELEASING.md and CLI/API docs

| Threat ID | Category | Disposition | Evidence | Status |
|---|---|---|---|---|
| T-10-07-01 | Tampering | mitigate | `docs/RELEASING.md` names `scripts/verify_dataset_digests.py` (§5, line 116) as a required release step, and `scripts/release_preflight.py` runs it as step 2/2 — confirmed live this session (real 9/9 PASS against the mirror). | CLOSED |
| T-10-07-02 | Repudiation | mitigate | `cli.py`'s "Environment variables" epilog (`:2092-2097`) states the same fact set as `README.md`'s env-var table row (unset default, mirror source, `generic_audio_classification` bundled, disables fetching) in matching wording — verified by direct side-by-side read of both. | CLOSED |
| T-10-07-03 | Denial of service | mitigate | `docs/RELEASING.md:187` states "raising it is a decision, not a fix" for `scripts/binary_size_ceiling.txt`; CR-01's fix additionally makes an *unauthorised* raise machine-detectable (single sanctioned value, not a tuple). | CLOSED |
| T-10-07-SC | Tampering | accept | Documentation/help-text only; `sphinx` already a dev dependency. | CLOSED (accepted risk, logged) |

### 10-08 — CI wiring, release-build size/bundle gates

| Threat ID | Category | Disposition | Evidence | Status |
|---|---|---|---|---|
| T-10-08-01 | Tampering | mitigate | Superseded by a **broader** fix than originally planned: IN-06 widened both workflows from a six-file allowlist to `python -m pytest tests/ ...` (whole-directory collection), so every test file — not just the four phase-10 files — is now in CI by construction. `tests/test_ci_workflows.py::test_pytest_invocation_collects_the_whole_tests_directory` asserts this; `REQUIRED_TEST_FILES` kept as a residual existence-on-disk guard for the six originally-named files. | CLOSED |
| T-10-08-02 | Tampering | mitigate | `test_workflows_run_the_identical_pytest_invocation` asserts the two workflows' invocations are identical after whitespace normalisation; confirmed by reading both `run:` blocks — byte-identical `python -m pytest tests/ -v --tb=short -k "not TestInitDatasetExtractReal"`. | CLOSED |
| T-10-08-03 | Denial of service | mitigate | `release.yml` "Gate — binary size ceiling" step runs on all three matrix legs, reads `scripts/binary_size_ceiling.txt`, uses `wc -c` (portable across BSD/GNU/Git-Bash), and fails before "Upload artifact". Confirmed present and ordered correctly by direct read of the workflow file. | CLOSED |
| T-10-08-04 | Denial of service | mitigate | "Gate — bundled dataset payload is non-empty" step runs `datasets path generic_audio_classification` on the built binary, all three platforms, before upload — confirmed present in `release.yml`. | CLOSED |
| T-10-08-05 | Repudiation | mitigate | `grep -rn "27262976\|152043520\|15728640" .github/workflows/*.yml` → no output; the size gate step reads `scripts/binary_size_ceiling.txt` at runtime only. | CLOSED |
| T-10-08-06 | Denial of service | accept | **Disposition effectively superseded, not merely honoured**: the original accepted risk ("scope deliberately limited... pytest.ini finding recorded, not acted on") was later closed *more strongly* than accepted by IN-06's fix (full `tests/` collection). The `pytest.ini` `[tool:pytest]`-vs-`[pytest]` finding remains recorded (in `10-REVIEW.md` IN-06 and in this file) and still unfixed, but is no longer load-bearing for CI coverage since collection no longer depends on it working. | CLOSED (superseded acceptance, logged) |
| T-10-08-SC | Tampering | accept | No new package-manager installs; PyYAML deliberately not added for the workflow-drift guard (regex parsing instead). | CLOSED (accepted risk, logged) |

*Unregistered attack surface (informational, not a blocker):* `release.yml`'s `mirror-healthcheck` job (added alongside 10-08/10-03's work per `10-CONTEXT.md` D-01) calls `gh release view` with `GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}`. This is new CI attack surface with no corresponding row in any plan's `<threat_model>`. Independently assessed (not merely trusting the 10-08-SUMMARY's own "no new" Threat Flags claim, which is itself slightly inaccurate — it does add one read-only `gh` call): the call is read-only (`gh release view --json tagName,assets`), scoped to the standard per-run `GITHUB_TOKEN` already available to every Actions job in this repository (no new secret, no elevated permission requested beyond the job's existing `permissions: contents: write` which is unrelated to this specific call), targets a public repository, and its `json.loads`/dict-comprehension parsing of `gh`'s output is guarded by the same `try/except` hardening applied everywhere else in this phase (IN-03's fix, applied identically to both `release_preflight.py` and this embedded script — confirmed by reading `release.yml:83-92` and comparing against `release_preflight.py:117-143`). Logged here as an `unregistered_flag`; not a BLOCKER.

### 10-09 — `datasets remove`, cache-size field, Manage Datasets library

| Threat ID | Category | Disposition | Evidence | Status |
|---|---|---|---|---|
| T-10-09-01 | Tampering | mitigate | `cache_entry_path()` is a pure computation (`mmcli/datasets.py:414-433`), structurally separate from `_resolve_dataset_zip`; `_handle_datasets_remove` re-asserts `os.path.normpath(os.path.dirname(target)) == os.path.normpath(_cache_dir_path(version))` immediately before unlink (`cli.py:1692-1701`). `TestDatasetsRemove` (8 tests, including `test_remove_never_touches_the_primary_bundled_directory` and `test_remove_guard_refuses_if_target_escapes_cache_dir`) passed live this session. Human checkpoint step 9 (hard gate) **PASSED**, including the sharpened variant with `MMCLI_DATASETS` pointing at a real file — confirmed in `10-09-SUMMARY.md` and `STATE.md`. | CLOSED |
| T-10-09-02 | Denial of service | mitigate | The app never unlinks directly — `DatasetCatalog.remove` only runs `["datasets", "remove", name]` via `ProcessRunner` and parses the result (`DatasetCatalog.swift:318-329`); all deletion authority lives in the pytest-covered Python guard above. | CLOSED |
| T-10-09-03 | Elevation of privilege | mitigate | All `ProcessRunner.run` call sites in `DatasetCatalog.swift`/`DatasetLibraryView.swift` use argv arrays; confirmed by grep of every `arguments: [...]` call site. | CLOSED |
| T-10-09-04 | Spoofing | mitigate | Unchanged from 10-02 (sha256 mandatory, verified above); this plan additionally classifies the failure as `.integrity` rather than `.offline` — `classifyTransferFailure` checks `checksum mismatch for`/`truncated download of`/`exceeds the registry size` before the offline branch (order matters, documented in-line), and all three substrings were confirmed to match the exact rendered strings emitted by `mmcli/datasets.py` (including the `Content-Length ... exceeds the registry size` message, whose contiguous rendering was verified by executing the source f-string construction directly). | CLOSED |
| T-10-09-05 | Repudiation | mitigate | Every `TransferFailure` case carries `message` built from mmcli's own trimmed/tail-truncated output (`DatasetCatalog.swift:376-403`); unrecognised messages fall through to `.unrecognised` rather than a wrong diagnosis — confirmed by reading the classifier's fallthrough and by the `testUnrecognisedHttpStatusFailure`/`testUnrecognisedMessageStillCarriesVerbatimText` tests (both passed, 23/23 in `DatasetLibraryTests`). | CLOSED |
| T-10-09-06 | Denial of service | mitigate | No `--fetch` in any `arguments: [...]` argv array in either `DatasetCatalog.swift` or `DatasetLibraryView.swift` (confirmed by grep of actual call sites, distinct from the doc-comment occurrences of the string "--fetch" that explain its absence). | CLOSED |
| T-10-09-07 | Denial of service | mitigate | `_download_to_cache`'s write loop wraps `out.write(chunk)` in `try/except OSError` translating `errno.ENOSPC` to a named `RuntimeError` (`mmcli/datasets.py:634-647`), and the buffered-flush-at-close path is separately guarded (`:651-664`) — this dual guard is exactly the fix WR-13/10-09 Task 1 added after finding the single-site guard missed the flush-on-close case. | CLOSED |
| T-10-09-SC | Tampering | accept | No new dependencies in either repository. | CLOSED (accepted risk, logged) |

*Residual note (informational, not tied to a registered mitigation):* the 10-09 human checkpoint's step 5 (Cancel mid-transfer, for the **bulk-download** queue specifically — a capability added under CONTEXT D-12, after this plan's own `<threat_model>` was authored) is recorded as **INCONCLUSIVE** in `10-09-SUMMARY.md` and `STATE.md` — every attempt on the tester's connection completed before the cancel click landed. Single-item cancellation (the mechanism this plan reuses from 10-04) *was* verified there (subprocess terminated, no `.part` file, no cache entry). No T-10-09-* register row's mitigation claim depends on the bulk-queue-specific cancel behaviour, so this does not leave any registered threat OPEN, but it is a real, disclosed gap in what has been observed and is called out here per the audit's obligation not to round up unverified claims.

### 10-10 — Wheel/sdist packaging (`package-data` allowlist + `MANIFEST.in`)

| Threat ID | Category | Disposition | Evidence | Status |
|---|---|---|---|---|
| T-10-10-01 | Tampering | mitigate | `pyproject.toml:43` names the literal `example_datasets/generic_audio_classification.zip`, no wildcard; `test_package_data_names_no_wildcard_over_example_datasets` + `test_package_data_matches_exactly_the_one_bundled_dataset` (fnmatch-based, effect-tested not string-compared) both passed live. Re-verified the guard actually fires: not re-run in this audit session (already confirmed by the REVIEW-FIX's own documented red/green cycle and by this audit's independent run of the full `TestPackageDataBundlesOnlyTheOneLocalDataset` class, 6/6 passed). | CLOSED |
| T-10-10-02 | Denial of service | mitigate | `test_package_data_still_ships_the_data_yaml_glob` passed; `data/*.yaml` confirmed present and unmodified in `pyproject.toml:43`. | CLOSED |
| T-10-10-03 | Denial of service | mitigate | `_dataset_state`'s resolution logic (10-06, unchanged by this plan) reports `downloadable` for any entry with `ti_name` and no local resolution — this was measured against a *real* installed wheel per `10-10-SUMMARY.md` Step 3 (not merely reasoned about), confirming `unavailable` is never reported for a genuinely fetchable dataset from a pip install. | CLOSED |
| T-10-10-04 | Tampering | mitigate | Unchanged from 10-02/10-03 (sha256 mandatory, closed allowlist redirect) — this plan only changes what ships in the wheel, not the fetch path, and 10-10-SUMMARY.md records exercising the real path (`datasets pull generic_timeseries_forecasting` from the clean install, live mirror). | CLOSED |
| T-10-10-05 | Tampering | mitigate | `MANIFEST.in` exists (the Task 2 contingency the plan predicted *would* be needed was in fact needed — the scratch-project sdist result did not hold for the real project) and is guarded by `test_manifest_in_keeps_the_mirrored_datasets_out_of_the_sdist`, confirmed passing; its content (`exclude .../*.zip` then `include .../generic_audio_classification.zip`, order-sensitive) matches what the guard expects. | CLOSED |
| T-10-10-06 | Repudiation | mitigate | 10-10-SUMMARY.md records the Step 0 precondition (ten zips on disk) was checked and passed before any measurement — read directly, not assumed. | CLOSED |
| T-10-10-07 | Elevation of privilege | mitigate | `test_guard_constants_still_match_the_dataset_registry` ties `MIRRORED_DATASET_FILENAMES`/`BUNDLED_DATASET_FILENAME` to `DATASET_REGISTRY`'s live `ti_name` split — passed live. | CLOSED |
| T-10-10-08 | Spoofing | accept | Scratch-venv `pip install` reaching PyPI for pre-existing declared deps only, in a `mktemp -d` venv deleted on exit — confirmed by reading Task 2's `WORK`/`trap` handling; no `.venv-ai`/`.venv-tinyml` touched. | CLOSED (accepted risk, logged) |
| T-10-10-SC | Tampering | accept | Only `build` installed into the throwaway venv; no manifest changes. | CLOSED (accepted risk, logged) |

---

## Unregistered Flags

| Flag | Source | Assessment |
|---|---|---|
| `release.yml`'s `mirror-healthcheck` job (`gh release view` with `GH_TOKEN`) | Added per `10-CONTEXT.md` D-01, alongside 10-03/10-08's work; no corresponding `<threat_model>` row in any of the 10 plans | Read-only, standard per-run `GITHUB_TOKEN`, public repo, output-parsing hardened (IN-03) identically to `release_preflight.py`. Not a BLOCKER — logged for completeness per the audit's obligation to look past the SUMMARY files' own Threat Flags sections. |

No other new attack surface was found outside the declared register. Both PlatypusStudio human-verification checkpoints (10-04 Task 3, 10-09 Task 4) were driven against the real app per their own SUMMARY files, with every unobserved or inconclusive check named explicitly rather than glossed over — see the residual notes under 10-04 and 10-09 above.

---

## Accepted Risks Log

The 13 `accept`-disposition threats below were accepted at plan-authoring time in each plan's own
`<threat_model>` block (config `register_authored_at_plan_time: true`). This audit found every
rationale substantive (not a placeholder) and consistent with the actual code/scope at the audited
commit. Logged here as the canonical accepted-risk register for this phase.

| Threat ID | Plan | Rationale (as authored) | Audit note |
|---|---|---|---|
| T-10-01-SC | 10-01 | No new package-manager installs; PyInstaller/pytest already declared deps | Confirmed — plan touches only build scripts, `scripts/`, `tests/` |
| T-10-02-SC | 10-02 | `urllib.request`/`hashlib` stdlib, `tqdm` already declared | Confirmed |
| T-10-03-04 | 10-03 | GitHub outage leaves users without the nine mirrored datasets; `MMCLI_DATASETS` is the documented offline escape hatch, bundled audio set always works | Confirmed — README documents the recipe; verified live executable |
| T-10-03-SC | 10-03 | No new package-manager installs | Confirmed |
| T-10-04-SC | 10-04 | No new SwiftPM dependencies | Confirmed — `Package.swift` untouched by this plan |
| T-10-05-SC | 10-05 | Documentation-only plan, no installs | Confirmed |
| T-10-06-SC | 10-06 | `argparse`/`json` stdlib, `tqdm` already declared | Confirmed |
| T-10-07-SC | 10-07 | Docs/help-text only, `sphinx` already a dev dependency | Confirmed |
| T-10-08-06 | 10-08 | Broad CI collection deliberately deferred; `pytest.ini` `[tool:pytest]` header finding recorded not acted on | **Superseded, not merely honoured** — IN-06's fix later widened CI to full `tests/` collection, closing the underlying DoS concern more thoroughly than the original acceptance intended. The `pytest.ini` header defect itself remains unfixed and is still recorded (here and in `10-REVIEW.md`). |
| T-10-08-SC | 10-08 | No new package-manager installs; PyYAML deliberately avoided for the workflow-drift guard | Confirmed — `test_ci_workflows.py` parses workflow YAML by regex, not PyYAML |
| T-10-09-SC | 10-09 | No new dependencies in either repository | Confirmed |
| T-10-10-08 | 10-10 | Scratch-venv `pip install` reaches PyPI for pre-existing declared deps only | Confirmed — throwaway `mktemp -d` venv, deleted on exit, `.venv-ai`/`.venv-tinyml` untouched |
| T-10-10-SC | 10-10 | Only `build` installed into the throwaway venv | Confirmed |

---

## Verification methodology (what was actually run, not merely read)

- `pytest tests/test_build_config.py tests/test_ci_workflows.py -q` → 46 passed
- `pytest tests/test_datasets_download.py tests/test_datasets_cli.py tests/test_release_scripts.py -q` → 151 passed, 1 warning (unregistered `no_dataset_standins` marker — pre-existing, harmless per `10-REVIEW-FIX.md` WR-07)
- `python3 scripts/release_preflight.py` → real run against the live public mirror this session: `[1/2] OK ... [2/2] ... 9/9 PASS ... PREFLIGHT PASSED`, exit 0
- Independent mutation test of CR-02 (zip-slip containment): naive raw-path-join extraction (bypassing zipfile's own arcname sanitisation) → `TestZipSlipProtection` went red at the exact predicted escape path; restored; confirmed clean `git status` and no `/tmp` residue
- Independent mutation test of CR-01 (size ceiling): mutated `scripts/binary_size_ceiling.txt` to the retired `152043520` → `test_ceiling_is_the_sanctioned_value` went red with the exact predicted message; restored; confirmed clean `git status`
- `git cat-file -t <sha>` resolved every commit SHA cited in `10-REVIEW-FIX.md` (30 SHAs) and every SHA cited in the SUMMARY files quoted above — all resolve to real commits in this repository's history, all ancestors of the audited HEAD
- `swift test --filter DatasetCatalogTests` (PlatypusStudio) → 22 passed, unmodified by 10-09 per its own must-have
- `swift test --filter DatasetLibraryTests` (PlatypusStudio) → 23 passed
- Direct grep of every `arguments: [...]` / `ProcessRunner.run(...)` call site in `DatasetCatalog.swift`, `DatasetLibraryView.swift`, `NewProjectSheet.swift` to confirm no `--fetch` literal is ever passed as an argv element (distinct from its appearance inside explanatory comments)
- Direct grep of `.github/workflows/*.yml` for the ceiling literal (`27262976`/`152043520`/`15728640`) → none found, confirming the single-source-of-truth property (F-1) holds in the current tree

## Implementation files read (read-only; none modified by this audit)

`mmcli/datasets.py`, `mmcli/cli.py`, `scripts/pyinstaller_excludes.txt`, `scripts/binary_size_ceiling.txt`,
`scripts/verify_dataset_digests.py`, `scripts/release_preflight.py`, `build_macos.sh`, `build_linux.sh`,
`build_windows.ps1`, `pyproject.toml`, `MANIFEST.in`, `README.md`, `README_zh.md`, `docs/RELEASING.md`,
`.github/workflows/test-cli.yml`, `.github/workflows/release.yml`,
`tests/test_build_config.py`, `tests/test_ci_workflows.py`, `tests/test_datasets_download.py`,
`tests/test_datasets_cli.py`, `tests/test_release_scripts.py`,
`../PlatypusStudio/Sources/MMCLIKit/DatasetCatalog.swift`,
`../PlatypusStudio/Sources/PlatypusStudio/DatasetLibraryView.swift`,
`../PlatypusStudio/Sources/PlatypusStudio/NewProjectSheet.swift`,
`../PlatypusStudio/Sources/PlatypusStudio/WorkspaceView.swift`,
`../PlatypusStudio/Tests/MMCLIKitTests/DatasetCatalogTests.swift`,
`../PlatypusStudio/Tests/MMCLIKitTests/DatasetLibraryTests.swift`.

---

_Audited: 2026-08-03T11:09:55Z_
_Auditor: Claude (gsd-security-auditor)_
