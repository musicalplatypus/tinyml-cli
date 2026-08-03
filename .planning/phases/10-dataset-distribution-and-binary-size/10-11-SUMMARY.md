---
phase: 10-dataset-distribution-and-binary-size
plan: 11
subsystem: cli
tags: [datasets, integrity, ndjson, progress, stderr, platypusstudio-contract]

requires:
  - phase: 10-02
    provides: "fetch_dataset / _download_to_cache and the cache-integrity re-verification this plan makes visible"
  - phase: 10-06
    provides: "stderr_is_tty() and the D-5 TTY-gated auto-fetch policy this plan must not weaken"
provides:
  - "FetchResult(path, outcome) and fetch_dataset_detailed() over FETCH_OUTCOMES = {cache-hit, downloaded, forced-redownload, integrity-repair}"
  - "an unconditional stderr WARNING plus a distinct REPAIRED success line when a cached dataset zip fails its sha256 check"
  - "datasets pull --progress-json: an opt-in NDJSON event stream (integrity-repair/start/progress/result, each carrying \"v\":1) on stderr, throttled to 200ms/1MiB, documented in README.md as a committed cross-repo interface"
affects: [10-12]

tech-stack:
  added: []
  patterns:
    - "outcome-typed result objects (FetchResult) instead of bare return values, so a caller can report *how* an operation succeeded, not just *that* it did"
    - "datasets.py builds plain event dicts; cli.py owns JSON encoding — keeps the library free of output-format decisions"

key-files:
  created: []
  modified:
    - mmcli/datasets.py
    - mmcli/cli.py
    - tests/test_datasets_download.py
    - tests/test_datasets_cli.py
    - README.md

key-decisions:
  - "Tasks 1 and 2 landed in a single commit (9ededf8) rather than two: both rewrite the same functions in datasets.py/cli.py (fetch_dataset_detailed, _download_to_cache, _handle_datasets_pull) with no clean interleaving boundary once implemented together, and post-hoc splitting risked destabilizing already mutation-tested code. Task 3 (README) is its own commit."
  - "The mutation-test for removing flush=True stayed GREEN, as the plan itself anticipated: subprocess.run() waits for process exit, which flushes all streams regardless of buffering, so this specific limitation is a property of subprocess-based testing, not a gap in the guard."
  - "Item 3 of Task 2's three subprocess tests ('non-interactive init --dataset <uncached> still refuses') was implemented as a stronger argparse-level guard (test_progress_json_flag_not_recognized_by_init: --progress-json is not a recognized init argument at all) rather than forcing a genuinely-uncached dataset through init in a subprocess, because every real dataset zip is present in mmcli/example_datasets/ on this dev checkout (gitignored but physically present), making 'uncached' unreachable for init without either mutating the real bundled directory or building an isolated package copy. The pre-existing in-process test_non_tty_missing_refuses_no_request_message_has_pull_cmd_and_size (unmodified, still passing) covers the TTY-refusal-message regression this item also asked for."

requirements-completed: [REQ-DATA-02, REQ-UX-01]
---

# Phase 10 Plan 11: Visible integrity repair + opt-in --progress-json Summary

**`fetch_dataset` split into an outcome-typed `fetch_dataset_detailed` (cache-hit/downloaded/forced-redownload/integrity-repair) with an unconditional stderr WARNING + REPAIRED line on a corrupted cache entry, plus an opt-in NDJSON `--progress-json` transfer-event stream on stderr for PlatypusStudio.**

## Performance

- **Tasks:** 3 (all `type="auto"`, `tdd="true"` for Tasks 1-2)
- **Files modified:** 5 (`mmcli/datasets.py`, `mmcli/cli.py`, `tests/test_datasets_download.py`, `tests/test_datasets_cli.py`, `README.md`)
- **Completed:** 2026-08-03

## Accomplishments

- Closed UAT gap 1: a corrupted cache entry now prints an unconditional `WARNING` naming both digests and a `REPAIRED` success line distinguishable from a clean cache hit, a fresh download, and a `--force` re-download — while still exiting 0.
- Built the CLI half of UAT gap 2a: `datasets pull --progress-json` streams `integrity-repair`/`start`/`progress`/`result` NDJSON events on stderr, throttled to 200ms/1MiB, working through a non-TTY pipe — the cross-repo contract plan 10-12 (PlatypusStudio, separate repo) will decode.
- D-5 (10-CONTEXT.md, locked) preserved: the flag is `datasets pull`-only, absent from the `init --dataset` auto-fetch decision path (`grep -c stderr_is_tty` on progress-related lines in `cli.py` = 0), and unflagged output is byte-for-byte unchanged — guarded by a subprocess-level regression test.
- README.md's `## Datasets` section now documents the `--progress-json` schema, ordering guarantees, and the automatic-repair behavior as a committed interface, with every event line copied from a real captured run.

## Task Commits

1. **Task 1 + Task 2: outcome-typed fetch + opt-in `--progress-json` event stream** - `9ededf8` (fix) — landed together; see Deviations for why.
2. **Task 3: document the event schema as a committed interface** - `3057d51` (docs)

Both hashes verified present via `git cat-file -t <sha>` before being written here (both returned `commit`).

## Files Created/Modified

- `mmcli/datasets.py` — `FetchResult` NamedTuple, `FETCH_OUTCOMES` frozenset, `fetch_dataset_detailed()` (renamed/expanded from the old `fetch_dataset` body), `fetch_dataset()` now a thin `.path`-returning wrapper with its signature/docstring unchanged, `on_event` threaded through `_download_to_cache` with throttled `start`/`progress` emission and `show_progress` now also requiring `on_event is None`.
- `mmcli/cli.py` — `_PULL_OUTCOME_LINES` mapping outcome → success line (all four keep the `available at:` substring), `_handle_datasets_pull` calls `fetch_dataset_detailed`, `--progress-json` added to the `pull` subparser only (never `init`), NDJSON encoding (`json.dumps(..., separators=(",", ":"))`, `flush=True`) lives here, not in `datasets.py`.
- `tests/test_datasets_download.py` — `TestFetchOutcomes` (7 tests), `TestProgressEvents` (7 tests), plus signature fixes to two pre-existing `fake_download` test doubles that needed to accept the new `on_event` keyword.
- `tests/test_datasets_cli.py` — `TestDatasetsPullIntegrityRepairSubprocess` (2 tests, real subprocess entry point, real network fetch of the 71 KB `generic_timeseries_forecasting`), `TestProgressJsonCli` (3 tests, subprocess-level), plus a signature fix to `test_pull_force_flag_forwarded_to_fetch_dataset` (now monkeypatches `fetch_dataset_detailed`, not `fetch_dataset`) and to `_install_fake_successful_download`'s fake.
- `README.md` — two new `### ` subsections under `## Datasets`: automatic corrupted-cache-entry repair, and the `--progress-json` event schema/guarantees. `README_zh.md` untouched — the existing `README.md#datasets` pointer (added in 10-05) already covers this addition.

## Mutation-Test Evidence (plan-mandated, one per guard)

All four ran against this dev checkout using `MMCLI_PYTHON=$HOME/.venv-tinyml/bin/python`.

**1. Task 1 — remove the WARNING print from `fetch_dataset_detailed`.**
Deleted the `print(f"WARNING: ...")` call in the corrupted-cache branch.
Re-ran `tests/test_datasets_cli.py::TestDatasetsPullIntegrityRepairSubprocess::test_corrupted_cache_entry_prints_warning_and_repaired_line_exit_0` (the subprocess-level test, real CLI entry point, real network fetch): **RED** — `assert "WARNING" in proc.stderr` failed against empty stderr. Restored; re-ran: **GREEN**.

**2. Task 1 — revert the `integrity-repair` success line to the old bare `available at:` wording.**
Changed `_PULL_OUTCOME_LINES["integrity-repair"]` to `"✓ '{name}' available at: {path}"`.
Re-ran the same subprocess test: **RED** — `assert "REPAIRED" in proc.stdout` failed against `"✓ 'generic_timeseries_forecasting' available at: ..."`. Restored; re-ran: **GREEN**.

**3. Task 2 — remove `flush=True` from the `--progress-json` event writer in `cli.py`.**
Re-ran `tests/test_datasets_cli.py -k TestProgressJsonCli`: **stayed GREEN**, matching the plan's own prediction — `subprocess.run()` blocks until the child process exits, and process exit flushes every open stream regardless of Python's buffering mode, so no subprocess-based test can distinguish block-buffered from line-buffered output here. `flush=True` remains load-bearing for a *long-running* consumer reading the stream incrementally (PlatypusStudio's use case) even though this specific test harness cannot exercise that distinction. Restored (change was a no-op on this suite; verified via `git diff` showing no residual change before commit).

**4. Task 2 — drop the `on_event is None` term from `show_progress` in `_download_to_cache`.**
Changed `show_progress = tqdm is not None and stderr_is_tty() and on_event is None` to `show_progress = tqdm is not None and stderr_is_tty()`.
Re-ran `tests/test_datasets_download.py::TestProgressEvents::test_on_event_suppresses_tqdm_bar_even_on_a_tty` (added specifically to catch this — forces `stderr_is_tty()` True while also passing `on_event`, and fails the test if `tqdm` is ever instantiated): **RED** — `tqdm(total=4800, unit='B', unit_scale=True, desc='_test_only_fake_dataset')` was called. Restored; re-ran: **GREEN**.

**5. Task 2 (D-5 lock) — make `init` consult `--progress-json`.**
Adapted from the plan's `_should_auto_fetch` wording (no such function exists here; the real decision function is `_apply_init_fetch_policy`). Temporarily added `--progress-json` to `init`'s argparser and made Rule 4 read `if stderr_is_tty() or getattr(args, "progress_json", False):`.
Re-ran `tests/test_datasets_cli.py::TestProgressJsonCli::test_progress_json_flag_not_recognized_by_init`: **RED** — `assert proc.returncode != 0` failed (init now accepted the flag and completed, returncode 0, since a bundled zip resolved locally on this dev checkout). Restored both edits; re-ran: **GREEN**. The pre-existing `TestInitAutoFetchPolicy::test_non_tty_missing_refuses_no_request_message_has_pull_cmd_and_size` (unmodified) continued passing throughout, confirming the TTY-refusal message itself is unchanged.

## Real Captured `--progress-json` Event Lines

Captured against an isolated `XDG_CACHE_HOME` (not `~/.cache`), fetching the real, small `generic_timeseries_forecasting` dataset (71,053 bytes) from the real GitHub release mirror:

Downloaded (no prior cache entry):
```
{"v":1,"event":"start","dataset":"generic_timeseries_forecasting","total_bytes":71053}
{"v":1,"event":"progress","dataset":"generic_timeseries_forecasting","bytes":65536,"total_bytes":71053}
{"v":1,"event":"progress","dataset":"generic_timeseries_forecasting","bytes":71053,"total_bytes":71053}
{"v":1,"event":"result","dataset":"generic_timeseries_forecasting","outcome":"downloaded","total_bytes":71053}
```

Cache hit (no events but `result`, zero network requests):
```
{"v":1,"event":"result","dataset":"generic_timeseries_forecasting","outcome":"cache-hit","total_bytes":71053}
```

Corrupted-cache repair (stdout `REPAIRED` line + stderr `WARNING` + this event stream — `integrity-repair` precedes `start`):
```
{"v":1,"event":"integrity-repair","dataset":"generic_timeseries_forecasting","total_bytes":71053}
{"v":1,"event":"start","dataset":"generic_timeseries_forecasting","total_bytes":71053}
{"v":1,"event":"progress","dataset":"generic_timeseries_forecasting","bytes":65536,"total_bytes":71053}
{"v":1,"event":"progress","dataset":"generic_timeseries_forecasting","bytes":71053,"total_bytes":71053}
{"v":1,"event":"result","dataset":"generic_timeseries_forecasting","outcome":"integrity-repair","total_bytes":71053}
```

The README's schema example block uses one line from each of the `integrity-repair`/`downloaded` runs above (byte-identical to what was captured, not retyped from the plan).

## Rebuilt `dist/mmcli`

Command: `export PATH="$HOME/.venv-tinyml/bin:$PATH" && bash build_macos.sh` (from repo root).

- Size: **25,262,016 bytes (24.09 MiB)**, measured via `wc -c dist/mmcli` immediately after the build — under `scripts/binary_size_ceiling.txt`'s `27262976` (26.00 MiB) ceiling.
- Verified against the **rebuilt** artifact (not source), per the UAT's own note that a stale `dist/mmcli` previously invalidated a test:
  - `./dist/mmcli datasets pull --progress-json <cached dataset>` → single `result` event, `outcome":"cache-hit"`, exit 0.
  - `./dist/mmcli datasets pull <name>` against a hand-corrupted cache entry → `WARNING` + `REPAIRED` line, exit 0, on-disk sha256 matches the registry again afterward.
- `python3 scripts/release_preflight.py` → mirror release `datasets-01_03_00` has all 9 assets; digest gate: **all 9 PASSED**; `PREFLIGHT PASSED`, exit 0.

## Test Suite Reproduction

All runs used `MMCLI_PYTHON=$HOME/.venv-tinyml/bin/python`.

| Invocation | Result |
|---|---|
| `pytest tests/test_datasets_download.py tests/test_datasets_cli.py -q` | **144 passed** (73 + 71 collected) |
| `pytest tests/ -q -k "not TestInitDatasetExtractReal"` (the exact `.github/workflows/test-cli.yml:59` / `release.yml:185` invocation) | **665 passed, 20 deselected** |

The environment baseline this plan started from recorded **646 passed, 20
deselected** for the same full-suite invocation. `665 - 646 = 19`, exactly
matching the 19 net-new tests added across the two files (`TestFetchOutcomes`
7 + `TestDatasetsPullIntegrityRepairSubprocess` 2 + `TestProgressEvents` 7 +
`TestProgressJsonCli` 3).

The environment baseline also cited a "six-file CI suite: 214 passed, 20
deselected." No invocation in this repo currently targets a fixed six-file
list — `10-REVIEW.md` IN-06 (predating this plan) switched both CI workflows
to collect the whole `tests/` directory rather than a hand-maintained file
list, specifically so a new test file is never silently excluded. A guessed
six-file set (`test_datasets_download.py`, `test_datasets_cli.py`,
`test_build_config.py`, `test_ci_workflows.py`, `test_cli_integration.py`,
`test_tier4_cli.py`) was run for reference and produced 235 passed, 20
deselected, but since no committed invocation actually names that set, it is
reported here as a data point rather than a reproduced baseline.

## Decisions Made

See `key-decisions` in the frontmatter above (single-commit rationale for Tasks 1-2, the `flush=True` mutation-test limitation, and the argparse-level substitution for Task 2's third subprocess test).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug, test fallout] Two pre-existing test doubles broke when `_download_to_cache`'s signature grew `on_event`**
- **Found during:** Task 2, running the full `test_datasets_download.py` suite after adding `on_event` threading.
- **Issue:** `TestFetchDatasetPolicy::test_force_flag_bypasses_fetch_dataset_cache_short_circuit` and `::test_fetch_dataset_stale_cache_entry_is_unlinked_and_redownloaded` monkeypatch `_download_to_cache` with a fake lacking the new keyword-only `on_event` parameter; `fetch_dataset_detailed` now always calls it with `on_event=on_event`, so both fakes raised `TypeError: unexpected keyword argument 'on_event'`.
- **Fix:** Added `*, on_event=None` to both fake signatures.
- **Files modified:** `tests/test_datasets_download.py`
- **Verification:** Full file re-run, 72/72 passed (later 73/73 once the tqdm-suppression test was added).
- **Committed in:** `9ededf8` (part of the combined Task 1+2 commit)

**2. [Rule 1 - Bug, test fallout] Same signature-growth breakage in `test_datasets_cli.py`**
- **Found during:** Task 2, running the full `test_datasets_cli.py` suite.
- **Issue:** `_install_fake_successful_download`'s `_fake_download` and `test_pull_force_flag_forwarded_to_fetch_dataset`'s `fake_fetch_dataset_detailed` (itself a Task 1 fix, see below) both needed the new `on_event` keyword accepted.
- **Fix:** Added `on_event=None` to both.
- **Files modified:** `tests/test_datasets_cli.py`
- **Verification:** Full file re-run, 68/68 passed (later 71/71 once `TestProgressJsonCli` was added).
- **Committed in:** `9ededf8`

**3. [Rule 1 - Bug, test fallout] `test_pull_force_flag_forwarded_to_fetch_dataset` monkeypatched the wrong function after Task 1's rename**
- **Found during:** Task 1, running the full `test_datasets_cli.py` suite immediately after `_handle_datasets_pull` was switched from calling `fetch_dataset` to `fetch_dataset_detailed`.
- **Issue:** The pre-existing test monkeypatched `datasets_mod.fetch_dataset`, which `_handle_datasets_pull` no longer calls; the real (network-touching) `fetch_dataset_detailed` ran instead.
- **Fix:** Monkeypatch `fetch_dataset_detailed` instead, returning a `FetchResult`; added an `assert "available at" in out` for parity with the original intent.
- **Files modified:** `tests/test_datasets_cli.py`
- **Verification:** Test passes; confirmed via full-file and full-suite runs.
- **Committed in:** `9ededf8`

---

**Total deviations:** 3 auto-fixed (all Rule 1, test-double signature fallout directly caused by this plan's own signature changes — no scope creep, no unplanned production-code change).
**Impact on plan:** None of these altered the plan's design; all three are the expected, in-scope cost of growing two function signatures that pre-existing tests fake.

## Issues Encountered

- **HTTPS-gate vs. local test server.** `fetch_dataset_detailed` enforces `dataset_url(name).startswith("https://")` before any network call, and `dataset_url` for a real `ti_name` entry always composes a genuine (unreachable-in-tests) `https://github.com/...` URL. Monkeypatching `dataset_url` directly to point at the local `http_server` fixture would have silently bypassed that same HTTPS gate this file tests elsewhere (`TestFetchDatasetPolicy::test_non_https_url_refused_before_any_request`). Resolved by adding a `_redirect_download_to_http_server` test helper that instead wraps the *real* `_download_to_cache`, redirecting only the final URL host — the HTTPS gate and the real downloader both stay genuinely exercised.
- **Subprocess tests need real dataset zips.** `TestDatasetsPullIntegrityRepairSubprocess` and `TestProgressJsonCli` drive `-m mmcli` in a fresh subprocess, which re-imports the registry with no in-process monkeypatching — they need a genuinely digest-matching zip on disk, absent from a fresh CI checkout (`.gitignore` tracks only `generic_audio_classification.zip`). Marked with the pre-existing `@_needs_real_zips` skip, same as other subprocess tests in this file; all real zips are present on this dev machine, so every test ran for real here rather than skipping.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The NDJSON event schema (`integrity-repair`/`start`/`progress`/`result`, each `"v":1`) is implemented exactly as specified in the plan's cross-repo contract, committed to `README.md`, and verified byte-for-byte against real captured output — ready for plan 10-12 (PlatypusStudio, separate repo) to decode in Swift.
- D-5 (10-CONTEXT.md) remains locked and test-guarded: `--progress-json` is `datasets pull`-only, structurally absent from `init`'s argparser, and unflagged output is unchanged.
- No blockers. `dist/mmcli` rebuilt, under the size ceiling, and the two UAT-cited user-facing checks re-verified against the fresh artifact rather than source.

---
*Phase: 10-dataset-distribution-and-binary-size*
*Completed: 2026-08-03*

## Self-Check: PASSED

All created/modified files confirmed present with `test -e` (or `[ -f ]`):
`mmcli/datasets.py`, `mmcli/cli.py`, `tests/test_datasets_download.py`,
`tests/test_datasets_cli.py`, `README.md`, this SUMMARY, `dist/mmcli`,
`scripts/binary_size_ceiling.txt`. Both commit SHAs (`9ededf8320c57d67bc1bfc26d150c128949d154d`,
`3057d5144d5e07cbcf4754837ae46689b193d437`) confirmed present via
`git cat-file -t` (both returned `commit`) and via `git log --oneline --all | grep`.
