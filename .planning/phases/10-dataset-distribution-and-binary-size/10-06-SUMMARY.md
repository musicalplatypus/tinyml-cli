---
phase: 10-dataset-distribution-and-binary-size
plan: 06
subsystem: dataset-distribution
tags: [cli, argparse, json-contract, tty-detection, datasets]

requires:
  - phase: 10-02
    provides: "fetch_dataset(), dataset_url(), _resolve_dataset_zip(), stderr_is_tty() — the fetch mechanism this plan exposes as a CLI surface"
provides:
  - "mmcli datasets list --format json — cross-repo contract (name/version/state/bytes + descriptive fields), all 10 datasets"
  - "mmcli datasets pull <name> [--force] — thin wrapper over fetch_dataset(), no reimplemented download logic"
  - "mmcli datasets path <name> — prints resolved path or exits non-zero"
  - "D-5 auto-fetch policy for init --dataset: --fetch/--no-fetch flags, MMCLI_AUTO_FETCH env var, TTY-gated default"
affects: [10-04, 10-05, 10-07, 10-08]

tech-stack:
  added: []
  patterns:
    - "State classification (_dataset_state) computed by calling 10-02's _resolve_dataset_zip() directly and comparing the returned directory, rather than re-deriving the MMCLI_DATASETS/bundled/cache precedence a second time"
    - "Policy helper (_apply_init_fetch_policy) documents its 4-rule precedence in its own docstring in the literal decision order, per the plan's explicit instruction"
    - "Successful fetch simulated in tests by monkeypatching dataset_url/_download_to_cache to copy an already-verified bundled zip, rather than fabricating bytes or standing up TLS"

key-files:
  created:
    - tests/test_datasets_cli.py
  modified:
    - mmcli/cli.py

key-decisions:
  - "JSON contract shape: {\"datasets\": [...]} — a dict wrapper around the list, not a bare array — emitted via mmcli/output.py's existing format_json(). Each record carries name, version, state, bytes, task_types, module, description. This exact key set and state vocabulary is documented below for 10-04 to decode."
  - "state is computed from _resolve_dataset_zip()'s return path, not re-derived: if the resolved path's directory equals _datasets_dir()'s current value, state is 'bundled'; otherwise (must be the cache) 'cached'. If nothing resolves: 'unavailable' when MMCLI_DATASETS is set (fetch_dataset refuses unconditionally, so it is not really fetchable regardless of ti_name) or the entry has no ti_name; 'downloadable' otherwise."
  - "version is null for entries with no ti_name (generic_audio_classification) since no TI version pin applies to it — the JSON contract test only requires the key be present, not non-null."
  - "Split the plan's two tasks into two commits despite both touching the same two files: Task 1 (list/pull/path) was committed as a real intermediate state with only its own 17 tests, then Task 2 (D-5 policy) was layered on top and committed with its 7 additional tests — each stage verified green before committing, following the pattern 10-02 established."
  - "_apply_init_fetch_policy is skipped entirely (not just short-circuited) when the dataset already resolves locally, verified by a test that monkeypatches the policy function itself to fail if called — proving 'no policy involvement' for already-present datasets is structural, not incidental."

patterns-established:
  - "Any new mmcli subcommand that needs to know whether it may fetch (rather than just whether a file happens to exist) should reuse stderr_is_tty() and the same 4-rule precedence rather than inventing a new heuristic."

requirements-completed: [REQ-DATA-01, REQ-DATA-03, REQ-UX-01]

duration: 100min
completed: 2026-07-22
---

# Phase 10 Plan 06: `mmcli datasets` subcommand + D-5 auto-fetch policy Summary

Added `mmcli datasets list/pull/path` (a committed JSON interface for PlatypusStudio) and the D-5 decision that only `init --dataset` may auto-fetch, and only when stderr is a TTY or `--fetch` is explicit.

## Performance

- **Duration:** ~100 min
- **Started:** 2026-07-22 (session start, after reading 10-02/10-04/PLAN.md context)
- **Completed:** 2026-07-22
- **Tasks:** 2/2 completed
- **Files modified:** 2 (`mmcli/cli.py`, `tests/test_datasets_cli.py`)

## Accomplishments

- **`mmcli datasets` subcommand** with three actions, none of which
  reimplement download, verification, or resolution logic — everything
  network- or cache-related is imported directly from `mmcli.datasets`
  (10-02):
  - `list [--format text|json] [-t TASK_TYPE] [-m MODULE]` — human table by
    default; `--format json` is the committed interface below.
  - `pull <name> [--force]` — calls `fetch_dataset()` verbatim; unknown
    names exit 2 listing every registered name.
  - `path <name>` — prints the resolved path via `_resolve_dataset_zip()`,
    or exits 1 naming the exact `datasets pull` command when unavailable.
- **`_dataset_state(name)`** classifies each entry into `bundled` / `cached`
  / `downloadable` / `unavailable` by calling `_resolve_dataset_zip()`
  itself and comparing directories, so the CLI's notion of availability can
  never drift from `init`'s.
- **D-5 auto-fetch policy** for `init --dataset`: added `--fetch`/
  `--no-fetch` (mutually exclusive) plus `MMCLI_AUTO_FETCH=0|1`, and
  `_apply_init_fetch_policy()` implementing the exact 4-rule precedence
  (`MMCLI_DATASETS` hard-blocks everything → explicit refusal → explicit
  permission → `stderr_is_tty()` default), reusing 10-02's `stderr_is_tty()`
  verbatim. The policy function is invoked only when the dataset does not
  already resolve locally, so present datasets behave exactly as before
  this phase.
- **Refusal messages** name the dataset, its size in MB, and the exact
  `mmcli datasets pull <name>` command — never a bare "not available".
- `mmcli init --list`'s existing text-table output is byte-for-byte
  unchanged (not touched by this plan).

## The JSON contract (for 10-04 / DatasetCatalog.swift)

**Updated 2026-08-02 by 10-09 (CONTEXT.md D-10):** an additive `cache_bytes`
field was added below. This is an *addition* to the contract as originally
committed by this plan — every field and behaviour documented below from
`name` through `description` is unchanged and still authoritative; existing
consumers that read only those fields are unaffected.

```
mmcli datasets list --format json
```
emits:
```json
{
  "datasets": [
    {
      "name": "fan_blade_fault",
      "version": "01_03_00",
      "state": "bundled",
      "bytes": 56595859,
      "cache_bytes": null,
      "task_types": ["motor_fault"],
      "module": "timeseries",
      "description": "Fan blade fault classification (vibration data, 3-axis)"
    },
    ...
  ]
}
```
- Top level is a **dict with a `datasets` key** holding the array (not a
  bare array).
- `name`: registry key, stable identifier.
- `version`: the TI engine version path this entry is pinned to
  (`meta.ti_version` or `DATASETS_DEFAULT_VERSION`), or `null` for the one
  entry with no TI upstream (`generic_audio_classification`).
- `state`: one of exactly `bundled`, `cached`, `downloadable`, `unavailable`
  — computed live against the current machine's disk/env state, not a
  static registry field.
  - `bundled`: resolves via `MMCLI_DATASETS` or the packaged directory.
  - `cached`: resolves from `~/.cache/mmcli/datasets/<version>/`.
  - `downloadable`: has a TI source (`ti_name`) and is not present, and
    `MMCLI_DATASETS` is not set (so a fetch is actually possible).
  - `unavailable`: neither present nor fetchable right now — either no
    `ti_name` and not present, or `MMCLI_DATASETS` is set (which blocks
    `fetch_dataset` unconditionally, per REQ-DATA-03) and the file is not in
    that directory.
- `bytes`: from the registry (`meta.bytes`), never a HEAD request — listing
  all 10 costs zero network round-trips.
- `cache_bytes` **(added 10-09, CONTEXT.md D-10)**: the actual on-disk size
  in bytes of this dataset's version-scoped cache entry (from `os.stat`, not
  the registry's recorded `bytes`), or `null` if no cache entry exists.
  Deliberately independent of `state`: a dataset can report `state: bundled`
  (the packaged copy, or a file in the user's own `MMCLI_DATASETS`
  directory, wins resolution) while `cache_bytes` is still non-null, because
  a stale entry from an earlier download can sit in the cache underneath the
  winning copy. Without this field that disk usage is invisible and
  unreclaimable from a GUI — see `mmcli datasets remove` (10-09-PLAN.md),
  which is the only supported way to reclaim it. Computed via
  `cache_entry_path()`, a helper deliberately separate from
  `_resolve_dataset_zip()` (the resolution answer): it always names where
  the cache entry *would* be, never where the dataset currently resolves
  from.
- `task_types`, `module`, `description`: descriptive fields carried through
  unchanged from `DATASET_REGISTRY`.

All 10 registry entries are always present in the array regardless of
filters (filters only apply to `-t`/`-m`, tested separately). 10-04's Swift
model should treat this key set and state vocabulary as fixed; changing
either requires a coordinated update to `DatasetCatalog.swift`. (10-09 is
the one coordinated update on record: it added `cache_bytes` to both sides
in the same plan.)

## Task Commits

1. **Task 1: `mmcli datasets` subcommand with a machine-readable list** —
   `3aae067` (feat) — `datasets list/pull/path`, `_dataset_state`,
   `_dataset_record`, 17 tests.
2. **Task 2: Auto-fetch policy for `init --dataset` (D-5)** — `eb6e392`
   (feat) — `--fetch`/`--no-fetch`, `_resolve_explicit_fetch`,
   `_apply_init_fetch_policy`, `_do_init_fetch`, 7 additional tests.

Both tasks touch the same two files (`mmcli/cli.py`,
`tests/test_datasets_cli.py`); each commit was built and verified as a
genuine intermediate state (Task 2's additions temporarily removed, full
suite re-run green, then re-applied) rather than a post-hoc split of one
combined diff, following the precedent 10-02 set.

## Files Created/Modified

- `mmcli/cli.py` — added `_add_datasets_parser` (list/pull/path
  subparsers), `--fetch`/`--no-fetch` on `init`'s parser,
  `_dataset_state`/`_dataset_record`/`_handle_datasets_list`/
  `_handle_datasets_pull`/`_handle_datasets_path`,
  `_resolve_explicit_fetch`/`_do_init_fetch`/`_apply_init_fetch_policy`, and
  the `datasets`/init-policy wiring in `main()`. Top-level `--help` text now
  lists `datasets` alongside the other subcommands.
- `tests/test_datasets_cli.py` — new, 24 tests: JSON contract (all 10
  records, required keys, valid state enum, descriptive fields), all four
  state transitions (`bundled` via bundled dir and via `MMCLI_DATASETS`,
  `cached`, `downloadable`, `unavailable`), text-format list, task/module
  filters, `pull`/`path` error handling and force-flag forwarding, and all
  seven D-5 policy behaviour cases plus the argparse mutual-exclusion
  check.

## Decisions Made

- **JSON shape**: `{"datasets": [...]}`, not a bare array — documented
  above as the committed interface.
- **`state` computed, not stored**: rather than adding a `state` field to
  `DATASET_REGISTRY` (which would need to be kept in sync with disk),
  `_dataset_state()` calls 10-02's `_resolve_dataset_zip()` on every
  invocation and classifies the result. This is the same function `init`
  uses internally, so the two can never disagree.
- **Policy is opt-in per dataset, not global**: `_apply_init_fetch_policy`
  is only called from the `init --dataset` handler, and only when the
  dataset does not already resolve locally. No other subcommand consults
  it — `datasets pull` always calls `fetch_dataset` directly regardless of
  TTY, since a user typing that command has already made the explicit
  choice the policy exists to gate.
- **Test technique for "successful fetch" without network**: rather than
  standing up a TLS server or hand-crafting bytes with a matching sha256,
  tests monkeypatch `mmcli.datasets.dataset_url` and
  `mmcli.datasets._download_to_cache` so that `fetch_dataset()`'s own logic
  runs unmodified but the "download" step copies the real, already-bundled
  zip (byte-identical, so its digest already matches the registry) into the
  cache directory. This is the same technique 10-02's
  `test_force_flag_bypasses_fetch_dataset_cache_short_circuit` established.
- **Manual verification caveat**: the plan's own suggested manual repro
  (`mmcli init --dataset fan_blade_fault ... 2>&1 | cat` with an empty
  cache) does **not** currently demonstrate a refusal on this machine,
  because 10-03 (which unbundles the TI zips) has not landed yet in this
  repo — the zip is still physically present in `mmcli/example_datasets/`,
  so `_resolve_dataset_zip` finds it via the bundled path and the policy is
  correctly never consulted (this is the documented "already present, no
  policy involvement" behaviour, not a bug). The automated test suite
  proves the refusal path itself by hiding the bundled directory via the
  `hide_bundled` fixture, which is unaffected by whether 10-03 has run.
  Once 10-03 lands, the plan's manual repro will demonstrate the refusal
  directly.

## Deviations from Plan

None — plan executed as written. `--format json`'s exact key set
(`name`, `version`, `state`, `bytes`) and D-5's exact 4-rule precedence
were implemented as specified, both pinned by tests.

## Issues Encountered

None. All verification steps in the plan (both automated pytest subsets
and the standalone JSON-contract shell check) pass; the full pre-existing
regression suite (`test_init.py`, `test_tier4_cli.py`,
`test_cli_parsing.py`, `test_datasets_download.py`) passes unmodified
alongside the new tests.

## User Setup Required

None. No network access is required to run this plan's tests; the full
`test_datasets_cli.py` suite was additionally verified to pass with all
non-loopback network access blocked at the socket layer.

## Next Phase Readiness

- **10-04** (PlatypusStudio download affordance) can now decode
  `mmcli datasets list --format json` against the exact schema documented
  above and drive `mmcli datasets pull <name>` for its download button. Its
  Task 3 checkpoint step 7 (confirming `init --dataset` does not
  auto-fetch under a piped invocation) will pass once 10-03 unbundles the
  datasets it targets (`fan_blade_fault`), since this plan's D-5 policy is
  already correctly wired and tested against the "not present" case via
  `hide_bundled`.
- **10-05** (README offline recipe) can reference `mmcli datasets pull` and
  `mmcli datasets path generic_audio_classification` directly; both are
  implemented and tested.
- **10-08** (CI wiring) has a new test file (`tests/test_datasets_cli.py`)
  to add to the explicit pytest file lists in both workflows.
- No blockers. `mmcli init --list` is unmodified; `mmcli init --dataset`
  for an already-present dataset behaves identically to before this phase.

## Self-Check: PASSED

- FOUND: `mmcli/cli.py` (contains `_add_datasets_parser`, `_dataset_state`,
  `_handle_datasets_list`, `_handle_datasets_pull`, `_handle_datasets_path`,
  `_apply_init_fetch_policy`, `_resolve_explicit_fetch`, `_do_init_fetch`)
- FOUND: `tests/test_datasets_cli.py` (24 tests)
- FOUND commit `3aae067` (Task 1)
- FOUND commit `eb6e392` (Task 2)
- `pytest tests/test_datasets_cli.py -q`: 24 passed
- `pytest tests/test_datasets_cli.py -q` re-run with all non-loopback
  `socket.connect` calls raising `AssertionError`: 24 passed (proves no
  test contacts the real network)
- `pytest tests/test_init.py tests/test_tier4_cli.py tests/test_cli_parsing.py tests/test_datasets_download.py -q -k "not TestInitDatasetExtractReal"`:
  93 passed (no regressions)
- `mmcli datasets list --format json | python3 -c "..."` (plan's own
  Task 1 verify script): `JSON CONTRACT OK`, exit 0
- `mmcli datasets path generic_audio_classification` (empty cache, no
  `MMCLI_DATASETS`): prints an existing path, exit 0
- `mmcli init --fetch --no-fetch ...`: argparse rejects with exit 2
- `mmcli init --list` output unchanged (10 datasets listed, same columns)

---
*Phase: 10-dataset-distribution-and-binary-size*
*Completed: 2026-07-22*
