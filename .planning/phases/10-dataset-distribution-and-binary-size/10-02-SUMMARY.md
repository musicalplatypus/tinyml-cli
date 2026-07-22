---
phase: 10-dataset-distribution-and-binary-size
plan: 02
subsystem: dataset-distribution
tags: [urllib, sha256, cache, download, security, datasets]

requires:
  - phase: 10-01
    provides: shared PyInstaller exclude list and binary size ceiling (unrelated file surface, sequencing dependency only)
provides:
  - "DATASET_REGISTRY entries with ti_name/ti_version/sha256/bytes for the nine TI-fetchable datasets"
  - "dataset_url(name) — version-pathed TI URL, KeyError on unknown name"
  - "_cache_dir(version) — XDG_CACHE_HOME-aware, version-scoped cache directory"
  - "_resolve_dataset_zip(name) — MMCLI_DATASETS -> bundled -> version cache -> None resolution, used by extract_dataset()"
  - "fetch_dataset(name, *, force=False) -> str — download, sha256-verify, atomic cache, MMCLI_DATASETS/HTTPS-only guarded"
  - "stderr_is_tty() — shared TTY predicate for the tqdm progress bar and (in 10-06) the init --dataset auto-fetch policy"
affects: [10-03, 10-06, 10-04, 10-05, 10-07]

tech-stack:
  added: []
  patterns:
    - "download-to-temp-in-same-dir, verify, then os.replace() for atomic cache writes"
    - "version-keyed cache directory so a pinned-version bump cannot reuse stale data"
    - "single shared isatty() predicate reused for two different policy decisions (progress bar here, auto-fetch permission in 10-06) instead of two independent heuristics"
    - "low-level downloader (_download_to_cache) kept scheme-agnostic so it is testable against a plain local http.server, while the public fetch_dataset() is the only place that enforces HTTPS"

key-files:
  created:
    - tests/test_datasets_download.py
  modified:
    - mmcli/datasets.py

key-decisions:
  - "Split network-mechanics testing from policy testing: _download_to_cache (scheme-agnostic) is exercised directly against a local http.server for every failure mode (truncated/oversized/redirect/404/timeout/checksum-mismatch), while fetch_dataset's HTTPS-only/MMCLI_DATASETS/no-ti_name refusals are tested via monkeypatching so no TLS certificate machinery was needed in the unit suite."
  - "_resolve_dataset_zip wraps _datasets_dir() rather than reimplementing its MMCLI_DATASETS/bundled precedence, per the plan's explicit instruction, so every existing caller and test of _datasets_dir() is untouched."
  - "Split the plan's 3 tasks into 3 sequential commits despite both files being touched by every task, since Task 2 depends on Task 1's dataset_url/registry and Task 3 depends on Task 2's _cache_dir — verified buildable and green at each intermediate stage before committing."

patterns-established:
  - "Any new fetchable dataset must add both sha256 and bytes or the module refuses to import (_validate_registry, REQ-DATA-02)."
  - "Any code path that decides whether to show progress or start an unnarrated download should call stderr_is_tty() rather than a fresh isatty() check (10-06 depends on this)."

requirements-completed: [REQ-DATA-01, REQ-DATA-02, REQ-DATA-03, REQ-DATA-05]

duration: 90min
completed: 2026-07-22
---

# Phase 10 Plan 02: Registry digests/versioning, version-scoped cache, verified fetch_dataset Summary

Added TI download URLs, sha256/bytes to the dataset registry, a version-keyed on-disk cache, and a stdlib-urllib `fetch_dataset()` that verifies-then-atomically-replaces so a corrupted, truncated, or substituted download can never become a poisoned cache entry.

## Performance

- **Duration:** ~90 min
- **Started:** 2026-07-22T18:05:00Z (approx, session start)
- **Completed:** 2026-07-22T19:37:19Z
- **Tasks:** 3/3 completed
- **Files modified:** 2 (`mmcli/datasets.py`, `tests/test_datasets_download.py`)

## Accomplishments

- `DATASET_REGISTRY`'s nine TI-fetchable entries now carry `ti_name`, `sha256`, and `bytes`
  matching the measured provenance table in `10-RESEARCH.md` verbatim; `generic_audio_classification`
  carries `sha256`/`bytes` but no `ti_name` (it has no TI upstream).
- `dataset_url(name)` composes the version-pathed TI URL
  (`https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/<ti_name>`), returns `None`
  for the locally-authored entry, and raises `KeyError` on an unknown name rather than ever
  interpolating a caller-supplied string into a URL.
- Import-time enforcement (`_validate_registry`) makes a `ti_name` entry without a valid
  64-hex-char `sha256`/positive `bytes` a hard import failure naming the offending entry
  (REQ-DATA-02) — verified against both the real registry and a deliberately-broken copy of
  the module loaded via `importlib.util` (see Self-Check).
- `_cache_dir(version)` and `_resolve_dataset_zip(name)` implement the four-step resolution
  order (`MMCLI_DATASETS` → bundled → version cache → not present), reusing `_datasets_dir()`
  rather than replacing it, so every existing test and caller of `_datasets_dir()` is
  unaffected. Cache hits are re-verified against the registry sha256 on every resolution; a
  mismatch is treated as absent and reported to stderr, never silently used.
- `fetch_dataset(name, *, force=False)` downloads via stdlib `urllib.request` only, verifies
  sha256 before an atomic `os.replace()`, and refuses unconditionally when `MMCLI_DATASETS` is
  set (even on this explicit call, per REQ-DATA-03) or when the URL is not HTTPS. The
  lower-level `_download_to_cache` guards against truncated bodies, oversized bodies
  (1%/1 KiB tolerance over the registry `bytes`), cross-host redirects, HTTP 404 (naming the
  version and URL), and connect/read timeouts — every failure path removes the temp file and
  leaves the cache directory untouched.
- Added a zip-slip regression test (threat T-10-02-06) confirming `extract_dataset()` cannot
  be used to write outside the target project directory via a malicious `../` or absolute zip
  member path. Python 3.10's stdlib `zipfile._extract_member` already strips such paths (since
  3.6), so this is a confirming test rather than a new guard — documented rather than silently
  assumed, per the plan's threat model.

## Task Commits

Each task was committed atomically. Since Tasks 1-3 share the same two files
(`mmcli/datasets.py`, `tests/test_datasets_download.py`) and are intentionally layered
(Task 2 calls Task 1's `dataset_url`/registry; Task 3 calls Task 2's `_cache_dir`), each
commit was built as a genuine intermediate state of the file — not a post-hoc split of one
combined diff — and verified green (`pytest tests/test_datasets_download.py -q` plus the
existing dataset test suites) before being committed:

1. **Task 1: Registry gains ti_name/ti_version/sha256/bytes, URL is derived** - `aa33ba4` (feat)
2. **Task 2: Version-scoped cache and resolution order** - `e075dc3` (feat)
3. **Task 3: fetch_dataset with mandatory verification** - `95b8f90` (feat)

**Plan metadata:** (this commit)

## Files Created/Modified

- `mmcli/datasets.py` — registry augmented with `ti_name`/`ti_version`/`sha256`/`bytes`;
  added `TI_DATASETS_BASE`, `DATASETS_DEFAULT_VERSION`, `DOWNLOAD_TIMEOUT_SECONDS`,
  `stderr_is_tty()`, `_cache_dir()`, `_sha256_of()`, `dataset_url()`, `_resolve_dataset_zip()`,
  `_HostLockedRedirectHandler`, `_download_to_cache()`, `fetch_dataset()`; `extract_dataset()`
  now resolves its zip path through `_resolve_dataset_zip()` instead of building it from
  `_datasets_dir()` directly.
- `tests/test_datasets_download.py` — new, 668 lines, 44 tests: registry invariants, URL
  derivation, cache directory/resolution order, zip-slip confirmation, and the full
  `fetch_dataset`/`_download_to_cache` failure-mode matrix against a local `http.server`.

## Exact signatures for 10-03 and 10-06

```python
def dataset_url(name: str) -> str | None: ...          # KeyError on unknown name
def _cache_dir(version: str) -> str: ...                # creates 0700 dir, returns path
def _resolve_dataset_zip(name: str) -> str | None: ...   # env -> bundled -> cache -> None
def fetch_dataset(name: str, *, force: bool = False) -> str: ...  # returns cached path
def stderr_is_tty() -> bool: ...                         # shared TTY predicate for 10-06's D-5
```

`DATASETS_DEFAULT_VERSION = "01_03_00"`. `TI_DATASETS_BASE = "https://software-dl.ti.com/C2000/esd/mcu_ai"`.

## Decisions Made

- Reused `_datasets_dir()` unchanged and wrapped it inside `_resolve_dataset_zip()` rather than
  inlining its env/bundled logic a second time, per the plan's explicit instruction — this kept
  every pre-existing test in `tests/test_init.py` and `tests/test_tier4_cli.py` passing
  unmodified.
- Split `_download_to_cache` (scheme-agnostic network mechanics) from `fetch_dataset` (policy:
  `MMCLI_DATASETS`, HTTPS-only, no-ti_name refusal, cache short-circuit) so the download
  mechanics are testable against a plain local `http.server` without needing self-signed TLS
  certificates in the unit suite, while `fetch_dataset`'s own policy gates are tested via
  monkeypatching to prove they refuse before any request is attempted. This is a deviation
  from a literal single-function reading of the plan's "Implement `fetch_dataset`..." action
  text, but it satisfies every behavior bullet and the plan's own instruction to "Test against
  a local `http.server` fixture ... not against TI" — a single `fetch_dataset(https_url)` could
  not be driven against a plain-HTTP local server at all.
- Named the shared TTY predicate `stderr_is_tty()` (not `_is_tty` or similar) since 10-06-PLAN.md
  explicitly requires importing this exact helper for the `init --dataset` auto-fetch policy
  (D-5) rather than writing a second `isatty()` check.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - blocking issue] Caught raw `OSError`/`TimeoutError` from `opener.open()`, not only `urllib.error.URLError`**
- **Found during:** Task 3, writing the hung-server timeout test
- **Issue:** A socket-level timeout while reading the HTTP response status line
  (`http.client.HTTPResponse.begin()` internals) surfaces as a raw `OSError`/`TimeoutError`
  from `opener.open()`, not wrapped in `urllib.error.URLError` as I initially assumed. The
  original `except urllib.error.URLError` clause did not catch it, so the hung-server test
  failed with an uncaught `TimeoutError` escaping `_download_to_cache` instead of the intended
  `RuntimeError`.
- **Fix:** Added an `except OSError as exc:` clause alongside the existing `HTTPError`/`URLError`
  handling in `_download_to_cache`, converting it to the same `RuntimeError` message pattern.
- **Files modified:** `mmcli/datasets.py`
- **Commit:** `95b8f90` (part of Task 3's commit — found and fixed before that commit, not
  requiring a follow-up commit)

No other deviations. Design constraint D-5 (auto-fetch policy, `stderr.isatty()` gating) was
not implicated by this plan's actual task text — `10-02-PLAN.md`'s three tasks scope
strictly to the registry, cache, and `fetch_dataset()` library function; the `init --dataset`
CLI wiring and auto-fetch decision belong entirely to `10-06-PLAN.md`, which this plan does
not touch (`mmcli/cli.py` is outside this plan's `files_modified`). `stderr_is_tty()` was
added here specifically so 10-06 has the one shared predicate to import, per that plan's own
instruction.

**Total deviations:** 1 auto-fixed (Rule 3).
**Impact on plan:** Necessary for correctness of the documented timeout behavior; no scope
creep beyond what Task 3's action text already called for ("Apply a connect and read timeout
so a hung server fails rather than stalling forever").

## Issues Encountered

None beyond the OSError-catching gap above, which was found and fixed during the same task
before committing.

## User Setup Required

None — no external service configuration required. No real network access to
`software-dl.ti.com` was needed or used in this plan; all 44 new tests run against a local
`http.server` fixture and were additionally verified to pass with all non-loopback network
access blocked at the socket layer.

## Next Phase Readiness

- 10-03 (GET-and-hash gate + unbundle) can call `fetch_dataset()` directly against the real
  TI URLs for its one-time verification gate, and lower `scripts/binary_size_ceiling.txt` once
  the datasets are removed from `--add-data`.
- 10-06 (`mmcli datasets list/pull/path` + D-5 auto-fetch policy) can import `fetch_dataset`,
  `dataset_url`, `_resolve_dataset_zip`, and `stderr_is_tty` directly — no reimplementation of
  download, verification, or TTY-detection logic should be needed in `cli.py`.
- No blockers. `mmcli init --list` and `mmcli init --dataset <name>` (zip present locally)
  behave identically to before this plan, confirmed by the full existing `test_init.py` /
  `test_tier4_cli.py` suites passing unmodified.

## Self-Check: PASSED

- FOUND: mmcli/datasets.py (contains `def fetch_dataset`, `def dataset_url`, `def _cache_dir`,
  `def _resolve_dataset_zip`, `def stderr_is_tty`)
- FOUND: tests/test_datasets_download.py (668 lines, 44 tests)
- FOUND commit `aa33ba4` (Task 1)
- FOUND commit `e075dc3` (Task 2)
- FOUND commit `95b8f90` (Task 3)
- `pytest tests/test_datasets_download.py -q`: 44 passed
- `pytest tests/test_init.py tests/test_tier4_cli.py tests/test_dataset_manager.py tests/test_dataset_preset.py tests/test_build_config.py -q`: 91 passed (no regressions)
- `pytest tests/test_datasets_download.py -q` re-run with all non-loopback `socket.connect`
  calls raising `AssertionError`: 44 passed (proves no test contacts `software-dl.ti.com`)
- `python3 -c "from mmcli.datasets import DATASET_REGISTRY as R, dataset_url; ..."` (plan's
  own Task 1 verify script): `BAD: none`, exit 0
- `python3 -c "import mmcli.datasets as d, inspect; ..."` (plan's own Task 3 verify script):
  `MISSING: none`, exit 0
- Deliberately stripped `sha256` from `fan_blade_fault` in a throwaway copy of the module
  loaded via `importlib.util`: import raised `ValueError` naming the entry, as required
- `mmcli init --list` output unchanged (10 datasets listed)

---
*Phase: 10-dataset-distribution-and-binary-size*
*Completed: 2026-07-22*
