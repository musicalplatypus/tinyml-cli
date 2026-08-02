---
phase: 10-dataset-distribution-and-binary-size
plan: 09
subsystem: datasets, platypusstudio
tags: [python, swift, swiftui, cross-repo, cache-management, human-verified]

requires:
  - phase: 10-04
    provides: "DatasetCatalog, extended here rather than forked"
  - phase: 10-06
    provides: "the datasets list/pull/path JSON contract, extended here with cache_bytes"
provides:
  - "mmcli datasets remove — cache-only deletion with a re-asserted path guard"
  - "cache_bytes: additive per-dataset reclaimable size in datasets list --format json (D-10)"
  - "Manage Datasets library: per-dataset state/size, bulk download, task/module filters, cache footer"
affects: [10-07]

requirements-completed: [REQ-UX-02]
---

# 10-09: Standalone dataset library + `mmcli datasets remove`

**Spans two repositories.** tinyml-cli: `47c76f3`, `420adde`, `0bc9703`, `2f639eb`.
PlatypusStudio: `6d66d7f`, `560aa63`, `e5142bf`.

## What shipped

**`mmcli datasets remove`** deletes only from the version-scoped cache. The target comes from
`cache_entry_path()` — deliberately *not* `_resolve_dataset_zip()`, which can return the
packaged copy or a file in the user's `MMCLI_DATASETS` directory — and the containment check is
re-asserted immediately before `unlink`, so it survives a future refactor of the path helper.
Never recursive, never touches another version's copy.

**`cache_bytes`** (CONTEXT D-10) is an additive field on 10-06's contract, reporting on-disk
cache size *independently of `state`*. This is what makes a shadowed entry visible: a dataset
can resolve as `bundled` while still holding a stale cache entry underneath it.

**Manage Datasets** hangs off the workspace toolbar and opens with no project selected. Per-row
state and size, bulk selection with a single-active-subprocess queue (D-12), task/module filters
(D-13), and a footer reporting cached count, disk used and cache location (D-11).

## Checkpoint — driven against the real app

Preconditions: cache cleared, `MMCLI_DATASETS` unset, freshly built binary (25,258,768 bytes,
under the 27,262,976 ceiling) copied to the managed location the app resolves from.

| # | Check | Result |
|---|---|---|
| 1 | Library reachable, no project open | **Observed** — opens from the toolbar, lists all 10 with sizes |
| 2 | Affordances match the approved table | **Observed** — downloadable rows get a size-labelled Download and a bulk checkbox; `generic_audio_classification` shows "Already available locally" with neither |
| 3 | Download a small dataset | **Observed** — 71,053 bytes cached, row → "Downloaded" + Remove, footer → "1 dataset(s) using 0.1 MB of disk, in …/01_03_00" |
| 4 | Remove it | **Observed** — file gone, row → Download, footer reset, **cache directory itself preserved** (removal is not recursive) |
| 5 | **Cancel mid-transfer** | **INCONCLUSIVE — see below** |
| 6 | Offline failure message | **Not exercised** — network was not disabled |
| 7 | Checksum-mismatch message | **Not exercised** — needs deliberate tampering; wording is unit-covered only |
| 8 | `MMCLI_DATASETS` → empty dir | **Observed at CLI level** — all 10 report `unavailable`, none `downloadable`. **Not** verified in the GUI: a `.app` launched via `open` does not inherit shell environment, and `MMCLI_DATASETS` is not among the variables the app injects |
| 9 | **Hard gate — removal safety** | **PASSED** — see below |
| — | D-10 `cache_bytes` | **Observed** — null when uncached, 71053 after download |
| — | D-12 bulk download | **Observed** — "Select All Downloadable" → "Download Selected (8)", queue processed serially |
| — | D-13 filters | Present and populated; **not exercised** |

### Check 9 passed, including the sharpened variant

With `MMCLI_DATASETS` pointing at a directory holding a real `pir_detection.zip`:

- `datasets remove pir_detection` → exit 0, removed the **cache** entry, printed a NOTE that
  resolution is unchanged in this environment, and **left the user's air-gapped file intact**
- Repeated with nothing cached (the sharpening suggested by the gemma4 review) → exit 0,
  "not cached; nothing to remove", air-gapped file still intact — it *refuses by construction*,
  not merely by failing safe
- `generic_audio_classification` (packaged, no upstream) → "not cached; nothing to remove"; the
  packaged copy is never a deletion target
- Unknown name → exit **2**, listing every registered dataset

### Check 5 is inconclusive, and that is not a pass

Three attempts, none conclusive:

1. Single 56.6 MB download — the transfer completed before the Cancel click landed.
2. Bulk download, cancel mid-queue — the queue stopped at 5 of 8, consistent with "cancel drops
   the remainder", but two datasets completed *after* the click, so the stop cannot be cleanly
   attributed to it.
3. Deterministic retry (Download then Cancel back-to-back on the 56.6 MB row) — 4 seconds later
   the file was absent with no surviving subprocess, which looked like a pass; a further check
   showed **all nine datasets cached**, meaning the download had simply still been running and
   then finished. The click had not landed on Cancel.

On this connection every dataset downloads faster than a screenshot-click round trip, so the
window is smaller than the automation's latency. **This says nothing about whether cancel
works.** Note that 10-04 *did* verify cancellation through the same `ProcessRunner` mechanism in
the New Project sheet (subprocess terminated, no cache entry, no `.part` file). What remains
genuinely unverified is this plan's own addition: that cancelling also **drops the rest of the
bulk queue**. That needs a slow link, a throttle, or a unit-level test — not a faster clicker.

## Defects found and fixed during verification

**Cache inspection created directories.** Found by CodeRabbit (recorded in `10-REVIEWS.md`),
then found to be **incompletely fixed** by running the real binary. `cache_entry_path()` was
corrected first, but `_resolve_dataset_zip()` built its step-3 path the same way, so
`datasets list` still created `<cache-home>/mmcli/datasets/<version>/` on a read-only listing —
and would fail outright on an unwritable cache home with no download requested.

The regression test missed it for an instructive reason: it called `cache_entry_size()` in a
loop instead of invoking the CLI, exercising a path that *resembled* the real one. It was
replaced with a subprocess test against the actual CLI **plus** a test that reaches step 3
directly — in a source tree all ten zips sit in the package directory, so resolution returns at
step 1 and the cache branch is never taken, which is why the subprocess test alone still passed
with the bug reintroduced. Both were mutation-checked.

Two smaller fixes from the same review: ENOSPC escaping on the buffered flush at block exit
(only `out.write` was guarded), and `os.path.getsize` sitting outside the `try` around `unlink`.

## Cosmetic issue, not fixed

`generic_audio_classification` displays **"0 MB"**. It is 18,371 bytes, which rounds to zero at
MB precision, so a dataset that is present reads as having no size. Worth a sub-MB unit or a
"<0.1 MB" floor. Left alone: it is presentation-only and outside this plan's stated scope.

## Deviations

1. `DatasetInfo.sizeBytes` added — the model carried a byte count only for `.downloadable`,
   which could not satisfy this plan's own must-have that every dataset shows its size.
2. Swapped `///` doc comments on `.downloadable` / `.cached` corrected (from 10-04);
   documentation only, all call sites were already correct.
3. The plan's verification names a literal `grep -- '--fetch'` expected to return nothing on
   `DatasetCatalog.swift`, but that file already contained the string inside a doc comment
   written in 10-04. The accurate comment was left alone rather than reworded to satisfy a
   grep — **the grep in the plan is the thing that is wrong**. The real constraint, that
   `--fetch` is never passed as a subprocess argument, holds and was verified by inspection.
4. CONTEXT D-10..D-13 were implemented here although they postdate the plan text.

## Test counts

tinyml-cli: `test_datasets_cli.py` **41 passed**; download/build/CI suites **92 passed**.
PlatypusStudio: **163 passed, 1 skipped** (baseline 140 + 23 new).

## Note for 10-07

`datasets remove` now exists and is verified, so its help text and `docs/RELEASING.md` can
document it. The NOTE printed under `MMCLI_DATASETS` is part of its observable behaviour and
worth documenting alongside.

## State left on the machine

All nine fetchable datasets are cached (~125 MB in `~/.cache/mmcli/datasets/01_03_00/`) as a
by-product of the bulk-download test. Harmless and re-downloadable; removable with
`mmcli datasets remove <name>` or by deleting that directory.

## Self-Check: PASSED

- Tasks 1-3 committed atomically in the correct repos; Task 4 driven against the real app
- Every claim above is something observed; the three unexercised checks and the inconclusive
  cancel test are named as such rather than implied
- The hard gate that matters — removal never touching a packaged or user-supplied dataset —
  passed, including the sharpened variant from the cross-AI review
- REQ-UX-02 satisfied: a library reachable at any time, independent of project creation, showing
  per-dataset size and state, with download and removal
