---
phase: 10-dataset-distribution-and-binary-size
plan: 12
type: execute
gap_closure: true
status: complete-with-open-items
requirements-completed: [REQ-UX-01]
requirements-partial: [REQ-UX-02]
repos_touched: [PlatypusStudio]
---

# 10-12 — Determinate transfer progress + success confirmation

Closes UAT gap 2 (test 9): the New Project sheet's dataset download showed no progress and no
completion signal, so a working download was visually indistinguishable from a crash.

**Written retrospectively on 2026-08-04.** The plan reached its blocking human checkpoint and no
SUMMARY was written at the time; the executor stopped at Task 3 and the continuation agent
declined to write one while the checkpoint was open. This records what actually happened,
including the parts that did not work.

## Commits (all in PlatypusStudio — no commit spans repos)

| Commit | What |
|---|---|
| `941f8c6` | `DatasetTransferProgress` — total decoder for mmcli's `--progress-json` NDJSON stream |
| `f17d34a` | Determinate progress in both surfaces + session-scoped success confirmation |
| `0e9bca0` | `@MainActor` on `NewProjectSheet`/`DatasetLibraryView`; confirmation-flicker fix |

Upstream contract supplied by 10-11 (`9ededf8`, tinyml-cli): `integrity-repair` / `start` /
`progress` / `result`, each carrying `"v":1`, on stderr, only under `--progress-json`.

## What was verified, and how

Task 3's checkpoint was driven against the real app via computer-use on 2026-08-03. Full record:
`10-12-CHECKPOINT-OBSERVATIONS.md`.

| Check | Result |
|---|---|
| 2 — success confirmation | **PASS** |
| 3 — already-local dataset renders nothing | **PASS** |
| 4 — cancel mid-transfer | **PASS** — first verification in this project |
| 8 — no JSON leaks into the UI | **PASS** |
| 1 — determinate progress | **FAIL**, then fixed, then **re-verified only for the fix's absence of regression** |
| 5, 6, 7 | **NOT RUN** |

Check 4 matters beyond this plan: 10-09 recorded cancel-mid-transfer as INCONCLUSIVE because every
dataset downloaded faster than a click. A 56.6 MB dataset made it reachable — the row returned to
`Download`, no traceback, and **no `.part` or temp files were left behind**.

## The defect this plan shipped, and the two errors on the way to it

Check 1 failed: the byte counter read `Zero KB of 56.6 MB` for an entire 56.6 MB transfer.

**First diagnosis was wrong.** It claimed the label proved the `start` event was ingested. It did
not — `NewProjectSheet.swift:139-140` composes the label from the `.downloadable(let bytes)`
availability case, not the event stream. That number proved nothing.

**The fix applied on that diagnosis (`0e9bca0`, `@MainActor`) did not resolve it either** — the
counter stayed frozen when re-tested. What it *did* fix is real: the post-transfer flicker, where
`downloading` was cleared before `refreshAvailability()` completed, so a finished download briefly
reverted to "not on this machine yet" before confirming.

**A stale artifact hid the failure once more.** The first re-test ran against an `.app` built at
13:49 while the fix committed at 13:51 — the bundle predated the fix. Rebuilding and re-testing
showed the counter still frozen. Three separate stale-artifact traps occurred in this phase
(`dist/mmcli`, a worktree base, this `.app`); the lesson is in `unplanned-work.md`.

Ruled out by measurement, not inference: NDJSON is emitted correctly and is stderr-only; pipe
tagging in `ProcessRunner.swift:130-131` is correct; the stream is incremental, not buffered to
completion; `@MainActor` is applied; both binaries were current.

## Status

- **REQ-UX-01 — met.** The download affordance works end to end: uncached dataset → Download →
  transfer → availability refresh → `Create` unlocks without reopening the sheet. Verified.
- **REQ-UX-02 — partial.** The library surface exists and was verified in 10-09, but this plan's
  additions to it (progress, bulk cancel) were never exercised.

## Open items — carried to PlatypusStudio's own roadmap

Determinate progress is **still not working** and the cause is not established. Checks 5, 6 and 7
were never run. These are now `REQ-UAT-01..03` in
`../../../PlatypusStudio/.planning/ROADMAP.md` Phase 2, which is where PlatypusStudio work is
planned from 2026-08-04 onward.

Recommended next step, unchanged from the checkpoint record: instrument the consumer — log every
line received from `proc.lines` with its `isStderr` flag during a real download. That single
observation separates "events not arriving" from "arriving but not rendering" and ends the
guessing. Everything cheaper has been ruled out.
