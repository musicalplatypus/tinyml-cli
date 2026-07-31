---
phase: 10-dataset-distribution-and-binary-size
plan: 04
subsystem: platypusstudio
tags: [swift, swiftui, cross-repo, dataset-download, human-verified]

requires:
  - phase: 10-06
    provides: "`mmcli datasets list --format json` and `datasets pull`, and the D-5 TTY policy that makes the app own the prompting"
  - phase: 10-03
    provides: "the unbundled binary, without which every dataset would already be local and this affordance dead code"
provides:
  - "DatasetCatalog — four-state availability decoded from JSON, plus failure classification and size formatting"
  - "Gated Create in NewProjectSheet with an explicit, cancellable, size-labelled download"
  - "A Setup row exposing the dataset cache location and counts"
  - "Managed mmcli copy in Application Support (CONTEXT D-07)"
affects: [10-09]

requirements-completed: [REQ-UX-01]
---

# 10-04: PlatypusStudio download affordance

**Code lands in the `PlatypusStudio` repo, not `tinyml-cli`.** Six commits from baseline
`783acb0`; 140 tests pass (106 before this plan).

| Commit | What |
|---|---|
| `13f2f33` | Task 1 — `DatasetCatalog` + Setup cache row |
| `e2ffeec` | Task 2 — gated Create + download affordance |
| `9a51421` | deviation — reject an mmcli that cannot run |
| `ecbafcf` | deviation — report an unreadable folder as a permission problem |
| `068e4a4` | CONTEXT D-07 — managed mmcli copy outside the gated folders |
| `658f71b` | checkpoint finding — cancelled download reported itself as a failure |

## What was built

`DatasetCatalog` decodes `mmcli datasets list --format json` into `bundled` / `cached` /
`downloadable(bytes:)` / `unavailable`, taking size from the registry's `bytes` rather than a
HEAD per dataset. A failed or unparseable invocation stays distinct from a successful-but-empty
one, and an unrecognised state fails safe to `unavailable` so a future mmcli cannot produce a
download button that cannot work.

`NewProjectSheet` disables Create for a `.downloadable` selection and offers a size-labelled
Download that runs `mmcli datasets pull`, streams its output, and re-reads availability on
success so Create unlocks without reopening the sheet. Bundled and cached datasets show no
affordance at all. **The app never passes `--fetch`** — `ProcessRunner` pipes stderr, so mmcli
sees a non-terminal and will not fetch on its own; the app owns the prompting by construction.

Failure classification lives in MMCLIKit so it is testable without a subprocess, keyed to
mmcli's real wording. Writing those tests caught the network matcher missing the commonest
case: urllib reports a DNS failure as `<urlopen error … nodename nor servname provided>`, which
contains none of "network", "connection" or "resolve".

## Task 3 — the human-verification checkpoint

Driven against the real macOS app (CONTEXT D-09), with the cache cleared and nine datasets
`downloadable`.

| Check | Result |
|---|---|
| Size shown before any transfer | **Observed** — "Download (13.3 MB)"; 56.6 MB for `fan_blade_fault` |
| Create gated on availability | **Observed** — clicked with a valid name and destination: no project dir, no `mmcli init` process. Genuinely disabled, not merely dimmed |
| Progress visible | **Observed** — spinner, "Downloading 13.3 MB…", Cancel |
| Cancel terminates the subprocess | **Observed** — process gone, no `.part` file, no cache entry |
| Download completes → Create unlocks → project created | **Observed** — `gate_probe` created, dataset extracted, `platypus.json` seeded, project window opened with 4,836,258 samples analyzed |
| Cached/bundled show no affordance | **Observed** — the row disappears once cached |
| Integrity of a GUI-initiated download | **Observed** — bytes matched the registry sha256 exactly |
| D-5 holds underneath the app | **Observed** — piped `init --dataset fan_blade_fault` refuses, names the size, and points at `datasets pull`; no download, no traceback |
| Setup cache row | **Not observed** — the Setup panel was not opened during the pass |
| Offline failure message | **Not observed** — network was not disabled |

### The defect the checkpoint found

Cancelling rendered a **raw Python traceback** in the sheet. Interrupting mmcli mid-transfer
makes Python unwind through `KeyboardInterrupt`, and that stack was surfaced as the download
failure — a wall of red frames for an action that did exactly what was asked.

**All 138 unit tests passed with this present.** That is the plan's own stated reason for being
`autonomous: false` ("every UI defect found in this project so far passed its tests"), and it
held again.

Fixed in `658f71b`: a user-initiated cancel is no longer treated as a failure, and failure text
is capped to its last lines so no future unbounded output can push the sheet off screen.

**Caveat, stated rather than glossed:** the fix is verified by unit test but was **not**
re-observed in the UI. Rebuilding to verify it reset the ad-hoc signature, macOS treated the
app as new, and a privacy prompt blocked further input. Answering a system privacy dialog is
the user's decision, not something to automate. The cancel *behaviour* (subprocess terminated,
nothing cached) was observed before the fix and is unchanged by it; only the absence of the
traceback message is unconfirmed visually.

## Deviations

Four changes outside this plan's declared `files_modified`, all found by driving the app.
Recorded here and in `unplanned-work.md` §2.

1. **`MMCLIBinary.swift` — resolution accepted an mmcli that could not run.** It probed each
   candidate with `--version` and ignored the result, so a stale `pip` console-script shim in
   `~/.venv-ai/bin` — left behind when this repo moved from `Documents/repos/TexasInstruments/`
   — won over the working build purely by being first on PATH. Every command then failed with
   `ModuleNotFoundError`. Only a candidate that answers `--version` is now accepted; when
   nothing runs, the first candidate is returned carrying its failure text rather than reported
   as missing.
2. **`Preflight.swift` + `NewProjectSheet.swift` — an unreadable folder read as absence.** macOS
   gates Documents/Desktop/Downloads per application and a denied `stat` is indistinguishable
   from a missing file, so a lost grant made the app claim mmcli was not found while the binary
   sat in plain sight and the project list emptied itself. Both surfaces now name the folder,
   where to grant access, and that a rebuilt unsigned app must be granted again.
3. **CONTEXT D-07 — a managed mmcli copy in Application Support.** Resolution fell back to a
   build under `~/Documents`; that path is now the last resort and a working binary found
   anywhere else is copied to the app's own directory on launch. Confirmed live: the app invokes
   `~/Library/Application Support/PlatypusStudio/bin/mmcli`. **Partial by design** — projects
   still live in `~/Documents`, so the workspace list continues to need a grant. This removes
   the permission dependency from *finding mmcli*, which is what made the app look broken.
4. **Test isolation.** D-07's default let the resolution tests see the real binary installed on
   this machine, so they passed or failed on developer state rather than on code. They now pass
   `managedCopy: nil` explicitly.

## Notes

- `10-04-PLAN.md` listed `DatasetCatalogTests.swift` and three sources in `files_modified`;
  actual sources touched were `DatasetCatalog.swift`, `MMCLIBinary.swift`, `Preflight.swift`,
  `NewProjectSheet.swift` and `WorkspaceStore.swift`.
- Failure classification and byte formatting were placed in MMCLIKit rather than the SwiftUI
  target, per the design spec's architecture rule — the executable has no test target.
- **Ad-hoc signing makes this recur.** Every rebuild is a new application to macOS, so the
  Documents grant resets. Accepted for now (CONTEXT D-08); a stable signing identity is
  deferred to its own phase.
- `~/Documents/PlatypusStudio Projects/gate_probe` is a real project created during the pass.
  The user asked for it to be kept.
- Zero "TI" / "Texas Instruments" / "Edge AI Studio" strings in any user-facing text added here.

## Self-Check: PASSED

- Both automated tasks committed atomically in the PlatypusStudio repo; no tinyml-cli commits
- 140 tests pass (106 baseline + 34), `swift build` clean
- Checkpoint driven against the real app; every claim above is something observed, and the two
  unobserved checks plus the unverified fix are named rather than implied
- REQ-UX-01 satisfied: size shown before transfer, Create gated on local availability, download
  explicit and cancellable, distinct failure messages, no implicit fetch
