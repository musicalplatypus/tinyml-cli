# Deferred items — out of scope for the plan that found them

Items discovered during plan execution that are pre-existing, out of the
executing plan's file scope, and therefore logged here rather than fixed
inline (per the executor's scope-boundary rule).

## Found during 10-05 (README truth-up)

- **`mmcli/cli.py` still says "from TI" in two places, stale after 10-03's
  repoint to the GitHub release mirror:**
  - `datasets pull`'s subparser description: `"Fetch a dataset from TI and
    cache it. ..."` (around `_add_datasets_parser`'s `pull_p = sub.add_parser(...)`).
  - `init --fetch`'s help text: `"Force-fetch a missing dataset from TI,
    regardless of whether stderr is a terminal. ..."`.
  - Not fixed here: `10-05-PLAN.md`'s `files_modified` is `[README.md,
    README_zh.md]` only; `mmcli/cli.py` is out of this plan's scope. The
    behavior is correct (it fetches from the GitHub mirror, not TI) — only
    the CLI's own `--help` prose is stale. Low severity (in-tool help text,
    not a user-facing README claim), but should be corrected the next time
    `mmcli/cli.py` is touched, e.g. by 10-07 (CLI help / Sphinx docs plan)
    or a follow-up.

## Found during an exploratory pass over the training-report and NAS pages (2026-08-02)

**Not phase 10 scope.** Phase 10 covers dataset distribution and binary size; the only
PlatypusStudio surfaces in it are the New Project download affordance (10-04) and the dataset
library (10-09). These pages were never in scope. They were driven at the user's request after
they asked whether the live training-report and NAS pages had been tested — they had not, and
structurally could not be: `Package.swift` declares one test target, `MMCLIKitTests`, which
depends only on `MMCLIKit`. The SwiftUI executable target has no test target and no test file
imports SwiftUI, so `RunTabView`, `MetricCharts`, `ConfusionMatrixView`, `CompareView`,
`RunsPanel`, `NASSearchView` and `ArchitectureView` have **zero automated coverage**.

Observed against `~/Documents/edgeai/ecg_classification`, which holds three archived runs
(one `completed` NAS run, two `failed`). All findings below were confirmed against code and
on-disk state, not inferred from the screen alone.

### D-A. Archived runs capture metadata only — no metrics, no artifacts, no log

`.platypus/runs/20260711-195603-NAS_m/run.json` contains `"metrics": {}`, `"artifacts": {}`
and no `nas` key, despite `"status": "completed"`. The archive directory holds **only**
`run.json` — no `run.log`. This is the root cause of D-B, D-C and D-D below, so it is the one
worth fixing first.

`RunArchive` has tests (`RunArchiveTests`), so those presumably exercise the archive mechanics
with synthetic input rather than asserting that a real completed run yields non-empty metrics.

### D-B. A completed run's tab shows two empty headings and no explanation

`RunTabView.swift:16-18` builds `logURL` as
`<project>/.platypus/runs/<id>/<artifacts.log ?? "run.log">`. That file does not exist for
these archives, so `HistoricalRunView`'s `guard let text = try? String(contentsOf: logURL)`
(`:115`) returns early, the parser stays empty, and `MetricCharts` renders "Loss" and
"Accuracy %" as bare headings over blank space. `MetricCharts.swift` has no `isEmpty` /
`ContentUnavailableView` path at all — there is no empty state to fall back to.

A user cannot tell "this run recorded no metrics" from "the chart failed to draw".

### D-C. A failed run explains nothing

A `failed` run opens the same view as a completed one: `Best —`, `Status failed`, `Device`, and
the same two empty headings. No error message, no exit status, no log excerpt, no pointer to
where the failure is recorded. For the one status where the user most needs a reason, none is
offered.

### D-D. An archived NAS run never reaches the NAS view

`RunTabView.swift:19-22` routes to `NASSearchView` only when `record.nas != nil`. The archive
does not record `nas`, so a NAS run opens the ordinary `HistoricalRunView` — the architecture
and search-result surfaces are unreachable for any historical run. For the same reason the
"Searched" badge in `RunsPanel.swift:23` (`if r.nas != nil`) never appears on a NAS run.

`NASSearchView` and `ArchitectureView` are therefore unverified end-to-end. Reaching them
requires launching a real NAS search, which was deliberately not started during this pass — it
is a long-running compute job on the user's machine.

### D-E. The Date column prints a run id, not a date

`RunsPanel.swift:28` is `Text(String(r.id.prefix(13)))`, which renders `20260711-1956`. The
manifest carries a real ISO `startedAt` (`"2026-07-11T22:56:03Z"` in the file above), so a
formatted date is available and simply not used.

### D-F. Compare could not be enabled; multi-select did not respond

`RunsPanel` is written for multi-select — `@State private var selection = Set<RunManifest.ID>()`
and `Table(session.runs, selection: $selection)` — and `canCompare` requires 2+ runs of one
task type. In the running app, neither cmd-click nor shift-click extended the selection: each
click replaced it and the Compare button stayed disabled, leaving `CompareView` unreachable.

**Cause unconfirmed.** A candidate is the `.simultaneousGesture(TapGesture(count: 2))` attached
to the Run column at `:26`, but the click automation used for this pass is also a plausible
explanation, and the two were not separated. Worth reproducing by hand before changing code.

### D-G. Train form wastes most of the window width

At full-screen width the "Create a run" form places Configuration, Hyperparameters and Pipeline
in a narrow right-hand column and leaves roughly the left 60% of the pane empty. Cosmetic, and
only visible when the window is large.

### What did work

The Train tab's NAS mode switch behaves correctly: selecting "Search for a model" swaps the
button to "Start Search", removes the Model row (NAS supplies it), and reveals a Search section
with Size, an Optimize Memory/Compute toggle, and Search epochs. Model and feature-extraction
pickers populate from `mmcli info`. Dataset Overview renders real analysis (694,722 samples,
2 classes, class-distribution chart).

### Suggested shape of the follow-up

A PlatypusStudio verification phase, not a phase 10 plan. It would need to fix D-A first (an
archive that records nothing cannot be displayed), then give the run views honest empty and
error states, then decide whether the executable target gets a test target at all — noting that
the 10-04 checkpoint found a defect (a cancelled download rendering a traceback) that all 138
unit tests passed, which is the argument for driving the UI rather than only testing MMCLIKit.

Related deferred items already recorded in `10-CONTEXT.md`: `ProjectScanner.scan` silently
dropping unreadable directories, and ad-hoc signing resetting privacy grants on every rebuild —
the latter makes any repeatable UI verification painful until it is addressed.
