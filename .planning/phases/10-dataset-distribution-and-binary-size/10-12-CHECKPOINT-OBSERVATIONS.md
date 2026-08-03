# 10-12 Task 3 — checkpoint observations (agent-driven)

Driven by the agent via computer-use against the real app on the built-in Retina display,
2026-08-03 13:35-13:41. App: `PlatypusStudio/dist/PlatypusStudio.app`, rebuilt by the executor.
mmcli at `~/Library/Application Support/PlatypusStudio/bin/mmcli` verified byte-identical
(sha256 `3fa773e7194005ff…`) to the freshly built `tinyml-cli/dist/mmcli`.

Nothing below is inferred. Each line is what appeared on screen or on disk.

## Results

| # | Check | Result |
|---|-------|--------|
| 1 | Determinate progress, New Project sheet | **FAIL** |
| 2 | Success confirmation | **PASS** |
| 3 | Common case (already-local dataset) renders nothing | **PASS** |
| 4 | Cancel mid-transfer | **PASS** — first ever verification |
| 5 | Determinate progress, Manage Datasets | NOT RUN |
| 6 | Bulk Cancel All | NOT RUN |
| 7 | Integrity repair surfaces in GUI | NOT RUN |
| 8 | No JSON leaks into the UI | **PASS** |

## 1. FAIL — the byte counter never moves

Selected `fan_blade_fault` (uncached, 56.6 MB), clicked Download, sampled the row repeatedly.

Observed timeline:
- t≈1s, 2s, 3s: spinner glyph + **`Zero KB of 56.6 MB`** + Cancel button. Identical at all three samples.
- t≈4s, 6s: row reverted to **`This dataset is not on this machine yet.` + `Download (56.6 MB)`**
- t≈9s onward: **`Downloaded (56.6 MB) ✓`**

No filling bar was ever displayed — only the indeterminate spinner. `bytesTransferred` stayed 0
for the entire transfer.

**The producer side is NOT at fault.** Verified directly, piping stderr exactly as the app does:

```
635.654  {"v":1,"event":"start","dataset":"fan_blade_fault","total_bytes":56595859}
635.685  {"v":1,"event":"progress",...,"bytes":65536,...}
635.723  {"v":1,"event":"progress",...,"bytes":1114112,...}
635.814  {"v":1,"event":"progress",...,"bytes":2162688,...}
...
```
Events arrive ~90 ms apart with incrementing byte counts, correctly flushed through a pipe.

**Diagnosis — the label itself is the evidence.** `Zero KB of 56.6 MB` proves the `start` event
WAS ingested (that is where `total_bytes` comes from) while `bytesTransferred` never advanced.
So events reach the process but the `progress` folds never reach the view.

Most probable cause: `NewProjectSheet` is a plain `struct … : View` with no `@MainActor`, and
`private func download(_:) async` (`NewProjectSheet.swift:196`) is therefore **nonisolated** — a
nonisolated `async` function does not run on the caller's actor. `transfer.ingest(line.text)`
inside `for await line in proc.lines` mutates `@State` off the main actor, so SwiftUI does not
publish the intermediate updates. The terminal assignments still land because nothing overwrites
them, which matches exactly what was observed: frozen counter, correct final state.

Suggested fix: mark `download(_:)` `@MainActor` (or hop to the main actor per ingested event).
This is unverified — it is a hypothesis from reading the code, not a tested fix.

`ProcessRunner` is fine: it streams via `AsyncStream` with blocking `availableData` reads on a
dedicated drain thread (`ProcessRunner.swift:104-130`), not `readDataToEndOfFile`.

## Secondary finding (new, not in the plan) — post-transfer flicker

Between the transfer finishing and `refreshAvailability()` completing, the row reverts to
`This dataset is not on this machine yet.` + `Download (56.6 MB)` for ~2-3 s before the
confirmation appears. `downloading` is set to nil before availability refreshes, so the
`.downloadable` else-branch renders in the gap. A user watching sees the download appear to
fail and reset, then succeed. Same "looks broken" class as the original UAT gap.

## Cosmetic

`Zero KB` is `ByteCountFormatter`'s rendering of 0 bytes. Even once the counter works, the
opening frame will read "Zero KB of 56.6 MB" rather than "0 MB of 56.6 MB".

## 2. PASS — success confirmation

`Downloaded (56.6 MB) ✓` appeared where the Download row had been. The dataset picker remained
populated and functional.

## 3. PASS — common case unchanged

Closed and reopened the sheet, selected the now-cached `fan_blade_fault`: nothing rendered
between the picker and the Project name field. The `Downloaded ✓` confirmation correctly did NOT
persist across sheet sessions — the session-scoping works, and `arc_fault_classification`
(cached, default selection) likewise rendered no row on open.

## 4. PASS — cancel mid-transfer (first verification in this project)

10-09 recorded this INCONCLUSIVE because every transfer finished faster than a click. With
`fan_blade_fault` at 56.6 MB it was reachable. Clicked Download, then Cancel ~1 s in:
- Row returned immediately to `This dataset is not on this machine yet.` + `Download (56.6 MB)`
- No Python traceback, no red error line, sheet intact
- `mmcli datasets list` → `fan_blade_fault  downloadable`
- Cache directory: **no `.part`, `.fetch-` or temp files left behind**

Note: an *unrelated* interrupted transfer (a SIGPIPE from the agent's own `head -20` on a CLI
run) also left no partial file, independently corroborating the cleanup path.

## 8. PASS — no JSON leaked

No string beginning `{"v":1` appeared anywhere on screen across every screenshot taken.

## Not run

Checks 5, 6 and 7 were not exercised. Check 5 (library progress) is expected to share check 1's
defect since both surfaces consume the same stream; 6 (bulk Cancel All) and 7 (integrity repair
in the GUI) remain genuinely unverified and should not be recorded as passing.

## Agent interference disclosed

- The agent's own CLI test (`… --progress-json | head -20`) SIGPIPE'd a transfer, leaving
  `fan_blade_fault` uncached. A first attempt at check 3 was invalidated by this and was
  re-run correctly afterwards rather than reported.
- Cache was left whole: 9 mirrored datasets cached, same as before the session.
- A project named `asdasdasdasda` exists in the project list from the user's earlier manual
  Create-button test; it was not created by this checkpoint.
