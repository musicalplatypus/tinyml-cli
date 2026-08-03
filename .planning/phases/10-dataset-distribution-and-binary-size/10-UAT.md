---
status: complete
phase: 10-dataset-distribution-and-binary-size
source:
  - 10-01-SUMMARY.md
  - 10-02-SUMMARY.md
  - 10-03-SUMMARY.md
  - 10-04-SUMMARY.md
  - 10-05-SUMMARY.md
  - 10-06-SUMMARY.md
  - 10-07-SUMMARY.md
  - 10-08-SUMMARY.md
  - 10-09-SUMMARY.md
  - 10-10-SUMMARY.md
started: 2026-08-03T12:00:00Z
updated: 2026-08-03T13:30:00Z
---

## Current Test

[testing complete]

## Test Environment Note

Tests 1-8, 11 and 12 were executed by the agent against the real CLI, not self-reported.
**`dist/mmcli` was stale when testing began** — built 2026-08-02 15:53, which predates
`f580a9f` (17:49, CLI help) and `306310a` (19:05, zip-slip containment guard), i.e. it
predated all 23 code-review fixes. It was rebuilt from current source
(2026-08-03 07:33, 25,260,384 bytes) and every affected test re-run against the fresh
artifact. Results below are against the rebuilt binary.

**Relevant to tests 9-10:** PlatypusStudio resolves mmcli from
`~/Library/Application Support/PlatypusStudio/bin/mmcli`, which is a *separate copy*. Refresh
it from the newly built `dist/mmcli` before running the GUI tests, or the app will exercise
the same stale pre-fix binary.

## Tests

### 1. Cold start from an empty cache
expected: With the cache cleared, `dist/mmcli --version` and `dist/mmcli datasets list` both work. Binary boots in a few seconds. All ten datasets listed — nine downloadable, one bundled. No traceback or hang.
result: pass
evidence: "Ran with XDG_CACHE_HOME pointed at an empty temp dir (the user's real ~125 MB cache was deliberately NOT cleared). `--version` → `mmcli 1.1.2` in 7s; `datasets list` → all 10 rows, 9 downloadable + 1 bundled, in 6s. No traceback. Re-run on the rebuilt binary: 7s."

### 2. Binary is small and starts acceptably
expected: `dist/mmcli` is roughly 25 MB — not the old ~260 MB, and not over the 26 MiB ceiling. It starts in well under 8 seconds. The training engine (torch, modelmaker) is not inside it.
result: pass
evidence: "Rebuilt binary 25,260,384 bytes (24.09 MiB) vs ceiling 27,262,976 (26.00 MiB) — under. Zero `torch/` or `tinyml_modelmaker/` entries in the archive. Startup 7s against the <8s bound."
note: "7s leaves ~1s of margin on an 8s bound. ROADMAP's own doc-audit already flags this bound as tighter than it looks (10-03 recorded 6.6-9.6s on this machine). Not a failure; a thin margin worth watching."

### 3. `mmcli datasets list` reports honest state
expected: Lists all ten datasets with a state per dataset that matches reality — cached ones say cached, uncached say downloadable, and the bundled one says bundled. `--format json` produces valid JSON with the same information.
result: pass
evidence: "Against the real cache: table showed arc_fault=cached, pir_detection=downloadable, generic_audio=bundled — matching actual disk contents verified independently by `ls ~/.cache/mmcli/datasets/01_03_00/`. `--format json` parsed cleanly: 10 entries, Counter({'cached': 8, 'bundled': 1, 'downloadable': 1})."

### 4. `mmcli datasets pull` fetches and verifies
expected: `mmcli datasets pull pir_detection` downloads from the project's GitHub release mirror with visible progress, verifies the sha256, and lands it in the cache. Re-running without `--force` makes no network request. A corrupted or tampered download is rejected rather than accepted.
result: issue
reported: "Tamper handling is correct but silent. Appending bytes to a cached zip then re-running `datasets pull` printed only the normal success line; afterwards the on-disk sha256 matched the registry again, so the corruption WAS detected and repaired. The user is given no indication that the cached copy was bad and re-fetched."
severity: minor
evidence: "pull of uncached pir_detection → cached + exit 0. Re-run without --force → no network request. Corrupted entry → digest d75470c9ba7f56fd matches registry after the run, i.e. silently re-downloaded."

### 5. `mmcli datasets remove` removes only what it should
expected: `mmcli datasets remove <name>` deletes a cached copy and reports the space freed. It refuses to touch the packaged dataset inside the binary, and refuses when MMCLI_DATASETS supplies the file — printing a clear NOTE rather than deleting your own data.
result: pass
evidence: "Cached removal reported '1579936 bytes freed'. Bundled dataset → 'is not cached; nothing to remove' (packaged copy untouched). Under MMCLI_DATASETS it printed the D-11 NOTE that removing the cache entry does not change what the name resolves to, then removed only the ~/.cache entry — the MMCLI_DATASETS-supplied file was not touched."
note: "The test's wording ('refuses') was stricter than the actual design. The security-relevant guarantee — never delete a packaged or MMCLI_DATASETS-supplied file — holds. Removing only the inert cache entry while explaining it changes nothing is defensible and arguably better than a blanket refusal."

### 6. `mmcli init --dataset` produces a usable project
expected: `mmcli init --dataset <name> -t <task> -p <dir>` creates the project, extracts the dataset into `<dir>/dataset/`, and prints the next-step train command. In a terminal a missing dataset is auto-fetched; piped/scripted it refuses and tells you the exact `mmcli datasets pull` command instead of silently downloading megabytes.
result: pass
evidence: "Project created; 79 files extracted under dataset/ with classes/ and annotations/ present; train command printed. Non-interactive invocation of an uncached dataset refused: 'stderr is not a terminal, so mmcli will not start an unnarrated download' plus the exact `mmcli datasets pull pir_detection` recovery command."

### 7. Offline / air-gapped path works
expected: With MMCLI_DATASETS pointing at a directory of dataset zips and the network unreachable, `mmcli init --dataset` still works for every dataset present, and no command attempts a download. The README's offline recipe can be followed as written.
result: pass
evidence: "With http_proxy/https_proxy pointed at unroutable 192.0.2.1:9 and MMCLI_DATASETS staged, init extracted 79 files successfully. A missing dataset refused immediately (no hang) with a message naming the variable and the two ways out."

### 8. CLI help tells the truth
expected: `mmcli datasets pull --help` and `mmcli init --help` describe fetching from the project's GitHub release mirror — no leftover "from TI" wording. Each datasets subcommand shows a usage example. MMCLI_DATASETS is documented in `mmcli --help` and matches the README.
result: pass
evidence: "On the rebuilt binary all four datasets subcommands show 1 example section each, and `datasets pull --help` reads 'Fetch a dataset from the project's GitHub release mirror'. An apparent 'from TI' hit was a false positive — the substring inside 'from TInyml-modelzoo examples'. Initially failed against the stale pre-f580a9f binary; see Test Environment Note."

### 8b. Zip-slip resistance of the shipped binary (added during testing)
expected: A malicious archive delivered through the deliberately un-digested MMCLI_DATASETS path cannot write outside the project directory.
result: pass
evidence: "Constructed a real zip with member '../../../../../../tmp/MMCLI_UAT_ZIPSLIP_PWNED.txt' and ran the REBUILT binary's `init --dataset` against it. No file was written at /tmp or anywhere within a 3-level filesystem scan; the benign member still extracted to dataset/classes/class_a/ok.csv. This is the first time CR-02's containment guard has been exercised through the shipped entry point rather than a unit test."

### 9. PlatypusStudio — New Project download affordance
expected: In the app's New Project sheet, picking an uncached dataset offers to download it, shows progress, and can be cancelled without leaving a broken project or a traceback. After download the project is created normally.
result: issue
reported: "Download didn't show any progress. Then the dataset picker line just disappeared."
severity: minor
evidence: |
  The download itself SUCCEEDED — fan_blade_fault.zip, 56,595,859 bytes, written to
  ~/.cache/mmcli/datasets/01_03_00/ at 09:30, and `datasets list` now reports it cached.
  This is a feedback defect, not a functional one.

  Symptom 1 (no progress) — by construction. NewProjectSheet.swift:88-91 pairs an
  INDETERMINATE `ProgressView()` with `downloadLog.last ?? "Downloading \(bytes)…"`.
  `downloadLog` is fed from mmcli's stderr, but the sheet pipes stderr (its own comment at
  :124 — "its stderr is piped here, never a terminal"), and mmcli deliberately suppresses
  progress output on a non-TTY under the D-5 policy. So downloadLog never receives a line,
  the fallback string never changes, and a 54 MB transfer shows a spinner plus one static
  sentence. No percentage, no byte counter, no determinate bar.

  Symptom 2 (row vanished) — intended, but indistinguishable from a crash. On completion
  availability flips .downloadable -> .cached and :83-84 renders `EmptyView()`. The affordance
  disappears with NO success confirmation.

  AMBIGUITY RESOLVED (user screenshot): the Dataset dropdown is INTACT and still shows
  fan_blade_fault. Only the availability row vanished — the by-design :83-84 EmptyView() path.
  This is a feedback/UX defect, NOT a functional bug. Severity downgraded major -> minor.

  Incidental positives confirmed by the same screenshot:
  - `refreshAvailability()` (:151) WORKS. The row could only vanish if the post-download
    catalog reload saw fan_blade_fault flip .downloadable -> .cached. That path ("Create can
    unlock without reopening the sheet") had never been verified before; it is now.
  - `Create` greyed out is CORRECT, not a defect: `canCreate` (:122) requires
    `!projectName.isEmpty`, and the field was empty in the screenshot. User then confirmed
    with a second screenshot that entering a project name enables Create immediately.

  FUNCTIONAL PATH FULLY VERIFIED end-to-end, without reopening the sheet:
  uncached dataset -> Download affordance appears -> download succeeds and verifies ->
  refreshAvailability() flips state -> row clears -> Create unlocks on name entry.
  Every part of REQ-UX-01's user-visible contract holds. The defect is confined to feedback
  DURING and AT THE END of the transfer.

  STILL GENUINELY UNVERIFIED: cancel-mid-transfer. The user did not report clicking Cancel,
  and the transfer completed. This remains the same gap 10-09 recorded as INCONCLUSIVE.

  RECOMMENDED FIX (two small, independent changes, both in NewProjectSheet.swift):
  1. Determinate progress. mmcli suppresses progress on a non-TTY by design (D-5), so the
     sheet cannot get byte counts from stderr as currently wired. Either have the sheet poll
     the partial file size in the version-scoped cache dir, or give mmcli a
     machine-readable progress mode (e.g. `--progress-json` on stderr) that does not depend
     on isatty. The second is cleaner and reusable by the dataset library sheet too.
  2. Success confirmation. Replace the bare EmptyView() for a dataset that was downloaded
     *in this session* with a transient "Downloaded (54.0 MB) ✓" line, so a completed
     transfer is distinguishable from a crash. Keep EmptyView() for datasets that were
     already local on open, preserving the "common case stays a single click" intent.

### 10. PlatypusStudio — Manage Datasets library
expected: The Manage Datasets surface is reachable from the workspace toolbar with no project open. It lists every dataset with size and state, downloads and removes them, and shows a cache-size footer that updates. Removal never touches a packaged or MMCLI_DATASETS-supplied file.
result: pass
evidence: "Confirmed by the user driving the real app. Closes the app-side half of REQ-UX-02; the CLI-side removal boundary (never touches a packaged or MMCLI_DATASETS-supplied file) was verified independently in test 5."

### 11. Release process is followable
expected: A maintainer can follow `docs/RELEASING.md` end to end. `python3 scripts/release_preflight.py` runs and reports mirror + digest status, exiting non-zero if the mirror is wrong. The doc is explicit that publishing the mirror release is a human step and that a published release is never deleted.
result: pass
evidence: "Preflight run live against the public mirror: 'All 9 fetchable dataset(s) PASSED. PREFLIGHT PASSED', exit 0. RELEASING.md §4 'Publish the mirror release — human-only' and §9 'Mirror releases are never deleted' both present and explicit."

### 12. pip install stays small
expected: The wheel and sdist no longer carry the nine mirrored datasets — roughly 0.1 MB rather than ~108 MB. A clean `pip install` of the package still resolves datasets by fetching them from the mirror on demand.
result: pass
evidence: "Built a real wheel: mmcli-1.1.2-py3-none-any.whl, 110,164 bytes (0.10 MB), down from ~108 MB. Only dataset inside is example_datasets/generic_audio_classification.zip at 18,371 bytes — the locally-authored bundled one. All nine mirrored datasets absent, matching the BUNDLED_DATASETS allowlist."
note: "`python -m build` is not installed in the project venv; used `pip wheel . --no-deps` instead. Same artifact, different front-end."

## Summary

total: 13
passed: 11
issues: 2
pending: 0
skipped: 0

## Gaps

- truth: "A corrupted or tampered cached dataset is rejected rather than accepted, and the user can tell it happened"
  status: failed
  reason: "User-observable half is missing. Corruption IS detected and silently repaired by re-download (verified: post-run sha256 matches the registry again), but `datasets pull` prints only its normal success line. A user cannot distinguish a clean cache hit from a corrupted entry that was quietly re-fetched. The security property holds; the reporting property does not."
  severity: minor
  test: 4
  artifacts: []
  missing: []

- truth: "A dataset download in PlatypusStudio's New Project sheet shows progress, and its completion is distinguishable from a failure"
  status: failed
  reason: "User reported: 'Download didn't show any progress. Then the dataset picker line just disappeared.' The transfer succeeded (56,595,859 bytes cached) and every functional step verified, but a 54 MB download renders only an indeterminate spinner beside one static string, and on completion the affordance row is replaced by EmptyView() with no success confirmation. Root cause of the missing progress: the sheet feeds downloadLog from mmcli's stderr, but pipes that stderr, and mmcli suppresses progress on a non-TTY under the D-5 policy — so no progress line can ever arrive through the current wiring."
  severity: minor
  test: 9
  artifacts:
    - ../PlatypusStudio/Sources/PlatypusStudio/NewProjectSheet.swift
  missing:
    - "Determinate progress: either poll the partial file size in the version-scoped cache dir, or add a machine-readable progress mode to mmcli (e.g. --progress-json on stderr) that does not depend on isatty. The second is reusable by the dataset library sheet."
    - "Transient success confirmation for a dataset downloaded in this session, instead of a bare EmptyView(); keep EmptyView() for datasets already local on open."
