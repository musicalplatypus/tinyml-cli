# Phase 10 — Plan Review

**Date:** 2026-07-22
**Reviewer:** main agent, inline (goal-backward verification against ROADMAP.md requirements)

## Provenance of this review — read before trusting it

`gsd-plan-checker` did **not** produce this. It could not run: `CLAUDE_CODE_SUBAGENT_MODEL`
was set to `qwen3-coder:30b` while `ANTHROPIC_BASE_URL` pointed at `api.anthropic.com`, so
every subagent spawn 404'd. That env key has since been removed from `~/.claude/settings.json`,
so the checker should work in a fresh session — re-running it against these plans is still
worthwhile.

The same review payload **was** run against `qwen3-coder:30b` locally (24,318 prompt tokens).
It returned "VERIFICATION PASSED", 9/9 requirements "Covered", and found none of the findings
below. It marked REQ-DATA-02 as "Covered" despite being told in the same prompt that the `url`
field does not exist. Treat that run as no evidence either way.

The plans themselves were written inline rather than by `gsd-planner`, and have never been
through the checker.

## Verified as accurate (no action)

- The `datasets.py` seam is exactly as the plans describe: `_datasets_dir()` at line 18,
  `DATASET_REGISTRY` with 10 entries, `extract_dataset()` at line 166.
- All 10 zips present in `mmcli/example_datasets/`.
- `build_macos.sh` prints `mmcli modules bundled: N` (line 80), so 10-01's verify grep is real.
- 10-04's cross-repo paths resolve in PlatypusStudio; `DatasetCatalog.swift` and
  `DatasetCatalogTests.swift` are absent because 10-04 creates them.
- **The recorded sha256 digests match what TI serves at the version-pathed URLs.** The
  provenance table compared against the *flat* path while the plans pin the *versioned* one —
  a real gap, since those are different URLs. Spot-checked 4 files by download-and-hash
  (`generic_timeseries_forecasting` at flat / `01_03_00` / `01_04_00`, and
  `pir_detection_classification_dsk` at `01_03_00`). All matched. The reversal's central
  assumption holds.

## Findings

### F-1 — CI size ceiling would pass a 3x regression (10-01-PLAN.md:116)

Task 3 says "fail above 45 MB once 10-03 lands". 45 MB is a leftover from Option 2 (unbundle
only the two giants), which was **not** chosen. REQ-SIZE-01 is <= 15 MB and 10-03's own check
uses 15,728,640 bytes.

A build that silently re-bundled `arc_fault` + `ecg` + `mnist` would pass CI while violating
the phase's headline requirement. Change the post-10-03 ceiling to 15 MB.

Note: the two `145 MB` figures in the same file are correct and must stay — a naive
`grep "45 MB"` matches them as substrings.

### F-2 — REQ-DATA-02 is vacuous after the TI reversal (ROADMAP.md:49)

> Any registry entry carrying a `url` must carry a `sha256`, verified before extraction

After the reversal no entry carries `url` — the URL is derived by `dataset_url()` from
`ti_name`. The requirement is now trivially satisfied by a registry with no digests at all,
while the invariant the plans actually enforce (`ti_name` implies `sha256`) is stated nowhere
at requirement level.

Restate as: any entry carrying a `ti_name` must carry a `sha256`, verified before extraction.

### F-3 — 10-03 Task 1 verifies reachability, not content (10-03-PLAN.md)

Task 1 gates the build change and is titled "Record digests and confirm upstream availability",
but its `<automated>` block issues only HEAD requests. HEAD proves a URL resolves; it says
nothing about what it serves. Since the digests were computed from local files, the one failure
mode that breaks *every* user — TI serving different bytes at the pinned path — is exactly what
this gate cannot see.

Change to GET-and-hash for all nine fetchable datasets, comparing against the registry `sha256`.
Four were checked by hand (see above); the plan should not rely on that.

### F-4 — The documented offline recipe cannot be completed (10-05-PLAN.md:66)

Task 1 tells air-gapped users to "download the ten assets once". Only nine are downloadable.
`generic_audio_classification.zip` has no TI upstream — it exists only inside the binary and in
git. REQ-DATA-04 ("all 10 obtainable offline") therefore has no covering task for the tenth.

Fix in the plan text: nine come from `software-dl.ti.com`, the tenth from
`mmcli/example_datasets/` in the repo.

### F-5 — 10-02 and 10-04 disagree about who fetches (10-02-PLAN.md:186)

10-02 Task 4 makes `init --dataset X` auto-fetch when missing. 10-04 Task 2 disables Create and
requires an explicit "Download (54 MB)" click. Both cannot own the decision, and mmcli
auto-fetching underneath a GUI is precisely the silent multi-megabyte stall REQ-UX-01 exists to
prevent.

The layering note ("mmcli owns the download, the app owns the prompting") is the right
principle; 10-02 Task 4 contradicts it. This is a design decision, not a text fix — it needs an
explicit resolution before either plan executes.

Suggested: gate auto-fetch on `stderr.isatty()`. Interactive CLI keeps its ergonomics, CI stays
non-interactive and fails loudly, and the GUI path falls to the app's explicit affordance.

## Recommended action

Re-run `gsd-plan-checker` in a fresh session first — it may find more than this pass did, and
these plans have never had a real check. Then replan with `--reviews` so F-1 through F-5 are
addressed rather than re-derived.
