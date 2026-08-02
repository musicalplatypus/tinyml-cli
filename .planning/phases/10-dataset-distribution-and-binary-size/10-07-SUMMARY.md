---
phase: 10-dataset-distribution-and-binary-size
plan: 07
subsystem: docs
tags: [release-process, cli-help, sphinx, github-releases, gh-cli]

requires:
  - phase: 10-03
    provides: "the GitHub release mirror (datasets-<version>), dataset_url()/fetch_dataset(), the mirror-healthcheck pattern release.yml already runs"
  - phase: 10-05
    provides: "the corrected README.md MMCLI_DATASETS env-var wording this plan quotes verbatim in mmcli/cli.py"
  - phase: 10-09
    provides: "mmcli datasets remove and its MMCLI_DATASETS NOTE, documented here"
provides:
  - "docs/RELEASING.md — the dataset release-process document REQ-DOC-01/REQ-DATA-05 needed, folding in CONTEXT.md D-04/D-05/D-06"
  - "scripts/release_preflight.py — D-05's scripted preflight (mirror tag/asset check + full digest gate), executed for real, not just described"
  - "corrected CLI help text: no more stale \"from TI\" wording anywhere in mmcli/cli.py"
affects: []

tech-stack:
  added: []
  patterns:
    - "release_preflight.py reuses release.yml's mirror-healthcheck gh invocation verbatim rather than reimplementing it, so a local preflight failure matches the CI failure a maintainer would otherwise wait for"

key-files:
  created:
    - docs/RELEASING.md
    - scripts/release_preflight.py
  modified:
    - mmcli/cli.py

key-decisions:
  - "scripts/release_preflight.py was added even though the plan's files_modified only names docs/RELEASING.md, docs/mmcli.rst and mmcli/cli.py — CONTEXT.md D-05 (which postdates the plan) requires the ordered checklist to be backed by an actual scripted preflight, not left as prose, and explicitly says it should be executed, not just written."
  - "docs/mmcli.rst needed no edit: dataset_url()/fetch_dataset() are already public with docstrings good enough for the existing :members: autodoc; _resolve_dataset_zip()/_cache_dir() stay private (leading underscore) and are correctly excluded without a :private-members: directive, which the plan's own read_first list treats as needing autodoc but which is not what the existing directive does — confirmed by a real -W build showing 0 warnings before and after this plan's cli.py edits."

requirements-completed: [REQ-DOC-01, REQ-DATA-05]

duration: ~50min
completed: 2026-08-02
---

# Phase 10 Plan 07: Release-process documentation and CLI help corrections Summary

**Wrote `docs/RELEASING.md` (the release-process document Phase 10 never had) with a real, executed D-05 preflight script (`scripts/release_preflight.py`), and corrected both stale "from TI" strings in `mmcli/cli.py`'s help text, verified by invoking the real CLI.**

## Performance

- **Duration:** ~50 min
- **Tasks:** 2 (plus one Rule 2 addition — the preflight script — required by CONTEXT.md D-05)
- **Files modified:** 3 (2 created, 1 modified)

## Accomplishments

- `docs/RELEASING.md` (242 lines): nine numbered sections covering the plan's seven original
  release obligations plus CONTEXT.md D-04 (never delete a mirror release — track deprecation in
  a table instead) and D-06 (the mirror publish is human-only; document why and refuse to run it
  even here). Names `scripts/verify_dataset_digests.py` and `scripts/binary_size_ceiling.txt` by
  path, as the plan's automated gate requires.
- `scripts/release_preflight.py` (new): D-05's scripted preflight. Step 1 re-runs the exact `gh
  release view <tag> --json tagName,assets` check `release.yml`'s `mirror-healthcheck` CI job
  already runs (reused, not reimplemented). Step 2 runs `scripts/verify_dataset_digests.py` as a
  subprocess — the real ~131 MB GET-and-hash gate over all nine fetchable datasets via
  `fetch_dataset(name, force=True)`, the same function every real `mmcli datasets pull` runs.
- **Ran the preflight script for real**, twice: once with `--skip-digests` (fast tag/asset check
  only), once the full run. Both against the live mirror (`datasets-01_03_00`,
  `musicalplatypus/tinyml-cli`). Full run: mirror tag/assets OK, all 9 fetchable datasets `PASS`,
  script exit 0. Real output recorded verbatim (with per-dataset PASS lines elided and that
  elision stated) in `docs/RELEASING.md` §5. The failure path was independently confirmed the
  same session: `gh release view datasets-99_99_99 --repo musicalplatypus/tinyml-cli` (a
  nonexistent tag) returns `release not found`, exit 1 — the exact condition the script's step 1
  turns into a `FATAL:` line.
- Corrected both stale "from TI" strings deferred-items.md flagged: the `datasets pull`
  subparser description and `init --fetch`'s help text now say "the project's GitHub release
  mirror" / "the GitHub release mirror", matching the actual fetch source since 10-03's repoint.
  Also fixed the same staleness in `datasets list --help`'s state-table row for `downloadable`
  ("has a TI source" → "fetchable from the GitHub release mirror"), found while this file was
  open — not separately flagged in deferred-items.md, but the identical defect.
- Added a usage `Example:` to each of `datasets list/pull/path/remove --help` (previously only
  the parent `datasets --help` had examples; each action's own `--help` did not). `remove` was
  not named in the plan text (it postdates the plan, added by 10-09) but is documented here
  since the plan's own note said 10-07 should cover it and `mmcli/cli.py` was already open.
- `init --help`'s description now states the D-5 auto-fetch policy in one line, pointing at
  `--fetch`/`--no-fetch`, "since that is where a scripted user meets it" (plan's own phrasing).
- Added an `MMCLI_DATASETS` row to the top-level `mmcli --help` env-var block. It was previously
  **entirely absent** from that block (only `MMCLI_PYTHON`/`MMCLI_MODELMAKER` were documented
  there), not merely stale — despite STATE.md's 02-05 session note claiming it had been added to
  a "module-level docstring"; that claim does not describe the `--help` block main() actually
  builds (verified by reading the real argparse `description=` construction, not the note). The
  new row's description sentence is matched verbatim to README.md's env-var table row for
  `MMCLI_DATASETS` (T-10-07-02's mitigation), with the same default-value wording.

## Task Commits

1. **Task 1: Write docs/RELEASING.md** — `d49e679` (docs) — includes `scripts/release_preflight.py`
2. **Task 2: CLI help and API docs** — `f580a9f` (docs)

## Files Created/Modified

- `docs/RELEASING.md` — release-process checklist: version-bump decision, digest contract, the
  digest gate, human-only mirror publish, scripted preflight (with real recorded output),
  new-dataset procedure, pre-announce clean-cache verification, binary-size ceiling policy,
  never-delete-a-mirror-release table, `datasets remove` cross-reference, and an explicit
  "why the order matters" closing section.
- `scripts/release_preflight.py` — D-05's scripted preflight (mirror tag/asset check +
  `verify_dataset_digests.py` subprocess), executable, run for real (see above).
- `mmcli/cli.py` — stale "from TI" wording corrected in 3 places; usage examples added to 4
  `datasets` actions; D-5 one-liner added to `init --help`; `MMCLI_DATASETS` added to the
  top-level env-var help block, matched verbatim to README.md.

## Decisions Made

- **Added `scripts/release_preflight.py` despite it not being in the plan's `files_modified`.**
  CONTEXT.md D-05 (which the executor's Required Reading list explicitly says postdates and
  expands the plan) requires an actual scripted preflight, not a description of one, and requires
  it be executed for real — both done. Documented here rather than silently expanding scope: this
  is a Rule 2 auto-add (missing critical functionality mandated by context that supersedes the
  plan text), not a deviation from what was actually asked of this plan.
- **Did not touch `docs/mmcli.rst`.** The plan's `files_modified` lists it, but no change was
  needed: `mmcli.datasets`'s existing `:members: :undoc-members: :show-inheritance:` autodoc
  directive already picks up `dataset_url()` and `fetch_dataset()` (both public, both already
  well-docstringed from 10-02/10-03), and the plan's own read_first list names
  `_resolve_dataset_zip`/`_cache_dir` as needing docs — both are private (leading underscore) and
  are correctly excluded from autodoc without a `:private-members:` directive, which this plan
  does not add (adding one would pull in every other private helper in the module, well beyond
  this plan's scope). Confirmed by running the real `-W` build both before and after the
  `mmcli/cli.py` edits: 0 warnings, exit 0, in both cases.
- **Did not touch the `mmcli/cli.py` module-level docstring** (lines 1-24, Python docstring, not
  the argparse `--help` text). It documents `MMCLI_PYTHON`/`MMCLI_MODELMAKER` only. The plan's
  key_link (T-10-07-02) names the argparse env-var help block specifically ("mmcli/cli.py env var
  help" → "README.md env var table"), which is what `mmcli --help` actually shows a user; the
  module docstring is a secondary, Sphinx-only surface not covered by the plan's must-haves. Left
  alone to keep this plan's scope to what was asked.
- **Left `unplanned-work.md`'s and `10-DOC-AUDIT.md`'s figures uncited in `docs/RELEASING.md`
  without a fresh re-measurement.** Per the audit's M-3 finding, binary size varies build to
  build; the doc explicitly says not to treat any single figure as exact and points to the audit
  finding rather than repeating one of the four disagreeing numbers as if it were authoritative.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] `scripts/release_preflight.py` did not exist; wrote and ran it**
- **Found during:** Task 1 (docs/RELEASING.md)
- **Issue:** The plan text (pre-CONTEXT.md) only asked for a documented checklist. CONTEXT.md
  D-05, which the executor's mandatory reading list says postdates and expands the plan, requires
  an actual scripted preflight enforcing the publish-then-verify-then-build order, and requires
  it be executed for real against the live repo state.
- **Fix:** Wrote `scripts/release_preflight.py` (two-step: mirror tag/asset check via `gh
  release view`, then `scripts/verify_dataset_digests.py` as a subprocess). Ran it twice for
  real against the live `datasets-01_03_00` mirror; both runs' actual output is recorded in
  `docs/RELEASING.md`.
- **Files modified:** `scripts/release_preflight.py` (new), `docs/RELEASING.md`
- **Verification:** `python3 scripts/release_preflight.py` exit 0 (full run, 9/9 digests PASS);
  `python3 scripts/release_preflight.py --skip-digests` exit 0 (tag/asset-only run); failure path
  independently confirmed via a direct `gh release view` call against a nonexistent tag.
- **Committed in:** `d49e679`

**2. [Rule 1 - Bug] `datasets list --help`'s "downloadable" state description also said "TI"**
- **Found during:** Task 2 (CLI help corrections)
- **Issue:** Same staleness class as the two strings deferred-items.md named
  ("has a TI source but is not present locally"), just not separately flagged there — found
  while `mmcli/cli.py` was open for the flagged fixes.
- **Fix:** Reworded to "fetchable from the GitHub release mirror, not present locally".
- **Files modified:** `mmcli/cli.py`
- **Verification:** `python -m mmcli datasets list --help` invoked for real, output confirmed.
- **Committed in:** `f580a9f`

**3. [Rule 2 - Missing Critical] Removed 9 stray stand-in dataset zips this session's own
concurrent test runs left behind, out of scope for a commit but corrupting local test state**
- **Found during:** post-Task-2 verification (running `tests/test_datasets_cli.py`)
- **Issue:** Running the same pytest invocation twice concurrently (to verify no regression)
  raced against `test_datasets_cli.py`'s `dataset_zips_present` autouse fixture, which
  materialises a digest-matching stand-in zip for any dataset missing from
  `mmcli/example_datasets/` and monkeypatches that entry's registry `sha256`/`bytes` to match —
  cleaning up only the files *it itself* created. The concurrent second process found files the
  first process had already created, skipped creating (and therefore skipped monkeypatching) its
  own copies, and left 9 gitignored stand-in zips (173-239 bytes each) on disk after both
  processes exited, with registry digests no longer monkeypatched to match. This broke every
  subsequent test run in this working tree, including tests unrelated to my changes.
- **Fix:** Deleted the 9 stray stand-in files (`arc_fault_classification.zip`,
  `ecg_classification.zip`, `fan_blade_fault.zip`, `generic_timeseries_anomalydetection.zip`,
  `generic_timeseries_classification.zip`, `generic_timeseries_forecasting.zip`,
  `generic_timeseries_regression.zip`, `mnist_image_classification.zip`,
  `pir_detection.zip`), all gitignored and untracked, restoring the checkout to only the one
  real tracked bundled zip (`generic_audio_classification.zip`).
- **Files modified:** none tracked (all deleted files were gitignored/untracked; `git status`
  confirmed clean before and after)
- **Verification:** `python -m pytest tests/test_datasets_cli.py -q` — 39 passed, 2 skipped
  (the 2 skips are the intended `_needs_real_zips`-marked tests, which correctly skip on a
  checkout without real dataset zips, not a regression).
- **Committed in:** N/A — no tracked files changed; this was local working-tree hygiene, not a
  plan deliverable, and out of this plan's own file scope. Recorded here per the executor's
  scope-boundary rule (fixed inline since it was self-caused and blocking further verification,
  rather than left for `deferred-items.md`, since leaving it would have silently broken the next
  session's test runs).

---

**Total deviations:** 3 (1 Rule 2 addition mandated by CONTEXT.md D-05, 1 Rule 1 wording fix,
1 Rule 2 local-state cleanup of this session's own test-run side effect).
**Impact on plan:** All within scope of what CONTEXT.md's Required Reading list already told this
plan to deliver, or self-inflicted and self-corrected. No scope creep beyond D-04/D-05/D-06 and
the deferred-items.md item this plan was explicitly told it owed.

## Issues Encountered

- **Sphinx is not installed in the project's `MMCLI_PYTHON` venv (`~/.venv-tinyml`)** — confirmed
  by import error. The system `python3` (`/opt/homebrew/bin/python3`, Sphinx 8.1.3) was used
  instead for the `-W` build, since Sphinx is a docs-build-time tool, not an mmcli runtime
  dependency, and this project convention is scoped to mmcli's own CLI test/run environment, not
  documentation tooling. Both runs (before and after the `cli.py` edits) exited 0 with 0
  warnings; commands and full output recorded in the task commit messages.
- **`gsd-sdk query task.is-behavior-adding`, MVP+TDD gate, and other SDK verbs were not invoked**
  — this is not a TDD-tagged plan (`type="auto"` throughout, no `tdd="true"` tasks), so the
  MVP+TDD gate does not apply.

## User Setup Required

None — no external service configuration required. Note per CONTEXT.md D-06: publishing or
re-publishing a `datasets-<version>` mirror release remains human-only
(`gh release create`/`gh release upload` refused by the agent permission classifier); this plan
did not attempt either, consistent with the verification requirements given to this executor.

## Next Phase Readiness

Phase 10 is now **10/10 planned plans complete** (10-04's checkpoint and 10-09's checkpoint were
both already driven to completion in prior sessions per `.continue-here.md`; this plan, 10-07,
was the only one remaining). `REQ-DOC-01` and `REQ-DATA-05` should now both be markable fully
met — `.planning/REQUIREMENTS.md`'s "partially discharged" notes for both named `docs/RELEASING.md`
as the missing piece, which now exists, is 242 lines, and passes the plan's own automated grep
gate. Per this plan's operating instructions, `.planning/REQUIREMENTS.md`, `STATE.md`, and
`ROADMAP.md` are **not** updated by this worktree agent — that is the orchestrator's job after
all wave agents complete.

No blockers. The one open item from `.continue-here.md`'s `<blockers>` section (10-09's
cancel-drops-queue behaviour unverified) is unrelated to this plan's scope and remains as
recorded there.

---
*Phase: 10-dataset-distribution-and-binary-size*
*Completed: 2026-08-02*

## Self-Check: PASSED

- `docs/RELEASING.md` — FOUND on disk
- `scripts/release_preflight.py` — FOUND on disk, executable bit set
- `.planning/phases/10-dataset-distribution-and-binary-size/10-07-SUMMARY.md` — FOUND on disk
- Commit `d49e679` (Task 1) — resolves via `git cat-file -t`
- Commit `f580a9f` (Task 2) — resolves via `git cat-file -t`
- All claimed real-command output (Sphinx `-W` build, `python -m mmcli ... --help` invocations,
  `python3 scripts/release_preflight.py` full run, `gh release view` failure-path check,
  `pytest tests/test_datasets_cli.py`) was executed in this session, not inferred; commands and
  results are reproduced verbatim in the task commit messages above and in `docs/RELEASING.md`.
