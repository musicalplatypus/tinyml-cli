---
phase: 10-dataset-distribution-and-binary-size
plan: 05
subsystem: dataset-distribution
tags: [documentation, readme, offline-recipe, pyinstaller, dataset-mirror]

requires:
  - phase: 10-03
    provides: "GitHub release mirror repoint, unbundle to one dataset, real ~31.84 MB / ~6.7-9.6s macOS measurement — the ground truth this plan documents"
  - phase: 10-06
    provides: "mmcli datasets list/pull/path CLI surface and the D-5 init --dataset auto-fetch policy this plan documents"
provides:
  - "README.md and README_zh.md with no false statement about dataset location, MMCLI_DATASETS semantics, or binary size"
  - "A README-only ten-dataset offline/air-gapped recipe, executed literally end-to-end and proven correct against the real dist/mmcli binary"
  - "Discovery + documented workaround for a PyInstaller onefile limitation: a bundled resource's printed path is invalid the instant the process exits, so 'datasets path' output cannot be piped into a following shell command"
affects: [10-07, 10-08]

tech-stack:
  added: []
  patterns:
    - "Materialize a PyInstaller-onefile-bundled resource via its own extraction side-effect (init --dataset writes real files to disk before the process exits) rather than trying to read the resource's printed temp path after the fact"

key-files:
  created:
    - .planning/phases/10-dataset-distribution-and-binary-size/deferred-items.md
  modified:
    - README.md
    - README_zh.md

key-decisions:
  - "Offline recipe's tenth-dataset step does NOT use 'datasets path' + cp as the plan's own Background section assumed. Verified live: PyInstaller onefile deletes its _MEI extraction directory synchronously at process exit, so by the time a shell command substitution `$(mmcli datasets path ...)` returns, the printed path already points at nothing. The recipe instead runs `mmcli init --dataset generic_audio_classification -t audio_classification -p <tmp>` (extraction happens while the process is alive, durable on disk) and re-zips the resulting dataset/ directory under the expected filename. This is safe because MMCLI_DATASETS-resolved files are explicitly not sha256-checked (asymmetric-by-design in mmcli/datasets.py's _resolve_dataset_zip), so a re-zipped, byte-different-but-content-identical copy resolves and extracts correctly."
  - "README_zh.md: corrected the two false facts (9-bundled-from-TI claim, ~10 MB size) directly in Chinese, but did not translate the new English Datasets section (resolution order, mmcli datasets CLI, D-5 policy, full offline recipe) — added an explicit visible pointer to README.md#datasets instead, per the plan's explicit instruction not to present a machine translation of that content as reviewed text."
  - "Binary size/startup claims use 10-03's measured figures (~31.8 MB, no startup number restated) rather than the phase's aspirational ~14/15 MB target, per explicit instruction not to let an unverified approximate number back into the README."

requirements-completed: [REQ-DOC-01, REQ-DATA-03, REQ-DATA-04]

duration: ~90min
completed: 2026-07-23
---

# Phase 10 Plan 05: README truth-up + ten-dataset offline recipe Summary

**Corrected two false README claims (9 bundled TI datasets, ~10 MB binary), added a Datasets section documenting the real GitHub-mirror source/resolution order/CLI surface/D-5 policy, and replaced an unworkable "print bundled path, then cp" offline-recipe step (discovered broken against the real PyInstaller onefile binary) with a working extract-and-re-zip approach — proven by literally executing the final committed recipe text twice against `dist/mmcli` with network disabled.**

## Performance

- **Duration:** ~90 min
- **Tasks:** 3/3 completed
- **Files modified:** 2 (`README.md`, `README_zh.md`); 1 file created (`deferred-items.md`)

## Accomplishments

- Removed every false statement this plan targeted: "9 bundled example datasets... downloaded from TI's servers" (README.md:12,490 pre-edit) and "lightweight binary (~10 MB)" (README.md:17 pre-edit) both corrected to the real, measured facts from 10-03 (one 18 KB bundled dataset + nine mirror-fetched; ~31.8 MB measured macOS build).
- Replaced `## Example Datasets` with `## Datasets`, covering: the mirror source and *why* it replaced TI (TI's CDN 302s to a 404 now), a ten-row table (all datasets including the previously-omitted `generic_audio_classification`), the explicit `MMCLI_DATASETS` → bundled → cache → download resolution order, the `mmcli datasets list/pull/path` surface, the D-5 auto-fetch policy (`--fetch`/`--no-fetch`, TTY gating, non-interactive refusal), and the full offline/air-gapped recipe.
- Fixed the env var table row for `MMCLI_DATASETS` (both README.md and README_zh.md): the stale "bundled `example_datasets/`" default and "Override directory" description are gone; the row now states the real default and that setting the variable disables fetching unconditionally.
- **Executed the offline recipe literally, twice, against the real `dist/mmcli` binary** (see below) — not merely documented. Found and fixed a real bug in the recipe's assumed approach for the tenth dataset before it was ever committed.
- README_zh.md: corrected the same two facts in Chinese and added a visible pointer to the English README's Datasets section for the parts not translated (per the plan's explicit instruction against presenting a machine translation as reviewed text).

## Task Commits

1. **Task 1: Correct and extend README.md** — `6d42add` (docs) — includes the offline-recipe fix that Task 2 required, because the broken step was discovered and corrected during drafting, before this commit was made (see "How Task 2 was actually executed" below).
2. **Task 2: Execute the offline recipe as written** — no separate commit. The literal execution surfaced the PyInstaller-onefile bug described below, which was fixed inline as part of Task 1's text before that commit landed; the subsequent literal re-run against the already-committed text (see verification below) required zero further changes, so there is nothing left for Task 2 to commit.
3. **Task 3: Keep the Chinese README honest** — `6280ad7` (docs)

## The PyInstaller-onefile bug this plan found (why Task 2 forced a doc rewrite)

The plan's own Background section assumed `mmcli datasets path generic_audio_classification` could be piped into `cp` to obtain the tenth dataset's zip without a repo checkout. Verified false against the real binary:

```bash
$ P=$(./dist/mmcli datasets path generic_audio_classification); cp "$P" /tmp/x.zip
cp: /var/folders/.../_MEIxxxxxx/mmcli/example_datasets/generic_audio_classification.zip: No such file or directory
```

`dist/mmcli` is a PyInstaller `--onefile` build: it extracts itself into a fresh `_MEI<random>` temp directory on each launch and deletes that directory synchronously when the process exits — before a shell command substitution `$(...)` even returns control to the calling script. So a "print the path" step can never be chained into a following command for a bundled resource in this build mode. Confirmed reproducible across three independent single-invocation tests, not a race.

**Fix applied to the README recipe:** instead of trying to read the resource after the process exits, materialize it through the process's own durable side effect. `mmcli init --dataset generic_audio_classification -t audio_classification -p <tmp>` extracts the zip into `<tmp>/dataset/` *while `mmcli` is still running* — that write survives process exit. The recipe then re-zips `<tmp>/dataset/` under the exact filename `MMCLI_DATASETS` expects. This works because `_resolve_dataset_zip()` in `mmcli/datasets.py` explicitly does **not** sha256-check files found via `MMCLI_DATASETS` (only bundled/cache-resolved files are digest-checked) — a re-zipped, byte-different-but-content-identical copy resolves and extracts without complaint.

## Literal recipe execution — commands run and outcomes

Performed twice: once while drafting (to find/fix the bug above), and once more as a final proof, extracting the exact bash blocks from the *already-committed* `README.md` text and running them verbatim with a clean cache and `mmcli` resolved via `PATH` (a symlink to `dist/mmcli`, matching "Copy `dist/mmcli` anywhere on your PATH" from the Setup section):

```bash
# Step 1-2: pull all nine, assemble cache copies
for n in arc_fault_classification ecg_classification fan_blade_fault \
         generic_timeseries_anomalydetection generic_timeseries_classification \
         generic_timeseries_forecasting generic_timeseries_regression \
         mnist_image_classification pir_detection; do
  mmcli datasets pull "$n"
done
mkdir -p ~/mmcli-offline-datasets
cp ~/.cache/mmcli/datasets/*/*.zip ~/mmcli-offline-datasets/

# Step 3: materialize + re-zip the tenth
mmcli init --dataset generic_audio_classification -t audio_classification -p /tmp/mmcli-audio-seed
(cd /tmp/mmcli-audio-seed/dataset && zip -qr ~/mmcli-offline-datasets/generic_audio_classification.zip .)
rm -rf /tmp/mmcli-audio-seed
```

Result: `ls ~/mmcli-offline-datasets/*.zip | wc -l` → **10**.

```bash
export MMCLI_DATASETS=~/mmcli-offline-datasets
mmcli datasets list --format json
```
Result: all 10 records report `"state": "bundled"`; none report `downloadable` or `unavailable`.

Then, with `http_proxy`/`https_proxy` pointed at an unroutable address (`http://127.0.0.1:1`, so any real network attempt fails fast rather than merely "not used") and `MMCLI_DATASETS` still exported:

```bash
mmcli init --dataset <name> -t <matching-task-type> -p <fresh-tmp-dir>
```
run once per dataset for all ten names/task types (`generic_timeseries_classification`, `generic_timeseries_regression`, `generic_timeseries_anomalydetection`, `generic_timeseries_forecasting`, `arc_fault_classification`→`arc_fault`, `ecg_classification`, `fan_blade_fault`→`motor_fault`, `pir_detection`, `mnist_image_classification`→`image_classification`, `generic_audio_classification`→`audio_classification`).

Result: **all ten exited 0** ("✓ Project created: ...") both in the initial run and in the final re-verification against the exact committed recipe text. `MMCLI_DATASETS`'s hard block on `fetch_dataset()` (unconditional refusal per REQ-DATA-03, confirmed by reading `mmcli/datasets.py`) means these results hold regardless of whether the unroutable-proxy network simulation is airtight — the code path cannot reach the network at all once the variable is set.

## Files Created/Modified

- `README.md` — corrected intro (dataset count/source, binary size), "How it works" size claim, env var table row for `MMCLI_DATASETS`; replaced `## Example Datasets` with a `## Datasets` section (resolution order, `mmcli datasets` CLI, D-5 policy, ten-dataset offline recipe, TI-name fallback mapping for 5 renamed datasets).
- `README_zh.md` — corrected the same intro/size facts in Chinese; replaced `## 示例数据集`'s stale claim with a corrected short summary plus an explicit pointer to `README.md#datasets` for the untranslated recipe/CLI detail; corrected the ten-row table (added the audio row, added a Source column); corrected the `MMCLI_DATASETS` env var table row.
- `.planning/phases/10-dataset-distribution-and-binary-size/deferred-items.md` — new; logs an out-of-scope discovery (see below).

## Decisions Made

See `key-decisions` in the frontmatter above. In short: the offline recipe's tenth-dataset step uses extract-then-re-zip rather than the plan's assumed path-then-copy, because the latter is provably broken against a PyInstaller `--onefile` build; and README_zh.md corrects short factual claims directly but points at the English README rather than translating the new, more complex recipe/CLI section.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - bug, found via Task 2's literal execution] Offline recipe's tenth-dataset step (`datasets path` + `cp`) does not work against the real binary**
- **Found during:** Task 2 (drafting/verifying Task 1's recipe text before committing)
- **Issue:** `cp "$(mmcli datasets path generic_audio_classification)" <dest>` fails with "No such file or directory" — PyInstaller onefile deletes its `_MEI<random>` extraction directory synchronously at process exit, before the calling shell can act on the printed path.
- **Fix:** Recipe now runs `mmcli init --dataset generic_audio_classification -t audio_classification -p <tmp>` (extraction happens while the process is alive) and re-zips the resulting `<tmp>/dataset/` directory under the expected filename. Verified this reconstructed zip resolves and extracts correctly (not digest-checked when found via `MMCLI_DATASETS`, per `mmcli/datasets.py`).
- **Files modified:** `README.md`
- **Verification:** Full ten-dataset offline recipe re-executed verbatim from the committed text, twice; both times all ten datasets report `bundled` in `datasets list --format json` and `init --dataset` exits 0 for all ten with the network unreachable.
- **Committed in:** `6d42add` (the bug was found and fixed before this commit was made, so there is no separate "before" state committed to git)

**Total deviations:** 1 auto-fixed (Rule 1), found via literal execution exactly as the plan intended Task 2 to surface.
**Impact on plan:** None beyond the intended one — this is precisely the class of gap Task 2 exists to catch ("a recipe nobody has run is a hypothesis"). No scope creep: the fix stays within `README.md`.

## Issues Encountered

- The plan's own suggested Task 2 verification script (using `cp "$(./dist/mmcli datasets path generic_audio_classification)"`) was re-run standalone (not as the README's recipe) purely to confirm the failure mode was real and not an artifact of my rewritten recipe: it failed identically (`No such file or directory`), confirming the bug is in the assumption, not in my test harness.
- Found (not fixed, out of scope): `mmcli/cli.py`'s `datasets pull` and `init --fetch` help text still say "from TI" — stale after 10-03's repoint, but outside this plan's `files_modified` (`README.md`, `README_zh.md` only). Logged in `deferred-items.md` for a future CLI-help-touching plan (10-07 is a candidate).

## User Setup Required

None. All verification ran against the pre-built `dist/mmcli` (31,839,872 bytes) already present in the repo from 10-03; no new build was performed.

## Next Phase Readiness

- **10-07** (docs/RELEASING.md, CLI help, Sphinx) can reuse this plan's accurate Datasets section as source material, and should also fix the stale "from TI" CLI help text logged in `deferred-items.md`.
- **10-08** (CI wiring) is unaffected — this plan touched no build scripts or CI config.
- No blockers for either.

## Self-Check: PASSED

- FOUND: `README.md` contains `datasets pull`, `cache/mmcli/datasets`, `generic_audio_classification`, `MMCLI_DATASETS`
- FOUND: `grep -c 'bundled `example_datasets/`' README.md` → 0 (stale row gone)
- FOUND: no remaining `software-dl.ti.com`/"downloaded from TI"/"9 bundled" false claims in `README.md` (the one `software-dl.ti.com` mention remaining is the accurate historical explanation of why the mirror exists)
- FOUND: no remaining `~10 MB`/`~14 MB`/`~15 MB`/`260 MB` size claims in `README.md`
- FOUND: `README_zh.md` contains `MMCLI_DATASETS` and a `README.md` pointer; no remaining "9 个示例数据集"/"从 TI 服务器下载的示例数据集"/"约 10 MB" claims
- FOUND: commit `6d42add` (`git log --oneline --all | grep 6d42add`)
- FOUND: commit `6280ad7`
- Ran the exact bash blocks extracted from the committed `README.md`'s offline recipe, fresh (`~/.cache/mmcli` and `~/mmcli-offline-datasets` removed first): 10/10 zips assembled, `datasets list --format json` reports all 10 as `bundled`, `init --dataset` exits 0 for all 10 with `http_proxy`/`https_proxy` pointed at an unroutable address and `MMCLI_DATASETS` set

## Threat Flags

None. This plan is documentation-only; no new network endpoints, auth paths, or trust boundaries were introduced. The one new pattern (re-zipping a locally-extracted dataset for offline use) does not change what `mmcli` trusts — `MMCLI_DATASETS`-provided files were already explicitly untrusted/unverified by design (REQ-DATA-03), and this recipe does not alter that.

---
*Phase: 10-dataset-distribution-and-binary-size*
*Completed: 2026-07-23*
