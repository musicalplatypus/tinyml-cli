# Phase 10: Dataset Distribution and Binary Size

**Milestone:** v1.2
**Depends on:** Phase 9

**Requirements:** REQ-SIZE-01/02, REQ-DATA-01/02/03/04/05, REQ-UX-01, REQ-DOC-01
(defined in ROADMAP.md)

**Research:** `10-RESEARCH.md` — all figures measured on macOS arm64, 2026-07-22.

## Goal

Cut the distributed `mmcli` binary from 260 MB to roughly 14 MB by fetching the TI example
datasets from their upstream versioned URLs on demand, so a dataset can be release-specific
and updated without rebuilding the binary.

Two independent causes, discovered by measuring the binary rather than reading the build:

1. **Unenforced exclusions (defect).** `build_macos.sh` states the binary is "lightweight
   (~10 MB) because tinyml_modelmaker is NOT bundled", but the generated spec had
   `excludes=[]`. mmcli drives the engine through `MMCLI_PYTHON` as a subprocess and never
   needs it in-process, yet three guarded probes — `import tinyml_modelmaker` in
   `recommend.py` and `diagnose.py`, `import tvm` in `diagnose.py` — are visible to
   PyInstaller's static analysis, which follows them and bundles torch, TVM and the whole
   engine. The comment asserted an invariant nothing enforced.

2. **Bundled dataset payload (design choice).** `--add-data` bakes
   `mmcli/example_datasets/` into the binary: 125 MB of zips, of which
   `fan_blade_fault.zip` (54 MB) and `mnist_image_classification.zip` (45 MB) are 71%.

### Measurements

| Build | Size | Startup (steady, 3 runs) |
|-------|------|--------------------------|
| Before | 260.3 MB | ~6.2 s |
| After `--exclude-module` | 138.7 MB | ~5.2 s |
| Projected, nothing bundled (chosen) | ~14 MB | ~1–2 s |

Startup matters disproportionately because `--onefile` extracts the entire archive on
*every* launch, so payload size is paid per invocation, not once at install.

Composition of the current 138.7 MB: example datasets 125 MB, pandas 18 MB, cryptography
10 MB, PIL 8 MB, numpy 6.5 MB. `numpy` and `pandas` are load-bearing —
`analyze.py::_row_count` uses them — so they stay. `cryptography` and `PIL` are unverified
transitive pull-ins and are worth investigating, not assuming.

### Three artifacts, not one

`.github/workflows/release.yml` builds and publishes three binaries per release:
`build_linux.sh`, `build_windows.ps1`, `build_macos.sh`. Every measurement above was taken on
macOS, and only `build_macos.sh` had the exclusions. Anything this phase asserts about "the
binary" has to hold for all three, or REQ-SIZE-02 fails for a third of the download page. The
exclude list therefore lives in one shared file (`scripts/pyinstaller_excludes.txt`) read by
all three scripts, and the size ceiling is enforced per-artifact in the release job rather
than on one developer's laptop.

## Options considered

| # | Option | Size | Trade-off |
|---|--------|------|-----------|
| 1 | Keep everything bundled | 138.7 MB | Zero network, but every dataset ships in every release |
| 2 | Unbundle the two giants | ~40 MB | 8 small datasets still work offline; 2 need fetching |
| 3 | **Fetch all TI sets — CHOSEN (D-1/D-2)** | **~14 MB** | Only the 18 KB local audio set stays bundled; TI sets need network or `MMCLI_DATASETS` |

**Chosen: Option 3**, refined by D-1. The nine TI datasets are fetched from TI rather than
mirrored; only the 18 KB locally authored audio set stays bundled, since bundling it costs
nothing and it has no upstream. Trade-off accepted: a first `init --dataset` on a TI set
needs network, where today it does not.

## Design

`datasets.py` already has the seam this needs: `_datasets_dir()` resolves `MMCLI_DATASETS`
first and falls back to the bundled directory, `DATASET_REGISTRY` holds one record per
dataset (`filename`, `task_types`, `module`, `description`), and `extract_dataset()` is the
single choke point. The change is additive — existing resolution order is preserved, so
nothing regresses for users who bundle or set the env var.

Resolution order becomes:

```
1. MMCLI_DATASETS env var          (existing — offline / air-gap escape hatch)
2. bundled example_datasets/        (existing — only the 18 KB audio set after 10-03)
3. ~/.cache/mmcli/datasets/<ver>/   (new — previously downloaded)
4. download from TI (new — version-pinned, sha256-verified)
```

**Layering.** mmcli owns the download; PlatypusStudio owns the prompting. mmcli is used
headless and in CI, so it must work without a GUI; if the download lived only in the app,
CLI users could not obtain the datasets at all. The app already shells out to mmcli for
everything else, so surfacing a "Download (54 MB)" affordance that invokes
`mmcli datasets pull` fits the existing pattern rather than introducing a second mechanism.

**Integrity is mandatory, not optional.** Every registry entry that carries a `ti_name` must
carry a `sha256`, verified before extraction. Without it this introduces a remote-fetch
surface into a tool that then runs training jobs. A checksum mismatch must fail loudly and
leave no partial file in the cache.

**A guard CI does not run is not a guard.** Both workflows invoke pytest with an explicit file
list, so a new test file is invisible to CI until it is named there. Every regression guard
this phase adds is wired into both workflows by 10-08, the only plan that touches
`.github/workflows/`.

## Plans

Eight plans in five waves. Same-wave plans touch disjoint files and can run in parallel.

| Wave | Plan | Type | Requirements | Status |
|------|------|------|--------------|--------|
| 1 | 10-01 — PyInstaller exclusions across all three builds + single-source size ceiling | fix | REQ-SIZE-01/02 | PENDING |
| 2 | 10-02 — Registry digests/versioning, version-scoped cache, verified `fetch_dataset` | feat | REQ-DATA-01/02/03/05 | PENDING |
| 3 | 10-03 — GET-and-hash gate over all nine TI URLs, then unbundle | chore | REQ-SIZE-01, REQ-DATA-04/05 | PENDING |
| 3 | 10-06 — `mmcli datasets list/pull/path` + D-5 auto-fetch policy | feat | REQ-DATA-01/03, REQ-UX-01 | PENDING |
| 4 | 10-04 — PlatypusStudio download affordance (separate repo) | feat | REQ-UX-01 | PENDING |
| 4 | 10-05 — README, ten-dataset offline recipe, executed as written | doc | REQ-DOC-01, REQ-DATA-03/04 | PENDING |
| 4 | 10-08 — Wire the new guards into CI + per-artifact size and bundle gates | chore | REQ-SIZE-01/02 | PENDING |
| 5 | 10-07 — docs/RELEASING.md, CLI help, Sphinx | doc | REQ-DOC-01, REQ-DATA-05 | PENDING |

10-06 and 10-07 were split out of the original 10-02 and 10-05, which had four tasks each and
would have run past their context budget. 10-08 was added in the second review revision to
close the CI-wiring gap. Plan numbers were appended rather than renumbered so existing
references stay valid; the wave column, not the number, gives execution order.

10-01's macOS build change is already committed (143dd7e); that plan now also covers the Linux
and Windows scripts, which never had it, plus the regression guard that keeps it from silently
reverting.

### Wave placement

Waves follow `wave = max(depends_on) + 1`, with no plan held back beyond that unless the
dependency is stated:

- **10-05 → wave 4.** Depends only on 10-03 and 10-06 (both wave 3). An earlier revision put
  it at wave 5 behind 10-04 for no reason; 10-04 is cross-repo, `autonomous: false`, and this
  plan reads nothing it produces.
- **10-07 → wave 5, behind 10-05.** This one *is* a dependency: 10-07 Task 2 must copy the
  corrected README env-var sentence into `cli.py` verbatim, which is the whole of mitigation
  T-10-07-02. Running them in parallel would copy the stale sentence and still pass its grep.
- **10-08 → wave 4.** Needs every test file it wires (10-01, 10-02, 10-06) and the lowered
  ceiling (10-03) to exist first.

### File ownership (no same-wave overlap)

| Wave | Plan | Owns |
|------|------|------|
| 1 | 10-01 | `scripts/pyinstaller_excludes.txt`, `build_macos.sh`, `build_linux.sh`, `build_windows.ps1`, `scripts/binary_size_ceiling.txt`, `tests/test_build_config.py` |
| 2 | 10-02 | `mmcli/datasets.py`, `tests/test_datasets_download.py` |
| 3 | 10-03 | `scripts/verify_dataset_digests.py`, `build_macos.sh`, `build_linux.sh`, `build_windows.ps1`, `scripts/binary_size_ceiling.txt`, `tests/test_build_config.py` |
| 3 | 10-06 | `mmcli/cli.py`, `tests/test_datasets_cli.py` |
| 4 | 10-04 | `../PlatypusStudio/...` only |
| 4 | 10-05 | `README.md`, `README_zh.md` |
| 4 | 10-08 | `.github/workflows/test-cli.yml`, `.github/workflows/release.yml`, `tests/test_ci_workflows.py` |
| 5 | 10-07 | `docs/RELEASING.md`, `docs/mmcli.rst`, `mmcli/cli.py` |

Checked pairwise within each wave: wave 3 (10-03 ∩ 10-06 = ∅), wave 4 (10-04 ∩ 10-05 ∩ 10-08
= ∅). The repeated files — the three build scripts, `binary_size_ceiling.txt` and
`tests/test_build_config.py` between 10-01 (wave 1) and 10-03 (wave 3), `mmcli/cli.py` between
10-06 and 10-07 — are all cross-wave, so they serialise. 10-01 creates
`tests/test_build_config.py`; 10-03 extends it two waves later.

## Success Criteria

- `dist/mmcli` is ≤ 15 MB and starts in under 2.5 s (steady state, 3-run median)
- `mmcli --version`, `mmcli init --list`, `mmcli info`, `mmcli analyze` and `mmcli diagnose`
  all behave identically to the 260 MB build
- `mmcli datasets pull fan_blade_fault` fetches, verifies sha256, caches, and a subsequent
  `mmcli init --dataset fan_blade_fault` uses the cache without network
- A corrupted or truncated download fails with a clear error and leaves no cache entry
- With `MMCLI_DATASETS` set to a directory holding all 10 zips, no network access occurs
- All three published artifacts — Linux, Windows, macOS — exclude the same module set and
  bundle exactly `generic_audio_classification.zip`
- A build that loses the exclusions fails CI rather than shipping a 260 MB binary — meaning
  the guard runs inside `.github/workflows/`, not only in a local pytest invocation
- A release build over the ceiling, or with an empty dataset bundle, fails before the artifact
  is uploaded, on every platform

## Decisions (RESOLVED 2026-07-22)

- **D-1 Source:** fetch from TI (`software-dl.ti.com`), do not mirror. Nine of the ten local
  zips were verified **byte-identical** to files TI already publishes (five under different
  names) — see the provenance table in `10-RESEARCH.md`. Mirroring would duplicate a working
  CDN, raise a redistribution question for third-party data, and create a mirror that can
  drift.
- **D-2 Bundling:** bundle only `generic_audio_classification.zip` (18 KB, locally authored,
  already tracked in git). The nine TI datasets are fetched. Binary target ~14 MB unaffected.
- **D-3 Versioning:** version axis is TI's engine version path (`/01_03_00/datasets/…`), with
  a per-entry `ti_version` override. Cache is keyed by version so a bump cannot silently
  reuse an older dataset.
- **D-4 Naming:** five local names differ from TI's, so entries carry both `filename` and
  `ti_name`.
- **D-5 Who may start a download (added 2026-07-22, resolves review finding F-5):** mmcli
  never starts a multi-megabyte transfer it cannot narrate. `init --dataset` auto-fetches
  only when `stderr` is a TTY; piped, scripted and GUI-driven invocations fail with the exact
  `mmcli datasets pull <name>` command and the size. Precedence: `MMCLI_DATASETS` set → never
  fetch; `--no-fetch` / `MMCLI_AUTO_FETCH=0` → never; `--fetch` / `MMCLI_AUTO_FETCH=1` →
  always; otherwise TTY decides. The TTY predicate is the same one that gates the `tqdm`
  progress bar, so progress and permission are one question rather than two heuristics.
  Rationale and the options considered are recorded in `10-06-PLAN.md`.

**Risk from D-1:** availability now depends on `software-dl.ti.com`. Two URL shapes already
coexist, so TI has reorganised at least once. Mitigated by pinning the versioned form, sha256
verification, and `MMCLI_DATASETS` as the offline escape hatch; mirroring stays available as
a fallback if TI ever breaks the paths.

**Correction:** an earlier draft claimed these files were unbacked and existed only on one
machine. That was wrong — inferred from `.gitignore` without checking — and it is what made
mirroring look necessary.

## Notes

- `cryptography` (10 MB) and `PIL` (8 MB) appear to be transitive pull-ins mmcli never uses.
  Worth confirming with a dependency trace before adding them to the exclude list — they are
  small enough that guessing is not worth a broken build.
- `mmcli/example_datasets/` stays in the repo regardless; unbundling changes only what the
  *binary* carries, not what the source tree contains.
- `pytest.ini` declares `[tool:pytest]`, which is the `setup.cfg` section name — in a
  `pytest.ini` the section must be `[pytest]`. So `testpaths` and the `--cov` `addopts` are
  inert today (verified: bare `pytest` collects `test_sigsegv.py` at the repo root). Recorded
  in 10-08 and deliberately not fixed here: correcting the header would enable `--cov` in
  workflows that do not install pytest-cov, and would broaden collection to thirty-odd test
  files CI has never run.

## Review findings resolved (REVIEWS.md, 2026-07-22)

| Finding | Resolution | Where |
|---------|------------|-------|
| F-1 — CI ceiling said 45 MB, a leftover from an unchosen option | Ceiling moved into a single file `scripts/binary_size_ceiling.txt`, created at the interim 152,043,520 and lowered by 10-03 to 15,728,640 (REQ-SIZE-01). `tests/test_build_config.py` asserts it is one of those two values, so a typo cannot loosen it. 10-08's CI gate reads the same file rather than restating the number. The three correct `145 MB` figures are unchanged. | 10-01 Task 3, 10-03 Task 2, 10-08 Task 2 |
| F-2 — REQ-DATA-02 keyed off a `url` field that no longer exists | Requirement restated against `ti_name` and made import-enforced. The two remaining `url` phrasings in `10-RESEARCH.md` were corrected in revision 2 so the finding cannot be re-derived from the research doc. | ROADMAP.md, 10-02 Task 1, 10-RESEARCH.md |
| F-3 — digest gate issued HEAD requests only | Replaced with a committed GET-and-hash script driving the real `fetch_dataset` for all nine, blocking on failure and re-runnable from `docs/RELEASING.md`. | 10-03 Task 1, 10-07 Task 1 |
| F-4 — offline recipe said "download the ten assets"; only nine exist upstream | Recipe now pulls nine and takes the tenth from `mmcli datasets path generic_audio_classification`, and a task executes the recipe with the network disabled. The five-file TI rename is documented as a manual-download trap. | 10-05 Tasks 1-2 |
| F-5 — 10-02 and 10-04 disagreed about who fetches | Resolved as D-5 above: TTY-gated with explicit overrides. 10-04 verifies the guarantee (checkpoint step 7) instead of assuming it. | 10-06 Task 2, 10-04 Task 3 |

## Checker findings resolved (gsd-plan-checker, revision 2)

| Finding | Resolution | Where |
|---------|------------|-------|
| B-1 — `build_windows.ps1` entirely unaddressed; REQ-SIZE-02 failed for 1 of 3 shipped artifacts | Exclude list extracted to `scripts/pyinstaller_excludes.txt` and read by all three build scripts (PowerShell via `Get-Content` + array splat — the bash idiom does not transfer, and interpolating an array into a backtick-continued line stringifies it into one malformed flag). `tests/test_build_config.py` parametrises over all three scripts. The stale `~10 MB` comment, present verbatim in all three, is corrected in all three. **Also found while fixing it:** `build_linux.sh` and `build_windows.ps1` have no `--add-data` at all, so both already ship zero datasets and `datasets path generic_audio_classification` cannot work there — 10-03 Task 2 now *adds* the staged allowlist to those two, with the Windows `;` separator called out (T-10-03-05). Kept in 10-01/10-03 rather than a new plan: the shared exclude list and `tests/test_build_config.py` are single-owner artifacts, and splitting Windows off would give two plans write access to them, reintroducing the drift failure mode 10-01 exists to prevent. | 10-01 Tasks 2-3, 10-03 Tasks 2-3 |
| B-2 — new regression tests never wired into CI | New plan **10-08** (wave 4) is the single owner of `.github/workflows/`. Task 1 appends the four new test files to both invocations, preserving `-k "not TestInitDatasetExtractReal"`, and adds `tests/test_ci_workflows.py` asserting the two workflows name the same set and that each new file actually contributes selected tests. Task 2 adds a per-artifact size gate reading `scripts/binary_size_ceiling.txt` and a `datasets path generic_audio_classification` empty-bundle probe to the release build job, both before upload. The explicit-file-list pattern is kept deliberately — see the `pytest.ini` note above. Given its own plan because three plans in three different waves produce these test files; wiring them separately means three edits to the same two lines and three chances to diverge. | 10-08 |
| W-1 — 10-05 and 10-07 wave placement unjustified | 10-05 moved to wave 4 (`max(deps)+1`); it was serialised behind 10-04 for no reason. 10-07 stays at wave 5 with `depends_on` now naming 10-05 explicitly, because its Task 2 copies the corrected README wording verbatim and that wording does not exist until 10-05 Task 1 lands. | 10-05, 10-07 frontmatter + Wave placement sections |
| W-3 — 10-03 added `--add-data` staging with no source-level test, so a Windows/Linux bundling regression would merge silently and surface only at release | `tests/test_build_config.py` added to 10-03's `files_modified`; Task 2 now extends it with `--add-data` assertions over all three scripts — one flag, naming `generic_audio_classification`, destination `mmcli/example_datasets`, and the platform-correct separator (`:` on the shell scripts, `;` on PowerShell) — plus an assertion that no TI zip is named. Reason it matters: `test-cli.yml` builds no binary, so 10-08 Task 2 Step B's runtime probe only runs at tag time; the exclusions are source-tested on every push and the bundling now is too. New threat T-10-03-06. | 10-03 Task 2 |
| W-2 — non-portable `stat -f%z` | Both occurrences (10-01 Task 1, 10-03 Task 2) now use `stat -f%z 2>/dev/null \|\| stat -c%s`. 10-08's CI gate uses `wc -c` instead, which behaves identically on all three runners including Windows Git Bash. | 10-01, 10-03, 10-08 |

## Source coverage audit

| Source | Item | Covered by |
|--------|------|-----------|
| GOAL | Binary 260 MB → ~14 MB via on-demand TI fetch, release-specific datasets | 10-01, 10-02, 10-03, 10-06 |
| REQ | REQ-SIZE-01 (≤15 MB, <2.5 s) | 10-03 Task 2 (ceiling + startup, macOS), 10-08 Task 2 (per-artifact gate for Linux/Windows, which no local host can measure) |
| REQ | REQ-SIZE-02 (lost exclusions fail CI) | 10-01 Tasks 1-3 (all three build scripts + guard), 10-08 Task 1 (the guard actually runs in CI) |
| REQ | REQ-DATA-01 (env → bundled → cache → download) | 10-02 Task 2, 10-06 Task 1 |
| REQ | REQ-DATA-02 (ti_name ⇒ sha256, verified) | 10-02 Tasks 1 and 3 |
| REQ | REQ-DATA-03 (MMCLI_DATASETS disables fetching) | 10-02 Task 3, 10-06 Task 2, 10-05 Task 2 |
| REQ | REQ-DATA-04 (all 10 obtainable offline) | 10-03 Task 2 (incl. the Linux/Windows `--add-data` gap), 10-05 Task 2 (executed recipe), 10-08 Task 2 (per-platform bundle probe) |
| REQ | REQ-DATA-05 (pinned version path, per-entry override, version-keyed cache) | 10-02 Tasks 1-2, 10-03 Task 1, 10-07 Task 1 |
| REQ | REQ-UX-01 (visible size + explicit download, never a silent stall) | 10-06 Task 2 (D-5), 10-04 Tasks 2-3 |
| REQ | REQ-DOC-01 (no false README claim; offline recipe; RELEASING.md) | 10-05 Tasks 1-3, 10-07 Tasks 1-2 |
| RESEARCH | Cause 1 — unenforced exclusions, numpy/pandas retained | 10-01 Tasks 1-3 |
| RESEARCH | Cause 2 — 125 MB bundled payload | 10-03 Task 2 |
| RESEARCH | Provenance table, ten digests, local↔TI name mapping | 10-02 Task 1, 10-05 Task 1 |
| RESEARCH | Constraints: atomic replace, HTTPS+stdlib urllib, immutable published zips | 10-02 Task 3, 10-07 Task 1 |
| RESEARCH | D-1 risk: TI path reorganisation, mirroring as fallback | 10-03 Task 1 gate, 10-07 Task 1 point 7 |
| CONTEXT | D-1 fetch from TI, do not mirror | 10-02 Task 1, 10-03 |
| CONTEXT | D-2 bundle only `generic_audio_classification.zip` | 10-03 Task 2 (all three build scripts) |
| CONTEXT | D-3 TI engine version axis, version-keyed cache | 10-02 Tasks 1-2 |
| CONTEXT | D-4 filename mapping | 10-02 Task 1, 10-05 Task 1 |
| CONTEXT | D-5 who may start a download | 10-06 Task 2, 10-04 |

No source item is unplanned. `cryptography` (10 MB) and `PIL` (8 MB) remain deliberately
out of scope — `10-RESEARCH.md` marks them unverified, and excluding a module on a guess is
what a dependency trace is for. They do not block REQ-SIZE-01. The `pytest.ini` section-header
defect and the thirty-odd never-collected test files are recorded in 10-08's "Out of scope":
both are real, neither is required by any Phase 10 requirement, and each would change what CI
enforces repository-wide rather than for this phase.
</content>
