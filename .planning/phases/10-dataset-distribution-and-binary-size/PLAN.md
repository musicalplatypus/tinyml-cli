# Phase 10: Dataset Distribution and Binary Size

**Milestone:** v1.2
**Depends on:** Phase 9

**Requirements:** REQ-SIZE-01, REQ-SIZE-02, REQ-DATA-01, REQ-DATA-02, REQ-DATA-03,
REQ-DATA-04, REQ-UX-01 (defined in ROADMAP.md)

**Research:** `10-RESEARCH.md` — all figures measured on macOS arm64, 2026-07-22.

## Goal

Cut the distributed `mmcli` binary from 260 MB to roughly 14 MB, and publish the example
datasets as versioned GitHub Release assets fetched on demand, so a dataset can be
release-specific and updated without rebuilding the binary.

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

## Options considered

| # | Option | Size | Trade-off |
|---|--------|------|-----------|
| 1 | Keep everything bundled | 138.7 MB | Zero network, but every dataset ships in every release |
| 2 | Unbundle the two giants | ~40 MB | 8 small datasets still work offline; 2 need fetching |
| 3 | **Bundle nothing — CHOSEN (D-2)** | **~14 MB** | Meets the original "~10 MB" claim; every `init --dataset` needs network or `MMCLI_DATASETS` |

**Chosen: Option 3** (D-2). Bundling nothing keeps one mechanism instead of two, and the
zero-network path is served by `MMCLI_DATASETS` rather than by a partial bundle. The
trade-off accepted: a first `init --dataset` needs network, where today it does not.

## Design

`datasets.py` already has the seam this needs: `_datasets_dir()` resolves `MMCLI_DATASETS`
first and falls back to the bundled directory, `DATASET_REGISTRY` holds one record per
dataset (`filename`, `task_types`, `module`, `description`), and `extract_dataset()` is the
single choke point. The change is additive — existing resolution order is preserved, so
nothing regresses for users who bundle or set the env var.

Resolution order becomes:

```
1. MMCLI_DATASETS env var          (existing — offline / air-gap escape hatch)
2. bundled example_datasets/        (existing — now only the small datasets)
3. ~/.cache/mmcli/datasets/         (new — previously downloaded)
4. download from registry `url`     (new — on demand, sha256-verified)
```

**Layering.** mmcli owns the download; PlatypusStudio owns the prompting. mmcli is used
headless and in CI, so it must work without a GUI; if the download lived only in the app,
CLI users could not obtain the datasets at all. The app already shells out to mmcli for
everything else, so surfacing a "Download (54 MB)" affordance that invokes
`mmcli datasets pull` fits the existing pattern rather than introducing a second mechanism.

**Integrity is mandatory, not optional.** Every registry entry that carries a `url` must
carry a `sha256`, verified before extraction. Without it this introduces a remote-fetch
surface into a tool that then runs training jobs. A checksum mismatch must fail loudly and
leave no partial file in the cache.

## Plans

| Plan | Type | Status |
|------|------|--------|
| 10-01-PLAN.md — Enforce PyInstaller exclusions + size regression guard | fix | PENDING |
| 10-02-PLAN.md — Registry url/sha256, cache layer, `mmcli datasets pull` | feat | PENDING |
| 10-03-PLAN.md — Publish assets, unbundle entirely + docs | chore | PENDING |
| 10-04-PLAN.md — PlatypusStudio download affordance (separate repo) | feat | PENDING |

10-01 is already implemented but uncommitted in `build_macos.sh`; the plan covers
committing it plus the regression guard that keeps it from silently reverting.

## Success Criteria

- `dist/mmcli` is ≤ 45 MB and starts in under 2.5 s (steady state, 3-run median)
- `mmcli --version`, `mmcli init --list`, `mmcli info`, `mmcli analyze` and `mmcli diagnose`
  all behave identically to the 260 MB build
- `mmcli datasets pull fan_blade_fault` fetches, verifies sha256, caches, and a subsequent
  `mmcli init --dataset fan_blade_fault` uses the cache without network
- A corrupted or truncated download fails with a clear error and leaves no cache entry
- With `MMCLI_DATASETS` set to a directory holding all 10 zips, no network access occurs
- A build that loses the exclusions fails CI rather than shipping a 260 MB binary

## Decisions (RESOLVED 2026-07-22)

- **D-1 Hosting:** GitHub Release assets on `musicalplatypus/tinyml-cli`, reusing the
  existing `PlatypusCLI_X.Y.Z_Release` tags that already carry the per-platform binaries.
  Not committed to git — `.gitignore` already excludes the zips, and 125 MB in history would
  be permanent and paid by every clone.
- **D-2 Bundling:** Option 3 — bundle nothing (~14 MB). One mechanism rather than two.
- **D-3 Versioning:** per-dataset `tag` override with a global `DATASETS_DEFAULT_TAG`, so a
  dataset republishes only when it actually changes. The cache is keyed by tag so an mmcli
  upgrade cannot silently reuse an older dataset.

**Risk surfaced by D-1:** nine of the ten zips are in no repository or release — they exist
only on one developer machine. 10-03 Task 1 is their first backup, which is why it runs
before anything that touches the build.

## Notes

- `cryptography` (10 MB) and `PIL` (8 MB) appear to be transitive pull-ins mmcli never uses.
  Worth confirming with a dependency trace before adding them to the exclude list — they are
  small enough that guessing is not worth a broken build.
- `mmcli/example_datasets/` stays in the repo regardless; unbundling changes only what the
  *binary* carries, not what the source tree contains.
