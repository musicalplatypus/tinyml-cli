---
phase: 10-dataset-distribution-and-binary-size
plan: 10
subsystem: packaging
tags: [packaging, setuptools, wheel, sdist, regression-guard]

requires:
  - phase: 10-03
    provides: "the BUNDLED_DATASETS allowlist pattern and the release mirror the nine datasets are fetched from"
provides:
  - "wheel and sdist that carry only generic_audio_classification.zip"
  - "include-package-data = false, without which package-data is additive and non-binding"
  - "MANIFEST.in, without which the sdist keeps its own full file list"
  - "TestPackageDataBundlesOnlyTheOneLocalDataset — 6 source-level guards, all mutation-tested"
affects: [10-07, 10-08]

requirements-completed: [REQ-SIZE-03]
---

# 10-10: Stop the wheel and sdist shipping the mirrored datasets

**Phase 10 unbundled the PyInstaller binary but never the pip channel. Closing that took three
independent levers, not the one the plan anticipated — verified by building real artifacts at
each step rather than reasoning about setuptools' behaviour.**

## What changed

| File | Change |
|---|---|
| `pyproject.toml` | `package-data` narrowed to the literal `example_datasets/generic_audio_classification.zip`; `data/*.yaml` untouched; **`include-package-data = false`** added |
| `MANIFEST.in` | new — `exclude` the dataset zips, then `include` the one packageable dataset |
| `tests/test_build_config.py` | new `TestPackageDataBundlesOnlyTheOneLocalDataset` (6 tests), `_package_data_patterns()` helper, `PYPROJECT_FILE` constant |

## Measured result

Built from a working tree holding all ten zips (macOS, setuptools 83.0.0, `python -m build`):

| Artifact | Before | After |
|---|---|---|
| wheel | 108.22 MB | **0.10 MB** |
| sdist | 108.26 MB | **0.17 MB** |
| installed package | — | **608 KB** |

## The plan was wrong about the mechanism, and the artifacts proved it

The plan's premise was that narrowing `package-data` covers both artifacts, on the reasoning
that "setuptools' sdist derives package data from package-data; it is not wheel-only." Building
the real thing contradicted that twice:

1. **After narrowing the glob, the wheel was still 108.22 MB.** Setuptools defaults
   `include-package-data` to **true** for pyproject.toml projects, and with it on `package-data`
   is *additive*, not restrictive — every file in the package directory is swept in regardless
   of the allowlist. The allowlist was inert until `include-package-data = false` was set.
2. **After the wheel was fixed, the sdist was still 108.26 MB.** The sdist builds its own file
   list and does not consult `package-data` at all. `MANIFEST.in` was required — the plan named
   this as a contingency, and it fired.

A third false lead was eliminated along the way: a stale `build/lib/` from an earlier build held
ten copies and setuptools reuses that tree without pruning files that no longer match. It was
not the cause here (removing it changed nothing), but it is a real way to ship a fat artifact
from a correct source tree.

The sdist mattering is not cosmetic: `python -m build` builds the wheel **from** the sdist, so a
fat sdist is a live route back to a fat wheel.

## Guards, and proof they fail

Six source-level assertions in the existing `tests/test_build_config.py`, beside 10-03's
build-script class so both distribution channels read as one decision and share
`BUNDLED_DATASET_FILENAME` / `MIRRORED_DATASET_FILENAMES`. One of them ties those constants back
to `DATASET_REGISTRY`'s `ti_name` split, so a dataset added later cannot end up unguarded in the
binary and the wheel simultaneously.

Every guard was mutation-tested — each of these fails exactly one assertion and the suite returns
to 36 green when reverted:

| Mutation | Caught |
|---|---|
| restore `example_datasets/*.zip` glob | 2 assertions |
| flip `include-package-data` to true | 1 |
| delete `MANIFEST.in` | 1 |
| re-include a mirrored dataset in `MANIFEST.in` | 1 |

Source-level rather than build-and-measure, and here that is the **stronger** choice, not the
cheaper one: `.gitignore:10` ignores `mmcli/example_datasets/*.zip` and only the audio zip is
tracked, so a CI checkout holds one dataset and a re-added wildcard would still build a small
artifact there. An artifact-size gate in CI would pass vacuously. The parser is regex-over-text
rather than `tomllib` because both workflows pin python 3.10, where `tomllib` is absent and
`tomli` is not installed — and it asserts it actually found the section and array, so a
restructured file fails loudly instead of returning `[]` and satisfying everything downstream.

## End-to-end verification

Wheel installed into a throwaway venv, run from outside the repo, `MMCLI_DATASETS` unset and a
scratch `XDG_CACHE_HOME`:

- `mmcli --version` → `mmcli 1.1.2`
- `datasets list --format json` → **9 `downloadable`, 1 `bundled`**
- `datasets path generic_audio_classification` → resolves inside site-packages, not the repo
- `datasets pull generic_timeseries_forecasting` → **really fetched from the live GitHub mirror**,
  sha256-verified, state flipped to `cached`
- `MMCLI_DATASETS` pointed at a directory holding all ten → **10 `bundled`** (REQ-DATA-04 intact)

One process note: the first introspection ran with the repo as cwd, so `import mmcli` resolved to
the source tree and reported 125 MB — measuring the wrong thing entirely. Re-run from `/private/tmp`
it reported 608 KB. Every figure above is from outside the repo.

## Scope correction worth carrying forward

`.gitignore` ignores the nine zips and only `generic_audio_classification.zip` is tracked, so a
build from a **clean clone** already produced a small wheel. The 108 MB figure is a property of a
maintainer working tree. This is therefore a release-safety fix — it stops whoever cuts a release
from publishing a fat artifact — not a defect every pip user hits today. Anyone re-checking this
from a fresh clone will find nothing wrong, which is why the numbers are recorded in
`pyproject.toml` next to the code they justify.

## Flagged, not done here

- **10-08 (CI):** the guards reach CI free via `test_build_config.py`, which it already wires in.
  It should *not* add a wheel-size gate believing it would catch this — CI's checkout lacks the
  nine zips, so such a gate passes vacuously unless it first seeds them from the mirror (~131 MB
  per run). That cost is a decision to take deliberately.
- **10-07 (RELEASING.md):** publishing a wheel for a new version obliges publishing the matching
  `datasets-<version>` mirror release **first**, or the nine go `downloadable` and then 404 for
  every pip user. Also worth documenting: build from a clean tree, or remove `build/` first.

## Self-Check: PASSED

- Both tasks executed and committed atomically (`b36faa8`, `15b3ee6`)
- Both artifacts measured, not inferred; every claim above is from a real build
- Ten zips still on disk and still ignored by git; REQ-DATA-04 re-verified after the change
- Throwaway venvs only — `.venv-ai` and `.venv-tinyml` were not written to; `$WORK` deleted
- Working tree clean; no commits outside tinyml-cli
