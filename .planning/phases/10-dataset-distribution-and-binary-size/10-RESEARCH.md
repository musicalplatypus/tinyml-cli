# Phase 10 Research: Dataset Distribution & Binary Size

All figures below were measured on this machine (macOS arm64, 2026-07-22), not estimated.

## Problem

`dist/mmcli` was 260.3 MB with ~6.2 s startup. `build_macos.sh:4` claims:

```
# The binary is lightweight (~10 MB) because tinyml_modelmaker is NOT bundled.
```

The claim was never enforced. Two independent causes.

## Cause 1 — unenforced PyInstaller exclusions (defect)

The generated `mmcli.spec` had `excludes=[]`. mmcli reaches the training engine through
`MMCLI_PYTHON` as a **subprocess** and never needs it in-process, but three real `import`
statements are visible to PyInstaller's static analysis:

| Site | Statement | Guard |
|------|-----------|-------|
| `mmcli/recommend.py:121` | `import tinyml_modelmaker` | `try/except ImportError: pass` |
| `mmcli/diagnose.py:131` | `import tinyml_modelmaker` | `try/except ImportError` |
| `mmcli/diagnose.py:208` | `import tvm` | `try/except ImportError` |

PyInstaller follows reachable imports regardless of the guard, pulling in torch + TVM + the
engine. Other occurrences (`info.py:40`, `compare.py:30`, `cli.py:218`) sit inside **string
literals** passed to subprocesses and were never a factor.

Because all three sites already handle `ImportError`, excluding them is behaviour-preserving.

**Measured effect of adding `--exclude-module`:**

| Build | Size | Startup (3-run steady) |
|-------|------|------------------------|
| Before | 260.3 MB | 8.43 / 6.21 / 6.13 s |
| After | 138.7 MB | 5.10 / 5.20 / 5.21 s |

First-run timings are noisy (a cold 9.33 s was observed); use the steady-state median.

Verified still working after exclusion: `--version`, `init --list` (10 datasets), `info`
(subprocess path), and the script's own "mmcli modules bundled: 17" check.

**`numpy` and `pandas` must NOT be excluded** — `analyze.py::_row_count` imports them.

## Cause 2 — bundled dataset payload (design choice)

`--add-data mmcli/example_datasets:mmcli/example_datasets` bakes in 125 MB:

| Dataset | Size |
|---------|------|
| fan_blade_fault.zip | 54 MB |
| mnist_image_classification.zip | 45 MB |
| arc_fault_classification.zip | 13 MB |
| ecg_classification.zip | 4.4 MB |
| generic_timeseries_anomalydetection.zip | 4.0 MB |
| generic_timeseries_classification.zip | 2.5 MB |
| pir_detection.zip | 1.5 MB |
| generic_timeseries_regression.zip | 888 KB |
| *(2 further small zips)* | — |

The top two are 99 MB — 71% of the payload.

Composition of the remaining 138.7 MB binary: datasets 125 MB, pandas 18 MB, cryptography
10 MB, PIL 8 MB, numpy 6.5 MB. **`cryptography` and `PIL` are unverified** — they look like
transitive pull-ins mmcli never uses, but that needs a dependency trace before excluding.

`--onefile` extracts the entire archive on **every launch**, so payload is paid per
invocation, not once at install. That is why startup tracks size so closely.

## Existing seam in the code

`mmcli/datasets.py` already has what an on-demand design needs:

```python
def _datasets_dir() -> str:
    env = os.environ.get("MMCLI_DATASETS")
    if env and os.path.isdir(env):
        return env
    return os.path.join(os.path.dirname(__file__), "example_datasets")
```

- `DATASET_REGISTRY` — one record per dataset: `filename`, `task_types`, `module`, `description`
- `extract_dataset()` — the single place a zip is opened
- No download path of any kind exists today

Adding `url`/`sha256` and a cache layer is **additive**: existing precedence is unchanged, so
current invocations behave identically.

## Options

| # | Option | Size | Trade-off |
|---|--------|------|-----------|
| 1 | Keep all bundled | 138.7 MB | Zero network; every dataset ships in every release |
| 2 | Unbundle the two giants | ~40 MB | 8 small datasets still offline; 2 need fetching |
| 3 | Bundle nothing | ~14 MB | Meets the original "~10 MB" claim; all datasets need network or `MMCLI_DATASETS` |

**Chosen: Option 3** (see D-2). Bundling nothing gives one mechanism rather than two; the
zero-network path is served by `MMCLI_DATASETS` instead.

## Layering decision

**mmcli owns the download; PlatypusStudio owns the prompting.** mmcli is used headless and in
CI so it must work without a GUI; if the fetch lived in the app, CLI users could not obtain
datasets at all. The app already shells out to mmcli for every other operation, so invoking
`mmcli datasets pull` keeps one mechanism instead of a second implementation that can drift.

## Constraints that are requirements, not preferences

- **sha256 mandatory** for any `url`. This adds a remote-fetch path to a tool that then runs
  training jobs.
- **Atomic download → verify → `os.replace()`.** A partial file at the final path becomes a
  poisoned cache hit on the next run.
- **`MMCLI_DATASETS` disables fetching entirely** — it signals a managed/air-gapped
  environment and must never be silently overridden by a cached copy.
- **Published zips are immutable.** Clients key on the digest; regenerating a zip breaks
  every cached client. Publish under a new name instead.
- Use stdlib `urllib.request`. `requests` is not a declared dependency, and adding one to
  shrink a binary is self-defeating.

## Provenance of the local zips (measured 2026-07-22)

An earlier draft of this document claimed these files existed only on one developer machine
and were unbacked. **That was wrong**, inferred from `.gitignore` without checking. Verified
by downloading TI's copies and comparing digests:

| Local file | TI source (`.../mcu_ai/datasets/`) | Match |
|------------|-----------------------------------|-------|
| generic_timeseries_classification.zip | same name | byte-identical |
| generic_timeseries_regression.zip | same name | byte-identical |
| generic_timeseries_anomalydetection.zip | same name | byte-identical |
| generic_timeseries_forecasting.zip | same name | byte-identical |
| fan_blade_fault.zip | `fan_blade_fault_dsi.zip` | byte-identical, renamed |
| mnist_image_classification.zip | `mnist_classes.zip` | byte-identical, renamed |
| pir_detection.zip | `pir_detection_classification_dsk.zip` | byte-identical, renamed |
| arc_fault_classification.zip | `arc_fault_classification_dsi.zip` | byte-identical, renamed |
| ecg_classification.zip | `ecg_classification_2class.zip` | byte-identical, renamed |
| generic_audio_classification.zip | *(none — local origin)* | 18 KB, added by `be06559`, tracked in git |

Nine of ten are verbatim TI files under friendlier names; nothing was repackaged. The tenth
is a synthetic yes/no WAV set authored here.

Recorded sha256 for all ten (use directly in the registry):

```
bcee7b54fb42079bfac1f4a39266fb836c2ef73c3f8fffd8fa04c41671f7656e  arc_fault_classification.zip
881ac26e95378eca9c1979cf1c70a8d1b8f2cb73da65e264a03bf1849c6addc6  ecg_classification.zip
5194925e0f97387a54be989923ec34bef8e65e03fe21652552d7bbcdc21a959e  fan_blade_fault.zip
dfc463e6a0aac80b2db36770e9fc56090f319d400d416b391d160d70382dbc5d  generic_audio_classification.zip
7cb2f9fd183fa5c6730abdd0a144e1ce57f7ece9ed93d8663b19a983cde6d6b5  generic_timeseries_anomalydetection.zip
7b2c0980bb30c3bc661004d66373d7ea35ea13ab5b6f8b74f5182c3bc6bc09c1  generic_timeseries_classification.zip
4ae6e7e436817a8ee5f3e528e70741b9c6fabfeb6c19a9fdb321dabad0a804ce  generic_timeseries_forecasting.zip
078d212b00112bcaca4b1bb68b871e8c24eb3ed809b610d64642a74a7854cc23  generic_timeseries_regression.zip
7fa4be9944a364074dc796d5d802dad8f1636f2f4daa6fd735d15f5fe05f3db8  mnist_image_classification.zip
d75470c9ba7f56fd4e8801c9f10424262e9935513b9011f55f5f5ed406ae0b0e  pir_detection.zip
```

### TI URL forms

Both work and both serve `application/zip` with range support (`HTTP 206`):

- flat: `https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/<ti_name>.zip`
- version-pathed: `https://software-dl.ti.com/C2000/esd/mcu_ai/<VER>/datasets/<ti_name>.zip`
  (`01_02_00`, `01_03_00`, `01_04_00` observed)

Version-pathed URLs resolve even for datasets not referenced at that version in engine
source, so the versioned form covers all nine.

## Decisions (REVISED 2026-07-22 after the provenance check)

**D-1 — Fetch from TI, do not mirror.** The datasets are TI's, publicly hosted, and already
version-pathed. Mirroring 125 MB into this repo's releases would duplicate a working CDN,
add a redistribution question for third-party data, and create a mirror that can silently
drift from upstream. The registry points at TI URLs instead.

Supersedes the earlier decision to publish them as GitHub Release assets, which rested on the
false premise that the files were unbacked.

**D-2 — Bundle only what is ours.** `generic_audio_classification.zip` is 18 KB, locally
authored and already tracked in git, so it stays bundled at negligible cost. The nine TI
datasets are fetched on demand. Binary target ~14 MB is unaffected.

**D-3 — Version axis is TI's engine version, not an mmcli tag.** A global
`DATASETS_DEFAULT_VERSION` (e.g. `01_03_00`) with a per-entry `ti_version` override gives the
"release specific" property, sourced from the authority that actually versions these files.

Consequence: **the cache key includes the version** —
`~/.cache/mmcli/datasets/<version>/<name>.zip` — so changing the pinned version cannot
silently reuse an older dataset.

**D-4 — Filename mapping is required.** Five local names differ from TI's, so each entry
needs both its local `filename` and its `ti_name`.

## Risk introduced by D-1

Availability now depends on `software-dl.ti.com`. Two URL shapes already coexist (flat and
version-pathed), which suggests TI has reorganised paths at least once, so a future
reorganisation is plausible. Mitigations: pin the version-pathed form, keep sha256
verification, and document `MMCLI_DATASETS` as the offline escape hatch. If TI ever breaks
the paths, mirroring to release assets remains available as a fallback — the registry change
would be a URL swap, not a redesign.

## Status

The Cause-1 fix is **implemented but uncommitted** in `build_macos.sh` (exclude list +
`--exclude-module` expansion). It needs a regression guard so it cannot silently revert.
