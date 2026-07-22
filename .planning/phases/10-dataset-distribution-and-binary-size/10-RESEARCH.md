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

Recommended: **Option 2**. The synthetic timeseries sets are what most first runs use and
cost ~8 MB, preserving a zero-network happy path.

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

## Open decisions (need a human)

1. **Hosting.** GitHub Releases on the fork is free and versioned but couples availability to
   GitHub reachability — a real constraint for TI/corporate/air-gapped users. Alternative: a
   TI-internal location, or publish both and let `MMCLI_DATASETS` cover restricted networks.
2. **How many datasets stay bundled** — Option 2 (8 bundled, ~40 MB) vs Option 3 (none, ~14 MB).

## Status

The Cause-1 fix is **implemented but uncommitted** in `build_macos.sh` (exclude list +
`--exclude-module` expansion). It needs a regression guard so it cannot silently revert.
