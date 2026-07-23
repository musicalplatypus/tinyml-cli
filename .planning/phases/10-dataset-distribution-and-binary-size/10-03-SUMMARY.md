---
phase: 10-dataset-distribution-and-binary-size
plan: 03
subsystem: dataset-distribution
tags: [pyinstaller, build-scripts, binary-size, dataset-verification, blocked]

requires:
  - phase: 10-02
    provides: "fetch_dataset(), dataset_url(), DATASET_REGISTRY digests — the fetch path this plan's gate exercises verbatim"
provides:
  - "scripts/verify_dataset_digests.py — re-runnable GET-and-hash gate over every fetchable registry entry, driven through fetch_dataset(force=True)"
affects: [10-04, 10-05, 10-07, 10-08]

tech-stack:
  added: []
  patterns:
    - "Content-verification gate driven through the real fetch_dataset() code path (not a parallel downloader) so the gate proves what users actually run, per review finding F-3"

key-files:
  created:
    - scripts/verify_dataset_digests.py
  modified: []

key-decisions:
  - "STOPPED before Task 2/3 per the plan's own explicit blocking instruction: the gate must pass before any unbundling, and it did not pass. No --add-data line was touched, no ceiling was lowered, no build-script comments were rewritten."
  - "Independently verified (via curl -L + sha256, bypassing the code's host-lock) that all nine TI datasets' actual bytes match the registry digests exactly — the failure is not a tampering event or a stale digest, it is TI's CDN issuing a 302 that the code's cross-host redirect refusal (correctly, by design) does not follow."
  - "Did not modify mmcli/datasets.py to work around the redirect — that file is 10-02's deliverable and is not in this plan's files_modified list, and loosening a Tampering-mitigation (T-10-02-01/05) is a security-relevant architectural decision, not a Rule 1-3 auto-fix. Flagged for follow-up instead."

requirements-completed: []

duration: ~35min (blocked)
completed: 2026-07-23
---

# Phase 10 Plan 03: GET-and-hash gate over every fetchable dataset — BLOCKED before unbundling

**This plan did not complete.** Task 1's gate ran for real against `software-dl.ti.com` and
failed for all nine fetchable datasets — not because any digest is wrong, but because TI's
CDN now issues a cross-host redirect that `fetch_dataset()`'s security-hardened redirect
handler correctly refuses. Per the plan's own explicit instruction ("This gate is blocking.
If any dataset fails, stop: do not proceed to Task 2"), Tasks 2 and 3 (unbundling the build
scripts, lowering the size ceiling, correcting build-script comments) were **not executed**.

## What was built

**Task 1 — `scripts/verify_dataset_digests.py` (committed, working as designed).**

A re-runnable, committed script (not a heredoc) that, for every `DATASET_REGISTRY` entry
whose `dataset_url()` is not `None`, calls `mmcli.datasets.fetch_dataset(name, force=True)`
against a throwaway `XDG_CACHE_HOME` temp directory (with `MMCLI_DATASETS` unset before
import) and prints one `PASS`/`FAIL` line per dataset, exiting non-zero if any fails.
Supports `--only <name>` for iterating on a single dataset. Drives the exact code path a real
`mmcli datasets pull`/`init --dataset --fetch` invocation uses — no parallel downloader — so
a `FAIL` line here means a real user hitting the same failure today, not a gate artifact.

## The actual gate run (verbatim, `PYTHONPATH=$PWD ~/.venv-tinyml/bin/python scripts/verify_dataset_digests.py`)

```
arc_fault_classification: GET https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/arc_fault_classification_dsi.zip ...
ecg_classification: GET https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/ecg_classification_2class.zip ...
fan_blade_fault: GET https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/fan_blade_fault_dsi.zip ...
generic_timeseries_anomalydetection: GET https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/generic_timeseries_anomalydetection.zip ...
generic_timeseries_classification: GET https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/generic_timeseries_classification.zip ...
generic_timeseries_forecasting: GET https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/generic_timeseries_forecasting.zip ...
generic_timeseries_regression: GET https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/generic_timeseries_regression.zip ...
mnist_image_classification: GET https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/mnist_classes.zip ...
pir_detection: GET https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/pir_detection_classification_dsk.zip ...
arc_fault_classification: ...: FAIL — Refusing cross-host redirect while fetching 'arc_fault_classification': https://software-dl.ti.com/.../arc_fault_classification_dsi.zip -> https://downloads.ti.com/.../arc_fault_classification_dsi.zip
ecg_classification: ...: FAIL — Refusing cross-host redirect ... -> https://downloads.ti.com/.../ecg_classification_2class.zip
fan_blade_fault: ...: FAIL — Refusing cross-host redirect ... -> https://downloads.ti.com/.../fan_blade_fault_dsi.zip
generic_timeseries_anomalydetection: ...: FAIL — Refusing cross-host redirect ... -> https://downloads.ti.com/.../generic_timeseries_anomalydetection.zip
generic_timeseries_classification: ...: FAIL — Refusing cross-host redirect ... -> https://downloads.ti.com/.../generic_timeseries_classification.zip
generic_timeseries_forecasting: ...: FAIL — Refusing cross-host redirect ... -> https://downloads.ti.com/.../generic_timeseries_forecasting.zip
generic_timeseries_regression: ...: FAIL — Refusing cross-host redirect ... -> https://downloads.ti.com/.../generic_timeseries_regression.zip
mnist_image_classification: ...: FAIL — Refusing cross-host redirect ... -> https://downloads.ti.com/.../mnist_classes.zip
pir_detection: ...: FAIL — Refusing cross-host redirect ... -> https://downloads.ti.com/.../pir_detection_classification_dsk.zip

9/9 fetchable dataset(s) FAILED: arc_fault_classification, ecg_classification, fan_blade_fault, generic_timeseries_anomalydetection, generic_timeseries_classification, generic_timeseries_forecasting, generic_timeseries_regression, mnist_image_classification, pir_detection
```

Exit code: 1. (URLs abbreviated with `...` above only for readability in this summary; the
real output prints them in full, per the script's design.)

## Root cause: real TI infrastructure, not a bad digest

`_HostLockedRedirectHandler` (added in 10-02, mitigating threat T-10-02-01/05: an
unexplained cross-host redirect should not be followed silently, since a hijacked or
compromised redirect to an attacker-controlled host would otherwise be indistinguishable
from a legitimate one) refuses **any** redirect whose target host differs from the request
host. TI's CDN currently 302-redirects every `software-dl.ti.com/C2000/esd/mcu_ai/...` URL to
the equivalent `downloads.ti.com/C2000/esd/mcu_ai/...` path — for all nine datasets,
consistently. That is exactly the class of redirect the handler is designed to refuse, and it
does so correctly.

To determine whether this is TI republishing different bytes (D-1 revisited) or simply TI's
domain having moved (an operational fact, not tampering), all nine datasets were independently
downloaded with `curl -sL` (following the redirect, bypassing the code's host-lock entirely)
and hashed:

```
arc_fault_classification: MATCH
ecg_classification: MATCH
fan_blade_fault: MATCH
generic_timeseries_anomalydetection: MATCH
generic_timeseries_classification: MATCH
generic_timeseries_forecasting: MATCH
generic_timeseries_regression: MATCH
mnist_image_classification: MATCH
pir_detection: MATCH

9/9 content-verified via curl -L (bypassing the code's host-lock)
```

Every one of the nine registry `sha256`/`bytes` pairs matches the content TI actually serves
at `downloads.ti.com` byte-for-byte. **The registry digests are correct and current.** The
sole blocker is that `fetch_dataset()` — the exact function every real `mmcli datasets pull`
and `init --dataset --fetch` invocation calls today — refuses the redirect TI's CDN now
issues unconditionally, for all nine datasets. This means **10-02's fetch mechanism is
currently non-functional against TI's real production infrastructure**, for every dataset,
right now — not a hypothetical or a gate technicality.

## Why this was not auto-fixed here

Two candidate fixes exist, and both touch `mmcli/datasets.py`:

1. Point `TI_DATASETS_BASE` directly at `downloads.ti.com`, eliminating the redirect.
2. Extend `_HostLockedRedirectHandler` to allow this one specific, verified TI redirect pair
   (`software-dl.ti.com` → `downloads.ti.com`) rather than requiring exact host equality.

`mmcli/datasets.py` is **not** in this plan's `files_modified` list — it is 10-02's
deliverable — and either fix is a security-relevant change to a Tampering mitigation
(T-10-02-01/05), which this plan's own deviation rules classify as Rule 4 (architectural,
requires a decision) rather than a Rule 1-3 auto-fix. Making that call unilaterally inside a
plan scoped to build scripts and a verification script would be exactly the kind of scope
creep the plan boundaries exist to prevent. This is flagged here for a follow-up
plan/decision rather than patched in place.

**Tasks 2 and 3 were not started.** No `--add-data` line was touched in any of the three
build scripts, `scripts/binary_size_ceiling.txt` still reads `152043520`, and no build-script
comment was rewritten. `dist/mmcli` was not built in this session.

## Recommendation for follow-up

- Update `mmcli/datasets.py` (in a dedicated fix plan, or as a 10-02 patch) to either repoint
  `TI_DATASETS_BASE` at `downloads.ti.com` or allowlist this specific TI redirect pair in
  `_HostLockedRedirectHandler`.
- Re-run `python3 scripts/verify_dataset_digests.py` after that fix — it requires no changes
  itself and will report 9/9 PASS once `fetch_dataset()` can follow (or avoid) the redirect.
- Only then resume 10-03 Tasks 2/3 (unbundle, lower the ceiling to `15728640`, correct the
  build-script comments).

## Files Created/Modified

- `scripts/verify_dataset_digests.py` — new, 120 lines. GET-and-hash gate over every
  fetchable `DATASET_REGISTRY` entry, driven through `fetch_dataset(name, force=True)` against
  a throwaway cache directory; `--only <name>` supported; exits non-zero on any failure.

## Task Commits

1. **Task 1: GET-and-hash gate over every fetchable dataset** — `ffe1a67` (feat)

Tasks 2 and 3 were not started; no commits exist for them.

## Deviations from Plan

None in the sense of unauthorized changes — the plan's own text anticipated exactly this
outcome ("This gate is blocking. If any dataset fails, stop: do not proceed to Task 2... A
gate that cannot fail... is the exact hazard this plan exists to prevent") and this execution
followed that instruction to the letter. The one judgment call was *not* modifying
`mmcli/datasets.py` to route around the redirect — out of scope per `files_modified`, and a
Rule 4 (architectural/security) decision rather than an auto-fixable bug.

## Known Stubs

None introduced.

## Threat Flags

None introduced by this plan's own changes (only `scripts/verify_dataset_digests.py` was
added, and it makes no new claims about trust). Worth noting for the record, not as a new
flag from this plan: the pre-existing `_HostLockedRedirectHandler` mitigation (T-10-02-01/05)
is functioning exactly as designed — it is TI's own infrastructure change that now trips it,
which is the correct failure mode for a security control encountering an unanticipated
redirect, not a defect in the control itself.

## Self-Check: PASSED (for what was executed)

- FOUND: `scripts/verify_dataset_digests.py`
- FOUND commit `ffe1a67`
- Ran `PYTHONPATH=$PWD ~/.venv-tinyml/bin/python scripts/verify_dataset_digests.py`: exit 1,
  9/9 FAIL (redirect refusal), matching the verbatim output above.
- Independently verified all nine datasets' content via `curl -sL` + sha256: 9/9 MATCH against
  the registry, confirming the digests are correct and the failure is redirect-refusal only.
- Confirmed `scripts/binary_size_ceiling.txt` unchanged (`152043520`), `build_macos.sh` /
  `build_linux.sh` / `build_windows.ps1` unchanged (`git status --short` shows only the new
  script), `tests/test_build_config.py` unchanged and still passing (15 passed, baseline
  re-confirmed before this run).

## Next Phase Readiness

**Not ready.** 10-03 Tasks 2/3 (and anything downstream that assumes the binary is
unbundled — 10-04, 10-05, 10-08's per-artifact size gate) are blocked on a fix to
`mmcli/datasets.py`'s redirect handling or `TI_DATASETS_BASE`, landed and re-verified via
`scripts/verify_dataset_digests.py`, before this plan can resume.

---
*Phase: 10-dataset-distribution-and-binary-size*
*Completed: 2026-07-23 (blocked — Task 1 only)*
