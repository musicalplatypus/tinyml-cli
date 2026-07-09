# Phase 7 Research: Device & Task Coverage

**Date:** 2026-07-09  
**Phase:** 07 — Add F28E12 to TARGET_DEVICES + TASK_TYPES_AUDIO constant

---

## Finding 1: F28E12 Rejection Scope

`TARGET_DEVICES` (cli.py:98–111) is a Python list used in **two distinct ways**:

### Hard rejection (choices=)
`choices=TARGET_DEVICES` is applied at three argparse call sites:
- `info` subcommand `--device` argument (line 907) — hard reject, argparse error on unknown device
- `deploy sdk` subcommand `--device` (line 993) — hard reject
- `deploy crt` subcommand `--device` (line 1024) — hard reject

### Soft hint (metavar= only, no choices=)
`_add_common_args` wires `--device` with `metavar="DEVICE"` for `train`/`run`/`compile` — no choices enforcement. `F28E12` is accepted as a string at parse time for these subcommands.

### Insertion point
Current C2000 block (lines 99–101):
```python
# C2000
"F280013", "F280015", "F28003", "F28004", "F2837",
"F28P55", "F28P65", "F29H85", "F29P58", "F29P32",
```
`F28E12` belongs between `F2837` and `F28P55` (family order).

### Help text
The `--device` help string at lines 277–278 also lists C2000 devices as a human-readable block. Both the list and the help text need updating.

---

## Finding 2: Audio Task Discoverability

### No TASK_TYPES_AUDIO constant exists
`mmcli/cli.py` defines:
- `TASK_TYPES_TIMESERIES` (lines 84–95) — 9 task types  
- `TASK_TYPES_VISION` (line 96) — `["image_classification"]`

No `TASK_TYPES_AUDIO` constant exists, making `audio_classification` invisible in the CLI.

### audio module IS supported in info.py
`mmcli/info.py` lines 34–35 already conditionally import the audio module:
```python
elif module_name == "audio":
    from tinyml_modelmaker.ai_modules.audio import constants, training
```
The backend exists; only the CLI discoverability constant is missing.

### --task has no choices= enforcement
The `--task` argument across all subcommands uses `metavar="TASK_TYPE"` only (no `choices=`). Adding `TASK_TYPES_AUDIO` is purely a **discoverability** and **help text** improvement — not a gate change.

### NAS_SUPPORTED_TASKS is separate
`NAS_SUPPORTED_TASKS` (line 120) is the only task list with downstream validation logic (line 1282). `audio_classification` is NOT a NAS task — do not add it there.

---

## Impact Summary for Plans 07-01 and 07-02

**07-01 (implementation):**
1. Add `"F28E12"` between `"F2837"` and `"F28P55"` in `TARGET_DEVICES` — fixes hard rejection at info/deploy
2. Add `TASK_TYPES_AUDIO = ["audio_classification"]` after `TASK_TYPES_VISION` — enables help text referencing it
3. Update `--task` help text to append `"Audio tasks:\n  audio_classification"` in `_add_common_args`
4. Update `--device` help text C2000 line to include `F28E12`

**07-02 (tests):** Tests should verify:
- `F28E12` in `TARGET_DEVICES` (not rejected by choices=)
- `mmcli info -d F28E12` exits 0 (or sensible error) rather than argparse hard reject
- `TASK_TYPES_AUDIO` constant exists and contains `"audio_classification"`
- `--task` help includes `audio_classification`
- `TASK_TYPES_TIMESERIES`, `TASK_TYPES_VISION`, `TASK_TYPES_AUDIO` are mutually disjoint

---

## No changes needed to

- `NAS_SUPPORTED_TASKS` — audio_classification is not a NAS task
- `_add_common_args` device argument — already soft (`metavar=` only), F28E12 will work for train/run once in TARGET_DEVICES
- `mmcli/info.py` — already handles audio module imports
