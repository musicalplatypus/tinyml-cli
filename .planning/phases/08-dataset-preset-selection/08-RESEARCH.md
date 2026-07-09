# Phase 8 Research: Dataset Preset Selection

**Date:** 2026-07-09  
**Phase:** 08 — Add `--dataset-preset` flag + extend `mmcli info` with dataset preset listing

---

## Finding 1: builder.py dataset_name is NOT hardcoded to "default"

The existing plan background states: "The builder currently hardcodes `"default"`".
This is **incorrect**. The skeleton default is "default" but it is immediately overridden:

```python
# builder.py lines 162-170
project_dir = getattr(args, "project", None)
if project_dir:
    project_dir = os.path.abspath(project_dir)
    _set(config, "dataset", "input_data_path", os.path.join(project_dir, "dataset"))
    _set(config, "dataset", "dataset_name", os.path.basename(project_dir))  # NOT "default"
    _set(config, "training", "train_output_path", os.path.join(project_dir, "run"))
```

`dataset_name` is set to `basename(project_dir)`. Since `--project` defaults to
`data/projects/default`, the effective value IS `"default"` for default runs, but via
`basename()`, not by hardcoding.

**Impact on wiring:** The `_set(config, "dataset", "dataset_name", dataset_preset)` call MUST
come AFTER the project_dir block (line 162+) so it overrides the basename when a preset is
provided. Omitting `--dataset-preset` leaves `dataset_name = basename(project_dir)`.

**Impact on tests:** `test_preset_none_uses_default` passes because
`basename("data/projects/default") == "default"` — the assertion is correct but the
mechanism is not "hardcoded". The test remains valid as written.

---

## Finding 2: _add_training_args insertion point

`_add_training_args` is defined at `mmcli/cli.py:345`. The `--feature-extraction` argument
is at lines 358–376. The `--dataset-preset` flag should be added **immediately after** the
closing parenthesis of the `--feature-extraction` `add_argument` call (after line 376).

The pattern is: `group.add_argument("--dataset-preset", dest="dataset_preset", metavar="PRESET", default=None, help=...)`

This is the `group` variable from `parser.add_argument_group("training options")` — same
group as `--feature-extraction`. Correct.

`_add_training_args` is called by `train` subcommand (line 621) and `run` subcommand (line 690).
NOT called by `compile` or `init` — so `--dataset-preset` correctly appears only in train/run.

---

## Finding 3: _QUERY_SCRIPT insertion for dataset presets

Current `_QUERY_SCRIPT` (info.py lines 22–101) ends with:
```python
    result["fe_presets"] = fe_presets

print(json.dumps(result))
```

Dataset presets should be added after `result["fe_presets"] = fe_presets` and before
`print(json.dumps(result))`.

**Pattern to use** (mirrors how fe_presets is constructed, with graceful fallback):
```python
    # --- Dataset presets ---
    if task_type:
        try:
            params = training.ModelRunner.init_params()
            dataset_preset_descs = training.ModelRunner.get_dataset_preset_descriptions(params)
            result["dataset_presets"] = list(dataset_preset_descs.keys())
        except Exception:
            result["dataset_presets"] = []
```

`training` is already imported at the top of `_QUERY_SCRIPT` — no extra import needed.
The try/except ensures that if `get_dataset_preset_descriptions` doesn't exist on this
ModelRunner version, `dataset_presets` silently returns `[]`.

**Alternative pattern from existing plan** uses `get_target_module` re-import — also fine,
but redundant given `training` is already in scope. Prefer the simpler approach.

---

## Finding 4: info.py text output insertion point

The text output function `_print_task_details` (info.py line ~220+) currently outputs:
1. Task name + supported devices
2. Models table
3. Feature Extraction Presets (line 258–263)
4. Example Datasets (line 265–280)

Dataset presets should appear between FE Presets and Example Datasets. After:
```python
    # --- FE Presets ---
    fe_presets = data.get("fe_presets", [])
    print(f"\nFeature Extraction Presets ({len(fe_presets)} available):")
    ...
```

Add:
```python
    # --- Dataset Presets ---
    dataset_presets = data.get("dataset_presets", [])
    if dataset_presets:
        print(f"\nDataset Presets --dataset-preset ({len(dataset_presets)} available):")
        for p in sorted(dataset_presets):
            print(f"  {p}")
```

Skip section if `dataset_presets` is empty (no need to print "0 available").

---

## Finding 5: JSON output requires no extra changes

`mmcli info --format json` at line 306 calls `format_json(data)` on the full result dict.
Adding `dataset_presets` to `result` in `_QUERY_SCRIPT` automatically includes it in JSON
output. No changes needed to the JSON path.

---

## Finding 6: `test_flag_absent_from_compile_help` — verify compile has no dataset_preset

The test `test_flag_absent_from_compile_help` asserts `--dataset-preset` does NOT appear in
`mmcli compile --help`. This is correct because `_add_training_args` is not called for
`compile`. The test is a valid safety check.

---

## No changes needed to

- `_set` helper in builder.py (already skips None)
- `mmcli/cli.py` TARGET_DEVICES or TASK_TYPES constants (Phase 7 work)
- `mmcli/info.py` JSON format dispatch (format_json(data) already handles new keys)
- Any module other than `mmcli/cli.py`, `mmcli/builder.py`, `mmcli/info.py`

---

## Plan corrections needed in existing plans

**08-01 Task background**: The background text claiming `dataset_name` is "hardcoded" to
`"default"` is misleading. The implementation is correct; only the description needs fixing.

**08-02 Task 1 `_QUERY_SCRIPT` insertion**: The plan uses `get_target_module` re-import.
Simpler to use `training` (already in scope). Both approaches work with try/except.
