"""Channel-aware feature-extraction preset selection.

Why this exists
---------------
`mmcli train` without ``--feature-extraction`` used to fail for nearly every
timeseries task, because the preset in play expected a different number of input
channels than the dataset actually has. Two measured cases:

* ``generic_timeseries_classification``: shipped dataset is 1 column, the task's
  default preset is ``..._3InputChannel_...`` (``variables=3``) →
  ``Not enough dimensions present. Extract more features``.
* ``generic_timeseries_regression``: shipped dataset is ``x,y`` (1 input + 1
  target), the only non-default preset is ``variables=11`` →
  ``index 2 is out of bounds for axis 0 with size 2`` — where ``size 2`` is
  literally the dataset's column count.

Every preset in ``FEATURE_EXTRACTION_PRESET_DESCRIPTIONS`` carries a
machine-readable ``variables`` (channel count) and ``common.task_type``, so the
right preset can be chosen from facts rather than guessed from its name.

See ``.planning/SPEC-channel-aware-preset-selection.md`` and
``.planning/FINDINGS-training-matrix.md`` F-5.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

# Tasks whose LAST column is a continuous target rather than an input channel.
# Contract read from tinyml_tinyverse GenericTSDatasetReg:
#   "Load data file with continuous target values from the last column."
#   (timeseries_dataset.py:1172, y_temp = data[:, -1] at :1188)
TASKS_WITH_TRAILING_TARGET = frozenset({
    "generic_timeseries_regression",
})

# Matching the channel count is necessary but NOT sufficient. `Custom_Default`
# declares variables=1 yet has `feat_ext_transform=[]` and no frame structure —
# it is an empty template to be filled in by user-supplied custom parameters, not
# a usable default. Selecting it produces no features at all, the tensor stays
# 2-D, and _rearrange_dims raises the very error this module exists to prevent.
# Verified: choosing Custom_Default for both classification and regression still
# failed with "Not enough dimensions present".
#
# A preset is usable only if it actually extracts something and declares the
# frame structure that yields a 3-D tensor.
_FRAME_STRUCTURE_KEYS = ("frame_size", "feature_size_per_frame", "num_frame_concat")

# Among usable candidates, prefer a RAW passthrough: it makes no assumption about
# the signal's frequency content, which is the safest automatic choice when we
# know nothing about the user's data beyond its shape.
PREFERRED_TRANSFORM_PREFIX = "RAW_FE"

# Query executed inside MMCLI_PYTHON, where tinyml_modelmaker is importable.
# The packaged binary excludes modelmaker by design, so this must not run in
# mmcli's own interpreter.
_PRESET_QUERY = '''
import json
from tinyml_modelmaker.ai_modules.timeseries import constants

task_type = {task_type!r}
out = []
for name, desc in constants.FEATURE_EXTRACTION_PRESET_DESCRIPTIONS.items():
    common = desc.get("common", {{}}) if hasattr(desc, "get") else {{}}
    pt = common.get("task_type") if hasattr(common, "get") else None
    types = pt if isinstance(pt, list) else ([pt] if pt is not None else [])
    if types and task_type not in types:
        continue
    fe = desc.get("data_processing_feature_extraction", {{}}) if hasattr(desc, "get") else {{}}
    out.append({{
        "name": name,
        "variables": fe.get("variables"),
        "feat_ext_transform": fe.get("feat_ext_transform"),
        "frame_size": fe.get("frame_size"),
        "feature_size_per_frame": fe.get("feature_size_per_frame"),
        "num_frame_concat": fe.get("num_frame_concat"),
    }})
print(json.dumps(out))
'''


class PresetSelectionError(Exception):
    """Raised when no compatible preset can be chosen.

    Deliberately fatal rather than falling back to an arbitrary preset: an
    arbitrary choice would reproduce the exact failure this module exists to
    prevent, one layer further from the cause.
    """


class PresetQueryError(Exception):
    """Raised when the preset catalog could not be read at all.

    Distinct from PresetSelectionError on purpose. An earlier version returned
    an empty list on any subprocess failure, so a broken MMCLI_PYTHON
    environment was reported to the user as "this task has no feature-extraction
    presets available" — blaming the catalog for what was actually an unusable
    interpreter. Not being able to ask is not the same as being told there are
    none.
    """


def _is_number(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def count_columns_csv(path: str) -> int | None:
    """Count usable input columns in a CSV, applying the loader's own rules.

    Mirrors two behaviours of tinyml_tinyverse's reader:
      * a header row is tolerated (regression's dataset has ``x,y``; the
        classification one has none), and
      * columns whose first value contains "time" are dropped
        (timeseries_dataset.py:708).
    """
    try:
        with open(path, newline="") as fh:
            rows = []
            for row in csv.reader(fh):
                if row:
                    rows.append(row)
                if len(rows) >= 2:
                    break
    except OSError:
        return None
    if not rows:
        return None

    first = rows[0]
    header = None
    if not all(_is_number(c) for c in first):
        header = first
        data_row = rows[1] if len(rows) > 1 else None
    else:
        data_row = first
    if data_row is None:
        return None

    keep = [
        i for i in range(len(data_row))
        if "time" not in str((header[i] if header and i < len(header) else data_row[i])).lower()
    ]
    return len(keep) or None


def count_columns_npy(path: str) -> int | None:
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        arr = np.load(path, allow_pickle=False, mmap_mode="r")
    except Exception:
        return None
    if getattr(arr, "ndim", 0) == 1:
        return 1
    return int(arr.shape[-1]) if getattr(arr, "ndim", 0) >= 2 else None


def detect_channels(dataset_dir: str, task: str, sample: int = 3) -> int | None:
    """Detect the dataset's input-channel count.

    Samples up to *sample* files from different class directories and requires
    them to agree — an inconsistent dataset is a real condition and is reported
    (as ``None``) rather than averaged into a plausible-looking answer.
    """
    candidates: list[str] = []
    for root, _dirs, files in os.walk(dataset_dir):
        if os.path.basename(root) == "annotations":
            continue
        for fn in sorted(files):
            if fn.lower().endswith((".csv", ".npy")):
                candidates.append(os.path.join(root, fn))
                break  # one file per directory, so we span classes
        if len(candidates) >= sample:
            break
    if not candidates:
        return None

    counts = set()
    for path in candidates[:sample]:
        n = (count_columns_npy(path) if path.lower().endswith(".npy")
             else count_columns_csv(path))
        if n:
            counts.add(n)
    if len(counts) != 1:
        if len(counts) > 1:
            logger.warning(
                "Dataset files disagree on column count (%s); "
                "skipping automatic feature-extraction preset selection.",
                sorted(counts),
            )
        return None

    columns = counts.pop()
    if task in TASKS_WITH_TRAILING_TARGET:
        # The last column is the continuous target, not an input channel.
        columns -= 1
    return columns or None


def query_presets(python_exe: str, task: str) -> list[dict]:
    """Return ``[{"name": ..., "variables": ...}]`` for *task*.

    Runs inside MMCLI_PYTHON because tinyml_modelmaker is not importable from
    the packaged mmcli binary.
    """
    script = _PRESET_QUERY.format(task_type=task)
    try:
        proc = subprocess.run([python_exe, "-c", script],
                              capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        raise PresetQueryError(
            f"could not run {python_exe!r} to read the preset catalog: {exc}"
        ) from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        detail = tail[-1] if tail else "(no stderr)"
        raise PresetQueryError(
            f"reading the preset catalog via {python_exe!r} failed "
            f"(exit {proc.returncode}): {detail}"
        )
    if not proc.stdout.strip():
        raise PresetQueryError(
            f"{python_exe!r} produced no output when reading the preset catalog"
        )
    try:
        return json.loads(proc.stdout.strip())
    except ValueError as exc:
        raise PresetQueryError(
            f"could not parse the preset catalog returned by {python_exe!r}: {exc}"
        ) from exc


def choose_preset(presets: list[dict], channels: int, task: str) -> str:
    """Pick a preset whose ``variables`` matches *channels*.

    Raises PresetSelectionError when nothing matches, rather than guessing.
    """
    if not presets:
        raise PresetSelectionError(
            f"Task '{task}' has no feature-extraction presets available, so one "
            f"cannot be selected automatically. This is a gap in the upstream "
            f"preset catalog, not a problem with your dataset."
        )
    right_width = [p for p in presets if p.get("variables") == channels]
    # Discard presets that would not actually produce features (see the
    # _FRAME_STRUCTURE_KEYS comment above).
    usable = [
        p for p in right_width
        if p.get("feat_ext_transform")
        and all(p.get(k) is not None for k in _FRAME_STRUCTURE_KEYS)
    ]
    if not usable:
        available = ", ".join(
            f"{p['name']} (expects {p.get('variables')}"
            + ("" if p.get("feat_ext_transform") else ", no transforms")
            + ")"
            for p in presets
        ) or "none"
        matched_but_unusable = [p["name"] for p in right_width]
        detail = ""
        if matched_but_unusable:
            detail = (
                f"\n  Matched on channel count but unusable (no feature "
                f"transforms declared): {', '.join(matched_but_unusable)}"
            )
        raise PresetSelectionError(
            f"No usable feature-extraction preset for task '{task}' matches the "
            f"{channels} input channel(s) detected in your dataset."
            f"{detail}\n"
            f"  Available for this task: {available}\n"
            f"  Pass --feature-extraction explicitly to override this check."
        )
    raw_first = [
        p["name"] for p in usable
        if (p.get("feat_ext_transform") or [None])[0] == PREFERRED_TRANSFORM_PREFIX
    ]
    return sorted(raw_first)[0] if raw_first else sorted(p["name"] for p in usable)[0]


def select_for_project(project: str, task: str, python_exe: str) -> str | None:
    """Full selection. Returns a preset name, or None if detection was not possible.

    Detection failure is non-fatal (we simply do not intervene); an *incompatible*
    dataset is fatal, via PresetSelectionError from choose_preset().
    """
    dataset_dir = os.path.join(project, "dataset")
    if not os.path.isdir(dataset_dir):
        return None
    channels = detect_channels(dataset_dir, task)
    if not channels:
        return None
    try:
        presets = query_presets(python_exe, task)
    except PresetQueryError as exc:
        # Not being able to read the catalog is not the user's dataset being
        # wrong, and it must not be reported as such. Decline to intervene and
        # let modelmaker surface its own error, which will be the real one.
        print(
            f"WARNING: skipping automatic feature-extraction preset selection — "
            f"{exc}",
            file=sys.stderr,
        )
        return None
    chosen = choose_preset(presets, channels, task)
    compatible_n = sum(1 for p in presets if p.get("variables") == channels)
    print(
        f"Detected {channels} input channel(s); selected feature-extraction "
        f"preset '{chosen}' ({compatible_n} compatible). "
        f"Override with --feature-extraction.",
        file=sys.stderr,
    )
    return chosen
