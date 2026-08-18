"""
Dataset analysis for mmcli.

Inspects a project dataset directory (classes/ or files/ layout) and
reports total size, class distribution, min sequence length, and the
dataset size bucket needed by 'mmcli recommend'.
"""

import glob
import os
import pickle
import sys
from pathlib import Path
from typing import Any

# Import output formatting functions
from mmcli.output import format_json, format_csv, format_yaml, format_table


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bin_dataset(n: int) -> str:
    if n < 500:
        return "tiny"
    if n < 5000:
        return "small"
    if n < 50000:
        return "medium"
    return "large"


def _find_data_files(directory: str) -> list[str]:
    """Find data files in directory, case-insensitively.

    Globbing both `*.csv` and `*.CSV` is only safe where glob matches
    case-sensitively. It does not on Windows: glob compares through
    os.path.normcase(), which lowercases on Windows and is the identity on
    POSIX, so there both patterns match the *same* file and every result was
    returned twice. That silently doubled everything downstream -- file
    counts, per-class sample counts, and row totals -- so `mmcli analyze`
    reported twice the real numbers rather than failing visibly.

    Note this is a property of glob, not of the filesystem: macOS's default
    APFS is case-insensitive too, yet glob still matches case-sensitively
    there (verified), which is why the doubling was Windows-only.

    Dedupe on normcase -- the same function that causes the collision -- so
    the two cases stay consistent: on Windows the duplicate pair collapses,
    while on POSIX `a.csv` and `a.CSV` are genuinely different files and both
    are still returned.
    """
    exts = (".csv", ".txt", ".npy", ".pkl")
    found: dict[str, str] = {}
    for ext in exts:
        for pattern in (f"*{ext}", f"*{ext.upper()}"):
            for path in glob.glob(os.path.join(directory, pattern)):
                found.setdefault(os.path.normcase(path), path)
    return sorted(found.values())


def _row_count(file_path: str) -> int:
    """Return number of samples in a data file (first dimension)."""
    import numpy as np
    ext = Path(file_path).suffix.lower()
    if ext == ".csv":
        import pandas as pd
        return len(pd.read_csv(file_path))
    if ext == ".txt":
        import pandas as pd
        return len(pd.read_csv(file_path, sep=r"\s+|,|\t", engine="python"))
    if ext == ".npy":
        return int(np.load(file_path, mmap_mode="r").shape[0])
    if ext == ".pkl":
        import pandas as pd
        with open(file_path, "rb") as fh:
            obj = pickle.load(fh)
        if isinstance(obj, pd.DataFrame):
            return len(obj)
        if isinstance(obj, np.ndarray):
            return obj.shape[0]
        raise TypeError(f"Unsupported pickle type: {type(obj)}")
    raise ValueError(f"Unsupported extension: {ext}")


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _analyse_classes(dataset_path: str) -> dict[str, Any]:
    classes_dir = os.path.join(dataset_path, "classes")
    total, global_min = 0, float("inf")
    dist: dict[str, int] = {}

    for cls in sorted(os.listdir(classes_dir)):
        cls_path = os.path.join(classes_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        cls_count = 0
        for f in _find_data_files(cls_path):
            try:
                n = _row_count(f)
                global_min = min(global_min, n)
                total += n
                cls_count += n
            except Exception as exc:
                print(f"Warning: could not load {f}: {exc}", file=sys.stderr)
        dist[cls] = cls_count

    return {
        "layout": "classes",
        "total_samples": total,
        "min_sample_length": int(global_min) if global_min != float("inf") else None,
        "class_distribution": dist,
        "num_classes": len(dist),
        "dataset_bucket": _bin_dataset(total),
    }


def _analyse_files(dataset_path: str) -> dict[str, Any]:
    files_dir = os.path.join(dataset_path, "files")
    total, global_min = 0, float("inf")

    for f in _find_data_files(files_dir):
        try:
            n = _row_count(f)
            global_min = min(global_min, n)
            total += n
        except Exception as exc:
            print(f"Warning: could not load {f}: {exc}", file=sys.stderr)

    return {
        "layout": "files",
        "total_samples": total,
        "min_seq_length": int(global_min) if global_min != float("inf") else None,
        "dataset_bucket": _bin_dataset(total),
    }


def analyse_dataset(dataset_path: str) -> dict[str, Any]:
    """Auto-detect layout and return analysis dict. Never raises."""
    classes_dir = os.path.join(dataset_path, "classes")
    files_dir = os.path.join(dataset_path, "files")
    if os.path.isdir(classes_dir):
        return _analyse_classes(dataset_path)
    if os.path.isdir(files_dir):
        return _analyse_files(dataset_path)
    return {"error": f"No 'classes/' or 'files/' directory found in {dataset_path}"}


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_BUCKET_NOTES = {
    "tiny":   "< 500 samples — prefer simple models (≤ 2k params)",
    "small":  "500–4,999 samples — prefer models ≤ 10k params",
    "medium": "5,000–49,999 samples — models up to 30k params",
    "large":  "≥ 50,000 samples — no model size restriction",
}


def print_analysis(stats: dict, dataset_path: str) -> None:
    if "error" in stats:
        print(f"ERROR: {stats['error']}", file=sys.stderr)
        return

    bucket = stats["dataset_bucket"]
    total = stats["total_samples"]
    layout = stats["layout"]

    print(f"\nDataset Analysis: {dataset_path}\n")
    print(f"  Layout:        {layout}/")
    print(f"  Total samples: {total:,}")
    print(f"  Size bucket:   {bucket}  ({_BUCKET_NOTES[bucket]})")

    if layout == "classes":
        print(f"  Num classes:   {stats['num_classes']}")
        if stats.get("min_sample_length") is not None:
            print(f"  Min seq len:   {stats['min_sample_length']}")
        print(f"\n  Class distribution:")
        for cls, count in sorted(stats["class_distribution"].items()):
            pct = 100.0 * count / max(total, 1)
            bar = "█" * min(40, max(1, round(40 * count / max(total, 1))))
            print(f"    {cls:<24} {count:>6,}  {pct:>5.1f}%  {bar}")
    else:
        if stats.get("min_seq_length") is not None:
            print(f"  Min seq len:   {stats['min_seq_length']}")

    print()
    print("  Next step — pick a model with:")
    print(f"    mmcli recommend -m <module> -t <task_type> -d <device> "
          f"--dataset-size-bucket {bucket}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_analyze(args) -> None:
    project_dir = os.path.abspath(args.project)
    dataset_path = os.path.join(project_dir, "dataset")

    # Allow pointing directly at the dataset dir
    if not os.path.isdir(dataset_path):
        has_layout = (
            os.path.isdir(os.path.join(project_dir, "classes")) or
            os.path.isdir(os.path.join(project_dir, "files"))
        )
        if has_layout:
            dataset_path = project_dir
        else:
            print(
                f"ERROR: No dataset/ directory in {project_dir}.\n"
                "Use -i/--project to point at the project root.",
                file=sys.stderr,
            )
            sys.exit(2)

    stats = analyse_dataset(dataset_path)

    # Format output based on arguments
    if args.format == "json":
        output_text = format_json(stats)
    elif args.format == "csv":
        output_text = format_csv(stats)
    elif args.format == "yaml":
        output_text = format_yaml(stats)
    else:
        # Default to text format
        print_analysis(stats, dataset_path)
        if "error" in stats:
            sys.exit(1)
        return

    # Write to file or stdout
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_text)
    else:
        print(output_text)

    if "error" in stats:
        sys.exit(1)
