"""
Dataset registry and project initialisation for mmcli.

Provides:
  - DATASET_REGISTRY — mapping of dataset names → metadata
  - list_datasets()  — query available datasets, optionally filtered by task
  - extract_dataset() — unzip an example dataset into a new project directory
"""

import os
import sys
import zipfile

# ---------------------------------------------------------------------------
# Where example zips are stored
# ---------------------------------------------------------------------------

def _datasets_dir() -> str:
    """Return the directory that holds example dataset zips.

    Priority:
      1. MMCLI_DATASETS env var
      2. mmcli/example_datasets/ (bundled with the package)
    """
    env = os.environ.get("MMCLI_DATASETS")
    if env and os.path.isdir(env):
        return env
    return os.path.join(os.path.dirname(__file__), "example_datasets")


# ---------------------------------------------------------------------------
# Registry — add entries as you add zips to example_datasets/
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, dict] = {
    "generic_timeseries_classification": {
        "filename": "generic_timeseries_classification.zip",
        "task_types": ["generic_timeseries_classification"],
        "module": "timeseries",
        "description": "Synthetic waveform classification (sawtooth, sine, square)",
    },
    "generic_timeseries_regression": {
        "filename": "generic_timeseries_regression.zip",
        "task_types": ["generic_timeseries_regression"],
        "module": "timeseries",
        "description": "Synthetic timeseries regression dataset",
    },
    "generic_timeseries_anomalydetection": {
        "filename": "generic_timeseries_anomalydetection.zip",
        "task_types": ["generic_timeseries_anomalydetection"],
        "module": "timeseries",
        "description": "Synthetic anomaly detection (amplitude/frequency shifts)",
    },
    "generic_timeseries_forecasting": {
        "filename": "generic_timeseries_forecasting.zip",
        "task_types": ["generic_timeseries_forecasting"],
        "module": "timeseries",
        "description": "Simulated thermostat temperature forecasting",
    },
    "arc_fault_classification": {
        "filename": "arc_fault_classification.zip",
        "task_types": ["arc_fault"],
        "module": "timeseries",
        "description": "DC arc fault current classification (DSI sensor)",
    },
    "ecg_classification": {
        "filename": "ecg_classification.zip",
        "task_types": ["ecg_classification", "generic_timeseries_classification"],
        "module": "timeseries",
        "description": "ECG 2-class heartbeat classification (normal vs abnormal)",
    },
    "fan_blade_fault": {
        "filename": "fan_blade_fault.zip",
        "task_types": ["motor_fault"],
        "module": "timeseries",
        "description": "Fan blade fault classification (vibration data, 3-axis)",
    },
    "pir_detection": {
        "filename": "pir_detection.zip",
        "task_types": ["pir_detection"],
        "module": "timeseries",
        "description": "PIR motion detection classification (human vs non-human)",
    },
    "mnist_image_classification": {
        "filename": "mnist_image_classification.zip",
        "task_types": ["image_classification"],
        "module": "vision",
        "description": "MNIST handwritten digit classification (28×28 images)",
    },
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def list_datasets(task_type: str | None = None,
                  module: str | None = None) -> list[dict]:
    """Return datasets matching the optional *task_type* and *module* filters.

    Each returned dict has keys: name, filename, task_types, module, description.
    """
    results = []
    for name, meta in DATASET_REGISTRY.items():
        if task_type and task_type not in meta.get("task_types", []):
            continue
        if module and meta.get("module") != module:
            continue
        results.append({"name": name, **meta})
    return results


def print_dataset_list(task_type: str | None = None,
                       module: str | None = None) -> None:
    """Print a formatted table of available datasets.

    Used by ``mmcli init --list`` to show the user what datasets are
    available before they create a project.
    """
    datasets = list_datasets(task_type=task_type, module=module)

    if not datasets:
        filters = []
        if task_type:
            filters.append(f"task={task_type}")
        if module:
            filters.append(f"module={module}")
        print(f"No datasets found matching: {', '.join(filters)}")
        return

    # Column widths (minimum padding)
    max_name = max(len(d["name"]) for d in datasets)
    max_tasks = max(len(", ".join(d["task_types"])) for d in datasets)
    max_mod = max(len(d.get("module", "")) for d in datasets)

    # Header
    print("\nAvailable example datasets:\n")
    hdr = (f" {'Dataset':<{max_name}}  {'Task Types':<{max_tasks}}  "
           f"{'Module':<{max_mod}}  Description")
    print(hdr)
    print("─" * len(hdr))

    for d in datasets:
        tasks_str = ", ".join(d["task_types"])
        print(f" {d['name']:<{max_name}}  {tasks_str:<{max_tasks}}  "
              f"{d.get('module', ''):<{max_mod}}  {d.get('description', '')}")

    print(f"\n{len(datasets)} dataset(s) available. Create a project with:")
    print("  mmcli init -t TASK_TYPE --dataset DATASET -p ./my_project\n")


def get_dataset(name: str) -> dict | None:
    """Look up a single dataset by name. Returns None if not found."""
    meta = DATASET_REGISTRY.get(name)
    if meta is None:
        return None
    return {"name": name, **meta}


def extract_dataset(dataset_name: str, project_path: str,
                    task_type: str | None = None) -> None:
    """Create *project_path* and extract the named dataset into it.

    The TI example dataset zips contain ``classes/`` (or ``files/``) and
    ``annotations/`` at the zip root.  We extract into
    ``<project_path>/dataset/`` so that the resulting project matches the
    layout expected by ``mmcli train``.

    Raises
    ------
    SystemExit
        On any error (unknown dataset, incompatible task, zip not found,
        target directory already exists).
    """
    meta = DATASET_REGISTRY.get(dataset_name)
    if meta is None:
        available = ", ".join(sorted(DATASET_REGISTRY.keys())) or "(none)"
        print(
            f"ERROR: Unknown dataset '{dataset_name}'.\n"
            f"Available datasets: {available}",
            file=sys.stderr,
        )
        sys.exit(2)

    # Validate task compatibility
    if task_type and task_type not in meta.get("task_types", []):
        compatible = ", ".join(meta["task_types"])
        print(
            f"ERROR: Dataset '{dataset_name}' is not compatible with "
            f"task '{task_type}'.\n"
            f"Compatible tasks: {compatible}",
            file=sys.stderr,
        )
        sys.exit(2)

    # Resolve paths
    project_path = os.path.abspath(project_path)
    if os.path.exists(project_path):
        print(
            f"ERROR: Project directory already exists: {project_path}\n"
            "Choose a different name or remove the existing directory.",
            file=sys.stderr,
        )
        sys.exit(2)

    zip_path = os.path.join(_datasets_dir(), meta["filename"])
    if not os.path.isfile(zip_path):
        print(
            f"ERROR: Dataset zip not found: {zip_path}\n"
            f"Place '{meta['filename']}' in the datasets directory:\n"
            f"  {_datasets_dir()}",
            file=sys.stderr,
        )
        sys.exit(2)

    # Create project directory and extract into dataset/ subdirectory
    dataset_dir = os.path.join(project_path, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dataset_dir)
    except zipfile.BadZipFile:
        print(
            f"ERROR: '{zip_path}' is not a valid zip file.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Verify expected subdirectories
    data_subdirs = ["classes", "files", "images"]
    has_data = any(
        os.path.isdir(os.path.join(dataset_dir, d)) for d in data_subdirs
    )
    if not has_data:
        print(
            f"WARNING: The extracted dataset does not contain any of "
            f"{', '.join(data_subdirs)}.  The project may not be usable "
            f"with 'mmcli train' as-is.",
            file=sys.stderr,
        )

    print(f"✓ Project created: {project_path}")
    print(f"  Dataset: {dataset_name} — {meta.get('description', '')}")
    print()
    print("Next steps:")
    print(f"  mmcli train -m {meta.get('module', 'MODULE')} "
          f"-t {(task_type or meta['task_types'][0])} "
          f"-d DEVICE -n MODEL -i {project_path}")

