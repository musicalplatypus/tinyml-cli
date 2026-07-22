"""
Dataset registry and project initialisation for mmcli.

Provides:
  - DATASET_REGISTRY — mapping of dataset names → metadata
  - list_datasets()  — query available datasets, optionally filtered by task
  - extract_dataset() — unzip an example dataset into a new project directory
  - dataset_url()    — compose the version-pinned TI download URL for a dataset

REQ-DATA-02 invariant, enforced at import time (see _validate_registry below):
every entry that carries a ``ti_name`` (i.e. every dataset fetchable from TI)
MUST also carry a 64-hex-character ``sha256`` and a positive ``bytes``. A
fetchable dataset without a recorded digest is a configuration error and must
fail at import, not halfway through a multi-megabyte download. If you add a
new fetchable entry, you must add both fields or the module will refuse to
import.
"""

import hashlib
import os
import sys
import zipfile

# ---------------------------------------------------------------------------
# TI download source (D-1/D-3/D-4 in 10-RESEARCH.md)
# ---------------------------------------------------------------------------

TI_DATASETS_BASE = "https://software-dl.ti.com/C2000/esd/mcu_ai"

# TI's *engine* version, not an mmcli release tag. This is the version axis
# datasets are pinned to; a future plan keys the on-disk cache path on this
# value so bumping the pinned version can never silently reuse a dataset
# fetched under an older TI release. An individual registry entry may
# override this with its own `ti_version` key.
DATASETS_DEFAULT_VERSION = "01_03_00"

# ---------------------------------------------------------------------------
# Where example zips are stored
# ---------------------------------------------------------------------------

def _datasets_dir() -> str:
    """Return the directory that holds example dataset zips.

    Priority:
      1. MMCLI_DATASETS env var
      2. mmcli/example_datasets/ (bundled with the package)

    This function's behaviour is unchanged by the fetch/cache mechanism added
    alongside it — `_resolve_dataset_zip()` wraps this rather than replacing
    it, so any existing caller of `_datasets_dir()` keeps working exactly as
    before.
    """
    env = os.environ.get("MMCLI_DATASETS")
    if env and os.path.isdir(env):
        return env
    return os.path.join(os.path.dirname(__file__), "example_datasets")


def _cache_dir(version: str) -> str:
    """Return the version-scoped cache directory for downloaded datasets,
    creating it (mode 0700) if it does not already exist.

    Honours XDG_CACHE_HOME, falling back to ~/.cache. Resolves to
    ``<cache-home>/mmcli/datasets/<version>/``.

    The version is part of the path *deliberately*: a flat, version-less
    cache would let bumping DATASETS_DEFAULT_VERSION (or an entry's
    ti_version override) silently reuse a dataset downloaded under an older
    TI release — exactly the failure D-3 (10-RESEARCH.md) exists to prevent.
    Two versions therefore always cache independently.
    """
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    path = os.path.join(base, "mmcli", "datasets", version)
    os.makedirs(path, exist_ok=True)
    try:
        # os.makedirs's mode= argument is masked by umask, so set the
        # permission explicitly rather than relying on it.
        os.chmod(path, 0o700)
    except OSError:  # pragma: no cover - best effort, e.g. on some CI runners
        pass
    return path


def _sha256_of(path: str) -> str:
    """Return the hex sha256 digest of the file at *path*, read in chunks."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()




# ---------------------------------------------------------------------------
# Registry — add entries as you add zips to example_datasets/
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, dict] = {
    "generic_timeseries_classification": {
        "filename": "generic_timeseries_classification.zip",
        "task_types": ["generic_timeseries_classification"],
        "module": "timeseries",
        "description": "Synthetic waveform classification (sawtooth, sine, square)",
        "ti_name": "generic_timeseries_classification.zip",
        "sha256": "7b2c0980bb30c3bc661004d66373d7ea35ea13ab5b6f8b74f5182c3bc6bc09c1",
        "bytes": 2579940,
    },
    "generic_timeseries_regression": {
        "filename": "generic_timeseries_regression.zip",
        "task_types": ["generic_timeseries_regression"],
        "module": "timeseries",
        "description": "Synthetic timeseries regression dataset",
        "ti_name": "generic_timeseries_regression.zip",
        "sha256": "078d212b00112bcaca4b1bb68b871e8c24eb3ed809b610d64642a74a7854cc23",
        "bytes": 906660,
    },
    "generic_timeseries_anomalydetection": {
        "filename": "generic_timeseries_anomalydetection.zip",
        "task_types": ["generic_timeseries_anomalydetection"],
        "module": "timeseries",
        "description": "Synthetic anomaly detection (amplitude/frequency shifts)",
        "ti_name": "generic_timeseries_anomalydetection.zip",
        "sha256": "7cb2f9fd183fa5c6730abdd0a144e1ce57f7ece9ed93d8663b19a983cde6d6b5",
        "bytes": 4242845,
    },
    "generic_timeseries_forecasting": {
        "filename": "generic_timeseries_forecasting.zip",
        "task_types": ["generic_timeseries_forecasting"],
        "module": "timeseries",
        "description": "Simulated thermostat temperature forecasting",
        "ti_name": "generic_timeseries_forecasting.zip",
        "sha256": "4ae6e7e436817a8ee5f3e528e70741b9c6fabfeb6c19a9fdb321dabad0a804ce",
        "bytes": 71053,
    },
    "arc_fault_classification": {
        "filename": "arc_fault_classification.zip",
        "task_types": ["arc_fault"],
        "module": "timeseries",
        "description": "DC arc fault current classification (DSI sensor)",
        "ti_name": "arc_fault_classification_dsi.zip",
        "sha256": "bcee7b54fb42079bfac1f4a39266fb836c2ef73c3f8fffd8fa04c41671f7656e",
        "bytes": 13290076,
    },
    "ecg_classification": {
        "filename": "ecg_classification.zip",
        "task_types": ["ecg_classification", "generic_timeseries_classification"],
        "module": "timeseries",
        "description": "ECG 2-class heartbeat classification (normal vs abnormal)",
        "ti_name": "ecg_classification_2class.zip",
        "sha256": "881ac26e95378eca9c1979cf1c70a8d1b8f2cb73da65e264a03bf1849c6addc6",
        "bytes": 4651662,
    },
    "fan_blade_fault": {
        "filename": "fan_blade_fault.zip",
        "task_types": ["motor_fault"],
        "module": "timeseries",
        "description": "Fan blade fault classification (vibration data, 3-axis)",
        "ti_name": "fan_blade_fault_dsi.zip",
        "sha256": "5194925e0f97387a54be989923ec34bef8e65e03fe21652552d7bbcdc21a959e",
        "bytes": 56595859,
    },
    "pir_detection": {
        "filename": "pir_detection.zip",
        "task_types": ["pir_detection"],
        "module": "timeseries",
        "description": "PIR motion detection classification (human vs non-human)",
        "ti_name": "pir_detection_classification_dsk.zip",
        "sha256": "d75470c9ba7f56fd4e8801c9f10424262e9935513b9011f55f5f5ed406ae0b0e",
        "bytes": 1579936,
    },
    "mnist_image_classification": {
        "filename": "mnist_image_classification.zip",
        "task_types": ["image_classification"],
        "module": "vision",
        "description": "MNIST handwritten digit classification (28×28 images)",
        "ti_name": "mnist_classes.zip",
        "sha256": "7fa4be9944a364074dc796d5d802dad8f1636f2f4daa6fd735d15f5fe05f3db8",
        "bytes": 46993516,
    },
    "generic_audio_classification": {
        "filename": "generic_audio_classification.zip",
        "task_types": ["audio_classification"],
        "module": "audio",
        "description": "Synthetic 2-class audio (yes/no) — 16kHz sine-wave WAV files",
        # No ti_name: this set is locally authored (be06559) and has no TI
        # upstream, so it stays bundled and is never fetched. It still
        # carries sha256/bytes for completeness and future integrity checks.
        "sha256": "dfc463e6a0aac80b2db36770e9fc56090f319d400d416b391d160d70382dbc5d",
        "bytes": 18371,
    },
}


def _validate_registry(registry: dict) -> None:
    """Enforce REQ-DATA-02 at import time: every entry with a ``ti_name``
    (i.e. every dataset that can be fetched from TI) must carry a valid
    64-hex-character ``sha256`` and a positive ``bytes``. Raises ValueError
    naming the offending entry rather than letting a fetchable-but-undigested
    dataset surface as a runtime surprise partway through a download.
    """
    hex_digits = set("0123456789abcdef")
    for name, meta in registry.items():
        if not meta.get("ti_name"):
            continue
        sha256 = meta.get("sha256", "")
        size = meta.get("bytes", 0)
        valid_sha256 = (
            isinstance(sha256, str)
            and len(sha256) == 64
            and set(sha256.lower()) <= hex_digits
        )
        valid_bytes = isinstance(size, int) and size > 0
        if not (valid_sha256 and valid_bytes):
            raise ValueError(
                f"DATASET_REGISTRY['{name}'] has a ti_name "
                f"({meta['ti_name']!r}) but is missing a valid sha256/bytes "
                f"pair (REQ-DATA-02). Every entry that can be fetched from "
                f"TI must carry a 64-hex-character sha256 and a positive "
                f"byte count, checked at import so a missing digest is a "
                f"configuration error, not a runtime surprise."
            )


_validate_registry(DATASET_REGISTRY)


def dataset_url(name: str) -> str | None:
    """Return the version-pathed TI download URL for *name*, or ``None`` when
    the entry has no ``ti_name`` (i.e. it is locally authored and bundled
    only, such as ``generic_audio_classification``).

    Looks the name up through ``DATASET_REGISTRY`` — never composes a URL
    from a caller-supplied string directly — so an unknown name raises
    ``KeyError`` instead of ever reaching URL construction.

    Uses the version-pathed URL form (``.../<version>/datasets/<ti_name>``),
    not the flat form: the flat path is effectively "latest" and can drift
    underneath a pinned digest, while the versioned path is the closest
    thing TI offers to immutability.
    """
    meta = DATASET_REGISTRY[name]  # KeyError on unknown name, deliberately
    ti_name = meta.get("ti_name")
    if ti_name is None:
        return None
    version = meta.get("ti_version") or DATASETS_DEFAULT_VERSION
    return f"{TI_DATASETS_BASE}/{version}/datasets/{ti_name}"



def _resolve_dataset_zip(name: str) -> str | None:
    """Resolve the on-disk path for *name*'s zip, or ``None`` if it is not
    available anywhere (the caller — `extract_dataset` or the CLI — decides
    whether to fetch it).

    Resolution order:
      1. MMCLI_DATASETS env var (existing offline/air-gap escape hatch)
      2. bundled example_datasets/ (existing, wraps `_datasets_dir()`)
      3. ~/.cache/mmcli/datasets/<version>/ (new — previously downloaded)

    The cache sits *below* the env var and the bundled directory
    deliberately: a user who pointed MMCLI_DATASETS at an air-gapped or
    reproducible dataset directory must never be silently overridden by a
    cached copy, and a bundled zip (shipped with the package) is preferred
    over a cache entry for the same reason existing behaviour did not
    consult a cache at all.

    Digest verification is asymmetric on purpose: files resolved through
    MMCLI_DATASETS are **not** digest-checked, because that directory is
    explicitly user-managed and may legitimately hold a locally prepared
    substitute dataset (REQ-DATA-03). Cache hits **are** re-verified against
    the registry sha256 on every resolution, not only at download time,
    because the cache directory may be writable by another process or user
    that can plant a file there — re-hashing costs tens of milliseconds
    against a download that costs seconds. A cache file that fails
    verification is treated as absent (not returned) and reported to
    stderr, never used.
    """
    meta = DATASET_REGISTRY.get(name)
    if meta is None:
        return None
    filename = meta["filename"]

    # Steps 1-2: MMCLI_DATASETS env var, then bundled example_datasets/.
    # _datasets_dir() already implements exactly this precedence as a single
    # directory choice; we look inside it for the specific file rather than
    # replacing its logic.
    primary_dir = _datasets_dir()
    candidate = os.path.join(primary_dir, filename)
    if os.path.isfile(candidate):
        return candidate

    env = os.environ.get("MMCLI_DATASETS")
    if env and os.path.isdir(env):
        # MMCLI_DATASETS is set and is a real directory, but does not hold
        # this file. It is the user's authoritative, explicitly configured
        # source — do not fall through to the cache, which would silently
        # override an air-gapped setup with a network-fetched copy.
        return None

    # Step 3: version-scoped cache, digest-verified on every hit.
    version = meta.get("ti_version") or DATASETS_DEFAULT_VERSION
    cache_path = os.path.join(_cache_dir(version), filename)
    if os.path.isfile(cache_path):
        expected = meta.get("sha256")
        if expected and _sha256_of(cache_path) == expected:
            return cache_path
        print(
            f"WARNING: cached copy of '{name}' at {cache_path} does not "
            f"match the recorded sha256; treating it as absent. Run "
            f"'mmcli datasets pull {name} --force' to redownload.",
            file=sys.stderr,
        )
        return None

    return None



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

    zip_path = _resolve_dataset_zip(dataset_name)
    if zip_path is None:
        hint = (
            f"Run `mmcli datasets pull {dataset_name}` to fetch it, "
            if meta.get("ti_name")
            else ""
        )
        print(
            f"ERROR: Dataset zip not found for '{dataset_name}'.\n"
            f"{hint}or place '{meta['filename']}' in the datasets "
            f"directory:\n  {_datasets_dir()}",
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

