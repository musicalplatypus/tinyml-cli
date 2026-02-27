"""
Build the nested config dict (and write YAML) from parsed CLI args.

The config structure mirrors what run_tinyml_modelmaker.py expects:
  common / dataset / data_processing_feature_extraction / training / testing / compilation
"""

import copy
import logging
import os
import tempfile

import yaml

logger = logging.getLogger(__name__)

# Mirrors constants.COMPILATION_DEFAULT in tinyml-modelmaker
COMPILATION_DEFAULT_PRESET = "default_preset"

# ---------------------------------------------------------------------------
# Minimal skeleton — every section that tinyml_modelmaker.main() reads
# ---------------------------------------------------------------------------

_SKELETON: dict = {
    "common": {
        "target_module": None,
        "task_type": None,
        "target_device": None,
        "run_name": "{date-time}/{model_name}",
        "verbose_mode": True,
    },
    "dataset": {
        "enable": False,
        "dataset_name": "default",
    },
    "data_processing_feature_extraction": {
        "feature_extraction_name": "default",
    },
    "training": {
        "enable": False,
        "model_name": None,
    },
    "testing": {
        "enable": True,
    },
    "compilation": {
        "enable": False,
        "model_path": None,
        "compile_preset_name": COMPILATION_DEFAULT_PRESET,
    },
}


def _load_yaml(path: str) -> dict:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}, got {type(data)}")
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _set(config: dict, *path_and_value) -> None:
    """Set config[key0][key1]... = value — skipped when value is None."""
    *path, value = path_and_value
    if value is None:
        return
    node = config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


def build_config(args) -> dict:
    """
    Build the nested config dict from a parsed argparse namespace.

    Priority (highest wins):
      CLI args  >  --config YAML file  >  built-in skeleton defaults
    """
    config = copy.deepcopy(_SKELETON)

    # Merge base YAML if provided
    if getattr(args, "config", None):
        base = _load_yaml(args.config)
        config = _deep_merge(config, base)
        logger.debug("Merged base config from %s", args.config)

    command = args.command

    # --- enable/disable pipeline steps ---
    config["dataset"]["enable"] = command in ("train", "run")
    config["training"]["enable"] = command in ("train", "run")
    config["compilation"]["enable"] = command in ("compile", "run")

    # --- common ---
    _set(config, "common", "target_module", getattr(args, "module", None))
    _set(config, "common", "task_type", getattr(args, "task", None))
    _set(config, "common", "target_device", getattr(args, "device", None))
    _set(config, "common", "run_name", getattr(args, "run_name", None))

    # --- project directory → dataset paths ---
    # -i/--project points to a project dir containing dataset/.
    # We set input_data_path to the original data and train_output_path to a
    # separate "run" directory so that modelmaker creates a working copy
    # (symlinks) in project_dir/run/dataset rather than clobbering the original.
    project_dir = getattr(args, "project", None)
    if project_dir:
        project_dir = os.path.abspath(project_dir)
        _set(config, "dataset", "input_data_path",
             os.path.join(project_dir, "dataset"))
        _set(config, "dataset", "dataset_name",
             os.path.basename(project_dir))
        _set(config, "training", "train_output_path",
             os.path.join(project_dir, "run"))

    # --- feature extraction ---
    _set(
        config,
        "data_processing_feature_extraction",
        "feature_extraction_name",
        getattr(args, "feature_extraction", None),
    )

    # --- training ---
    _set(config, "training", "model_name", getattr(args, "model", None))
    _set(config, "training", "training_epochs", getattr(args, "epochs", None))
    _set(config, "training", "batch_size", getattr(args, "batch_size", None))
    _set(config, "training", "learning_rate", getattr(args, "lr", None))
    _set(config, "training", "num_gpus", getattr(args, "gpus", None))
    _set(config, "training", "quantization", getattr(args, "quantization", None))

    # Performance optimization flags
    _set(config, "training", "compile_model", getattr(args, "compile_model", None))
    _set(config, "training", "native_amp", getattr(args, "native_amp", None))

    # --training-device: map 'auto' → omit (let tinyml_modelmaker decide),
    # 'mps'/'cuda'/'cpu' → set training_device explicitly.
    # tinyml_modelmaker selects MPS when num_gpus > 0 and MPS is available,
    # so 'mps' is achieved by setting training_device='mps' AND num_gpus=1.
    td = getattr(args, "training_device", None)
    if td and td != "auto":
        _set(config, "training", "training_device", td)
        # Ensure num_gpus is consistent: mps/cuda need num_gpus >= 1
        if td in ("mps", "cuda") and config["training"].get("num_gpus") is None:
            config["training"]["num_gpus"] = 1
        if td == "cpu":
            config["training"]["num_gpus"] = 0

    # --- compilation ---
    _set(config, "compilation", "model_path", getattr(args, "onnx", None))
    _set(config, "compilation", "compile_preset_name", getattr(args, "preset", None))

    # compile subcommand still needs model_name for preset lookup
    if command == "compile" and config["training"]["model_name"] is None:
        _set(config, "training", "model_name", getattr(args, "model", "unknown"))

    return config


def write_temp_yaml(config: dict) -> str:
    """Write config dict to a named temp YAML file and return its path."""
    fd, path = tempfile.mkstemp(prefix="mmcli_", suffix=".yaml")
    with os.fdopen(fd, "w") as fh:
        yaml.dump(config, fh, default_flow_style=False, sort_keys=False)
    logger.debug("Wrote temp config to %s", path)
    return path
