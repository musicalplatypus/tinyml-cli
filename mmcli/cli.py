"""
mmcli — command-line interface for tinyml-modelmaker.

Subcommands:
  train    Train a model and export ONNX (no compilation)
  compile  Compile an existing ONNX file (no training)
  run      Full pipeline: train then compile
  help     Show detailed help for all subcommands

Runtime requirements (not bundled in the binary):
  MMCLI_PYTHON  Path to the Python interpreter that has tinyml_modelmaker
                installed, e.g. /path/to/venv/bin/python
                Defaults to the 'python' found on PATH.

  MMCLI_MODELMAKER  Path to the tinyml-modelmaker source directory
                    (the folder containing tinyml_modelmaker/).
                    Defaults to the directory of run_tinyml_modelmaker.py
                    resolved from the tinyml_modelmaker package itself.
"""

import argparse
import logging
import os
import subprocess
import sys

from mmcli import __version__
from mmcli.builder import build_config, write_temp_yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known enumerations (for --help text)
# ---------------------------------------------------------------------------

MODULES = ["timeseries", "vision"]

TASK_TYPES_TIMESERIES = [
    "generic_timeseries_classification",
    "generic_timeseries_regression",
    "generic_timeseries_anomalydetection",
    "generic_timeseries_forecasting",
    "arc_fault",
    "ecg_classification",
    "motor_fault",
    "blower_imbalance",
    "pir_detection",
]

TASK_TYPES_VISION = ["image_classification"]

TARGET_DEVICES = [
    # C2000
    "F280013", "F280015", "F28003", "F28004", "F2837",
    "F28P55", "F28P65", "F29H85", "F29P58", "F29P32",
    # MSPM0
    "MSPM0G3507", "MSPM0G3519", "MSPM0G5187",
    "MSPM33C32", "MSPM33C34",
    # SimpleLink
    "CC2755", "CC1352", "CC1354", "CC35X1",
    # Sitara
    "AM263", "AM263P", "AM261", "AM13E2",
]

QUANTIZATION_OPTIONS = ["NO_QUANTIZATION", "QUANTIZATION_TINPU"]

TRAINING_DEVICES = ["auto", "mps", "cuda", "cpu"]

NAS_SIZES = ["s", "m", "l", "xl"]

NAS_OPTIMIZE_MODES = ["Memory", "Compute"]

# Task types that support NAS (classification only)
NAS_SUPPORTED_TASKS = [
    "generic_timeseries_classification",
    "arc_fault",
    "ecg_classification",
    "motor_fault",
    "blower_imbalance",
    "pir_detection",
    "image_classification",
]


# ---------------------------------------------------------------------------
# Metal / MPS detection
# ---------------------------------------------------------------------------

def _detect_training_device() -> str:
    """
    Return the best available training device on this machine.
    Preference order: MPS (Apple Metal) > CUDA > CPU.

    Uses platform heuristics so that torch does NOT need to be importable
    in the CLI binary itself (torch lives in the external MMCLI_PYTHON env).
    """
    import platform
    import subprocess as _sp
    import sys as _sys

    system = platform.system()

    # Fast macOS heuristic: MPS is available on Apple Silicon or Intel Macs
    # with macOS 12.3+ and a GPU.  Check via system_profiler if available.
    if system == "Darwin":
        try:
            out = _sp.check_output(
                ["system_profiler", "SPDisplaysDataType"],
                stderr=_sp.DEVNULL, timeout=3, text=True,
            )
            if "Metal" in out or "Apple" in out:
                return "mps"
        except Exception:
            pass
        # Fallback: any macOS likely has Metal
        return "mps"

    # On Linux/Windows check for CUDA via nvidia-smi
    if system in ("Linux", "Windows"):
        try:
            _sp.check_output(["nvidia-smi"], stderr=_sp.DEVNULL, timeout=3)
            return "cuda"
        except Exception:
            pass

    return "cpu"


# ---------------------------------------------------------------------------
# Locate the run_tinyml_modelmaker.py entry point
# ---------------------------------------------------------------------------

def _find_runner_script(python_exe: str) -> str:
    """
    Return the path to run_tinyml_modelmaker.py.

    Search order:
      1. MMCLI_MODELMAKER env var (user-supplied path to the modelmaker dir)
      2. Ask the Python interpreter where tinyml_modelmaker is installed
    """
    env_dir = os.environ.get("MMCLI_MODELMAKER")
    if env_dir:
        script = os.path.join(env_dir, "tinyml_modelmaker", "run_tinyml_modelmaker.py")
        if os.path.isfile(script):
            return script
        script2 = os.path.join(env_dir, "run_tinyml_modelmaker.py")
        if os.path.isfile(script2):
            return script2
        raise FileNotFoundError(
            f"MMCLI_MODELMAKER is set to '{env_dir}' but "
            f"run_tinyml_modelmaker.py was not found there."
        )

    probe = subprocess.run(
        [
            python_exe, "-c",
            "import tinyml_modelmaker, os; "
            "print(os.path.join(os.path.dirname(tinyml_modelmaker.__file__), "
            "'run_tinyml_modelmaker.py'))"
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            f"Could not locate tinyml_modelmaker using '{python_exe}'.\n"
            f"stderr: {probe.stderr.strip()}\n\n"
            "Set MMCLI_PYTHON to a Python interpreter that has "
            "tinyml_modelmaker installed, or set MMCLI_MODELMAKER to the "
            "tinyml-modelmaker source directory."
        )
    return probe.stdout.strip()


def _get_python_exe() -> str:
    return os.environ.get("MMCLI_PYTHON", "python")


# ---------------------------------------------------------------------------
# Shared argument groups
# ---------------------------------------------------------------------------

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    # These are logically required but marked optional here so that
    # --config <file> can supply them without repeating on the CLI.
    # Validated post-parse in _validate_args().
    parser.add_argument(
        "-m", "--module",
        default=None,
        choices=MODULES,
        metavar="MODULE",
        help=(
            "AI module to use.\n"
            "  timeseries  Time-series classification, regression, etc.\n"
            "  vision      Image classification\n"
            "Required unless provided via --config."
        ),
    )
    parser.add_argument(
        "-t", "--task",
        default=None,
        metavar="TASK_TYPE",
        help=(
            "Task type (required unless provided via --config).\n"
            "Timeseries tasks:\n"
            "  generic_timeseries_classification\n"
            "  generic_timeseries_regression\n"
            "  generic_timeseries_anomalydetection\n"
            "  generic_timeseries_forecasting\n"
            "  arc_fault  ecg_classification  motor_fault\n"
            "  blower_imbalance  pir_detection\n"
            "Vision tasks:\n"
            "  image_classification"
        ),
    )
    parser.add_argument(
        "-d", "--device",
        default=None,
        metavar="DEVICE",
        help=(
            "Target microcontroller device (required unless provided via --config).\n"
            "C2000:     F280013  F280015  F28003  F28004  F2837\n"
            "           F28P55  F28P65  F29H85  F29P58  F29P32\n"
            "MSPM0:     MSPM0G3507  MSPM0G3519  MSPM0G5187\n"
            "           MSPM33C32  MSPM33C34\n"
            "SimpleLink: CC2755  CC1352  CC1354  CC35X1\n"
            "Sitara:    AM263  AM263P  AM261  AM13E2"
        ),
    )
    parser.add_argument(
        "-n", "--model",
        default=None,
        metavar="MODEL_NAME",
        help=(
            "Model name from the catalog (required unless provided via --config).\n"
            "Classification:\n"
            "  NPU:   CLS_100_NPU  CLS_500_NPU  CLS_1k_NPU  CLS_2k_NPU\n"
            "         CLS_4k_NPU  CLS_6k_NPU  CLS_8k_NPU  CLS_13k_NPU\n"
            "         CLS_20k_NPU  CLS_55k_NPU  ECG_55k_NPU\n"
            "  Res:   CLS_ResAdd_3k  CLS_ResCat_3k\n"
            "  App:   ArcFault_model_{200,300,700,1400}_t\n"
            "         MotorFault_model_{1,2,3}_t\n"
            "         FanImbalance_model_{1,2,3}_t  PIRDetection_model_1_t\n"
            "Regression:     REGR_{1k,2k,3k,4k,10k,13k}\n"
            "                REGR_{500,2k,6k,8k,20k}_NPU\n"
            "Anomaly:        AD_{1k,4k,16k,17k}  AD_Linear\n"
            "                AD_{500,2k,6k,8k,10k,20k}_NPU\n"
            "Forecasting:    FCST_{3k,13k}  FCST_LSTM{8,10}\n"
            "                FCST_{500,1k,2k,4k,6k,8k,10k,20k}_NPU\n"
            "Vision:         Lenet5"
        ),
    )
    parser.add_argument(
        "-c", "--config",
        metavar="YAML_FILE",
        default=None,
        help=(
            "Optional base YAML config file.\n"
            "CLI arguments override values from this file.\n"
            "When provided, all other flags become optional."
        ),
    )
    parser.add_argument(
        "--run-name",
        dest="run_name",
        metavar="NAME",
        default=None,
        help=(
            "Name for the output run folder.\n"
            "Supports {date-time} and {model_name} placeholders.\n"
            "Default: '{date-time}/{model_name}'"
        ),
    )
    # Note: --output removed; projects_path is now derived from the -i/--project path.
    # The project directory IS the project_path, and its parent is projects_path.


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    detected = _detect_training_device()
    group = parser.add_argument_group("training options")
    group.add_argument(
        "-i", "--project",
        default=os.path.join("data", "projects", "default"),
        metavar="PROJECT_DIR",
        help=(
            "Path to project directory (must contain dataset/).\n"
            "Default: ./data/projects/default"
        ),
    )
    group.add_argument(
        "--feature-extraction",
        dest="feature_extraction",
        metavar="PRESET",
        default=None,
        help=(
            "Feature extraction preset name. Default: 'default'.\n"
            "Examples:\n"
            "  Generic_1024Input_FFTBIN_64Feature_8Frame\n"
            "  Generic_512Input_FFTBIN_32Feature_8Frame\n"
            "  Generic_512Input_RAW_512Feature_1Frame\n"
            "  Generic_256Input_RAW_256Feature_1Frame"
        ),
    )
    group.add_argument(
        "--epochs",
        type=int,
        default=None,
        metavar="N",
        help="Number of training epochs.",
    )
    group.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=None,
        metavar="N",
        help="Training batch size.",
    )
    group.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="RATE",
        help="Learning rate.",
    )
    group.add_argument(
        "--training-device",
        dest="training_device",
        default=detected,
        choices=TRAINING_DEVICES,
        metavar="BACKEND",
        help=(
            "Training backend to use.\n"
            "  auto  Let tinyml_modelmaker detect\n"
            "  mps   Apple Metal (macOS)\n"
            "  cuda  NVIDIA GPU\n"
            "  cpu   CPU only\n"
            f"Default (detected): {detected}"
        ),
    )
    group.add_argument(
        "--gpus",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of GPUs / accelerators to use.\n"
            "  0  CPU only\n"
            "  1  Single GPU or Apple Metal (MPS) — default\n"
            " >1  Multi-GPU distributed training (CUDA only)\n"
            "When set to 1 on macOS, Metal (MPS) is used automatically\n"
            "if available via --training-device auto."
        ),
    )
    group.add_argument(
        "--quantization",
        choices=QUANTIZATION_OPTIONS,
        default=None,
        metavar="METHOD",
        help=(
            "Quantization method.\n"
            "  NO_QUANTIZATION    Float model (default)\n"
            "  QUANTIZATION_TINPU Quantize for TI NPU target"
        ),
    )

    # Performance optimization flags (advanced)
    perf = parser.add_argument_group("performance options (advanced)")
    perf.add_argument(
        "--compile-model",
        dest="compile_model",
        type=int,
        default=None,
        choices=[0, 1],
        metavar="0|1",
        help=(
            "Enable torch.compile for optimized training.\n"
            "  0  Disabled (default)\n"
            "  1  Enable (CUDA: inductor backend, MPS: aot_eager)\n"
            "Best results on CUDA with inductor. May add overhead on MPS."
        ),
    )
    amp_group = perf.add_mutually_exclusive_group()
    amp_group.add_argument(
        "--native-amp",
        dest="native_amp",
        action="store_true",
        default=None,
        help=(
            "Enable PyTorch native mixed precision (torch.amp.autocast).\n"
            "Reduces memory and may improve throughput.\n"
            "Auto-enabled on MPS (Apple Silicon)."
        ),
    )
    amp_group.add_argument(
        "--no-native-amp",
        dest="native_amp",
        action="store_false",
        help="Disable native mixed precision (overrides MPS auto-enable).",
    )

    # Neural Architecture Search (NAS)
    nas = parser.add_argument_group(
        "neural architecture search (NAS)",
        description=(
            "Automatically discover an optimal model architecture instead of\n"
            "using a predefined model from the catalog. When --nas is used,\n"
            "--model/-n is optional (a synthetic name is generated).\n"
            "NAS is supported for classification tasks only."
        ),
    )
    nas.add_argument(
        "--nas",
        dest="nas_size",
        choices=NAS_SIZES,
        default=None,
        metavar="SIZE",
        help=(
            "Enable NAS with a model size preset.\n"
            "  s   Small  — fast search, compact model\n"
            "  m   Medium — balanced search\n"
            "  l   Large  — deeper search, larger model\n"
            "  xl  XL     — extensive search, largest model\n"
            "When set, --model/-n becomes optional."
        ),
    )
    nas.add_argument(
        "--nas-epochs",
        dest="nas_epochs",
        type=int,
        default=None,
        metavar="N",
        help="Number of NAS search epochs (default: 10).",
    )
    nas.add_argument(
        "--nas-optimize",
        dest="nas_optimize",
        choices=NAS_OPTIMIZE_MODES,
        default=None,
        metavar="MODE",
        help=(
            "Resource optimization target for NAS.\n"
            "  Memory   Prefer fewer parameters (default)\n"
            "  Compute  Prefer fewer MACs (lower latency)"
        ),
    )


def _add_compilation_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("compilation options")
    group.add_argument(
        "-o", "--onnx",
        default=None,
        metavar="ONNX_FILE",
        help=(
            "Path to an existing ONNX model file to compile.\n"
            "Required for 'compile' subcommand unless provided via --config.\n"
            "Not needed for 'run' (uses the model produced by training)."
        ),
    )
    group.add_argument(
        "--preset",
        metavar="PRESET_NAME",
        default=None,
        help="Compilation preset name. Default: 'default_preset'.",
    )


# ---------------------------------------------------------------------------
# Subcommand parsers
# ---------------------------------------------------------------------------

def _add_train_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "train",
        help="Train a model and export ONNX. Compilation is skipped.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Train a model using the tinyml-modelmaker pipeline.\n"
            "Outputs model.onnx in the project run directory.\n"
            "On macOS, Apple Metal (MPS) is used automatically when available.\n\n"
            "Example:\n"
            "  mmcli train -m timeseries -t generic_timeseries_classification \\\n"
            "              -d F28P55 -n TimeSeries_Generic_1k_t -i ./my_project\n\n"
            "  # Use default project (./data/projects/default):\n"
            "  mmcli train -m timeseries -t generic_timeseries_classification \\\n"
            "              -d F28P55 -n TimeSeries_Generic_1k_t\n\n"
            "  # Force CPU:\n"
            "  mmcli train ... --training-device cpu\n\n"
            "  # Explicit Metal:\n"
            "  mmcli train ... --training-device mps"
        ),
    )
    _add_common_args(p)
    _add_training_args(p)


def _add_compile_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "compile",
        help="Compile an existing ONNX file. Training is skipped.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Compile a pre-trained ONNX model for a target microcontroller.\n"
            "No training data or training step is required.\n\n"
            "Note: compilation requires ti_mcu_nnc (Linux/Windows only).\n"
            "On macOS: train here, then compile on Linux.\n\n"
            "Example:\n"
            "  mmcli compile -m timeseries -t generic_timeseries_classification \\\n"
            "                -d F28P55 -n TimeSeries_Generic_1k_t \\\n"
            "                -o ./data/projects/my_run/model.onnx"
        ),
    )
    _add_common_args(p)
    _add_compilation_args(p)


def _add_run_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "run",
        help="Full pipeline: train then compile.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Run the full tinyml-modelmaker pipeline:\n"
            "  1. Train the model (using Metal/MPS on macOS if available)\n"
            "  2. Export model.onnx\n"
            "  3. Compile for the target device\n\n"
            "Note: compilation requires ti_mcu_nnc (Linux/Windows only).\n\n"
            "Example:\n"
            "  mmcli run -m timeseries -t generic_timeseries_classification \\\n"
            "            -d F28P55 -n TimeSeries_Generic_1k_t -i ./my_project"
        ),
    )
    _add_common_args(p)
    _add_training_args(p)
    _add_compilation_args(p)


def _add_help_parser(subparsers) -> None:
    subparsers.add_parser(
        "help",
        help="Show detailed help for all subcommands and options.",
        description="Print full help for every subcommand.",
    )


# ---------------------------------------------------------------------------
# Full help printer
# ---------------------------------------------------------------------------

def _print_full_help(main_parser: argparse.ArgumentParser) -> None:
    """Print help for the top-level parser and every subcommand."""
    detected = _detect_training_device()
    print(f"mmcli {__version__}  —  tinyml-modelmaker command-line interface")
    print(f"Detected training backend on this machine: {detected}\n")

    main_parser.print_help()

    # Print each subcommand's help
    for action in main_parser._subparsers._group_actions:  # type: ignore[attr-defined]
        for name, subparser in action.choices.items():
            if name == "help":
                continue
            print(f"\n{'─' * 70}")
            print(f"  mmcli {name}")
            print(f"{'─' * 70}\n")
            subparser.print_help()


# ---------------------------------------------------------------------------
# Post-parse validation
# ---------------------------------------------------------------------------

def _validate_args(args: argparse.Namespace) -> None:
    """
    Validate args after parsing. Exits with a clear error message on failure.

    Required fields (module/task/device/model/data/onnx) may come from
    --config rather than the CLI, so we defer their check to here instead
    of marking them required=True in argparse.
    """
    errors = []
    command = args.command

    # --config path must exist
    if getattr(args, "config", None) and not os.path.isfile(args.config):
        errors.append(f"--config file not found: {args.config}")

    # NAS mode: --model/-n is optional when --nas is set
    using_nas = bool(getattr(args, "nas_size", None))

    # For fields that can come from --config, only enforce when --config not given
    using_config = bool(getattr(args, "config", None))
    if not using_config:
        for flag, attr in [("--module/-m", "module"), ("--task/-t", "task"),
                           ("--device/-d", "device")]:
            if not getattr(args, attr, None):
                errors.append(f"{flag} is required when --config is not provided")
        # --model/-n is required unless NAS is enabled
        if not getattr(args, "model", None) and not using_nas:
            errors.append(
                "--model/-n is required when --config and --nas are not provided")
        if command == "compile" and not getattr(args, "onnx", None):
            errors.append("--onnx/-o is required when --config is not provided")

    # NAS is only supported for classification tasks
    if using_nas:
        task = getattr(args, "task", None)
        if task and task not in NAS_SUPPORTED_TASKS:
            errors.append(
                f"--nas is only supported for classification tasks, "
                f"not '{task}'.\n"
                f"  Supported: {', '.join(NAS_SUPPORTED_TASKS)}"
            )

    # Validate project directory structure for train/run
    project = getattr(args, "project", None)
    if project and command in ("train", "run"):
        project = os.path.abspath(project)
        args.project = project  # normalize to absolute
        dataset_dir = os.path.join(project, "dataset")
        annotations_dir = os.path.join(dataset_dir, "annotations")
        # Data can live in classes/ (classification/anomaly), files/
        # (regression/forecasting), or images/ (vision)
        data_subdirs = ["classes", "files", "images"]
        if not os.path.isdir(project):
            errors.append(f"Project directory not found: {project}")
        elif not os.path.isdir(dataset_dir):
            errors.append(
                f"Project directory missing 'dataset/' subdirectory: {project}")
        else:
            # Validate dataset contents
            if not os.path.isdir(annotations_dir):
                errors.append(
                    f"Dataset missing 'annotations/' subdirectory: {dataset_dir}")
            if not any(os.path.isdir(os.path.join(dataset_dir, d))
                       for d in data_subdirs):
                errors.append(
                    f"Dataset missing data subdirectory (one of "
                    f"{', '.join(data_subdirs)}): {dataset_dir}")

    # Path existence checks
    if getattr(args, "onnx", None) and not os.path.isfile(args.onnx):
        errors.append(f"--onnx file not found: {args.onnx}")

    if errors:
        for msg in errors:
            print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(2)


def _validate_config(config: dict) -> None:
    """Check that required config fields are non-null before dispatching."""
    required = [
        ("common", "target_module"),
        ("common", "task_type"),
        ("common", "target_device"),
        ("training", "model_name"),
    ]
    missing = [
        f"{section}.{key}"
        for section, key in required
        if not config.get(section, {}).get(key)
    ]
    if config["training"]["enable"] and not config.get("dataset", {}).get("input_data_path"):
        missing.append("dataset.input_data_path")
    if config["compilation"]["enable"] and not config.get("compilation", {}).get("model_path"):
        if not config["training"]["enable"]:
            missing.append("compilation.model_path")

    if missing:
        print(
            "ERROR: The following required config fields are missing or null:\n  "
            + "\n  ".join(missing)
            + "\nProvide them via CLI flags or a --config YAML file.",
            file=sys.stderr,
        )
        sys.exit(2)


# ---------------------------------------------------------------------------
# Dispatch to tinyml_modelmaker via subprocess
# ---------------------------------------------------------------------------

def _dispatch(config: dict, python_exe: str, verbose: bool) -> int:
    """Write config to a temp YAML and invoke run_tinyml_modelmaker.py."""
    try:
        runner_script = _find_runner_script(python_exe)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    yaml_path = None
    try:
        yaml_path = write_temp_yaml(config)
        logger.debug("Config YAML: %s", yaml_path)
        logger.debug("Runner script: %s", runner_script)
        logger.debug("Python interpreter: %s", python_exe)

        cmd = [python_exe, runner_script, yaml_path]
        if verbose:
            print(f"Running: {' '.join(cmd)}", flush=True)

        result = subprocess.run(cmd, check=False)
        return result.returncode
    finally:
        if yaml_path:
            try:
                os.unlink(yaml_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(name)s - %(message)s",
    )

    detected = _detect_training_device()

    parser = argparse.ArgumentParser(
        prog="mmcli",
        description=(
            "mmcli — command-line interface for tinyml-modelmaker\n\n"
            "Subcommands:\n"
            "  train    Train a model and export ONNX (uses Metal/MPS on macOS)\n"
            "  compile  Compile an existing ONNX file (Linux/Windows only)\n"
            "  run      Full pipeline: train then compile\n"
            "  help     Show detailed help for all subcommands\n\n"
            f"Detected training backend: {detected}\n\n"
            "Environment variables:\n"
            "  MMCLI_PYTHON      Python interpreter with tinyml_modelmaker installed\n"
            "                    Default: 'python' on PATH\n"
            "  MMCLI_MODELMAKER  Path to the tinyml-modelmaker source directory\n"
            "                    (auto-detected if MMCLI_PYTHON is set correctly)\n\n"
            "Run 'mmcli help' to see all subcommands and options at once.\n"
            "Run 'mmcli <subcommand> --help' for subcommand-specific help."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"mmcli {__version__}")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable debug logging and show the subprocess command.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Build the config and print the generated YAML, but do not run anything.",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="SUBCOMMAND")
    subparsers.required = True

    _add_train_parser(subparsers)
    _add_compile_parser(subparsers)
    _add_run_parser(subparsers)
    _add_help_parser(subparsers)

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle help subcommand before any other processing
    if args.command == "help":
        _print_full_help(parser)
        sys.exit(0)

    _validate_args(args)
    config = build_config(args)
    _validate_config(config)

    if args.dry_run:
        import yaml as _yaml
        print("# Generated config (dry run — not executed)")
        print(_yaml.dump(config, default_flow_style=False, sort_keys=False))
        sys.exit(0)

    python_exe = _get_python_exe()
    rc = _dispatch(config, python_exe, verbose=args.verbose)
    sys.exit(rc)
