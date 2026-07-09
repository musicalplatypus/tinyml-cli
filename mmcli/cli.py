"""
mmcli — command-line interface for tinyml-modelmaker.

Subcommands:
  init       Create a new project from an example dataset
  train      Train a model and export ONNX (no compilation)
  compile    Compile an existing ONNX file (no training)
  run        Full pipeline: train then compile
  info       Show supported devices, models, and feature extraction presets
  analyze    Analyse a project dataset (size, classes, sequence length)
  recommend  Recommend a model and FE preset from tinyml-modelzoo examples
  deploy     Deploy compiled artifacts to target hardware (check-sdk/artifacts/create/build/flash)
  help       Show detailed help for all subcommands

Runtime requirements (not bundled in the binary):
  MMCLI_PYTHON  Path to the Python interpreter that has tinyml_modelmaker
                installed, e.g. /path/to/venv/bin/python
                Defaults to 'python' or 'python3' found on PATH.

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

# Batch utilities
from mmcli.batch import expand_project_paths, run_batch, format_batch_results

# ---------------------------------------------------------------------------
# Security helpers (restored from Phase 1; removed in Phase 5 eb0a1bd)
# ---------------------------------------------------------------------------

import tempfile
from pathlib import Path


def _is_safe_path(path: str, base_dir=None) -> bool:
    """Return True if *path* resolves under *base_dir* (default: cwd) or the OS temp dir.

    Uses pathlib.resolve() — immune to iterative-replace bypasses and encoded traversal.
    Shell metacharacters in paths are irrelevant because all subprocess calls use shell=False.
    """
    if not path or not path.strip():
        return False
    try:
        resolved = Path(path).resolve()
        anchor = Path(base_dir).resolve() if base_dir else Path.cwd()
        if resolved.is_relative_to(anchor):
            return True
        return resolved.is_relative_to(Path(tempfile.gettempdir()).resolve())
    except (ValueError, OSError):
        return False


def _sanitize_input(input_str: str, max_length: int = 1024) -> str:
    """Enforce a length cap. Raises ValueError if exceeded. Does NOT strip chars.

    Shell injection is prevented by shell=False in subprocess calls, not char-stripping.
    """
    if not isinstance(input_str, str):
        input_str = str(input_str)
    if len(input_str) > max_length:
        raise ValueError(f"Input exceeds maximum length of {max_length}")
    return input_str


# ---------------------------------------------------------------------------
# Known enumerations (for --help text)
# ---------------------------------------------------------------------------

MODULES = ["timeseries", "vision", "audio"]

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

TASK_TYPES_AUDIO = ["audio_classification"]

TARGET_DEVICES = [
    # C2000
    "F280013", "F280015", "F28003", "F28004", "F2837", "F28E12",
    "F28P55", "F28P65", "F29H85", "F29P58", "F29P32",
    # MSPM0
    "MSPM0G3507", "MSPM0G3519", "MSPM0G5187",
    "MSPM33C32", "MSPM33C34",
    # SimpleLink
    "CC1312", "CC1314", "CC1352", "CC1354", "CC2755", "CC35X1",
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
    env = os.environ.get("MMCLI_PYTHON")
    if env:
        return env
    # Try common Python executable names (macOS/Linux often lack 'python')
    import shutil
    for name in ("python", "python3"):
        if shutil.which(name):
            return name
    return "python"  # fallback, will produce a clear error


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
            "  audio       Audio / speech classification\n"
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
            "  image_classification\n"
            "Audio tasks:\n"
            "  audio_classification"
        ),
    )
    parser.add_argument(
        "-d", "--device",
        default=None,
        metavar="DEVICE",
        help=(
            "Target microcontroller device (required unless provided via --config).\n"
            "C2000:     F280013  F280015  F28003  F28004  F2837\n"
            "           F28E12  F28P55  F28P65  F29H85  F29P58  F29P32\n"
            "MSPM0:     MSPM0G3507  MSPM0G3519  MSPM0G5187\n"
            "           MSPM33C32  MSPM33C34\n"
            "SimpleLink: CC1312  CC1314  CC1352  CC1354  CC2755  CC35X1\n"
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

    # Report generation
    parser.add_argument(
        "--report",
        action="store_true",
        default=False,
        help=(
            "Generate a live-updating HTML training report with\n"
            "accuracy/loss charts and confusion matrix.\n"
            "Output: <project_dir>/run/report.html"
        ),
    )


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
    aq = group.add_mutually_exclusive_group()
    aq.add_argument(
        "--auto-quantization",
        dest="auto_quantization",
        action="store_true",
        default=None,
        help=(
            "Enable auto-quantization: binary-search for the lowest bitwidth\n"
            "that keeps accuracy within the tolerance threshold (default: on)."
        ),
    )
    aq.add_argument(
        "--no-auto-quantization",
        dest="auto_quantization",
        action="store_false",
        help="Disable auto-quantization and use fixed 8-bit quantization.",
    )
    group.add_argument(
        "--autoquant-tolerance-classification",
        dest="autoquant_tolerance_classification",
        type=float,
        default=None,
        metavar="FRAC",
        help="Max accuracy drop tolerated by auto-quantization (default: 0.05 = 5%%).",
    )
    group.add_argument(
        "--autoquant-tolerance-regression",
        dest="autoquant_tolerance_regression",
        type=float,
        default=None,
        metavar="FRAC",
        help="Max R² drop tolerated by auto-quantization (default: 0.05 = 5%%).",
    )
    group.add_argument(
        "--autoquant-tolerance-forecasting",
        dest="autoquant_tolerance_forecasting",
        type=float,
        default=None,
        metavar="MULT",
        help="Max SMAPE increase tolerated (default: 2.0 = 3× float baseline).",
    )
    group.add_argument(
        "--autoquant-tolerance-anomaly",
        dest="autoquant_tolerance_anomaly",
        type=float,
        default=None,
        metavar="MULT",
        help="Max MSE increase tolerated for anomaly detection (default: 2.0).",
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
            "When set, --model/-n becomes optional.\n"
            "For timeseries tasks you must also set --feature-extraction."
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
        help=(
            "Compilation preset name. Options:\n"
            "  default_preset           Best available path (NPU if device supports it)\n"
            "  forced_soft_npu_preset   Force CPU path (recommended for anomaly/forecasting)\n"
            "  compress_npu_layer_data  NPU path with compressed weights (tight SRAM)\n"
            "  auto                     Pick the best preset for this device and task\n"
            "Default: 'default_preset'."
        ),
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
            "              -d F28P55 -n CLS_1k_NPU -i ./my_project\n\n"
            "  # Use default project (./data/projects/default):\n"
            "  mmcli train -m timeseries -t generic_timeseries_classification \\\n"
            "              -d F28P55 -n CLS_1k_NPU\n\n"
            "  # Force CPU:\n"
            "  mmcli train ... --training-device cpu\n\n"
            "  # Explicit Metal:\n"
            "  mmcli train ... --training-device mps\n\n"
            "  # Show progress bar:\n"
            "  mmcli train -m timeseries -t classification -d F28P55 --progress"
        ),
    )
    _add_common_args(p)
    _add_training_args(p)
    p.add_argument(
        "--progress",
        action="store_true",
        help="Display progress bar during training",
    )
    # Batch mode flags
    batch_group = p.add_mutually_exclusive_group()
    batch_group.add_argument(
        "--batch", "-b",
        nargs="+",
        metavar="PROJECT",
        help="Process multiple projects (supports glob patterns)"
    )
    batch_group.add_argument(
        "--directory", "-D",
        type=str,
        dest="directory",
        metavar="DIR",
        help="Process all .yaml files in directory"
    )


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
            "                -d F28P55 -n CLS_1k_NPU \\\n"
            "                -o ./data/projects/my_run/model.onnx\n\n"
            "  # Show progress bar:\n"
            "  mmcli compile -m timeseries -t classification -d F28P55 --progress"
        ),
    )
    _add_common_args(p)
    _add_compilation_args(p)
    p.add_argument(
        "--progress",
        action="store_true",
        help="Display progress bar during compilation",
    )


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
            "            -d F28P55 -n CLS_1k_NPU -i ./my_project\n\n"
            "  # Show progress bar for both training and compilation:\n"
            "  mmcli run -m timeseries -t classification -d F28P55 --progress"
        ),
    )
    _add_common_args(p)
    _add_training_args(p)
    p.add_argument(
        "--progress",
        action="store_true",
        help="Display progress bar during training and compilation",
    )
    # Batch mode flags
    batch_group = p.add_mutually_exclusive_group()
    batch_group.add_argument(
        "--batch", "-b",
        nargs="+",
        metavar="PROJECT",
        help="Process multiple projects (supports glob patterns)"
    )
    batch_group.add_argument(
        "--directory", "-D",
        type=str,
        dest="directory",
        metavar="DIR",
        help="Process all .yaml files in directory"
    )
    _add_compilation_args(p)


def _add_info_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "info",
        help="Show supported devices, models, and feature extraction presets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Query the model registry and display available options.\n\n"
            "Examples:\n"
            "  mmcli info -m timeseries                        # list task types\n"
            "  mmcli info -m timeseries -t arc_fault           # details for arc_fault\n"
            "  mmcli info -m timeseries -t arc_fault -d F28P55 # models for F28P55\n"
            "  mmcli info -m timeseries --format json         # JSON output\n"
        ),
    )
    p.add_argument(
        "-m", "--module",
        required=True,
        choices=MODULES,
        metavar="MODULE",
        help="AI module (timeseries, vision, or audio).",
    )
    p.add_argument(
        "-t", "--task",
        default=None,
        metavar="TASK_TYPE",
        help="Task type to show details for. Omit to list all task types.",
    )
    p.add_argument(
        "-d", "--device",
        default=None,
        metavar="DEVICE",
        help="Target device to filter models.",
    )
    p.add_argument(
        "--format",
        choices=["text", "json", "csv", "yaml"],
        default="text",
        help="Output format (default: text)",
    )
    p.add_argument(
        "-o", "--output",
        type=str,
        metavar="FILE",
        help="Write output to file instead of stdout",
    )


def _add_init_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "init",
        help="Create a new project from an example dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Initialise a new project directory from a bundled example\n"
            "dataset.  The dataset zip is extracted into the project so\n"
            "it is ready for 'mmcli train'.\n\n"
            "Example:\n"
            "  mmcli init -t arc_fault --dataset arc_fault_sample \\\n"
            "             -p ./my_arc_project\n\n"
            "List available datasets:\n"
            "  mmcli init --list\n"
            "  mmcli init --list -m timeseries\n\n"
            "Set MMCLI_DATASETS to override the built-in datasets directory."
        ),
    )
    p.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available example datasets and exit.",
    )
    p.add_argument(
        "-t", "--task",
        required=False,
        default=None,
        metavar="TASK_TYPE",
        help=(
            "Task type for the new project (required for extraction).\n"
            "Also used as a filter with --list."
        ),
    )
    p.add_argument(
        "-p", "--project",
        required=False,
        default=None,
        metavar="PROJECT_DIR",
        help="Path for the new project directory (must not already exist).",
    )
    p.add_argument(
        "--dataset",
        required=False,
        default=None,
        metavar="DATASET_NAME",
        help=(
            "Name of the example dataset to use.\n"
            "Use 'mmcli init --list' to see available datasets."
        ),
    )
    p.add_argument(
        "-m", "--module",
        default=None,
        choices=MODULES,
        metavar="MODULE",
        help="AI module (auto-detected from dataset if omitted).\n"
             "Also used as a filter with --list.",
    )


def _add_analyze_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "analyze",
        help="Analyse a project dataset (size, classes, sequence length).",
        description=(
            "Inspect a project's dataset/ directory and report:\n"
            "  - Total sample count and per-class distribution\n"
            "  - Minimum sequence length\n"
            "  - Dataset size bucket (tiny/small/medium/large)\n\n"
            "The 'size bucket' output can be passed directly to 'mmcli recommend'\n"
            "via --dataset-size-bucket to improve model selection accuracy.\n\n"
            "Examples:\n"
            "  mmcli analyze -i ./my-project --format json\n"
            "  mmcli analyze -i ./my-project -o results.csv\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "-i", "--project",
        required=True,
        metavar="DIR",
        help=(
            "Project directory containing dataset/. "
            "May also point directly at the dataset folder."
        ),
    )
    p.add_argument(
        "--format",
        choices=["text", "json", "csv", "yaml"],
        default="text",
        help="Output format (default: text)",
    )
    p.add_argument(
        "-o", "--output",
        type=str,
        metavar="FILE",
        help="Write output to file instead of stdout",
    )
    # Batch mode flags
    batch_group = p.add_mutually_exclusive_group()
    batch_group.add_argument(
        "--batch", "-b",
        nargs="+",
        metavar="PROJECT",
        help="Process multiple projects (supports glob patterns)"
    )
    batch_group.add_argument(
        "--directory", "-d",
        type=str,
        dest="directory",
        metavar="DIR",
        help="Process all .yaml files in directory"
    )


def _add_recommend_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "recommend",
        help="Recommend a model and FE preset from tinyml-modelzoo examples.",
        description=(
            "Score every tinyml-modelzoo example against your task, device,\n"
            "and (optionally) variable count and dataset size, then print a\n"
            "ranked shortlist of the best-matching models and feature extraction\n"
            "presets.\n\n"
            "Modelzoo path resolution (in order):\n"
            "  1. --modelzoo-path argument\n"
            "  2. MMCLI_MODELZOO_PATH env var (set to repo root)\n"
            "  3. Auto-detected sibling of the installed tinyml_modelmaker package\n"
            "  4. Common default paths (~/tinyml-tensorlab/tinyml-modelzoo, etc.)\n\n"
            "Examples:\n"
            "  mmcli recommend -t arc_fault -d F28P55 --format yaml\n"
            "  mmcli recommend -t classification -d F28P55 -o recommendations.json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    req = p.add_argument_group("required")
    req.add_argument(
        "-t", "--task",
        required=True,
        metavar="TASK_TYPE",
        help="Task type (e.g. arc_fault, generic_timeseries_classification).",
    )
    req.add_argument(
        "-d", "--device",
        required=True,
        metavar="DEVICE",
        choices=TARGET_DEVICES,
        help="Target MCU device (see 'mmcli info' for the full list).",
    )
    opt = p.add_argument_group("optional")
    opt.add_argument(
        "-m", "--module",
        choices=MODULES,
        default=None,
        metavar="MODULE",
        help=(
            "Target module (timeseries/vision/audio). "
            "Auto-inferred from --task when omitted."
        ),
    )
    opt.add_argument(
        "--variables",
        type=int,
        default=None,
        metavar="N",
        help="Number of sensor channels / input variables. Boosts score by +2 for exact matches.",
    )
    opt.add_argument(
        "--dataset-size-bucket",
        dest="dataset_size_bucket",
        choices=["tiny", "small", "medium", "large"],
        default=None,
        metavar="BUCKET",
        help=(
            "Dataset size bucket from 'mmcli analyze'.\n"
            "  tiny   < 500 samples   → prefer models ≤ 2k params\n"
            "  small  500–4,999       → prefer models ≤ 10k params\n"
            "  medium 5k–49,999       → prefer models ≤ 30k params\n"
            "  large  ≥ 50,000        → no restriction"
        ),
    )
    opt.add_argument(
        "--top",
        type=int,
        default=5,
        metavar="N",
        help="Number of ranked matches to display (default: 5).",
    )
    opt.add_argument(
        "--modelzoo-path",
        dest="modelzoo_path",
        default=None,
        metavar="DIR",
        help="Path to the tinyml-modelzoo repo root (or its examples/ subdirectory).",
    )
    p.add_argument(
        "--format",
        choices=["text", "json", "csv", "yaml"],
        default="text",
        help="Output format (default: text)",
    )
    p.add_argument(
        "-o", "--output",
        type=str,
        metavar="FILE",
        help="Write output to file instead of stdout",
    )


def _add_deploy_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "deploy",
        help="Deploy compiled model to target hardware.",
        description=(
            "Device deployment workflow — 5 subcommands:\n\n"
            "  check-sdk   Verify SDK installation for a target device\n"
            "  artifacts   Locate compiled ModelMaker outputs\n"
            "  create      Create CCS project from SDK template + artifacts\n"
            "  build       Headless CCS build (requires CCS 12+)\n"
            "  flash       Flash compiled .out to device via dslite"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="deploy_subcommand", metavar="DEPLOY_SUBCOMMAND")
    sub.required = True

    # ── check-sdk ──────────────────────────────────────────────────────────
    sdk_p = sub.add_parser(
        "check-sdk",
        help="Verify that the device-specific SDK is installed.",
        description="Locate and validate the SDK required for a target device.",
    )
    sdk_p.add_argument("-d", "--device", required=True, choices=TARGET_DEVICES,
                       help="Target MCU device (e.g. F28P55).")
    sdk_p.add_argument("--sdk-path", dest="sdk_path", default=None, metavar="PATH",
                       help="Explicit path to SDK root (skips auto-detection).")

    # ── artifacts ─────────────────────────────────────────────────────────
    art_p = sub.add_parser(
        "artifacts",
        help="Locate compiled ModelMaker outputs for a training run.",
        description=(
            "Validate that all 4 files needed for CCS project creation exist:\n"
            "  mod.a, tvmgen_default.h, test_vector.c, user_input_config.h"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    art_p.add_argument("-t", "--task", required=True, metavar="TASK_TYPE",
                       help="Task type string (e.g. motor_fault).")
    art_p.add_argument("--run-id", dest="run_id", required=True, metavar="RUN_ID",
                       help="Timestamp run directory name (e.g. 20240115_143022).")
    art_p.add_argument("--model-id", dest="model_id", required=True, metavar="MODEL_ID",
                       help="Model name from training config (e.g. CLS_1k_NPU).")
    art_p.add_argument("--no-quantization", dest="no_quantization", action="store_true",
                       default=False, help="Use base (float) golden vectors instead of quantized.")
    art_p.add_argument("--tinyml-base", dest="tinyml_base", default=None, metavar="PATH",
                       help="tinyml-tensorlab root (default: ~/tinyml-tensorlab).")

    # ── create ─────────────────────────────────────────────────────────────
    crt_p = sub.add_parser(
        "create",
        help="Create a CCS project from SDK template with compiled artifacts.",
    )
    crt_p.add_argument("-d", "--device", required=True, choices=TARGET_DEVICES,
                       help="Target MCU device.")
    crt_p.add_argument("-t", "--task", required=True, metavar="TASK_TYPE",
                       help="Task type string (e.g. motor_fault).")
    crt_p.add_argument("--run-id", dest="run_id", required=True, metavar="RUN_ID",
                       help="Training run timestamp directory name.")
    crt_p.add_argument("--model-id", dest="model_id", required=True, metavar="MODEL_ID",
                       help="Model name from training artifacts.")
    crt_p.add_argument("--project-name", dest="project_name", required=True, metavar="NAME",
                       help="Name for the new CCS project folder.")
    crt_p.add_argument("--device-type", dest="device_type", default=None, metavar="CCS_TYPE",
                       help=(
                           "CCS device_type string (e.g. f28p55x). "
                           "Auto-resolved from --device when omitted."
                       ))
    crt_p.add_argument("--no-quantization", dest="no_quantization", action="store_true",
                       default=False, help="Use base (float) golden vectors.")
    crt_p.add_argument("--tinyml-base", dest="tinyml_base", default=None, metavar="PATH",
                       help="tinyml-tensorlab root (default: ~/tinyml-tensorlab).")
    crt_p.add_argument("--sdk-path", dest="sdk_path", default=None, metavar="PATH",
                       help="Explicit SDK root path (overrides auto-detection).")
    crt_p.add_argument("--ccs-templates-path", dest="ccs_templates_path", default=None,
                       metavar="PATH",
                       help="Explicit AI examples directory (overrides sdk_path + auto-detect).")

    # ── build ──────────────────────────────────────────────────────────────
    bld_p = sub.add_parser(
        "build",
        help="Headless CCS build of a CCS project (requires CCS 12+).",
    )
    bld_p.add_argument("--project-path", dest="project_path", required=True, metavar="PATH",
                       help="Absolute path to the CCS project folder.")
    bld_p.add_argument("--ccs-path", dest="ccs_path", required=True, metavar="PATH",
                       help="CCS installation root (e.g. /opt/ti/ccs1260).")
    bld_p.add_argument("--workspace", default=None, metavar="PATH",
                       help="Eclipse workspace directory (default: /tmp/ccs_ws_<pid>).")
    bld_p.add_argument("--build-type", dest="build_type", choices=["full", "incremental"],
                       default="full", help="Build type (default: full).")

    # ── flash ──────────────────────────────────────────────────────────────
    fls_p = sub.add_parser(
        "flash",
        help="Flash compiled .out to target device via dslite.",
        description="Connect the device via USB/JTAG before running this.",
    )
    fls_p.add_argument("--project-path", dest="project_path", required=True, metavar="PATH",
                       help="Absolute path to the CCS project folder.")
    fls_p.add_argument("--ccs-path", dest="ccs_path", required=True, metavar="PATH",
                       help="CCS installation root.")
    fls_p.add_argument("--project-name", dest="project_name", default=None, metavar="NAME",
                       help="Project binary name (defaults to folder name).")
    fls_p.add_argument("--ccxml", default=None, metavar="PATH",
                       help="Path to the device .ccxml target config (auto-searched if omitted).")
    fls_p.add_argument("--out-file", dest="out_file", default=None, metavar="PATH",
                       help="Explicit path to the .out binary (auto-searched if omitted).")


def _add_about_parser(subparsers) -> None:
    subparsers.add_parser(
        "about",
        help="Show copyright, credits, and version information.",
        description="Display copyright, version, and ASCII art.",
    )


def _add_shell_parser(subparsers) -> None:
    """Add the ``shell`` subcommand which launches an interactive REPL.

    The command takes no additional arguments – it simply starts the mmcli
    interactive shell implemented in :mod:`mmcli.interactive`.
    """
    subparsers.add_parser(
        "shell",
        help="Enter interactive REPL-like shell mode.",
        description=(
            "Launch an interactive mmcli shell where commands such as 'info',\n"
            "'recommend', 'analyze', and others can be executed in a REPL.\n"
            "Use 'exit' or Ctrl‑D to leave the shell."
        ),
    )


def _add_diagnose_parser(subparsers) -> None:
    """Add the ``diagnose`` subcommand for troubleshooting."""
    p = subparsers.add_parser(
        "diagnose",
        help="Run diagnostic checks to troubleshoot issues.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Run diagnostic checks to identify common issues.\n\n"
            "Examples:\n"
            "  mmcli diagnose              # Run basic diagnostics\n"
            "  mmcli diagnose --full       # Run extended diagnostics\n"
            "  mmcli diagnose --error 'Cannot import tinyml_modelmaker'  # Get fix for specific error\n"
            "  mmcli diagnose --full --format json   # JSON output for automation\n"
        ),
    )

    diag_group = p.add_mutually_exclusive_group()
    diag_group.add_argument(
        "--full", "-f",
        action="store_true",
        help="Run extended diagnostics (includes disk space, etc.)",
    )
    diag_group.add_argument(
        "--error", "-e",
        type=str,
        metavar="MESSAGE",
        help="Get fix suggestion for a specific error message",
    )


def _add_compare_parser(subparsers) -> None:
    """Add the ``compare`` subcommand for comparing models."""
    p = subparsers.add_parser(
        "compare",
        help="Compare multiple models side-by-side.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Compare model availability and attributes across task types.\n\n"
            "Examples:\n"
            "  mmcli compare -m timeseries \\\n"
            "      --model1 generic_timeseries_classification \\\n"
            "      --model2 generic_timeseries_regression\n"
            "  mmcli compare -m timeseries --all-models --device F28P55\n"
        ),
    )

    p.add_argument(
        "-m", "--module",
        required=True,
        choices=["timeseries", "vision", "audio"],
        help="AI module type.",
    )

    p.add_argument(
        "--model1",
        type=str,
        required=True,
        metavar="MODEL",
        help="First model to compare (task type).",
    )

    p.add_argument(
        "--model2",
        type=str,
        required=True,
        metavar="MODEL",
        help="Second model to compare (task type).",
    )

    p.add_argument(
        "-d", "--device",
        type=str,
        default=None,
        metavar="DEVICE",
        help="Target device to filter results.",
    )

    p.add_argument(
        "--all-models",
        action="store_true",
        help="Compare all available models for this module.",
    )

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

    # --- Input sanitization (length cap on free-text string flags) -----------
    for attr in ("module", "task", "device", "model"):
        val = getattr(args, attr, None)
        if val is not None:
            try:
                setattr(args, attr, _sanitize_input(str(val)))
            except ValueError as exc:
                errors.append(f"--{attr.replace('_', '-')}: {exc}")

    # --- Path traversal guard (relative paths only) --------------------------
    # Absolute paths are user-specified filesystem locations and are accepted;
    # only relative paths can traverse outside the working directory via '..'.
    for path_attr, flag in [
        ("config", "--config"),
        ("onnx", "--onnx"),
        ("project", "--project/-i"),
    ]:
        val = getattr(args, path_attr, None)
        if val and not os.path.isabs(val) and not _is_safe_path(val):
            errors.append(
                f"{flag} contains an unsafe path traversal sequence: {val!r}"
            )

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

def _dispatch(config: dict, python_exe: str, verbose: bool,
              report_path: str = None, nas_enabled: bool = False) -> int:
    """Write config to a temp YAML and invoke run_tinyml_modelmaker.py."""
    try:
        runner_script = _find_runner_script(python_exe)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Set up report handler if requested
    feed_line = finalize = None
    if report_path:
        from mmcli.report import create_report_handler
        feed_line, finalize = create_report_handler(report_path, nas_enabled=nas_enabled)
        logger.info("Report will be written to: %s", report_path)

    yaml_path = None
    try:
        yaml_path = write_temp_yaml(config)
        logger.debug("Config YAML: %s", yaml_path)
        logger.debug("Runner script: %s", runner_script)
        logger.debug("Python interpreter: %s", python_exe)

        cmd = [python_exe, runner_script, yaml_path]
        if verbose:
            print(f"Running: {' '.join(cmd)}", flush=True)

        if feed_line:
            # Capture stdout/stderr line-by-line for report generation
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                print(line, end='', flush=True)
                feed_line(line)
            proc.wait()
            finalize()
            return proc.returncode
        else:
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
            "  init       Create a new project from an example dataset\n"
            "  train      Train a model and export ONNX (uses Metal/MPS on macOS)\n"
            "  compile    Compile an existing ONNX file (Linux/Windows only)\n"
            "  run        Full pipeline: train then compile\n"
            "  info       Show supported devices, models, and presets\n"
            "  analyze    Analyse a project dataset (size, classes, sequence length)\n"
            "  recommend  Recommend a model and FE preset from modelzoo examples\n"
            "  about      Show copyright, credits, and version info\n"
            "  help       Show detailed help for all subcommands\n\n"
            f"Detected training backend: {detected}\n\n"
            "Environment variables:\n"
            "  MMCLI_PYTHON      Python interpreter with tinyml_modelmaker installed\n"
            "                    Default: 'python' or 'python3' on PATH\n"
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

    _add_init_parser(subparsers)
    _add_train_parser(subparsers)
    _add_compile_parser(subparsers)
    _add_run_parser(subparsers)
    _add_info_parser(subparsers)
    _add_analyze_parser(subparsers)
    _add_recommend_parser(subparsers)
    _add_compare_parser(subparsers)
    _add_diagnose_parser(subparsers)
    _add_deploy_parser(subparsers)
    _add_about_parser(subparsers)
    _add_shell_parser(subparsers)
    _add_help_parser(subparsers)

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle help, info, and about subcommands before any other processing
    if args.command == "help":
        _print_full_help(parser)
        sys.exit(0)

    if args.command == "about":
        from mmcli.about import run_about
        run_about()
        sys.exit(0)

    if args.command == "init":
        if args.list:
            from mmcli.datasets import print_dataset_list
            print_dataset_list(task_type=args.task, module=args.module)
            sys.exit(0)
        # Validate required args for extraction
        missing = []
        if not args.task:
            missing.append("--task / -t")
        if not args.project:
            missing.append("--project / -p")
        if not args.dataset:
            missing.append("--dataset")
        if missing:
            print(
                f"ERROR: The following arguments are required: "
                f"{', '.join(missing)}\n"
                f"Use 'mmcli init --list' to see available datasets.",
                file=sys.stderr,
            )
            sys.exit(2)
        from mmcli.datasets import extract_dataset
        extract_dataset(
            dataset_name=args.dataset,
            project_path=args.project,
            task_type=args.task,
        )
        sys.exit(0)

    if args.command == "info":
        from mmcli.info import run_info
        run_info(args, _get_python_exe())
        sys.exit(0)

    if args.command == "analyze":
        # Batch mode handling
        if getattr(args, "batch", None) or getattr(args, "directory", None):
            import copy, glob, os
            if args.batch:
                paths = expand_project_paths(args.batch)
            elif args.directory:
                yaml_files = glob.glob(os.path.join(args.directory, "**/*.yaml"), recursive=True)
                paths = yaml_files
            else:
                paths = [args.project]

            def analyze_one(project_path):
                local_args = copy.deepcopy(args)
                local_args.project = project_path
                setattr(local_args, "batch", None)
                setattr(local_args, "directory", None)
                from mmcli.analyze import run_analyze as _run_analyze
                try:
                    _run_analyze(local_args)
                    return True
                except SystemExit as e:
                    # If the function exits with non-zero code, treat as failure
                    return False

            results = run_batch(analyze_one, paths)
            print(format_batch_results(results))
            sys.exit(0 if all(r["success"] for r in results.values()) else 1)
        else:
            from mmcli.analyze import run_analyze
            run_analyze(args)
            sys.exit(0)

    if args.command == "recommend":
        from mmcli.recommend import run_recommend
        run_recommend(args)
        sys.exit(0)

    # Batch processing for train command
    if args.command == "train" and (getattr(args, "batch", None) or getattr(args, "directory", None)):
        import copy, glob, os
        # Determine project paths
        if args.batch:
            paths = expand_project_paths(args.batch)
        elif args.directory:
            yaml_files = glob.glob(os.path.join(args.directory, "**/*.yaml"), recursive=True)
            paths = yaml_files
        else:
            paths = [args.project]

        def train_one(project_path):
            # Clone arguments and set project path
            local_args = copy.deepcopy(args)
            local_args.project = project_path
            # Clear batch flags to avoid recursion
            setattr(local_args, "batch", None)
            setattr(local_args, "directory", None)
            _validate_args(local_args)
            cfg = build_config(local_args)
            _validate_config(cfg)
            py_exe = _get_python_exe()
            rc = _dispatch(cfg, py_exe, verbose=getattr(local_args, "verbose", False), report_path=None)
            return rc == 0

        results = run_batch(train_one, paths)
        print(format_batch_results(results))
        sys.exit(0 if all(r["success"] for r in results.values()) else 1)

    if args.command == "deploy":
        from mmcli.deploy import (
            run_deploy_check_sdk, run_deploy_artifacts,
            run_deploy_create, run_deploy_build, run_deploy_flash,
        )
        _deploy_handlers = {
            "check-sdk": run_deploy_check_sdk,
            "artifacts": run_deploy_artifacts,
            "create":    run_deploy_create,
            "build":     run_deploy_build,
            "flash":     run_deploy_flash,
        }
        handler = _deploy_handlers.get(args.deploy_subcommand)
        if handler:
            handler(args)
        sys.exit(0)

    if args.command == "shell":
        from mmcli.interactive import run_shell
        run_shell()
        sys.exit(0)

    if args.command == "compare":
        from mmcli.compare import compare_models, format_comparison

        if args.all_models:
            task_types = [
                "generic_timeseries_classification",
                "generic_timeseries_regression",
                "generic_timeseries_forecasting",
                "generic_timeseries_anomalydetection",
            ]
        else:
            task_types = [args.model1, args.model2]

        comparison = compare_models(
            module_type=args.module,
            task_types=task_types,
            device=args.device,
        )

        print(format_comparison(comparison))
        sys.exit(0)

    if args.command == "diagnose":
        from mmcli.diagnose import (
            run_diagnostic_checks,
            format_diagnostic_results,
            get_fix_for_error
        )

        if args.error:
            # Get fix for specific error
            severity, suggestion = get_fix_for_error(args.error)

            print("=" * 60)
            print("ERROR FIX SUGGESTION")
            print("=" * 60)
            print(f"Error: {args.error}")
            print("")
            print(f"Severity: {severity.upper()}")
            print(f"Suggestion: {suggestion}")
            sys.exit(0)

        else:
            # Run full diagnostic
            result = run_diagnostic_checks(full=args.full)

            print(format_diagnostic_results(result))
            print("")

            if not result.is_healthy:
                print("Some checks failed. Please address the issues above.")
                sys.exit(1)

            print("All checks passed! Your system is ready to use mmcli.")
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

    # Determine report output path if --report is enabled
    report_path = None
    if getattr(args, 'report', False):
        train_output = config.get('training', {}).get('train_output_path')
        if train_output:
            report_path = os.path.join(train_output, 'report.html')
        else:
            project = getattr(args, 'project', None)
            if project:
                report_path = os.path.join(os.path.abspath(project), 'run', 'report.html')
            else:
                report_path = 'report.html'

    nas_enabled = getattr(args, 'nas_size', None) is not None
    rc = _dispatch(config, python_exe, verbose=args.verbose,
                   report_path=report_path, nas_enabled=nas_enabled)
    sys.exit(rc)
