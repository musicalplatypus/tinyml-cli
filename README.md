# mmcli

Command-line interface for [tinyml-modelmaker](https://github.com/musicalplatypus/tinyml-tensorlab).

`mmcli` is a **self-contained native macOS binary** that drives the tinyml-modelmaker
training and compilation pipeline entirely from the command line — no YAML config file
required (though one can optionally be used as a base).

## How it works

`mmcli` is a lightweight binary (~10 MB) that does **not** bundle
`tinyml_modelmaker`, PyTorch, or TVM. Instead it:

1. Translates your CLI arguments into the config dict that `tinyml_modelmaker` expects
2. Writes a temporary YAML file
3. Calls `$MMCLI_PYTHON run_tinyml_modelmaker.py <tempfile>` as a subprocess

This means `tinyml_modelmaker` runs in your existing Python 3.10 environment with all
its native dependencies (including MPS/Metal for macOS training).

> **Note on compilation:** The TVM compiler backend (`ti_mcu_nnc`) currently ships
> Linux and Windows wheels only. The `mmcli compile` and `mmcli run` subcommands will
> work on Linux/Windows. On macOS you can use `mmcli train` to train with Metal/MPS,
> then transfer the resulting `model.onnx` to a Linux machine for compilation.

---

## Setup

### 1. Install tinyml_modelmaker (Python 3.10 environment)

```bash
# Create and activate a Python 3.10 venv
python3.10 -m venv ~/.venv-tinyml
source ~/.venv-tinyml/bin/activate

# Install tinyml_modelmaker
pip install -e ~/Documents/repos/TexasInstruments/tinyml-tensorlab/tinyml-modelmaker
```

### 2. Set the environment variable

Add to your `~/.zshrc` (or `~/.bash_profile`):

```bash
export MMCLI_PYTHON="$HOME/.venv-tinyml/bin/python"
```

### 3. Get the binary

**Option A — Use the pre-built binary:**
```bash
cp dist/mmcli /usr/local/bin/mmcli   # or anywhere on your PATH
```

**Option B — Build it yourself** (requires any Python + PyInstaller in an active venv):
```bash
cd ~/Documents/repos/TexasInstruments/tinyml-cli
source ~/.venv-ai/bin/activate        # any venv with PyInstaller
pip install pyinstaller -q
pip install -e .
bash build_macos.sh                   # → dist/mmcli  (~10 MB)
```

---

## Subcommands

### `mmcli train` — train only

Train a model and export `model.onnx`. Compilation is skipped.

```
mmcli train -m MODULE -t TASK -d DEVICE -n MODEL -i DATA_PATH [options]
```

| Flag | Short | Description |
|------|-------|-------------|
| `--module` | `-m` | `timeseries` or `vision` |
| `--task` | `-t` | Task type (see list below) |
| `--device` | `-d` | Target device (e.g. `F28P55`) |
| `--model` | `-n` | Model name from catalog |
| `--data` | `-i` | Path to input data directory |
| `--config` | `-c` | Optional base YAML file (CLI args override) |
| `--feature-extraction` | | Feature extraction preset name |
| `--epochs` | | Training epochs |
| `--batch-size` | | Batch size |
| `--lr` | | Learning rate |
| `--gpus` | | Number of GPUs (0 = CPU/MPS, default on macOS) |
| `--quantization` | | `NO_QUANTIZATION` or `QUANTIZATION_TINPU` |
| `--run-name` | | Output folder name (supports `{date-time}`, `{model_name}`) |
| `--output` | | Root output directory |

**Example:**
```bash
mmcli train \
  -m timeseries \
  -t generic_timeseries_classification \
  -d F28P55 \
  -n TimeSeries_Generic_1k_t \
  -i ./data/my_dataset \
  --epochs 30 \
  --batch-size 256
```

---

### `mmcli compile` — compile only

Compile a pre-trained ONNX file. No training data needed.

```
mmcli compile -m MODULE -t TASK -d DEVICE -n MODEL -o ONNX_FILE [options]
```

| Flag | Short | Description |
|------|-------|-------------|
| `--onnx` | `-o` | Path to existing ONNX model file **(required)** |
| `--preset` | | Compilation preset (default: `default_preset`) |
| *(common flags above)* | | |

**Example:**
```bash
mmcli compile \
  -m timeseries \
  -t generic_timeseries_classification \
  -d F28P55 \
  -n TimeSeries_Generic_1k_t \
  -o ./data/projects/my_run/model.onnx
```

---

### `mmcli run` — full pipeline

Train then compile. Accepts all flags from both `train` and `compile`.

```bash
mmcli run \
  -m timeseries \
  -t generic_timeseries_classification \
  -d F28P55 \
  -n TimeSeries_Generic_1k_t \
  -i ./data/my_dataset \
  --quantization QUANTIZATION_TINPU
```

---

## Useful options

### Dry run — inspect the generated YAML without running anything

```bash
mmcli --dry-run train \
  -m timeseries -t generic_timeseries_classification \
  -d F28P55 -n TimeSeries_Generic_1k_t -i ./data
```

### Override a base YAML config

```bash
mmcli train --config examples/hello_world/config.yaml \
            --epochs 50 --device F29H85
```

### Verbose output (shows subprocess command + debug logs)

```bash
mmcli --verbose train ...
```

---

## Available task types

**Timeseries:**
`generic_timeseries_classification` · `generic_timeseries_regression`
`generic_timeseries_anomalydetection` · `generic_timeseries_forecasting`
`arc_fault` · `motor_fault` · `blower_imbalance` · `pir_detection`

**Vision:**
`image_classification`

## Available target devices

`F280013` `F280015` `F28003` `F28004` `F2837` `F28P55` `F28P65` `F29H85`
`MSPM0G3507` `MSPM0G5187` `CC2755` `AM263`

## Example model names (timeseries)

`TimeSeries_Generic_100_t` `TimeSeries_Generic_1k_t` `TimeSeries_Generic_4k_t`
`TimeSeries_Generic_6k_t` `TimeSeries_Generic_13k_t`
`ArcFault_model_200_t` `ArcFault_model_1400_t`
`MotorFault_model_1_t` `FanImbalance_model_1_t`

---

## Building the binary

```bash
bash build_macos.sh              # arm64 (Apple Silicon, default)
ARCH=x86_64 bash build_macos.sh  # Intel Mac
ARCH=universal2 bash build_macos.sh  # fat binary (both)
# Output: dist/mmcli
```

Copy `dist/mmcli` anywhere on your `PATH`. No Python environment needed to
**run** the binary — only `MMCLI_PYTHON` must point to a Python 3.10 install
that has `tinyml_modelmaker`.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MMCLI_PYTHON` | `python` on PATH | Python interpreter with `tinyml_modelmaker` installed |
| `MMCLI_MODELMAKER` | auto-detected | Path to tinyml-modelmaker source dir (only needed if auto-detection fails) |
