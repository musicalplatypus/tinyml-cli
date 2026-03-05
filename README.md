# mmcli

Command-line interface for [tinyml-modelmaker](https://github.com/musicalplatypus/tinyml-tensorlab).

`mmcli` is a **self-contained native macOS binary** that drives the tinyml-modelmaker
training and compilation pipeline entirely from the command line — no YAML config file
required (though one can optionally be used as a base).

It ships with **9 bundled example datasets** covering all task types, so you can
create a project and start training with a single command.

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

# Install tinyml_modelmaker from the release tag
pip install "tinyml_modelmaker @ git+https://github.com/musicalplatypus/tinyml-tensorlab.git@PlatypusCLI_0.9.0_Release#subdirectory=tinyml-modelmaker"
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
git clone --branch PlatypusCLI_0.9.0_Release https://github.com/musicalplatypus/tinyml-cli.git
cd tinyml-cli
source ~/.venv-ai/bin/activate        # any venv with PyInstaller
pip install pyinstaller -q
pip install -e .
bash build_macos.sh                   # → dist/mmcli  (~10 MB)
```

---

## Subcommands

### `mmcli init` — create a project from an example dataset

Create a new project directory pre-populated with a dataset, ready for training.

```
mmcli init -t TASK --dataset DATASET_NAME -p PROJECT_DIR [-m MODULE]
```

| Flag | Short | Description |
|------|-------|-------------|
| `--task` | `-t` | Task type **(required)** |
| `--dataset` | | Name of the example dataset **(required)** |
| `--project` | `-p` | Path for the new project directory **(required, must not exist)** |
| `--module` | `-m` | AI module (auto-detected from dataset if omitted) |

**Example:**
```bash
# Create a project for arc fault classification
mmcli init -t arc_fault --dataset arc_fault_classification -p ./my_arc_project

# Then train with it
mmcli train -m timeseries -t arc_fault -d F28P55 -n CLS_1k_NPU -i ./my_arc_project
```

> **Tip:** Run `mmcli info -m timeseries -t <task>` to see which datasets are available
> for a given task.

---

### `mmcli train` — train only

Train a model and export `model.onnx`. Compilation is skipped.

```
mmcli train -m MODULE -t TASK -d DEVICE -n MODEL -i PROJECT_DIR [options]
mmcli train -m MODULE -t TASK -d DEVICE --nas SIZE -i PROJECT_DIR [options]
```

| Flag | Short | Description |
|------|-------|-------------|
| `--module` | `-m` | `timeseries` or `vision` |
| `--task` | `-t` | Task type (see list below) |
| `--device` | `-d` | Target device (e.g. `F28P55`) |
| `--model` | `-n` | Model name from catalog (optional with `--nas`) |
| `--project` | `-i` | Path to project directory containing `dataset/` |
| `--config` | `-c` | Optional base YAML file (CLI args override) |
| `--feature-extraction` | | Feature extraction preset name |
| `--epochs` | | Training epochs |
| `--batch-size` | | Batch size |
| `--lr` | | Learning rate |
| `--gpus` | | Number of GPUs (0 = CPU/MPS, default on macOS) |
| `--quantization` | | `NO_QUANTIZATION` or `QUANTIZATION_TINPU` |
| `--run-name` | | Output folder name (supports `{date-time}`, `{model_name}`) |
| `--output` | | Root output directory |
| `--compile-model` | | `0` (default) or `1` to enable torch.compile (CUDA recommended) |
| `--native-amp` | | Enable native mixed precision (CUDA recommended, not for MPS) |
| `--report` | | Generate a live-updating HTML training report (charts + confusion matrix) |
| `--nas` | | NAS model size preset: `s`, `m`, `l`, `xl` (classification only) |
| `--nas-epochs` | | NAS search epochs (default: 10) |
| `--nas-optimize` | | NAS resource target: `Memory` (default) or `Compute` |

**Example:**
```bash
mmcli train \
  -m timeseries \
  -t generic_timeseries_classification \
  -d F28P55 \
  -n CLS_1k_NPU \
  -i ./data/my_project \
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
  -n CLS_1k_NPU \
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
  -n CLS_1k_NPU \
  -i ./data/my_dataset \
  --quantization QUANTIZATION_TINPU
```

---

## Example Datasets

`mmcli` bundles 9 example datasets downloaded from TI's servers. Use `mmcli init`
to extract them into a new project.

| Dataset Name | Task Type | Size | Description |
|-------------|-----------|------|-------------|
| `generic_timeseries_classification` | classification | 2.5 MB | Synthetic waveforms (sawtooth, sine, square) |
| `generic_timeseries_regression` | regression | 885 KB | Synthetic regression data |
| `generic_timeseries_anomalydetection` | anomaly detection | 4.0 MB | Amplitude/frequency anomalies |
| `generic_timeseries_forecasting` | forecasting | 69 KB | Simulated thermostat temperatures |
| `arc_fault_classification` | arc_fault | 13 MB | DC arc fault currents (DSI sensor) |
| `ecg_classification` | ecg_classification | 4.4 MB | ECG 2-class heartbeat (normal vs abnormal) |
| `fan_blade_fault` | motor_fault | 54 MB | Fan blade vibration (3-axis accelerometer) |
| `pir_detection` | pir_detection | 1.5 MB | PIR motion detection (human vs non-human) |
| `mnist_image_classification` | image_classification | 45 MB | MNIST handwritten digits (28×28 images) |

To use an external datasets directory instead of the bundled one, set:
```bash
export MMCLI_DATASETS=/path/to/your/datasets
```


## Useful options

### `mmcli info` — query the model registry

Show supported task types, models, devices, feature extraction presets,
and available example datasets.

```
mmcli info -m MODULE [-t TASK] [-d DEVICE]
```

| Flag | Short | Description |
|------|-------|-------------|
| `--module` | `-m` | `timeseries` or `vision` **(required)** |
| `--task` | `-t` | Task type to show details for. Omit to list all task types. |
| `--device` | `-d` | Target device to filter models. |

**Examples:**
```bash
mmcli info -m timeseries                        # list task types
mmcli info -m timeseries -t arc_fault           # details for arc_fault
mmcli info -m timeseries -t arc_fault -d F28P55 # models for F28P55
```

---

### Dry run — inspect the generated YAML without running anything

```bash
mmcli --dry-run train \
  -m timeseries -t generic_timeseries_classification \
  -d F28P55 -n CLS_1k_NPU -i ./data
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

### Training report

Generate a self-contained HTML report with live-updating accuracy/loss charts
and a heatmap confusion matrix:

```bash
mmcli train \
  -m timeseries \
  -t generic_timeseries_classification \
  -d F28P55 \
  -n CLS_1k_NPU \
  -i ./data/my_project \
  --report
```

The report is written to `<project_dir>/run/report.html` and auto-refreshes
every 5 seconds while training is in progress. Once training completes, the
auto-refresh is removed and the final report includes the confusion matrix
and file-level classification summary (if available).

---

## Neural Architecture Search (NAS)

Instead of picking a model from the catalog (`-n`), you can let NAS automatically
discover an optimal architecture for your dataset. NAS is supported for
**classification tasks only** (timeseries and vision).

When `--nas` is set, `--model/-n` becomes optional — a synthetic name like
`NAS_m` is generated automatically.

### Feature extraction with NAS

> **Important:** For timeseries tasks, you **must** specify `--feature-extraction`
> with a preset that is appropriate for your dataset when using `--nas`.
>
> With catalog models (`-n`), the feature extraction configuration is provided
> automatically by the model description. NAS has no catalog entry, so the
> pipeline does not know which feature extraction to apply. Without this flag,
> training will fail with *"Not enough dimensions present"* because the raw
> sensor data has not been transformed into the feature representation that
> the classification pipeline expects.

To see which feature extraction presets are available for your task:

```bash
mmcli info -m timeseries -t generic_timeseries_classification
```

Common presets for generic timeseries classification include:
- `Generic_256Input_FFTBIN_16Feature_8Frame`
- `Generic_1024Input_FFTBIN_32Feature_32Frame`

### NAS flags

| Flag | Description |
|------|-------------|
| `--nas SIZE` | Enable NAS with a model size preset: `s` (small), `m` (medium), `l` (large), `xl` (extra-large). Controls the search space complexity and resulting model size. |
| `--nas-epochs N` | NAS search epochs (default: 10). Higher values explore more architectures but take longer. |
| `--nas-optimize MODE` | Resource optimization target: `Memory` (fewer parameters, default) or `Compute` (fewer MACs, lower latency). |
| `--feature-extraction` | Feature extraction preset **(required for timeseries NAS)**. Use `mmcli info` to list available presets. |

### NAS examples

```bash
# Basic NAS — medium-sized model with feature extraction
mmcli train \
  -m timeseries \
  -t generic_timeseries_classification \
  -d F28P55 \
  -i ./data/my_project \
  --nas m \
  --feature-extraction Generic_256Input_FFTBIN_16Feature_8Frame \
  --epochs 50

# NAS with explicit search budget and compute optimization
mmcli train \
  -m timeseries \
  -t motor_fault \
  -d F28P55 \
  -i ./data/motor_project \
  --nas l \
  --nas-epochs 20 \
  --nas-optimize Compute \
  --feature-extraction Generic_256Input_FFTBIN_16Feature_8Frame

# NAS for vision classification (no --feature-extraction needed)
mmcli train \
  -m vision \
  -t image_classification \
  -d F29H85 \
  -i ./data/image_project \
  --nas s

# Dry-run to inspect NAS config without running
mmcli --dry-run train \
  -m timeseries -t generic_timeseries_classification \
  -d F28P55 -i ./data/my_project --nas m \
  --feature-extraction Generic_256Input_FFTBIN_16Feature_8Frame
```

### Supported tasks for NAS

`generic_timeseries_classification` · `arc_fault` · `ecg_classification`
`motor_fault` · `blower_imbalance` · `pir_detection` · `image_classification`

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

**Classification:** `CLS_100_NPU` `CLS_500_NPU` `CLS_1k_NPU` `CLS_2k_NPU`
`CLS_4k_NPU` `CLS_6k_NPU` `CLS_8k_NPU` `CLS_13k_NPU` `CLS_20k_NPU`
`CLS_55k_NPU` `CLS_ResAdd_3k` `CLS_ResCat_3k`

**Regression:** `REGR_1k` `REGR_2k` `REGR_3k` `REGR_4k` `REGR_10k` `REGR_13k`
`REGR_500_NPU` `REGR_2k_NPU` `REGR_6k_NPU` `REGR_8k_NPU` `REGR_20k_NPU`

**Anomaly Detection:** `AD_1k` `AD_4k` `AD_16k` `AD_17k` `AD_Linear`
`AD_500_NPU` `AD_2k_NPU` `AD_6k_NPU` `AD_8k_NPU` `AD_10k_NPU` `AD_20k_NPU`

**Forecasting:** `FCST_3k` `FCST_13k` `FCST_LSTM8` `FCST_LSTM10`
`FCST_500_NPU` `FCST_1k_NPU` `FCST_2k_NPU` `FCST_4k_NPU` `FCST_6k_NPU`
`FCST_8k_NPU` `FCST_10k_NPU` `FCST_20k_NPU`

**Application-specific:** `ArcFault_model_200_t` `ArcFault_model_300_t`
`ArcFault_model_700_t` `ArcFault_model_1400_t`
`MotorFault_model_1_t` `MotorFault_model_2_t` `MotorFault_model_3_t`
`FanImbalance_model_1_t` `FanImbalance_model_2_t` `FanImbalance_model_3_t`
`ECG_55k_NPU` `PIRDetection_model_1_t`

> **Tip:** Run `mmcli info -m timeseries -t <task>` to see models available for a specific task.

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
| `MMCLI_PYTHON` | `python` or `python3` on PATH | Python interpreter with `tinyml_modelmaker` installed |
| `MMCLI_MODELMAKER` | auto-detected | Path to tinyml-modelmaker source dir (only needed if auto-detection fails) |
| `MMCLI_DATASETS` | bundled `example_datasets/` | Override directory containing example dataset zips |
