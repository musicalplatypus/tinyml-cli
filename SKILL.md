---
name: mmcli
description: Use when the user wants to train, compile, analyze, or deploy a TI tinyml model via the mmcli command-line interface. Covers dataset inspection, model recommendation, training, compilation, and hardware deployment to TI MCUs. Always verify MMCLI_PYTHON is set before running any command.
---

# mmcli — TI TinyML CLI Skill

## Environment setup

Before running any `mmcli` command, verify the Python interpreter is configured:

```bash
# Set MMCLI_PYTHON to the venv that has tinyml-modelmaker installed
export MMCLI_PYTHON=~/.venv-tinyml/bin/python

# Verify:
$MMCLI_PYTHON -c "import tinyml_modelmaker; print('OK')"
```

For C2000-target compilation (F28P55, F28P65, etc.) on any platform, set:
```bash
# Point to the TI C2000 CGT installation root (version 25.11.1.LTS or compatible)
export C2000_CG_ROOT=~/ti/ti-cgt-c2000_25.11.1.LTS
```
modelmaker reads `C2000_CG_ROOT` at import time; it must be set before invoking `mmcli`.

**macOS ARM64 note:** on macOS, `mmcli run` / `mmcli compile` may exit with code 245
after the pipeline completes. This is a crash in onnxsim's C extension during Python
interpreter shutdown — all artifacts (`mod.a`, ONNX files) are written before it fires.

If `tinyml-modelzoo` recommendations are needed, also set:
```bash
export MMCLI_MODELZOO_PATH=~/tinyml-tensorlab/tinyml-modelzoo
```

---

## Core workflow

```
1. mmcli analyze    — inspect dataset (size, classes, sample count)
2. mmcli recommend  — pick model + FE preset from modelzoo examples
3. mmcli info       — browse supported devices, tasks, models, FE presets
4. mmcli init       — create a project from a bundled example dataset
5. mmcli train      — train (ONNX export only)
6. mmcli compile    — compile an existing ONNX → TVM artifacts
7. mmcli run        — train + compile in one pass
8. mmcli deploy     — create CCS project, build, and flash to device
```

---

## 1. Dataset analysis

```bash
mmcli analyze -i <project_dir>
# <project_dir>/dataset/ must contain classes/ or files/ layout

# Output: row counts, class balance bar chart, dataset size bucket
# (tiny / small / medium / large) — use bucket in recommend step
```

---

## 2. Model recommendation

```bash
mmcli recommend -t <task_type> -d <device> [options]

# Examples:
mmcli recommend -t motor_fault -d F28P55 --variables 3 --dataset-size-bucket small
mmcli recommend -t arc_fault   -d F28P55 --top 10
mmcli recommend -t generic_timeseries_forecasting -d MSPM0G5187

# Key options:
#   --variables N          sensor channel count (+2 score bonus for exact match)
#   --dataset-size-bucket  tiny | small | medium | large
#   --top N                how many ranked matches to show (default 5)
#   --modelzoo-path DIR    explicit path to modelzoo repo root
```

Scoring: task(+1) device(+1) module(+1) variables-exact(+2) size(+1) = max 6.
Use the recommended model name with `mmcli train -n <model>`.

---

## 3. Device / task / preset browser

```bash
mmcli info -m timeseries -t motor_fault -d F28P55
mmcli info -m audio
mmcli info -m timeseries -t generic_timeseries_classification
```

---

## 4. Create project from bundled dataset

```bash
mmcli init --list                                  # show available datasets
mmcli init --dataset motor_fault -t motor_fault -p /path/to/project
mmcli init --dataset generic_audio_classification -t audio_classification -p /path/to/project
```

Bundled datasets: generic_timeseries_classification, generic_timeseries_regression,
generic_timeseries_anomalydetection, generic_timeseries_forecasting,
arc_fault_classification, ecg_classification, fan_blade_fault,
pir_detection, mnist_image_classification, generic_audio_classification.

---

## 5–7. Train / compile / run

```bash
# Train (exports ONNX, no compilation)
mmcli train -m timeseries -t motor_fault -d F28P55 -n CLS_1k_NPU -i /path/to/project

# Compile only (from existing ONNX)
mmcli compile -m timeseries -t motor_fault -d F28P55 -n CLS_1k_NPU \
  --onnx /path/to/model.onnx

# Full pipeline
mmcli run -m timeseries -t motor_fault -d F28P55 -n CLS_1k_NPU -i /path/to/project

# Key options (all three commands):
#   --quantization QUANTIZATION_TINPU | NO_QUANTIZATION
#   --auto-quantization              binary search for optimal bitwidth
#   --epochs N  --batch-size N  --lr FLOAT
#   --preset auto | default_preset | forced_soft_npu_preset | compress_npu_layer_data
#   --report                         generate live HTML training report
#   --dry-run                        print YAML config without executing
```

### Preset auto-resolution

`--preset auto` picks the right compilation path:
- NPU device + standard task + QUANTIZATION_TINPU → `default_preset` (hardware NPU)
- NPU device + anomaly/forecasting task           → `forced_soft_npu_preset` (soft-NPU)
- Non-NPU device or non-TI quantization           → `default_preset`

NPU-capable devices: F28P55, F28P65, MSPM0G5187, MSPM33C34, CC2755, CC35X1, AM13E2.

---

## 8. Device deployment

### 8A. Verify SDK

```bash
mmcli deploy check-sdk -d F28P55
mmcli deploy check-sdk -d CC1312 --sdk-path ~/ti/simplelink_cc13xx_sdk_7.10.00
```

### 8B. Locate compiled artifacts

```bash
mmcli deploy artifacts -t motor_fault --run-id 20240115_143022 --model-id CLS_1k_NPU
# Add --no-quantization for float model artifacts
# Add --tinyml-base <PATH> if checkout is not at ~/tinyml-tensorlab
```

### 8C. Create CCS project

```bash
mmcli deploy create \
  -d F28P55 -t motor_fault \
  --run-id 20240115_143022 --model-id CLS_1k_NPU \
  --project-name my_motor_fault_project
# --device-type f28p55x  (auto-resolved; override if needed)
# --sdk-path / --ccs-templates-path  (override SDK auto-detection)
```

CCS device_type strings (auto-resolved from --device):
| Device      | CCS type    |
|-------------|-------------|
| F28P55      | f28p55x     |
| F28P65      | f28p65x     |
| F28004      | f28004x     |
| MSPM0G3507  | mspm0g3507  |
| MSPM0G5187  | mspm0g5187  |
| AM263       | am263       |
| CC2755      | cc2755      |
| CC1312      | cc1312      |
| CC1314      | cc1314      |
| CC1352      | cc1352      |

### 8D. Build

```bash
mmcli deploy build \
  --project-path /path/to/project \
  --ccs-path /opt/ti/ccs1260
# --build-type incremental  (default: full)
```

### 8E. Flash

```bash
mmcli deploy flash \
  --project-path /path/to/project \
  --ccs-path /opt/ti/ccs1260
# --ccxml TMS320F28P550SJ9_LaunchPad.ccxml  (auto-searched if omitted)
```

---

## Supported devices

| Family    | Devices                                          | SDK           |
|-----------|--------------------------------------------------|---------------|
| c2000     | F280013, F280015, F28003, F28004, F2837, F28P55, F28P65, F29H85, F29P58, F29P32 | C2000Ware |
| mspm0     | MSPM0G3507, MSPM0G3519, MSPM0G5187              | MSPM0 SDK     |
| mspm33    | MSPM33C32, MSPM33C34                            | MSPM33 SDK    |
| am13      | AM13E2                                          | MCU+ SDK      |
| am26x     | AM263, AM263P, AM261                            | MCU+ SDK      |
| simplelink| CC1312, CC1314, CC1352, CC1354, CC2755, CC35X1  | SimpleLink LP |

---

## Supported task types

**Timeseries:** generic_timeseries_classification, generic_timeseries_regression,
generic_timeseries_anomalydetection, generic_timeseries_forecasting,
arc_fault, ecg_classification, motor_fault, blower_imbalance, pir_detection

**Vision:** image_classification

**Audio:** audio_classification

---

## Common flag reference

| Flag                    | Commands          | Notes                             |
|-------------------------|-------------------|-----------------------------------|
| `-m / --module`         | most              | timeseries / vision / audio       |
| `-t / --task`           | most              | task type string                  |
| `-d / --device`         | most              | canonical device ID               |
| `-n / --model`          | train/run         | model name from modelzoo          |
| `-i / --project`        | train/run         | project dir (contains dataset/)   |
| `--quantization`        | train/run/compile | QUANTIZATION_TINPU / NO_QUANTIZATION |
| `--auto-quantization`   | train/run         | binary search for best bitwidth   |
| `--preset`              | compile/run       | auto / default_preset / forced_soft_npu_preset |
| `--feature-extraction`  | train/run         | FE preset name from schema        |
| `--report`              | train/run         | generate HTML training report     |
| `--dry-run`             | train/compile/run | print YAML, do not execute        |
| `--verbose / -v`        | all               | debug logging                     |
