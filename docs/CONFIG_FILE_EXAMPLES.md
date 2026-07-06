# Config File Examples for mmcli

This document provides working examples of YAML configuration files for use with `mmcli --config`.

## Quick Start

Create a `config.yaml` file and reference it with the `--config` flag:

```bash
mmcli train --config config.yaml
```

CLI arguments override values from the config file, so you can still specify individual flags:

```bash
mmcli train --config config.yaml -t image_classification  # overrides common.task_type
```

---

## Common Sections

All config files share these sections:

### `common`
Required fields:
- `target_module`: `timeseries`, `vision`, or `audio`
- `task_type`: The task to perform (e.g., `generic_timeseries_classification`)
- `target_device`: Target MCU device (e.g., `F28P55`)

### `training`
Required when training:
- `model_name`: Model from catalog (e.g., `CLS_1k_NPU`)
- Optional: `epochs`, `batch_size`, `learning_rate`

### `compilation`
Required when compiling:
- `enable`: `true` or `false`
- `model_path`: Path to ONNX file

### `dataset`
Required when training:
- `input_data_path`: Path to dataset directory

---

## Example 1: Training Configuration (`train.yaml`)

```yaml
# Training configuration example
common:
  target_module: timeseries      # AI module (timeseries, vision, audio)
  task_type: generic_timeseries_classification  # Task type
  target_device: F28P55          # Target microcontroller

training:
  enable: true                   # Enable training phase
  model_name: CLS_1k_NPU         # Model from catalog
  epochs: 50                     # Training epochs (optional)
  batch_size: 32                 # Batch size (optional)
  learning_rate: 0.001           # Learning rate (optional)

dataset:
  input_data_path: ./data/projects/my_project/dataset

# Compilation is separate - you can train without compiling
compilation:
  enable: false
```

**Usage:**
```bash
mmcli train --config train.yaml
```

---

## Example 2: Compilation Configuration (`compile.yaml`)

```yaml
# Compilation-only configuration example
common:
  target_module: timeseries      # Module must match the trained model
  task_type: generic_timeseries_classification
  target_device: F28P55

compilation:
  enable: true                   # Enable compilation phase
  model_path: ./data/projects/my_run/model.onnx   # Path to ONNX file
  preset: default_preset         # Compilation preset (optional)
```

**Usage:**
```bash
mmcli compile --config compile.yaml
```

---

## Example 3: Full Pipeline (`full_pipeline.yaml`)

```yaml
# Complete pipeline: training + compilation
common:
  target_module: vision          # Image classification module
  task_type: image_classification
  target_device: F28P55

training:
  enable: true
  model_name: Lenet5             # Vision model from catalog
  epochs: 100                    # More epochs for complex models
  batch_size: 64

compilation:
  enable: true
  preset: auto                   # Auto-select best preset for device
```

**Usage:**
```bash
mmcli run --config full_pipeline.yaml
# Or separately:
mmcli train --config full_pipeline.yaml   # Step 1: Train
mmcli compile --config full_pipeline.yaml # Step 2: Compile (ONNX must exist)
```

---

## Example 4: Model Recommendations (`recommend.yaml`)

```yaml
# Configuration for mmcli recommend command
common:
  target_module: audio           # Module to search in
  task_type: audio_classification
  target_device: CC1352

# Recommendation criteria (non-common fields)
recommend:
  variables: 8                   # Number of sensor channels (optional)
  dataset_size_bucket: small     # tiny/small/medium/large (optional)

dataset:
  input_data_path: ./data/projects/audio_project/dataset
```

**Usage:**
```bash
mmcli recommend -t audio_classification -d CC1352 --config recommend.yaml
```

---

## Example 5: With Custom Paths (`custom_paths.yaml`)

```yaml
# Configuration using environment variable overrides

# Use environment variables for sensitive paths:
#   MMCLI_PYTHON      - Python interpreter path
#   MMCLI_MODELMAKER  - Modelmaker source directory
common:
  target_module: timeseries
  task_type: motor_fault
  target_device: F28P55

training:
  enable: true
  model_name: MotorFault_model_1_t
  epochs: 30

dataset:
  input_data_path: ./data/projects/motor_project/dataset

compilation:
  enable: false
```

**Usage with env vars:**
```bash
export MMCLI_PYTHON=/path/to/venv/bin/python
mmcli train --config custom_paths.yaml
```

---

## CLI Flag Overrides

CLI arguments take precedence over config values:

| Config Field | CLI Flag | Example |
|-------------|----------|---------|
| `common.target_module` | `-m`, `--module` | `mmcli train -m vision` |
| `common.task_type` | `-t`, `--task` | `mmcli train -t image_classification` |
| `common.target_device` | `-d`, `--device` | `mmcli train -d F28P55` |
| `training.model_name` | `-n`, `--model` | `mmcli train -n Lenet5` |

---

## Best Practices

1. **Keep configs in version control** for reproducibility
2. **Use environment variables** for paths (MMCLI_PYTHON, MMCLI_MODELMAKER)
3. **Name configs descriptively**: `train_classification_f28p55.yaml`
4. **Don't commit sensitive data** to config files
5. **Test with --dry-run** first: `mmcli train --config config.yaml --dry-run`

---

## See Also

- `mmcli help` - Full subcommand documentation
- `mmcli <subcommand> --help` - Subcommand-specific options
- [Environment Variables](../mmcli/cli.py#L16-L38) - MMCLI_* env var reference
