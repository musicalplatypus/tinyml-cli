# GSD (Gradual Software Development) Setup for mmcli

This document outlines the gradual software development approach for the mmcli tinyML CLI project.

## Project Overview

mmcli is a command-line interface for [tinyml-modelmaker](https://github.com/musicalplatypus/tinyml-tensorlab) that provides a self-contained native binary (macOS, Linux, Windows) for driving the entire training and compilation pipeline from the command line.

## Gradual Software Development Approach

### Phase 1: Core CLI Functionality
Focus on implementing and testing core CLI commands:
- `init` - Create projects from example datasets
- `train` - Train models 
- `compile` - Compile ONNX files
- `run` - Full pipeline training + compilation
- `info` - Query supported models, devices, and presets

### Phase 2: Enhanced Features
Implement additional functionality:
- `analyze` - Dataset analysis
- `recommend` - Model recommendations from modelzoo
- `deploy` - Hardware deployment capabilities

### Phase 3: Testing and Documentation
Establish comprehensive testing and documentation:
- Unit tests for CLI commands
- Integration tests for full pipelines
- End-to-end tests with real datasets
- Detailed documentation and examples

## Implementation Structure

### Core Components

1. **Command Parser** (`cli.py`)
   - Argument parsing using argparse
   - Subcommand handling
   - Help system

2. **Configuration Builder** (`builder.py`)  
   - Translates CLI arguments to config dictionaries
   - YAML file generation
   - Temporary file management

3. **Core Modules**
   - `init` - Project initialization from datasets
   - `train` - Model training pipeline
   - `compile` - ONNX compilation
   - `run` - Full pipeline execution
   - `info` - Model and device information
   - `analyze` - Dataset analysis
   - `recommend` - Model recommendation system
   - `deploy` - Hardware deployment

### Development Workflow

1. **Start with minimal functionality**
   - Implement basic CLI structure
   - Add core commands: init, train, compile, run

2. **Incremental feature addition**
   - Add info command for device/model listings
   - Add analyze for dataset inspection
   - Add recommend for model selection
   - Add deploy for hardware deployment

3. **Testing and validation**
   - Unit tests for each module
   - Integration tests for full pipelines
   - End-to-end testing with example datasets

4. **Documentation and examples**
   - README updates with usage examples
   - Command reference documentation
   - Tutorial walkthroughs

## GSD Implementation Steps

### Step 1: Basic CLI Structure (Week 1)
- Set up argument parser with core subcommands
- Implement basic help system
- Add version command
- Create initial configuration structure

### Step 2: Core Commands (Week 2)
- Implement `init` command for project creation
- Implement `train` command for model training
- Implement `compile` command for ONNX compilation
- Implement `run` command for full pipeline

### Step 3: Advanced Features (Week 3)
- Add `info` command for device/model listings
- Implement `analyze` command for dataset inspection
- Add `recommend` command for model selection
- Implement `deploy` command for hardware deployment

### Step 4: Testing and Documentation (Week 4)
- Write unit tests for all commands
- Create integration tests for full pipelines
- Add end-to-end test cases
- Update documentation with usage examples

## Testing Strategy

### Unit Tests
Test individual components and functions:
- Argument parsing validation
- Configuration generation 
- Command execution logic

### Integration Tests
Test complete command flows:
- Full pipeline execution from init to run
- Dataset processing workflows
- Model training and compilation

### End-to-End Tests  
Test with real datasets:
- Example dataset processing
- Hardware deployment scenarios
- Error handling in various conditions

## Dependencies

The mmcli project requires:
- Python 3.10+ environment
- `tinyml_modelmaker` package from the sibling repo
- PyYAML for configuration files
- PyInstaller for binary creation (optional)

## Environment Setup

### Prerequisites
```bash
# Create and activate a Python 3.10 venv
python3.10 -m venv ~/.venv-tinyml
source ~/.venv-tinyml/bin/activate

# Install tinyml_modelmaker from the release tag
pip install "tinyml_modelmaker @ git+https://github.com/musicalplatypus/tinyml-tensorlab.git@PlatypusCLI_1.0.0_Release#subdirectory=tinyml-modelmaker"
```

### Environment Variables
```bash
export MMCLI_PYTHON="$HOME/.venv-tinyml/bin/python"
```

## Usage Examples

### Basic Project Creation
```bash
mmcli init -t arc_fault --dataset arc_fault_classification -p ./my_arc_project
```

### Training a Model
```bash
mmcli train \
  -m timeseries \
  -t generic_timeseries_classification \
  -d F28P55 \
  -n CLS_1k_NPU \
  -i ./my_project \
  --epochs 30 \
  --batch-size 256
```

### Full Pipeline Execution
```bash
mmcli run \
  -m timeseries \
  -t generic_timeseries_classification \
  -d F28P55 \
  -n CLS_1k_NPU \
  -i ./my_project \
  --quantization QUANTIZATION_TINPU
```

## GSD Benefits

1. **Modular Development**: Features are implemented incrementally
2. **Early Validation**: Core functionality is tested early in the process
3. **Risk Mitigation**: Issues are caught and addressed at each phase
4. **Clear Milestones**: Well-defined development stages with deliverables
5. **Maintainable Codebase**: Each phase builds upon a solid foundation

This gradual approach ensures that the mmcli project is developed systematically, with each stage building upon the previous one to create a robust and well-tested command-line interface for tinyML model development.