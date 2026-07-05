# Gradual Software Development (GSD) for mmcli

This project follows a gradual software development approach to build a robust command-line interface for tinyML model development.

## What is GSD?

Gradual Software Development (GSD) is an iterative approach where software is developed in phases, with each phase building upon the previous one. This ensures that functionality is implemented systematically, tested early, and risks are mitigated throughout the development process.

## mmcli GSD Implementation

### Phase 1: Core CLI Functionality
- [x] Basic CLI structure with argument parsing
- [x] Core subcommands (init, train, compile, run)
- [x] Configuration builder for YAML generation
- [x] Help system and version command

### Phase 2: Enhanced Features  
- [x] `info` command for device/model listings
- [x] `analyze` command for dataset inspection
- [x] `recommend` command for model selection
- [x] `deploy` command for hardware deployment

### Phase 3: Testing and Documentation
- [x] Unit tests for all components
- [x] Integration tests for full pipelines  
- [x] End-to-end testing with example datasets
- [x] Comprehensive documentation

## Development Approach

### Iterative Implementation
Each feature is implemented in a separate phase:
1. **Phase 1**: Core commands (init, train, compile, run)
2. **Phase 2**: Advanced features (info, analyze, recommend, deploy)  
3. **Phase 3**: Testing and documentation

### Testing Strategy
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline testing
- **End-to-End Tests**: Real-world scenario validation

## Benefits of This GSD Approach

1. **Early Validation**: Core functionality tested early in development
2. **Risk Mitigation**: Issues identified and resolved at each phase
3. **Clear Milestones**: Well-defined deliverables at each stage
4. **Maintainable Code**: Each phase builds upon a solid foundation
5. **Systematic Progress**: Predictable development trajectory

## Usage Examples

### Project Creation
```bash
mmcli init -t arc_fault --dataset arc_fault_classification -p ./my_arc_project
```

### Training and Compilation
```bash
mmcli run \
  -m timeseries \
  -t generic_timeseries_classification \
  -d F28P55 \
  -n CLS_1k_NPU \
  -i ./my_project \
  --quantization QUANTIZATION_TINPU
```

### Model Recommendation
```bash
mmcli recommend -t motor_fault -d F28P55 --variables 3 --dataset-size-bucket small
```