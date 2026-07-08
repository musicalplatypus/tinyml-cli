# Context: Phase 6 - Fix Apple Silicon SIGSEGV Crash

## Project Background
This repository contains the mmcli tool, a command-line interface for the tinyml-modelmaker project. The tool is designed to work on macOS (arm64), Linux (x86_64), and Windows (x86_64) platforms.

## Known Issue - Apple Silicon SIGSEGV
Based on the project documentation, there's a known issue where macOS ARM64 processes may exit with code 245 after pipeline completion. This is a known crash in the `onnxsim` C extension during Python shutdown and does not affect output artifacts (`compilation/artifacts/mod.a` is written before it).

## Current State
- Phase 5 has been completed successfully with all new features implemented
- The repository contains a phase directory `06-fix-apple-silicon-sigsegv` indicating this is an active planned phase
- VERIFICATION.md exists with specific criteria for completion
- RESEARCH.md exists with audit approach

## Phase Objectives
1. Identify architecture-specific binary linking issues causing SIGSEGV crashes on Apple Silicon
2. Resolve compatibility problems with native extensions and libraries  
3. Ensure full pipeline workflow completes successfully on ARM64 hardware
4. Maintain backward compatibility with existing x86_64 platforms

## Technical Considerations
- The issue is specifically related to ARM64 architecture compatibility
- Focus on compiled extension binaries (.so files) that may be built for wrong architecture (x86_64)
- Memory access violations during interpreter shutdown are the primary concern
- Metal/MPS performance and stability need to be verified

## Dependencies
- Phase 5: All new features implemented (progress visualization, export formats, etc.)
- Existing tool functionality must remain intact
- Testing environment with Apple Silicon hardware required

## Success Metrics
- No segmentation faults during mmcli operations on ARM64
- Full pipeline workflow completes successfully 
- Performance characteristics maintained
- No regressions in existing functionality