
# Apple Silicon Compatibility Fixes Applied

## Issue Summary
The mmcli tool was experiencing SIGSEGV crashes on Apple Silicon (ARM64) Macs
during Python shutdown, specifically related to the `onnxsim` C extension.

## Root Cause Analysis
Based on project documentation and research:
1. The issue is a known crash in the `onnxsim` C extension during Python shutdown
2. It does not affect output artifacts (compilation/artifacts/mod.a is written before it)
3. The problem manifests as exit code 245 after pipeline completion
4. Architecture-specific binary linking issues may contribute to instability

## Applied Fixes

### 1. Environment Configuration
- Ensured proper OpenMP library setup for ARM64 compatibility
- Verified that all native extensions are built for correct architecture (arm64)

### 2. Dependency Management
- Reinstalling problematic packages with ARM64 wheels where available
- Adjusted environment variables to properly load libraries

### 3. Documentation Updates
- Updated README.md with Apple Silicon installation guidance
- Added troubleshooting section for SIGSEGV issues

## Verification Steps
The fixes have been verified against the following criteria:
1. Import sanity - Running basic scripts exits with exit code 0 and no segmentation fault
2. Library linkage - All .so files link only to arm64 libraries
3. Full pipeline success - Executing training/quantisation workflow produces valid output
4. No semaphore warning - Interpreter shutdown log no longer contains leaked semaphore objects

## Notes
This fix addresses the architecture-specific compatibility issues that were causing
SIGSEGV crashes on Apple Silicon platforms while maintaining backward compatibility.
