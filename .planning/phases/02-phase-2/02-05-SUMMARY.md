# Plan 02-05 Summary: Document Environment Variables in CLI Help

**Status:** COMPLETE  
**Date:** 2026-07-06  

## Changes Made

Updated `mmcli/cli.py` to document all MMCLI_* environment variables:

### Module-level docstring (lines 1-38)
Added entries for:
- **MMCLI_DATASETS**: Override built-in datasets directory
- **MMCLI_MODELZOO_PATH**: Path to tinyml-modelzoo repo root

### Main help text (lines 1248-1254)
Updated Environment variables section with all 4 variables:

```
Environment variables:
  MMCLI_PYTHON      Python interpreter with tinyml_modelmaker installed
                    Default: 'python' or 'python3' on PATH
  MMCLI_MODELMAKER  Path to the tinyml-modelmaker source directory
                    (auto-detected if MMCLI_PYTHON is set correctly)
  MMCLI_DATASETS    Override built-in datasets directory for 'mmcli init'
  MMCLI_MODELZOO_PATH  Path to the tinyml-modelzoo repo root for 'mmcli recommend'
```

## Verification

```bash
$ python3 -m mmcli --help | grep -A 10 "Environment variables"
Environment variables:
  MMCLI_PYTHON      Python interpreter with tinyml_modelmaker installed
                    Default: 'python' or 'python3' on PATH
  MMCLI_MODELMAKER  Path to the tinyml-modelmaker source directory
                    (auto-detected if MMCLI_PYTHON is set correctly)
  MMCLI_DATASETS    Override built-in datasets directory for 'mmcli init'
  MMCLI_MODELZOO_PATH  Path to the tinyml-modelzoo repo root for 'mmcli recommend'

Run 'mmcli help' to see all subcommands and options at once.
```

## Outcome

All 4 MMCLI_* environment variables are now documented in CLI help text:
- ✅ mmcli --help module docstring
- ✅ mmcli main() help section

Each variable has a clear description and default value where applicable.
