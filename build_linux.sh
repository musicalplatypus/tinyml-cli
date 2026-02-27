#!/usr/bin/env bash
# build_linux.sh — Build a standalone Linux binary for mmcli using PyInstaller
#
# The binary is lightweight (~10 MB) because tinyml_modelmaker is NOT bundled.
# At runtime the binary calls out to an external Python interpreter via the
# MMCLI_PYTHON environment variable.
#
# Requirements (in the active venv):
#   pip install pyinstaller mmcli  (or pip install -e .)
#
# Output: dist/mmcli  (single-file native binary)
#
# Usage:
#   bash build_linux.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building mmcli for Linux..."

# Ensure PyInstaller is available
if ! python -c "import PyInstaller" 2>/dev/null; then
    echo "Installing PyInstaller..."
    pip install pyinstaller
fi

# Clean previous build artifacts
rm -rf "${SCRIPT_DIR}/build" "${SCRIPT_DIR}/dist/mmcli"

pyinstaller \
    --onefile \
    --name mmcli \
    --hidden-import mmcli \
    --hidden-import mmcli.builder \
    --hidden-import mmcli.cli \
    "${SCRIPT_DIR}/mmcli/__main__.py"

echo ""
echo "Build complete: ${SCRIPT_DIR}/dist/mmcli"
echo ""
echo "Usage:"
echo "  export MMCLI_PYTHON=/path/to/venv/bin/python"
echo "  ./dist/mmcli --version"
echo "  ./dist/mmcli --help"
echo "  ./dist/mmcli train --help"
echo "  ./dist/mmcli --dry-run train -m timeseries -t generic_timeseries_classification \\"
echo "      -d F28P55 -n CLS_1k_NPU -i ./data"
