#!/usr/bin/env bash
# build_macos.sh — Build a standalone macOS binary for mmcli using PyInstaller
#
# The training engine (torch, TVM, tinyml_modelmaker and friends) is excluded via
# --exclude-module, driven by scripts/pyinstaller_excludes.txt, because mmcli calls out
# to it via the MMCLI_PYTHON subprocess and never needs it in-process. The bundled
# example datasets are still the largest remaining component until a later phase
# unbundles them; see scripts/binary_size_ceiling.txt for the current size ceiling.
# At runtime the binary calls out to an external Python interpreter via the
# MMCLI_PYTHON environment variable.
#
# Requirements (in the active venv):
#   pip install pyinstaller mmcli  (or pip install -e .)
#
# Output: dist/mmcli  (single-file native binary)
#
# Usage:
#   bash build_macos.sh              # arm64 (Apple Silicon, default)
#   ARCH=x86_64 bash build_macos.sh  # Intel
#   ARCH=universal2 bash build_macos.sh  # fat binary (both)

set -euo pipefail

ARCH="${ARCH:-arm64}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building mmcli for macOS ${ARCH}..."

# Ensure PyInstaller is available
if ! python -c "import PyInstaller" 2>/dev/null; then
    echo "Installing PyInstaller..."
    pip install pyinstaller
fi

# Install mmcli into site-packages (non-editable, so PyInstaller can find it).
# Always reinstall to pick up latest source changes.
echo "Installing mmcli into site-packages..."
pip install "${SCRIPT_DIR}" --force-reinstall --no-deps -q

python -c "import mmcli; print(mmcli.__file__)"

# Clean previous build artifacts
rm -rf "${SCRIPT_DIR}/build" "${SCRIPT_DIR}/dist/mmcli" "${SCRIPT_DIR}/mmcli.spec"

# mmcli drives the training engine through MMCLI_PYTHON as a subprocess; it never needs
# the engine in its own address space. But a handful of `import tinyml_modelmaker` /
# `import tvm` probes sit in try/except fallbacks, and PyInstaller's static analysis
# follows them, dragging in torch, TVM and the whole engine — a ~10 MB launcher became
# 260 MB with a 6.5 s startup. Excluding them keeps those probes as the no-ops they
# already handle. numpy and pandas stay: analyze.py genuinely uses them.
#
# The exclude list is shared across all three build scripts
# (scripts/pyinstaller_excludes.txt) so it cannot drift between platforms.
EXCLUDE_ARGS=()
while IFS= read -r m; do
    [ -n "$m" ] && EXCLUDE_ARGS+=(--exclude-module "$m")
done < "${SCRIPT_DIR}/scripts/pyinstaller_excludes.txt"

pyinstaller \
    --onefile \
    --name mmcli \
    --target-arch "${ARCH}" \
    --paths "${SCRIPT_DIR}" \
    --collect-submodules mmcli \
    --add-data "${SCRIPT_DIR}/mmcli/example_datasets:mmcli/example_datasets" \
    "${EXCLUDE_ARGS[@]}" \
    "${SCRIPT_DIR}/mmcli/__main__.py"

# Verify the binary contains mmcli modules
echo ""
echo "Verifying binary contents..."
python -c "
import os, tempfile
from PyInstaller.archive.readers import CArchiveReader, ZlibArchiveReader
arch = CArchiveReader('${SCRIPT_DIR}/dist/mmcli')
pyz_data = arch.extract('PYZ.pyz')
tmp = tempfile.NamedTemporaryFile(suffix='.pyz', delete=False)
tmp.write(pyz_data[1] if isinstance(pyz_data, tuple) else pyz_data)
tmp.close()
pyz = ZlibArchiveReader(tmp.name)
entries = [k for k in pyz.toc.keys() if k.startswith('mmcli')]
os.unlink(tmp.name)
print(f'  mmcli modules bundled: {len(entries)}')
for e in sorted(entries):
    print(f'    {e}')
if not entries:
    print('  ERROR: No mmcli modules found in binary!')
    exit(1)
"

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

