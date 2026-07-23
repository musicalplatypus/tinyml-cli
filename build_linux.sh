#!/usr/bin/env bash
# build_linux.sh — Build a standalone Linux binary for mmcli using PyInstaller
#
# The training engine (torch, TVM, tinyml_modelmaker and friends) is excluded via
# --exclude-module, driven by scripts/pyinstaller_excludes.txt, because mmcli calls out
# to it via the MMCLI_PYTHON subprocess and never needs it in-process. Only
# generic_audio_classification.zip (the one locally-authored example dataset, D-2 in
# 10-03-PLAN.md) is bundled into the binary; the other nine example datasets are
# fetched on demand from this project's own GitHub release mirror
# (github.com/musicalplatypus/tinyml-cli releases, tag datasets-<version>) via
# `mmcli datasets pull` rather than shipped. See scripts/binary_size_ceiling.txt for
# the enforced size bound. At runtime the binary calls out to an external Python
# interpreter via the MMCLI_PYTHON environment variable.
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

# mmcli drives the training engine through MMCLI_PYTHON as a subprocess; it never needs
# the engine in its own address space. Excluding it keeps the guarded
# `import tinyml_modelmaker` / `import tvm` probes as the no-ops their except branches
# already handle. numpy and pandas stay: analyze.py genuinely uses them. The exclude
# list is shared across all three build scripts (scripts/pyinstaller_excludes.txt) so
# it cannot drift between platforms.
EXCLUDE_ARGS=()
while IFS= read -r m; do
    [ -n "$m" ] && EXCLUDE_ARGS+=(--exclude-module "$m")
done < "${SCRIPT_DIR}/scripts/pyinstaller_excludes.txt"

# Explicit bundling allowlist (10-03-PLAN.md D-2/T-10-03-02): this script previously
# shipped zero datasets (no --add-data at all), leaving generic_audio_classification
# unreachable on Linux — this staging fixes REQ-DATA-04 here. Stage exactly the one
# locally-authored dataset into a fresh temp directory and --add-data *that*, so the
# bundled set is a property of this script, not of whatever zips sit in the
# developer's working tree. The stage dir is removed on EXIT so a failed build leaves
# nothing behind.
BUNDLED_DATASETS=(generic_audio_classification.zip)
DATASET_STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "${DATASET_STAGE_DIR}"' EXIT
for f in "${BUNDLED_DATASETS[@]}"; do
    cp "${SCRIPT_DIR}/mmcli/example_datasets/${f}" "${DATASET_STAGE_DIR}/"
done

pyinstaller \
    --onefile \
    --name mmcli \
    --hidden-import mmcli \
    --hidden-import mmcli.builder \
    --hidden-import mmcli.cli \
    --add-data "${DATASET_STAGE_DIR}:mmcli/example_datasets" \
    "${EXCLUDE_ARGS[@]}" \
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
