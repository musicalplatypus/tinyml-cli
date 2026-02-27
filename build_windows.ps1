# build_windows.ps1 — Build a standalone Windows executable for mmcli using PyInstaller
#
# The binary is lightweight (~10 MB) because tinyml_modelmaker is NOT bundled.
# At runtime the binary calls out to an external Python interpreter via the
# MMCLI_PYTHON environment variable.
#
# Requirements (in the active venv):
#   pip install pyinstaller mmcli  (or pip install -e .)
#
# Output: dist\mmcli.exe  (single-file executable)
#
# Usage:
#   .\build_windows.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Building mmcli for Windows..."

# Ensure PyInstaller is available
try {
    python -c "import PyInstaller" 2>$null
} catch {
    Write-Host "Installing PyInstaller..."
    pip install pyinstaller
}

# Clean previous build artifacts
if (Test-Path "$ScriptDir\build") { Remove-Item -Recurse -Force "$ScriptDir\build" }
if (Test-Path "$ScriptDir\dist\mmcli.exe") { Remove-Item -Force "$ScriptDir\dist\mmcli.exe" }

pyinstaller `
    --onefile `
    --name mmcli `
    --hidden-import mmcli `
    --hidden-import mmcli.builder `
    --hidden-import mmcli.cli `
    "$ScriptDir\mmcli\__main__.py"

Write-Host ""
Write-Host "Build complete: $ScriptDir\dist\mmcli.exe"
Write-Host ""
Write-Host "Usage:"
Write-Host "  `$env:MMCLI_PYTHON = 'C:\path\to\venv\Scripts\python.exe'"
Write-Host "  .\dist\mmcli.exe --version"
Write-Host "  .\dist\mmcli.exe --help"
Write-Host "  .\dist\mmcli.exe train --help"
Write-Host "  .\dist\mmcli.exe --dry-run train -m timeseries -t generic_timeseries_classification ``"
Write-Host "      -d F28P55 -n CLS_1k_NPU -i .\data"
