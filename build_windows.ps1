# build_windows.ps1 — Build a standalone Windows executable for mmcli using PyInstaller
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

# mmcli drives the training engine through MMCLI_PYTHON as a subprocess; it never needs
# the engine in its own address space. Excluding it keeps the guarded
# `import tinyml_modelmaker` / `import tvm` probes as the no-ops their except branches
# already handle. numpy and pandas stay: analyze.py genuinely uses them. The exclude
# list is shared across all three build scripts (scripts/pyinstaller_excludes.txt) so
# it cannot drift between platforms. Splatted via @ExcludeArgs rather than interpolated
# as "$ExcludeArgs" — PowerShell stringifies an interpolated array into one
# space-joined argument, which PyInstaller would receive as a single malformed flag.
$ExcludeFile = Join-Path (Join-Path $ScriptDir "scripts") "pyinstaller_excludes.txt"
$ExcludeArgs = Get-Content $ExcludeFile |
    Where-Object { $_.Trim() -ne "" } |
    ForEach-Object { '--exclude-module', $_ }

pyinstaller `
    --onefile `
    --name mmcli `
    --hidden-import mmcli `
    --hidden-import mmcli.builder `
    --hidden-import mmcli.cli `
    @ExcludeArgs `
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
