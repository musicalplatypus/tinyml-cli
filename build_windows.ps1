# build_windows.ps1 — Build a standalone Windows executable for mmcli using PyInstaller
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

# Explicit bundling allowlist (10-03-PLAN.md D-2/T-10-03-02): this script previously
# shipped zero datasets (no --add-data at all), leaving generic_audio_classification
# unreachable on Windows — this staging fixes REQ-DATA-04 here. Stage exactly the one
# locally-authored dataset into a fresh temp directory and --add-data *that*, so the
# bundled set is a property of this script, not of whatever zips sit in the
# developer's working tree. Windows --add-data uses ";" as the source/dest separator,
# not ":" — the POSIX form yields a silently empty bundle here, not an error. Splatted
# via @DataArgs (same technique as @ExcludeArgs) rather than interpolated into the
# backtick-continued command line, for the same stringification reason. The stage dir
# is removed in a finally block so a failed build leaves nothing behind.
$BundledDatasets = @("generic_audio_classification.zip")
$DatasetStageDir = Join-Path ([System.IO.Path]::GetTempPath()) ("mmcli-dataset-stage-" + [System.Guid]::NewGuid())
New-Item -ItemType Directory -Path $DatasetStageDir | Out-Null
try {
    foreach ($f in $BundledDatasets) {
        Copy-Item (Join-Path $ScriptDir "mmcli\example_datasets\$f") $DatasetStageDir
    }
    $DataArgs = @('--add-data', "$DatasetStageDir;mmcli/example_datasets")

    pyinstaller `
        --onefile `
        --name mmcli `
        --hidden-import mmcli `
        --hidden-import mmcli.builder `
        --hidden-import mmcli.cli `
        @ExcludeArgs `
        @DataArgs `
        "$ScriptDir\mmcli\__main__.py"
} finally {
    Remove-Item -Recurse -Force $DatasetStageDir
}

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
