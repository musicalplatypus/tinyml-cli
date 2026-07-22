"""Source-level regression guard over the PyInstaller exclude flags and the binary
size ceiling, shared by all three release build scripts.

This is a source-level assertion deliberately, not a build-and-measure test: building
the binary takes minutes per platform and needs PyInstaller and its per-target
toolchain, which is too heavy for the unit suite. The failure mode this guards
against is someone editing one of the three pyinstaller invocations and dropping the
exclude flags, or a stale/typo'd size ceiling silently loosening the gate — exactly
what shipped the original 260 MB binary (build_macos.sh asserted "~10 MB" while
excludes=[] in the generated spec enforced nothing).

The built-artifact size check itself belongs in the release CI job (10-08), not here.
"""
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXCLUDES_FILE = REPO_ROOT / "scripts" / "pyinstaller_excludes.txt"
CEILING_FILE = REPO_ROOT / "scripts" / "binary_size_ceiling.txt"

BUILD_SCRIPTS = [
    REPO_ROOT / "build_macos.sh",
    REPO_ROOT / "build_linux.sh",
    REPO_ROOT / "build_windows.ps1",
]

# 152043520 = 145 MiB, the interim ceiling while the dataset payload is still bundled.
# 15728640  = 15 MiB (REQ-SIZE-01), the final ceiling once 10-03 unbundles the datasets.
SANCTIONED_CEILINGS = (152043520, 15728640)

NEVER_EXCLUDE = ("numpy", "pandas")

EXPECTED_EXCLUDED_MODULES = {
    "torch",
    "torchvision",
    "torchaudio",
    "tinyml_modelmaker",
    "tinyml_tinyverse",
    "tinyml_torchmodelopt",
    "tinyml_modelzoo",
    "tvm",
    "matplotlib",
    "scipy",
    "sklearn",
    "onnx",
    "onnxruntime",
}


def _read_excludes_list():
    """Parse scripts/pyinstaller_excludes.txt: one module name per line."""
    lines = EXCLUDES_FILE.read_text().splitlines()
    return [line.strip() for line in lines if line.strip()]


def _strip_comment_lines(text):
    """Drop full-line '#' comments so substring checks aren't fooled by prose
    (e.g. a comment mentioning "numpy stays" must not read as an exclusion)."""
    return "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("#")
    )


def _script_reads_shared_list(text):
    """True if the script's (non-comment) source references the shared exclude-list
    file by name — i.e. it derives its flags from scripts/pyinstaller_excludes.txt
    rather than hardcoding a second copy."""
    return "pyinstaller_excludes" in _strip_comment_lines(text)


def _literal_exclude_modules(text):
    """Modules named by a literal '--exclude-module <name>' pair in the script text.

    Matches only the parsed flag/argument pair, not a bare substring, so prose in a
    comment ('excludes torch and TVM') can never be mistaken for an actual flag.
    """
    code = _strip_comment_lines(text)
    return set(re.findall(r'--exclude-module\s+"?\'?([A-Za-z0-9_.]+)"?\'?', code))


class TestExcludesList:
    """scripts/pyinstaller_excludes.txt is the single source of truth for what gets
    excluded from every shipped binary."""

    def test_excludes_file_is_non_empty(self):
        modules = _read_excludes_list()
        assert modules, "scripts/pyinstaller_excludes.txt must not be empty"

    def test_excludes_file_has_thirteen_expected_entries(self):
        modules = _read_excludes_list()
        assert set(modules) == EXPECTED_EXCLUDED_MODULES
        assert len(modules) == 13

    def test_excludes_file_never_lists_numpy_or_pandas(self):
        modules = set(_read_excludes_list())
        for never in NEVER_EXCLUDE:
            assert never not in modules, (
                f"{never} must never be excluded — analyze.py::_row_count imports it"
            )


class TestBuildScriptsExcludeEveryModule:
    """Every module in the shared list must be excluded by every one of the three
    shipped build scripts (Linux, Windows, macOS) — parametrised so a fourth build
    script means adding one path here, not writing a fourth test."""

    @pytest.mark.parametrize("script_path", BUILD_SCRIPTS, ids=lambda p: p.name)
    def test_script_excludes_every_shared_module(self, script_path):
        text = script_path.read_text()
        shared_modules = set(_read_excludes_list())

        if _script_reads_shared_list(text):
            # Reads the shared file at build time: every module in the file is
            # excluded by construction, provided the read actually feeds
            # --exclude-module flags into the pyinstaller invocation.
            assert "--exclude-module" in _strip_comment_lines(text), (
                f"{script_path.name} references the shared exclude list but never "
                f"builds --exclude-module flags from it"
            )
            return

        # Documented fallback: a script that cannot read the shared file must carry
        # a hardcoded copy equal to it.
        literal_modules = _literal_exclude_modules(text)
        missing = shared_modules - literal_modules
        assert not missing, (
            f"{script_path.name} hardcodes its own exclude list and is missing: "
            f"{sorted(missing)}"
        )

    @pytest.mark.parametrize("script_path", BUILD_SCRIPTS, ids=lambda p: p.name)
    def test_script_never_excludes_numpy_or_pandas(self, script_path):
        text = script_path.read_text()
        literal_modules = _literal_exclude_modules(text)
        for never in NEVER_EXCLUDE:
            assert never not in literal_modules, (
                f"{script_path.name} must not exclude {never} — analyze.py needs it"
            )


class TestBuildScriptsReadSharedList:
    """Each script must actually reference the shared list by name, so a silent
    revert to a hardcoded array is caught even if that array happens to still be
    complete today."""

    @pytest.mark.parametrize("script_path", BUILD_SCRIPTS, ids=lambda p: p.name)
    def test_script_references_shared_exclude_file(self, script_path):
        text = script_path.read_text()
        assert _script_reads_shared_list(text), (
            f"{script_path.name} does not reference scripts/pyinstaller_excludes.txt "
            f"— it may have reverted to a hardcoded exclude list"
        )


def test_build_windows_is_named_in_this_file():
    """Regression guard for checker finding B-1: build_windows.ps1 was entirely
    unaddressed by the original build fix, which only touched build_macos.sh. This
    test file must always name all three scripts, not silently drop the odd one out."""
    this_file_text = Path(__file__).read_text()
    assert re.search(r"build_windows\.ps1", this_file_text)
    assert re.search(r"build_linux\.sh", this_file_text)
    assert re.search(r"build_macos\.sh", this_file_text)


class TestBinarySizeCeiling:
    """scripts/binary_size_ceiling.txt is the single source of truth for the CI size
    gate — a number repeated across the CI job, the test and the build script is a
    number that will disagree with itself (checker finding F-1)."""

    def test_ceiling_parses_as_positive_integer(self):
        raw = CEILING_FILE.read_text().strip()
        value = int(raw)
        assert value > 0

    def test_ceiling_is_a_sanctioned_value(self):
        value = int(CEILING_FILE.read_text().strip())
        assert value in SANCTIONED_CEILINGS, (
            f"{value} is not one of the sanctioned ceilings {SANCTIONED_CEILINGS} — "
            f"a typo here would loosen the size gate by an order of magnitude"
        )
