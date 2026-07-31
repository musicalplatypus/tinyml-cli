"""Source-level regression guard over the PyInstaller exclude flags, the binary size
ceiling shared by all three release build scripts, and the Python distribution's
`package-data` allowlist.

This is a source-level assertion deliberately, not a build-and-measure test: building
the binary takes minutes per platform and needs PyInstaller and its per-target
toolchain, which is too heavy for the unit suite. The failure mode this guards
against is someone editing one of the three pyinstaller invocations and dropping the
exclude flags, or a stale/typo'd size ceiling silently loosening the gate — exactly
what shipped the original 260 MB binary (build_macos.sh asserted "~10 MB" while
excludes=[] in the generated spec enforced nothing).

The built-artifact size check itself belongs in the release CI job (10-08), not here.
"""
import fnmatch
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXCLUDES_FILE = REPO_ROOT / "scripts" / "pyinstaller_excludes.txt"
CEILING_FILE = REPO_ROOT / "scripts" / "binary_size_ceiling.txt"
PYPROJECT_FILE = REPO_ROOT / "pyproject.toml"

BUILD_SCRIPTS = [
    REPO_ROOT / "build_macos.sh",
    REPO_ROOT / "build_linux.sh",
    REPO_ROOT / "build_windows.ps1",
]

# 152043520 = 145 MiB, the interim ceiling while the dataset payload is still bundled.
# 15728640  = 15 MiB (REQ-SIZE-01), the final ceiling once 10-03 unbundles the datasets.
SANCTIONED_CEILINGS = (152043520, 15728640)

NEVER_EXCLUDE = ("numpy", "pandas")

# 10-03-PLAN.md D-2: only this one locally-authored dataset is bundled into the
# binary on every platform; the other nine (below) are fetched on demand from the
# GitHub release mirror and must never be named by a script's --add-data flag.
BUNDLED_DATASET_FILENAME = "generic_audio_classification.zip"
MIRRORED_DATASET_FILENAMES = {
    "arc_fault_classification.zip",
    "ecg_classification.zip",
    "fan_blade_fault.zip",
    "generic_timeseries_anomalydetection.zip",
    "generic_timeseries_classification.zip",
    "generic_timeseries_forecasting.zip",
    "generic_timeseries_regression.zip",
    "mnist_image_classification.zip",
    "pir_detection.zip",
}
DATASET_DEST = "mmcli/example_datasets"

# Separator PyInstaller's --add-data expects between the source and destination
# halves of its argument: ":" on macOS/Linux, ";" on Windows. Getting this wrong on
# Windows silently produces an empty bundle rather than an error (T-10-03-05).
EXPECTED_ADD_DATA_SEPARATOR = {
    "build_macos.sh": ":",
    "build_linux.sh": ":",
    "build_windows.ps1": ";",
}

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


def _package_data_patterns():
    """The patterns in pyproject.toml's `[tool.setuptools.package-data]` `mmcli` array.

    Parsed by regex over the raw text rather than with `tomllib`: both workflows pin
    python 3.10, where tomllib is not in the stdlib, and neither installs tomli. Adding
    a parser dependency for one assertion is the call 10-08 declined to make for PyYAML.

    Raises rather than returning [] when the section or array is missing, so a
    restructured pyproject.toml fails loudly. An empty return would silently satisfy
    every "no mirrored dataset matches" assertion downstream — a guard that cannot fail.
    """
    text = _strip_comment_lines(PYPROJECT_FILE.read_text())
    section = re.search(
        r"^\[tool\.setuptools\.package-data\]\s*$(.*?)(?=^\[|\Z)",
        text, re.MULTILINE | re.DOTALL,
    )
    assert section, (
        "no [tool.setuptools.package-data] section in pyproject.toml — either it was "
        "removed (the wheel then ships no datasets at all) or renamed, and this guard "
        "can no longer see what is packaged"
    )
    array = re.search(r"\bmmcli\s*=\s*\[(.*?)\]", section.group(1), re.DOTALL)
    assert array, (
        "no `mmcli = [...]` array under [tool.setuptools.package-data] — this guard "
        "cannot verify what the wheel and sdist ship"
    )
    return re.findall(r"['\"]([^'\"]+)['\"]", array.group(1))


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


def _add_data_values(text):
    """Return the raw quoted argument value(s) that follow a literal --add-data flag.

    Matches two shapes across the three scripts:
      - bash form:        --add-data "SRC:DST"
      - PowerShell splat:  @('--add-data', "SRC;DST")   (comma between flag and value,
                            since the flag and its value are two separate array
                            elements rather than one whitespace-joined token)
    Matching the parsed argument, not a bare substring, so a comment or an unrelated
    string can never be mistaken for an actual --add-data value.
    """
    code = _strip_comment_lines(text)
    bash_style = re.findall(r'--add-data\s+["\']([^"\']+)["\']', code)
    ps_style = re.findall(r"--add-data['\"]?\s*,\s*[\"']([^\"']+)[\"']", code)
    return bash_style + ps_style


def _bundled_dataset_allowlist(text):
    """Return the set of *.zip filenames literally present in the script's
    BUNDLED_DATASETS (bash) / $BundledDatasets (PowerShell) allowlist array."""
    code = _strip_comment_lines(text)
    match = re.search(r"BUNDLED_DATASETS=\(([^)]*)\)", code) or re.search(
        r"\$BundledDatasets\s*=\s*@\(([^)]*)\)", code
    )
    if not match:
        return set()
    return set(re.findall(r"([A-Za-z0-9_]+\.zip)", match.group(1)))


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


class TestBuildScriptsBundleOnlyTheOneLocalDataset:
    """10-03-PLAN.md D-2/T-10-03-02/T-10-03-05: every one of the three build scripts
    must stage exactly generic_audio_classification.zip into the shipped bundle via a
    single --add-data argument, with the platform-correct separator, and must never
    name any of the nine mirrored datasets in that flag. Parametrised so a fourth
    build script means adding one path to BUILD_SCRIPTS, not writing a fourth test."""

    @pytest.mark.parametrize("script_path", BUILD_SCRIPTS, ids=lambda p: p.name)
    def test_script_has_exactly_one_add_data_argument(self, script_path):
        text = script_path.read_text()
        values = _add_data_values(text)
        assert len(values) == 1, (
            f"{script_path.name} must pass exactly one --add-data argument, found "
            f"{len(values)}: {values}"
        )

    @pytest.mark.parametrize("script_path", BUILD_SCRIPTS, ids=lambda p: p.name)
    def test_script_add_data_uses_platform_separator_and_correct_destination(
        self, script_path
    ):
        text = script_path.read_text()
        [value] = _add_data_values(text)
        sep = EXPECTED_ADD_DATA_SEPARATOR[script_path.name]
        assert sep in value, (
            f"{script_path.name}'s --add-data value {value!r} does not contain the "
            f"platform-correct separator {sep!r}"
        )
        source, _, dest = value.rpartition(sep)
        assert source, f"{script_path.name}'s --add-data has no source half: {value!r}"
        assert dest == DATASET_DEST, (
            f"{script_path.name}'s --add-data destination is {dest!r}, expected "
            f"{DATASET_DEST!r} (mmcli._datasets_dir() resolves this exact path)"
        )

    @pytest.mark.parametrize("script_path", BUILD_SCRIPTS, ids=lambda p: p.name)
    def test_script_add_data_source_is_the_staging_variable(self, script_path):
        # The source half of --add-data is a staging variable (not a literal
        # filename), since the bundled set is staged into a temp dir rather than
        # filtered from mmcli/example_datasets in place. Per plan D-2's parenthetical:
        # when the source is a staging variable, the BUNDLED_DATASETS allowlist
        # literal must appear in the script and must name exactly the one bundled
        # dataset.
        text = script_path.read_text()
        [value] = _add_data_values(text)
        sep = EXPECTED_ADD_DATA_SEPARATOR[script_path.name]
        source, _, _dest = value.rpartition(sep)
        if BUNDLED_DATASET_FILENAME in source:
            return  # source names the file directly — no allowlist needed
        allowlist = _bundled_dataset_allowlist(text)
        assert allowlist == {BUNDLED_DATASET_FILENAME}, (
            f"{script_path.name}'s --add-data source {source!r} is a staging "
            f"variable, but its BUNDLED_DATASETS allowlist is {allowlist!r}, "
            f"expected exactly {{{BUNDLED_DATASET_FILENAME!r}}}"
        )

    @pytest.mark.parametrize("script_path", BUILD_SCRIPTS, ids=lambda p: p.name)
    def test_script_add_data_never_names_a_mirrored_dataset(self, script_path):
        text = script_path.read_text()
        [value] = _add_data_values(text)
        named = MIRRORED_DATASET_FILENAMES & set(re.findall(r"[A-Za-z0-9_]+\.zip", value))
        assert not named, (
            f"{script_path.name}'s --add-data names mirrored dataset(s) {named} — "
            f"only {BUNDLED_DATASET_FILENAME} may ever be bundled"
        )

    @pytest.mark.parametrize("script_path", BUILD_SCRIPTS, ids=lambda p: p.name)
    def test_script_never_stages_a_mirrored_dataset_in_its_allowlist(self, script_path):
        text = script_path.read_text()
        allowlist = _bundled_dataset_allowlist(text)
        named = MIRRORED_DATASET_FILENAMES & allowlist
        assert not named, (
            f"{script_path.name}'s BUNDLED_DATASETS allowlist stages mirrored "
            f"dataset(s) {named} — only {BUNDLED_DATASET_FILENAME} may be staged"
        )


class TestPackageDataBundlesOnlyTheOneLocalDataset:
    """The packaging-channel sibling of the class above.

    The build scripts stop the PyInstaller binary shipping the nine mirrored datasets;
    `[tool.setuptools.package-data]` is the same decision for the wheel and the sdist,
    and it was missed when the binary was unbundled. A wheel built from a working tree
    holding all ten measured 108.2 MB (REQ-SIZE-03).

    Deliberately a source-level check rather than a build-and-measure one, and here
    that is the *stronger* choice, not the cheaper one: `.gitignore` ignores
    `mmcli/example_datasets/*.zip` and only the audio zip is tracked, so a CI checkout
    holds one dataset and a re-added wildcard would still produce a small artifact
    there. Measuring the built wheel in CI would pass vacuously.
    """

    def test_package_data_matches_exactly_the_one_bundled_dataset(self):
        patterns = _package_data_patterns()
        # fnmatch is used rather than string equality so an equivalent-but-differently
        # written glob is judged on what setuptools would actually select. Note fnmatch
        # lets `*` cross `/` while setuptools' glob does not, which makes this strictly
        # more willing to report a match — it can over-report a mirrored dataset, never
        # miss one. That is the safe direction for a guard.
        bundled = f"example_datasets/{BUNDLED_DATASET_FILENAME}"
        assert any(fnmatch.fnmatchcase(bundled, p) for p in patterns), (
            f"package-data no longer ships {BUNDLED_DATASET_FILENAME}; it is the one "
            f"dataset with no mirror asset, so a pip install can never obtain it "
            f"(patterns: {patterns})"
        )
        for name in sorted(MIRRORED_DATASET_FILENAMES):
            target = f"example_datasets/{name}"
            matched = [p for p in patterns if fnmatch.fnmatchcase(target, p)]
            assert not matched, (
                f"package-data pattern(s) {matched} ship {name} inside the wheel/sdist. "
                f"It is fetched from the release mirror on demand; packaging it re-adds "
                f"the payload REQ-SIZE-03 removed"
            )

    def test_package_data_names_no_wildcard_over_example_datasets(self):
        # Belt to the above's braces, and the assertion that survives if the ten-name
        # list ever drifts: it names the actual mistake ("a wildcard came back") instead
        # of reporting ten fnmatch results.
        offenders = [
            p for p in _package_data_patterns()
            if p.startswith("example_datasets/") and any(c in p for c in "*?[")
        ]
        assert not offenders, (
            f"package-data uses wildcard(s) {offenders} over example_datasets/. List the "
            f"one bundled dataset literally — a wildcard silently re-ships every zip that "
            f"happens to be in the working tree at build time"
        )

    def test_package_data_still_ships_the_data_yaml_glob(self):
        # Over-narrowing is the live risk of this change: mmcli/data/schema.yaml is
        # unrelated to datasets, is tracked in git, and dropping it is a runtime
        # regression rather than a size win.
        patterns = _package_data_patterns()
        assert any(fnmatch.fnmatchcase("data/schema.yaml", p) for p in patterns), (
            f"package-data no longer ships mmcli/data/*.yaml (patterns: {patterns}) — "
            f"narrowing the dataset glob must not narrow the yaml glob"
        )

    def test_guard_constants_still_match_the_dataset_registry(self):
        """Protects both this class and TestBuildScriptsBundleOnlyTheOneLocalDataset.

        Both key off the same two constants, so a dataset added to the registry but not
        to them would be unguarded in the binary *and* the wheel simultaneously.
        """
        from mmcli.datasets import DATASET_REGISTRY

        mirrored = {e["filename"] for e in DATASET_REGISTRY.values() if e.get("ti_name")}
        bundled = {e["filename"] for e in DATASET_REGISTRY.values() if not e.get("ti_name")}
        assert mirrored == MIRRORED_DATASET_FILENAMES, (
            f"DATASET_REGISTRY's fetchable datasets {mirrored} no longer match "
            f"MIRRORED_DATASET_FILENAMES {MIRRORED_DATASET_FILENAMES}. Update the constant, "
            f"or a newly added dataset ships unguarded in both the binary and the wheel"
        )
        assert bundled == {BUNDLED_DATASET_FILENAME}, (
            f"DATASET_REGISTRY's non-fetchable datasets {bundled} no longer match "
            f"{{{BUNDLED_DATASET_FILENAME!r}}} — the set of datasets that must be packaged "
            f"has changed"
        )

    def test_no_manifest_in_reintroduces_the_mirrored_datasets(self):
        # No MANIFEST.in exists today, and setuptools derives sdist package data from
        # package-data, so narrowing it covers both artifacts. A MANIFEST.in could
        # re-include the zips in the sdist behind package-data's back; this keeps adding
        # one a deliberate act.
        manifest = REPO_ROOT / "MANIFEST.in"
        if not manifest.exists():
            return
        for line in _strip_comment_lines(manifest.read_text()).splitlines():
            directive = line.strip().lower()
            if not directive.startswith(("include", "graft", "recursive-include")):
                continue
            assert "example_datasets" not in directive and ".zip" not in directive, (
                f"MANIFEST.in line {line.strip()!r} re-includes dataset zips in the sdist, "
                f"bypassing the package-data allowlist (REQ-SIZE-03)"
            )


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
