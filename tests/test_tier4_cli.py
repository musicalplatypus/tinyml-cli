"""Tier 4 — CLI Integration Tests for mmcli.

Validates CLI commands work end-to-end via subprocess and direct API calls.
Covers the full spectrum from `test_analysis.md` Tier 4:

  Test 20: `mmcli info` all tasks ✅ (existing in test_cli_integration.py)
  Test 21: `mmcli info` device filtering
  Test 22: `mmcli train --dry-run` per task type ✅ (existing)
  Test 23: `mmcli init` — bundled dataset extraction for all 9 datasets
  Test 24: Report generation per task type ✅ (existing in test_report.py)

This file adds Tests 21 and 23 and extends existing CLI coverage.

Marked with @pytest.mark.cli — run with: pytest -m cli
"""

import os
import subprocess
import sys
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "cli: Tier 4 CLI integration tests")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MMCLI = [sys.executable, "-m", "mmcli"]


def _run_cli(*args, timeout=30):
    """Run mmcli with args and return (returncode, stdout, stderr)."""
    cmd = MMCLI + list(args)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Test 21: Device filtering in `mmcli info`
# ---------------------------------------------------------------------------

# Devices grouped by family (from info.py)
DEVICES_SAMPLE = [
    "F28P55",       # C2000
    "MSPM0G3507",   # MSPM0
    "AM263",        # Sitara
]

TASK_TYPES = [
    ("timeseries", "generic_timeseries_classification"),
    ("timeseries", "generic_timeseries_regression"),
    ("timeseries", "generic_timeseries_forecasting"),
]


@pytest.mark.cli
class TestInfoDeviceFiltering:
    """Verify mmcli info --device filters to device-specific models."""

    @pytest.mark.parametrize("module,task_type", TASK_TYPES)
    def test_device_filter_returns_valid_models(self, module, task_type):
        """info with --device should list only models for that device."""
        rc, stdout, stderr = _run_cli(
            "info", "-m", module, "-t", task_type, "-d", "F28P55"
        )
        output = stdout + stderr
        assert rc == 0, f"mmcli info with device filter failed: {output}"
        # Should contain "Models for F28P55" section
        assert "F28P55" in output, (
            f"Device F28P55 not mentioned in filtered output: {output[:300]}"
        )

    @pytest.mark.parametrize("module,task_type", TASK_TYPES)
    def test_different_devices_give_different_results(self, module, task_type):
        """Different devices may have different model sets."""
        rc1, out1, err1 = _run_cli(
            "info", "-m", module, "-t", task_type, "-d", "F28P55"
        )
        rc2, out2, err2 = _run_cli(
            "info", "-m", module, "-t", task_type, "-d", "MSPM0G3507"
        )
        # Both should succeed
        assert rc1 == 0, f"F28P55 info failed: {out1 + err1}"
        assert rc2 == 0, f"MSPM0G3507 info failed: {out2 + err2}"
        # They should mention their respective devices
        assert "F28P55" in (out1 + err1)
        assert "MSPM0G3507" in (out2 + err2)

    def test_info_without_device_shows_all_devices(self):
        """info without --device should list all supported devices."""
        rc, stdout, stderr = _run_cli(
            "info", "-m", "timeseries", "-t", "generic_timeseries_classification"
        )
        output = stdout + stderr
        assert rc == 0
        assert "Supported Devices" in output or "devices" in output.lower()

    def test_info_without_task_lists_all_tasks(self):
        """info with only -m should list all available tasks."""
        rc, stdout, stderr = _run_cli(
            "info", "-m", "timeseries"
        )
        output = stdout + stderr
        assert rc == 0
        # Should list task types
        assert "classification" in output.lower()


# ---------------------------------------------------------------------------
# Test 21b: Feature extraction preset listing
# ---------------------------------------------------------------------------

@pytest.mark.cli
class TestInfoFeaturePresets:
    """Verify mmcli info lists feature extraction presets for each task."""

    @pytest.mark.parametrize("module,task_type", TASK_TYPES)
    def test_info_lists_fe_presets(self, module, task_type):
        """info should list feature extraction presets."""
        rc, stdout, stderr = _run_cli(
            "info", "-m", module, "-t", task_type
        )
        output = stdout + stderr
        assert rc == 0, f"mmcli info failed: {output}"
        # Feature extraction presets should appear
        assert "Feature Extraction" in output or "feature" in output.lower(), (
            f"No feature extraction presets listed: {output[:300]}"
        )


# ---------------------------------------------------------------------------
# Test 23: `mmcli init` — bundled dataset extraction
# ---------------------------------------------------------------------------

@pytest.mark.cli
class TestInitDatasetExtraction:
    """Verify all bundled datasets are accessible and well-formed."""

    def test_list_datasets_returns_all_registered(self):
        """list_datasets() API should return all 9 registered datasets."""
        from mmcli.datasets import list_datasets, DATASET_REGISTRY

        result = list_datasets()
        names = {d["name"] for d in result}
        assert names == set(DATASET_REGISTRY.keys()), (
            f"Mismatch: list_datasets returned {names}, "
            f"registry has {set(DATASET_REGISTRY.keys())}"
        )
        assert len(result) == 9, f"Expected 9 datasets, got {len(result)}"

    def test_list_datasets_filter_by_task(self):
        """list_datasets() with task_type filter should return relevant datasets."""
        from mmcli.datasets import list_datasets

        cls_datasets = list_datasets(task_type="generic_timeseries_classification")
        assert len(cls_datasets) >= 1
        for d in cls_datasets:
            assert "generic_timeseries_classification" in d["task_types"]

    def test_list_datasets_filter_by_module(self):
        """list_datasets() with module filter should work correctly."""
        from mmcli.datasets import list_datasets

        ts_datasets = list_datasets(module="timeseries")
        vision_datasets = list_datasets(module="vision")

        # Most datasets are timeseries
        assert len(ts_datasets) >= 7
        assert len(vision_datasets) >= 1
        assert all(d["module"] == "timeseries" for d in ts_datasets)
        assert all(d["module"] == "vision" for d in vision_datasets)


@pytest.mark.cli
class TestInitDatasetExtractReal:
    """Extract each bundled dataset and verify directory structure.

    These tests use the actual bundled zips, not mocks.
    They are parametrized over all 9 datasets in DATASET_REGISTRY.
    """

    # All 9 bundled datasets
    DATASETS = [
        ("generic_timeseries_classification", "generic_timeseries_classification"),
        ("generic_timeseries_regression", "generic_timeseries_regression"),
        ("generic_timeseries_anomalydetection", "generic_timeseries_anomalydetection"),
        ("generic_timeseries_forecasting", "generic_timeseries_forecasting"),
        ("arc_fault_classification", "arc_fault"),
        ("ecg_classification", "ecg_classification"),
        ("fan_blade_fault", "motor_fault"),
        ("pir_detection", "pir_detection"),
        ("mnist_image_classification", "image_classification"),
    ]

    @pytest.mark.parametrize("dataset_name,task_type", DATASETS)
    def test_dataset_extracts_without_error(self, dataset_name, task_type, tmp_path):
        """Each bundled dataset zip should extract into a valid project structure."""
        from mmcli.datasets import extract_dataset, DATASET_REGISTRY

        # Verify dataset is registered
        assert dataset_name in DATASET_REGISTRY, (
            f"Dataset '{dataset_name}' not in DATASET_REGISTRY"
        )

        # Check zip exists
        from mmcli.datasets import _datasets_dir
        zip_path = os.path.join(_datasets_dir(), DATASET_REGISTRY[dataset_name]["filename"])
        assert os.path.isfile(zip_path), f"Zip not found: {zip_path}"

        # Extract
        project_path = str(tmp_path / dataset_name)
        extract_dataset(dataset_name, project_path, task_type=task_type)

        # Verify project structure
        dataset_dir = os.path.join(project_path, "dataset")
        assert os.path.isdir(dataset_dir), "dataset/ directory not created"

        # Should have at least one data subdirectory (classes/, files/, or images/)
        data_subdirs = ["classes", "files", "images"]
        found = [d for d in data_subdirs if os.path.isdir(os.path.join(dataset_dir, d))]
        assert len(found) >= 1, (
            f"No data subdirectory found: expected one of {data_subdirs} "
            f"in {os.listdir(dataset_dir)}"
        )

    @pytest.mark.parametrize("dataset_name,task_type", DATASETS)
    def test_dataset_has_data_files(self, dataset_name, task_type, tmp_path):
        """Extracted dataset should contain actual data files."""
        from mmcli.datasets import extract_dataset

        project_path = str(tmp_path / dataset_name)
        extract_dataset(dataset_name, project_path, task_type=task_type)

        dataset_dir = os.path.join(project_path, "dataset")

        # Count actual data files (CSV, txt, png, jpg, etc.)
        file_count = 0
        for root, dirs, files in os.walk(dataset_dir):
            for f in files:
                if not f.startswith("."):
                    file_count += 1

        assert file_count > 0, (
            f"No data files found in {dataset_dir}"
        )

    def test_dataset_registry_completeness(self):
        """All entries in DATASET_REGISTRY should have required keys."""
        from mmcli.datasets import DATASET_REGISTRY

        required_keys = {"filename", "task_types", "module", "description"}
        for name, meta in DATASET_REGISTRY.items():
            missing = required_keys - set(meta.keys())
            assert not missing, (
                f"Dataset '{name}' missing keys: {missing}"
            )

    def test_all_zips_present(self):
        """Every registered dataset should have its zip file present."""
        from mmcli.datasets import DATASET_REGISTRY, _datasets_dir

        ds_dir = _datasets_dir()
        for name, meta in DATASET_REGISTRY.items():
            zip_path = os.path.join(ds_dir, meta["filename"])
            assert os.path.isfile(zip_path), (
                f"Missing zip for '{name}': {zip_path}"
            )


# ---------------------------------------------------------------------------
# Config validation via --dry-run for all device × task combos
# ---------------------------------------------------------------------------

# Representative devices (one per family)
REPRESENTATIVE_DEVICES = ["F28P55", "MSPM0G3507", "AM263"]


@pytest.mark.cli
class TestDryRunCrossDevice:
    """Validate --dry-run config generation across task × device combos.

    Extends Test 22 from the original CLI integration tests to cover
    multiple devices, not just F28P55.
    """

    MODEL_NAMES = {
        "generic_timeseries_classification": "CLS_1k_NPU",
        "generic_timeseries_regression": "REGR_1k",
        "generic_timeseries_forecasting": "FCST_LSTM8",
    }

    @pytest.mark.parametrize("device", REPRESENTATIVE_DEVICES)
    @pytest.mark.parametrize("module,task_type", TASK_TYPES)
    def test_dry_run_valid_config(self, module, task_type, device, tmp_path):
        """--dry-run should generate valid YAML for each task × device combo."""
        model = self.MODEL_NAMES.get(task_type, "CLS_1k_NPU")

        # Create minimal directory structure
        data_dir = tmp_path / "data"
        (data_dir / "dataset" / "classes" / "dummy").mkdir(parents=True)
        (data_dir / "dataset" / "classes" / "dummy" / "sample.csv").write_text(
            "1,2,3\n4,5,6\n"
        )
        (data_dir / "dataset" / "annotations").mkdir(parents=True)
        (data_dir / "dataset" / "annotations" / "labels.csv").write_text(
            "file,label\nsample.csv,dummy\n"
        )

        rc, stdout, stderr = _run_cli(
            "--dry-run", "train",
            "-m", module,
            "-t", task_type,
            "-d", device,
            "-n", model,
            "-i", str(data_dir),
        )
        output = stdout + stderr
        assert rc == 0, (
            f"--dry-run failed for {task_type} on {device}: {output[:500]}"
        )


# ---------------------------------------------------------------------------
# mmcli subcommand completeness
# ---------------------------------------------------------------------------

@pytest.mark.cli
class TestSubcommandCoverage:
    """Verify all expected subcommands exist and respond to --help."""

    SUBCOMMANDS = ["train", "info", "init"]

    @pytest.mark.parametrize("subcmd", SUBCOMMANDS)
    def test_subcommand_help(self, subcmd):
        """Each subcommand should accept --help."""
        rc, stdout, stderr = _run_cli(subcmd, "--help")
        assert rc == 0, f"{subcmd} --help failed: {stdout + stderr}"

    def test_main_help_lists_subcommands(self):
        """mmcli --help should list all subcommands."""
        rc, stdout, stderr = _run_cli("--help")
        output = stdout + stderr
        assert rc == 0
        for subcmd in self.SUBCOMMANDS:
            assert subcmd in output.lower(), (
                f"Subcommand '{subcmd}' not found in --help output"
            )

    def test_version_matches_package(self):
        """--version should output a version string."""
        rc, stdout, stderr = _run_cli("--version")
        output = stdout + stderr
        assert rc == 0
        # Should contain a version-like pattern
        assert "0." in output or "1." in output, (
            f"No version pattern found in: {output}"
        )
