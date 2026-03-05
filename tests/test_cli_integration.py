"""CLI Integration Tests for mmcli.

Validates that mmcli commands work correctly when invoked via subprocess,
testing the CLI interface that users actually interact with.
"""

import os
import subprocess
import sys
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MMCLI = [sys.executable, "-m", "mmcli"]

# Task types that mmcli info supports (registered in TASK_DESCRIPTIONS)
TASK_TYPES = [
    ("timeseries", "generic_timeseries_classification"),
    ("timeseries", "generic_timeseries_regression"),
    ("timeseries", "generic_timeseries_forecasting"),
]

# Task types not yet in mmcli TASK_DESCRIPTIONS (tested separately)
UNSUPPORTED_INFO_TASKS = [
    ("timeseries", "generic_timeseries_anomalydetection"),
]


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
# Tests
# ---------------------------------------------------------------------------


class TestVersion:
    """Verify mmcli --version works."""

    def test_version_output(self):
        rc, stdout, stderr = _run_cli("--version")
        output = stdout + stderr
        assert "mmcli" in output.lower(), f"Version output missing 'mmcli': {output}"
        assert rc == 0


class TestInfoCommand:
    """Verify mmcli info returns valid output for each task type."""

    @pytest.mark.parametrize("module,task_type", TASK_TYPES)
    def test_info_lists_models(self, module, task_type):
        """info command should list available models for each task."""
        rc, stdout, stderr = _run_cli(
            "info", "-m", module, "-t", task_type, "-d", "F28P55"
        )
        output = stdout + stderr
        assert rc == 0, f"mmcli info failed: {output}"
        assert "Models for F28P55" in output or "models" in output.lower(), (
            f"Info output doesn't list models: {output[:200]}"
        )

    @pytest.mark.parametrize("module,task_type", TASK_TYPES)
    def test_info_lists_devices(self, module, task_type):
        """info command should list supported devices."""
        rc, stdout, stderr = _run_cli("info", "-m", module, "-t", task_type)
        output = stdout + stderr
        assert rc == 0, f"mmcli info failed: {output}"
        assert "Supported Devices" in output or "devices" in output.lower(), (
            f"Info output doesn't list devices: {output[:200]}"
        )

    @pytest.mark.parametrize("module,task_type", UNSUPPORTED_INFO_TASKS)
    @pytest.mark.xfail(reason="Task type not yet registered in mmcli TASK_DESCRIPTIONS")
    def test_info_unsupported_task(self, module, task_type):
        """Anomaly detection not yet registered — expected to fail."""
        rc, stdout, stderr = _run_cli(
            "info", "-m", module, "-t", task_type, "-d", "F28P55"
        )
        assert rc == 0, f"mmcli info failed: {stdout + stderr}"


class TestDryRun:
    """Verify mmcli --dry-run train generates valid YAML configs."""

    @pytest.mark.parametrize("module,task_type", TASK_TYPES + UNSUPPORTED_INFO_TASKS)
    def test_dry_run_generates_config(self, module, task_type, tmp_path):
        """--dry-run train should print the YAML config without running."""
        # Pick a minimal model name based on task type
        model_names = {
            "generic_timeseries_classification": "CLS_1k_NPU",
            "generic_timeseries_regression": "REGR_1k",
            "generic_timeseries_anomalydetection": "AD_1k",
            "generic_timeseries_forecasting": "FCST_LSTM8",
        }
        model = model_names.get(task_type, "CLS_1k_NPU")

        # Create the full directory structure mmcli validates
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
            "-d", "F28P55",
            "-n", model,
            "-i", str(data_dir),
        )
        output = stdout + stderr
        assert rc == 0, f"mmcli --dry-run failed for {task_type}: {output}"


class TestHelpCommands:
    """Verify help text is accessible."""

    def test_main_help(self):
        rc, stdout, stderr = _run_cli("--help")
        output = stdout + stderr
        assert rc == 0
        assert "train" in output.lower()
        assert "info" in output.lower()

    def test_train_help(self):
        rc, stdout, stderr = _run_cli("train", "--help")
        output = stdout + stderr
        assert rc == 0
        assert "--model" in output or "-m" in output

    def test_info_help(self):
        rc, stdout, stderr = _run_cli("info", "--help")
        output = stdout + stderr
        assert rc == 0

    def test_init_help(self):
        rc, stdout, stderr = _run_cli("init", "--help")
        output = stdout + stderr
        assert rc == 0
