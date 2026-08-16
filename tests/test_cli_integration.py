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
    ("timeseries", "generic_timeseries_anomalydetection"),
]


# 120s, not 30: these shell out to `mmcli info`, which imports the training
# engine (torch and friends) in a subprocess. That takes ~8s warm on a developer
# machine but far longer on a cold hosted runner — macOS timed out at 30s while
# ubuntu passed. The tests assert what the command prints, not how fast it runs,
# so the bound only needs to be generous enough not to fail for being slow.
def _run_cli(*args, timeout=120):
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
    def test_info_lists_models(self, module, task_type, mock_tinyml_modelmaker_registry):
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
    def test_info_lists_devices(self, module, task_type, mock_tinyml_modelmaker_registry):
        """info command should list supported devices."""
        rc, stdout, stderr = _run_cli("info", "-m", module, "-t", task_type)
        output = stdout + stderr
        assert rc == 0, f"mmcli info failed: {output}"
        assert "Supported Devices" in output or "devices" in output.lower(), (
            f"Info output doesn't list devices: {output[:200]}"
        )


class TestDryRun:
    """Verify mmcli --dry-run train generates valid YAML configs."""

    @pytest.mark.parametrize("module,task_type", TASK_TYPES)
    def test_dry_run_generates_config(self, module, task_type, tmp_path, mock_tinyml_modelmaker_registry):
        """--dry-run train should print the YAML config without running.

        generic_timeseries_forecasting and generic_timeseries_anomalydetection
        are excepted below: both have zero feature-extraction presets in the
        upstream catalog (finding F-2, .planning/FINDINGS-training-matrix.md),
        so no dataset shape could ever make their dry-run succeed.
        generic_timeseries_regression is also excepted: it has presets, but the
        only one that produces features requires an 11-channel input (finding
        F-9), and this fixture's dataset detects as 2 channels, so it can never
        match. mmcli's response to each — a specific, actionable error naming
        the catalog gap — is correct and is pinned as such instead of
        asserting a success that cannot happen. Revisit these branches once
        upstream ships the missing presets.
        """
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

        if task_type in (
            "generic_timeseries_forecasting",
            "generic_timeseries_anomalydetection",
        ):
            # F-2: zero feature-extraction presets upstream — dry-run cannot
            # succeed. Assert the *specific* upstream-gap error rather than a
            # bare non-zero exit, so a crash, typo, or unrelated regression
            # here still fails the test.
            assert rc != 0, (
                f"expected --dry-run to fail for {task_type} "
                f"(F-2: zero feature-extraction presets upstream) but it "
                f"succeeded: {output[:500]}"
            )
            assert "no feature-extraction presets available" in output, (
                f"expected the F-2 upstream-gap message for {task_type}, "
                f"got: {output[:500]}"
            )
            assert "gap in the upstream preset catalog" in output, (
                f"expected the F-2 upstream-gap message for {task_type}, "
                f"got: {output[:500]}"
            )
            return

        if task_type == "generic_timeseries_regression":
            # F-9: presets exist, but the only usable one requires 11 input
            # channels — no realistic dataset (including this fixture) can
            # match. Assert the specific channel-mismatch error rather than
            # a bare non-zero exit.
            assert rc != 0, (
                f"expected --dry-run to fail for {task_type} "
                f"(F-9: no usable preset matches this channel count) but it "
                f"succeeded: {output[:500]}"
            )
            assert "No usable feature-extraction preset for task" in output, (
                f"expected the F-9 channel-mismatch message for {task_type}, "
                f"got: {output[:500]}"
            )
            assert "input channel(s) detected in your dataset" in output, (
                f"expected the F-9 channel-mismatch message for {task_type}, "
                f"got: {output[:500]}"
            )
            return

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
