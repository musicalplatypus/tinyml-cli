"""Tests for --dataset-preset flag and mmcli info dataset preset listing."""
import subprocess
import sys
import pytest
from argparse import Namespace
from mmcli.builder import build_config


class TestDatasetPresetFlag:
    def test_flag_in_train_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "mmcli", "train", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--dataset-preset" in result.stdout

    def test_flag_in_run_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "mmcli", "run", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--dataset-preset" in result.stdout

    def test_flag_absent_from_compile_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "mmcli", "compile", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--dataset-preset" not in result.stdout


class TestDatasetPresetBuilderWiring:
    """Verify build_config routes --dataset-preset into config['dataset']['dataset_name']."""

    def _make_args(self, **kwargs):
        defaults = dict(
            command="train",
            module="timeseries",
            task="motor_fault",
            device="F28P55",
            model="CLS_1k_NPU",
            config=None,
            run_name=None,
            project="data/projects/default",
            feature_extraction=None,
            dataset_preset=None,
            epochs=None,
            batch_size=None,
            lr=None,
            training_device="cpu",
            gpus=None,
            quantization=None,
            auto_quantization=None,
            autoquant_tolerance_classification=None,
            autoquant_tolerance_regression=None,
            autoquant_tolerance_forecasting=None,
            autoquant_tolerance_anomaly=None,
            compile_model=None,
            native_amp=None,
            nas_size=None,
            nas_epochs=None,
            nas_optimize=None,
            onnx=None,
            preset=None,
            report=False,
        )
        defaults.update(kwargs)
        return Namespace(**defaults)

    def test_preset_none_uses_project_basename(self):
        """Omitting --dataset-preset leaves dataset_name as basename(project_dir)."""
        args = self._make_args(dataset_preset=None, project="data/projects/default")
        config = build_config(args)
        assert config["dataset"]["dataset_name"] == "default"

    def test_preset_name_propagates(self):
        args = self._make_args(dataset_preset="motor_fault_sample")
        config = build_config(args)
        assert config["dataset"]["dataset_name"] == "motor_fault_sample"

    def test_explicit_default_is_harmless(self):
        args = self._make_args(dataset_preset="default")
        config = build_config(args)
        assert config["dataset"]["dataset_name"] == "default"

    def test_preset_overrides_project_basename(self):
        """--dataset-preset takes precedence over project dir basename."""
        args = self._make_args(
            dataset_preset="motor_fault_sample",
            project="data/projects/my_project",
        )
        config = build_config(args)
        assert config["dataset"]["dataset_name"] == "motor_fault_sample"
