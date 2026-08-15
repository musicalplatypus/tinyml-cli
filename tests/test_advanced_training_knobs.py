"""Tests for --nn-feature-extraction, --gof-test, and --quantization-mode flags."""
import subprocess
import sys
import pytest
from argparse import Namespace
from mmcli.builder import build_config


def _make_args(**kwargs):
    defaults = dict(
        command="train",
        module="timeseries",
        task="generic_timeseries_classification",
        device="F28P55",
        model="CLS_1k_NPU",
        config=None,
        run_name=None,
        project="data/projects/default",
        feature_extraction=None,
        dataset_preset=None,
        nn_feature_extraction=False,
        gof_test=False,
        epochs=None,
        batch_size=None,
        lr=None,
        training_device="cpu",
        gpus=None,
        quantization=None,
        quantization_mode=None,
        run_quant_train_only=None,
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


class TestNNFeatureExtractionFlag:
    def test_flag_in_train_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "mmcli", "train", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--nn-feature-extraction" in result.stdout

    def test_flag_in_run_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "mmcli", "run", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--nn-feature-extraction" in result.stdout

    def test_false_by_default_key_absent(self):
        args = _make_args()
        config = build_config(args)
        fe = config.get("data_processing_feature_extraction", {})
        assert "nn_for_feature_extraction" not in fe

    def test_true_when_flag_set(self):
        args = _make_args(nn_feature_extraction=True)
        config = build_config(args)
        assert config["data_processing_feature_extraction"]["nn_for_feature_extraction"] is True


class TestGofTestFlag:
    def test_flag_in_train_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "mmcli", "train", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--gof-test" in result.stdout

    def test_false_by_default_key_absent(self):
        args = _make_args()
        config = build_config(args)
        fe = config.get("data_processing_feature_extraction", {})
        assert "gof_test" not in fe

    def test_true_when_flag_set(self):
        args = _make_args(gof_test=True)
        config = build_config(args)
        assert config["data_processing_feature_extraction"]["gof_test"] is True

    def test_independent_of_nn_fe(self):
        args = _make_args(gof_test=True, nn_feature_extraction=False)
        config = build_config(args)
        fe = config["data_processing_feature_extraction"]
        assert fe["gof_test"] is True
        assert "nn_for_feature_extraction" not in fe


class TestQuantizationModeFlag:
    def test_flag_in_train_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "mmcli", "train", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--quantization-mode" in result.stdout

    def test_none_by_default_key_absent(self):
        args = _make_args()
        config = build_config(args)
        assert "quantization_mode" not in config.get("training", {})

    def test_ptq_mode_propagates(self):
        args = _make_args(quantization_mode="ptq")
        config = build_config(args)
        assert config["training"]["quantization_mode"] == "ptq"

    def test_qat_mode_propagates(self):
        args = _make_args(quantization_mode="qat")
        config = build_config(args)
        assert config["training"]["quantization_mode"] == "qat"

    def test_invalid_mode_rejected(self):
        result = subprocess.run(
            [sys.executable, "-m", "mmcli", "train",
             "-m", "timeseries", "-t", "motor_fault",
             "-d", "F28P55", "--model", "CLS_1k_NPU",
             "--quantization-mode", "invalid"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "invalid choice" in result.stderr or "error" in result.stderr.lower()


class TestQuantTrainOnlyFlag:
    """--quant-train-only: skip float training, run only the quantisation
    training pass. Modelmaker raises ValueError when quantization is
    NO_QUANTIZATION; mmcli must refuse that combination itself, at
    argument-parse time, naming both flags."""

    def test_flag_in_train_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "mmcli", "train", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--quant-train-only" in result.stdout

    def test_none_by_default_key_absent(self):
        args = _make_args()
        config = build_config(args)
        assert "run_quant_train_only" not in config.get("training", {})

    def test_true_when_flag_set(self):
        args = _make_args(run_quant_train_only=True, quantization="QUANTIZATION_TINPU")
        config = build_config(args)
        assert config["training"]["run_quant_train_only"] is True

    def test_rejected_with_no_quantization_explicit(self):
        result = subprocess.run(
            [sys.executable, "-m", "mmcli", "train",
             "-m", "timeseries", "-t", "motor_fault",
             "-d", "F28P55", "--model", "CLS_1k_NPU",
             "--quantization", "NO_QUANTIZATION",
             "--quant-train-only"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "--quant-train-only" in result.stderr
        assert "--quantization" in result.stderr

    def test_rejected_with_quantization_omitted(self):
        # --quantization omitted defaults to NO_QUANTIZATION downstream in
        # modelmaker, so the precondition must also be enforced here.
        result = subprocess.run(
            [sys.executable, "-m", "mmcli", "train",
             "-m", "timeseries", "-t", "motor_fault",
             "-d", "F28P55", "--model", "CLS_1k_NPU",
             "--quant-train-only"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "--quant-train-only" in result.stderr
        assert "--quantization" in result.stderr

    def test_accepted_with_quantization_tinpu(self, tmp_path):
        data_dir = tmp_path / "data"
        (data_dir / "dataset" / "classes" / "dummy").mkdir(parents=True)
        (data_dir / "dataset" / "classes" / "dummy" / "sample.csv").write_text(
            "1,2,3\n4,5,6\n"
        )
        result = subprocess.run(
            [sys.executable, "-m", "mmcli", "--dry-run", "train",
             "-m", "timeseries", "-t", "motor_fault",
             "-d", "F28P55", "--model", "CLS_1k_NPU",
             "-i", str(data_dir),
             "--quantization", "QUANTIZATION_TINPU",
             "--quant-train-only"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "run_quant_train_only: true" in result.stdout
