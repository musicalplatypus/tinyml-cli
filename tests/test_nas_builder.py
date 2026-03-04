"""Tests for NAS config building in mmcli builder."""

import argparse

import pytest

from mmcli.builder import build_config


def _make_nas_args(**overrides):
    """Build a minimal argparse Namespace with NAS flags."""
    defaults = dict(
        command="train",
        config=None,
        module="timeseries",
        task="generic_timeseries_classification",
        device="F28P55",
        model=None,
        project=None,
        run_name=None,
        feature_extraction=None,
        epochs=None,
        batch_size=None,
        lr=None,
        gpus=None,
        quantization=None,
        training_device=None,
        compile_model=None,
        native_amp=None,
        onnx=None,
        preset=None,
        nas_size="m",
        nas_epochs=None,
        nas_optimize=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestNASConfigBuilder:
    """Tests for NAS-related config generation in builder.py."""

    def test_nas_enabled_set(self):
        """When --nas is provided, config should have nas_enabled=True."""
        args = _make_nas_args()
        config = build_config(args)
        assert config["training"]["nas_enabled"] is True

    def test_nas_model_size_propagated(self):
        """NAS model size should be written to config."""
        for size in ("s", "m", "l", "xl"):
            args = _make_nas_args(nas_size=size)
            config = build_config(args)
            assert config["training"]["nas_model_size"] == size

    def test_nas_synthetic_model_name(self):
        """When --model is not given, a synthetic NAS_<size> name is generated."""
        args = _make_nas_args(nas_size="l", model=None)
        config = build_config(args)
        assert config["training"]["model_name"] == "NAS_l"

    def test_nas_explicit_model_name_preserved(self):
        """When --model IS given alongside --nas, the explicit name takes priority."""
        args = _make_nas_args(nas_size="m", model="MyCustomModel")
        config = build_config(args)
        assert config["training"]["model_name"] == "MyCustomModel"
        assert config["training"]["nas_enabled"] is True

    def test_nas_epochs_propagated(self):
        """--nas-epochs should propagate to config."""
        args = _make_nas_args(nas_epochs=20)
        config = build_config(args)
        assert config["training"]["nas_epochs"] == 20

    def test_nas_epochs_default_omitted(self):
        """When --nas-epochs is not set, the key should be absent (using backend default)."""
        args = _make_nas_args(nas_epochs=None)
        config = build_config(args)
        assert "nas_epochs" not in config["training"]

    def test_nas_optimize_mode(self):
        """--nas-optimize Memory/Compute should propagate."""
        for mode in ("Memory", "Compute"):
            args = _make_nas_args(nas_optimize=mode)
            config = build_config(args)
            assert config["training"]["nas_optimization_mode"] == mode

    def test_no_nas_no_flags(self):
        """Without --nas, NAS fields should be absent from config."""
        args = _make_nas_args(nas_size=None, model="SomeModel")
        config = build_config(args)
        assert "nas_enabled" not in config["training"]
        assert "nas_model_size" not in config["training"]

    def test_nas_training_enabled(self):
        """NAS config should have training.enable=True for train command."""
        args = _make_nas_args()
        config = build_config(args)
        assert config["training"]["enable"] is True
