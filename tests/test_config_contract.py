"""Pin the mmcli <-> modelmaker config contract for `training.compile_model` and
`training.native_amp`.

WHY THIS TEST EXISTS (read before deleting anything here):

tinyml-modelmaker's `hardware_defaults.explicit_training_keys()` decides whether to
auto-enable `torch.compile` and native AMP on a CUDA host by checking whether the key
is *present* in the resolved `training` section of the config -- not by checking its
value. Presence means "the user has an opinion, leave it alone." Absence means
"the user has no opinion, modelmaker may auto-default it on CUDA."

mmcli currently honours this by construction: `builder._set()` (mmcli/builder.py:127-135)
skips writing a key entirely when the CLI value is `None`, so an unset
`--compile-model`/`--native-amp` never appears in the emitted config. That is
compatibility *by accident* -- nothing stops a future refactor of `build_config()`
from writing `training["compile_model"] = 0` and `training["native_amp"] = False`
as explicit defaults instead of omitting them. If that happens, this file's contract
still holds syntactically (the keys have values), but the *meaning* flips: modelmaker
will read those explicit `0`/`False` entries as "the user explicitly asked for this
to stay off" and will no longer auto-enable compile/AMP on CUDA hosts. No exception is
raised, no log line is emitted -- torch.compile and AMP simply stop turning on
fleet-wide on every CUDA machine, silently.

This test pins two directions of that contract:
  1. Omission: when the CLI flag is not passed, the key must be ABSENT from
     `training`, not merely falsy.
  2. Pinning: when the CLI flag IS passed -- including to explicitly turn the
     feature OFF (`--compile-model 0`, `--no-native-amp`) -- the key must be
     PRESENT and carry the requested value. Deliberately turning something off is
     exactly the case `explicit_training_keys()` exists to respect; a test that only
     covered "unset omits" would miss a regression that dropped explicit zeros/False
     values on the way to the config dict.

If you are reading this because an absence assertion below looks "redundant" and
you're about to delete it: don't. It is the only thing standing between a
refactor and a silent, fleet-wide, unloggable change to training behaviour on every
CUDA host. See .planning/ANALYSIS-cuda-auto-defaults.md (F-3, F-4) for the full
writeup.

Scope note: this test only exercises mmcli's side (`build_config()`). It does not
import tinyml_modelmaker and asserts nothing about modelmaker's actual runtime
behaviour -- that policy lives upstream and may change independently. What is
pinned here is purely: does mmcli emit the key, and with what value, for a given
set of CLI args. That is hardware-independent and needs no CUDA host.
"""
import pytest
from argparse import Namespace

from mmcli.builder import build_config

# Keys under test. Both the "must be absent" and "must be present" assertions
# below reference these same constants rather than re-typing the string, so a
# typo in the key name cannot cause the absence check to silently diverge from
# the presence check.
_COMPILE_MODEL_KEY = "compile_model"
_NATIVE_AMP_KEY = "native_amp"


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


class TestAbsenceAssertionsAreNotVacuous:
    """Self-check on the idiom used below, not on build_config().

    `assert key not in dict` passes trivially if the key name is misspelled, if
    the dict is empty for an unrelated reason, or if the call that should have
    populated it silently failed. This class proves, independently of
    build_config(), that the exact assertion form used in this file WOULD fail
    if the key were actually present -- so the absence tests below are not
    vacuous.
    """

    def test_compile_model_absence_assertion_catches_presence(self):
        poisoned_training = {_COMPILE_MODEL_KEY: 0}
        with pytest.raises(AssertionError):
            assert _COMPILE_MODEL_KEY not in poisoned_training

    def test_native_amp_absence_assertion_catches_presence(self):
        poisoned_training = {_NATIVE_AMP_KEY: False}
        with pytest.raises(AssertionError):
            assert _NATIVE_AMP_KEY not in poisoned_training


class TestCompileModelOmissionContract:
    def test_absent_when_flag_not_passed(self):
        args = _make_args()
        config = build_config(args)
        training = config["training"]
        assert _COMPILE_MODEL_KEY not in training
        # Prove `training` is a normally-populated dict, not empty/broken --
        # otherwise the absence check above would pass for the wrong reason.
        assert "model_name" in training

    def test_pinned_when_explicitly_enabled(self):
        args = _make_args(compile_model=1)
        config = build_config(args)
        assert config["training"][_COMPILE_MODEL_KEY] == 1

    def test_pinned_when_explicitly_disabled(self):
        # The critical case: turning it OFF must still PIN the key at 0, not
        # omit it -- 0 is falsy but not None, and _set() only skips None.
        args = _make_args(compile_model=0)
        config = build_config(args)
        training = config["training"]
        assert _COMPILE_MODEL_KEY in training
        assert training[_COMPILE_MODEL_KEY] == 0


class TestNativeAmpOmissionContract:
    def test_absent_when_flag_not_passed(self):
        args = _make_args()
        config = build_config(args)
        training = config["training"]
        assert _NATIVE_AMP_KEY not in training
        assert "model_name" in training

    def test_pinned_when_explicitly_enabled(self):
        args = _make_args(native_amp=True)
        config = build_config(args)
        assert config["training"][_NATIVE_AMP_KEY] is True

    def test_pinned_when_explicitly_disabled(self):
        # The critical case: --no-native-amp must still PIN the key at False,
        # not omit it -- False is falsy but not None, and _set() only skips None.
        args = _make_args(native_amp=False)
        config = build_config(args)
        training = config["training"]
        assert _NATIVE_AMP_KEY in training
        assert training[_NATIVE_AMP_KEY] is False
