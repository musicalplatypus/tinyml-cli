"""
Test configuration and fixtures for mmcli.
This file extends the root conftest.py with test-specific mocks.
"""
import sys
from unittest import mock


@pytest.fixture
def mock_tinyml_modelmaker_registry():
    """Create a stub tinyml_modelmaker package for testing."""

    class MockConstants:
        TASK_DESCRIPTIONS = {
            "generic_timeseries_classification": {
                "task_name": "Generic Timeseries Classification",
                "target_devices": ["F28P55", "MSPM0G3507"]
            },
            "generic_timeseries_regression": {
                "task_name": "Generic Timeseries Regression",
                "target_devices": ["F28P55", "MSPM0G3507"]
            },
            "generic_timeseries_forecasting": {
                "task_name": "Generic Timeseries Forecasting",
                "target_devices": ["F28P55"]
            },
            "generic_timeseries_anomalydetection": {
                "task_name": "Generic Timeseries Anomaly Detection",
                "target_devices": ["F28P55", "MSPM0G3507", "AM263"]
            }
        }
        FEATURE_EXTRACTION_PRESET_DESCRIPTIONS = {
            "FE_DEFAULT": {"common": {"task_type": "generic_timeseries_classification"}},
            "FE_LOW_POWER": {"common": {"task_type": ["generic_timeseries_classification", "generic_timeseries_regression"]}},
        }

    class MockTraining:
        @staticmethod
        def get_model_descriptions(task_type=None, target_device=None):
            """Return mock model descriptions."""
            models = {
                "CLS_1k_NPU": type("ModelDesc", (), {
                    "get": lambda self, key, default=None: {"training": {"target_devices": {"F28P55": {}}}} if key == "training" else None,
                    **({"training": type("TrainingInfo", (), {"target_devices": {"F28P55": {}}})()} if target_device == "F28P55" or not target_device else {})
                })(),
                "REGR_1k": type("ModelDesc", (), {
                    "get": lambda self, key, default=None: {"training": {"target_devices": {"F28P55": {}}}} if key == "training" else None,
                })(),
                "FCST_LSTM8": type("ModelDesc", (), {
                    "get": lambda self, key, default=None: {"training": {"target_devices": {"F28P55": {}}}} if key == "training" else None,
                })(),
                "AD_1k": type("ModelDesc", (), {
                    "get": lambda self, key, default=None: {"training": {"target_devices": {"F28P55": {}}}} if key == "training" else None,
                })(),
            }

            if task_type:
                return {name: desc for name, desc in models.items()}
            return models

    # Create stub modules
    mock_ai_modules = type("MockAIModules", (), {})()

    mock_timeseries = type("TimeseriesModule", (), {
        "constants": MockConstants(),
        "training": MockTraining(),
    })()

    mock_vision = type("VisionModule", (), {
        "constants": MockConstants(),
        "training": MockTraining(),
    })()

    mock_audio = type("AudioModule", (), {
        "constants": MockConstants(),
        "training": MockTraining(),
    })()

    # Build the stub tinyml_modelmaker package
    stub_package = type("MockTinyMLModelMaker", (), {
        "__file__": "/tmp/fake_tinyml_modelmaker",
        "ai_modules": mock_ai_modules,
    })()

    stub_package.ai_modules.timeseries = mock_timeseries
    stub_package.ai_modules.vision = mock_vision
    stub_package.ai_modules.audio = mock_audio

    # Patch sys.modules before info.py is imported
    with mock.patch.dict('sys.modules', {
        'tinyml_modelmaker': stub_package,
        'tinyml_modelmaker.ai_modules': mock_ai_modules,
        'tinyml_modelmaker.ai_modules.timeseries': mock_timeseries,
        'tinyml_modelmaker.ai_modules.vision': mock_vision,
        'tinyml_modelmaker.ai_modules.audio': mock_audio,
    }):
        yield
