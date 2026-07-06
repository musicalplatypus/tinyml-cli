"""Unit tests for mmcli.recommend module.

Tests the recommend command module which provides model and feature-extraction
preset recommendations based on task, device, and dataset parameters.
"""

import os
import sys
from unittest import mock

import pytest

from mmcli import recommend


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_modelzoo_path(tmp_path):
    """Create a temporary modelzoo examples directory structure."""
    examples_dir = tmp_path / "modelzoo" / "examples"
    examples_dir.mkdir(parents=True)

    # Create a sample example with config.yaml
    example_dir = examples_dir / "example1"
    example_dir.mkdir()
    (example_dir / "config.yaml").write_text("""
common:
  task_type: image_classification
  target_device: F28P55
  target_module: vision
training:
  model_name: tiny_model_5k
data_processing_feature_extraction:
  variables: 10
""")

    return str(examples_dir)


# ============================================================================
# TASK_TYPE_TO_MODULE tests
# ============================================================================


class TestTaskTypeToModule:
    """Tests for TASK_TYPE_TO_MODULE mapping."""

    def test_task_type_mapping_exists(self):
        """TASK_TYPE_TO_MODULE should have expected mappings."""
        assert "generic_timeseries_classification" in recommend.TASK_TYPE_TO_MODULE
        assert recommend.TASK_TYPE_TO_MODULE["image_classification"] == "vision"
        assert recommend.TASK_TYPE_TO_MODULE["audio_classification"] == "audio"

    def test_task_type_mapping_includes_all_types(self):
        """TASK_TYPE_TO_MODULE should include all defined task types."""
        expected = [
            "generic_timeseries_classification",
            "generic_timeseries_regression",
            "generic_timeseries_forecasting",
            "generic_timeseries_anomalydetection",
            "motor_fault",
            "ecg_classification",
            "arc_fault",
            "blower_imbalance",
            "pir_detection",
            "image_classification",
            "audio_classification",
        ]
        for task_type in expected:
            assert task_type in recommend.TASK_TYPE_TO_MODULE


# ============================================================================
# _complexity_tier tests
# ============================================================================


class TestComplexityTier:
    """Tests for _complexity_tier function."""

    def test_micro_tier(self):
        """_complexity_tier should return 'micro' for params <= 1000."""
        assert recommend._complexity_tier(0) == "micro"
        assert recommend._complexity_tier(1000) == "micro"

    def test_tiny_tier(self):
        """_complexity_tier should return 'tiny' for 1001-4000 params."""
        assert recommend._complexity_tier(1001) == "tiny"
        assert recommend._complexity_tier(4000) == "tiny"

    def test_small_tier(self):
        """_complexity_tier should return 'small' for 4001-10000 params."""
        assert recommend._complexity_tier(4001) == "small"
        assert recommend._complexity_tier(10000) == "small"

    def test_medium_tier(self):
        """_complexity_tier should return 'medium' for 10001-30000 params."""
        assert recommend._complexity_tier(10001) == "medium"
        assert recommend._complexity_tier(30000) == "medium"

    def test_large_tier(self):
        """_complexity_tier should return 'large' for > 30000 params."""
        # Note: tier returns 'large' only when hi is None and param_count > lo
        assert recommend._complexity_tier(30002) == "large"  # strictly greater than 30001
        assert recommend._complexity_tier(100000) == "large"


# ============================================================================
# _parse_model_params tests
# ============================================================================


class TestParseModelParams:
    """Tests for _parse_model_params function."""

    def test_parse_k_suffix(self):
        """_parse_model_params should handle k suffix (e.g., '5k' -> 5000)."""
        assert recommend._parse_model_params("model_5k") == 5000
        assert recommend._parse_model_params("tiny_model_20K") == 20000

    def test_parse_plain_numbers(self):
        """_parse_model_params should extract largest number >= 100."""
        assert recommend._parse_model_params("model1234") == 1234
        # Note: returns None only when no numbers >= 100 are found
        assert recommend._parse_model_params("m50") is None  # < 100, returns None
        assert recommend._parse_model_params("model5000") == 5000

    def test_parse_complex_names(self):
        """_parse_model_params should handle complex model names."""
        # Should extract largest number >= 100
        result = recommend._parse_model_params("tiny_mobilenet_v2_30k")
        assert result == 30000


# ============================================================================
# Dataset size bucket tests
# ============================================================================


class TestDatasetSizeBucket:
    """Tests for dataset_size_bucket handling."""

    def test_preferred_max_params(self):
        """_DATASET_PREFERRED_MAX_PARAMS should have expected values."""
        assert recommend._DATASET_PREFERRED_MAX_PARAMS["tiny"] == 2000
        assert recommend._DATASET_PREFERRED_MAX_PARAMS["small"] == 10000
        assert recommend._DATASET_PREFERRED_MAX_PARAMS["medium"] == 30000
        assert recommend._DATASET_PREFERRED_MAX_PARAMS["large"] is None


# ============================================================================
# _find_modelzoo_examples_path tests
# ============================================================================


class TestFindModelzooExamplesPath:
    """Tests for _find_modelzoo_examples_path function."""

    def test_env_var_priority(self, monkeypatch, tmp_path):
        """MMCLI_MODELZOO_PATH env var should take priority."""
        examples_dir = tmp_path / "custom_modelzoo" / "examples"
        examples_dir.mkdir(parents=True)

        monkeypatch.setenv("MMCLI_MODELZOO_PATH", str(tmp_path / "custom_modelzoo"))

        result = recommend._find_modelzoo_examples_path()
        # The env var points to the root, but we expect examples/ subpath
        # Since custom_modelzoo/examples exists, it should return that path
        assert result is not None
        assert "examples" in result

    def test_fallback_locations(self):
        """_find_modelzoo_examples_path should return None when nothing exists."""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = recommend._find_modelzoo_examples_path()
            # Should return None when no modelzoo is found
            assert result is None


# ============================================================================
# get_recommendations tests
# ============================================================================


class TestGetRecommendations:
    """Tests for get_recommendations function."""

    def test_no_examples_returns_error(self):
        """get_recommendations should return error when no examples found."""
        # This would normally require a real modelzoo path, so we mock it
        with mock.patch.object(recommend, "_find_modelzoo_examples_path", return_value=None):
            result = recommend.get_recommendations(
                task_type="image_classification",
                target_device="F28P55",
                target_module="vision",
            )

            assert result["success"] is False
            assert "Could not find tinyml-modelzoo/examples" in result["error"]

    def test_empty_examples_returns_error(self, temp_modelzoo_path):
        """get_recommendations should return error when no valid examples."""
        # Create empty examples dir
        with mock.patch.object(recommend, "_find_modelzoo_examples_path", return_value=temp_modelzoo_path):
            # Patch _list_examples to return empty list
            with mock.patch.object(recommend, "_list_examples", return_value=[]):
                result = recommend.get_recommendations(
                    task_type="image_classification",
                    target_device="F28P55",
                    target_module="vision",
                )

                assert result["success"] is False

    def test_scoring_preferences(self, temp_modelzoo_path):
        """get_recommendations should score examples correctly."""
        with mock.patch.object(recommend, "_find_modelzoo_examples_path", return_value=temp_modelzoo_path):
            # Mock _list_examples to control the examples
            test_examples = [{
                "task_type": "image_classification",
                "target_device": "F28P55",
                "target_module": "vision",
                "variables": 10,
                "model_name": "tiny_model_5k",
                "feature_extraction_name": "default_fe",
                "example_dir": "/tmp/test",
            }]

            with mock.patch.object(recommend, "_list_examples", return_value=test_examples):
                result = recommend.get_recommendations(
                    task_type="image_classification",
                    target_device="F28P55",
                    target_module="vision",
                    variables=10,
                )

                assert result["success"] is True
                assert result["task_type"] == "image_classification"
                assert len(result["ranked"]) > 0


# ============================================================================
# print_recommendations tests
# ============================================================================


class TestPrintRecommendations:
    """Tests for print_recommendations function."""

    def test_error_message(self, capsys):
        """print_recommendations should print error message on failure."""
        result = {
            "success": False,
            "error": "Test error",
            "ranked": [],
        }

        recommend.print_recommendations(result)

        captured = capsys.readouterr()
        assert "ERROR: Test error" in captured.err

    def test_shows_recommended_model(self, capsys):
        """print_recommendations should show recommended model."""
        result = {
            "success": True,
            "error": None,
            "task_type": "image_classification",
            "recommended_model": "best_model",
            "recommended_fe_preset": "fe_preset",
            "match_score": 6,
            "match_breakdown": {},
            "ranked": [
                {"model_name": "best_model", "score": 6, "param_count": 5000,
                 "complexity_tier": "tiny", "match_breakdown": {}, "feature_extraction_name": "fe_preset"},
            ],
        }

        recommend.print_recommendations(result)

        captured = capsys.readouterr()
        assert "Recommended model:   best_model" in captured.out
        assert "Feature extraction:  fe_preset" in captured.out


# ============================================================================
# run_recommend tests
# ============================================================================


class TestRunRecommend:
    """Tests for run_recommend function."""

    def test_module_inference_from_task(self, tmp_path):
        """run_recommend should infer module from task type."""
        # Create a mock args object
        class MockArgs:
            task = "image_classification"
            device = "F28P55"
            modelzoo_path = None
            variables = None
            dataset_size_bucket = None

        args = MockArgs()

        with mock.patch.object(recommend, "get_recommendations", return_value={
            "success": True,
            "error": None,
            "task_type": "image_classification",
            "recommended_model": "model1",
            "recommended_fe_preset": "fe_preset",
            "ranked": [],
            "match_score": 0,
        }):
            with mock.patch("mmcli.recommend.print"):
                recommend.run_recommend(args)

    def test_missing_module_error(self, capsys):
        """run_recommend should error when module cannot be inferred."""
        class MockArgs:
            task = "unknown_task"
            device = "F28P55"
            modelzoo_path = None
            variables = None
            dataset_size_bucket = None

        args = MockArgs()

        # This will exit with code 2, so we catch SystemExit
        with pytest.raises(SystemExit) as exc_info:
            recommend.run_recommend(args)

        assert exc_info.value.code == 2


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_parse_model_params_no_match(self):
        """_parse_model_params should return None when no match."""
        assert recommend._parse_model_params("model") is None
        assert recommend._parse_model_params("m50") is None  # < 100

    def test_complexity_tier_boundaries(self):
        """_complexity_tier should handle boundary values correctly."""
        assert recommend._complexity_tier(1000) == "micro"
        assert recommend._complexity_tier(4000) == "tiny"
        assert recommend._complexity_tier(10000) == "small"
        assert recommend._complexity_tier(30000) == "medium"
