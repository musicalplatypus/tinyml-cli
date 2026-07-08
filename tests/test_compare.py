"""Tests for the compare module functionality."""

import pytest
from unittest.mock import patch
from mmcli.compare import compare_models, format_comparison


_MOCK_CLS = {
    "module": "timeseries",
    "task_type": "generic_timeseries_classification",
    "models": {
        "CLS_1k_NPU": {"quantization": 2, "learning_rate": 0.002, "batch_size": 64,
                        "device_count": 24, "devices": ["F28P55", "F28P65"]},
        "CLS_6k_NPU": {"quantization": 2, "learning_rate": 0.002, "batch_size": 64,
                        "device_count": 24, "devices": ["F28P55", "F28P65"]},
    },
}

_MOCK_REGR = {
    "module": "timeseries",
    "task_type": "generic_timeseries_regression",
    "models": {
        "REGR_1k": {"quantization": 2, "learning_rate": 0.001, "batch_size": 32,
                     "device_count": 20, "devices": ["F28P55"]},
    },
}


class TestCompareModels:
    """Unit tests for compare_models (subprocess mocked)."""

    def test_basic_comparison(self):
        side_effects = [_MOCK_CLS, _MOCK_REGR]
        with patch("mmcli.compare._query_models", side_effect=side_effects):
            result = compare_models(
                "timeseries",
                ["generic_timeseries_classification", "generic_timeseries_regression"],
            )

        assert result["module"] == "timeseries"
        assert result["device"] is None
        assert "generic_timeseries_classification" in result["tasks"]
        assert "generic_timeseries_regression" in result["tasks"]

    def test_model_count_aggregated(self):
        with patch("mmcli.compare._query_models", side_effect=[_MOCK_CLS, _MOCK_REGR]):
            result = compare_models(
                "timeseries",
                ["generic_timeseries_classification", "generic_timeseries_regression"],
            )
        assert result["tasks"]["generic_timeseries_classification"]["model_count"] == 2
        assert result["tasks"]["generic_timeseries_regression"]["model_count"] == 1

    def test_device_filter_passed_through(self):
        calls = []

        def capture(*args, **kwargs):
            calls.append(args)
            return _MOCK_CLS

        with patch("mmcli.compare._query_models", side_effect=capture):
            compare_models("timeseries", ["generic_timeseries_classification"], device="F28P55")

        assert calls[0][2] == "F28P55"

    def test_error_in_query_recorded(self):
        err = {"error": "Cannot import tinyml_modelmaker"}
        with patch("mmcli.compare._query_models", return_value=err):
            result = compare_models("timeseries", ["generic_timeseries_classification"])

        task_info = result["tasks"]["generic_timeseries_classification"]
        assert "error" in task_info

    def test_quantization_types_extracted(self):
        with patch("mmcli.compare._query_models", return_value=_MOCK_CLS):
            result = compare_models("timeseries", ["generic_timeseries_classification"])

        task = result["tasks"]["generic_timeseries_classification"]
        assert 2 in task["quantization_types"]


class TestFormatComparison:
    """Unit tests for format_comparison."""

    def _make_result(self):
        with patch("mmcli.compare._query_models", side_effect=[_MOCK_CLS, _MOCK_REGR]):
            return compare_models(
                "timeseries",
                ["generic_timeseries_classification", "generic_timeseries_regression"],
            )

    def test_returns_string(self):
        formatted = format_comparison(self._make_result())
        assert isinstance(formatted, str)

    def test_contains_task_names(self):
        formatted = format_comparison(self._make_result())
        assert "generic_timeseries_classification" in formatted
        assert "generic_timeseries_regression" in formatted

    def test_contains_model_names(self):
        formatted = format_comparison(self._make_result())
        assert "CLS_1k_NPU" in formatted
        assert "REGR_1k" in formatted

    def test_empty_tasks(self):
        formatted = format_comparison({"module": "timeseries", "device": None, "tasks": {}})
        assert "No task data" in formatted

    def test_quant_label_shown(self):
        formatted = format_comparison(self._make_result())
        assert "TI-NPU" in formatted
