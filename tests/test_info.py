"""Unit tests for mmcli.info module.

Tests the info command module which queries the tinyml-modelmaker registry
and displays available devices, models, and feature extraction presets.
"""

import json
import sys
from unittest import mock

import pytest

from mmcli import info


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_subprocess_run():
    """Fixture to patch subprocess.run for testing."""
    with mock.patch("mmcli.info.subprocess.run") as mock_run:
        yield mock_run


@pytest.fixture
def mock_sys_exit():
    """Fixture to patch sys.exit for testing error handling."""
    with mock.patch("mmcli.info.sys.exit") as mock_exit:
        yield mock_exit


# ============================================================================
# _run_query tests
# ============================================================================


class TestRunQuery:
    """Tests for _run_query function."""

    def test_run_query_returns_parsed_json(self, mock_subprocess_run):
        """_run_query should return parsed JSON dict for valid output."""
        expected_data = {"module": "timeseries", "task_descriptions": {}}
        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(expected_data)
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        result = info._run_query("python3", "test script")

        assert result == expected_data
        mock_subprocess_run.assert_called_once()

    def test_run_query_prints_json_error_and_exits_for_invalid_json(
        self, mock_subprocess_run, capsys
    ):
        """_run_query should print JSON parse error and call sys.exit(1) for invalid JSON."""
        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stdout = "not valid json{"
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        with pytest.raises(SystemExit) as exc_info:
            info._run_query("python3", "test script")

        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "Could not parse query output as JSON" in captured.err

    def test_run_query_prints_error_and_exits_for_nonzero_exit(
        self, mock_subprocess_run, capsys
    ):
        """_run_query should print error and call sys.exit(1) for non-zero exit code."""
        mock_result = mock.Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error message"
        mock_subprocess_run.return_value = mock_result

        with pytest.raises(SystemExit) as exc_info:
            info._run_query("python3", "test script")

        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "Query failed (exit 1)" in captured.err
        assert "Error message" in captured.err

    def test_run_query_prints_error_and_exits_for_empty_output(
        self, mock_subprocess_run, capsys
    ):
        """_run_query should print error and call sys.exit(1) for empty stdout."""
        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = "No output"
        mock_subprocess_run.return_value = mock_result

        with pytest.raises(SystemExit) as exc_info:
            info._run_query("python3", "test script")

        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "No output from registry query" in captured.err
        assert "No output" in captured.err

    def test_run_query_handles_unicode_output(self, mock_subprocess_run):
        """_run_query should handle unicode characters in output."""
        expected_data = {"module": "test", "name": "日本語"}
        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(expected_data)
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        result = info._run_query("python3", "test script")

        assert result == expected_data


# ============================================================================
# _group_devices tests
# ============================================================================


class TestGroupDevices:
    """Tests for _group_devices function."""

    def test_group_devices_groups_by_family(self):
        """_group_devices should group devices by family correctly."""
        devices = ["F28P55", "CC1312", "MSPM0G3507", "AM263"]

        result = info._group_devices(devices)

        assert set(result.keys()) == {"C2000", "SimpleLink", "MSPM0", "Sitara"}
        assert "F28P55" in result["C2000"]
        assert "CC1312" in result["SimpleLink"]
        assert "MSPM0G3507" in result["MSPM0"]
        assert "AM263" in result["Sitara"]

    def test_group_devices_unknown_device(self):
        """_group_devices should place unknown devices in 'Other' group."""
        devices = ["UNKNOWN123", "F28P55"]

        result = info._group_devices(devices)

        assert "C2000" in result
        assert "Other" in result
        assert "UNKNOWN123" in result["Other"]

    def test_group_devices_preserves_order(self):
        """_group_devices should preserve device order within groups."""
        devices = ["F28P55", "F29P58", "F28P65"]  # Same family

        result = info._group_devices(devices)

        assert result["C2000"] == devices

    def test_group_devices_empty_list(self):
        """_group_devices should return empty dict for empty list."""
        result = info._group_devices([])
        assert result == {}


# ============================================================================
# _build_query_script tests
# ============================================================================


class TestBuildQueryScript:
    """Tests for _build_query_script function."""

    def test_build_query_script_handles_none_task(self):
        """_build_query_script should handle None task_type."""
        script = info._build_query_script("timeseries", None, None)
        assert "timeseries" in script
        # Script uses empty string for None values
        assert "task_type   = ''" in script
        assert "target_device = ''" in script

    def test_build_query_script_handles_provided_values(self):
        """_build_query_script should include provided task_type and device."""
        script = info._build_query_script("vision", "classification", "F28P55")
        assert "vision" in script
        assert "task_type   = 'classification'" in script
        assert "target_device = 'F28P55'" in script

    def test_build_query_script_escapes_special_chars(self):
        """_build_query_script should escape single quotes in module name."""
        script = info._build_query_script("module'with'single", None, None)
        # Python repr() handles the escaping
        assert "module'with'single" in script


# ============================================================================
# run_info tests
# ============================================================================


class TestRunInfo:
    """Tests for run_info function."""

    @pytest.fixture
    def mock_print(self):
        """Fixture to patch print for testing."""
        with mock.patch("mmcli.info.print") as mock_print:
            yield mock_print

    @pytest.fixture
    def mock_list_datasets(self):
        """Fixture to patch mmcli.datasets.list_datasets for testing."""
        with mock.patch("mmcli.info.list_datasets") as mock_list:
            mock_list.return_value = []
            yield mock_list

    def test_run_info_displays_task_list_when_no_task_specified(
        self, mock_subprocess_run, mock_print
    ):
        """run_info should display task list when task is None."""
        data = {
            "module": "timeseries",
            "task_descriptions": {"classification": {"task_name": "Classification", "target_devices": ["F28P55"]}}
        }
        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(data)
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        args = mock.Mock(module="timeseries", task=None, device=None)

        info.run_info(args, "python3")

        assert mock_print.called

    def test_run_info_displays_task_details_when_task_specified(
        self, mock_subprocess_run, mock_print
    ):
        """run_info should display detailed task info when task is specified."""
        data = {
            "module": "timeseries",
            "task_descriptions": {"classification": {"task_name": "Classification", "target_devices": ["F28P55"]}},
            "models": {
                "model1": {"devices": ["F28P55"]},
                "model2": {"devices": ["F28P55", "CC1312"]}
            },
            "fe_presets": []
        }
        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(data)
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        args = mock.Mock(module="timeseries", task="classification", device="F28P55")

        info.run_info(args, "python3")

        output_calls = [str(call) for call in mock_print.call_args_list]
        output_text = "\n".join(output_calls)
        assert "Task: Classification" in output_text

    def test_run_info_handles_error_response_from_registry(
        self, mock_subprocess_run, capsys
    ):
        """run_info should print error and exit when registry returns error."""
        data = {"error": "Unknown module: invalid_module"}
        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(data)
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        args = mock.Mock(module="invalid_module", task=None, device=None)

        with pytest.raises(SystemExit) as exc_info:
            info.run_info(args, "python3")

        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "ERROR: Unknown module: invalid_module" in captured.err


# ============================================================================
# _print_task_list tests
# ============================================================================


class TestPrintTaskList:
    """Tests for _print_task_list function."""

    def test_print_task_list_shows_tasks(self, mock_print):
        """_print_task_list should display task names and device counts."""
        data = {
            "module": "timeseries",
            "task_descriptions": {
                "classification": {"task_name": "Classification", "target_devices": ["F28P55"]},
                "regression": {"task_name": "Regression", "target_devices": ["CC1312"]}
            }
        }

        info._print_task_list(data)

        output_calls = [str(call) for call in mock_print.call_args_list]
        output_text = "\n".join(output_calls)
        assert "Timeseries Task Types" in output_text
        assert "classification" in output_text

    def test_print_task_list_shows_none_when_empty(self, mock_print):
        """_print_task_list should show '(none found)' when no tasks."""
        data = {
            "module": "timeseries",
            "task_descriptions": {}
        }

        info._print_task_list(data)

        output_calls = [str(call) for call in mock_print.call_args_list]
        output_text = "\n".join(output_calls)
        assert "(none found)" in output_text


# ============================================================================
# _print_task_details tests
# ============================================================================


class TestPrintTaskDetails:
    """Tests for _print_task_details function."""

    @pytest.fixture
    def mock_list_datasets(self):
        """Fixture to patch mmcli.datasets.list_datasets for testing."""
        with mock.patch("mmcli.datasets.list_datasets") as mock_list:
            mock_list.return_value = []
            yield mock_list

    def test_print_task_details_shows_devices_and_models(
        self, mock_print, mock_list_datasets
    ):
        """_print_task_details should display devices and models."""
        data = {
            "task_descriptions": {"classification": {"target_devices": ["F28P55", "CC1312"]}},
            "models": {
                "model1": {"devices": ["F28P55"]},
                "model2": {"devices": ["F28P55", "CC1312"]}
            },
            "fe_presets": []
        }

        info._print_task_details(data, "classification", None)

        output_calls = [str(call) for call in mock_print.call_args_list]
        output_text = "\n".join(output_calls)
        assert "Supported Devices (2)" in output_text

    def test_print_task_details_shows_no_models_when_empty(
        self, mock_print, mock_list_datasets
    ):
        """_print_task_details should show '(none found)' for models when empty."""
        data = {
            "task_descriptions": {"classification": {"target_devices": ["F28P55"]}},
            "models": {},
            "fe_presets": []
        }

        info._print_task_details(data, "classification", None)

        output_calls = [str(call) for call in mock_print.call_args_list]
        output_text = "\n".join(output_calls)
        assert "(none found)" in output_text


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_group_devices_large_list(self):
        """_group_devices should group by family correctly."""
        devices = ["F28P55", "F28P65", "MSPM0G3507", "AM263"]
        result = info._group_devices(devices)

        assert len(result) == 3
        assert result["C2000"] == ["F28P55", "F28P65"]
        assert result["MSPM0"] == ["MSPM0G3507"]
        assert result["Sitara"] == ["AM263"]

    def test_build_query_script_handles_special_characters(self):
        """_build_query_script should handle module names with special chars."""
        # This tests the textwrap.dedent and .format handling
        script = info._build_query_script("audio", "anomalydetection", None)
        assert "audio" in script
        assert "anomalydetection" in script  # task_type is included

    def test_build_query_script_empty_task_type(self):
        """_build_query_script should handle empty task_type."""
        script = info._build_query_script("audio", "", None)
        assert "task_type   = ''" in script
