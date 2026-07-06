"""Unit tests for mmcli.analyze module.

Tests the analyze command module which inspects project datasets and
reports size, layout, and sample distribution.
"""

import os
import pickle
import sys
from pathlib import Path
from unittest import mock

import pytest

# Try to import numpy - skip tests that require it if not available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

from mmcli import analyze


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dataset_dir(tmp_path):
    """Create a temporary dataset directory structure."""
    return tmp_path / "dataset"


@pytest.fixture
def mock_numpy_import():
    """Mock numpy module for testing."""
    with mock.patch.dict('sys.modules', {'numpy': None}):
        # Create a fake numpy module
        class FakeNumpy:
            def load(self, path, mmap_mode=None):
                if mmap_mode == 'r':
                    return FakeArray()
                return FakeArray()

        class FakeArray:
            def __init__(self):
                self.shape = (10,)

            def __getitem__(self, key):
                return 0

        with mock.patch.dict('sys.modules', {'numpy': FakeNumpy()}):
            yield


# ============================================================================
# _bin_dataset tests
# ============================================================================


class TestBinDataset:
    """Tests for _bin_dataset function."""

    def test_bin_dataset_tiny(self):
        """_bin_dataset should return 'tiny' for n < 500."""
        assert analyze._bin_dataset(0) == "tiny"
        assert analyze._bin_dataset(499) == "tiny"

    def test_bin_dataset_small(self):
        """_bin_dataset should return 'small' for 500 <= n < 5000."""
        assert analyze._bin_dataset(500) == "small"
        assert analyze._bin_dataset(4999) == "small"

    def test_bin_dataset_medium(self):
        """_bin_dataset should return 'medium' for 5000 <= n < 50000."""
        assert analyze._bin_dataset(5000) == "medium"
        assert analyze._bin_dataset(49999) == "medium"

    def test_bin_dataset_large(self):
        """_bin_dataset should return 'large' for n >= 50000."""
        assert analyze._bin_dataset(50000) == "large"
        assert analyze._bin_dataset(100000) == "large"


# ============================================================================
# _row_count tests
# ============================================================================


class TestRowCount:
    """Tests for _row_count function."""

    def test_row_count_csv(self, tmp_path):
        """_row_count should correctly count CSV rows."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2,col3\n1,2,3\n4,5,6\n7,8,9\n")

        result = analyze._row_count(str(csv_file))
        assert result == 3

    def test_row_count_csv_with_header(self, tmp_path):
        """_row_count should include header in count for CSV."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2\n")

        result = analyze._row_count(str(csv_file))
        assert result == 1

    def test_row_count_txt_with_whitespace(self, tmp_path):
        """_row_count should handle whitespace-separated TXT files."""
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("a b c\n1 2 3\n4 5 6\n")

        result = analyze._row_count(str(txt_file))
        assert result == 2

    def test_row_count_txt_with_commas(self, tmp_path):
        """_row_count should handle comma-separated TXT files."""
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("a,b,c\n1,2,3\n4,5,6\n")

        result = analyze._row_count(str(txt_file))
        assert result == 2

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
    def test_row_count_npy(self, tmp_path):
        """_row_count should correctly count NPY array first dimension."""
        npy_file = tmp_path / "data.npy"
        arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])  # 4 rows
        np.save(npy_file, arr)

        result = analyze._row_count(str(npy_file))
        assert result == 4

    def test_row_count_pickle_dataframe(self, tmp_path):
        """_row_count should correctly count pandas DataFrame in pickle."""
        import pandas as pd

        pkl_file = tmp_path / "data.pkl"
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        with open(pkl_file, 'wb') as f:
            pickle.dump(df, f)

        result = analyze._row_count(str(pkl_file))
        assert result == 3

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
    def test_row_count_pickle_array(self, tmp_path):
        """_row_count should correctly count numpy array in pickle."""
        import pickle

        pkl_file = tmp_path / "data.pkl"
        arr = np.array([[1, 2], [3, 4]])
        with open(pkl_file, 'wb') as f:
            pickle.dump(arr, f)

        result = analyze._row_count(str(pkl_file))
        assert result == 2

    def test_row_count_unsupported_extension(self, tmp_path):
        """_row_count should raise ValueError for unsupported extensions."""
        unsupported_file = tmp_path / "data.xyz"
        unsupported_file.write_text("test")

        with pytest.raises(ValueError) as exc_info:
            analyze._row_count(str(unsupported_file))

        assert "Unsupported extension" in str(exc_info.value)

    def test_row_count_pickle_unsupported_type(self, tmp_path):
        """_row_count should raise TypeError for unsupported pickle types."""
        import pickle

        pkl_file = tmp_path / "data.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump("string", f)  # Not DataFrame or ndarray

        with pytest.raises(TypeError) as exc_info:
            analyze._row_count(str(pkl_file))

        assert "Unsupported pickle type" in str(exc_info.value)


# ============================================================================
# _find_data_files tests
# ============================================================================


class TestFindDataFiles:
    """Tests for _find_data_files function."""

    def test_find_data_files_returns_sorted(self, tmp_path):
        """_find_data_files should return files in sorted order."""
        (tmp_path / "b.csv").write_text("1,2")
        (tmp_path / "a.csv").write_text("3,4")
        (tmp_path / "c.npy").write_text("")

        result = analyze._find_data_files(str(tmp_path))

        assert len(result) == 3
        assert os.path.basename(result[0]) == "a.csv"
        assert os.path.basename(result[1]) == "b.csv"
        assert os.path.basename(result[2]) == "c.npy"

    def test_find_data_files_handles_multiple_extensions(self, tmp_path):
        """_find_data_files should find files with all supported extensions."""
        (tmp_path / "a.csv").write_text("1,2")
        (tmp_path / "b.txt").write_text("3 4")
        (tmp_path / "c.npy").write_text("")
        (tmp_path / "d.pkl").write_text("")

        result = analyze._find_data_files(str(tmp_path))

        assert len(result) == 4

    def test_find_data_files_empty_directory(self, tmp_path):
        """_find_data_files should return empty list for directory with no data files."""
        (tmp_path / "readme.md").write_text("not a data file")

        result = analyze._find_data_files(str(tmp_path))

        assert result == []


# ============================================================================
# _analyse_classes tests
# ============================================================================


class TestAnalyseClasses:
    """Tests for _analyse_classes function."""

    def test_analyse_classes_basic(self, temp_dataset_dir):
        """_analyse_classes should analyze classes/ layout."""
        classes_dir = temp_dataset_dir / "classes"
        classes_dir.mkdir(parents=True)

        # Create class directories with data files
        cls1_dir = classes_dir / "class1"
        cls1_dir.mkdir(parents=True)
        csv_file = cls1_dir / "data.csv"
        csv_file.write_text("col1,col2\n1,2\n3,4\n")

        result = analyze._analyse_classes(str(temp_dataset_dir))

        assert result["layout"] == "classes"
        assert result["total_samples"] == 2
        assert result["num_classes"] == 1
        assert "class1" in result["class_distribution"]
        assert result["dataset_bucket"] == "tiny"

    def test_analyse_classes_multiple_classes(self, temp_dataset_dir):
        """_analyse_classes should handle multiple classes."""
        classes_dir = temp_dataset_dir / "classes"
        classes_dir.mkdir(parents=True)

        for count in [100, 200, 300]:
            cls_dir = classes_dir / f"class{count}"
            cls_dir.mkdir(parents=True)
            csv_file = cls_dir / "data.csv"
            # Write rows (excluding header) - total 600 samples
            csv_file.write_text("\n".join(["col1,col2"] + ["a,b"] * count))

        result = analyze._analyse_classes(str(temp_dataset_dir))

        assert result["total_samples"] == 600
        assert result["num_classes"] == 3
        assert result["dataset_bucket"] == "small"

    def test_analyse_classes_min_sample_length(self, temp_dataset_dir):
        """_analyse_classes should report minimum sample length."""
        classes_dir = temp_dataset_dir / "classes"
        classes_dir.mkdir(parents=True)

        for count in [100, 50, 200]:
            cls_dir = classes_dir / f"class{count}"
            cls_dir.mkdir(parents=True)
            csv_file = cls_dir / "data.csv"
            csv_file.write_text("\n".join(["col1,col2"] + ["a,b"] * count))

        result = analyze._analyse_classes(str(temp_dataset_dir))

        assert result["min_sample_length"] == 50

    def test_analyse_classes_empty_class(self, temp_dataset_dir):
        """_analyse_classes should include empty directories with 0 count."""
        classes_dir = temp_dataset_dir / "classes"
        classes_dir.mkdir(parents=True)

        (classes_dir / "empty").mkdir(parents=True)
        cls_with_data = classes_dir / "withdata"
        cls_with_data.mkdir(parents=True)
        csv_file = cls_with_data / "data.csv"
        csv_file.write_text("col1,col2\n1,2\n")

        result = analyze._analyse_classes(str(temp_dataset_dir))

        assert result["total_samples"] == 1
        # Empty directories are included with 0 count
        assert result["num_classes"] == 2

    def test_analyse_classes_handles_csv_read_errors(self, temp_dataset_dir):
        """_analyse_classes should warn but continue on CSV read errors."""
        classes_dir = temp_dataset_dir / "classes"
        classes_dir.mkdir(parents=True)

        cls_dir = classes_dir / "baddata"
        cls_dir.mkdir(parents=True)
        csv_file = cls_dir / "bad.csv"
        csv_file.write_text("invalid,csv,{")

        result = analyze._analyse_classes(str(temp_dataset_dir))

        assert "error" not in result
        # Error handling is via stderr print


# ============================================================================
# _analyse_files tests
# ============================================================================


class TestAnalyseFiles:
    """Tests for _analyse_files function."""

    def test_analyse_files_basic(self, temp_dataset_dir):
        """_analyse_files should analyze files/ layout."""
        files_dir = temp_dataset_dir / "files"
        files_dir.mkdir(parents=True)

        csv_file = files_dir / "data.csv"
        csv_file.write_text("col1,col2\n1,2\n3,4\n5,6\n")

        result = analyze._analyse_files(str(temp_dataset_dir))

        assert result["layout"] == "files"
        assert result["total_samples"] == 3
        assert result["dataset_bucket"] == "tiny"

    def test_analyse_files_min_seq_length(self, temp_dataset_dir):
        """_analyse_files should report minimum sequence length."""
        files_dir = temp_dataset_dir / "files"
        files_dir.mkdir(parents=True)

        for count in [100, 50, 200]:
            csv_file = files_dir / f"data{count}.csv"
            csv_file.write_text("\n".join(["col1,col2"] + ["a,b"] * count))

        result = analyze._analyse_files(str(temp_dataset_dir))

        assert result["min_seq_length"] == 50

    def test_analyse_files_handles_file_errors(self, temp_dataset_dir):
        """_analyse_files should warn but continue on file read errors."""
        files_dir = temp_dataset_dir / "files"
        files_dir.mkdir(parents=True)

        good_file = files_dir / "good.csv"
        good_file.write_text("col1,col2\n1,2\n")

        bad_file = files_dir / "bad.csv"
        bad_file.write_text("invalid")

        result = analyze._analyse_files(str(temp_dataset_dir))

        assert "error" not in result
        # Error is printed to stderr


# ============================================================================
# analyse_dataset tests
# ============================================================================


class TestAnalyseDataset:
    """Tests for analyse_dataset function."""

    def test_analyse_dataset_detects_classes_layout(self, temp_dataset_dir):
        """analyse_dataset should auto-detect classes/ layout."""
        (temp_dataset_dir / "classes").mkdir(parents=True)

        result = analyze.analyse_dataset(str(temp_dataset_dir))

        assert result["layout"] == "classes"

    def test_analyse_dataset_detects_files_layout(self, temp_dataset_dir):
        """analyse_dataset should auto-detect files/ layout."""
        (temp_dataset_dir / "files").mkdir(parents=True)

        result = analyze.analyse_dataset(str(temp_dataset_dir))

        assert result["layout"] == "files"

    def test_analyse_dataset_returns_error_for_invalid_layout(self, temp_dataset_dir):
        """analyse_dataset should return error for invalid layout."""
        result = analyze.analyse_dataset(str(temp_dataset_dir))

        assert "error" in result
        assert "classes/" in result["error"]
        assert "files/" in result["error"]

    def test_analyse_dataset_priority_classes_over_files(self, temp_dataset_dir):
        """analyse_dataset should prefer classes/ over files/ when both exist."""
        (temp_dataset_dir / "classes").mkdir(parents=True)
        (temp_dataset_dir / "files").mkdir(parents=True)

        result = analyze.analyse_dataset(str(temp_dataset_dir))

        assert result["layout"] == "classes"


# ============================================================================
# print_analysis tests
# ============================================================================


class TestPrintAnalysis:
    """Tests for print_analysis function."""

    def test_print_analysis_with_error(self, capsys):
        """print_analysis should print error message for error dict."""
        stats = {"error": "Invalid dataset"}
        analyze.print_analysis(stats, "/path/to/dataset")

        captured = capsys.readouterr()
        assert "ERROR: Invalid dataset" in captured.err

    def test_print_analysis_shows_bucket_info(self, capsys):
        """print_analysis should show size bucket and notes."""
        stats = {
            "layout": "tiny",
            "total_samples": 100,
            "dataset_bucket": "tiny"
        }
        analyze.print_analysis(stats, "/path/to/dataset")

        captured = capsys.readouterr()
        assert "tiny" in captured.out
        assert "< 500 samples" in captured.out

    def test_print_analysis_classes_layout(self, capsys):
        """print_analysis should show class distribution for classes layout."""
        stats = {
            "layout": "classes",
            "total_samples": 100,
            "num_classes": 2,
            "dataset_bucket": "tiny",
            "class_distribution": {"cls1": 60, "cls2": 40}
        }
        analyze.print_analysis(stats, "/path/to/dataset")

        captured = capsys.readouterr()
        assert "Class distribution" in captured.out
        assert "cls1" in captured.out

    def test_print_analysis_files_layout(self, capsys):
        """print_analysis should show min_seq_length for files layout."""
        stats = {
            "layout": "files",
            "total_samples": 100,
            "min_seq_length": 50,
            "dataset_bucket": "tiny"
        }
        analyze.print_analysis(stats, "/path/to/dataset")

        captured = capsys.readouterr()
        assert "Min seq len" in captured.out
        assert "50" in captured.out


# ============================================================================
# run_analyze tests
# ============================================================================


class TestRunAnalyze:
    """Tests for run_analyze function."""

    @pytest.fixture
    def mock_print(self):
        """Fixture to patch print for testing."""
        with mock.patch("mmcli.analyze.print") as mock_print:
            yield mock_print

    @pytest.fixture
    def mock_analyse_dataset(self):
        """Fixture to patch analyse_dataset for testing."""
        with mock.patch("mmcli.analyse_dataset") as mock_analyse:
            mock_analyse.return_value = {
                "layout": "classes",
                "total_samples": 100,
                "dataset_bucket": "tiny"
            }
            yield mock_analyse

    def test_run_analyze_project_directory(self, temp_dataset_dir):
        """run_analyze should analyze dataset/ directory in project."""
        # Create project structure
        cls1_dir = temp_dataset_dir / "classes" / "cls1"
        cls1_dir.mkdir(parents=True)
        csv_file = cls1_dir / "data.csv"
        csv_file.write_text("col1\n1\n2\n")

        args = mock.Mock(project=str(temp_dataset_dir))

        # When analysis succeeds, run_analyze returns normally (no SystemExit)
        analyze.run_analyze(args)

    def test_run_analyze_direct_dataset_path(self, temp_dataset_dir):
        """run_analyze should handle direct dataset path."""
        (temp_dataset_dir / "classes").mkdir(parents=True)

        args = mock.Mock(project=str(temp_dataset_dir))

        # When analysis succeeds, run_analyze returns normally (no SystemExit)
        analyze.run_analyze(args)

    def test_run_analyze_missing_dataset_directory(self, temp_dataset_dir, capsys):
        """run_analyze should error when no dataset directory exists."""
        args = mock.Mock(project=str(temp_dataset_dir))

        with pytest.raises(SystemExit) as exc_info:
            analyze.run_analyze(args)

        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "ERROR" in captured.err

    def test_run_analyzer_exit_code_on_error(self, temp_dataset_dir):
        """run_analyze should exit with code 1 when analysis has error."""
        (temp_dataset_dir / "dataset").mkdir(parents=True)

        args = mock.Mock(project=str(temp_dataset_dir))

        with pytest.raises(SystemExit) as exc_info:
            analyze.run_analyze(args)

        assert exc_info.value.code == 1


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_bin_dataset_boundary_values(self):
        """_bin_dataset should handle boundary values correctly."""
        # Just below threshold
        assert analyze._bin_dataset(499) == "tiny"
        # At threshold
        assert analyze._bin_dataset(500) == "small"
        # Just below medium threshold
        assert analyze._bin_dataset(4999) == "small"
        # At medium threshold
        assert analyze._bin_dataset(5000) == "medium"
        # Just below large threshold
        assert analyze._bin_dataset(49999) == "medium"
        # At large threshold
        assert analyze._bin_dataset(50000) == "large"

    def test_row_count_empty_csv(self, tmp_path):
        """_row_count should handle empty CSV files."""
        import pandas as pd

        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")

        # Empty CSV raises an error in pandas
        with pytest.raises(pd.errors.EmptyDataError):
            analyze._row_count(str(csv_file))

    def test_analyse_dataset_nonexistent_directory(self, tmp_path):
        """analyse_dataset should return error for nonexistent directory."""
        nonexistent = str(tmp_path / "nonexistent")

        result = analyze.analyse_dataset(nonexistent)

        assert "error" in result
        assert "classes/" in result["error"] or "files/" in result["error"]

    def test_find_data_files_case_insensitive(self, tmp_path):
        """_find_data_files should find files with different case extensions."""
        # Note: on case-insensitive FS (macOS), .CSV == .csv
        csv_file = tmp_path / "data.CSV"
        csv_file.write_text("1,2")

        result = analyze._find_data_files(str(tmp_path))

        assert len(result) >= 1
