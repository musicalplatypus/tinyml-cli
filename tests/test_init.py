"""Tests for the mmcli init command and datasets module."""

import os
import shutil
import tempfile
import zipfile

import pytest

from mmcli.datasets import (
    DATASET_REGISTRY,
    extract_dataset,
    get_dataset,
    list_datasets,
)


# ---------------------------------------------------------------------------
# Helpers — create a realistic dataset zip in a temp directory
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_zip(tmp_path):
    """Create a temporary zip file that mimics a real TI dataset.

    The TI dataset zips contain classes/ and annotations/ at the zip root.
    extract_dataset() will place these under <project>/dataset/.
    """
    build_dir = tmp_path / "build"
    annotations = build_dir / "annotations"
    classes = build_dir / "classes"
    class_a = classes / "class_a"
    class_b = classes / "class_b"

    annotations.mkdir(parents=True)
    class_a.mkdir(parents=True)
    class_b.mkdir(parents=True)

    # Add minimal data files
    (annotations / "labels.csv").write_text("file,label\na.csv,class_a\n")
    (class_a / "a.csv").write_text("1,2,3\n4,5,6\n")
    (class_b / "b.csv").write_text("7,8,9\n")

    zip_path = tmp_path / "test_dataset.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for root, dirs, files in os.walk(build_dir):
            for f in files:
                abs_path = os.path.join(root, f)
                arc_name = os.path.relpath(abs_path, build_dir)
                zf.write(abs_path, arc_name)

    return str(zip_path)


@pytest.fixture
def _register_dataset(sample_zip, monkeypatch):
    """Temporarily register a test dataset and point MMCLI_DATASETS to it."""
    zip_dir = os.path.dirname(sample_zip)
    zip_name = os.path.basename(sample_zip)

    DATASET_REGISTRY["test_dataset"] = {
        "filename": zip_name,
        "task_types": ["arc_fault", "generic_timeseries_classification"],
        "module": "timeseries",
        "description": "A test dataset for unit tests",
    }
    monkeypatch.setenv("MMCLI_DATASETS", zip_dir)

    yield

    DATASET_REGISTRY.pop("test_dataset", None)


# ---------------------------------------------------------------------------
# list_datasets
# ---------------------------------------------------------------------------

class TestListDatasets:
    """Tests for the list_datasets() function."""

    def test_list_all(self, _register_dataset):
        result = list_datasets()
        names = [d["name"] for d in result]
        assert "test_dataset" in names

    def test_filter_by_task(self, _register_dataset):
        result = list_datasets(task_type="arc_fault")
        assert len(result) >= 1
        assert all("arc_fault" in d["task_types"] for d in result)

    def test_filter_by_task_excludes(self, _register_dataset):
        result = list_datasets(task_type="image_classification")
        names = [d["name"] for d in result]
        assert "test_dataset" not in names

    def test_filter_by_module(self, _register_dataset):
        result = list_datasets(module="timeseries")
        names = [d["name"] for d in result]
        assert "test_dataset" in names

    def test_filter_by_wrong_module(self, _register_dataset):
        result = list_datasets(module="vision")
        names = [d["name"] for d in result]
        assert "test_dataset" not in names


# ---------------------------------------------------------------------------
# get_dataset
# ---------------------------------------------------------------------------

class TestGetDataset:
    def test_found(self, _register_dataset):
        result = get_dataset("test_dataset")
        assert result is not None
        assert result["name"] == "test_dataset"

    def test_not_found(self):
        result = get_dataset("nonexistent_dataset")
        assert result is None


# ---------------------------------------------------------------------------
# extract_dataset
# ---------------------------------------------------------------------------

class TestExtractDataset:
    """Tests for the extract_dataset() function."""

    def test_extract_creates_project(self, _register_dataset, tmp_path):
        project = str(tmp_path / "new_project")
        extract_dataset("test_dataset", project, task_type="arc_fault")

        assert os.path.isdir(project)
        assert os.path.isdir(os.path.join(project, "dataset"))
        assert os.path.isdir(os.path.join(project, "dataset", "annotations"))
        assert os.path.isdir(os.path.join(project, "dataset", "classes"))

    def test_extract_data_files_present(self, _register_dataset, tmp_path):
        project = str(tmp_path / "new_project")
        extract_dataset("test_dataset", project, task_type="arc_fault")

        labels = os.path.join(project, "dataset", "annotations", "labels.csv")
        assert os.path.isfile(labels)

    def test_extract_project_exists_error(self, _register_dataset, tmp_path):
        project = str(tmp_path / "existing_project")
        os.makedirs(project)

        with pytest.raises(SystemExit) as exc_info:
            extract_dataset("test_dataset", project, task_type="arc_fault")
        assert exc_info.value.code == 2

    def test_extract_unknown_dataset_error(self, tmp_path):
        project = str(tmp_path / "new_project")
        with pytest.raises(SystemExit) as exc_info:
            extract_dataset("totally_fake", project)
        assert exc_info.value.code == 2

    def test_extract_incompatible_task_error(self, _register_dataset, tmp_path):
        project = str(tmp_path / "new_project")
        with pytest.raises(SystemExit) as exc_info:
            extract_dataset("test_dataset", project,
                            task_type="image_classification")
        assert exc_info.value.code == 2
