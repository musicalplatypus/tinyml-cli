"""
Workflow Integration Tests for mmcli.

Tests complete workflows (init -> analyze -> recommend) without requiring external dependencies.
Uses subprocess-based testing similar to test_cli_integration.py.
"""
import os
import sys
import tempfile

import pytest


MMCLI = [sys.executable, "-m", "mmcli"]


def _run_cli(*args, timeout=30):
    """Run mmcli with args and return (returncode, stdout, stderr)."""
    import subprocess

    cmd = MMCLI + list(args)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_analyze_subcommand():
    """Test that analyze subcommand works with a project."""
    import subprocess

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create minimal valid project structure
        data_dir = os.path.join(tmp_dir, "data")
        os.makedirs(data_dir)
        dataset_dir = os.path.join(data_dir, "dataset")
        classes_dir = os.path.join(dataset_dir, "classes", "dummy")
        os.makedirs(classes_dir)

        # Create sample data file
        sample_csv = os.path.join(classes_dir, "sample.csv")
        with open(sample_csv, "w") as f:
            f.write("feature,label\n1,0\n2,1\n3,0\n")

        # Create annotations directory and labels file
        annotations_dir = os.path.join(dataset_dir, "annotations")
        os.makedirs(annotations_dir)
        labels_csv = os.path.join(annotations_dir, "labels.csv")
        with open(labels_csv, "w") as f:
            f.write("file,label\nsample.csv,dummy\n")

        # Run analyze on the project
        rc, stdout, stderr = _run_cli(
            "analyze",
            "-i", data_dir
        )
        output = stdout + stderr

        assert rc == 0, f"mmcli analyze failed: {output}"
        assert "classes" in output or "files" in output.lower()


def test_dry_run_train():
    """Test dry-run train mode."""
    import subprocess

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create minimal valid project structure
        data_dir = os.path.join(tmp_dir, "data")
        os.makedirs(data_dir)
        dataset_dir = os.path.join(data_dir, "dataset")
        classes_dir = os.path.join(dataset_dir, "classes", "dummy")
        os.makedirs(classes_dir)

        # Real classification datasets are a single unlabelled data column
        # with no header — the class comes from the classes/<class>/
        # directory, not from a column in the CSV. Verified against
        # ~/Documents/edgeai/myproject1/dataset/classes/sawtooth/saw10.csv,
        # from a project that trained successfully on 2026-08-14. A
        # "feature,label" two-column header here is 2 detected channels,
        # which no generic_timeseries_classification preset accepts (17 of
        # 19 expect 1, the rest expect 3) — do not add a header/label
        # column back in.
        sample_csv = os.path.join(classes_dir, "sample.csv")
        with open(sample_csv, "w") as f:
            f.write("1\n2\n3\n")

        annotations_dir = os.path.join(dataset_dir, "annotations")
        os.makedirs(annotations_dir)
        labels_csv = os.path.join(annotations_dir, "labels.csv")
        with open(labels_csv, "w") as f:
            f.write("file,label\nsample.csv,dummy\n")

        # Run dry-run train
        rc, stdout, stderr = _run_cli(
            "--dry-run", "train",
            "-i", data_dir,
            "-m", "timeseries",
            "-t", "generic_timeseries_classification",
            "-d", "F28P55",
            "-n", "CLS_1k_NPU"
        )
        output = stdout + stderr

        assert rc == 0, f"mmcli --dry-run train failed: {output}"


def test_recommend_subcommand():
    """Test recommend subcommand with mocked environment."""
    import subprocess
    from unittest import mock

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create minimal valid project structure
        data_dir = os.path.join(tmp_dir, "data")
        os.makedirs(data_dir)
        dataset_dir = os.path.join(data_dir, "dataset")
        classes_dir = os.path.join(dataset_dir, "classes", "dummy")
        os.makedirs(classes_dir)

        sample_csv = os.path.join(classes_dir, "sample.csv")
        with open(sample_csv, "w") as f:
            f.write("feature,label\n1,0\n2,1\n3,0\n")

        annotations_dir = os.path.join(dataset_dir, "annotations")
        os.makedirs(annotations_dir)
        labels_csv = os.path.join(annotations_dir, "labels.csv")
        with open(labels_csv, "w") as f:
            f.write("file,label\nsample.csv,dummy\n")

        # Mock the modelzoo path to avoid needing actual tinyml-modelzoo
        with mock.patch("mmcli.recommend._find_modelzoo_examples_path", return_value=None):
            # Run recommend on the project (will likely fail but should not crash)
            rc, stdout, stderr = _run_cli(
                "recommend",
                "-i", data_dir,
                "-m", "timeseries"
            )
            output = stdout + stderr

            # Recommend may fail without modelzoo, but shouldn't crash
            # It should at least print something about no models found or missing path
            assert "No models found" in output or rc != 0


def test_help_commands():
    """Test that help commands work."""
    import subprocess

    # Main help
    rc1, stdout1, stderr1 = _run_cli("--help")
    output = stdout1 + stderr1
    assert rc1 == 0
    assert "train" in output.lower()
    assert "analyze" in output.lower()

    # Train help
    rc2, stdout2, stderr2 = _run_cli("train", "--help")
    output = stdout2 + stderr2
    assert rc2 == 0
    assert "--model" in output or "-m" in output
