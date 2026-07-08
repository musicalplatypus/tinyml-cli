"""
Tests for mmcli error handling and recovery.

Covers: bad arguments → non-zero exit with stderr message, missing project
directories, missing onnx files, config not found, and help exiting 0.
All tests run via subprocess so they exercise the real argument parsing and
_validate_args() path including the security wiring.
"""
import subprocess
import sys
import os
import tempfile
import pytest

PYTHON = sys.executable
MMCLI = [PYTHON, "-m", "mmcli"]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(*args, cwd=None):
    return subprocess.run(
        [*MMCLI, *args],
        capture_output=True, text=True,
        cwd=cwd or REPO,
    )


class TestErrorRecovery:
    def test_missing_required_flags_exits_nonzero(self):
        """mmcli train with no flags should exit 2 and print error."""
        r = _run("train")
        assert r.returncode != 0
        assert r.stderr  # some error output expected

    def test_nonexistent_project_dir_exits_with_message(self, tmp_path):
        """Passing a project dir that doesn't exist should exit non-zero with a clear message."""
        r = _run("train", "-m", "timeseries", "-t", "generic_timeseries_classification",
                 "-d", "F28P55", "-n", "generic_timeseries_classification",
                 "-i", str(tmp_path / "does_not_exist"))
        assert r.returncode != 0
        assert "not found" in r.stderr.lower() or "error" in r.stderr.lower()

    def test_nonexistent_config_file_exits_with_message(self):
        """--config pointing to a missing file should produce a clear error."""
        r = _run("train", "--config", "/tmp/totally_missing_file_xyz.yaml")
        assert r.returncode != 0
        assert "not found" in r.stderr.lower() or "error" in r.stderr.lower()

    def test_nonexistent_onnx_file_exits_with_message(self):
        """--onnx pointing to a missing file should produce a clear error."""
        r = _run("compile", "-m", "timeseries", "-t", "generic_timeseries_classification",
                 "-d", "F28P55", "-n", "generic_timeseries_classification",
                 "--onnx", "/tmp/totally_missing_model_xyz.onnx")
        assert r.returncode != 0
        assert "not found" in r.stderr.lower() or "error" in r.stderr.lower()

    def test_unknown_subcommand_exits_nonzero(self):
        """Unrecognised subcommand should exit non-zero."""
        r = _run("nonexistent_subcommand_xyz")
        assert r.returncode != 0

    def test_resource_cleanup_on_failure(self, tmp_path):
        """After a failed invocation no leftover temp files accumulate in tmp_path."""
        before = set(tmp_path.iterdir())
        _run("train", "-m", "timeseries", "-t", "generic_timeseries_classification",
             "-d", "F28P55", "-n", "generic_timeseries_classification",
             "-i", str(tmp_path / "no_such_project"))
        after = set(tmp_path.iterdir())
        assert before == after, "Failed invocation left unexpected files behind"


def test_help_exits_zero():
    """--help must always exit 0 (regression guard)."""
    r = _run("--help")
    assert r.returncode == 0
    assert "mmcli" in r.stdout.lower()


def test_version_exits_zero():
    """--version must exit 0."""
    r = _run("--version")
    assert r.returncode == 0


def test_path_traversal_rejected():
    """Relative path traversal in --project should be rejected before any I/O."""
    r = _run("train", "-m", "timeseries", "-t", "generic_timeseries_classification",
             "-d", "F28P55", "-n", "generic_timeseries_classification",
             "-i", "../../etc/passwd")
    assert r.returncode != 0
    assert "unsafe" in r.stderr.lower() or "traversal" in r.stderr.lower() \
        or "error" in r.stderr.lower()


def test_oversized_module_flag_rejected():
    """A module value exceeding 1024 chars should be rejected with an error."""
    r = _run("train", "-m", "x" * 1025)
    assert r.returncode != 0
