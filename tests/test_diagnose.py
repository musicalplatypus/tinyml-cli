"""
Tests for the diagnose module functionality.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from mmcli.diagnose import (
    DiagnosticIssue,
    DiagnosticResult,
    check_python_version,
    check_environment_var,
    check_tinyml_modelmaker,
    check_tvm_compiler,
    check_tiarmclang,
    check_c2000_compiler,
    run_diagnostic_checks,
    format_diagnostic_results,
    get_fix_for_error
)


class TestDiagnosticIssue:
    """Tests for DiagnosticIssue class."""

    def test_diagnostic_issue_creation(self):
        """Test creating a diagnostic issue."""
        issue = DiagnosticIssue(
            name="Test Issue",
            severity="critical",
            status="pass",
            message="Test message",
            fix_suggestion="Test suggestion"
        )

        assert issue.name == "Test Issue"
        assert issue.severity == "critical"
        assert issue.status == "pass"
        assert issue.message == "Test message"
        assert issue.fix_suggestion == "Test suggestion"

    def test_diagnostic_issue_to_dict(self):
        """Test converting diagnostic issue to dictionary."""
        issue = DiagnosticIssue(
            name="Test Issue",
            severity="warning",
            status="fail",
            message="Test message",
            fix_suggestion="Test suggestion"
        )

        result = issue.to_dict()

        assert isinstance(result, dict)
        assert result["name"] == "Test Issue"
        assert result["severity"] == "warning"
        assert result["status"] == "fail"


class TestDiagnosticResult:
    """Tests for DiagnosticResult class."""

    def test_diagnostic_result_is_healthy(self):
        """Test healthy result detection."""
        issues = [
            DiagnosticIssue("Test", "info", "pass"),
            DiagnosticIssue("Test2", "warning", "pass"),
        ]
        result = DiagnosticResult(issues)

        assert result.is_healthy is True

    def test_diagnostic_result_critical_failures(self):
        """Test critical failure detection."""
        issues = [
            DiagnosticIssue("Test", "critical", "fail"),
        ]
        result = DiagnosticResult(issues)

        assert len(result.critical_failures) == 1
        assert result.critical_failures[0].name == "Test"

    def test_diagnostic_result_warnings(self):
        """Test warning detection."""
        issues = [
            DiagnosticIssue("Test", "warning", "fail"),
        ]
        result = DiagnosticResult(issues)

        assert len(result.warnings) == 1
        assert result.warnings[0].name == "Test"


class TestDiagnosticChecks:
    """Tests for individual diagnostic checks."""

    def test_check_python_version_pass(self):
        """Test Python version check passes."""
        issue = check_python_version()
        assert issue.status == "pass"
        assert issue.severity == "critical"

    @patch("os.environ.get")
    def test_check_environment_var_not_set(self, mock_getenv):
        """Test missing environment variable detection."""
        mock_getenv.return_value = None
        issue = check_environment_var("TEST_VAR", "Test Variable")

        assert issue.status == "fail"
        assert issue.severity == "warning"

    def test_check_tinyml_modelmaker_missing(self):
        """Test missing tinyml_modelmaker detection."""
        from unittest.mock import patch
        with patch.dict("sys.modules", {"tinyml_modelmaker": None}):
            issue = check_tinyml_modelmaker()

        assert issue.status == "fail"
        assert "Install tinyml_modelmaker" in issue.fix_suggestion


class TestCompileToolChecks:
    """Tests for check_tvm_compiler and check_tiarmclang."""

    def test_check_tvm_compiler_present(self):
        """TVM importable → pass with warning severity."""
        import sys
        fake_tvm = type("tvm", (), {})()
        with patch.dict("sys.modules", {"tvm": fake_tvm}):
            issue = check_tvm_compiler()
        assert issue.status == "pass"
        assert issue.severity == "warning"
        assert "TVM" in issue.name

    def test_check_tvm_compiler_missing(self):
        """TVM not importable → fail with install hint."""
        with patch.dict("sys.modules", {"tvm": None}):
            issue = check_tvm_compiler()
        assert issue.status == "fail"
        assert issue.severity == "warning"
        assert "MMCLI_PYTHON" in issue.fix_suggestion

    def test_check_tiarmclang_via_env_path(self, tmp_path):
        """ARM_LLVM_CGT_PATH set to a dir containing executable tiarmclang → pass."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        tiarmclang = bin_dir / "tiarmclang"
        tiarmclang.write_text("#!/bin/sh\n")
        tiarmclang.chmod(0o755)

        with patch.dict("os.environ", {"ARM_LLVM_CGT_PATH": str(tmp_path)}):
            issue = check_tiarmclang()

        assert issue.status == "pass"
        assert str(tiarmclang) in issue.message

    def test_check_tiarmclang_via_which(self, tmp_path):
        """tiarmclang on PATH via shutil.which → pass."""
        fake_path = str(tmp_path / "tiarmclang")
        with patch("shutil.which", return_value=fake_path):
            with patch.dict("os.environ", {}, clear=False):
                os.environ.pop("ARM_LLVM_CGT_PATH", None)
                issue = check_tiarmclang()

        assert issue.status == "pass"
        assert fake_path in issue.message

    def test_check_tiarmclang_not_found(self):
        """No ARM_LLVM_CGT_PATH and not on PATH → fail with install hint."""
        with patch("shutil.which", return_value=None):
            env = {k: v for k, v in os.environ.items() if k != "ARM_LLVM_CGT_PATH"}
            with patch.dict("os.environ", env, clear=True):
                issue = check_tiarmclang()

        assert issue.status == "fail"
        assert issue.severity == "warning"
        assert "ARM_LLVM_CGT_PATH" in issue.fix_suggestion

    def test_check_c2000_compiler_via_env_path(self, tmp_path):
        """C2000_CG_ROOT set to a dir containing executable cl2000 → pass."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        cl2000 = bin_dir / "cl2000"
        cl2000.write_text("#!/bin/sh\n")
        cl2000.chmod(0o755)

        with patch.dict("os.environ", {"C2000_CG_ROOT": str(tmp_path)}):
            issue = check_c2000_compiler()

        assert issue.status == "pass"
        assert str(cl2000) in issue.message

    def test_check_c2000_compiler_via_which(self, tmp_path):
        """cl2000 on PATH via shutil.which → pass."""
        fake_path = str(tmp_path / "cl2000")
        with patch("shutil.which", return_value=fake_path):
            env = {k: v for k, v in os.environ.items() if k != "C2000_CG_ROOT"}
            with patch.dict("os.environ", env, clear=True):
                issue = check_c2000_compiler()

        assert issue.status == "pass"
        assert fake_path in issue.message

    def test_check_c2000_compiler_not_found(self):
        """No C2000_CG_ROOT and not on PATH → fail with install hint."""
        with patch("shutil.which", return_value=None):
            env = {k: v for k, v in os.environ.items() if k != "C2000_CG_ROOT"}
            with patch.dict("os.environ", env, clear=True):
                issue = check_c2000_compiler()

        assert issue.status == "fail"
        assert issue.severity == "warning"
        assert "C2000_CG_ROOT" in issue.fix_suggestion


class TestDetectCompileTools:
    """Tests for cli._detect_compile_tools (returns 3-tuple)."""

    def test_tvm_and_tiarmclang_available(self, tmp_path):
        """TVM and tiarmclang present → (True, path, None) for ARM env."""
        from mmcli.cli import _detect_compile_tools

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        tiarmclang = bin_dir / "tiarmclang"
        tiarmclang.write_text("#!/bin/sh\n")
        tiarmclang.chmod(0o755)

        fake_result = MagicMock()
        fake_result.returncode = 0

        clean_env = {k: v for k, v in os.environ.items()
                     if k not in ("ARM_LLVM_CGT_PATH", "C2000_CG_ROOT")}
        with patch("subprocess.run", return_value=fake_result):
            with patch.dict("os.environ", {**clean_env, "ARM_LLVM_CGT_PATH": str(tmp_path)},
                            clear=True):
                with patch("shutil.which", return_value=None):
                    tvm_ok, arm_path, c2000_path = _detect_compile_tools("/usr/bin/python3")

        assert tvm_ok is True
        assert arm_path == str(tiarmclang)
        assert c2000_path is None

    def test_tvm_and_c2000_available(self, tmp_path):
        """TVM and cl2000 present → (True, None, path) for C2000 env."""
        from mmcli.cli import _detect_compile_tools

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        cl2000 = bin_dir / "cl2000"
        cl2000.write_text("#!/bin/sh\n")
        cl2000.chmod(0o755)

        fake_result = MagicMock()
        fake_result.returncode = 0

        clean_env = {k: v for k, v in os.environ.items()
                     if k not in ("ARM_LLVM_CGT_PATH", "C2000_CG_ROOT")}
        with patch("subprocess.run", return_value=fake_result):
            with patch.dict("os.environ", {**clean_env, "C2000_CG_ROOT": str(tmp_path)},
                            clear=True):
                with patch("shutil.which", return_value=None):
                    tvm_ok, arm_path, c2000_path = _detect_compile_tools("/usr/bin/python3")

        assert tvm_ok is True
        assert arm_path is None
        assert c2000_path == str(cl2000)

    def test_tvm_missing(self):
        """subprocess returns non-zero → tvm_ok is False."""
        from mmcli.cli import _detect_compile_tools

        fake_result = MagicMock()
        fake_result.returncode = 1

        clean_env = {k: v for k, v in os.environ.items()
                     if k not in ("ARM_LLVM_CGT_PATH", "C2000_CG_ROOT")}
        with patch("subprocess.run", return_value=fake_result):
            with patch("shutil.which", return_value=None):
                with patch.dict("os.environ", clean_env, clear=True):
                    tvm_ok, arm_path, c2000_path = _detect_compile_tools("/usr/bin/python3")

        assert tvm_ok is False
        assert arm_path is None
        assert c2000_path is None

    def test_tiarmclang_via_which_fallback(self):
        """No ARM_LLVM_CGT_PATH but tiarmclang on PATH → path returned."""
        from mmcli.cli import _detect_compile_tools

        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_clang_path = "/usr/local/bin/tiarmclang"

        clean_env = {k: v for k, v in os.environ.items()
                     if k not in ("ARM_LLVM_CGT_PATH", "C2000_CG_ROOT")}
        with patch("subprocess.run", return_value=fake_result):
            with patch("shutil.which", return_value=fake_clang_path):
                with patch.dict("os.environ", clean_env, clear=True):
                    tvm_ok, arm_path, c2000_path = _detect_compile_tools("/usr/bin/python3")

        assert tvm_ok is True
        # shutil.which is used as fallback for both tiarmclang and cl2000;
        # since our mock returns a path, tiarmclang gets it first.
        assert arm_path == fake_clang_path


class TestDiagnosticResults:
    """Tests for diagnostic result formatting."""

    def test_format_results(self):
        """Test formatting diagnostic results."""
        issues = [
            DiagnosticIssue("Passing", "info", "pass"),
            DiagnosticIssue("Failing", "critical", "fail", "Error message", "Fix this"),
        ]
        result = DiagnosticResult(issues)

        formatted = format_diagnostic_results(result)

        assert isinstance(formatted, str)
        assert "DIAGNOSTIC REPORT" in formatted
        assert "Passing" in formatted
        assert "Failing" in formatted


class TestErrorFixes:
    """Tests for error-specific fix suggestions."""

    def test_tinyml_modelmaker_error(self):
        """Test specific error handling for tinyml_modelmaker."""
        severity, suggestion = get_fix_for_error("Cannot import tinyml_modelmaker")

        assert "tinyml-modelmaker" in suggestion.lower()

    def test_path_traversal_error(self):
        """Test path error handling."""
        severity, suggestion = get_fix_for_error("Invalid project path: /etc/passwd")

        assert "relative path" in suggestion.lower() or ".." in suggestion

    def test_unknown_error(self):
        """Test unknown error handling."""
        severity, suggestion = get_fix_for_error("Something unexpected happened")

        assert severity == "info"
        assert "documentation" in suggestion.lower()


class TestDiagnosticIntegration:
    """Integration tests for diagnostic functionality."""

    def test_run_diagnostic_checks(self):
        """Test running full diagnostic checks."""
        result = run_diagnostic_checks()

        assert isinstance(result, DiagnosticResult)
        assert len(result.checks) > 0

    @patch("mmcli.diagnose.check_python_version")
    def test_run_diagnostic_checks_with_full(self, mock_check):
        """Test running extended diagnostics."""
        # Mock a passing check
        mock_check.return_value = DiagnosticIssue(
            name="Python Version",
            severity="critical",
            status="pass",
            message="Python 3.10+ is supported"
        )

        result = run_diagnostic_checks(full=True)

        assert isinstance(result, DiagnosticResult)