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