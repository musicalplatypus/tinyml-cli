"""Diagnostic tools for troubleshooting common issues."""

import os
import subprocess
import sys
from typing import Dict, List, Tuple


class DiagnosticIssue:
    """Represents a diagnostic finding."""

    def __init__(
        self,
        name: str,
        severity: str,  # "critical", "warning", "info"
        status: str,  # "pass", "fail", "skipped"
        message: str = "",
        fix_suggestion: str = ""
    ):
        self.name = name
        self.severity = severity
        self.status = status
        self.message = message
        self.fix_suggestion = fix_suggestion

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "severity": self.severity,
            "status": self.status,
            "message": self.message,
            "fix_suggestion": self.fix_suggestion,
        }


class DiagnosticResult:
    """Represents the results of a diagnostic run."""

    def __init__(self, checks: List[DiagnosticIssue]):
        self.checks = checks

    @property
    def is_healthy(self) -> bool:
        return all(c.status == "pass" for c in self.checks)

    @property
    def critical_failures(self) -> List[DiagnosticIssue]:
        return [c for c in self.checks if c.severity == "critical" and c.status == "fail"]

    @property
    def warnings(self) -> List[DiagnosticIssue]:
        return [c for c in self.checks if c.severity == "warning" and c.status != "pass"]


def check_python_version() -> DiagnosticIssue:
    """Check Python version is supported."""
    current = sys.version_info

    if current >= (3, 10):
        return DiagnosticIssue(
            name="Python Version",
            severity="critical",
            status="pass",
            message=f"Python {current.major}.{current.minor} is supported"
        )
    else:
        return DiagnosticIssue(
            name="Python Version",
            severity="critical",
            status="fail",
            message=f"Python {current.major}.{current.minor} found, but 3.10+ required",
            fix_suggestion="Upgrade Python to version 3.10 or higher"
        )


def check_environment_var(name: str, description: str) -> DiagnosticIssue:
    """Check if an environment variable is set."""
    value = os.environ.get(name)

    if value:
        return DiagnosticIssue(
            name=description,
            severity="info",
            status="pass",
            message=f"{name} is set to {value[:30]}..."
        )
    else:
        return DiagnosticIssue(
            name=description,
            severity="warning",
            status="fail",
            message=f"{name} environment variable not set",
            fix_suggestion=f"Set the {name} environment variable"
        )


def check_tinyml_modelmaker() -> DiagnosticIssue:
    """Check if tinyml_modelmaker can be imported."""
    try:
        import tinyml_modelmaker
        return DiagnosticIssue(
            name="tinyml_modelmaker",
            severity="critical",
            status="pass",
            message=f"tinyml_modelmaker {getattr(tinyml_modelmaker, '__version__', 'unknown')} is installed"
        )
    except ImportError:
        return DiagnosticIssue(
            name="tinyml_modelmaker",
            severity="critical",
            status="fail",
            message="Cannot import tinyml_modelmaker",
            fix_suggestion=(
                "Install tinyml_modelmaker: pip install tinyml-modelmaker\n"
                "Or set MMCLI_PYTHON to point to a Python environment with the package installed"
            )
        )


def check_path_permission(path: str, description: str) -> DiagnosticIssue:
    """Check if a path exists and is accessible."""
    try:
        if os.path.exists(path):
            return DiagnosticIssue(
                name=description,
                severity="info",
                status="pass",
                message=f"{path} exists and is accessible"
            )
        else:
            return DiagnosticIssue(
                name=description,
                severity="warning",
                status="fail",
                message=f"{path} does not exist",
                fix_suggestion=f"Create {path} or update the path configuration"
            )
    except Exception as e:
        return DiagnosticIssue(
            name=description,
            severity="critical",
            status="fail",
            message=f"Cannot access {path}: {e}",
            fix_suggestion="Check file permissions and path validity"
        )


def check_tvm_compiler() -> DiagnosticIssue:
    """Check whether TVM (ti_mcu_nnc) is importable for compilation."""
    try:
        import tvm  # noqa: F401
        return DiagnosticIssue(
            name="TVM (ti_mcu_nnc)",
            severity="warning",
            status="pass",
            message="TVM is installed and importable — compilation is available",
        )
    except ImportError:
        return DiagnosticIssue(
            name="TVM (ti_mcu_nnc)",
            severity="warning",
            status="fail",
            message="TVM is not importable; 'compile' and 'run' commands will not work",
            fix_suggestion=(
                "Install ti-mcu-nnc into the environment pointed to by MMCLI_PYTHON.\n"
                "If using a separate venv, set MMCLI_PYTHON to that interpreter."
            ),
        )


def check_tiarmclang() -> DiagnosticIssue:
    """Check whether tiarmclang (TI ARM cross-compiler) is available."""
    import shutil

    candidate = None
    env_cgt = os.environ.get("ARM_LLVM_CGT_PATH")
    if env_cgt:
        path = os.path.join(env_cgt, "bin", "tiarmclang")
        if os.path.isfile(path) and os.access(path, os.X_OK):
            candidate = path
    if candidate is None:
        candidate = shutil.which("tiarmclang")

    if candidate:
        return DiagnosticIssue(
            name="tiarmclang (ARM cross-compiler)",
            severity="warning",
            status="pass",
            message=f"tiarmclang found at {candidate}",
        )
    else:
        return DiagnosticIssue(
            name="tiarmclang (ARM cross-compiler)",
            severity="warning",
            status="fail",
            message="tiarmclang not found — device-family compilation unavailable",
            fix_suggestion=(
                "Install the TI ARM LLVM toolchain and either:\n"
                "  • Add its bin/ directory to PATH, or\n"
                "  • Set ARM_LLVM_CGT_PATH to the toolchain root directory."
            ),
        )


def check_c2000_compiler() -> DiagnosticIssue:
    """Check whether the C2000 CGT (cl2000) is available for C2000-family targets."""
    import shutil

    candidate = None
    env_root = os.environ.get("C2000_CG_ROOT")
    if env_root:
        path = os.path.join(env_root, "bin", "cl2000")
        if os.path.isfile(path) and os.access(path, os.X_OK):
            candidate = path
    if candidate is None:
        candidate = shutil.which("cl2000")

    if candidate:
        return DiagnosticIssue(
            name="cl2000 (C2000 CGT)",
            severity="warning",
            status="pass",
            message=f"cl2000 found at {candidate}",
        )
    else:
        return DiagnosticIssue(
            name="cl2000 (C2000 CGT)",
            severity="warning",
            status="fail",
            message="cl2000 not found — compilation for C2000 targets (F28xxx/F29xxx) unavailable",
            fix_suggestion=(
                "Install the C2000 Code Generation Tools and either:\n"
                "  • Add its bin/ directory to PATH, or\n"
                "  • Set C2000_CG_ROOT to the toolchain root directory."
            ),
        )


def run_diagnostic_checks(full: bool = False) -> DiagnosticResult:
    """Run all diagnostic checks."""
    checks = [
        check_python_version(),
        check_environment_var("MMCLI_PYTHON", "MMCLI_PYTHON"),
        check_environment_var("MMCLI_MODELZOO_PATH", "MMCLI_MODELZOO_PATH"),
        check_tinyml_modelmaker(),
        check_tvm_compiler(),
        check_tiarmclang(),
        check_c2000_compiler(),
    ]

    if full:
        # Run extended diagnostics
        project_path = os.getcwd()
        checks.append(check_path_permission(project_path, "Current Directory"))
        checks.append(DiagnosticIssue(
            name="Disk Space",
            severity="info",
            status="pass" if get_disk_space() > 100 else "fail",
            message=f"{get_disk_space()} MB free disk space"
        ))

    return DiagnosticResult(checks)


def get_disk_space() -> int:
    """Get available disk space in MB."""
    try:
        stat = os.statvfs("/")
        return int(stat.f_bavail * stat.f_frsize / (1024 * 1024))
    except Exception:
        return -1


def format_diagnostic_results(result: DiagnosticResult) -> str:
    """Format diagnostic results as human-readable output."""
    lines = []

    lines.append("=" * 60)
    lines.append("MMCLI DIAGNOSTIC REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Summary
    total = len(result.checks)
    passed = sum(1 for c in result.checks if c.status == "pass")
    failed = sum(1 for c in result.checks if c.status == "fail")

    lines.append(f"Total Checks: {total}")
    lines.append(f"Passed: {passed}")
    lines.append(f"Failed: {failed}")
    lines.append("")

    # Results by severity
    critical_issues = result.critical_failures
    warnings = result.warnings

    if critical_issues:
        lines.append("CRITICAL ISSUES:")
        for issue in critical_issues:
            lines.append(f"  ✗ {issue.name}: {issue.message}")
            lines.append(f"    Fix: {issue.fix_suggestion}")
        lines.append("")

    if warnings:
        lines.append("WARNINGS:")
        for issue in warnings:
            lines.append(f"  ! {issue.name}: {issue.message}")
            if issue.fix_suggestion:
                lines.append(f"    Suggestion: {issue.fix_suggestion}")
        lines.append("")

    # All checks
    lines.append("ALL CHECKS:")
    for issue in result.checks:
        status_symbol = {"pass": "✓", "fail": "✗", "warning": "!"}[issue.status]
        severity_label = f"[{issue.severity.upper()}]"
        lines.append(f"  {status_symbol} {severity_label} {issue.name}")

    # Recommendations
    if not result.is_healthy:
        lines.append("")
        lines.append("RECOMMENDATIONS:")

        # Suggest fixing critical issues first
        for issue in critical_issues:
            lines.append(f"  1. {issue.fix_suggestion}")

        # Then warnings
        for i, issue in enumerate(warnings, start=2):
            if issue.fix_suggestion:
                lines.append(f"  {i}. {issue.fix_suggestion}")

    return "\n".join(lines)


def get_fix_for_error(error_message: str) -> Tuple[str, str]:
    """
    Get a fix suggestion for a specific error message.

    Returns:
        Tuple of (severity, fix_suggestion)
    """
    error_lower = error_message.lower()

    if "cannot import tinyml_modelmaker" in error_lower:
        return (
            "critical",
            "Install tinyml_modelmaker: pip install tinyml-modelmaker"
        )
    elif "invalid project path" in error_lower or "path traversal" in error_lower:
        return (
            "warning",
            "Use a relative path that doesn't contain .. or absolute paths starting with /"
        )
    elif "permission denied" in error_lower:
        return (
            "critical",
            "Check file/directory permissions and ensure you have write access"
        )
    elif "disk space" in error_lower or "no space" in error_lower:
        return (
            "critical",
            "Free up disk space before running the command"
        )
    else:
        return (
            "info",
            "See documentation for troubleshooting tips"
        )