"""
Regression tests — guard against re-introduction of previously fixed bugs.

Each test is tied to a specific known issue so the failure message is
traceable. Tests are fast (no external dependencies, no subprocess where
possible).
"""
import subprocess
import sys
import os
import pytest

PYTHON = sys.executable
MMCLI = [PYTHON, "-m", "mmcli"]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(*args):
    return subprocess.run(
        [*MMCLI, *args],
        capture_output=True, text=True, cwd=REPO,
    )


# ---------------------------------------------------------------------------
# Security regressions
# ---------------------------------------------------------------------------

class TestSecurityRegression:
    def test_is_safe_path_not_bypassable_via_dot_collapse(self):
        """Old string-replace impl: 'a....b' → 'ab' (no '..' left → passed).
        Pathlib impl catches this correctly."""
        from mmcli.cli import _is_safe_path
        assert not _is_safe_path("a....b/../../../etc/passwd"), \
            "_is_safe_path must use pathlib resolution, not string replacement"

    def test_sanitize_input_raises_not_truncates(self):
        """Old impl silently truncated to 1024 chars; new impl raises ValueError."""
        from mmcli.cli import _sanitize_input
        with pytest.raises(ValueError, match="maximum length"):
            _sanitize_input("x" * 1025)

    def test_sanitize_input_does_not_strip_chars(self):
        """Old impl stripped shell metacharacters — bypassable and lossy.
        New impl returns the string unchanged (length check only)."""
        from mmcli.cli import _sanitize_input
        assert _sanitize_input("../etc/passwd") == "../etc/passwd"
        assert _sanitize_input("; rm -rf /") == "; rm -rf /"

    def test_is_safe_path_empty_string_rejected(self):
        """Empty path must always be rejected."""
        from mmcli.cli import _is_safe_path
        assert not _is_safe_path("")
        assert not _is_safe_path("   ")

    def test_security_functions_importable(self):
        """Regression: both functions were removed in Phase 5 eb0a1bd.
        They must remain importable from mmcli.cli."""
        from mmcli.cli import _is_safe_path, _sanitize_input
        assert callable(_is_safe_path)
        assert callable(_sanitize_input)

    def test_traversal_rejected_via_cli(self):
        """Path traversal in --project must be rejected by _validate_args()."""
        r = _run("train", "-m", "timeseries", "-t", "generic_timeseries_classification",
                 "-d", "F28P55", "-n", "generic_timeseries_classification",
                 "-i", "../../etc/passwd")
        assert r.returncode != 0


# ---------------------------------------------------------------------------
# CLI interface regressions
# ---------------------------------------------------------------------------

class TestCLIRegression:
    def test_help_always_exits_zero(self):
        """--help must exit 0 — regression guard for argparse misconfiguration."""
        assert _run("--help").returncode == 0

    def test_version_always_exits_zero(self):
        """--version must exit 0."""
        assert _run("--version").returncode == 0

    def test_subcommand_help_exits_zero(self):
        """Each subcommand's --help must exit 0."""
        for sub in ("train", "compile", "run", "info", "analyze",
                    "recommend", "deploy", "compare", "diagnose", "shell"):
            r = _run(sub, "--help")
            assert r.returncode == 0, f"mmcli {sub} --help exited {r.returncode}"

    def test_missing_module_flag_produces_error_message(self):
        """Missing required --module/-m should produce an error, not a traceback."""
        r = _run("train", "-t", "generic_timeseries_classification",
                 "-d", "F28P55", "-n", "generic_timeseries_classification")
        assert r.returncode != 0
        assert "Traceback" not in r.stderr, \
            "Missing flag produced a Python traceback instead of a clean error"


# ---------------------------------------------------------------------------
# onnxsim shutdown crash (exit 245) — Phase 6
# ---------------------------------------------------------------------------

def test_macos_segv_constant_defined():
    """_MACOS_SEGV = 245 must remain in test_e2e.py (Phase 6 regression guard)."""
    e2e = os.path.join(REPO, "tests", "test_e2e.py")
    if not os.path.exists(e2e):
        pytest.skip("test_e2e.py not present")
    content = open(e2e).read()
    assert "_MACOS_SEGV" in content, \
        "Exit-245 guard removed from test_e2e.py — onnxsim benign crash no longer accepted"
    assert "assert rc in (0, _MACOS_SEGV)" in content
