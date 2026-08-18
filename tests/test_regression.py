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

    def test_sanitize_input_truncates_long_input(self):
        """Long input is silently truncated to 1024 chars — does not raise."""
        from mmcli.cli import _sanitize_input
        result = _sanitize_input("x" * 1025)
        assert len(result) == 1024

    def test_sanitize_input_strips_dangerous_chars(self):
        """Shell metacharacters are stripped as defence-in-depth alongside shell=False.
        Safe path characters (dots, slashes) are preserved."""
        from mmcli.cli import _sanitize_input
        assert _sanitize_input("../etc/passwd") == "../etc/passwd"
        assert ";" not in _sanitize_input("; rm -rf /")

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


# ---------------------------------------------------------------------------
# Output-encoding regressions
# ---------------------------------------------------------------------------

class TestNonUTF8StdoutRegression:
    """mmcli must not die because its output contains a non-ASCII character.

    Windows opens the console streams with the ANSI code page (cp1252 on a
    default en-US install), which cannot encode U+2500 (the box-drawing rule
    under every table header) or U+2713 (the "project created" checkmark).
    Before `_force_utf8_stdio()`, the first such character raised
    UnicodeEncodeError and killed the process mid-output: `mmcli info` died
    immediately after printing its table header, and `mmcli init` died *after*
    creating the project, so the command both failed and left its work behind.

    Reproduced on any platform with PYTHONIOENCODING=cp1252, which is what
    makes this a real test rather than a Windows-only one nobody runs.
    """

    CP1252 = dict(os.environ, PYTHONIOENCODING="cp1252")

    # encoding="utf-8" is about how *this* process decodes the child's bytes,
    # which is a separate question from PYTHONIOENCODING (how the child encodes
    # them). Without it, text=True decodes using the parent's locale encoding —
    # cp1252 on Windows — so the child's correctly-emitted UTF-8 came back as
    # mojibake whose characters all happen to BE cp1252-encodable, and the
    # vacuity check below concluded the output had no non-encodable characters
    # left. It failed on Windows for a reason that had nothing to do with the
    # behaviour under test.
    def _run_cp1252(self, *args):
        return subprocess.run(
            [*MMCLI, *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=REPO, env=self.CP1252,
        )

    def test_command_with_non_ascii_output_completes_under_cp1252(self):
        proc = self._run_cp1252("init", "--list")
        combined = proc.stdout + proc.stderr

        assert "UnicodeEncodeError" not in combined, combined[-1500:]
        assert proc.returncode == 0, combined[-1500:]
        # Completeness, not just survival: the U+2500 rule is printed near the
        # top, so a crash there still leaves plausible-looking output behind.
        # Assert on a line emitted *after* it — that is what distinguishes a
        # finished command from a truncated one.
        assert "Create a project with:" in proc.stdout, (
            "output stops before its final line — it was truncated, not completed:\n"
            + combined[-1500:]
        )

    def test_the_guard_is_not_vacuous_output_really_is_non_encodable(self):
        """Pin the premise: if mmcli's output ever became pure ASCII, the test
        above would pass for the wrong reason and silently stop guarding
        anything. Assert the characters that break cp1252 are still there."""
        proc = subprocess.run(
            [*MMCLI, "init", "--list"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=REPO, env=dict(os.environ, PYTHONIOENCODING="utf-8"),
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

        unencodable = sorted(
            {ch for ch in proc.stdout if ord(ch) > 127
             and not _encodable_cp1252(ch)}
        )
        assert unencodable, (
            "no cp1252-unencodable character in `init --list` output any more; "
            "the regression test above is now vacuous and needs a new target"
        )


def _encodable_cp1252(ch: str) -> bool:
    try:
        ch.encode("cp1252")
        return True
    except UnicodeEncodeError:
        return False
