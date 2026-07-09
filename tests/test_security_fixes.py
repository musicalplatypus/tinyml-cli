#!/usr/bin/env python3
"""
Security test suite for mmcli path validation and input length enforcement.

Design notes (updated from original):
- _sanitize_input() strips non-safe characters and truncates to max_length.
  Shell injection is also prevented by shell=False in all subprocess calls
  (defence in depth). Path traversal is prevented by _is_safe_path().
- _is_safe_path() uses pathlib.Path.resolve() + is_relative_to(), which is
  immune to iterative-replace bypasses (a....b collapse) and encoded traversal
  (%2e%2e). The old string-based '..' check was bypassable.
"""

import os
import sys
import tempfile
import pytest

from mmcli.cli import _sanitize_input, _is_safe_path


def test_sanitize_input_passthrough():
    """_sanitize_input returns the string unchanged when within length limit."""
    assert _sanitize_input("normal_input") == "normal_input"
    assert _sanitize_input("input.with.dots") == "input.with.dots"
    assert _sanitize_input("input_with_underscores") == "input_with_underscores"
    assert _sanitize_input("input-with-dashes") == "input-with-dashes"
    assert _sanitize_input("input with spaces") == "input with spaces"
    assert _sanitize_input("") == ""


def test_sanitize_input_strips_shell_metacharacters():
    """_sanitize_input strips shell metacharacters as defence-in-depth.

    Safe path characters (dots, slashes, underscores, hyphens) are preserved.
    Shell injection is also prevented by shell=False subprocess calls.
    """
    assert _sanitize_input("../etc/passwd") == "../etc/passwd"
    assert ";" not in _sanitize_input("; rm -rf /")
    assert "$" not in _sanitize_input("$(rm -rf /)")
    assert "`" not in _sanitize_input("`rm -rf /`")


def test_sanitize_input_truncates_on_length_exceeded():
    """_sanitize_input truncates to max_length — does not raise."""
    long_input = "a" * 1025
    result = _sanitize_input(long_input)
    assert len(result) == 1024


def test_sanitize_input_custom_length():
    """Custom max_length is respected — truncates to that length."""
    assert _sanitize_input("abc", max_length=3) == "abc"
    assert _sanitize_input("abcd", max_length=3) == "abc"


def test_sanitize_input_non_string_coerced():
    """Non-string inputs are coerced to str before length check."""
    assert _sanitize_input(42) == "42"
    assert _sanitize_input(3.14) == "3.14"


def test_sanitize_input_unicode_preserved():
    """Unicode strings are preserved unchanged."""
    assert _sanitize_input("path_ñ_ç_ü") == "path_ñ_ç_ü"


def test_is_safe_path_relative_under_cwd():
    """Relative paths that resolve inside cwd are safe."""
    assert _is_safe_path("normal/path")
    assert _is_safe_path("./relative/path")
    assert _is_safe_path("path/with.dots")
    assert _is_safe_path("path_with_underscores")
    assert _is_safe_path("path-with-dashes")


def test_is_safe_path_blocks_traversal():
    """Paths that resolve outside cwd/tempdir via .. are rejected."""
    assert not _is_safe_path("../etc/passwd")
    assert not _is_safe_path("../../root")


def test_is_safe_path_blocks_absolute_outside_temp():
    """Absolute paths not under cwd or OS temp dir are rejected."""
    assert not _is_safe_path("/absolute/path")
    assert not _is_safe_path("/etc/passwd")


def test_is_safe_path_allows_temp_dir():
    """Paths inside the OS temp directory are allowed (cross-platform)."""
    tmp = tempfile.mkdtemp()
    assert _is_safe_path(tmp)
    from pathlib import Path
    assert _is_safe_path(str(Path(tmp) / "subdir"))


def test_is_safe_path_rejects_empty():
    """Empty or whitespace-only paths are rejected."""
    assert not _is_safe_path("")
    assert not _is_safe_path("   ")


def test_is_safe_path_shell_metachar_paths_not_blocked():
    """Paths containing shell metacharacters resolve under cwd — not blocked.

    _is_safe_path() guards path traversal, not shell injection.
    Shell injection is prevented by shell=False in subprocess calls.
    """
    # These resolve to cwd/path... which is relative to cwd → safe
    assert _is_safe_path("path; rm -rf /")
    assert _is_safe_path("path$(echo x)")


def main():
    """Run via python test_security_fixes.py for quick manual check."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v"],
        cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
