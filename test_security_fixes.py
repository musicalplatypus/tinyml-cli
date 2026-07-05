#!/usr/bin/env python3
"""
Security test suite for mmcli command injection fixes.
"""

import os
import sys
import tempfile
from unittest.mock import patch

# Add the mmcli module to the path
sys.path.insert(0, '/Users/martin/Documents/repos/PlatypusVibes/tinyml-cli')

from mmcli.cli import _sanitize_input, _is_safe_path


def test_sanitize_input():
    """Test input sanitization functions."""
    print("Testing input sanitization...")

    # Test normal inputs
    assert _sanitize_input("normal_input") == "normal_input"
    assert _sanitize_input("input.with.dots") == "input.with.dots"
    assert _sanitize_input("input_with_underscores") == "input_with_underscores"
    assert _sanitize_input("input-with-dashes") == "input-with-dashes"
    assert _sanitize_input("input with spaces") == "input with spaces"

    # Test dangerous inputs
    assert _sanitize_input("../etc/passwd") == "etcpasswd"  # Should remove ..
    assert _sanitize_input("; rm -rf /") == "rmrf"  # Should remove semicolon
    assert _sanitize_input("$(rm -rf /)") == "rmrf"  # Should remove command substitution
    assert _sanitize_input("`rm -rf /`") == "rmrf"  # Should remove backticks

    print("✓ Input sanitization tests passed")


def test_is_safe_path():
    """Test path safety validation."""
    print("Testing path safety...")

    # Test safe paths
    assert _is_safe_path("normal/path")
    assert _is_safe_path("./relative/path")
    assert _is_safe_path("path/with.dots")
    assert _is_safe_path("path_with_underscores")
    assert _is_safe_path("path-with-dashes")

    # Test unsafe paths
    assert not _is_safe_path("../etc/passwd")  # Path traversal
    assert not _is_safe_path("/absolute/path")  # Absolute path
    assert not _is_safe_path("path; rm -rf /")  # Command injection attempt
    assert not _is_safe_path("path$(rm -rf /)")  # Command substitution

    print("✓ Path safety tests passed")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("Testing edge cases...")

    # Empty input
    assert _sanitize_input("") == ""
    assert not _is_safe_path("")

    # Very long input (should be truncated)
    long_input = "a" * 1030  # Longer than 1024 limit
    sanitized = _sanitize_input(long_input)
    assert len(sanitized) == 1024

    # Unicode characters (should be preserved if valid)
    unicode_input = "path_ñ_ç_ü"
    assert _sanitize_input(unicode_input) == unicode_input

    print("✓ Edge case tests passed")


def main():
    """Run all security tests."""
    print("Running mmcli security fix tests...\n")

    try:
        test_sanitize_input()
        test_is_safe_path()
        test_edge_cases()

        print("\n✅ All security tests passed! Command injection vulnerabilities are mitigated.")
        return 0

    except Exception as e:
        print(f"\n❌ Security test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())