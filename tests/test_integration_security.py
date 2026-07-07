"""
Integration tests for security with potentially malicious inputs.
Tests CLI behavior when passed attacker-controlled input.
"""
import pytest

try:
    from mmcli.cli import _sanitize_input, _is_safe_path
except ImportError:
    cli = None


class TestCLIWithMaliciousInput:
    """Test that malicious inputs are properly sanitized."""

    def test_sanitize_blocks_semicolon(self):
        """Semicolons should be removed."""
        result = _sanitize_input(";rm -rf /")
        assert ';' not in result

    def test_sanitize_blocks_dollar_sign(self):
        """Dollar signs should be removed."""
        result = _sanitize_input("$(whoami)")
        assert '$' not in result

    def test_sanitize_blocks_backtick(self):
        """Backticks should be removed."""
        result = _sanitize_input("`whoami`")
        assert '`' not in result


class TestPathValidationSecurity:
    """Test path validation against attacks."""

    @pytest.mark.parametrize("traversal", [
        "../../etc/passwd",
        "..\\..\\windows\\system32",
    ])
    def test_path_traversal_blocked(self, traversal):
        """Path traversal attempts should be blocked."""
        result = _is_safe_path(traversal)
        assert result is False
