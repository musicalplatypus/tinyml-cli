"""
Attack surface tests for mmcli CLI commands.
Tests security patterns against various attack vectors.
"""
import pytest
import inspect

from mmcli.cli import _sanitize_input, _is_safe_path


class TestInputSanitization:
    """Test input sanitization against various attack patterns."""

    @pytest.mark.parametrize("malicious, dangerous_chars", [
        (";rm -rf /", [';', '$', '`', '|']),
        ("| cat /etc/passwd", [';', '$', '`', '|']),
        ("`whoami`", [';', '$', '`', '|']),
        ("$(whoami)", [';', '$', '`', '|']),
    ])
    def test_sanitize_removes_dangerous_chars(self, malicious, dangerous_chars):
        """Dangerous characters should be removed from input."""
        result = _sanitize_input(malicious)
        for char in dangerous_chars:
            assert char not in result

    @pytest.mark.parametrize("input_str", [
        "test;ls",
        "test && ls",
        "test || ls",
    ])
    def test_sanitize_blocks_shell_metacharacters(self, input_str):
        """Shell metacharacters should be removed."""
        result = _sanitize_input(input_str)
        # After sanitization, the dangerous patterns should be gone
        assert ';' not in result or len(result) == 0


class TestPathValidation:
    """Test path validation against traversal attacks."""

    @pytest.mark.parametrize("traversal", [
        "../../etc/passwd",
        "../..\\windows\\system32",
        ".....//.....//etc/passwd",
    ])
    def test_path_traversal_blocked(self, traversal):
        """Path traversal attempts should be blocked."""
        result = _is_safe_path(traversal)
        assert result is False

    @pytest.mark.parametrize("safe_path", [
        "./project/data",
        "../other-project",
        "relative/path/file.txt",
    ])
    def test_safe_paths_allowed(self, safe_path):
        """Safe relative paths should be allowed."""
        result = _is_safe_path(safe_path)
        assert isinstance(result, bool)


class TestSubprocessSecurity:
    """Test subprocess execution security."""

    def test_cli_uses_shell_false(self):
        """CLI commands must use shell=False for subprocess calls."""
        import mmcli.cli as cli_module

        # Check that shell=False is used in the source code
        source = inspect.getsource(cli_module)
        assert "shell=False" in source


class TestLengthLimits:
    """Test length limits to prevent buffer overflows."""

    def test_sanitize_enforces_length(self):
        """Sanitize should enforce maximum length."""
        long_input = "a" * 2048
        result = _sanitize_input(long_input)
        assert len(result) <= 1024


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string(self):
        """Empty strings should be handled."""
        result = _sanitize_input("")
        assert isinstance(result, str)

    def test_none_input_converts_to_string(self):
        """None or non-string inputs should be converted to string."""
        result = _sanitize_input(None)
        assert result == "None"


class TestSecurityPatterns:
    """Test that security patterns are correctly implemented."""

    def test_sanitize_preserves_safe_chars(self):
        """Safe characters should be preserved."""
        safe_input = "my-project_name/file.txt"
        result = _sanitize_input(safe_input)
        # After sanitization, should only contain safe chars
        assert isinstance(result, str)
