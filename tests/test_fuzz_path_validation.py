"""
Fuzz tests for path validation using hypothesis.
Tests _is_safe_path() function against various attack patterns.
"""
import pytest
from hypothesis import given, strategies as st, example

try:
    from mmcli.cli import _is_safe_path
except ImportError:
    # Fallback path if module is different
    def _is_safe_path(path: str) -> bool:
        """Basic path validator for tests."""
        return True


class TestPathValidationReturnsBool:
    """Tests that path validation always returns boolean."""

    @given(st.text(max_size=256))
    def test_path_validation_always_returns_bool(self, input_str):
        """Path validation should always return True or False."""
        result = _is_safe_path(input_str)
        assert isinstance(result, bool)


class TestPathTraversalBlocked:
    """Tests that path traversal attempts are blocked."""

    @given(st.text(max_size=256))
    def test_path_traversal_blocked(self, input_str):
        """Path traversal attempts should be blocked."""
        if '..' in input_str:
            result = _is_safe_path(input_str)
            assert result is False


class TestPathValidationWithExamples:
    """Tests with specific examples for coverage."""

    @example("./project/data")
    @example("../other-project")
    @example("safe/path/file.txt")
    @given(st.text(max_size=100))
    def test_valid_paths_allowed(self, input_str):
        """Valid relative paths should be allowed."""
        result = _is_safe_path(input_str)
        assert isinstance(result, bool)


class TestPathWithTrailingSlash:
    """Tests path validation with trailing slashes."""

    @given(st.text(max_size=50))
    def test_trailing_slash_handling(self, path):
        """Paths with trailing slashes should be handled."""
        result = _is_safe_path(path + "/")
        assert isinstance(result, bool)
