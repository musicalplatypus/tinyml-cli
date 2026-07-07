"""
Fuzz tests for input sanitization using hypothesis.
Automatically generates test cases to find edge cases and boundary conditions.
"""
import pytest
from hypothesis import given, strategies as st, example

# Import the sanitize function - it may be in cli module
try:
    from mmcli.cli import _sanitize_input
except ImportError:
    # Fallback if in different location
    def _sanitize_input(input_str: str) -> str:
        """Basic sanitizer for tests."""
        return input_str


class TestSanitizeDoesNotCrash:
    """Tests that sanitize never crashes on valid inputs."""

    @given(st.text(max_size=2048))
    def test_sanitize_does_not_crash(self, input_str):
        """Sanitize should never crash on any input."""
        result = _sanitize_input(input_str)
        assert isinstance(result, str)


class TestSanitizeBlocksCommandInjection:
    """Tests that malicious patterns are removed from inputs."""

    @given(st.text(max_size=256))
    def test_sanitize_blocks_semicolon(self, input_str):
        """Semicolons should be removed."""
        if ';' in input_str:
            result = _sanitize_input(input_str)
            assert ';' not in result

    @given(st.text(max_size=256))
    def test_sanitize_blocks_dollar_sign(self, input_str):
        """Dollar signs should be removed."""
        if '$' in input_str:
            result = _sanitize_input(input_str)
            assert '$' not in result

    @given(st.text(max_size=256))
    def test_sanitize_blocks_backtick(self, input_str):
        """Backticks should be removed."""
        if '`' in input_str:
            result = _sanitize_input(input_str)
            assert '`' not in result


class TestSanitizeEnforcesLengthLimit:
    """Tests that sanitize enforces maximum length."""

    @given(st.text(max_size=2048))
    def test_sanitize_enforces_length_limit(self, input_str):
        """Result should never exceed maximum length."""
        result = _sanitize_input(input_str)
        assert len(result) <= 1024


class TestSanitizeWithExamples:
    """Tests with specific examples for coverage."""

    @example("test")
    @example("my-project/data/file.txt")
    @example("path/to/safe/input")
    @given(st.text(max_size=100))
    def test_sanitize_preserves_safe_input(self, input_str):
        """Safe inputs should be preserved (within sanitization rules)."""
        result = _sanitize_input(input_str)
        # After sanitization, result should only contain safe characters
        import re
        sanitized_pattern = r'^[\w\-./_ ]*$'
        assert re.match(sanitized_pattern, result) is not None
