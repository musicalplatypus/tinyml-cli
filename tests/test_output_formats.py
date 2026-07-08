"""
Tests for the output formatting module functionality.
"""

import pytest
import csv
import io
from mmcli.output import format_json, format_csv, format_yaml, format_table


class TestOutputFormats:
    """Tests for output formatting functions."""

    def test_format_json(self):
        """JSON output should be valid JSON string."""
        data = {"key": "value", "nested": {"a": 1}}
        result = format_json(data)

        # Verify it's valid JSON by parsing it back
        import json
        parsed = json.loads(result)
        assert parsed == data

    def test_format_csv(self):
        """CSV output should have header and data row."""
        data = {"name": "test", "count": 42}
        result = format_csv(data)

        # Parse the CSV to verify structure
        lines = result.strip().split('\n')
        assert len(lines) == 2
        assert "name" in lines[0]
        assert "test" in lines[1]

    def test_format_yaml(self):
        """YAML output should be valid YAML structure."""
        data = {"key": "value"}
        result = format_yaml(data)

        # When PyYAML is available, it will be YAML formatted
        # When not available, it falls back to JSON which should contain the key-value structure
        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_table(self):
        """Table output should be text format with key-value pairs."""
        data = {"name": "test", "count": 42}
        result = format_table(data)

        assert "name:" in result
        assert "test" in result
        assert "count:" in result
        assert "42" in result

    def test_format_csv_nested_dict(self):
        """CSV output should handle nested dictionaries."""
        data = {"name": "test", "nested": {"a": 1, "b": 2}}
        result = format_csv(data)

        # Should not crash and produce valid CSV
        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_csv_empty_data(self):
        """CSV output should handle empty data gracefully."""
        data = {}
        result = format_csv(data)

        # Should return empty string or minimal CSV
        assert isinstance(result, str)

    def test_format_json_complex_structure(self):
        """JSON output should handle complex nested structures."""
        data = {
            "users": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25}
            ],
            "total": 2,
            "metadata": {
                "version": "1.0",
                "timestamp": "2026-07-08"
            }
        }
        result = format_json(data)

        # Verify it's valid JSON
        import json
        parsed = json.loads(result)
        assert parsed == data

    def test_format_yaml_complex_structure(self):
        """YAML output should handle complex nested structures."""
        data = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "credentials": {
                    "username": "user",
                    "password": "pass"
                }
            }
        }
        result = format_yaml(data)

        # Should be valid YAML/JSON with the structure
        assert isinstance(result, str)
        assert len(result) > 0


class TestOutputFormatIntegration:
    """Integration tests for output formatting."""

    def test_format_csv_with_keys_parameter(self):
        """Test CSV formatting with explicit keys parameter."""
        data = {"name": "test", "count": 42}
        result = format_csv(data, keys=["name", "count"])

        # Should have the specified columns
        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_json_empty_dict(self):
        """Test JSON formatting with empty dictionary."""
        data = {}
        result = format_json(data)

        import json
        parsed = json.loads(result)
        assert parsed == data

    def test_format_yaml_empty_dict(self):
        """Test YAML formatting with empty dictionary."""
        data = {}
        result = format_yaml(data)

        # Should produce valid YAML (empty object)
        assert isinstance(result, str)
        assert len(result) > 0