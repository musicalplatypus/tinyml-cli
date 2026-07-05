"""
Test configuration and fixtures for mmcli.
"""
import pytest
import tempfile
import os
from pathlib import Path
import psutil
import time

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def sample_config_data():
    """Provide sample configuration data for testing."""
    return {
        'common': {
            'project_name': 'test_project',
            'output_dir': '/tmp/test_output'
        },
        'training': {
            'epochs': 10,
            'batch_size': 32
        }
    }

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Provide mocked environment variables for testing."""
    monkeypatch.setenv('MMCLI_PYTHON', '/usr/bin/python3')
    monkeypatch.setenv('MMCLI_MODELMAKER', '/tmp/test_modelmaker')

@pytest.fixture
def performance_monitor():
    """Monitor system resources during test execution."""
    def _monitor():
        process = psutil.Process(os.getpid())
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()
        return {
            'cpu_percent': cpu_percent,
            'memory_mb': memory_info.rss / 1024 / 1024
        }
    return _monitor

@pytest.fixture
def baseline_performance():
    """Provide baseline performance measurements."""
    # This would be populated with actual baseline data
    return {
        'expected_cpu_usage': 50.0,
        'expected_memory_mb': 100.0,
        'expected_execution_time': 10.0
    }