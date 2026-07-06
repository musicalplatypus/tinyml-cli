"""
Test configuration and fixtures for mmcli.
"""
import sys
import pytest
import tempfile
import os
from pathlib import Path
import psutil
import time
from unittest import mock

# Try to import optional dependencies
try:
    import numpy
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_numpy():
    """Mock numpy module for testing without external dependencies."""
    fake_array = type('FakeArray', (), {
        'shape': (10,),
        '__getitem__': lambda self, key: 0,
        'load': lambda path, mmap_mode=None: (
            FakeArray() if mmap_mode == 'r' else FakeArray()
        )
    })

    class FakeNumpy:
        load = staticmethod(fake_array.load)

    with mock.patch.dict('sys.modules', {'numpy': FakeNumpy()}):
        yield


@pytest.fixture
def mock_pandas():
    """Mock pandas module for testing without external dependencies."""
    fake_df = type('FakeDataFrame', (), {
        'data': [],
        '__len__': lambda self: len(self.data),
        '__init__': lambda self, data=None: setattr(self, 'data', data if data else [])
    })

    with mock.patch.dict('sys.modules', {'pandas': fake_df}):
        yield


@pytest.fixture
def mock_filesystem_dependencies():
    """Mock all external filesystem dependencies (numpy, pandas)."""
    fake_array = type('FakeArray', (), {
        'shape': (10,),
        '__getitem__': lambda self, key: 0,
        'load': lambda path, mmap_mode=None: FakeArray()
    })

    class FakeNumpy:
        load = staticmethod(fake_array.load)

    fake_df = type('FakeDataFrame', (), {
        'data': [],
        '__len__': lambda self: len(self.data),
        '__init__': lambda self, data=None: setattr(self, 'data', data if data else [])
    })

    with mock.patch.dict('sys.modules', {
        'numpy': FakeNumpy(),
        'pandas': fake_df
    }):
        yield


@pytest.fixture
def mock_print():
    """Mock print function for testing."""
    with mock.patch("builtins.print") as mock_print:
        yield mock_print


@pytest.fixture
def require_numpy():
    """Skip test if numpy is not available."""
    pytest.importorskip("numpy")


@pytest.fixture
def require_pandas():
    """Skip test if pandas is not available."""
    pytest.importorskip("pandas")


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
