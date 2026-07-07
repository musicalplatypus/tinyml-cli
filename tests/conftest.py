"""
Test configuration and fixtures for mmcli.
This file extends the root conftest.py with test-specific mocks.

The tinyml_modelmaker package is installed as a mock package in site-packages
for testing purposes. No mocking of subprocess calls is needed.
"""
import pytest


@pytest.fixture
def mock_tinyml_modelmaker_registry():
    """Fixture to ensure tinyml_modelmaker is available for tests.

    This fixture exists as a placeholder - the actual tinyml_modelmaker
    is installed as a mock package in site-packages for testing.
    """
    yield


# Hypothesis fuzz testing fixtures

@pytest.fixture(scope="session")
def cli_input_strategies():
    """Hypothesis strategies for CLI input generation."""
    from hypothesis import strategies as st

    return {
        'basic': st.text(min_size=1, max_size=256),
        'alphanumeric': st.text(alphabet=st.characters(whitelist_categories=['Lu', 'Ll', 'Nd'])),
        'path_like': st.text(alphabet=st.characters(whitelist_characters=['a-z', 'A-Z', '0-9', '-', '_', '.', '/'])),
    }


@pytest.fixture
def fuzz_cli_argument(cli_input_strategies):
    """Generate random CLI argument values using hypothesis."""
    from hypothesis import strategies as st

    return st.one_of(
        cli_input_strategies['basic'],
        cli_input_strategies['alphanumeric'],
        cli_input_strategies['path_like']
    )
