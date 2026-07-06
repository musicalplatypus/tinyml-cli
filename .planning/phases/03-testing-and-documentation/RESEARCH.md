# Phase 3: Testing and Documentation - Research

**Researched:** 2026-07-06  
**Domain:** Python CLI testing, integration test fixtures, documentation  
**Confidence:** MEDIUM

## Summary

Phase 3 addresses issues identified in Phase 2's test execution:
1. Integration tests failing due to `tinyml_modelmaker` dependency not being available
2. E2E tests failing with "Invalid project path" errors related to temp directory cleanup
3. Test coverage gaps in non-command modules (builder, datasets)

**Key Finding:** The integration test failures are environmental - the code is correct but tests require external dependencies that aren't properly mocked or installed.

## Integration Test Failure Analysis

### Root Cause: Missing tinyml_modelmaker Mock

The failing tests (`test_info_lists_models`, `test_dry_run_generates_config`) all show:
```
AssertionError: mmcli info failed: ERROR: Cannot import tinyml_modelmaker
```

**Root cause:** Tests attempt to run `mmcli info` which internally calls subprocess to execute Python code that imports `tinyml_modelmaker`. The test environment doesn't have this package installed.

**Evidence from tests/test_cli_integration.py (lines 50-120):**
```python
def test_info_lists_models(test_project_dir):
    result = run_mmcli("info", "-m", "timeseries")
    assert result.returncode == 0
    # This fails because tinyml_modelmaker isn't installed in test env
```

**Solution approaches:**
1. **Mock the subprocess call** - Patch `subprocess.run` to return fake data
2. **Install tinyml_modelmaker as dev dependency** - Add to requirements-dev.txt
3. **Create a mock registry** - Simulate tinyml_modelmaker's API response

### Root Cause: Temp Directory Cleanup in E2E Tests

The e2e tests fail with:
```
AssertionError: dry-run failed: ERROR: Invalid project path: /private/var/f...
```

**Root cause:** The test creates a temp directory, but the mmcli tool validates paths and rejects `/private/var/folders/...` (macOS temp location) as potentially unsafe.

**Evidence from tests/test_e2e.py:**
```python
def test_dry_run_quantization_int_in_yaml(tmp_path):
    project_dir = tmp_path / "test_project"
    result = run_mmcli("train", "--dry-run", "-i", str(project_dir))
    # Fails because mmcli validates path format
```

**Solution approaches:**
1. **Update path validation logic** - Allow temp directory paths that are valid
2. **Use a different temp location** - Create test dirs in project root instead of system tmp
3. **Mock the path validation** - Patch `_is_safe_path()` for tests

## Test Coverage Analysis

### Current Coverage (from conftest.py fixtures)

| Module | Tests | Lines | Coverage Estimate |
|--------|-------|-------|-------------------|
| info.py | 22 | ~150 | High |
| analyze.py | 40 | ~200 | High |
| recommend.py | 22 | ~180 | Medium |
| deploy.py | 34 | ~250 | Medium |

### Missing Coverage Areas

| Component | Tests | Need |
|-----------|-------|------|
| builder.py | 0 | High - generates configs |
| datasets.py | 0 | Medium - dataset handling |
| report.py | 0 | Low - reporting only |
| about.py | 0 | Low - display only |

## Documentation Gaps

### Current State
- README.md exists (21KB)
- CONFIG_FILE_EXAMPLES.md created in Phase 2
- No API documentation

### Required Documentation

1. **API Reference** - Sphinx-generated from docstrings
2. **User Guide** - Common workflows and examples
3. **Troubleshooting** - Fix common error patterns

## Testing Strategy Recommendations

### Approach A: Mock External Dependencies (Recommended)
```python
# In tests/test_cli_integration.py
from unittest.mock import patch, MagicMock

@patch("mmcli.info._run_query")
def test_info_lists_models(mock_run_query):
    mock_run_query.return_value = {"models": ["model1", "model2"]}
    result = run_mmcli("info", "-m", "timeseries")
    assert "model1" in result.stdout
```

**Pros:** Fast, deterministic, no external dependencies  
**Cons:** May miss real-world integration issues

### Approach B: Dev Dependencies
```python
# requirements-dev.txt
-r requirements.txt
tinyml-modelmaker>=2024.1
```

**Pros:** Tests real behavior, catches integration issues  
**Cons:** Slower, environment-specific

### Hybrid Recommendation
- **Unit tests:** Mock external calls (fast, reliable)
- **Integration tests:** Run against mock server or test data
- **Smoke tests:** Optional, manual execution against real tinyml_modelmaker

## Code Examples

### Mocking Subprocess in Tests
```python
# Source: conftest.py (lines 100-150)
import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_tinyml_modelmaker():
    """Mock the tinyml_modelmaker subprocess calls."""
    with patch("mmcli.info._run_query") as mock_run_query:
        mock_run_query.return_value = {
            "models": ["test_model"],
            "devices": ["F28P55", "MSPM0G3507"],
            "tasks": ["timeseries_classification"]
        }
        yield mock_run_query
```

### Path Validation Fix Options

**Option 1: Update `_is_safe_path()` to allow temp dirs**
```python
# In mmcli/cli.py
def _is_safe_path(path: str) -> bool:
    """Check if path is safe for project operations."""
    # Normalize the path
    normalized = os.path.normpath(path)
    
    # Allow paths under current directory or absolute paths in /tmp
    if normalized.startswith(('.', '/tmp', '/private/var/folders')):
        return True
    
    # Reject path traversal attempts
    if '..' in os.path.relpath(normalized, '.'):
        return False
    
    return True
```

**Option 2: Use project-local temp dirs for tests**
```python
# In conftest.py
@pytest.fixture
def test_project_dir(tmp_path):
    """Create a project directory that passes path validation."""
    # Use a subdirectory of tmp_path that won't trigger validation issues
    return tmp_path / "workspace" / "project"
```

## Dependencies Verification

| Dependency | Required For | Status |
|------------|--------------|--------|
| pytest-mock | Unit tests with mocks | Already in pyproject.toml |
| tinyml_modelmaker | Integration tests | External, optional for dev |

## Next Steps

1. **03-01:** Fix integration test failures by mocking tinyml_modelmaker
2. **03-02:** Fix E2E temp directory issues by updating path validation or test setup
3. **03-03:** Add unit tests for builder, datasets modules
4. **03-06:** Generate API documentation

## Risks and Mitigations

| Risk | Impact |Mitigation |
|------|--------|-----------|
| tinyml_modelmaker changes break mocks | High | Keep mock responses in sync with actual API |
| Temp dir validation too strict | Medium | Update `_is_safe_path()` to allow standard temp locations |
| Test suite becomes slow with real deps | Low | Use mocking for unit tests, reserve integration tests for CI |
