# Phase 2.5: Testing Coverage for Phase 2 - Research

**Researched:** 2026-07-05
**Domain:** Python testing (pytest/mmcli)
**Confidence:** HIGH

## Summary

Phase 2.5 focuses on adding comprehensive test coverage for the four advanced commands (`info`, `analyze`, `recommend`, `deploy`) implemented in Phase 2. Research reveals:

**Primary recommendation:** Add unit and integration tests for each advanced command using pytest with appropriate mocking for external dependencies (tinyml_modelmaker subprocess, file I/O). Create a dedicated test file per command following existing patterns.

**Key findings:**
1. **Existing test infrastructure:** The project uses pytest with coverage reporting, conftest.py fixtures already defined
2. **Test coverage gap:** Current tests focus on `train`, `init`, and e2e pipeline; no specific tests for advanced commands
3. **Mocking strategy needed:** External dependencies (tinyml_modelmaker registry, modelzoo examples) must be mocked
4. **Environment isolation:** Tests need to handle MMCLI_PYTHON environment variable properly

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Test infrastructure setup | API / Backend | — | pytest configuration, fixtures, conftest.py |
| Command unit tests | Frontend Server | — | Test CLI argument parsing, command dispatch |
| Integration tests | API / Backend | Browser (if web UI) | Test full workflows with mocked dependencies |
| Security tests | Security | API / Backend | Test input validation, path traversal protection |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest | - | Testing framework | Already configured in pyproject.toml and conftest.py |
| Python stdlib (unittest.mock) | - | Mocking external dependencies | Standard library, no additional dependency |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest-cov | - | Coverage reporting | Already configured in pytest.ini |

**Installation:**
```bash
pip install pytest>=7.0  # Already available via requirements.txt
```

**Version verification:**
- pytest: Configured in pyproject.toml [tool.pytest.ini_options]
- Python stdlib (mock): Available by default

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pytest + mock | nose2, unittest | More verbose; pytest is already established |

## Package Legitimacy Audit

> **Note:** Phase 2.5 uses only pytest (already installed) and Python stdlib mocking.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| pytest | PyPI | 14+ years | 80M+/mo | github.com/pytest-dev/pytest | [OK] | Approved |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         mmcli CLI                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Subcommands: init, train, compile, run, info, analyze, recommend   │
└──────────────────┬──────────────────────────────────────────────────┘
                   │
          ┌────────┴──────────┐
          │                   │
    [subprocess]         [file I/O]
          │                   │
    ╔═════▼════╗        ╔════▼════╗
    ║ tinyml-  ║        ║ dataset ║
    ║modelmaker║        ║ files   ║
    ╚════▲═════╝        ╚════▲════╝
         │                   │
    ┌────┴────┐          ┌───┴────┐
    │  tests/ │          │  tests/│
    │test_info│          │test_   │
    │.py      │          │analyze │
    └─────────┘          │.py     │
                         └────────┘

Test Strategy:
- Mock subprocess calls to tinyml_modelmaker
- Mock file I/O for dataset operations
- Fixture-based project setup for integration tests
```

### Recommended Project Structure
```
tests/
├── __init__.py
├── test_info.py          # NEW: info command unit/integration tests
├── test_analyze.py       # NEW: analyze command unit/integration tests
├── test_recommend.py     # NEW: recommend command unit/integration tests
├── test_deploy.py        # NEW: deploy command unit/integration tests
├── test_integration.py   # EXISTING: End-to-end workflow tests
└── conftest.py           # SHARED: Fixtures and configuration

# Existing test files to extend/validate:
test_e2e.py               # Full pipeline with real dependencies
test_cli_integration.py   # CLI argument parsing
```

### Pattern 1: Mock Subprocess for tinyml_modelmaker
**What:** Mock the subprocess call to tinyml_modelmaker in info command

**When to use:** Any test that would call `mmcli info` which runs Python subprocess

**Example:**
```python
# Source: mmcli/info.py (lines 112-146)
def _run_query(python_exe: str, script: str) -> dict:
    result = subprocess.run(
        [python_exe, "-c", script],
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)

# Test pattern
@patch('mmcli.info.subprocess.run')
def test_info_query_success(mock_run):
    mock_run.returncode = 0
    mock_run.return_value.stdout = '{"module": "timeseries", ...}'
    result = info._run_query("/usr/bin/python3", "test script")
    assert result["module"] == "timeseries"
```

### Pattern 2: Temporary Directory Fixtures
**What:** Use pytest's temp_dir fixture for file-based tests

**When to use:** Tests that need to create dataset directories, config files, etc.

**Example:**
```python
# Source: conftest.py (lines 11-16)
@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

# Usage in tests
def test_analyze_classes_layout(temp_dir):
    ds_path = os.path.join(temp_dir, "dataset")
    classes_dir = os.path.join(ds_path, "classes", "class_a")
    os.makedirs(classes_dir)
    # Create test data files...
```

### Pattern 3: Parameterized Tests for Multiple Modules/Tasks
**What:** Test the same logic across different modules and task types

**When to use:** Commands support multiple modules (timeseries, vision, audio) or task types

**Example:**
```python
# Source: test_cli_integration.py (lines 21-27)
TASK_TYPES_TIMESERIES = [
    ("timeseries", "generic_timeseries_classification"),
    ("timeseries", "generic_timeseries_regression"),
]

@pytest.mark.parametrize("module,task_type", TASK_TYPES_TIMESERIES)
def test_info_lists_models(module, task_type):
    # Test logic...
```

### Pattern 4: Environment Variable Isolation
**What:** Mock environment variables for tests that depend on MMCLI_ vars

**When to use:** Tests that run commands requiring external tool paths

**Example:**
```python
# Source: conftest.py (lines 32-36)
@pytest.fixture
def mock_env_vars(monkeypatch):
    monkeypatch.setenv('MMCLI_PYTHON', '/usr/bin/python3')
    monkeypatch.setenv('MMCLI_MODELMAKER', '/tmp/test_modelmaker')

# Usage
def test_info_with_mocked_env(mock_env_vars, monkeypatch):
    # Environment is set up for this test
```

### Anti-Patterns to Avoid

- **Testing with real tinyml_modelmaker:** Tests should be fast and deterministic - use mocks instead of calling real modelmaker
- **Shared state between tests:** Each test should create its own temp directory, don't share data
- **Hardcoded paths in tests:** Use pytest fixtures or `os.path.join` for cross-platform compatibility
- **Ignoring subprocess errors:** Verify exit codes and stderr handling in error cases

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Test fixture management | Custom setup/teardown | pytest fixtures | Standard, reusable, automatic cleanup |
| Mocking subprocess calls | Manual patching | unittest.mock.patch | Built-in, handles context properly |
| Temporary file creation | manual tempfile.mktemp | temp_dir fixture | Automatic cleanup, cross-platform |
| Environment variable isolation | os.environ manipulation | monkeypatch fixture | Automatic restoration |

**Key insight:** The project's pytest configuration is already well-set up. Phase 2.5 should extend existing patterns rather than creating new testing infrastructure.

## Common Pitfalls

### Pitfall 1: Real Subprocess Calls in Tests
**What goes wrong:** Tests become slow (minutes) and flaky due to external dependency calls

**Why it happens:** Forgetting to patch subprocess.run when testing commands that call tinyml_modelmaker

**How to avoid:** Always mock subprocess for commands that delegate to external Python processes. Use `@patch('mmcli.info.subprocess.run')` decorator.

**Warning signs:** Tests take > 5 seconds, fail without tinyml_modelmaker installed

### Pitfall 2: Shared State Between Tests
**What goes wrong:** Test A modifies files in temp directory, Test B reads those modifications unexpectedly

**Why it happens:** Using module-level temp directories or not cleaning up after tests

**How to avoid:** Use pytest's `temp_dir` fixture for each test function. Never use global variables for test state.

**Warning signs:** Tests pass individually but fail when run together, flaky test results

### Pitfall 3: Path Hardcoding
**What goes wrong:** Tests work on Linux/macOS but fail on Windows due to path separator differences

**Why it happens:** Using string concatenation like `"/tmp/test_" + name` instead of os.path.join()

**How to avoid:** Always use `os.path.join()` for paths. For temp files, use the temp_dir fixture.

**Warning signs:** Tests work on developer machine but fail in CI/CD

### Pitfall 4: Testing Error Output Parsing
**What goes wrong:** Tests break when error message format changes slightly

**Why it happens:** Testing exact string matches of stderr output which is not API-stable

**How to avoid:** Test return codes and logical outcomes (e.g., "file exists") rather than exact error messages. Mock subprocess to control exactly what's returned.

## Code Examples

### Command Output Validation Pattern
```python
# Source: test_cli_integration.py (lines 60-71)
class TestInfoCommand:
    @pytest.mark.parametrize("module,task_type", TASK_TYPES)
    def test_info_lists_models(self, module, task_type):
        rc, stdout, stderr = _run_cli(
            "info", "-m", module, "-t", task_type, "-d", "F28P55"
        )
        output = stdout + stderr
        assert rc == 0, f"mmcli info failed: {output}"
        # Validate expected content exists
```

### Dataset Layout Testing Pattern
```python
# Based on mmcli/analyze.py patterns
def test_analyze_files_layout(temp_dir):
    """Test analyze with files/ layout (no classes subdirectory)."""
    project_path = os.path.join(temp_dir, "project")
    dataset_path = os.path.join(project_path, "dataset")
    files_dir = os.path.join(dataset_path, "files")
    os.makedirs(files_dir)
    
    # Create test data
    sample_csv = os.path.join(files_dir, "sample.csv")
    with open(sample_csv, "w") as f:
        f.write("1,2,3\n4,5,6\n")
    
    result = analyze.analyze_dataset(dataset_path)
    assert result["layout"] == "files"
    assert result["total_samples"] == 2
```

### Environment Variable Handling Test Pattern
```python
# Source: test_environment_integration.py pattern
def test_deploy_checks_sdk_with_mocked_env(monkeypatch, tmp_path):
    """Test deploy SDK check when MMCLI_MODELZOO_PATH is set."""
    monkeypatch.setenv("MMCLI_MODELZOO_PATH", str(tmp_path))
    
    # Test SDK detection with mocked modelzoo path
    result = deploy.check_sdk("F28P55")
    assert "c2000" in str(result.get("family", "")).lower()
```

### Full Integration Test Pattern (with skipping)
```python
# Source: test_e2e.py pattern for slow tests
@pytest.mark.e2e  # Mark as slow, skip by default
def test_full_workflow(temp_dir):
    """Full init → analyze → recommend workflow."""
    project = _create_project(temp_dir)
    
    # Step 1: Init
    rc, _, _ = _run("init", "-t", "arc_fault", "-p", project)
    assert rc == 0
    
    # Step 2: Analyze
    rc, out, _ = _run("analyze", "-i", project)
    assert rc == 0
    assert "size bucket" in out.lower()
    
    # Continue with remaining steps...
```

## Testing Strategy

### Unit Tests per Command
| Command | Test File | Coverage |
|---------|-----------|----------|
| `info` | test_info.py | Registry query, JSON parsing, output formatting |
| `analyze` | test_analyze.py | Dataset analysis (classes/files layout), file parsing |
| `recommend` | test_recommend.py | Scoring algorithm, modelzoo path resolution |
| `deploy` | test_deploy.py | SDK detection, artifact discovery, project creation |

### Integration Tests
1. **End-to-end workflow:** `init → analyze → recommend → train`
2. **Error recovery:** Invalid inputs, missing files, subprocess failures
3. **Cross-command validation:** Output from one command feeds into next

### Security Test Cases
| Attack Vector | Test Case |
|---------------|-----------|
| Path traversal | `../../etc/passwd` in project path |
| Shell injection | `; rm -rf /` in arguments |
| Environment variable injection | MMCLI_PYTHON with shell metacharacters |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | pytest.ini, conftest.py |
| Quick run command | `pytest tests/ -m "not e2e" -x` |
| Full suite command | `pytest tests/ --cov=mmcli` |

### Phase Requirements → Test Map
| Requirement | Test Type | Automated Command | File |
|-------------|-----------|-------------------|------|
| info lists devices | unit | pytest tests/test_info.py::TestInfoCommand::test_info_lists_devices -x | NEW |
| analyze dataset structure | unit | pytest tests/test_analyze.py::TestAnalyzeDataset -x | NEW |
| recommend scores models correctly | unit | pytest tests/test_recommend.py::TestRecommendScoring -x | NEW |
| deploy SDK detection works | unit | pytest tests/test_deploy.py::TestDeploySDKCheck -x | NEW |

### Sampling Rate
- **Per task commit:** `pytest tests/test_info.py tests/test_analyze.py tests/test_recommend.py tests/test_deploy.py -x`
- **Per wave merge:** `pytest tests/ --cov=mmcli --cov-report=term-missing`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_info.py` — info command unit and integration tests
- [ ] `tests/test_analyze.py` — analyze command tests
- [ ] `tests/test_recommend.py` — recommend command tests  
- [ ] `tests/test_deploy.py` — deploy command tests
- [ ] `tests/test_security.py` extension — security-focused tests for new commands

*(If no gaps: "None — existing test infrastructure covers all phase requirements")*

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| pytest | All tests | ✅ | 7.x (configured) | — |
| Python 3.10+ | Runtime | ✅ | 3.14.6 | — |
| tinyml_modelmaker | info command subprocess | ⚠️ Requires MMCLI_PYTHON | — | Skip tests requiring real modelmaker |

**Missing dependencies with no fallback:** none
**Missing dependencies with fallback:** Tests requiring real tinyml_modelmaker can be skipped (marked `@pytest.mark.skipif`)

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | N/A (CLI only) |
| V3 Session Management | no | N/A |
| V4 Access Control | yes | Path validation tests, input sanitization |
| V5 Input Validation | yes | Security tests for injection vectors |
| V6 Cryptography | no | Not applicable |

### Known Threat Patterns for mmcli

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal | Tamping | `_is_safe_path()` validation, path normalization |
| Command injection | Tampering | `shell=False` in subprocess, input sanitization |
| Environment variable injection | Spoofing | Sanitize MMCLI_* env vars before use |

## Next Steps

1. **Create test files:**
   - `tests/test_info.py` — info command tests (5-10 tests)
   - `tests/test_analyze.py` — analyze command tests (8-12 tests)
   - `tests/test_recommend.py` — recommend command tests (6-10 tests)
   - `tests/test_deploy.py` — deploy command tests (8-15 tests)

2. **Add integration test:** Extend existing e2e tests to include the four new commands

3. **Add security tests:** Create test cases for path traversal, shell injection in command arguments

4. **Documentation:** Update README with testing instructions and coverage status

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - pytest configuration verified
- Architecture: HIGH - existing patterns documented and understood
- Pitfalls: HIGH - from Phase 1 security lessons and test review

**Research date:** 2026-07-05  
**Valid until:** 2026-08-05 (30 days)
