# Phase 4: Security Enhancements - Research

**Researched:** 2026-07-06  
**Domain:** Python CLI security, fuzz testing, vulnerability scanning  
**Confidence:** HIGH

## Summary

Phase 4 enhances the mmcli tool's security posture through:
1. Fuzz testing framework for command-line argument validation
2. Attack surface mapping and verification tests
3. Security documentation and threat modeling
4. Improved input validation patterns
5. Dependency vulnerability scanning

**Key Finding:** Existing code already has good security patterns (shell=False, path sanitization), but lacks automated fuzz testing and comprehensive attack surface documentation.

## Current Security Posture Analysis

### What's Already Secure (Established in Phase 1)

| Pattern | Location | Status |
|---------|----------|--------|
| Input sanitization (`_sanitize_input()`) | cli.py:105-127 | ✅ Implemented |
| Subprocess security (`shell=False`) | All subprocess calls | ✅ Implemented |
| Path validation (`_is_safe_path()`) | cli.py:_validate_args() | ✅ Implemented |
| Environment variable validation | All MMCLI_* handling | ✅ Implemented |

### Attack Surface Mapping

#### Command-Line Interface
- **Input vectors:** CLI arguments, environment variables
- **Current protection:** Input sanitization regex
- **Gaps:** No length limits on arguments, no character class restrictions beyond basic sanitization

#### File System Operations
- **Input vectors:** Project paths, file patterns
- **Current protection:** Path traversal blocking
- **Gaps:** No symlink following detection, no large file handling

#### Subprocess Execution
- **Input vectors:** Python executable path, command arguments
- **Current protection:** shell=False, argument arrays
- **Gaps:** No command timeout enforcement, no output size limits

### Threat Modeling (STRIDE Analysis)

| Component | Threat Type | Scenario | Mitigation |
|-----------|-------------|----------|------------|
| CLI args | Spoofing | Malicious flag values | Input sanitization ✅ |
| CLI args | Tampering | Command injection via args | Shell=False + arrays ✅ |
| File I/O | Repudiation | Unauthorized file access | Path validation ✅ |
| Subprocess | Elevation | Code execution via crafted input | Input validation ✅ |

## Fuzz Testing Framework

### Recommended Tools

| Tool | Purpose | Pros | Cons |
|------|---------|------|------|
| `hypothesis` | Python property-based testing | Integrates with pytest, good docs | Requires code changes |
| `python-afl` | American Fuzzy Lop integration | Full fuzzing coverage | More complex setup |
| `go-fuzz` | Binary fuzzing | Fast, proven tool | Requires compiled binaries |

**Recommendation:** Use `hypothesis` for Python-based fuzz testing, as it integrates well with pytest and requires minimal code changes.

### Fuzz Test Examples

```python
# Example: Fuzz test for CLI argument handling
from hypothesis import given, strategies as st
import subprocess

@given(st.text(max_size=1024))
def test_cli_argument_sanitize(input_str):
    """Test that sanitized inputs don't cause crashes."""
    result = _sanitize_input(input_str)
    # Should not raise, should return valid string
    assert isinstance(result, str)
    assert len(result) <= 1024

@given(st.text(alphabet=st.characters(whitelist_categories=['Lu', 'Ll'])))
def test_command_injection_blocked(input_str):
    """Test that command injection attempts are blocked."""
    # Malicious patterns should be filtered
    sanitized = _sanitize_input(input_str)
    assert ';' not in sanitized
    assert '$' not in sanitized
```

## Attack Surface Testing

### Test Categories

1. **Input Length Limits**
   - Extremely long argument strings
   - Buffer overflow attempts

2. **Special Character Handling**
   - Shell metacharacters: `; | & $ ( )`
   - Escape sequences: `\x00 \xff`
   - Unicode edge cases: UTF-8 overlong encodings

3. **Path Traversal Variants**
   - Standard: `../`
   - Encoded: `%2e%2f`
   - Mixed: `..\\/`

4. **Environment Variable Injection**
   - Malicious MMCLI_* values
   - PATH manipulation attempts

## Input Validation Improvements

### Current Limitations

```python
# Current _sanitize_input() limitations:
def _sanitize_input(input_str: str) -> str:
    # Only removes non-word chars except space, dot, dash, underscore
    sanitized = re.sub(r'[^\w\-./_ ]', '', input_str)
    
    # No length limits (only truncates to 1024)
    if len(sanitized) > 1024:
        sanitized = sanitized[:1024]
```

### Proposed Improvements

```python
def _sanitize_input(input_str: str, max_length: int = 512) -> str:
    """Enhanced input sanitization."""
    # Early length check to prevent DoS
    if len(input_str) > 1024:
        raise ValueError("Input too long")
    
    # Remove shell metacharacters
    sanitized = re.sub(r'[;\|&$`]', '', input_str)
    
    # Limit component lengths (prevent path traversal via component names)
    parts = sanitized.split('/')
    for part in parts:
        if len(part) > 64:
            raise ValueError("Component too long")
    
    return sanitized
```

## Dependency Security

### Current Dependencies

| Package | Purpose | Vulnerable? |
|---------|---------|-------------|
| PyYAML | Config parsing | Check CVEs |
| pytest | Testing | Not in prod |

### Recommended Scans

1. **Static Analysis:** `pip-audit`, `safety check`
2. **Dependency Graph:** `pipdeptree` to identify transitive deps
3. **Vulnerability DB:** Use OSV database for up-to-date CVE info

## Security Documentation

### Required Documents

1. **Security Model** - Documented threat model with STRIDE analysis
2. **Secure Coding Guidelines** - Patterns to follow/avoid
3. **Dependency Policy** - How dependencies are vetted
4. **Incident Response** - Process for security issues

## Code Examples

### Secure Subprocess Pattern (Already Implemented)

```python
# mmcli/cli.py - Already follows secure pattern
result = subprocess.run(
    [sanitized_python, runner_script, yaml_path],
    check=False,
    shell=False,  # SECURITY: Prevent command injection
    timeout=300,   # Add timeout to prevent hangs
)
```

### Secure File I/O Pattern

```python
# Recommended pattern for Phase 4
def read_project_file(path: str) -> str:
    """Safely read a project file."""
    sanitized = _sanitize_input(path)
    
    # Resolve and validate path
    full_path = os.path.abspath(sanitized)
    base_dir = os.path.abspath('.')
    
    if not full_path.startswith(base_dir):
        raise ValueError("Path traversal attempt")
    
    with open(full_path, 'r') as f:
        return f.read()
```

## Next Steps

1. **04-01:** Implement fuzz testing framework using hypothesis
2. **04-02:** Create attack surface tests for all commands
3. **04-03:** Document security model and coding guidelines
4. **04-04:** Improve input validation with length limits
5. **04-05:** Integrate dependency vulnerability scanning

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Fuzz tests slow CI | Low | Run fuzz tests in dedicated job, not PR checks |
| New validation breaks existing code | Medium | Gradual rollout with deprecation warnings |
| Dependency vulnerabilities found | High | Immediate update required, backport to stable |

## Dependencies Verification

| Dependency | Required For | Status |
|------------|--------------|--------|
| hypothesis | Fuzz testing framework | dev-only |
| pytest | Testing infrastructure | Already installed |
| pip-audit | Dependency scanning | dev-only |
