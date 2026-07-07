# mmcli Security Model

## Threat Model (STRIDE)

### CLI Arguments
- **Spoofing:** Mitigated by input sanitization (`_sanitize_input()`)
- **Tampering:** Mitigated by shell=False and argument arrays
- **Denial of Service:** Mitigated by timeout enforcement and input length limits

### File System Operations
- **Repudiation:** Mitigated by path validation (`_is_safe_path()`)
- **Information Disclosure:** Limited to project directory access only
- **Denial of Service:** Mitigated by file size limits

### Subprocess Execution
- **Elevation of Privilege:** Not applicable - runs as current user
- **Tampering:** Mitigated by input sanitization before subprocess call

## Attack Vectors

1. **Command injection via CLI arguments**
   - Mitigation: `_sanitize_input()` strips dangerous characters (`;`, `$`, `` ` ``, `|`)
   - All subprocess calls use `shell=False` with argument arrays

2. **Path traversal via project path argument**
   - Mitigation: `_is_safe_path()` validates paths are within allowed directories
   - Blocks `..` sequences and absolute paths

3. **Environment variable manipulation**
   - Mitigation: Environment variables are sanitized before use
   - Only known-safe env vars (MMCLI_*) are accepted

4. **Buffer overflow**
   - Mitigation: All inputs have length limits (max 1024 characters)

## Security Features Checklist

- [x] Input validation on all user-provided strings
- [x] shell=False for all subprocess calls
- [x] Absolute path resolution with validation
- [x] Timeouts on subprocess execution
- [x] Output size limits to prevent DoS
- [x] Fuzz testing with hypothesis for edge case discovery

## Secure Coding Guidelines

### For Contributors

1. **Always sanitize user input** before processing:
   ```python
   from mmcli.cli import _sanitize_input
   
   safe_input = _sanitize_input(user_provided_string)
   ```

2. **Never use shell=True** in subprocess calls:
   ```python
   # Bad: vulnerable to command injection
   subprocess.run(command_string, shell=True)
   
   # Good: safe with argument arrays
   subprocess.run(['python', script_path], shell=False)
   ```

3. **Validate paths before file operations**:
   ```python
   from mmcli.cli import _is_safe_path
   
   if not _is_safe_path(project_path):
       raise ValueError("Invalid path")
   ```

4. **Set timeouts on subprocess execution** to prevent hanging.

## Known Limitations

- No built-in encryption for sensitive data
- Dependencies must be manually audited (use `scripts/scan-vulnerabilities.sh`)
- No automated security scanning in CI (planned)
