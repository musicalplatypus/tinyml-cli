# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in mmcli, please report it responsibly:

1. **DO NOT** open a public GitHub issue
2. Email the maintainers at security@mmcli.example.com
3. Include details about the vulnerability and reproduction steps

We aim to respond within 48 hours and fix critical issues within 7 days.

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.x     | :white_check_mark: |
| < 1.0   | :x:                |

## Security Features

- Input validation on all user-provided strings
- Subprocess execution with shell=False
- Path traversal protection via `_is_safe_path()`
- Environment variable sanitization
- Timeout enforcement on subprocess calls
- Fuzz testing with Hypothesis for edge case discovery

## Known Limitations

- No built-in encryption for sensitive data
- Dependencies must be manually audited (use `scripts/scan-vulnerabilities.sh`)
- No automated security scanning in CI (planned)

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
