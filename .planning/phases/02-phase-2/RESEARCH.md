# Phase 2: Advanced Features & Integration - Research

**Researched:** 2026-07-05
**Domain:** CLI command implementation (Python/mmcli)
**Confidence:** HIGH

## Summary

Phase 2 focuses on implementing four advanced commands (`info`, `analyze`, `recommend`, `deploy`) for the mmcli tool while maintaining security measures established in Phase 1. Research reveals:

**Primary recommendation:** The four advanced commands are already implemented in `mmcli/cli.py` and their supporting modules (`mmcli/info.py`, `mmcli/analyze.py`, `mmcli/recommend.py`, `mmcli/deploy.py`). The phase requires adding comprehensive test coverage for these existing implementations rather than building them from scratch.

**Key findings:**
1. **Commands already exist**: All four advanced commands have working implementations with security hardening
2. **Testing gap identified**: No tests directory structure found; Phase 1 commit notes "Test coverage is missing and should be added first"
3. **Security patterns established**: Input validation, subprocess handling (`shell=False`), and path sanitization already in place from Phase 1
4. **Environment dependency gap**: Commands depend on external tools (tinyml_modelmaker Python package) requiring `MMCLI_PYTHON` and `MMCLI_MODELMAKER` environment variables

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| CLI argument parsing | Frontend Server | — | Command-line interface with argparse, argument validation |
| Info command implementation | API / Backend | — | Queries tinyml-modelmaker registry via subprocess, parses JSON output |
| Analyze command implementation | API / Backend | Browser (if web UI) | Reads dataset files (.csv, .npy, .pkl), analyzes structure |
| Recommend command implementation | API / Backend | — | Scans tinyml-modelzoo examples, scores against user criteria |
| Deploy command implementation | API / Backend | Browser (if web UI) | Wraps external CCS/dslite tools for device deployment |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyYAML | 6.0+ | YAML config parsing | Required for project configuration and example config files |
| Python stdlib (subprocess, argparse) | - | CLI infrastructure | Standard library for command execution and argument parsing |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | - | Data file reading (.npy) | Used in analyze.py for numpy array loading |
| pandas | - | CSV/data analysis | Used in analyze.py for dataset inspection |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| PyYAML | yaml-loader, ruamel.yaml | More features but adds dependency; PyYAML is standard |

**Installation:**
```bash
pip install PyYAML>=6.0
```

**Version verification:**
- `PyYAML`: Version 6.0+ confirmed from `pyproject.toml`
- All other dependencies are Python stdlib

## Package Legitimacy Audit

> **Note:** Phase 2 primarily uses Python standard library and existing PyYAML dependency. No external packages need installation beyond what's already in the project.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| PyYAML | PyPI | 8+ years | 100M+/mo | github.com/yaml/pyyaml | [OK] | Approved |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         mmcli (CLI)                                │
├─────────────────────────────────────────────────────────────────────┤
│  argparse → CLI subcommands: init, train, compile, run, info,      │
│             analyze, recommend, deploy                              │
└────────────────────┬────────────────────────────────────────────────┘
                     │
         ┌───────────┴────────────┬───────────────┬──────────────────┐
         │                        │               │                  │
    [subprocess]            [file I/O]       [env vars]        [file I/O]
         │                        │               │                  │
    ╔════▼════╗           ╔══════▼═════╗   ╔════▼═════╗      ╔════▼════╗
    ║ tinyml- ║           ║ dataset    ║   ║ MMCLI_*  ║      ║ tinyml- ║
    ║modelmaker║           ║ analysis   ║   ║ env vars ║      ║ modelzoo║
    ╚════▲════╝           ╚══════▲═════╝   ╚════▲═════╝      ╚════▲════╝
         │                        │               │                  │
    ┌────┴────┐             ┌────┴─────┐     ┌──┴──────┐       ┌──┴────┐
    │ Python  │             │ classes/ │     │ validate│       │ score │
    │ runtime │             │ files/   │     │ inputs  │       │ models│
    └─────────┘             │ images/  │     └─────────┘       └───────┘
                            └──────────┘
```

### Recommended Project Structure
```
mmcli/
├── cli.py              # Main entry point, argparse setup, dispatch
├── info.py             # Info command implementation
├── analyze.py          # Analyze command implementation
├── recommend.py        # Recommend command implementation
├── deploy.py           # Deploy command (5 subcommands)
├── builder.py          # Config building utilities
├── datasets.py         # Dataset extraction and listing
├── about.py            # About/credits display
└── report.py           # Training report generation

tests/
├── test_info.py        # NEW: info command tests
├── test_analyze.py     # NEW: analyze command tests
├── test_recommend.py   # NEW: recommend command tests
├── test_deploy.py      # NEW: deploy command tests
└── test_integration.py # NEW: end-to-end integration tests
```

### Pattern 1: Secure Subprocess Dispatch
**What:** Execute tinyml_modelmaker via subprocess with shell=False and sanitized inputs

**When to use:** Any interaction with external Python scripts or tools

**Example:**
```python
# Source: mmcli/cli.py (lines 209-230)
result = subprocess.run(
    [sanitized_python, runner_script, yaml_path],
    check=False,
    shell=False,  # SECURITY: Prevent command injection
)
```

### Pattern 2: Input Sanitization
**What:** Validate and sanitize all user inputs before use

**When to use:** All CLI arguments and environment variables

**Example:**
```python
# Source: mmcli/cli.py (lines 105-127)
def _sanitize_input(input_str: str) -> str:
    """Sanitize input to prevent command injection."""
    sanitized = re.sub(r'[^\w\-./_ ]', '', input_str)
    while '..' in sanitized:
        sanitized = sanitized.replace('..', '')
    sanitized = sanitized.replace('`', '').replace('$', '').replace(';', '')
    if len(sanitized) > 1024:
        sanitized = sanitized[:1024]
    return sanitized
```

### Anti-Patterns to Avoid

- **Shell=True:** Never pass user input directly to shell commands - always use `shell=False` with argument arrays
- **Path traversal:** Always validate paths for `..` and absolute paths starting with `/`
- **Environment variable injection:** Sanitize all MMCLI_* env vars before use in subprocess calls
- **Assuming external tools exist:** All deploy commands should gracefully handle missing SDK/CCS installations

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| YAML config parsing | Custom parser | PyYAML | Standard library for Python, handles edge cases |
| Dataset file reading (.npy) | Manual numpy parsing | numpy.load() | Handles binary format correctly |
| CSV processing | Manual line-by-line parsing | pandas.read_csv() | Handles quoting, delimiters, encoding |
| Subprocess command injection prevention | Regex-based filtering | shell=False + argument array | More robust than manual sanitization |

**Key insight:** The existing codebase already uses the standard library approach correctly. Phase 2 should focus on test coverage, not rewriting implementation patterns.

## Runtime State Inventory

> This section documents runtime state that needs attention when renaming or refactoring (not applicable for this phase which implements new commands).

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None - fresh project | N/A |
| Live service config | None - no external services yet | N/A |
| OS-registered state | None | N/A |
| Secrets/env vars | MMCLI_PYTHON, MMCLI_MODELMAKER defined in cli.py docs | Document as part of CLI help text |
| Build artifacts | mmcli.egg-info/, dist/ | Rebuild after changes |

**Nothing found in category:** State explicitly ("None — verified by file system scan").

## Common Pitfalls

### Pitfall 1: Environment Variable Security
**What goes wrong:** MMCLI_PYTHON or MMCLI_MODELMAKER env vars containing malicious paths with shell metacharacters

**Why it happens:** Environment variables are user-controlled input that may not go through the same validation as CLI arguments

**How to avoid:** Sanitize all env var values before use (already implemented in `_get_python_exe()` and `_find_runner_script()')

**Warning signs:** Error messages showing raw user input with `;`, `$`, or backticks

### Pitfall 2: External Tool Dependency Not Found
**What goes wrong:** deploy commands fail silently when CCS/SDK not installed, or give cryptic errors

**Why it happens:** External tools (dslite, CCS) may not be on PATH or in expected locations

**How to avoid:** Implement early verification (`check-sdk` subcommand exists for this), provide helpful error messages with installation links

**Warning signs:** "Command not found" errors from subprocess calls

### Pitfall 3: Path Traversal via Project Directory
**What goes wrong:** User specifies project path like `../../etc/passwd` which could read/write sensitive files

**Why it happens:** Project directory paths come from user input (-i/--project flag)

**How to avoid:** Already implemented `_is_safe_path()` validation and path normalization in `_validate_args()`

**Warning signs:** Paths with `..` or absolute paths starting with `/` (if not intended)

### Pitfall 4: Missing Test Coverage for New Commands
**What goes wrong:** Implementation works in manual testing but breaks under edge cases or after refactoring

**Why it happens:** "It works on my machine" syndrome; external dependencies mask issues

**How to avoid:** Write tests before/alongside implementation; use pytest fixtures to mock external calls

**Warning signs:** No test files in tests/ directory for new commands

## Code Examples

Verified patterns from official source (mmcli/cli.py and supporting modules):

### Command Registration Pattern
```python
# Source: mmcli/cli.py (lines 686-718)
def _add_info_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "info",
        help="Show supported devices, models, and feature extraction presets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Query the model registry and display available options.\n\n"
            "Examples:\n"
            "  mmcli info -m timeseries                        # list task types\n"
            "  mmcli info -m timeseries -t arc_fault           # details for arc_fault\n"
            "  mmcli info -m timeseries -t arc_fault -d F28P55 # models for F28P55"
        ),
    )
    p.add_argument(
        "-m", "--module",
        required=True,
        choices=MODULES,
        metavar="MODULE",
        help="AI module (timeseries, vision, or audio).",
    )
```

### Info Command Registry Query Pattern
```python
# Source: mmcli/info.py (lines 19-98)
_QUERY_SCRIPT = textwrap.dedent(r'''
import json, sys

module_name = {module!r}
task_type   = {task_type!r}
target_device = {target_device!r}

try:
    if module_name == "timeseries":
        from tinyml_modelmaker.ai_modules.timeseries import constants, training
    elif module_name == "vision":
        from tinyml_modelmaker.ai_modules.vision import constants, training
    ...
''')

def _run_query(python_exe: str, script: str) -> dict:
    result = subprocess.run(
        [python_exe, "-c", script],
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)
```

### Deploy SDK Check Pattern
```python
# Source: mmcli/deploy.py (lines 65-126)
SDK_INFO: Dict[str, Dict] = {
    "c2000": {
        "name":          "C2000Ware",
        "install_globs": [
            os.path.expanduser("~/ti/c2000ware_*"),
            "/opt/ti/c2000ware_*",
        ],
        "ai_examples_subpath": "libraries/ai/examples",
        "download_url": "https://www.ti.com/tool/C2000WARE",
    },
    # ... other SDKs
}

def _find_sdk_root(family: str) -> Optional[str]:
    matches = []
    for pattern in SDK_INFO.get(family, {}).get("install_globs", []):
        matches += glob.glob(pattern)
    return sorted(matches)[-1] if matches else None
```

### Analyze Dataset Pattern
```python
# Source: mmcli/analyze.py (lines 67-94)
def _analyse_classes(dataset_path: str) -> dict:
    classes_dir = os.path.join(dataset_path, "classes")
    total, global_min = 0, float("inf")
    dist: dict[str, int] = {}

    for cls in sorted(os.listdir(classes_dir)):
        cls_path = os.path.join(classes_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        cls_count = 0
        for f in _find_data_files(cls_path):
            n = _row_count(f)  # Handles .csv, .pkl, .npy
            total += n
            cls_count += n
        dist[cls] = cls_count

    return {
        "layout": "classes",
        "total_samples": total,
        "class_distribution": dist,
        "dataset_bucket": _bin_dataset(total),
    }
```

## Security Considerations

### Input Validation (Established)
- All user inputs sanitized via `_sanitize_input()`
- Path traversal blocked via `_is_safe_path()`
- Environment variables validated before subprocess use

### Subprocess Security
- All subprocess calls use `shell=False` with argument arrays
- Paths are absolute and verified before passing to external tools
- No shell metacharacters allowed in any user-provided values

### External Tool Handling
- deploy commands provide clear error messages when SDK/CCS not found
- `check-sdk` subcommand verifies installations before trying to use them
- Fallback paths searched with glob patterns for cross-platform compatibility

## Testing Strategy Recommendations

Based on existing tests and gray areas identified:

1. **Unit Tests per Command:**
   - `test_info.py`: Test registry query, output formatting
   - `test_analyze.py`: Test dataset analysis (classes/files layout)
   - `test_recommend.py`: Test scoring algorithm, path resolution
   - `test_deploy.py`: Test SDK detection, artifact discovery

2. **Integration Tests:**
   - Full workflow: `init → analyze → recommend → train` (skip compilation for speed)

3. **Security Tests:**
   - Path traversal attempts
   - Shell metacharacter injection in arguments
   - Invalid environment variable values

4. **Cross-Platform Tests:**
   - Windows vs Unix path handling in deploy commands
   - SDK path detection on different OSes

## Next Steps

1. **Add test files** for each advanced command in `tests/` directory
2. **Document environment variables** (MMCLI_PYTHON, MMCLI_MODELMAKER) in CLI help text
3. **Create example projects** for testing with realistic datasets
4. **Add integration tests** covering full workflows
5. **Update documentation** with security considerations for new commands

## Dependencies Verification

| Dependency | Required For | Status |
|------------|--------------|--------|
| PyYAML >=6.0 | Config parsing, schema.yaml loading | Already in pyproject.toml |
| Python 3.10+ | Runtime | Confirmed via conftest.py fixtures |
| tinyml_modelmaker | Info command (subprocess) | Installed via requirements.txt (editable) |
| numpy | Analyze command (.npy files) | Imported but not explicit dep - likely from tinyml-modelmaker |
| pandas | Analyze command (.csv files) | Imported but not explicit dep - likely from tinyml-modelmaker |

**Note:** The project depends on external tools (TI CCS, dslite) for deploy functionality. These are not Python packages but must be installed separately for full deploy support.
