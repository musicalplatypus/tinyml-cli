---
phase: 1
reviewers: [ollama/qwen3.6]
reviewed_at: "2026-07-08T00:00:00Z"
plans_reviewed: [01-01-PLAN.md, 01-02-PLAN.md, 01-03-PLAN.md]
execution_status: complete  # Phase 1 completed pre-GSD; plans are retrospective
notes: >
  Retrospective review — Phase 1 predates formal GSD planning. Plans were written
  from git history (commits 69e31e9 through 768807e) for completeness.
  Gemini unavailable (account ineligible for free tier).
  Claude CLI skipped — running inside Claude Code (independence rule).
  One reviewer completed successfully.
  Context: Several Phase 1 security gaps were subsequently addressed in Phases 3
  (03-02: _is_safe_path temp paths) and 4 (04-01: hypothesis fuzz testing,
  04-04: ../ blocking fix).
---

# Cross-AI Plan Review — Phase 1

## Ollama Review (qwen3.6:latest)

### 01-01-PLAN.md: Core CLI Structure and Essential Commands

**Summary**
The plan establishes a clean, layered architecture for `mmcli`: argparse routing → YAML config
generation → isolated venv execution via `subprocess`. The separation of concerns is deliberate and
aligns well with the `MMCLI_PYTHON` isolation pattern. Auto-detection hints and PyInstaller/CI
scaffolding show distribution awareness, though the retrospective framing obscures whether all
components were consistently applied during implementation.

**Strengths**
- Clear data flow: CLI args → `builder.py` → YAML dict → venv runner eliminates direct command
  chaining in the main process
- Reusable `_add_common_args()` reduces duplication and ensures consistent flag behavior across
  subcommands
- Explicit Metal/MPS hints for macOS ARM64 improve out-of-the-box performance without bloating
  core logic
- Dedicated PyInstaller + GitHub Actions workflows mirror standard mature CLI distribution practices

**Concerns**

| Severity | Issue |
|----------|-------|
| **MEDIUM** | `builder.py` translating CLI args to YAML dicts creates a fragile translation layer; schema drift in `tinyml_modelmaker` will require manual updates here instead of native config parsing |
| **MEDIUM** | Bundling datasets with `init` increases binary size, complicates licensing compliance, and risks path/encoding issues across Windows/macOS/Linux |
| **LOW** | Auto-detection lacks explicit fallback semantics; silent CPU degradation on unsupported platforms or broken Metal/MPS symlinks could cause confusing runtime failures |

**Suggestions**
- Replace string/XML-based YAML translation with a validation-enabled schema (e.g., `pydantic` or
  `jsonschema`) to catch config errors at build time
- Externalize bundled datasets via installer artifacts or secure downloads; document licensing
  implications clearly
- Add explicit `--force-cpu` flag, platform probe logic, and fallback logging for hardware
  acceleration auto-detection

**Risk Assessment: MEDIUM**
Architectural separation is sound, but the translation layer and asset bundling introduce
maintainability and distribution risks.

---

### 01-02-PLAN.md: Security Hardening (Input Validation & Subprocess Isolation)

**Summary**
The plan addresses core injection vectors with explicit path guards, character sanitization, length
limits, and `shell=False` enforcement. The pattern is directionally correct for a Python CLI, but
the string-based validation logic contains well-documented bypass patterns and cross-platform
brittleness. Retrospective documentation often obscures whether these exact strings were used in
production or if they were iteratively patched.

**Strengths**
- `shell=False` + argument arrays correctly prevents subprocess interpreter abuse
- Length limiting (1024 chars) mitigates resource exhaustion and buffer manipulation vectors
- Dedicated `_is_safe_path()` and `_sanitize_input()` functions centralize security logic for
  auditability
- Explicit test files validate guard behavior against documented attack surfaces

**Concerns**

| Severity | Issue |
|----------|-------|
| **HIGH** | `'..' in path` fails on encoded/obfuscated traversal (`./../file`, `%2e%2e`, `..//`) and incorrectly rejects legitimate absolute paths needed on POSIX systems |
| **HIGH** | `re.sub(r'[^\w\-./_ ]', '', input_str)` strips platform-dependent punctuation and silently mangles Unicode filenames, breaking cross-platform compatibility |
| **HIGH** | Iterative `replace('..', '')` collapses nested traversal: `a....b` → `ab`, allowing path manipulation; also bypasses when combined with slashes (`../x/../../y`) |
| **MEDIUM** | `_is_safe_path()` and `_sanitize_input()` operate independently; order of operations is unspecified, leaving potential for pre-sanitization bypass |
| **MEDIUM** | `MMCLI_PYTHON` is "validated before use" but lacks path containment checks or signature verification, enabling arbitrary executable execution if env vars are tampered |
| **LOW** | Backtick/semicolon/`$` stripping targets bash but ignores PowerShell metacharacters (`&`, `%VAR%`, `\|`) and Windows NTFS reserved characters |

**Suggestions**
- Replace string traversal checks with semantic resolution: `PurePath(path).resolve().is_relative_to(target_dir)`
- Remove character whitelisting; validate paths via filesystem APIs (`os.path.realpath()` or
  Windows `GetFullPathNameW`)
- Process sanitization in a single pass: resolve → normalize → check containment → enforce length →
  escape/quote for subprocess argv
- Constrain `MMCLI_PYTHON` to expected venv directories; log warnings on mismatch
- Add PowerShell and Windows-specific quoting guards for cross-platform robustness

**Risk Assessment: HIGH**
Documented guards contain bypass-class vulnerabilities (traversal collapsing, Unicode stripping,
incomplete metachar blocking). Retrospective plans often normalize post-patch state; verify runtime
behavior matches documented strings before trusting as production baseline.

---

### 01-03-PLAN.md: Testing Infrastructure

**Summary**
The test plan provides a solid structural foundation: centralized fixtures, explicit cross-platform
coverage, CI matrix execution, and separation of release workflows. The inclusion of scaffolded tests
indicates forward planning. However, the reliance on mocking, incomplete error-path coverage, and
Windows CI bypass reduces confidence in real-world reliability claims.

**Strengths**
- `conftest.py` fixtures (`python_exe`, `tmp_project`) enforce isolation and reduce test coupling
- Explicit cross-platform test file addresses path separator and API drift risks early
- GitHub Actions matrix execution validates baseline compatibility across OSes before release
- Coverage configuration enforces measurability rather than relying on subjective completeness

**Concerns**

| Severity | Issue |
|----------|-------|
| **MEDIUM** | "Scaffolded" error recovery, regression, and performance tests indicate unimplemented failure modes; CLI tooling critically needs deterministic behavior for parse errors, venv failures, and subprocess timeouts |
| **MEDIUM** | Temp directory isolation (`tmp_project`) doesn't test concurrent executions, permission boundaries (POSIX ACLs, Windows UAC/Defender), or symlink resolution edge cases |
| **LOW** | `continue-on-error` on Windows CI masks platform-specific regressions until manual release validation |
| **LOW** | Mocking `psutil` and subprocess may pass tests while hiding real resource limits, signal handling, or zombie process accumulation in production |

**Suggestions**
- Replace scaffolds with deterministic assertions: YAML parse failures, missing venvs, non-zero
  subprocess exits, timeout handling, partial training states
- Use `pytest-subprocess` or strict `unittest.mock.patch('subprocess.run')` to verify exact
  argument arrays passed to runner
- Add permission/ownership test cases for POSIX and Windows (read-only dirs, restricted volumes,
  network drives)
- Remove `continue-on-error` from Windows CI; require explicit matrix pass gate before tag/release

**Risk Assessment: MEDIUM**
The test architecture is structurally sound but lacks depth in boundary conditions, error handling,
and real isolation testing. Windows CI bypass and scaffolded tests suggest undelivered coverage.

---

### Cross-Plan Observations

| Focus Area | Verdict | Notes |
|------------|---------|-------|
| **Security Design** | ⚠️ Directionally correct, implementation brittle | String-based guards bypassable via Unicode, encoded traversal, iterative-replace collapse |
| **Subprocess Isolation** | ✅ Architecturally sound | `subprocess.run(shell=False)` pattern is correct; MMCLI_PYTHON containment needs strengthening |
| **Cross-Platform Coverage** | ⚠️ Under-specified | Unicode path support, Windows metachar handling, and file permission semantics need OS-native resolution |
| **Test Coverage** | ⚠️ Scaffolded gaps | Error recovery, regression, and performance test files are placeholders |
| **Phase Goal Alignment** | ✅ Met | Core CLI, isolation pattern, security baseline, CI all delivered |

---

## Consensus Summary

Only one reviewer completed (Ollama/qwen3.6).

### Core Finding

**Phase 1 delivered the correct architectural foundation but the security implementation has
bypass-class vulnerabilities in the string-based path validation logic.**

The subprocess isolation pattern (`mmcli → MMCLI_PYTHON venv → tinyml-modelmaker`) is sound.
`shell=False` + argument arrays is correct. The `_is_safe_path()` and `_sanitize_input()`
functions, however, use string-based checks that have well-documented bypasses — the iterative
`replace('..', '')` approach is especially problematic, and the regex character whitelist silently
mangles Unicode paths.

**Historical note:** Several of these gaps were subsequently addressed:
- Phase 3 (03-02): Added `/private/var/folders` to _is_safe_path() temp allowlist
- Phase 4 (04-01): Hypothesis fuzz testing uncovered additional edge cases
- Phase 4 (04-04): Blocked single-dot traversal (`../`) explicitly; fix commit `4d3df52`

### Agreed Concerns (priority order)

1. **_is_safe_path() bypasses** — encoded traversal (`%2e%2e`, `../x/../../y`), iterative-replace
   collapse (`a....b` → `ab`), and Unicode path rejection all need semantic resolution instead
2. **_sanitize_input() Unicode breakage** — regex strips non-ASCII chars; breaks Chinese/Japanese
   project names and any non-POSIX-clean path
3. **MMCLI_PYTHON containment absent** — no check that the executable is within an expected venv
   directory; env var tampering → arbitrary executable execution
4. **Scaffolded tests never filled** — error_recovery, regression, performance test files remain
   as placeholders beyond Phase 1

### Agreed Strengths

1. Subprocess isolation pattern is architecturally correct and scales well
2. `shell=False` + argument arrays is enforced throughout
3. Centralized `_is_safe_path()` / `_sanitize_input()` functions — correct approach, implementation
   needs strengthening
4. Multi-platform CI/CD (GitHub Actions matrix) established early

### Recommended Actions

**Required (security):**
1. Replace iterative `replace('..', '')` with `pathlib.PurePath.resolve()` + containment check
2. Replace character-regex whitelist with OS-native path APIs; never strip Unicode silently
3. Scope `MMCLI_PYTHON` validation to known venv root paths (warn on deviation)

**Recommended (quality):**
4. Fill scaffolded test files with at minimum: non-zero subprocess exit, missing venv, timeout
5. Remove `continue-on-error` from Windows CI matrix
6. Add `--force-cpu` flag for Metal/MPS fallback transparency
