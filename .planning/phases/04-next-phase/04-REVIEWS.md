---
phase: 4
reviewers: [ollama/qwen3.6]
reviewed_at: "2026-07-08T00:00:00Z"
plans_reviewed: [04-01-PLAN.md, 04-02-PLAN.md, 04-03-PLAN.md, 04-04-PLAN.md, 04-05-PLAN.md]
execution_status: partial  # 04-01 complete; 04-02 through 04-05 not yet executed
notes: >
  Gemini unavailable (account ineligible for free tier).
  Claude CLI skipped — running inside Claude Code (independence rule).
  One reviewer completed successfully.
---

# Cross-AI Plan Review — Phase 4

## Ollama Review (qwen3.6:latest)

### Summary
Phase 4 demonstrates a strong security engineering mindset, particularly in Plan 04-01 where
hypothesis-based fuzzing successfully hardened path sanitization and resolved edge-case bugs.
However, plans 04-02 through 04-05 contain significant alignment gaps with the project's
established `argparse` architecture and standard CLI usability patterns. Several tasks introduce
breaking changes, contradictory validation logic, and overly restrictive execution constraints
that will likely cause test failures or developer friction if executed as written. With targeted
architectural corrections and requirement clarifications, these plans will deliver robust security
enhancements without compromising compatibility or workflow flexibility.

### Strengths
- **Effective Fuzzing Foundation (04-01):** Successfully integrated `hypothesis` to uncover real
  edge cases in `_is_safe_path()` (temp directories, Windows separators, `..../` patterns),
  setting a high bar for data-driven testing.
- **Structured Threat Modeling:** Adoption of STRIDE in 04-02 provides a clear, industry-standard
  framework for categorizing threats and deriving targeted mitigations.
- **Comprehensive Documentation Strategy:** Cross-referencing `SECURITY.md`, environment variables,
  and function docstrings ensures security context is accessible across user-facing and developer docs.
- **Clear Task Breakdown:** Each plan maintains explicit objectives, deliverables, and acceptance
  criteria, making execution tracking straightforward.

### Concerns

#### HIGH Severity (BLOCKING — must fix before executing 04-02 through 04-05)

| # | Plan | Issue |
|---|------|-------|
| 1 | 04-02 | `from click.testing import CliRunner` — `click` is not in the dependency tree. mmcli uses `argparse`. Will `ModuleNotFoundError` at import time. |
| 2 | 04-04 | Contradictory test: Task 3 `_is_safe_path()` blocks `../other-project` but Task 4 test asserts it returns True. Guarantees `pytest` failure. |
| 3 | 04-04 | `_sanitize_input()` changed from silent truncation to raising `ValueError` — breaking change for all existing callers. |
| 4 | 04-04 | CWD-anchoring: `abs_path.startswith(base_dir + os.sep)` makes all project paths relative to the CLI's execution directory. Breaks multi-project workflows and symlinks. |

#### MEDIUM Severity

| # | Plan | Issue |
|---|------|-------|
| 5 | 04-02 | Relies on `run_mmcli`, `SUBPROCESS_TIMEOUT` constants that may not exist under those exact names in codebase. |
| 6 | 04-02 | Weak assertions: `"command not found" in output.lower()` — no exit code check, no stderr filter. |
| 7 | 04-05 | `pip-audit --strict` will fail on transitive/build deps outside developer control — leads to CI fatigue. |

#### LOW Severity

| # | Plan | Issue |
|---|------|-------|
| 8 | 04-03 | Placeholder email `security@mmcli.example.com` — must be replaced before shipping. |
| 9 | 04-05 | `test_scan_script_runs_without_error` calls `pytest.skip()` immediately — permanently dead test code. |
| 10 | 04-05 | `scripts/scan-vulnerabilities.sh` creates `/tmp/vuln-scan-env` without cleanup on exit. |

### Suggestions for Unexecuted Plans

**Plan 04-02:**
- Replace `click.testing.CliRunner` with subprocess invocation via `sys.argv` injection or
  `pytest-subprocess` mock, since mmcli is argparse-based
- Validate security boundaries by exit codes and stderr: `assert result.returncode in (1, 2)`
- Remove `inspect.getsource()` checks — mock `subprocess.run` and assert kwargs instead
  (`shell=False`, `timeout=N`)

**Plan 04-03:**
- Replace placeholder email with actual maintainer alias or GitHub Security Advisory URL
- Add "Known Limitations" subsection clarifying threat model covers CLI surface only

**Plan 04-04:**
- Resolve the `../other-project` contradiction before executing: either block it (update test) or
  allow it (introduce explicit safe-base config rather than CWD anchoring)
- Keep `_sanitize_input()` non-raising for backward compatibility; move ValueError to argparse
  `type=` validators only (CLI boundary, not internal API)
- Use `os.path.realpath()` for CWD-anchor comparison; introduce a `--safe-dir` or config option
  rather than hardcoding CWD

**Plan 04-05:**
- Add `trap "rm -rf /tmp/vuln-scan-env" EXIT` to scan script
- Remove permanent `pytest.skip()` in the test or replace with a real assertion
- Use `pip-audit` without `--strict` for CI or add a CVE allowlist for non-exploitable transitive deps
- Pin Python version explicitly in security.yml: `python-version: '3.10'`

### Risk Assessment
**Overall Risk: MEDIUM-HIGH** (as written) → **LOW** (with corrections)

Plans 04-02 through 04-05 will fail immediately if executed without corrections: the click import
error alone blocks test collection, the contradictory path test will fail, and the ValueError
change will break callers. Once the argparse/click mismatch is fixed, the contradiction resolved,
and the breaking change scoped to the CLI boundary only, the remaining work is well-scoped and
achievable. Execution should pause until those three fixes are made.

---

## Consensus Summary

Only one reviewer completed (Ollama/qwen3.6).

### Core Finding

**Plans 04-02 through 04-05 contain blocking errors that must be fixed before execution.**

Plan 04-01 is solid and complete. The remaining four plans have implementation assumptions that
conflict with the actual codebase, producing tests that will fail at import time or contradict
their own implementations.

### Agreed Concerns (priority order)

1. **click import in 04-02** — `ModuleNotFoundError` at test collection, blocks entire test run
2. **`../other-project` contradiction in 04-04** — test asserts pass, impl blocks it; one must change
3. **ValueError breaking change in 04-04** — silent truncation → raise breaks existing callers
4. **CWD anchoring in 04-04** — ties all project paths to execution directory, too restrictive for CLI use

### Agreed Strengths

1. 04-01 fuzz testing is complete and excellent — real bugs found and fixed
2. STRIDE threat model in 04-02 is the right framework
3. Documentation scope (SECURITY.md + env doc + docstrings) is comprehensive

### Recommended Pivot

Before executing 04-02 through 04-05, apply these pre-flight fixes:

**Required (blocking):**
1. Replace `click.testing.CliRunner` in 04-02 with argparse-compatible test pattern
2. Resolve `../other-project` ambiguity in 04-04 (block OR configure safe-base)
3. Scope `_sanitize_input()` ValueError to argparse validators only in 04-04

**Recommended (quality):**
4. Add trap/cleanup to 04-05 scan script
5. Replace placeholder email in 04-03 SECURITY.md
6. Remove dead `pytest.skip()` from 04-05 test
