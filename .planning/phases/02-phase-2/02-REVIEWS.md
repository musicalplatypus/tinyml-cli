---
phase: 2
reviewers: [ollama/qwen3.6]
reviewed_at: "2026-07-08T00:00:00Z"
plans_reviewed: [02-01-PLAN.md, 02-02-PLAN.md, 02-03-PLAN.md, 02-04-PLAN.md, 02-05-PLAN.md, 02-06-PLAN.md]
execution_status: complete  # all 6 plans executed and summarized
notes: >
  Gemini unavailable (account ineligible for free tier).
  Claude CLI skipped — running inside Claude Code (independence rule).
  One reviewer completed successfully.
---

# Cross-AI Plan Review — Phase 2

## Ollama Review (qwen3.6:latest)

### Summary
Phase 2 delivers comprehensive unit test coverage for four advanced mmcli commands plus two
documentation updates. All six plans completed and align directly with their REQ-TESTS requirements.
However, the phase reveals a structural disconnect in three of four TDD plans: test files
pre-existed, meaning the "TDD" methodology was actually post-hoc test documentation rather than
design-driven test authoring. Beyond methodology labeling, cross-plan security gaps exist — none
of the test suites explicitly validate `_is_safe_path()` or `_sanitize_input()` integration for
user-supplied paths, and the deploy command plan lacks platform-aware subprocess mocking and temp
fixture cleanup guarantees.

### Strengths
- **Correct isolation architecture:** All four command test plans mock `subprocess.run`, correctly
  isolating the external tinyml_modelmaker venv dependency. Aligns with Phase 1's security pattern.
- **Failure mode coverage:** Plans explicitly test invalid JSON, non-zero exit codes, missing
  modelzoo, and empty results — the failure paths that commonly cause silent CLI crashes.
- **02-02 genuine TDD:** The analyze command tests were created from scratch (40 tests, 13 classes)
  with real boundary analysis (499/500, 4999/5000 thresholds) and temp fixture isolation.
- **Security-aware documentation:** 02-06 explicitly warns against committing sensitive paths to
  config files and references environment variables as the correct secrets mechanism.
- **Disciplined scope:** All plans map directly to Phase 2 requirements. No scope creep detected.

### Concerns

| Severity | Plan | Issue & Implication |
|:--------:|:----:|:-------------------|
| **HIGH** | 02-04 | **Platform-specific subprocess behavior unaddressed.** Windows `dslite`, macOS CCS launcher, and Linux cross-toolchain paths differ drastically. The mocking strategy does not stub per `sys.platform`, meaning tests that pass on macOS may fail on Windows CI runners without explicit platform routing. |
| **MEDIUM** | 02-01 | **Cross-file coupling: `_bin_dataset` referenced in info tests.** The plan lists `_bin_dataset` as a test target but notes it belongs to `analyze.py`. Importing or mocking across module boundaries introduces maintenance debt and potential flaky import errors on refactoring. |
| **MEDIUM** | 02-01, 02-03, 02-04 | **No explicit `_is_safe_path()` / `_sanitize_input()` integration tests.** All three command test plans omit validation that user-supplied registry queries, modelzoo paths, and deploy targets pass through Phase 1's security guards. Contradicts the "security testing for all new features" requirement (REQ-TESTS-07). |
| **MEDIUM** | 02-04 | **Temp fixture lifecycle unspecified.** Simulated SDK/artifact temp directories have no documented yield/cleanup guarantee. Leaked fixtures break subsequent CI jobs and cause false positives in path-existence checks. |
| **MEDIUM** | 02-05 | **No test guards help-text drift.** CLI refactors can silently drop MMCLI_* environment variable documentation without detection. A single `--help | grep MMCLI_` integration test would prevent regression. |
| **LOW** | 02-01, 02-03, 02-04 | **TDD methodology mislabeled.** Three plans claim `type: tdd` but execution notes confirm files pre-existed. Mislabeling obscures whether design intent drove test authoring or vice versa, reducing architectural confidence in these test suites. |
| **LOW** | 02-02 | **Binary fixture size risk.** Actual `.npy`/`.pkl` temp fixtures can bloat CI artifacts or slow test runs if datasets exceed a few MB. In-memory serialization is safer for unit scope. |

### Suggestions
- **Add platform-aware mock routing in 02-04:** Use `@patch('subprocess.run', side_effect=platform_mock_factory)` or `pytest.mark.skipif(sys.platform != 'darwin')` guards for CCS/dslite invocations.
- **Add security integration tests:** Each command test plan should include at least one test class validating that unsanitized paths (e.g., `../../../etc/passwd`) trigger `_is_safe_path()` rejection before reaching subprocess or filesystem calls.
- **Fix `_bin_dataset` cross-file reference in 02-01:** Remove from info tests; confirm the function is only tested in `test_analyze.py` where it belongs.
- **Specify temp fixture cleanup in 02-04:** Use `yield`-based fixtures or `tmp_path` (pytest built-in) to guarantee teardown on test failure.
- **Add help-text regression test:** `assert 'MMCLI_PYTHON' in subprocess.check_output(['python', '-m', 'mmcli', '--help'], text=True)` as a lightweight integration check in 02-05.
- **Correct metadata labels:** Update `type: tdd` to `type: post-functest` or `type: retrofit` in 02-01, 02-03, 02-04 to accurately reflect methodology.
- **Cap fixture sizes in 02-02:** Replace large binary mocks with minimal in-memory byte streams; document size limits in CI configuration.

### Risk Assessment
**Overall Risk: LOW-MEDIUM**

Plans are structurally sound and correctly scoped. Core risks stem from:
1. Platform-specific subprocess mocking absent in deploy tests (highest risk — CI failures on Windows)
2. Security guard coverage missing across all command test suites (contradicts REQ-TESTS-07 intent)
3. Temp fixture lifecycle unspecified in 02-04

Once these three issues are addressed, remaining concerns are polish-level. The documentation plans
(02-05, 02-06) are well-executed with LOW individual risk.

---

## Consensus Summary

Only one reviewer completed (Ollama/qwen3.6).

### Core Finding

**Phase 2 delivered correct test coverage but three of four TDD plans were post-hoc documentation
of pre-existing tests, and all command test suites omit security guard integration testing.**

All six plans completed against their requirements. The `analyze` command test (02-02) was the only
one genuinely authored test-first. The other three command test files pre-existed, making the plans
retrospective descriptions rather than design-driving specs. More critically, none of the test
suites explicitly verify that `_is_safe_path()` and `_sanitize_input()` are called for user inputs
— the central security guarantee from Phase 1.

### Agreed Concerns (priority order)

1. **Platform-specific mocking absent in 02-04** — Windows/macOS/Linux subprocess behavior differs
   for CCS/dslite; no per-platform mock routing specified; CI failures expected on non-macOS runners
2. **Security guard integration missing across 02-01/02-03/02-04** — REQ-TESTS-07 requires security
   testing for all new features, but no tests verify `_is_safe_path()` or `_sanitize_input()` are
   called for user-supplied paths in info, recommend, or deploy workflows
3. **`_bin_dataset` cross-file coupling in 02-01** — function is defined in `analyze.py`, not
   `info.py`; testing it from `test_info.py` creates an import coupling that breaks on refactoring
4. **Temp fixture cleanup unspecified in 02-04** — no yield/cleanup guarantee for simulated SDK
   directories; leaked fixtures can corrupt subsequent CI jobs

### Agreed Strengths

1. Subprocess mocking correctly isolates the external tinyml_modelmaker venv dependency
2. 02-02 analyze tests were genuinely TDD: 40 tests, 13 classes, created from scratch with correct
   boundary analysis
3. 02-06 config documentation explicitly warns about secrets in config files — correct security posture
4. All plans scope-disciplined — maps directly to REQ-TESTS requirements, no scope creep

### Recommended Actions (if replanning or executing future phases)

**Required (blocking for cross-platform CI):**
1. Add `sys.platform`-aware mock routing in 02-04 deploy tests for CCS/dslite path stubs
2. Add one security integration test per command verifying `_is_safe_path()` blocks `../`-style paths

**Recommended (quality):**
3. Remove `_bin_dataset` from `test_info.py`; test it only in `test_analyze.py`
4. Use `tmp_path` pytest fixture in 02-04 to guarantee SDK/artifact temp dir teardown
5. Add `--help | grep MMCLI_` assertion to prevent environment variable doc drift
6. Correct `type: tdd` metadata to `type: retrofit` in 02-01, 02-03, 02-04
