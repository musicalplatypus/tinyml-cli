---
phase: 6
reviewers: [ollama/qwen3.6]
reviewed_at: "2026-07-08T00:00:00Z"
plans_reviewed: [06-01-PLAN.md]
notes: >
  Gemini unavailable (account ineligible for free tier).
  Claude CLI skipped — running inside Claude Code (independence rule).
  One reviewer completed successfully.
---

# Cross-AI Plan Review — Phase 6

## Ollama Review (qwen3.6:latest)

### Summary
This plan is structurally organized for a dependency audit but fundamentally misaligned with
the documented root cause. It spends significant effort troubleshooting architecture mismatches
and environment variables across NumPy, Pandas, Torch, and OpenCV, despite explicit research
indicating the SIGSEGV originates from `onnxsim`'s C++ destructor during Python interpreter
shutdown. Because all pipeline artifacts are already written before the crash occurs, the plan
solves a phantom problem while ignoring the confirmed issue, making it an inefficient path to
Phase 6's goal.

### Strengths
- **Systematic audit methodology:** `faulthandler` + `otool -L` is a repeatable framework for
  diagnosing native extension linkage issues on macOS.
- **Clean environment discipline:** Fresh arm64 venv, forced pip/wheel updates, isolated
  dependencies reduce cross-contamination noise during debugging.
- **Pipeline verification focus:** Tying success to concrete artifacts (`mod.a`, `model.onnx`)
  and quantitative metrics (SNR/PSNR ≥ 0 dB) ensures business value is preserved regardless
  of crash semantics.
- **Multiprocessing safety:** Proposing `spawn` start method explicitly avoids fork-related
  macOS runtime conflicts.

### Concerns
- **HIGH: Solves the wrong diagnosed issue.** Explicitly targets NumPy/Pandas/OpenMP
  architecture mismatches despite known root cause being `onnxsim_cpp2py_export.so`'s C++
  destructor crash during interpreter teardown. Misdirects debugging effort and delays
  resolution.
- **HIGH: Misuse of `DYLD_LIBRARY_PATH`.** Setting and persisting this variable conflicts with
  modern macOS SIP restrictions, is deprecated for third-party library binding, and can cause
  silent fallback or security warnings. Linkage should be resolved via wheel rebuilds or
  `install_name_tool`, not env vars.
- **MEDIUM: `faulthandler` cannot catch shutdown destructor crashes.** It only handles signals
  during active execution. Post-pipeline Python interpreter teardown failure cannot be reliably
  verified with it — yields false negatives.
- **MEDIUM: Multiprocessing guard contradicts known facts.** The crash is unrelated to
  multiprocessing and fires *after* pipeline completion. Adding `spawn` logic adds complexity
  without addressing the actual exit behavior.
- **LOW: Over-engineering a benign shutdown artifact.** Since all outputs are already written,
  treating exit 245 as a hard failure introduces unnecessary rework and scope creep.

### Suggestions
- **Address `onnxsim` specifically:** Either accept the shutdown crash as benign (document it,
  strip exit code 245 from fail thresholds), or apply targeted fixes: downgrade/upgrade
  `onnxsim` to a patched version, or monkey-patch its destructor lifecycle.
- **Drop `DYLD_LIBRARY_PATH` workarounds.** Replace with native wheel verification
  (`pip install --force-reinstall --no-cache-dir --platform=arm64`) and strict `otool -L`
  checks on the specific `.so` file.
- **Shift verification to artifact integrity.** Validate success by post-pipeline checks
  (`test -f model.onnx && test -f mod.a`) rather than signal traps or exit codes.
- **Restrict audit scope.** Audit only relevant extensions: `onnxsim`, `tvm`, `pytorch`, and
  TI CGT Python bindings. Cross-auditing unrelated libraries (opencv-headless) adds friction.
- **If architecture mismatch exists, fix at build time.** Use `CMAKE_OSX_ARCHITECTURES=arm64`
  during custom extension builds, not runtime path overrides.

### Risk Assessment
**Overall Risk: HIGH**

The plan wastes development cycles on an unconfirmed dependency mismatch while ignoring the
verified `onnxsim` lifecycle bug. Persisting `DYLD_LIBRARY_PATH` introduces macOS SIP
compatibility risks and can silently break future CI/CD. Treating a post-artifact-write
shutdown crash as a hard failure forces unnatural workarounds that complicate exit semantics
without delivering tangible improvements.

---

## Consensus Summary

Only one reviewer completed (Ollama/qwen3.6). No cross-reviewer disagreement to synthesize,
but the single review is unambiguous.

### Core Finding

**The Phase 6 plan is solving the wrong problem.**

The SIGSEGV (exit 245) has been traced to `onnxsim_cpp2py_export.cpython-310-darwin.so`'s C++
destructor during Python **shutdown** — after all pipeline artifacts are written. The plan's 11
steps focus on NumPy/OpenMP/architecture mismatch remediation, which is unrelated to the
confirmed root cause.

### Agreed Concerns (HIGH priority)
1. Plan addresses NumPy/OpenMP mismatches; root cause is onnxsim C++ shutdown destructor
2. `DYLD_LIBRARY_PATH` persistence conflicts with macOS SIP; wrong remediation approach
3. `faulthandler` cannot intercept post-pipeline interpreter teardown crashes — verification
   strategy is unreliable for the actual failure mode

### Recommended Pivot

The plan should be rewritten around one of two valid approaches:

**Option A — Accept and document (low effort, already partially done):**
- The exit-245 crash is already handled in `test_e2e.py` (`assert rc in (0, _MACOS_SEGV)`)
- Document the behavior in SKILL.md and README (already done in this session)
- Close phase as "benign, documented, mitigated in test harness"

**Option B — Fix onnxsim specifically (targeted, higher effort):**
- `otool -L` audit scoped to `onnxsim_cpp2py_export.so` only
- Investigate onnxsim version downgrade/upgrade for a patched destructor
- Consider wrapping `mmcli run` invocation to intercept SIGABRT/SIGSEGV at process boundary
  and exit cleanly if all expected artifacts are present

**Option A is already functionally complete** given the work done in this session.
