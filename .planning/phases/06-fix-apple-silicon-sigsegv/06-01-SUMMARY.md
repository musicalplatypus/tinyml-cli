---
plan: 06-01
status: complete
completed_at: "2026-07-08"
---

# Summary: Plan 06-01 — Document and Mitigate onnxsim Shutdown Crash (macOS ARM64)

## What was done

All three tasks verified as already implemented at plan authoring time:

**Task 1 — Test harness (tests/test_e2e.py)**
- `_MACOS_SEGV = 245` defined on line 154
- Comment on line 152 explains: "245 = -11 mod 256 = SIGSEGV from onnxsim C++ cleanup during Python shutdown. All pipeline artifacts are written before the crash hits."
- `assert rc in (0, _MACOS_SEGV)` on line 155 — test never fails a successful pipeline for exit 245

**Task 2 — User docs (README.md)**
- Line 31: "On macOS ARM64, the process may exit with code 245 after the pipeline completes — this is a known crash in the onnxsim C extension during Python shutdown and does not affect output artifacts."

**Task 3 — Developer docs (SKILL.md)**
- Line 27: "on macOS, mmcli run / mmcli compile may exit with code 245 after the pipeline completes. This is a crash in onnxsim's C extension during Python shutdown."

## Acceptance criteria check

| Criterion | Result |
|-----------|--------|
| `grep "_MACOS_SEGV" tests/test_e2e.py` returns ≥2 lines | ✅ 2 matches |
| Comment on `_MACOS_SEGV` mentions onnxsim/shutdown | ✅ line 152 |
| `grep "245" README.md` exits 0 | ✅ line 31 |
| `grep -i "onnxsim" README.md` exits 0 | ✅ line 31 |
| `grep "245" SKILL.md` exits 0 | ✅ line 27 |
| `grep -i "onnxsim" SKILL.md` exits 0 | ✅ line 27 |
| `grep -i "shutdown" SKILL.md` exits 0 | ✅ line 27 |
| `pytest tests/test_e2e.py --co -q` exits 0 | ✅ |

## Notes

Plan pivoted from original NumPy/DYLD_LIBRARY_PATH remediation approach after cross-AI review (Ollama/qwen3.6) confirmed the crash is an onnxsim C++ destructor lifecycle issue, not an architecture mismatch. Option A (accept + document) was already functionally complete in the codebase — this plan verified and recorded that fact.
