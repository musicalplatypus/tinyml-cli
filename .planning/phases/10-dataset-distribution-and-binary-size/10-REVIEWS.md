---
phase: 10
reviewers: [gemma4-31b-cloud, deepseek-r1-14b, coderabbit]
reviewers_unavailable: [gemini, codex, opencode, qwen, cursor, lm_studio, llama_cpp]
reviewed_at: 2026-08-02
plans_reviewed: [10-09-PLAN.md, 10-07-PLAN.md]
---

# Cross-AI Plan Review — Phase 10

Scoped to the two **unfinished** plans. Phase 10 is 8/11 executed; feedback on executed plans
would be historical, so the prompt directed reviewers to 10-09 (mid-flight, Tasks 1-3 committed
across two repos, Task 4 checkpoint pending) and 10-07 (not started).

## Reviewer availability

| Reviewer | Outcome |
|---|---|
| **gemma4:31b-cloud** (via ollama) | ✅ substantive review |
| **deepseek-r1:14b** (local, 9 GB, `num_ctx` 32768) | ⚠️ ran, but low value — see below |
| **coderabbit** | ✅ code review of landed 10-09 code (not a plan review) |
| gemini | installed but **dead** — `IneligibleTierError`, Google discontinued this client tier for individuals |
| claude | skipped deliberately — `CLAUDE_CODE_ENTRYPOINT=claude-desktop`, it is the reviewing agent itself |
| kimi-k3, minimax-m2.5/m2.7 (cloud) | require an ollama.com subscription |
| codex, opencode, qwen, cursor, lm_studio, llama_cpp | not installed / not running |

**Correction to an earlier run of this command:** ollama was initially reported as missing. That
was wrong — the detection used `--max-time 2`, which the server did not answer inside. Ollama is
running on `*:11434` with 24 models. Local models were capped to fit well inside 40 GB RAM at
the user's instruction, and context was bounded explicitly because `OLLAMA_CONTEXT_LENGTH` is
set to 131072, which would otherwise allocate a KV cache far larger than the weights.

---

## gemma4:31b-cloud

### Summary
Phase 10 is nearly complete. The plans are rigorous about the safety of the `remove` operation —
ensuring the GUI cannot wipe packaged data — and about documenting the release process to
prevent silent dataset rot. Separation of concerns between CLI (authority) and GUI (orchestrator)
is well maintained.

### Strengths
- **Path guarding.** Implementing `mmcli datasets remove` rather than having the GUI unlink files
  is the correct architectural choice. Centring the "is this in the cache?" logic in Python under
  pytest effectively neutralises the risk of the GUI deleting a user's air-gapped
  `MMCLI_DATASETS` directory.
- **Failure classification.** Requiring the GUI to preserve mmcli's verbatim error while adding a
  remedy sentence prevents the "confident wrong diagnosis" problem common in wrapper apps.
- **Release continuity.** 10-07 treats the release process as a first-class citizen, addressing
  both the unstable upstream CDN and the danger of stale version pins.
- **CI realism.** D-01 and D-02 correctly acknowledge that hosted runners are too variable for
  tight startup-time gates, avoiding a flaky-test cycle.

### Concerns
- **LOW — dependency order.** 10-07 documents the `remove` command that 10-09 is still adding.
  An agent starting 10-07 before 10-09's Task 4 clears might document a surface that has not been
  human-verified.
- **LOW — cache-shadowing visibility.** A dataset can be `bundled` while still holding a stale
  cache entry. The approved affordance table hides Remove for bundled datasets, so that space is
  unreclaimable from the GUI — a discrepancy between "disk used" and "disk reclaimable".

### Suggestions
- 10-09 Task 4 step 9 should use a dataset that **exists in the registry but is not cached**, to
  prove `remove` specifically *refuses to touch* a primary directory rather than merely failing
  safe.
- 10-07's help-text update should use raw strings/explicit escaping when matching README wording,
  since argparse and Markdown render backticks and paths differently.

### Risk: **LOW**
The high-risk operation (file deletion) is guarded by a dedicated subcommand with its own unit
tests and a strict path assertion. Cross-repo risk is managed by a clear task split and dual-SHA
recording. **Verdict: proceed to 10-09 Task 4, then execute 10-07.**

---

## deepseek-r1:14b (local)

Ran in ~3m24s. **Low value, recorded for completeness rather than weight.** It restated the plans
back rather than analysing them, produced no specific or actionable defect, and its risk
assessment is circular — it rates 10-07 **HIGH** risk because "release processes lack
documentation", when writing that documentation *is* 10-07. Its output also leaked non-English
tokens mid-sentence.

Its only substantive points duplicate gemma4's, less precisely: deletions must affect cache
directories only, and dependency ordering needs managing.

This matches a finding already recorded in this project: a local `qwen3-coder:30b` previously
passed 9/9 plan checks and caught none of five real defects. **A local mid-size model is not a
review gate.** It should not be counted toward consensus.

---

## coderabbit — code review of 10-09's landed code

Not a plan review; CodeRabbit reviews a git diff. Pointed at 10-09 Task 1's commits, which had
never been peer-reviewed and which **delete files from disk**. Three findings, **all verified
against source before acting, all real**, fixed in `0bc9703`:

1. **MAJOR — cache inspection created directories.** `cache_entry_path()` resolved through
   `_cache_dir()`, which calls `os.makedirs`. A read-only `datasets list` therefore wrote to the
   filesystem, would fail on an unwritable cache home with no download requested, and
   `datasets remove`'s safety assertion created the very directory it was declining to touch.
   Split into a pure `_cache_dir_path()`; verified end to end that a listing now leaves nothing
   behind.
2. **MAJOR — ENOSPC escaped the buffered flush.** `out.write(chunk)` was guarded but the flush at
   block exit was not, so a disk filling on the last buffer produced a bare traceback — exactly
   what that handler existed to prevent.
3. **MINOR — `getsize` outside the try** around `unlink`, so a file vanishing between the check
   and the read tracebacked instead of printing the error message below it.

Four regression tests added, including one asserting `_cache_dir` *still* creates for downloads.

---

## Consensus summary

### Agreed strengths
- Routing removal through `mmcli datasets remove` rather than GUI unlinking is the right call
  (both plan reviewers).

### Agreed concerns
- **Dependency ordering between 10-07 and 10-09** — raised by both, at LOW severity by the
  credible reviewer. Already mitigated: 10-07 is sequenced after 10-09 and `.continue-here.md`
  records the in-flight state.
- **File-deletion safety** — flagged by both as the area to watch, and neither found an actual
  defect in it.

### Divergent views
Risk rating: gemma4 says **LOW** and "proceed"; deepseek says **MEDIUM/HIGH**. The divergence is
not substantive — deepseek's higher rating rests on the circular claim above, not on a defect.

### The finding that matters most
**The plan reviewers found no concrete defects; the code reviewer found three real bugs.** Both
plan reviewers praised the file-deletion design in the abstract, while the actual implementation
of that design was silently creating directories during read-only inspection and dropping an
ENOSPC on flush. Plan review and code review are not substitutes, and on this evidence plan
review is the weaker of the two for this codebase.

## Still undone
- **10-09's Swift half is unreviewed.** CodeRabbit ran only against `tinyml-cli`; the
  PlatypusStudio commits (`6d66d7f`, `560aa63`, `e5142bf`), including the new
  `DatasetLibraryView`, were not covered.
- gemma4's Task-4 suggestion (step 9 should use a registry dataset that is *not* cached) is a
  genuine sharpening of the hard gate and should be folded into the checkpoint run.
