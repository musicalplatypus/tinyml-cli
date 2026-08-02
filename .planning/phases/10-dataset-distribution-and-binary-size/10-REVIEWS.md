---
phase: 10
reviewers: [coderabbit]
reviewers_attempted: [gemini, coderabbit]
reviewers_unavailable: [claude, codex, opencode, qwen, cursor, ollama, lm_studio, llama_cpp]
reviewed_at: 2026-08-02
review_type: code (not plan — see caveat)
target: 10-09 Task 1 commits (47c76f3, 420adde) in tinyml-cli
---

# Cross-AI Review — Phase 10

## Caveat: this is not the review that was requested

`/gsd-review` performs **cross-AI plan review**. That could not run. Of the eight possible
reviewers:

| Reviewer | Outcome |
|---|---|
| gemini | installed but **dead** — `IneligibleTierError: This client is no longer supported for Gemini Code Assist for individuals` (Google requires migration to Antigravity) |
| claude | available, but **deliberately skipped** — `CLAUDE_CODE_ENTRYPOINT=claude-desktop`, so it is the reviewing agent itself and offers no independence |
| codex, opencode, qwen, cursor | not installed |
| ollama, lm_studio, llama_cpp | no local server running |
| coderabbit | available and authenticated |

CodeRabbit reviews a **git diff**, not a prompt — it cannot assess a plan. So no independent
model assessed the phase-10 plans. What was done instead, deliberately and with the substitution
stated up front: a code review of the work 10-09 Task 1 had just landed, which had never been
peer-reviewed and which **deletes files from disk**. That is arguably the higher-value target
right now, but it does not discharge plan review.

The workflow's `--prompt-only` flag also does not exist in this CodeRabbit version; `--committed
--base-commit <sha> --agent` was used.

## CodeRabbit review — 10-09 Task 1 (`mmcli datasets remove`, `cache_bytes`, ENOSPC)

Three findings. **All three were verified against the source before acting** — none were taken
on trust — and all three were real. Fixed in `0bc9703`.

### MAJOR — cache inspection created directories (`mmcli/datasets.py`)

`cache_entry_path()` resolved the directory through `_cache_dir()`, which calls
`os.makedirs(..., exist_ok=True)`. Asking *where a cache entry would be* therefore created it.

Consequences, in ascending order of seriousness:
- `datasets list --format json` calls `cache_entry_size()` for all ten datasets, so a purely
  read-only listing wrote to the filesystem
- it would raise `OSError` on a read-only or unwritable cache home even when no download had
  been requested
- `datasets remove`'s safety assertion computed `expected_dir` the same way, so the refusal
  path created the very cache directory it was declining to touch

Fixed by splitting a pure `_cache_dir_path()` from the creating `_cache_dir()`. Inspection and
the remove guard use the pure one; the download flow — the only caller that legitimately needs
the directory — keeps the creating one.

**Verified end to end:** `XDG_CACHE_HOME=<fresh dir> mmcli datasets list --format json` now
leaves nothing behind. Previously it created `<dir>/mmcli/datasets/01_03_00/`.

### MAJOR — ENOSPC escaped on the buffered flush (`mmcli/datasets.py`)

`out.write(chunk)` was wrapped and translated ENOSPC into a legible `RuntimeError`, but writes
are buffered: the final flush happens when the `with` block exits, outside that handler. A disk
filling on the last partial buffer produced a bare traceback — **precisely the failure the inner
handler was added to prevent**. The block now carries the same translation, so the user sees one
message whichever byte the disk ran out on.

Worth noting this is the second time this specific path has been found incomplete; 10-10's
planning found the first half of it.

### MINOR — `getsize` outside the try (`mmcli/cli.py`)

`freed_bytes = os.path.getsize(target)` sat above the `try:` guarding `os.unlink`. If the file
disappears between the `isfile` check and the read — another process, or a concurrent `remove` —
the `OSError` surfaced as a traceback instead of the error message written directly below it.
Both calls now share the handler.

## What was added beyond the fixes

Four regression tests (`TestCacheInspectionHasNoSideEffects`), covering the path query, the size
query, the user-visible case of listing every dataset, and that `_cache_dir` *still* creates for
the download flow — the last one guards against over-correcting the fix.

Counts after: `test_datasets_cli.py` 39 passed; `test_datasets_download.py` +
`test_build_config.py` + `test_ci_workflows.py` 92 passed.

## Consensus summary

Not applicable — one reviewer. There is no cross-model agreement to report, and no divergence.
Recording that plainly rather than presenting a single reviewer's output as consensus.

## What this leaves undone

- **Phase-10 plans remain un-peer-reviewed by an independent model.** The two unfinished plans
  (10-09, 10-07) were never assessed by anything other than this project's own planner/checker.
- **10-09's Swift half is unreviewed.** CodeRabbit ran only against `tinyml-cli`; the
  `PlatypusStudio` commits (`6d66d7f`, `560aa63`, `e5142bf`) — including the new
  `DatasetLibraryView` — were not covered by this run.
- To restore plan review, install a working second CLI (`codex` is the least friction) or run
  the local-server path with ollama/LM Studio. Note the project's own prior finding that a local
  qwen3-coder:30b passed 9/9 and caught none of five real plan defects, so a local model is not
  a substitute for an independent frontier one.
