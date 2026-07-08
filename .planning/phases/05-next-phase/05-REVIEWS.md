---
phase: 5
reviewers: [ollama/qwen3.6]
reviewed_at: "2026-07-08T00:00:00Z"
plans_reviewed: [05-01-PLAN.md, 05-02-PLAN.md, 05-03-PLAN.md, 05-04-PLAN.md, 05-05-PLAN.md, 05-06-PLAN.md]
notes: >
  Gemini unavailable (account ineligible for free tier).
  Claude CLI skipped — running inside Claude Code (independence rule).
  One reviewer completed successfully.
---

# Cross-AI Plan Review — Phase 5

## Ollama Review (qwen3.6:latest)

### Summary
Phase 5 plans demonstrate a pragmatic, execution-oriented approach to expanding `mmcli`'s feature
surface while preserving backward compatibility. The tasks are well-scoped, modularly separated into
dedicated modules (`progress.py`, `output.py`, `batch.py`, `interactive.py`), and include explicit
acceptance criteria. Critical design flaws from earlier drafts (e.g., hardcoded mock data in the
compare command) were successfully identified and resolved prior to delivery. However, several areas
lack defensive engineering: CLI convention inconsistency around directory flags, undefined dependency
boundaries for format/interactive extras, insufficient edge-case handling for file I/O and batch
execution, and an overly accelerated completion timeline for the interactive shell that suggests
rushed scaffolding over robust state management.

### Strengths
- **Pragmatic Deviation Management:** Successfully caught and fixed the mock-data trap in Plan 05-03
  before shipping, demonstrating strong self-review practices.
- **Backward Compatibility Preserved:** Default behavior remains unchanged for progress visualization
  (`--progress` opt-in) and export formats (`text` default), preventing regression breaks.
- **Clean Separation of Concerns:** Output formatting, progress tracking, and batch orchestration are
  isolated into dedicated modules, improving testability and future maintenance.
- **Environment-Aware Diagnostics:** `mmcli diagnose` checks concrete host-state requirements (Python
  ≥3.10, env vars, package importability, disk/dir access) rather than relying on brittle heuristics.
- **Clear Acceptance Criteria:** Each plan includes measurable success conditions, reducing ambiguity
  during implementation and QA.

### Concerns

| Severity | Concern | Impact Area |
|----------|---------|-------------|
| **HIGH** | `-d`/`-D` flag inconsistency across commands breaks CLI conventions. `recommend` is excluded from batch mode without documentation, creating fragmented UX expectations. | Consistency, Documentation, Scope |
| **HIGH** | Interactive shell completion time (5 min total) suggests scaffolding over resilience. Missing error recovery, state persistence, or graceful degradation strategies for REPL interruption. | Reliability, Security, UX |
| **MEDIUM** | Line-based subprocess streaming for progress (05-01) will be inaccurate or stall for non-line-buffered TI toolchains. No fallback if output buffering differs by command/device. | Performance, Correctness |
| **MEDIUM** | Export formats lack path validation, extension matching, or overwrite protection. Batch + `-o` could silently clobber files across multiple project runs. | Security, Data Loss, Edge Cases |
| **MEDIUM** | `compare` command shifted from model identifiers to `task_types`/`device`. Users may expect direct model-side-by-side comparison; registry abstraction introduces coupling to internal MMCLI naming conventions. | Usability, Scope Creep |
| **LOW** | Optional dependency management (`prompt_toolkit`, CSV/YAML serializers) not specified. May cause import errors or environment pollution if not scoped as extras/optional. | Dependency Ordering, Install Friction |
| **LOW** | `diagnose --full` disk/dir checks could block on large network mounts or trigger false negatives due to quota vs. actual available space logic gaps. | Performance, Edge Cases |

### Suggestions
- **Standardize Flag Conventions:** Replace `-D` with `--projects` or `--batch-dir` across all
  commands. Explicitly document why `recommend` is excluded from batch mode and consider adding it
  with a distinct flag if needed.
- **Define Dependency Boundaries:** List `tqdm`, `pyyaml`, `csv` (stdlib), and `prompt_toolkit` in
  `extras_require` of `pyproject.toml`. Ensure optional imports fail gracefully with clear install
  instructions.
- **Add I/O Safeguards:** Validate `-o` paths resolve to writable directories, enforce matching file
  extensions for `--format`, and add confirmation prompts or `--no-prompt` overrides in batch
  contexts to prevent silent clobbering.
- **Clarify Compare Abstraction:** Add a brief mapping section in help/docs explaining that
  comparison operates at the registry/task-type level rather than by raw model IDs.
- **Hardening Progress Tracking:** Document that line-based streaming is a subprocess-wrapping
  limitation. Add a note recommending users pipe toolchain output directly when precise epoch
  tracking is critical.
- **Interactive Shell Resilience:** Explicitly define lifecycle behavior (Ctrl-C handling, syntax
  error recovery, state reset between commands) and mark as `experimental` in CLI help until v1.0
  stability is proven.

### Risk Assessment
**Overall Risk: MEDIUM**

The plans successfully align with Phase 5's UX goal and avoid major architectural drift. Core risks
do not stem from fundamental design flaws but from insufficient defensive engineering around I/O,
flag consistency, and dependency scoping. The `-d`/`-D` inconsistency and undocumented batch
exclusion create friction that could impact adopter satisfaction. The interactive shell's rushed
timeline introduces latent reliability gaps, while format export lacks path validation and extension
enforcement. However, the modular structure, explicit acceptance criteria, and proactive mock-data
remediation significantly mitigate execution risk.

---

## Consensus Summary

Only one reviewer completed (Ollama/qwen3.6). No cross-reviewer disagreement to synthesize,
but the single review is substantive.

### Core Finding

**Phase 5 plans are solid in structure but have defensive engineering gaps.**

The plans correctly scope new features, maintain backward compatibility, and separate concerns
cleanly into dedicated modules. The critical mock-data bug in `compare.py` was caught and fixed
before shipping — a positive signal for plan quality.

### Agreed Concerns (priority order)

1. **Flag inconsistency** — `-d`/`-D` conflict in batch mode; `recommend` silently excluded from
   batch without explanation in docs or help text
2. **Interactive shell resilience** — 5-minute completion suggests scaffolding; missing Ctrl-C
   handling, state reset, and `experimental` label
3. **I/O edge cases** — `-o` flag in batch context can silently clobber files across runs; no path
   or extension validation
4. **Subprocess progress accuracy** — line-based streaming is a best-effort proxy for TI toolchain
   progress; may stall or miscount on non-line-buffered commands

### Agreed Strengths

1. Mock-data catch in compare.py before shipping
2. Backward-compatible defaults on all new flags
3. Module-per-feature separation (progress.py, output.py, batch.py, interactive.py)
4. Real environment checks in diagnose (not heuristics)

### Recommended Pivot (if replanning)

The concerns are polish-level, not architectural. If `/gsd:plan-phase 5 --reviews` is run:

- **Option A** (low effort): Document flag inconsistency in help text; add `experimental` tag to
  `mmcli shell` help; add path-writable check to `-o` flag handler
- **Option B** (refactor): Standardize `--batch-dir` flag naming; add I/O safeguards module; harden
  `interactive.py` with Ctrl-C + syntax error recovery before marking stable
