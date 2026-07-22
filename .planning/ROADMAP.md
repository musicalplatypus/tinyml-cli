# Roadmap

## v1.2 Milestone — Ecosystem Gap Closure (next)

**Phases:** 7-9  
**Source:** Gap analysis against tinyml-tensorlab (2026-07-09)

### Phase 7: Device & Task Coverage
**Goal:** Close two HIGH-priority discoverability gaps: missing `F28E12` device and invisible `audio_classification` task type.

**Depends on:** Phase 6 (complete)

**Plans:** 2/3 plans executed

- [x] 07-01-PLAN.md — Add F28E12 to TARGET_DEVICES + TASK_TYPES_AUDIO constant + help text
- [x] 07-02-PLAN.md — Tests for device coverage and audio task discoverability

### Phase 8: Dataset Preset Selection
**Goal:** Expose `dataset_name` preset selection via `--dataset-preset` flag on train/run; extend `mmcli info` to list dataset presets.

**Depends on:** Phase 7

**Plans:** 2/3 plans executed

- [x] 08-01-PLAN.md — Add --dataset-preset flag + builder.py wiring
- [x] 08-02-PLAN.md — Extend mmcli info with dataset preset listing + tests

### Phase 9: Advanced Training Knobs
**Goal:** Expose `nn_for_feature_extraction`, `gof_test`, and QAT/PTQ mode selection as CLI flags.

**Depends on:** Phase 8

**Plans:** 2 plans

- [ ] 09-01-PLAN.md — Add --nn-feature-extraction + --gof-test flags + builder wiring
- [ ] 09-02-PLAN.md — Add --quantization-mode flag + tests for all three knobs

### Phase 10: Dataset Distribution and Binary Size

**Goal:** Cut the distributed `mmcli` binary from 260 MB to ~14 MB by fetching the TI example
datasets from their upstream versioned URLs on demand, so a dataset can be release-specific
and updated without rebuilding the binary.

**Requirements**:
- REQ-SIZE-01: `dist/mmcli` ≤ 15 MB and starts in < 2.5 s (3-run median)
- REQ-SIZE-02: PyInstaller must not bundle the training engine in any of the three published
  artifacts (Linux, Windows, macOS); a build that loses the exclusions fails CI — meaning the
  guard runs inside `.github/workflows/` — rather than shipping a 260 MB binary
- REQ-DATA-01: Datasets resolvable via `MMCLI_DATASETS` → bundled → cache → download
- REQ-DATA-02: Any registry entry carrying a `ti_name` (i.e. any dataset that can be
  fetched) must carry a `sha256`, enforced at import and verified before extraction; a
  corrupt, truncated or substituted download fails loudly and leaves no cache entry
- REQ-DATA-03: `MMCLI_DATASETS` disables all fetching (offline / air-gap escape hatch)
- REQ-DATA-04: All 10 datasets remain obtainable offline via `MMCLI_DATASETS`
- REQ-DATA-05: TI datasets are fetched from software-dl.ti.com at a pinned engine-version
  path, with a per-dataset version override; the cache is keyed by version so a bump never
  silently reuses an older dataset
- REQ-UX-01: PlatypusStudio shows dataset size and an explicit download action, never a
  silent stall; mmcli never starts an implicit multi-megabyte transfer in a non-interactive
  invocation (D-5)
- REQ-DOC-01: No statement in README about dataset location or `MMCLI_DATASETS` is false
  after unbundling; the offline recipe is written down; `docs/RELEASING.md` states the
  dataset obligations of cutting a release and why they exist

**Depends on:** Phase 9
**Plans:** 8 plans in 5 waves

Plans:
- [ ] 10-01-PLAN.md — wave 1 — Enforce PyInstaller exclusions in all three build scripts + single-source size ceiling (REQ-SIZE-01/02)
- [ ] 10-02-PLAN.md — wave 2 — Registry digests/versioning, version-scoped cache, verified `fetch_dataset` (REQ-DATA-01/02/03/05)
- [ ] 10-03-PLAN.md — wave 3 — GET-and-hash gate over all nine TI URLs, then unbundle (REQ-SIZE-01, REQ-DATA-04/05)
- [ ] 10-06-PLAN.md — wave 3 — `mmcli datasets list/pull/path` + D-5 auto-fetch policy (REQ-DATA-01/03, REQ-UX-01)
- [ ] 10-04-PLAN.md — wave 4 — PlatypusStudio download affordance, cross-repo (REQ-UX-01)
- [ ] 10-05-PLAN.md — wave 4 — README, ten-dataset offline recipe, executed as written (REQ-DOC-01, REQ-DATA-03/04)
- [ ] 10-08-PLAN.md — wave 4 — Wire the new regression guards into CI + per-artifact size and empty-bundle gates (REQ-SIZE-01/02)
- [ ] 10-07-PLAN.md — wave 5 — docs/RELEASING.md, CLI help, Sphinx (REQ-DOC-01, REQ-DATA-05)

---

## v1.0 Milestone — Core Functionality & Security (complete)

**Phases:** 1-5

### Phase 1: Foundation & Core Functionality
**Goal:** Build core CLI structure with basic commands and security hardening.

**Status:** Completed

### Phase 2: Advanced Features & Integration  
**Goal:** Implement advanced features for the mmcli tool while maintaining security measures established in Phase 1.

**Depends on:** Phase 1

**Plans:** 6 plans (3 TDD test creation, 3 documentation/config updates)

**Tasks:**
- [x] `info` command - Show supported devices, models, and presets
- [x] `analyze` command - Analyze project dataset contents
- [x] `recommend` command - Recommend models and FE presets  
- [x] `deploy` command - Handle deployment operations
- [x] Security testing for all new features (REQ-TESTS-07)
- [x] Documentation updates with security best practices

### Phase 3: Testing and Documentation
**Goal:** Comprehensive testing and documentation for the mmcli tool.

**Depends on:** Phase 2

**Plans:** 6/7 plans executed

**Tasks:**
- [ ] Fix integration test failures (REQ-TESTS-07)
- [ ] Fix E2E temp directory issues
- [ ] Unit tests for builder and datasets modules (REQ-TESTS-08)
- [ ] Full workflow integration tests
- [ ] Cross-platform compatibility tests
- [ ] API documentation generation (REQ-TESTS-10)

Plans:
- [x] 02-01-PLAN.md — Unit tests for info command (TDD)
- [x] 02-02-PLAN.md — Unit tests for analyze command (TDD)
- [x] 02-03-PLAN.md — Unit tests for recommend command (TDD)
- [x] 02-04-PLAN.md — Unit tests for deploy command (TDD)
- [x] 02-05-PLAN.md — Document environment variables in CLI help
- [x] 02-06-PLAN.md — Config file examples documentation
- [x] 03-01-PLAN.md — Fix integration test failures
- [x] 03-02-PLAN.md — Fix E2E temp directory issues
- [x] 03-03-PLAN.md — Unit tests for builder, datasets modules
- [x] 03-04-PLAN.md — Full workflow integration tests
- [x] 03-05-PLAN.md — Cross-platform compatibility tests
- [x] 03-06-PLAN.md — API documentation generation

### Phase 4: Security Enhancements
**Goal:** Enhanced security testing, documentation, and input validation.

**Depends on:** Phase 3

**Plans:** 5 plans (2 tdd/fix, 1 sec, 1 doc)

**Tasks:**
- [ ] Fuzz testing framework using hypothesis (REQ-SEC-01)
- [ ] Attack surface mapping and verification tests
- [ ] Security documentation (SECURITY.md, threat model) (REQ-SEC-02)
- [ ] Improved input validation with length limits
- [ ] Dependency vulnerability scanning integration

Plans:
- [x] 04-01-PLAN.md — Fuzz testing framework using hypothesis
- [x] 04-02-PLAN.md — Attack surface mapping & tests
- [x] 04-03-PLAN.md — Security documentation (SECURITY.md, threat model)
- [x] 04-04-PLAN.md — Improved input validation with length limits
- [x] 04-05-PLAN.md — Dependency vulnerability scanning integration

### Phase 5: New Features & UX
**Goal:** Enhanced user experience and new features.

**Depends on:** Phase 4

**Plans:** 6 plans (all feat)

**Tasks:**
- [ ] Progress visualization for long-running operations (tqdm)
- [ ] Export formats (CSV, JSON, YAML) with -o flag
- [ ] Model comparison command (--compare)
- [ ] Batch processing for multiple projects/directories
- [ ] Troubleshooting assistant (diagnose command)
- [ ] Interactive shell mode (shell subcommand)

Plans:
- [x] 05-01-PLAN.md — Progress visualization (tqdm integration)
- [x] 05-02-PLAN.md — Export formats (CSV, JSON, YAML)
- [x] 05-03-PLAN.md — Model comparison command
- [x] 05-04-PLAN.md — Batch processing capabilities
- [x] 05-05-PLAN.md — Troubleshooting assistant (diagnose)
- [x] 05-06-PLAN.md — Interactive shell mode
