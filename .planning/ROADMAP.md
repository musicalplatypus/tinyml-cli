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
- REQ-SIZE-01: `dist/mmcli` ≤ 26 MiB (`27262976` bytes) and starts in < 8 s (3-run median).
  **REVISED 2026-07-31, replacing "≤ 15 MB and < 2.5 s", which was unreachable.** The original
  numbers were set before anything was measured; both were resolved against real builds on
  macOS arm64:

  | Build | Size | Startup (median of 5) |
  |---|---|---|
  | current excludes, `--onefile` | 31,840,752 B | 6.1–6.3 s |
  | **+ exclude PIL, cryptography, `--onefile`** | **25,256,048 B** | 6.1–6.3 s |
  | + those excludes, `--onedir` | 56 MB dir / 29.5 MB zipped | 2.39 s (6.2 s cold) |

  **Size — 26 MiB.** `PIL` and `cryptography` appear in zero mmcli source files; they were
  transitive. Excluding them saves 6.58 MB (21%) with `--version`, `init --list`,
  `datasets list` and `analyze` (4.8 M samples through the numpy/pandas path) all verified
  identical afterwards. 15 MB stays out of reach because the remainder is numpy and pandas,
  which `analyze` needs for CSV/npy/pkl and which are *already* lazily imported — so the usual
  deferral trick is spent, and going further means dropping that input support.

  **Startup — 8 s, and `--onefile` is kept.** Only `--onedir` reaches 2.5 s, and its cost is
  changing every platform's distribution shape from one file to a folder; the released assets
  are single binaries today. The ~6.1 s is PyInstaller unpacking the archive on each launch,
  not import work — exclusions did not move it at all. The bound is set to 8 s to gate against
  regression rather than to describe an aspiration.

  Consequence: `scripts/binary_size_ceiling.txt` is `27262976`, which a real build passes, so
  10-08's CI size gate is unblocked. This revision was made outside any plan; it is
  retro-documented in
  `.planning/phases/10-dataset-distribution-and-binary-size/unplanned-work.md` §1, and 10-08
  inherits it rather than re-deciding it.
- REQ-SIZE-02: PyInstaller must not bundle the training engine in any of the three published
  artifacts (Linux, Windows, macOS); a build that loses the exclusions fails CI — meaning the
  guard runs inside `.github/workflows/` — rather than shipping a 260 MB binary
- REQ-SIZE-03: the Python distribution artifacts (wheel and sdist) must not carry the nine
  mirrored datasets either — only the locally-authored `generic_audio_classification.zip`
  ships inside the package, so `pip install mmcli` fetches the rest from the mirror like every
  other install path. (Added 2026-07-28: a wheel built from the current `package-data` glob
  measures 108.2 MB, of which 124.9 MB uncompressed is the ten dataset zips. Phase 10
  unbundled the PyInstaller binary only; the pip channel was never covered by REQ-SIZE-01/02,
  which are both written about `dist/mmcli`.)
  **Scope of the bug, verified 2026-07-31:** `.gitignore:10` ignores
  `mmcli/example_datasets/*.zip` and only `generic_audio_classification.zip` is tracked, so a
  build from a *clean clone* already produces a small wheel. The 108.2 MB figure is a property
  of a maintainer working tree that holds all ten. That makes this a release-safety fix — it
  stops the person who cuts a release from publishing a fat wheel — rather than something every
  pip user hits today. It also means a CI artifact-size gate would pass vacuously.
- REQ-DATA-01: Datasets resolvable via `MMCLI_DATASETS` → bundled → cache → download
- REQ-DATA-02: Any registry entry carrying a `ti_name` (i.e. any dataset that can be
  fetched) must carry a `sha256`, enforced at import and verified before extraction; a
  corrupt, truncated or substituted download fails loudly and leaves no cache entry
- REQ-DATA-03: `MMCLI_DATASETS` disables all fetching (offline / air-gap escape hatch)
- REQ-DATA-04: All 10 datasets remain obtainable offline via `MMCLI_DATASETS`
- REQ-DATA-05: datasets are fetched from the project's own versioned GitHub release mirror
  (`releases/download/datasets-<version>/`), with a per-dataset version override; the cache is
  keyed by version so a bump never silently reuses an older dataset. (Reworded 2026-07-23: the
  original upstream, software-dl.ti.com, moved its paths in production — 302 → downloads.ti.com
  → 404 — so the nine datasets are mirrored to release assets from their digest-verified bytes.)
- REQ-UX-01: PlatypusStudio shows dataset size and an explicit download action, never a
  silent stall; creating an example project is blocked until that project's dataset is
  present locally, and the download is an explicit user step rather than a side effect of
  pressing Create; mmcli never starts an implicit multi-megabyte transfer in a
  non-interactive invocation (D-5)
- REQ-UX-02: PlatypusStudio offers a dataset library, reachable at any time and independent
  of project creation, from which any example dataset can be downloaded to local storage and
  removed again; it shows, per dataset, the download size and whether it is currently local
- REQ-DOC-01: No statement in README about dataset location or `MMCLI_DATASETS` is false
  after unbundling; the offline recipe is written down; `docs/RELEASING.md` states the
  dataset obligations of cutting a release and why they exist

**Depends on:** Phase 9
**Plans:** 6/11 plans executed

Plans:
- [x] 10-01-PLAN.md — wave 1 — Enforce PyInstaller exclusions in all three build scripts + single-source size ceiling (REQ-SIZE-01/02)
- [x] 10-02-PLAN.md — wave 2 — Registry digests/versioning, version-scoped cache, verified `fetch_dataset` (REQ-DATA-01/02/03/05)
- [x] 10-03-PLAN.md — wave 3 — Mirror the nine datasets to the project's own GitHub release assets, repoint fetch_dataset() (with a narrow github.com → release-assets.githubusercontent.com redirect allowance), verify the mirror end-to-end, then unbundle (REQ-SIZE-01, REQ-DATA-04/05). Replaces the dead TI-URL fetch: TI's CDN now 302s software-dl.ti.com → downloads.ti.com → 404. Registry digests confirmed correct; bytes mirrored from mmcli/example_datasets/.
- [x] 10-06-PLAN.md — wave 3 — `mmcli datasets list/pull/path` + D-5 auto-fetch policy (REQ-DATA-01/03, REQ-UX-01)
- [ ] 10-04-PLAN.md — wave 4 — PlatypusStudio download affordance, cross-repo (REQ-UX-01)
- [x] 10-05-PLAN.md — wave 4 — README, ten-dataset offline recipe, executed as written (REQ-DOC-01, REQ-DATA-03/04)
- [ ] 10-08-PLAN.md — wave 4 — Wire the new regression guards into CI + per-artifact size and empty-bundle gates (REQ-SIZE-01/02)
- [ ] 10-09-PLAN.md — wave 5 — PlatypusStudio standalone dataset library + `mmcli datasets remove`, cross-repo (REQ-UX-02)
- [ ] 10-07-PLAN.md — wave 5 — docs/RELEASING.md, CLI help, Sphinx (REQ-DOC-01, REQ-DATA-05)
- [x] 10-10-PLAN.md — wave 5 — Stop the wheel and sdist shipping the nine mirrored datasets: narrow `[tool.setuptools.package-data]` to the one locally-authored zip (the pip-channel sibling of 10-03's `BUNDLED_DATASETS` allowlist), guard it in tests/test_build_config.py, and verify against a real wheel + sdist and a clean-venv install that pulls from the live mirror (REQ-SIZE-03). A wheel built from the current glob measures 108.2 MB.

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
