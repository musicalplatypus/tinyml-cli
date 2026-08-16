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

  **Caveats on these figures, added by the 2026-08-02 document audit.** PyInstaller output is
  not byte-reproducible: successive builds of identical source measured 25,256,016 / 25,256,048 /
  25,258,768 bytes, and the pre-exclusion binary 31,839,872 / 31,840,752. Those differences are
  build variance, not disagreement between documents, and are immaterial against ~2 MB of
  ceiling headroom. **The startup bound is tighter than it looks:** `10-03-SUMMARY.md` records
  `~6.6–9.6 s` on this same machine, and the exclusions did not change startup — so the 9.6 s
  observation still applies and would fail the 8 s bound. Re-measured 2026-08-02 over 8 runs:
  6.19–7.17 s, median ~6.45. The bound holds today but has less margin than the 6.1–6.3 s row
  below implies. See `10-DOC-AUDIT.md` M-2 and M-3.

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
**Plans:** 12/13 plans executed

Plans:
- [x] 10-01-PLAN.md — wave 1 — Enforce PyInstaller exclusions in all three build scripts + single-source size ceiling (REQ-SIZE-01/02)
- [x] 10-02-PLAN.md — wave 2 — Registry digests/versioning, version-scoped cache, verified `fetch_dataset` (REQ-DATA-01/02/03/05)
- [x] 10-03-PLAN.md — wave 3 — Mirror the nine datasets to the project's own GitHub release assets, repoint fetch_dataset() (with a narrow github.com → release-assets.githubusercontent.com redirect allowance), verify the mirror end-to-end, then unbundle (REQ-SIZE-01, REQ-DATA-04/05). Replaces the dead TI-URL fetch: TI's CDN now 302s software-dl.ti.com → downloads.ti.com → 404. Registry digests confirmed correct; bytes mirrored from mmcli/example_datasets/.
- [x] 10-06-PLAN.md — wave 3 — `mmcli datasets list/pull/path` + D-5 auto-fetch policy (REQ-DATA-01/03, REQ-UX-01)
- [x] 10-04-PLAN.md — wave 4 — PlatypusStudio download affordance, cross-repo (REQ-UX-01)
- [x] 10-05-PLAN.md — wave 4 — README, ten-dataset offline recipe, executed as written (REQ-DOC-01, REQ-DATA-03/04)
- [x] 10-08-PLAN.md — wave 4 — Wire the new regression guards into CI + per-artifact size and empty-bundle gates (REQ-SIZE-01/02)
- [x] 10-09-PLAN.md — wave 5 — PlatypusStudio standalone dataset library + `mmcli datasets remove`, cross-repo (REQ-UX-02)
- [x] 10-07-PLAN.md — wave 5 — docs/RELEASING.md, CLI help, Sphinx (REQ-DOC-01, REQ-DATA-05)
- [x] 10-10-PLAN.md — wave 5 — Stop the wheel and sdist shipping the nine mirrored datasets: narrow `[tool.setuptools.package-data]` to the one locally-authored zip (the pip-channel sibling of 10-03's `BUNDLED_DATASETS` allowlist), guard it in tests/test_build_config.py, and verify against a real wheel + sdist and a clean-venv install that pulls from the live mirror (REQ-SIZE-03). A wheel built from the current glob measures 108.2 MB.

### Phase 11: PlatypusStudio run archive and training/NAS view verification

> **MOVED 2026-08-04 — this phase now lives in the PlatypusStudio repository**, at
> `../PlatypusStudio/.planning/ROADMAP.md` Phase 1. It was entirely PlatypusStudio work planned
> in this repo because PlatypusStudio had no `.planning/` of its own; it now does.
>
> The requirements source is unchanged and still lives here:
> `.planning/phases/10-dataset-distribution-and-binary-size/deferred-items.md`
> §"Found during an exploratory pass over the training-report and NAS pages" (D-A…D-G).
>
> **Do not plan or execute this phase from tinyml-cli.** The text below is retained only so the
> requirement IDs resolve for anything that already references them.

**Goal:** Make a finished training run reviewable in the app. Today a run archives its metadata
and nothing else, so every downstream view has nothing to show and says so by showing nothing —
blank chart headings for a completed run, no reason for a failed one, and no route at all to the
NAS views. Fix what is recorded first, then make the views honest about what they do and do not
have, then close the gap that let all of this ship: **the SwiftUI target has no test coverage
and cannot get any, because `Package.swift` declares one test target depending only on
`MMCLIKit`.**

**Source:** exploratory pass 2026-08-02 over `ecg_classification`'s three archived runs
(one completed NAS run, two failed). Findings D-A…D-G, with evidence and file/line references,
are in
`.planning/phases/10-dataset-distribution-and-binary-size/deferred-items.md`
§"Found during an exploratory pass over the training-report and NAS pages". **Read that section
before planning — it is the requirements source for this phase.**

**Requirements**:
- REQ-RUN-01: a completed run archives what is needed to review it later — its metrics, its
  artifact paths, its log, and, for a searched run, the fact that it was one. Today
  `run.json` records `"metrics": {}`, `"artifacts": {}`, no `nas` key and no `run.log`
  alongside it, despite `"status": "completed"` (D-A). This is the root cause of REQ-RUN-02..04
  and must be fixed first — an archive that records nothing cannot be displayed.
- REQ-RUN-02: a run view never presents absence as if it were data. A run with no recorded
  metrics says so; it does not render "Loss" and "Accuracy %" as bare headings over blank
  space, which is indistinguishable from a chart that failed to draw (D-B). `MetricCharts`
  has no empty-state path today.
- REQ-RUN-03: a failed run explains itself — status, and enough of the failure (exit status,
  log excerpt, or a pointer to where it is recorded) to act on. Today it renders identically
  to a completed one (D-C).
- REQ-RUN-04: a historical NAS run reaches the NAS surfaces. Routing keys off
  `record.nas != nil`, which the archive never sets, so `NASSearchView` and `ArchitectureView`
  are unreachable for any archived run and the "Searched" badge never appears (D-D). Both views
  remain unverified end-to-end; reaching them live requires launching a real search.
- REQ-RUN-05: the runs table shows a date, not an identifier. `RunsPanel.swift:28` prints
  `r.id.prefix(13)` (`20260711-1956`) while the manifest carries an ISO `startedAt` (D-E).
- REQ-RUN-06: comparison is reachable. `RunsPanel` is written for multi-select
  (`Set<RunManifest.ID>`, `Table(selection:)`) but neither cmd- nor shift-click extended the
  selection during the pass, leaving Compare permanently disabled and `CompareView` dead (D-F).
  **Cause unconfirmed** — the `.simultaneousGesture(TapGesture(count: 2))` at `:26` is a
  candidate, but so is the click automation used. Reproduce by hand before changing code.
- REQ-TEST-01: the SwiftUI target is testable, or the decision not to test it is deliberate and
  written down. One test target exists (`MMCLIKitTests` → `MMCLIKit`); no test imports SwiftUI.
  The 10-04 checkpoint found a defect — a cancelled download rendering mmcli's traceback as a
  failure — that all 138 unit tests passed, which is the argument for this being a real gap
  rather than a stylistic one.

**Also in scope** (recorded in `10-CONTEXT.md`, same subsystem, cheap to fold in):
- `ProjectScanner.scan` silently drops directories it cannot read, so a permissions problem is
  indistinguishable from "no projects" — the invisible-state trap the 10-04 Setup row exists to
  fix.
- Ad-hoc signing makes macOS treat every rebuild as a new application, resetting privacy
  grants. This is what makes repeatable UI verification painful, so it is worth settling here
  even though a stable signing identity is the larger question.

**Out of scope:** the Train launch form and NAS mode switch, which were driven during the pass
and behave correctly (mode swap, Model row removal, Size/Optimize/Search-epochs, pickers
populating from `mmcli info`), and dataset distribution, which is phase 10.

**Depends on:** Phase 10
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd-plan-phase 11 to break down)

### Phase 12: Radar point-cloud classification support

**Goal:** make the `radar` AI module reachable from `mmcli`. Modelmaker gained a radar module
(`ai_modules/radar/`, task type `radar_classification`) via PR #13; `mmcli` cannot see it at all.
`mmcli info -m radar` fails at argument parsing: `invalid choice: 'radar' (choose from
'timeseries', 'vision', 'audio')`.

**Source:** analysis of `tinyml-tensorlab` on 2026-08-06. Upstream commits `6f2c11c` (PR #13,
Radar Point Cloud Classification), `87408c0` (preprocessing + public dataset), `088e680`
(`radar_classification` `run()` dispatched to `main_debug` instead of `main`), `57bb052` (MPS
benchmark).

**What is actually there upstream — verified by executing the registry, not by reading it:**

- Task type `radar_classification`; **exactly one model**, `Pose_and_Fall_model`
  (`model_training_id: LINEAR_4L_PC`), target devices `F28P55` / `F28P65`, trains on cpu/cuda/mps.
- The model's `common.task_category` is **`timeseries_classification`**, while
  `radar/constants.py` defines `TASK_CATEGORY_RADAR_CLASSIFICATION = 'radar_classification'` and
  maps the task type to it. **These disagree.** `get_default_data_dir_for_task()` takes a task
  *category*, so this may change where the dataset is looked for. **Unverified — settle it before
  planning dataset layout.**
- `radar/training/tinyml_tinyverse/radar_classification.py:40` imports its model descriptions from
  the generic `tinyml_modelzoo.model_descriptions.classification` zoo; the radar-specific entry is
  filtered out of 37 registered descriptions by `task_type`.

**Known upstream defects — report, do not fix here** (modelmaker is a separate repo with PRs in
flight; this phase must tolerate these, not repair them):

- `radar/constants.py:522` — `FEATURE_EXTRACTION_PRESET_DESCRIPTIONS` contains **only
  `Mnist_Default`**, whose body is `image_height=28, image_width=28, image_num_channel=1,
  image_mean=0.1307, image_scale=0.3081`. That is MNIST image configuration inside the radar
  module, carried over from `vision/constants.py` (same keys, same relative position).
- `radar/constants.py:529` — `DATASET_EXAMPLES` contains only `mnist_image_classification`,
  pointing at `mnist_classes.zip`. Also vision residue.

**Requirements:**

- **REQ-RADAR-01** — `radar` is a selectable module. `MODULES` at `mmcli/cli.py:89`, the
  `choices=` at `:1437`, and the help strings at `:850` and `:1199` all hardcode the three-module
  list. **Four sites, and a partial fix is worse than none** — argparse would accept `radar` while
  dispatch still rejects it.
- **REQ-RADAR-02** — `info` and `compare` dispatch on radar. Both hardcode a three-branch
  if/elif over `ai_modules.{timeseries,vision,audio}` (`info.py:30-`, `compare.py:20-25`), three
  sites each. Prefer replacing the branching with a lookup keyed on the module name over adding a
  fourth branch in six places.
- **REQ-RADAR-03** — preset selection does not silently mis-serve radar.
  `mmcli/preset_selection.py` imports `ai_modules.timeseries.constants` unconditionally, so its
  channel-aware selection is timeseries-only. Given radar's only preset is an image preset, the
  correct behaviour is very likely to **decline to choose and say why**, not to pick
  `Mnist_Default`. Confirm against a real run before deciding.
- **REQ-RADAR-04** — `mmcli train` completes end to end for `radar_classification` with
  `Pose_and_Fall_model`, verified by an actual run producing an artifact. **Nothing here has ever
  been run through mmcli** — this requirement is where the real risk sits, and it should be
  attempted early rather than last.
- **REQ-RADAR-05** — the radar dataset layout is documented from evidence. Point-cloud data is not
  the CSV/`.npy` shape the timeseries path assumes, and `mmcli`'s project scaffolding
  (`cli.py:2001` and the `init` templates) encodes layout expectations. Determine the real layout
  from `87408c0`'s public dataset before writing any scaffolding.
- **REQ-RADAR-06** — the upstream residue above is reported upstream with evidence, so the MNIST
  preset and dataset example are fixed at the source rather than worked around forever.

**Risks:**

- **Six-plus hardcoded dispatch sites across three files.** The failure mode is a partial rollout
  that passes argparse and fails at import. Any plan should change all sites together or add the
  module through one registry.
- **The radar module is largely vision-derived.** Its constants diff against `vision/constants.py`
  by ~218 lines out of 709. Assume any radar constant is vision's until checked, not the reverse.
- **`mmcli info` exits 0 on an import error** (`mmcli/info.py:40`, found during the 2026-08-06
  analysis). A radar module that fails to import may look like a module with no models. Fix or
  work around this before using `info` as the verification signal for REQ-RADAR-01/02.

**Depends on:** nothing in this repo. Blocks PlatypusStudio Phase 4.
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd-plan-phase 12 to break down)

---

### Phase 13: Hold the modelmaker config contract, and close two knob gaps

**Goal:** stop `mmcli` from silently breaking modelmaker's CUDA auto-defaults, and pick up two
capabilities that exist upstream but are unreachable from the CLI.

**Source:** review of 105 upstream commits (`f484ddf..57bb052`) on 2026-08-06. Full analysis of
the CUDA policy, including a reproduced override bug, is in `.planning/ANALYSIS-cuda-auto-defaults.md`
(handed to the tensorlab session 2026-08-06).

**Requirements:**

- **REQ-CUDA-01** — pin the config contract with a regression test. Modelmaker's
  `apply_hardware_defaults` auto-enables `training.compile_model` and `training.native_amp` on
  CUDA hosts **only when those keys are absent from the config**. Verified empirically: with no
  flags passed, `mmcli` writes exactly `{enable, model_name}` under `training`, so the policy
  fires correctly today.

  **That compatibility is accidental, not designed.** It holds only because `_set()` skips `None`
  (`mmcli/builder.py:127-131`). A reasonable-looking refactor that emits explicit defaults would
  silently disable torch.compile and AMP on every CUDA host — no error, no log line, just slower
  training and different numerics. This is the same failure shape as the earlier
  `feature_extraction_name: "default"` bug, where emitting a key suppressed modelmaker's own
  resolution.

  The test must assert the **absence** of those keys when the flags are not passed, and their
  presence when they are. Absence-assertions are easy to write in a way that passes vacuously —
  make sure the test fails if the keys are emitted.

- **REQ-QUANT-01** — expose `run_quant_train_only`. Modelmaker supports quantisation-only
  retraining; `mmcli` has no flag for it. It was actively fixed upstream during this window
  (`fb5f0f8` — it crashed when the dataset-reuse cache was empty), which suggests real use.
  **Confirm what it actually does before designing the flag** — in particular whether it requires
  a prior float run's artifacts to exist, since that changes whether this is a `train` flag or its
  own subcommand.

- **REQ-COMPILE-01** — verify `--compile-model` across modules. `compile_model` is now wired into
  all four modelmaker modules (`3c900b2` vision, `baf334a` audio, `9a5facc` radar; timeseries
  already had it). `mmcli`'s flag is module-agnostic and *should* work for all of them, but has
  only ever been exercised on timeseries. Verify vision and audio by running them; the radar leg
  **depends on Phase 12** and should be deferred to there rather than blocking this phase.

**Deferred, deliberately — do not fold in:**

`mmcli` writes **17 of modelmaker's 60 `training` params** (measured, not estimated). Unexposed
knobs that look user-facing include `optimizer`, `lr_scheduler`, `weight_decay`, `warmup_epochs`,
`quantization_method`, `quantization_weight_bitwidth` / `activation_bitwidth`,
`partial_quantization`, `output_int`, `dual_op`, `load_saved_model`, `trainable_layers_from_last`,
`num_last_epochs`, and most `nas_*` knobs beyond `nas_epochs` / `nas_optimization_mode`.

This gap is **pre-existing, not caused by these commits**, and "expose all 43" is the wrong
instinct — PROJECT.md's constraint is that users should not need to know flag names. The
quantisation group is the one worth a scoped look, given how central auto-quantisation is to a
run. It needs its own phase and a design argument, not a bulk flag dump.

**Risks:**

- **REQ-CUDA-01 cannot be verified on this hardware.** No CUDA host is available; the analysis
  reproduced the branch logic by patching `torch.cuda.is_available`. The `mmcli`-side test does not
  need CUDA — it asserts what `mmcli` writes, which is hardware-independent — but any claim about
  *resulting training behaviour* does. Keep the test to the config contract and do not overclaim.
- **The upstream contract may change.** The analysis recommends modelmaker fail closed and log
  what it changed. If that recommendation is adopted, REQ-CUDA-01's test may need to track it.
  Coordinate before writing the test, or write it to assert `mmcli`'s side only — which is the
  more durable choice regardless.

**Depends on:** nothing. REQ-COMPILE-01's radar leg depends on Phase 12.
**Plans:** 4 plans — **all complete**

Plans:
- [x] 13-01-PLAN.md — pin the config omission contract (REQ-CUDA-01)
- [x] 13-02-PLAN.md — expose run_quant_train_only; verify --compile-model on vision/audio (REQ-QUANT-01, REQ-COMPILE-01)
- [x] 13-03-PLAN.md — gap closure: unrealistic classification fixture, forecasting's upstream gap
- [x] 13-04-PLAN.md — gap closure: the remaining regression/forecasting/anomalydetection cases, and F-9

#### Phase 13 outcome — closed 2026-08-15

**722 tests, 0 failures** — the first fully green run since channel-aware preset selection landed.
All three requirements met, and no production code changed in the gap-closure work.

**The larger finding came from finally running the full suite.** Ten tests were failing, none of
them a production defect: they had been asserting `rc == 0` on configs that could never have
trained. The selector moved those failures from training time to config-generation time, where the
error names the real cause — so the suite went red the moment it started telling the truth.

**The upstream preset catalog only supports classification:**

| task | presets | usable | usable widths |
|---|---|---|---|
| `generic_timeseries_classification` | 19 | 17 | [1, 3] — works |
| `generic_timeseries_regression` | 2 | 1 | [11] — **F-9** |
| `generic_timeseries_forecasting` | 0 | 0 | **F-2** |
| `generic_timeseries_anomalydetection` | 0 | 0 | **F-2** |

Three of the four generic timeseries task types cannot auto-select a preset. The affected tests now
pin the *specific* upstream-gap message each failure path emits, citing F-2 or F-9, so they become
meaningful again — rather than silently passing — when upstream fills a gap.

**Process note worth keeping:** this only surfaced because the full suite was run. It takes ~14
minutes, which is why it had not been, and two executors abandoned it mid-plan when instructed to.
A suite slow enough to skip is a suite that stops catching regressions.

---

### Phase 14: mmcli cold start — stop paying seconds before doing any work

**Source:** measured 2026-08-15 while investigating a user report that PlatypusStudio's Overview
and model pickers were unresponsive. The app was the symptom; mmcli is the cause.

**Every mmcli invocation pays ~6.6s before doing anything useful.** Decomposition:

| measurement | before | after |
|---|---|---|
| bare Python interpreter | 0.04s | 0.04s (unchanged) |
| `import mmcli` / `import mmcli.cli` | 0.05s / 0.02s | 0.05s / 0.02s (unchanged) |
| **`_detect_training_device()`** | **0.87s**, returns `'mps'`, **uncached**, **3 call sites** | memoised + lazy: **0 calls** for `--version`/subcommand `--help`, **at most 1** for top-level `--help`; pinned by `tests/test_cold_start.py` (14-01) |
| `python -m mmcli --version` from source | 2.75s | **0.04s** (measured 2026-08-15, 3-run median, freshly rebuilt) |
| **onefile binary `--version`** | **6.58s** — no real work done | **3.93s** (measured 2026-08-15, 3-run median against a binary rebuilt from this fix via `build_macos.sh`) |
| `mmcli info -m timeseries -t generic_timeseries_classification --format json` from source | not separately measured before | 2.78s (3-run median) |
| onefile `mmcli info` (same args) | 8–9s (≈1.7s of it real work) | **6.81s** (3-run median, same rebuilt binary) |

**Imports are not the problem.** Two costs dominate, and both are fixable.

**After (REQ-COLD-04, 14-01):** the `_detect_training_device()` fix (14-01-PLAN.md) accounts for
essentially all of the source-run improvement (2.75s → 0.04s, i.e. the full 2.71s, slightly more
than the 2.60s originally measured for the probe alone — within noise of a 3-run median). The
onefile binary drops from 6.58s to 3.93s, a ~2.65s improvement consistent with the same fix; the
remaining ~3.8s (3.93s − ~0.1s bare-binary overhead) is PyInstaller `--onefile` self-extraction,
unrelated to this plan and **not fixed here** — that is REQ-COLD-03's scope, a separate decision
about `--onefile` vs. a directory build (distribution trade-off), not claimed as resolved by 14-01.
The `mmcli info` figures move by roughly the same absolute amount (onefile: 8–9s → 6.81s), which is
consistent with `info` also passing through the now-lazy `_add_training_args()` subparser
construction rather than a change specific to `info` itself.

**Requirements:**

- **REQ-COLD-01** — `_detect_training_device()` stops costing 0.87s on every invocation. It runs
  `system_profiler SPDisplaysDataType` (0.90s standalone) and is called from `cli.py:2150` inside
  `main()` **before argparse is built**, so `--version` and `--help` pay it too. Its own macOS
  fallback is *"any macOS likely has Metal" → `mps`* — the same answer the probe returns in
  practice. Make it lazy (only when a training-device default is actually needed) or cached; its
  result is a constant for a given machine. **Three call sites** — `:411`, `:1910`, `:2150`.
  **Done (14-01):** memoised + made lazy at all three call sites; a real `train`/`run` still
  selects the identical device (verified via `--dry-run`).
- **REQ-COLD-02** — account for the rest. ~1.9s of the 2.75s source figure is unattributed;
  more than one detection call may run per invocation. **Measure before optimising** — do not
  assume the remainder is also detection.
  **Done (14-01-PLAN.md objective):** fully attributed before implementation started —
  `_detect_training_device()` called 3x = 2.60s, plus 0.02s imports = 2.75s; nothing left
  unaccounted for.
- **REQ-COLD-03** — evaluate moving off PyInstaller `--onefile`. The binary adds ~3.8s over running
  from source, which is unpacking on every launch. A directory build would remove it, but changes
  distribution and interacts with the wheel/sdist size work Phase 10 just finished
  (`BUNDLED_DATASETS`, the 108 MB wheel). **Scope the trade-off before committing** — a faster
  binary that is harder to ship may not be worth it.
  **Still open** — 14-01 did not touch this. After 14-01's fix, onefile `--version` is 3.93s
  against a source run of 0.04s; the ~3.8s gap is the unpacking cost this requirement describes,
  now isolated and undiluted by detection overhead.
- **REQ-COLD-04** — the win is verified by measurement, not assertion. Re-run the same
  decomposition afterwards and record before/after in this roadmap.
  **Done (14-01):** before/after table above, measured 2026-08-15 against a binary freshly
  rebuilt via `build_macos.sh` after the fix.

**Why this matters beyond the GUI:**
- Every `mmcli` call a user makes pays it.
- The tinyml-cli test suite takes **14 minutes**, largely from tests that spawn the real binary.
  A suite that slow does not get run — it hid 10 failures for four days (Phase 13).
- It stalled two executor agents mid-plan when they were told to run the full suite.

**Depends on:** nothing. Unblocks PlatypusStudio Phase 6, which may need less caching afterwards.
**Plans:** 1 of 2 complete

Plans:
- [x] 14-01-PLAN.md — detect at most once, and only when needed (REQ-COLD-01/02/04)
- [ ] 14-02 — REQ-COLD-03, the PyInstaller onefile question (unplanned; scope the trade-off first)

#### 14-01 outcome — 2026-08-15

`_detect_training_device()` was running **3× per invocation, 2.60s total**, shelling out to
`system_profiler` before argparse was even built. Now memoised and lazy: **0 calls** for `--version`.

| measurement | before | after |
|---|---|---|
| `python -m mmcli --version` (source) | 2.75s | **0.07s** |
| onefile binary `--version` | 6.58s | **3.93s** |
| onefile `mmcli info …` | 8–9s | **6.81s** |
| **full test suite** | **848s (14:08)** | **458s (7:38)** |

Device selection is unchanged — verified via `--dry-run` with default, explicit `cpu` and explicit
`auto`, all producing identical config. This was a *when*/*how often* fix, not a *what* fix.

**The suite result was not a goal and is the most useful number here.** It got 46% faster without a
single test being touched, confirming that its runtime was largely mmcli cold start paid repeatedly
by tests that spawn the real binary. A 14-minute suite is one people skip — this one hid 10 failures
for four days and stalled two executor agents mid-plan.

**REQ-COLD-03 remains open.** The onefile improvement above is a side effect; ~3.9s of PyInstaller
unpacking is still there and is a separate decision that trades against Phase 10's distribution-size
work.

---

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
