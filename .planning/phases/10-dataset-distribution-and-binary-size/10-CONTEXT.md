# Phase 10: Dataset Distribution and Binary Size - Context

**Gathered:** 2026-07-31
**Status:** Ready for planning (scoped to the unfinished plans only)

<domain>
## Phase Boundary

**Discussion was scoped to the remaining work, not the whole phase.** Phase 10 is 6/11
executed at the time of writing — 10-01, 10-02, 10-03, 10-05, 10-06 and 10-10 are committed
and their decisions are settled. The decisions below apply to the four unfinished plans:

| Plan | Delivers |
|---|---|
| 10-04 | PlatypusStudio New Project download affordance (code committed; human checkpoint open) |
| 10-07 | `docs/RELEASING.md`, CLI help corrections, Sphinx |
| 10-08 | CI enforcement of the guards this phase built |
| 10-09 | PlatypusStudio standalone dataset library + `mmcli datasets remove` |

Nothing here re-opens executed work. Where a decision below changes something already built
(D-05, D-10), that is called out explicitly.

</domain>

<decisions>
## Implementation Decisions

### CI gate depth (10-08)

- **D-01: Do not seed datasets in CI; add a mirror healthcheck instead.** CI checkouts hold
  one of ten zips (`.gitignore:10` ignores the rest), so any artifact-level dataset assertion
  passes vacuously. Rather than pulling ~131 MB per OS per run across three runners, add a
  cheap job that verifies the nine mirror assets are reachable (HEAD, not download) and fails
  if any is missing or mis-tagged. Catches a deleted or wrongly-named release — the realistic
  failure — without paying for payloads. Explicitly does **not** verify bytes; the sha256
  gate that does live in `scripts/verify_dataset_digests.py` and runs at release time (D-06).
- **D-02: Gate startup with generous headroom, not at the literal bound.** REQ-SIZE-01's 8 s
  was measured on the maintainer's laptop (6.1–6.6 s actual). Hosted runners are slower and
  load-variable, and macOS runners most of all. CI asserts a much looser bound (20–30 s) to
  catch catastrophic regressions such as a re-bundled training engine, accepting that it will
  not notice a 6 s → 12 s drift. A time-based check tight enough to catch drift would fail for
  reasons unrelated to the change under test.
- **D-03: The size gate is the real gate.** `scripts/binary_size_ceiling.txt` (`27262976`) is
  enforced per artifact on all three platforms before upload, as 10-08 already specifies.

### Release procedure (10-07)

- **D-04: Never delete a published `datasets-<version>` mirror release; document deprecation
  instead.** Binaries pin the dataset version they shipped with and the cache is keyed by it,
  so pruning an old release silently breaks any still-installed client, and there is no
  telemetry to know who that is. Storage is ~131 MB per version — negligible against that
  risk. `RELEASING.md` records which dataset versions are current and which are legacy, so it
  is clear what is still exercised without anything being removed.
- **D-05: `RELEASING.md` is an ordered checklist plus a scripted preflight.** The ordering
  rule — publish the mirror release, verify 9/9 digests, *then* build and ship — must be
  enforced rather than remembered, because getting it backwards means every fetch in a shipped
  binary 404s. The preflight script verifies the mirror release exists and all nine digests
  match before a release build is allowed to proceed. The checklist still names each step's
  failure mode in prose, so a reader understands why the order matters.
- **D-06: The mirror publish is human-only, and the docs must say so.** `gh release create`
  and `gh release upload` are refused by the permission classifier for any agent; the
  `datasets-01_03_00` release was published by the user directly. This is a property of the
  process, not an incident, and belongs in `RELEASING.md` as a named step with an owner.

### PlatypusStudio packaging and verification (10-04, 10-09)

- **D-07: Resolve mmcli from a non-protected location.** The app currently falls back to
  `~/Documents/repos/PlatypusVibes/tinyml-cli/dist/mmcli`, inside a macOS-gated folder, so a
  missing privacy grant makes a present binary report as absent. Prefer a location outside
  the protected folders (e.g. `~/.local/bin`, or an application-support path) so the common
  case needs no permission at all.
  **Partial fix, deliberately:** the user's projects live in `~/Documents/edgeai/`, so the
  workspace list still requires a Documents grant. This removes the permission dependency from
  *finding mmcli*, not from *listing projects*. The diagnostic added during 10-04 already
  explains the latter when it happens.
- **D-08: Ad-hoc signing is accepted; no signing-identity work in this phase.** Its cost is
  that each rebuild is a new application to macOS and grants reset. D-07 reduces how often
  that bites. A stable signing identity would remove it entirely but needs a developer account
  and belongs in its own phase — see Deferred.
- **D-09: Claude drives the app to resolve 10-04's checkpoint.** The user chose this over
  performing the eight checks themselves.
  **Recorded caveat, raised before the decision:** 10-04 is `autonomous: false` precisely
  because "every UI defect found in this project so far passed its tests", and the failures
  actually encountered so far have been permission- and packaging-shaped — the kind an
  automated pass judges worst. The verification must therefore report what was observed, name
  anything it could not exercise, and must not claim a clean pass on inference. The user
  retains the right to re-verify.

### Dataset library scope (10-09)

- **D-10: Expose reclaimable cache size per dataset, independent of state.** A dataset can be
  `bundled` *and* hold a stale cache entry from an earlier download — the packaged copy wins
  resolution, so that disk space is invisible and unreclaimable from the GUI. The four-state
  vocabulary cannot express it. **This requires a change to an executed plan's output:**
  `mmcli datasets list --format json` (10-06) must report cache-entry size separately from
  resolution state. That is an additive field, so existing consumers are unaffected, but
  10-09 must own it and the JSON contract note in `10-06-SUMMARY.md` should be updated to
  match.
- **D-11: The library shows total cache size and its location** — a header line, following the
  same "make invisible state visible" reasoning as the Setup row added in 10-04.
- **D-12: The library supports bulk download of selected datasets.** Preparing for offline or
  air-gapped work is the stated reason REQ-UX-02 exists, and pulling nine datasets one click
  at a time does not serve it.
- **D-13: The library supports filtering by task type and module**, mirroring the `-t` / `-m`
  flags `mmcli datasets list` already accepts — so the CLI and GUI narrow the same way.

### Claude's Discretion

- Exact CI healthcheck implementation (which job, HEAD vs `gh release view --json assets`),
  provided it fails loudly on a missing or mis-tagged asset and downloads no payloads.
- The precise loose startup bound within D-02's 20–30 s range.
- Preflight script's language and location, consistent with `scripts/` conventions.
- SwiftUI layout of the library surface, within D-10..D-13's content requirements.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase requirements and decisions
- `.planning/ROADMAP.md` — REQ-SIZE-01 (as revised 2026-07-31: 26 MiB, `--onefile`, 8 s),
  REQ-SIZE-02, REQ-SIZE-03, REQ-DATA-01..05, REQ-UX-01, REQ-UX-02, REQ-DOC-01
- `.planning/phases/10-dataset-distribution-and-binary-size/unplanned-work.md` — work
  committed outside any plan, including the REQ-SIZE-01 revision 10-08 inherits
- `.planning/phases/10-dataset-distribution-and-binary-size/10-RESEARCH.md` — D-1..D-4;
  **note D-1 ("fetch from TI, do not mirror") and the "TI URL forms" section are superseded**
  by 10-03's mirror
- `.planning/phases/10-dataset-distribution-and-binary-size/deferred-items.md` — two stale
  "from TI" strings in `mmcli/cli.py` help text, for 10-07 to correct

### Contracts the remaining plans consume
- `.planning/phases/10-dataset-distribution-and-binary-size/10-06-SUMMARY.md` — the
  `datasets list --format json` contract and the D-5 TTY auto-fetch policy; **D-10 above adds
  a field to it**
- `.planning/phases/10-dataset-distribution-and-binary-size/10-02-SUMMARY.md` — fetch, cache
  layout, `MMCLI_DATASETS` semantics
- `.planning/phases/10-dataset-distribution-and-binary-size/10-03-SUMMARY.md` — the mirror,
  the narrow `github.com` → `release-assets.githubusercontent.com` redirect allowance, and the
  real build measurements
- `.planning/phases/10-dataset-distribution-and-binary-size/10-10-SUMMARY.md` — why a
  wheel-size CI gate would pass vacuously (directly constrains 10-08)
- `.planning/phases/10-dataset-distribution-and-binary-size/10-03-SUMMARY-attempt1-blocked.md`
  — the blocked first attempt; why the upstream moved

### PlatypusStudio — code in `../PlatypusStudio`, specs in the PARENT checkout

**The specs are not in the PlatypusStudio repo.** They live in the enclosing `PlatypusVibes`
checkout, one level above both repos, and are given here as absolute paths because neither
repo root resolves them:

- `/Users/martin/Documents/repos/PlatypusVibes/docs/superpowers/specs/2026-07-09-platypus-studio-design.md`
  §"Dataset management" — the
  approved contract for the library and gated project creation, including the state/affordance
  table; also §"Workspace window" and §"Architecture" (MMCLIKit = no SwiftUI, unit-testable)
- `/Users/martin/Documents/repos/PlatypusVibes/docs/superpowers/specs/2026-07-09-platypus-studio-project-format.md` — project descriptor
  and the run-directory clobber finding

### Standing constraint
- No "TI", "Texas Instruments" or "Edge AI Studio" in any PlatypusStudio user-facing text or
  planning doc. Technical necessities from mmcli's own README (device names, `C2000_CG_ROOT`)
  are acceptable without the vendor prefix.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tests/test_build_config.py` — 36 source-level guards (excludes, ceiling, `package-data`,
  `MANIFEST.in`), all mutation-tested. 10-08 wires this file into CI; no new guard file needed.
- `scripts/verify_dataset_digests.py` — re-runnable 9/9 mirror gate; the natural core of
  D-05's release preflight.
- `Sources/MMCLIKit/DatasetCatalog.swift` — four-state availability model decoded from
  `datasets list --format json`, plus `explainDownloadFailure` and `formatBytes`. 10-09
  **extends** this; it must not fork a parallel model.
- `Sources/MMCLIKit/ProcessRunner.swift` — streaming subprocess with working cancellation;
  how the library's bulk download must terminate on cancel.

### Established Patterns
- Source-level guards over build-and-measure, because CI checkouts lack the datasets — this
  is what makes D-01 the right call rather than a cheap one.
- mmcli owns the download, the app owns the prompting (D-5). The app never passes `--fetch`;
  `ProcessRunner` pipes stderr so mmcli is non-interactive by construction.
- Removal goes through `mmcli datasets remove`, never the app unlinking a path from
  `datasets path` — that command resolves a *dataset*, not a cache entry, and can return
  packaged or `MMCLI_DATASETS` paths.

### Integration Points
- `.github/workflows/test-cli.yml` and `release.yml` currently collect only
  `test_cli_integration.py` and `test_tier4_cli.py` — every guard this phase built is inert in
  CI today. Both workflows pin python 3.10 (no `tomllib`).
- `release.yml` has no PyPI publish step, so the wheel channel is source/git install only.
- `docs/` has Sphinx (`conf.py`, `index.rst`, `mmcli.rst`); `docs/RELEASING.md` does not exist.

</code_context>

<specifics>
## Specific Ideas

- The mirror healthcheck should fail on a *mis-tagged* release, not only a missing one — the
  realistic mistake is publishing `datasets-1_03_00` or forgetting `--latest=false`, not
  deleting the release outright.
- `RELEASING.md` should state plainly that publishing the mirror is a human step and why
  (agents are refused by the permission classifier), so it reads as process rather than as an
  incident report.
- Bulk download exists for the offline/air-gap case specifically; that is the scenario to
  design it around.

</specifics>

<deferred>
## Deferred Ideas

- **Stable code-signing identity for PlatypusStudio** — would end the re-granting cycle
  permanently, needs an Apple Developer account and signing config in `make-app.sh`. Its own
  phase; D-08 accepts ad-hoc signing for now.
- **CI seeding the nine datasets to enable true artifact-level dataset gates** — rejected on
  cost (~393 MB per release across three runners) in favour of D-01's healthcheck. Revisit if
  a mirror-integrity failure ever reaches a user.
- **`ProjectScanner.scan` silently drops unreadable directories**, so a permissions problem is
  indistinguishable from "no projects" — the same invisible-state trap the Setup row fixes.
  Out of scope for phase 10; belongs with the Studio workspace work.
- **`pytest.ini`'s `[tool:pytest]` header makes its settings inert**, and `pyproject.toml`
  separately declares `[tool.pytest.ini_options]`. Deliberately untouched all phase: fixing it
  activates 30+ never-run test files and a missing `pytest-cov` in CI.
- **`dist/mmcli` was not reproducible from the committed scripts** before 2026-07-31 (a
  27,680,176-byte artifact of unknown provenance sat there). Now reproducible; noted in case
  it recurs.

</deferred>

---

*Phase: 10-dataset-distribution-and-binary-size*
*Context gathered: 2026-07-31*
