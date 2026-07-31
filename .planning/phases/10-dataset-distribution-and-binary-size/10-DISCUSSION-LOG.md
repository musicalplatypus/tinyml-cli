# Phase 10: Dataset Distribution and Binary Size - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-31
**Phase:** 10-dataset-distribution-and-binary-size
**Areas discussed:** CI gate depth, Release procedure, Studio app signing, Dataset library scope

**Scoping note:** the phase was 6/11 executed when this discussion ran, which is not the usual
order. The user chose to scope discussion to the remaining work (10-04, 10-07, 10-08, 10-09)
rather than re-open settled ground.

---

## Scope of discussion

| Option | Description | Selected |
|--------|-------------|----------|
| Discuss the remaining work | Focus gray areas on the 5 unfinished plans | ✓ |
| Discuss the whole phase | Cover executed plans too, as a fuller record | |
| Cancel | Leave decisions where they already live | |

**User's choice:** Discuss the remaining work
**Notes:** Avoids implying executed, committed work is up for change.

---

## CI gate depth (10-08)

| Option | Description | Selected |
|--------|-------------|----------|
| Don't seed — source gates only | Fast, but nothing ever proves the nine are fetchable | |
| Seed on release builds only | Real artifact gates, ~393 MB per release across three runners | |
| Don't seed, add a mirror healthcheck | HEAD the nine assets; catches a missing/mis-tagged release without payloads | ✓ |

**User's choice:** Don't seed, add a mirror healthcheck
**Notes:** Byte-level verification stays where it already is — `verify_dataset_digests.py` at
release time. The healthcheck covers availability only, deliberately.

| Option | Description | Selected |
|--------|-------------|----------|
| No startup gate in CI | Size only; 8 s stays a local expectation | |
| Gate with generous headroom | 20–30 s, catches catastrophic regressions only | ✓ |
| Gate at 8 s as written | Truest to the requirement, likely flaky on loaded runners | |

**User's choice:** Gate with generous headroom
**Notes:** 8 s was measured on the maintainer's laptop (6.1–6.6 s actual); hosted runners are
slower and load-variable.

---

## Release procedure (10-07)

| Option | Description | Selected |
|--------|-------------|----------|
| Keep them forever | Never delete a mirror release; publish-only | |
| Keep the last N versions | Caps storage, silently breaks pinned clients | |
| Keep forever, document deprecation | Never delete; record current vs legacy | ✓ |

**User's choice:** Keep forever, document deprecation
**Notes:** Binaries pin their dataset version and the cache is keyed by it, so pruning breaks
still-installed clients with no telemetry to identify them. ~131 MB per version is negligible
against that.

| Option | Description | Selected |
|--------|-------------|----------|
| Ordered checklist with failure modes | Step-by-step naming what breaks if reordered | |
| Prose explaining the obligations | Shorter, ages better, easier to get wrong | |
| Checklist plus a scripted preflight | Ordering enforced rather than remembered | ✓ |

**User's choice:** Checklist plus a scripted preflight
**Notes:** Shipping binaries before the mirror release means every fetch 404s — worth
enforcing mechanically rather than trusting to a reading of the docs.

---

## Studio app signing (10-04, 10-09)

| Option | Description | Selected |
|--------|-------------|----------|
| Accept it, document the workaround | Re-grant Documents access after each rebuild | |
| Sign with a stable identity | Grants persist; needs a developer account | |
| Move the binary out of Documents | Common case needs no permission at all | ✓ |

**User's choice:** Move the binary out of Documents
**Notes:** Recorded as a partial fix — projects still live in `~/Documents/edgeai/`, so the
workspace list continues to need a grant. Signing identity deferred to its own phase.

| Option | Description | Selected |
|--------|-------------|----------|
| You verify once the grant is given | Person drives the UI, matching `autonomous: false` | |
| I drive the simulator/app myself | Faster; weakest at permission/packaging failures | ✓ |
| Close 10-04 on unit tests, verify later | Unblocks 10-09 now, against the plan's own warning | |

**User's choice:** I drive the simulator/app myself
**Notes:** Concern was raised before the choice — 10-04 is `autonomous: false` because every
UI defect found in this project so far passed its tests, and the failures actually hit have
been permission- and packaging-shaped. User chose this anyway; verification must report what
was observed, name what could not be exercised, and not claim a clean pass on inference.

---

## Dataset library scope (10-09)

| Option | Description | Selected |
|--------|-------------|----------|
| Show reclaimable space per dataset | Needs an additive field in the datasets-list JSON | ✓ |
| Ignore it — terminal-only | Matches the signed-off spec; leaves invisible disk usage | |
| Add a 'clean cache' action | Coarser, no contract change | |

**User's choice:** Show reclaimable space per dataset
**Notes:** Changes an executed plan's output contract (10-06's JSON). Additive, so existing
consumers are unaffected, but 10-09 owns the change and 10-06's contract note needs updating.

| Option | Description | Selected |
|--------|-------------|----------|
| Total cache size and location | Header line, mirrors the 10-04 Setup row | ✓ |
| Bulk download selected | Multi-select pull for offline prep | ✓ |
| Filter by task type or module | Mirrors `mmcli datasets list -t/-m` | ✓ |
| Nothing further | Smallest surface | |

**User's choice:** All three (multi-select)
**Notes:** Bulk download is justified by the offline/air-gap scenario that REQ-UX-02 exists
to serve.

---

## Claude's Discretion

- Healthcheck implementation shape (HEAD vs `gh release view --json assets`), provided it
  downloads no payloads and fails loudly on a missing or mis-tagged asset
- The precise loose startup bound within 20–30 s
- Preflight script language and location, consistent with `scripts/` conventions
- SwiftUI layout of the library surface, within the agreed content requirements

## Deferred Ideas

- Stable code-signing identity for PlatypusStudio — own phase
- CI seeding the nine datasets for true artifact-level gates — revisit if a mirror-integrity
  failure ever reaches a user
- `ProjectScanner.scan` silently dropping unreadable directories — Studio workspace work
- `pytest.ini` / `pyproject.toml` duplicate pytest config — deliberately untouched all phase
