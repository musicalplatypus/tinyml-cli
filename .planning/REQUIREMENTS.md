# Requirements — index and traceability

**This is an index, not the source.** Each requirement's full text, rationale and any revision
history lives in `ROADMAP.md` under its phase. Duplicating the prose here would create two
versions to drift apart — several of these have already been revised once (REQ-SIZE-01 twice),
and a stale copy is worse than no copy.

What this file adds is the thing ROADMAP cannot show: **which plan claims each requirement, and
whether that claim has actually been discharged.**

Generated 2026-08-02 by auditing `requirements:` frontmatter across all phase-10 plans against
the IDs defined in ROADMAP.md. Every phase-10 requirement is claimed by at least one plan.

## Phase 10 — Dataset Distribution and Binary Size

| Requirement | Claimed by | Status |
|---|---|---|
| REQ-SIZE-01 | 10-01, 10-03 | ✅ met — **revised twice**; the 15 MB / 2.5 s original was unreachable. Now ≤ 26 MiB (`27262976`) / < 8 s against a measured 25,256,016-byte build. Decision made outside any plan; see `unplanned-work.md` §1 |
| REQ-SIZE-02 | 10-01, 10-08 | ✅ **met** — shared exclude list (36 local guards, mutation-checked) **and CI enforcement now demonstrated**: run 30767443908 on `7eeae8b` is the first green run since ≥2026-07-22. Ubuntu and macOS pass; Windows fails but is `continue-on-error` by pre-existing design. Getting there took four fixes — the tinyml-tensorlab ref, dataset zips absent from a checkout, a fixture masking registry invariants, and a 30s subprocess timeout too tight for a cold runner. See `10-DOC-AUDIT.md` H-2 |
| REQ-SIZE-03 | 10-10 | ✅ met — wheel 108.22 → 0.10 MB, sdist 108.26 → 0.17 MB. **Added mid-phase** (2026-07-28) after the pip channel was found uncovered |
| REQ-DATA-01 | 10-02, 10-06 | ✅ met |
| REQ-DATA-02 | 10-02 | ✅ met |
| REQ-DATA-03 | 10-02, 10-06, 10-05 | ✅ met |
| REQ-DATA-04 | 10-03, 10-05 | ✅ met — re-verified after unbundling and again after the wheel change |
| REQ-DATA-05 | 10-02, 10-03, 10-07 | ⚠️ **partially discharged** — reworded 2026-07-23 when the upstream CDN moved (302 → 404) and the datasets were mirrored to this project's own release assets. 10-07 still owes the release-process half |
| REQ-UX-01 | 10-04, 10-06 | ✅ met — checkpoint driven against the real app |
| REQ-UX-02 | 10-09 | ✅ met — checkpoint driven against the real app; removal hard gate passed. Cancel-drops-queue remains unverified (see SUMMARY) |
| REQ-DOC-01 | 10-05, 10-07 | ⚠️ **partially discharged** — README/README_zh done and the offline recipe executed as written (10-05). `docs/RELEASING.md` does not exist yet; 10-07 owes it |

**Outstanding for the phase:** 10-07 only (which alone closes REQ-DOC-01 and the
rest of REQ-DATA-05).

## Phase 11 — PlatypusStudio run archive and training/NAS views

Defined 2026-08-02, not yet planned. Full text in ROADMAP.md; the evidence behind each is in
phase 10's `deferred-items.md` §"Found during an exploratory pass over the training-report and
NAS pages", which is the requirements source for planning.

| Requirement | Covers | Status |
|---|---|---|
| REQ-RUN-01 | A completed run archives metrics, artifacts, log and its NAS flag (D-A) | ⬜ not planned — **root cause; fix first** |
| REQ-RUN-02 | A run view never presents absence as data (D-B) | ⬜ not planned |
| REQ-RUN-03 | A failed run explains itself (D-C) | ⬜ not planned |
| REQ-RUN-04 | A historical NAS run reaches the NAS surfaces (D-D) | ⬜ not planned |
| REQ-RUN-05 | The runs table shows a date, not an identifier (D-E) | ⬜ not planned |
| REQ-RUN-06 | Comparison is reachable (D-F) | ⬜ not planned — **cause unconfirmed**, reproduce by hand first |
| REQ-TEST-01 | The SwiftUI target is testable, or the decision not to is written down | ⬜ not planned |

## Earlier milestones

Requirement IDs from v1.0 (`REQ-SEC-01/02`, `REQ-TESTS-07/08/10`) appear in ROADMAP's completed
phases 1-5. They are not re-indexed here; that milestone is closed.

## Known gaps in this index

- **Cross-repo requirements have no verifier.** REQ-UX-01 and REQ-UX-02 are satisfied by code in
  the `PlatypusStudio` repo, which a phase verifier reading `tinyml-cli` cannot inspect. Both
  are marked met/in-flight on the strength of plan SUMMARYs and a driven checkpoint, not
  automated verification.
- **Status here is hand-audited**, not derived by tooling. Re-check it against plan frontmatter
  after any replan.
