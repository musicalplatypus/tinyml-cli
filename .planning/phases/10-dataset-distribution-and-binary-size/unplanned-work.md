# Unplanned work — substantive changes made outside any plan

Retro-documented 2026-07-31 at the user's request, after an audit found committed
work in this phase with no plan file, no SUMMARY, and no requirement traceability.

Everything here is real, verified, committed work. The problem is bookkeeping: a phase
verifier reading only `10-0N-PLAN.md` + `10-0N-SUMMARY.md` would find requirements
satisfied by changes it cannot attribute, and a reader asking "why is the ceiling
27262976?" would find the answer only in a commit message.

Recorded rather than re-done. None of it is being reverted.

---

## 1. REQ-SIZE-01 revision — the size/startup decision

**Commits:** `1372c09` (the decision), superseding `fb16a53` (the earlier deferral note).
**Requirement:** REQ-SIZE-01 — revised, and now satisfied by this work.
**Should have been:** a plan, through the planner and checker. This is the clearest
miss of the four: it amended a requirement, changed a CI gate's threshold, altered the
shared exclude list, and rebuilt the shipped binary.

**What happened.** REQ-SIZE-01's "≤ 15 MB and < 2.5 s" was written before anything was
measured, and 10-03 discovered neither half was reachable. The decision was deferred
(`fb16a53`), then taken directly in conversation once three real builds gave numbers:

| Build | Size | Startup (median of 5) |
|---|---|---|
| current excludes, `--onefile` | 31,840,752 B | 6.1–6.3 s |
| + exclude PIL, cryptography | **25,256,016 B** | 6.1–6.3 s |
| + those excludes, `--onedir` | 56 MB dir / 29.5 MB zipped | 2.39 s (6.2 s cold) |

Outcome: ceiling `15728640` → `27262976` (26 MiB); startup bound 2.5 s → 8 s;
`--onefile` kept; `PIL` and `cryptography` added to `scripts/pyinstaller_excludes.txt`
(zero references in any mmcli source file — both transitive).

**Verified:** `bash build_macos.sh` produces 25,256,016 bytes, within 32 bytes of the
standalone experiment, so the result is reproducible from the committed scripts.
`--version`, `init --list`, `datasets list`, `analyze` (4.8 M samples through the
numpy/pandas path) and the D-5 non-TTY refusal all unchanged.

**Full rationale** is in ROADMAP.md under REQ-SIZE-01, which carries the measurement
table and why 15 MB is unreachable (numpy and pandas are needed by `analyze` and are
already lazily imported, so the deferral trick is spent).

**Consumed by 10-08**, which owns REQ-SIZE-01/02 and could not previously write its CI
size gate against a ceiling no build could meet. 10-08 should state in its SUMMARY that
it inherits this decision rather than making it.

---

## 2. PlatypusStudio fixes made while debugging — belongs to 10-04

**Repo:** `PlatypusStudio` (separate git repo).
**Commits:** `9a51421` (binary resolution), and the follow-up adding the protected-folder
diagnostic.
**Requirement:** none directly; supports REQ-UX-01 by making the app able to find a
working mmcli at all.

Found by driving the real app, not by a plan:

- **`MMCLIBinary.resolve` accepted an mmcli that could not run.** It probed each
  candidate with `--version` and ignored the result, so a stale `pip` console-script shim
  in `~/.venv-ai/bin` (left behind when this repo moved from `Documents/repos/
  TexasInstruments/` and its editable install broke) won over the working build purely by
  being first on PATH. Now only a candidate that answers `--version` is accepted; when
  nothing runs, the first candidate is returned carrying its failure text rather than
  reported as missing.
- **An unreadable user folder was reported as absence.** macOS gates Documents/Desktop/
  Downloads per application and a denied `stat` is indistinguishable from a missing file,
  so a lost grant made the app claim mmcli was not found while the binary sat in plain
  sight and the project list emptied itself. Both now say so explicitly, including that a
  rebuilt ad-hoc-signed app is treated as a new application and must be granted again.

**Action:** 10-04's SUMMARY, when written after its human-verification checkpoint clears,
must list both as deviations — they are outside its stated `files_modified`
(`DatasetCatalog.swift`, `Preflight.swift`, `NewProjectSheet.swift`,
`DatasetCatalogTests.swift`), though `Preflight.swift` and `MMCLIBinary.swift` are the
files actually touched.

---

## 3. mmcli reinstalled into `~/.venv-ai`

**Commits:** none — environment repair outside any repo.
**Requirement:** none.

The venv's editable install pointed at `/Users/martin/Documents/repos/
TexasInstruments/tinyml-cli`, the repo's location before it moved, so the console script
failed with `ModuleNotFoundError` on every invocation while pip still reported mmcli
0.1.0 as installed. Reinstalled editable from the current path; now reports 1.1.2 and
imports from outside the repo.

No project artifact changed. Logged because it explains the failure in item 2 and
because editable installs breaking silently on a repo move is a recurring failure mode
here (the same thing happened previously with `torchmodelopt`).

---

## 4. `10-03-SUMMARY.md` renamed to `10-03-SUMMARY-attempt1-blocked.md`

**Commit:** `2f29d73`.

Deliberate, and noted here so the non-standard filename is not mistaken for a stray file.
10-03's first run was blocked when the upstream CDN moved its paths; that run's record —
including the curl-verified digests and the diagnosis — was preserved under the renamed
file rather than overwritten by the successful second attempt's SUMMARY.

---

## Process note

Two of these (1 and 2) would have gone through the planner under the project's own
convention, which exists because inline planning has previously missed blockers the
checker caught. Item 1 in particular changed a requirement and a release gate from
conversation. Worth watching for on the remaining plans: 10-04's checkpoint is still
open, and 10-07/10-08/10-09 have not started.

---

## 5. CI workflow ref fix (`56f6cc0`, 2026-07-23)

**Commit:** `56f6cc0` — one line in `.github/workflows/test-cli.yml`.
**Requirement:** none.

Corrected a `tinyml-tensorlab` ref pointing at `platypus_dev_1.4`, a branch that does not exist.
Unrelated to phase 10's subject matter; it landed during the phase window because CI was being
exercised. Recorded here only so that a commit-by-commit audit of the phase finds an
explanation for every code change rather than an unattributed one.

---

## Audit note (2026-08-02)

A traceability audit run at the user's request checked every commit in both repositories since
the phase began against `.planning/`. It found two gaps, both now closed:

- **A fabricated SHA.** `10-09-SUMMARY.md` cited `d9e4a1f` for the completed cache side-effect
  fix. That object does not exist; the real commit is `2f639eb`. The summary is corrected. This
  is the failure mode the project already has a standing note about — an unverified "fact" in a
  committed document is permanent, and this one was written without checking.
- **This CI commit**, previously unattributed.

Everything else reconciles: all nine PlatypusStudio commits and all phase-10 tinyml-cli code
commits are referenced from a SUMMARY, from this file, or from `10-REVIEWS.md`.
