# Adversarial review of phase 10's documents — 2026-08-02

Not a re-read. Every claim below was checked by executing something: resolving cited SHAs and
paths against both git objects and the filesystem, re-measuring figures, and querying the
remote. Findings are ordered by consequence, not by document.

Three findings are about the world being different from what the documents say. Three are about
the documents disagreeing with themselves.

---

## H-1 — 33 commits of finished work exist only on this machine

`tinyml-cli` is **24 commits ahead of `origin/main`**; `PlatypusStudio` is **9 ahead**.

`origin/main` sits at `1ff551a` (10-05). Everything after it is local only: 10-08's CI wiring,
10-10's packaging fix, 10-09's `datasets remove` and the whole Manage Datasets library, the
REQ-SIZE-01 revision, and every planning document written since.

**What this is not.** I initially framed this as "a fresh clone is broken". That is wrong and
worth stating: 10-03's mirror repoint landed *before* 10-05, so `origin/main` already fetches
from the release mirror correctly. A clone works.

**What it is.** The dataset library, `datasets remove`, the CI guards, and the packaging fix are
unreviewed by anyone else and unbacked by any remote. The `datasets-01_03_00` release is public
and live, so the *data* half of this phase is published while most of the *code* half is not.

**Not actionable by me** — pushing is the user's call.

## H-2 — CI has been red on all three platforms since at least 2026-07-22, and a requirement claims otherwise

The last three runs on `origin/main` all report `failure`, most recently 2026-07-25. All three
platforms fail at dependency install, before any test executes:

```
ERROR: .../tinyml-tensorlab/tinyml-modelzoo does not appear to be a Python project:
neither 'setup.py' nor 'pyproject.toml' found.
```

Unrelated to phase 10 — a submodule/checkout problem — but it means:

- **`REQUIREMENTS.md` marks REQ-SIZE-02 "✅ met — shared exclude list + CI enforcement".** The
  exclude-list half is real and tested locally. The CI half has **never executed**: the
  workflows 10-08 wired the guards into are unpushed (H-1), and even the workflows that *are*
  pushed fail before reaching pytest. REQ-SIZE-02's own text — "a build that loses the
  exclusions **fails CI**" — has never been demonstrated.
- 10-08's SUMMARY is honest that no workflow was executed. `REQUIREMENTS.md` then flattened that
  into a green tick. **The overclaim is in the index, not the SUMMARY.**

Status corrected below.

## M-1 — a MUST-READ reference in CONTEXT.md points at a path that does not resolve

`10-CONTEXT.md`'s `<canonical_refs>` opens with "Downstream agents MUST read these before
planning or implementing", and the template it follows requires a full relative path on every
entry. Under the heading **"PlatypusStudio (separate repo, `../PlatypusStudio`)"** it lists:

```
docs/superpowers/specs/2026-07-09-platypus-studio-design.md
docs/superpowers/specs/2026-07-09-platypus-studio-project-format.md
```

Those files are not in `PlatypusStudio` and not in `tinyml-cli`. They live in the **parent**
checkout, `/Users/martin/Documents/repos/PlatypusVibes/docs/`. An agent resolving either the
heading's repo or the document's own root finds nothing. `10-09-PLAN.md` cites the same spec
correctly with an absolute path, so the plan is right and the context that feeds planning is
wrong. Fixed below.

## M-2 — REQ-SIZE-01's startup bound is below a measurement recorded in this same phase

The requirement now reads `starts in < 8 s (3-run median)`, justified in ROADMAP by a measured
`6.1–6.3 s`.

But `10-03-SUMMARY.md` records, on the same machine, **`~6.6–9.6 s`** — stated four times, and
the upper end exceeds the bound. ROADMAP also asserts that excluding PIL and cryptography did
not move startup at all, which means the 9.6 s observation cannot be dismissed as belonging to
the older, larger binary.

Re-measured now, 8 runs: **6.19, 6.44, 6.19, 6.40, 6.41, 6.50, 6.38, 6.58** (median ~6.45), and
6.45–6.54 with the GUI also running. So the bound holds today with roughly 1.4 s of headroom —
but the phase's own worst recorded observation would fail it.

The bound is not obviously wrong; what is wrong is that it was set from the narrower of two
samples without acknowledging the wider one. Annotated below rather than changed, because
changing a bound to fit a measurement is what this phase already had to unwind once.

## M-3 — four different binary sizes are stated as fact, and one appears nowhere else

| Figure | Appears in |
|---|---|
| 25,256,016 | REQUIREMENTS.md, STATE.md, unplanned-work.md, 10-08-PLAN, 10-08-SUMMARY |
| **25,256,048** | **ROADMAP.md only** — the REQ-SIZE-01 table |
| 25,258,768 | 10-09-SUMMARY (and the binary on disk right now) |
| 31,840,752 / 31,839,872 | pre-exclusion, differing by 880 bytes across documents |

All are real measurements of different builds — PyInstaller output is not byte-reproducible.
Nothing anywhere says so, which leaves a reader unable to distinguish normal variance from a
typo, and makes the ROADMAP figure look authoritative when no other document shares it. The
variance is immaterial to the gate (~2 MB of headroom) but the silence about it is not. Noted
below.

## L-1 — nine paths cited by phases 1–5 do not exist

`docs/ENVIRONMENT_SECURITY.md`, `docs/SECURITY_AUDIT_LOG.md`, `docs/requirements.txt`,
`mmcli/compile.py`, `mmcli/security.py`, `mmcli/train.py`, `tests/test_batch.py`,
`tests/test_input_validation.py`, `tests/test_workflow_regression.py`.

All in the closed v1.0 milestone. Either they were planned and never built, or they were
renamed and the plans never updated. Out of scope here; recorded so a future audit does not
rediscover it as new.

## Already fixed before this review

A fabricated commit SHA (`d9e4a1f`) in `10-09-SUMMARY.md`, corrected to `2f639eb` and recorded
in `unplanned-work.md`.

---

## What held up

Worth stating, since an adversarial pass that only lists faults is not an audit:

- **Every cited commit SHA resolves** in one of the two repositories, after the one correction.
- **Every cited source path in phase-10 documents exists**, apart from M-1 and `docs/RELEASING.md`,
  which is correctly described as not yet written.
- **Test counts reconcile exactly**: 10-09-SUMMARY claims 41 + 92 = 133 for tinyml-cli and 163
  for PlatypusStudio; both reproduce on demand.
- **Every commit in both repos is attributed** to a SUMMARY, `unplanned-work.md`, or
  `10-REVIEWS.md`.
- **The inconclusive results stayed inconclusive.** 10-09's cancel test and 10-04's unverified
  traceback fix are both recorded as not established, and neither was later quietly upgraded to
  a pass in the index or in STATE.md.
