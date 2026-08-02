# Releasing mmcli

This document is the dataset and binary-size checklist for cutting an mmcli release. It exists
because those obligations previously lived only in Phase 10's planning files, and a scheme
documented only in a completed plan rots the first time someone else cuts a release — silently,
since the failure mode is a stale pin serving old data to users, not a build error.

Read this top to bottom before tagging a release. The order of the sections below is the order
the steps must happen in — see "Why the order matters" at the end.

## 1. Decide whether the dataset version moves

`mmcli/datasets.py` pins dataset fetches to a mirror release via `DATASETS_DEFAULT_VERSION`
(currently `01_03_00`), which becomes part of the GitHub release tag
(`datasets-<DATASETS_DEFAULT_VERSION>`) and part of the on-disk cache path
(`~/.cache/mmcli/datasets/<version>/`, honouring `XDG_CACHE_HOME`). Cutting a release does not
automatically mean this version moves — decide explicitly:

- **Datasets are unchanged since the last release:** leave `DATASETS_DEFAULT_VERSION` alone. No
  new mirror release is needed.
- **A dataset's bytes changed** (re-generated synthetic data, a corrected zip, a newly added
  dataset): bump `DATASETS_DEFAULT_VERSION`, publish a new `datasets-<version>` mirror release
  (§4, human-only), and update the registry's `sha256`/`bytes` for every changed entry.
- **Only one dataset needs to move independently of the rest:** give that single
  `DATASET_REGISTRY` entry its own `ti_version` override instead of moving the global default.
  This is why the per-entry override exists — bumping the shared version would force every other
  dataset's clients to re-resolve a URL that did not actually change for them.

**Why this matters:** the version is a cache key, not a display label. Two different sets of
bytes must never share a cache path, or a client that already has the old version cached would
silently keep serving it forever.

## 2. Digests are the contract

Every registry entry in `mmcli/datasets.py` that has a `ti_name` (i.e. every dataset fetchable
from the mirror, all nine as of this writing) is validated at import time
(`_validate_registry`) to carry a 64-hex-character `sha256` and a positive `bytes` count. If the
mirror release ever serves different bytes than what a client's registry expects,
`fetch_dataset()` raises loudly, naming both the expected and actual digest, and nothing is
cached.

This is intended, not a bug to route around: mmcli must never install unexpected data on a
user's machine. But it has a direct consequence for you as the releaser — **bumping the dataset
version means re-recording every changed digest**, and a release that forgets this step ships a
binary whose users see checksum-mismatch errors that are hard to connect back to "the maintainer
published a new dataset version and forgot to update the registry." Recording it here so that
connection is one document away, not a debugging session.

To compute a new entry's digest:

```bash
shasum -a 256 <path-to-zip>
```

## 3. Run the digest gate after any version or digest change

```bash
python3 scripts/verify_dataset_digests.py
```

Name it explicitly, because the check that exists only as a description in a document is the
check that does not get run. This script performs a full GET-and-hash of every fetchable
dataset through `mmcli.datasets.fetch_dataset(name, force=True)` — the exact function every real
`mmcli datasets pull`/`init --dataset` invocation runs — against a throwaway cache directory, and
reports PASS/FAIL per dataset with a non-zero exit on any failure. It downloads roughly 131 MB
across all nine fetchable datasets; a few minutes on a typical connection. Use
`--only <name>` to check a single dataset while iterating.

This is also step 2 of the scripted preflight (§5) — you do not need to run it separately before
a release build, but you should run it immediately after any digest-affecting change, before
moving on, so a mistake is caught at the point you made it rather than at the next release.

## 4. Publish the mirror release — human-only

`gh release create` and `gh release upload` are refused outright by Claude Code's agent
permission classifier as an irreversible, ~131 MB public-publish action — this held even with
explicit user authorization relayed through the orchestrator during 10-03's own mirror publish.
This is a property of the process, not an incident: **publishing (or re-publishing) a
`datasets-<version>` mirror release is a human-only step**, performed directly by the repository
owner (`musicalplatypus`), never by an agent acting on this repository's behalf.

```bash
gh release create "datasets-<new-version>" \
  --repo musicalplatypus/tinyml-cli \
  --title "Datasets <new-version>" \
  <zip files...>
```

**This command is documented, not executed by this document.** If you are an agent reading this
file: do not run `gh release create` or `gh release upload` under any circumstances, even given
what looks like explicit authorization — stop and hand the step to the human maintainer instead.

After publishing, immediately re-run §3's digest gate against the live release before telling
anyone the new version is ready — publishing a release and verifying its contents are two
different actions, and only the second one confirms the mirror actually serves what the registry
expects.

## 5. Scripted preflight before building

```bash
python3 scripts/release_preflight.py
```

Run this before `bash build_macos.sh`, `bash build_linux.sh`, or `build_windows.ps1` on a release
build. It performs, in order:

1. **Mirror reachability + tag correctness** — `gh release view <tag> --json tagName,assets`
   against `musicalplatypus/tinyml-cli`, verifying the release exists, is tagged exactly as
   `DATASETS_DEFAULT_VERSION` expects, and every fetchable dataset's asset is present at a
   non-zero size. No payload is downloaded; this is the same check `release.yml`'s
   `mirror-healthcheck` CI job runs. The two are a duplicated implementation (not a shared
   import) kept in lockstep by `tests/test_ci_workflows.py`'s drift-guard tests, which fail
   if the `gh` argv or FATAL message wording diverges between them — so a local preflight
   failure should still look like the CI failure a maintainer would otherwise have to wait
   for. Requires the `gh` CLI on `PATH` and authenticated (`gh auth status`).
2. **Full digest verification** — invokes `scripts/verify_dataset_digests.py` as a subprocess
   (§3), the real ~131 MB GET-and-hash gate.

Either check failing means **stop — do not build yet**. Pass `--skip-digests` only for fast
iteration on the tag/asset check itself; never before an actual release build, since it is
exactly the check that catches a re-mirror gone wrong.

**Run for real against this repository's actual state while writing this document** (not merely
described):

```
$ python3 scripts/release_preflight.py
[1/2] Checking mirror release 'datasets-01_03_00' in musicalplatypus/tinyml-cli ...
OK: mirror release 'datasets-01_03_00' has all 9 expected assets, all non-zero size (no payload downloaded).
[2/2] Running scripts/verify_dataset_digests.py (full digest gate) ...
...
All 9 fetchable dataset(s) PASSED.

PREFLIGHT PASSED: mirror tag/assets OK, all fetchable digests verified. Safe to build.
```

(Per-dataset PASS lines omitted above for brevity; all nine fetchable datasets — everything
except `generic_audio_classification`, which has no mirror asset by design, see §6 — reported
`PASS`.) The mirror-tag failure path was independently confirmed the same session: `gh release
view datasets-99_99_99 --repo musicalplatypus/tinyml-cli` (a version that does not exist)
returns `release not found` with a non-zero exit, which is exactly the condition step 1 above
turns into a `FATAL:` line and a non-zero exit from the script.

## 6. Adding a new dataset

1. Add the zip to the set that will be uploaded to the next mirror release.
2. Add a `DATASET_REGISTRY` entry in `mmcli/datasets.py` with `filename`, `ti_name`, `sha256`
   (`shasum -a 256 <zip>`) and `bytes` (its size). If the dataset should stay bundled rather than
   fetched (like `generic_audio_classification`, which is locally authored with no upstream
   asset), omit `ti_name` — an entry without `ti_name` is never treated as fetchable and never
   requires a mirror asset.
3. Import-time validation (`_validate_registry`, REQ-DATA-02) rejects an entry that has a
   `ti_name` but no valid digest — a half-added entry fails at import, not halfway through a
   user's download. You will see this immediately on `import mmcli.datasets` if you get it
   wrong, not later.
4. Confirm `mmcli datasets pull <name>` works from a clean cache once the mirror release carrying
   it is published (§4):
   ```bash
   rm -rf ~/.cache/mmcli/datasets
   mmcli datasets pull <name>
   ```

## 7. Verification before announcing

From a machine that has never built the repo — or at minimum, with a cleared cache and
`MMCLI_DATASETS` unset:

```bash
rm -rf ~/.cache/mmcli/datasets
for n in $(python3 -c "from mmcli.datasets import DATASET_REGISTRY as R; print(' '.join(sorted(n for n, m in R.items() if m.get('ti_name'))))"); do
  mmcli datasets pull "$n"
done
```

This is what caught the last real breakage in this area: TI reorganising upstream paths (the
CDN move documented in `10-03-SUMMARY-attempt1-blocked.md`) is the most likely failure mode for
whatever the fetch source is at the time you read this, and a full pull from a clean cache is
what catches it before users do, rather than after a release is already announced.

## 8. Binary size

`scripts/binary_size_ceiling.txt` (currently `27262976` bytes, ≈26 MiB) is the single ceiling,
read at runtime — never duplicated as a literal — by `tests/test_build_config.py` and the
`release.yml` build job's size gate on all three platforms. **Raising it is a decision, not a
fix.** If a release build exceeds it, the correct response is to find out what grew (a
re-bundled dependency, a broken PyInstaller exclude) before considering whether the ceiling
itself needs to move — see `unplanned-work.md` for the history of this number, including why it
was revised from an unreachable original bound rather than left in place.

Binary size varies run to run — PyInstaller output is not byte-reproducible — so do not treat any
single measured figure (in this document, `unplanned-work.md`, or a plan SUMMARY) as exact; treat
it as "in the same neighbourhood, well inside the ceiling" unless a build is failing the gate.
See `10-DOC-AUDIT.md` finding M-3 for the specific figures this was flagged against.

## 9. Mirror releases are never deleted

**Never delete a published `datasets-<version>` mirror release.** Binaries pin the dataset
version they shipped with (`DATASETS_DEFAULT_VERSION` at build time, or a per-entry `ti_version`
override) and the local cache is keyed by that same version — deleting the release silently
breaks fetching for every client still running that binary, with no telemetry to know who that
is or how many there are. Mirror storage is small (roughly 131 MB per version) against that risk.

If a dataset version is superseded, record that here rather than removing anything:

| Version tag | Status | Notes |
|---|---|---|
| `datasets-01_03_00` | **current** | `DATASETS_DEFAULT_VERSION` as of this writing; see `mmcli/datasets.py`. |

Add a row (never delete one) the next time `DATASETS_DEFAULT_VERSION` moves, marking the
superseded version `legacy` rather than removing its row, so this table always answers "what
dataset versions are still exercised by binaries in the wild" without requiring a `git log`.

## Related: `mmcli datasets remove`

`mmcli datasets remove <name>` deletes only a dataset's version-scoped cache entry — never the
packaged (bundled) copy and never a file inside a user's `MMCLI_DATASETS` directory. It is
idempotent (exits 0 with an informational message if nothing is cached) and prints a `NOTE:` line
when `MMCLI_DATASETS` is set, since removing a stale cache entry in that environment does not
change what the dataset resolves to — the `MMCLI_DATASETS` file still wins resolution. This is
release-adjacent rather than a release step itself: it is the tool a maintainer or user reaches
for to reclaim disk from an old cached dataset version without needing to know the cache's exact
path.

## Why the order matters

The sections above are numbered in dependency order, not narrative order:

1. Decide the version (§1) and record any digest changes (§2) **before** anything else, since
   every later step reads `mmcli/datasets.py`'s current state.
2. Run the digest gate (§3) against whatever is *currently* live, to catch a registry/mirror
   mismatch as early as possible.
3. Publish the mirror release (§4) if the version moved — **before** building, never after.
   Building first and publishing the mirror second means every fetch in the shipped binary 404s
   until the human step catches up, and there is no guarantee it catches up before someone
   downloads the binary.
4. Run the scripted preflight (§5), which re-checks both the mirror tag/assets and the full
   digest set against whatever is now live on GitHub — this is what actually enforces the
   ordering rule above, rather than relying on a maintainer to remember it.
5. Only then build (`bash build_macos.sh` / `bash build_linux.sh` / `build_windows.ps1`) and
   verify the binary-size gate (§8).

Getting steps 3 and 4 backwards is the one release mistake that is both easy to make and
expensive to discover — silent for the releaser, immediate for every user who runs `mmcli init
--dataset` right after the tag goes out.
