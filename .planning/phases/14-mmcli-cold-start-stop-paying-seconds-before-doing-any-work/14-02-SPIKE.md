# 14-02 Spike: should mmcli move off PyInstaller `--onefile`?

REQ-COLD-03. This is a scoping document, not an implementation. Nothing in `dist/`,
`build_macos.sh`, `mmcli.spec`, `release.yml`, or any source file was changed to produce it — see
"What was and wasn't touched" at the end.

## Baseline (already measured, from 14-01, not re-derived here)

| measurement | value |
|---|---|
| `python -m mmcli --version` (source) | 0.07s |
| onefile `dist/mmcli --version` | 3.93s |
| onefile `dist/mmcli info -m timeseries -t generic_timeseries_classification --format json` | 6.81s |
| onefile `dist/mmcli` size | 24 MB, 1 file |
| REQ-SIZE-01 ceiling | ≤ 26 MiB (27,262,976 bytes) **and** start < 8s (3-run median) |

The onefile binary is currently **inside both** REQ-SIZE-01 budgets. That is worth saying up front,
plainly, because it bears directly on the recommendation below.

## Task 1: what a directory (`--onedir`) build actually buys

### Method

Built with PyInstaller directly (not through `build_macos.sh`, which is `--onefile`-only and was
not modified), replicating its flags — same `--collect-submodules mmcli`, same
`scripts/pyinstaller_excludes.txt` exclude list, same staged `generic_audio_classification.zip`
`--add-data` — but with `--onedir` and `--distpath`/`--workpath`/`--specpath` pointed at a scratch
directory under `/private/tmp/...`. `dist/`, `build/`, and the checked-in `mmcli.spec` in the repo
were never touched; `git status` confirms this at the end.

Installed `mmcli` into `~/.venv-tinyml` the same way `build_macos.sh` does
(`pip install <repo> --force-reinstall --no-deps`) so the built artifact reflects the current
source tree. This only changes the venv's site-packages, not the repo.

### Size and file count

| | onefile | onedir |
|---|---|---|
| total on-disk size | 24 MB | **56 MB** |
| file count | 1 | **762** |
| directory count | 0 | 60 |

Onedir is **more than double** the on-disk footprint and ships 762 separate files instead of one.
This is the opposite of what "size" intuitively suggests a directory build would do — nothing here
is compressed the way the onefile archive's PKG/CArchive is.

### Startup timing — and the finding that matters most

The **already-measured** 3.93s / 6.81s onefile figures are stable every run: onefile re-extracts
into a fresh per-process temp directory on every single invocation, so there is no warm/cold
distinction to it — it's the same cost every time (this is exactly what the `runtime_tmpdir` test
below re-confirms).

Onedir does **not** behave that way. It has a real, reproducible, one-time-per-copy startup penalty
that appeared on every fresh copy tested and vanished on every subsequent run of that same copy:

| scenario | `--version` | `info … --format json` |
|---|---|---|
| **first run after copying the build to a new location** (3 independent fresh copies, 3-run median) | **4.07s** (4.05 / 4.07 / 4.20) | **6.61s** (6.55 / 6.61 / 7.60) |
| **steady state** (same already-run copy, 3-run median) | **0.08s** (0.08 / 0.08 / 0.09) | **2.58s** (2.58 / 2.58 / 2.59) |

Verified this isn't a `touch`/mtime artifact: touching the already-run binary's mtime and re-running
stayed fast (0.20s) — the slow path only re-triggers when the files are placed at a genuinely new
location (a fresh `cp -R`), not on every process launch. `time -l` on a fresh-copy first run showed
zero block I/O and only ~577M cycles — not disk-read-bound — with 699 involuntary context switches,
consistent with the kernel doing (and then caching) ad-hoc code-signature validation on first launch
of a binary at a new path, rather than PyInstaller doing anything expensive. I did not chase the
exact kernel mechanism further since it doesn't change the recommendation either way — the
behavior itself is what matters and it reproduced 3/3 times.

**What this means concretely:**
- **A one-off invocation (CI's "verify binary" step, a CLI user who runs the tool once) is not
  faster with onedir — it is very slightly worse for `--version` (4.07s vs 3.93s) and roughly a
  wash for `info` (6.61s vs 6.81s).**
- **Repeated invocations of the same already-placed copy are dramatically faster**: `--version`
  drops from 3.93s to 0.08s (98% reduction, matching source's 0.07s almost exactly — onedir's
  steady state has essentially no PyInstaller tax left), and `info` drops from 6.81s to 2.58s (62%
  reduction).

So onedir does not uniformly "help." It trades a one-time per-copy cost for near-zero cost on every
run after that. Whether that's a win depends entirely on how many times a given copy gets invoked —
which is exactly why the app/CLI-download asymmetry in Task 2 matters.

### `runtime_tmpdir`: tested, does not help

Built a second onefile variant with `--runtime-tmpdir` pointed at a persistent, pre-existing scratch
directory (rather than the default per-process OS temp dir) and ran it 3 times against the same
target. Result: the target directory was **empty before and after every run** — watching mid-run (a
background process, polled 0.3s in) showed a freshly created `_MEI<random>` subdirectory containing
the full unpacked payload (numpy, pandas, `Python.framework`, `mmcli`, etc.), which was gone again by
the time the process exited. Timing stayed flat at ~4s across all 3 runs regardless of the target
directory already existing from the prior run.

**Factual answer: no.** `runtime_tmpdir` only changes *where* PyInstaller extracts to, not *whether*
it re-extracts. There is no PyInstaller-level flag that turns onefile into "extract once, reuse."
Nobody should propose this again without also proposing patching the bootloader.

## Task 2: what breaks, and the recommendation

### Impact sites (real search, not just the three seeded in context)

**1. `PlatypusStudio/Sources/MMCLIKit/MMCLIBinary.swift`**
- `managedBinaryURL()` (line 61) resolves to a single file path,
  `~/Library/Application Support/PlatypusStudio/bin/mmcli`.
- `installManagedCopy()` (line 78) calls `FileManager.copyItem(at:to:)`, which **does** work
  unmodified on a directory — copying an onedir tree to that same destination path would work
  mechanically.
- **But `resolve()` (line 109) has a latent bug this would trigger**: it filters candidates with
  `fm.isExecutableFile(atPath: url.path)`. A directory's default permissions (`rwxr-xr-x`) make
  `isExecutableFile` return `true` for a directory too — it doesn't distinguish "traversable" from
  "runnable." If an onedir tree were copied straight to `managedBinaryURL()`'s path (a directory
  named `mmcli` containing an executable also named `mmcli` plus `_internal/`), the candidate would
  pass the filter, `ProcessRunner` would try to launch the *directory* as a process, and it would
  fail and get reported as `firstBroken` with `launchDiagnostic: "Could not be launched."` — a
  regression that looks like a broken install, not a build change.
- **Real fix required, not just "copy the folder"**: `managedBinaryURL()` would need to point one
  level deeper (e.g. `.../bin/mmcli/mmcli`), and `installManagedCopy` would need its destination
  logic adjusted so the copy lands as a sibling directory rather than at the leaf file path.
- `Tests/MMCLIKitTests/MMCLIBinaryTests.swift` asserts single-file behavior throughout (`isExecutableFile`
  checks, `installManagedCopy` round-trips) and would need onedir-shaped fixtures added.

**2. `PlatypusStudio/Sources/PlatypusStudio/SetupSheet.swift`**
- The manual "mmcli binary" picker (line 107-110) is `NSOpenPanel` configured
  `canChooseFiles = true, canChooseDirectories = false`. A user could still point it at the nested
  executable *inside* an already-placed onedir tree (it's still technically a file), so this isn't a
  hard break, but it's a usability wrinkle worth flagging: "pick the mmcli binary" now means
  "pick the file three levels inside the folder you unzipped, and don't move it away from its
  `_internal` sibling," which is a materially worse instruction than today's "pick the file you
  downloaded."

**3. `.github/workflows/release.yml`** — five separate places assume a single file:
- The size gate (`SIZE=$(wc -c < "$BINARY")`) — meaningless for a directory; would need to become a
  recursive byte sum (`find ... | xargs stat` or `du -sk`), and the ceiling itself would need
  redefining (56 MB onedir vs the current 26 MiB single-file ceiling aren't comparable numbers).
- `Verify binary` / `Gate — bundled dataset payload` steps — both just invoke `${{ matrix.binary }}`,
  which works unchanged as long as `matrix.binary` points at the nested executable.
- `Gate — startup regression` — currently loose (25s) enough to pass either way, but this spike's own
  finding means a fresh-per-CI-run onedir build would measure its **first-run** cost here
  (4-7s range measured), not the steady-state number that's the whole appeal of onedir. The gate
  would still pass, but anyone reading its output as "the user experience" would draw the wrong
  conclusion.
- Release job's binary rename/upload step (`cp artifacts/macos/mmcli release/mmcli-...`) assumes a
  single file it can `cp` and `chmod +x`. A directory needs archiving (zip/tar) before it can be a
  single downloadable release asset, and downloading it becomes "download, unzip, chmod, run" instead
  of "download, chmod, run."

**4. `docs/RELEASING.md` §8** — describes the size ceiling in single-file terms
(`scripts/binary_size_ceiling.txt`, read by `wc -c`-style logic); would need rewriting for whatever
new measurement onedir requires, not just a new number.

**5. `tests/test_build_config.py`** — `CEILING = 27262976` and its surrounding comments are written
assuming the shipped artifact is one file whose byte count that constant bounds. This test is
source-level (it parses build scripts, not built output) so it would not itself fail if onedir
shipped, but its stated rationale would become misleading — the constant would no longer describe
what actually gets measured against it in CI.

**6. `README.md` / `README_zh.md`** — explicit instructions
`cp dist/mmcli /usr/local/bin/mmcli` and "Copy `dist/mmcli` anywhere on your `PATH`." Both are the
literal CLI-download workflow this spike is evaluating, and both break for a directory: you cannot
usefully "copy a folder onto PATH" the same way, since PATH resolution expects an executable file at
`<dir>/mmcli`, not a directory. The instruction would become "unzip somewhere and add that location
to PATH" — a strictly worse instruction for a download-and-run tool.

**7. Tests that shell out to the binary**: checked — none do. `tests/test_advanced_training_knobs.py`
and the rest of `tests/` invoke `sys.executable -m mmcli` (the venv's Python running the module
directly), never `dist/mmcli`. The onefile/onedir choice has **no effect on the test suite's
runtime** — 14-01's win (848s → 458s) came entirely from fixing repeated device detection in the
source path, and nothing here changes that further either way.

### The asymmetry driving the recommendation

The app (PlatypusStudio) copies mmcli into Application Support **once** and then invokes it
repeatedly across a session — exactly the usage pattern where onedir's steady-state numbers
(0.08s / 2.58s) apply almost every time, and `copyItem` handles a directory as readily as a file.

A CLI user downloading a release asset gets **one file** today, drops it on PATH, and runs it —
sometimes once, sometimes occasionally, never with the app's "install once, invoke constantly"
profile. For that consumer, onedir's overhead (56MB vs 24MB download, unzip step, "don't move the
inner file away from `_internal`") is pure cost with no guaranteed payoff, since a single or
occasional invocation is the "first run after copy" case — measured the same or slightly worse than
onefile, not better.

These two consumers do not want the same tradeoff. That's why three options were evaluated, not two.

### Option 1: stay onefile

- **Cost:** ~3.86s tax on every invocation, everywhere, always.
- **Benefit:** zero changes to any of the seven impact sites above. One file to build, ship, test,
  size-gate, and document. What exists today.
- REQ-SIZE-01 (≤26 MiB, <8s) is **already met** at 24 MB / 3.93s.

### Option 2: move everything to onedir

- **Cost:** touches all seven impact sites — MMCLIBinary.swift's directory-vs-file bug, the picker,
  five things in release.yml, RELEASING.md, test_build_config.py's rationale, and both READMEs'
  CLI-download instructions. Ships a 56 MB / 762-file artifact instead of 24 MB / 1 file. Makes the
  CLI-download experience strictly worse (unzip step, don't-move-this-file caveat) for a consumer
  who was never the one paying the 3.86s tax in a way that matters to them (occasional invocation).
- **Benefit:** the app's managed-copy path gets the full win (0.08s / 2.58s steady state) — but so
  does *every* onedir consumer, including the CLI-download one where the win doesn't materialize the
  same way.
- This is the "fastest but changes what users download" option named in the plan, and the impact
  list above is the concrete size of that change.

### Option 3: hybrid — onedir for the app's managed copy, onefile for release downloads

- **Cost:** two build outputs from one PyInstaller configuration (the `--onefile`/`--onedir` flag is
  the only difference; everything else — excludes, staged dataset, target arch — stays identical, so
  this is not two divergent build scripts, just one invoked twice, or a `COLLECT` step added to the
  existing onefile spec's `EXE(exclude_binaries=True, ...)`). CI would build and upload two artifacts
  per platform instead of one. `MMCLIBinary.swift` still needs the directory-vs-file fix from Option
  2, scoped to just the app's own managed-copy path — the release/download side of `release.yml`,
  `RELEASING.md`, and the READMEs stay exactly as they are today.
- **Benefit:** the app gets the full onedir win (0.08s/2.58s) without changing anything a CLI-download
  user sees. Only the app-facing surface (MMCLIBinary.swift + its tests) changes; the release/CI
  surface changes only by adding a second build/upload step, not by rewriting the size gate or the
  README.
- **This is the option that isolates the fix from the users who don't want it** — but it's also the
  only one with ongoing dual-artifact maintenance cost: two things to build, two things that could
  independently regress, two things a future contributor needs to remember exist.

## Does the status quo already satisfy REQ-SIZE-01?

**Yes, plainly.** 24 MB is under the 26 MiB (27,262,976-byte) ceiling with roughly 2 MiB of headroom,
and 3.93s is well under the 8s startup bound — both already-revised, already-met numbers from Phase
10/14-01, not this spike. Nothing is broken today. Choosing to move off onefile is not fixing a
failing requirement; it's trading a real, measured 3.86s per-invocation tax for a real, measured
762-file/56MB artifact and 7 files' worth of touched surface, in exchange for a win the app would
feel constantly and a CLI-download user would barely feel at all.

## Recommendation

Of the three, **Option 3 (hybrid)** is the one that seems to fit the requirement's own framing
("scope the trade-off before committing... a faster binary that is harder to ship may not be worth
it") most cleanly: it gives the consumer that actually suffers the tax (the app, via constant
invocation) the full fix, while leaving the consumer for whom the fix doesn't help (an occasional
CLI download) completely alone. The added cost is a second build artifact per platform and the
`MMCLIBinary.swift` directory-handling fix — both bounded, both scoped to files already read for
this spike.

That said, this is a recommendation, not a decision — Martin decides, and there's real
reason to hesitate:

- The `--version` first-run number for onedir (4.07s) is not obviously better than onefile's
  3.93s — the entire case for onedir rests on repeated invocation, which is true of the app's usage
  pattern by construction but is an assumption, not something this spike measured about actual app
  session lengths or how often a real user restarts PlatypusStudio (a restart likely means a
  fresh-ish page-cache/signature-validation state, which would look more like the cold numbers than
  the warm ones).
- The maintenance cost of two build outputs is real and ongoing, not one-time — every future change
  to `build_macos.sh` (excludes, staged datasets, target arch) needs to land in both variants, and
  `test_build_config.py`'s source-level guards would need to grow a way to say "this flag must be
  used exactly once per platform per variant" without doubling every existing assertion.
- REQ-SIZE-01 is already met. This spike hasn't found anything broken — only something that could be
  faster for one specific consumer, at a scoped but nonzero cost.

**What would raise confidence:** actual data on PlatypusStudio session length / mmcli invocation
count per session (turns the "the app invokes it repeatedly" assumption into a number), and whether
the CI startup-regression gate's own artifact-build-then-verify-once pattern is representative enough
of a real CLI-download user's first experience to treat the cold numbers as the ones that matter for
that audience.

## What was and wasn't touched

All PyInstaller output for this spike went to scratch paths under `/private/tmp/...`. `dist/`,
`build/`, the checked-in `mmcli.spec`, `build_macos.sh`, and `release.yml` were never written to.
`git status` at the time of writing shows only this file and `14-02-SUMMARY.md` as new — no
production or build file appears as modified.
