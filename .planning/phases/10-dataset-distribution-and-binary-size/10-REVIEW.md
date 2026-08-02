---
phase: 10-dataset-distribution-and-binary-size
reviewed: 2026-08-02T22:13:15Z
depth: standard
files_reviewed: 13
files_reviewed_list:
  - .github/workflows/release.yml
  - .github/workflows/test-cli.yml
  - docs/RELEASING.md
  - mmcli/cli.py
  - mmcli/datasets.py
  - scripts/binary_size_ceiling.txt
  - scripts/pyinstaller_excludes.txt
  - scripts/release_preflight.py
  - scripts/verify_dataset_digests.py
  - tests/test_build_config.py
  - tests/test_ci_workflows.py
  - tests/test_datasets_cli.py
  - tests/test_datasets_download.py
findings:
  critical: 2
  warning: 14
  info: 7
  total: 23
status: issues_found
---

# Phase 10: Code Review Report

**Reviewed:** 2026-08-02T22:13:15Z
**Depth:** standard
**Files Reviewed:** 13
**Status:** issues_found

## Summary

The download core (`mmcli/datasets.py`) is the strongest part of this phase. Digest
verification is mandatory and unconditional, the atomic temp-file/`os.replace` pattern is
correct, cache hits are re-hashed on every resolution, the oversize/truncation guards are
real, and the `MMCLI_DATASETS` air-gap refusal is checked before any URL is composed. I
found no way to land unverified bytes in the cache and no way to make `fetch_dataset`
return a path whose sha256 does not match the registry.

The defects cluster in the **verification layer**, not the implementation layer. Two guards
that this phase built specifically to prevent regressions cannot fail:

* the binary-size ceiling test explicitly sanctions the retired 145 MiB value, so a
  one-line revert of `scripts/binary_size_ceiling.txt` passes every test and every CI gate
  and ships the exact artifact REQ-SIZE-01 exists to prevent;
* the zip-slip regression test asserts on a filesystem path that a *successful* escape
  would never write to, so it would pass against a `extract_dataset` with no protection at
  all — which is also the only verification the extraction path has, since
  `extract_dataset` carries no explicit member-path guard of its own.

Both are the same failure class the phase context flagged as having already bitten once.
Several other tests in the phase substitute a helper for the entry point they claim to
cover (`_download_to_cache` standing in for `fetch_dataset`'s stale-cache branch) or omit
the assertion that makes them non-vacuous (a subprocess test that never checks the exit
code).

On `scripts/release_preflight.py`: I could not construct a false PASS. Subprocess
invocation is `shell=False` with a fixed argv and no interpolated user input, there is no
injection surface, and every failure path I traced (missing `gh`, missing script, malformed
JSON, non-zero child exit) fails closed. Its problems are robustness and honesty: it
crashes with a traceback rather than a `FATAL:` line when `gh` is absent, it resolves the
digest script relative to CWD rather than `__file__`, and its docstring — echoed verbatim
in `docs/RELEASING.md` §5 — claims the mirror check is "reused rather than reimplemented"
from `release.yml` when it is in fact a verbatim copy with no drift guard, in a repo that
built `tests/test_ci_workflows.py` specifically to stop that class of drift.

## Critical Issues

### CR-01: Binary size gate is bypassable — the retired 145 MiB ceiling is still sanctioned

**File:** `tests/test_build_config.py:32-39, 489-494`
**Issue:** `SANCTIONED_CEILINGS = (152043520, 15728640, 27262976)` includes `152043520`
(145 MiB), which the comment directly above it labels as the **retired** interim ceiling
from when datasets were still bundled. `test_ceiling_is_a_sanctioned_value` asserts only
membership in that tuple, and its own failure message claims it exists because "a typo here
would loosen the size gate by an order of magnitude."

Editing `scripts/binary_size_ceiling.txt` from `27262976` to `152043520` therefore:

* passes `test_ceiling_parses_as_positive_integer` (positive int),
* passes `test_ceiling_is_a_sanctioned_value` (it is in the tuple),
* passes the `release.yml:215-233` size gate, which reads that file at runtime and has no
  independent bound,

and ships a binary up to 5.6× the REQ-SIZE-01 limit with a fully green pipeline. Nothing
else in the phase constrains the number: `docs/RELEASING.md` §8 says "raising it is a
decision, not a fix", but no code enforces that. `15728640` (15 MiB) is likewise a retired
value whose presence only matters in the loosening direction if it is ever restored.

**Fix:** Sanction exactly one value — the current one — and record history in a comment
rather than in the assertion set:

```python
# History (comment only — NOT sanctioned values):
#   152043520 (145 MiB) retired: interim ceiling while datasets were bundled
#    15728640 ( 15 MiB) retired 2026-07-31: unreachable
CEILING = 27262976  # 26 MiB, REQ-SIZE-01 as revised

def test_ceiling_is_the_sanctioned_value(self):
    value = int(CEILING_FILE.read_text().strip())
    assert value == CEILING, (
        f"{value} != the sanctioned ceiling {CEILING}. Raising this is a "
        f"deliberate decision (docs/RELEASING.md §8) that must edit this "
        f"constant in the same commit — not a value a typo can reach."
    )
```

### CR-02: The zip-slip regression test asserts on a path a successful escape can never create

**File:** `tests/test_datasets_download.py:423-459` (marker computed at 441, asserted at 450)
**Issue:** The malicious member is `"../../../../tmp/evil_zip_slip_marker.txt"` (line 429).
`extract_dataset` extracts into `<project_path>/dataset/`, i.e. `tmp_path/proj/dataset/`.
If `zipfile` did *not* sanitize member paths, four `..` segments from
`tmp_path/proj/dataset/` resolve to `tmp_path/../..`, so the escaped file would land at
`<pytest-basetemp-parent>/tmp/evil_zip_slip_marker.txt`.

The test asserts on `marker = tmp_path / "tmp" / "evil_zip_slip_marker.txt"` — one level
*inside* `tmp_path`. That path is written by neither the safe behaviour nor the escape. The
assertion `not marker.exists()` is true unconditionally; the test would pass identically
against an `extract_dataset` with no protection whatsoever, and against one that wrote the
member to an arbitrary absolute path.

This matters because `mmcli/datasets.py:846-847` uses a bare `zf.extractall(dataset_dir)`
with no explicit member-path guard. The security property is entirely delegated to
`zipfile._extract_member`'s sanitisation — correct in current CPython, but undocumented as
a stability guarantee and, per this test, unverified. The class docstring
(`tests/test_datasets_download.py:416-421`) states the plan's threat model "calls for an
explicit test rather than an assumption"; that test does not currently exist. Note this
path is reachable with attacker-influenced content via `MMCLI_DATASETS`, which is
deliberately *not* digest-verified (`mmcli/datasets.py:342-347`).

**Fix:** Assert on the escape's real destination, and assert positively that every
extracted file stays under the project directory:

```python
project_path = tmp_path / "proj"
escape_root = (project_path / "dataset" / "../../../../tmp").resolve()
marker = escape_root / "evil_zip_slip_marker.txt"
assert not marker.exists(), f"zip-slip member escaped to {marker}"

extracted = [p.resolve() for p in project_path.rglob("*") if p.is_file()]
assert extracted, "nothing was extracted — the test proved nothing"
for p in extracted:
    assert p.is_relative_to(project_path.resolve()), f"{p} escaped {project_path}"
```

Additionally, add the explicit guard in `extract_dataset` so the property is enforced by
mmcli rather than inherited:

```python
with zipfile.ZipFile(zip_path, "r") as zf:
    root = os.path.realpath(dataset_dir)
    for member in zf.namelist():
        target = os.path.realpath(os.path.join(dataset_dir, member))
        if not (target == root or target.startswith(root + os.sep)):
            print(f"ERROR: refusing to extract '{member}' from '{zip_path}': "
                  f"it escapes {dataset_dir}", file=sys.stderr)
            sys.exit(2)
    zf.extractall(dataset_dir)
```

## Warnings

### WR-01: `init --dataset` downloads up to 56 MB before validating the destination or the task

**File:** `mmcli/cli.py:2156-2171`; `mmcli/datasets.py:807-825`
**Issue:** `main()` runs the D-5 auto-fetch policy — including the actual
`fetch_dataset()` call — *before* `extract_dataset()`, which is where the
"project directory already exists" check (`datasets.py:819-825`) and the task-compatibility
check (`datasets.py:807-815`) live. So:

```
mmcli init -t motor_fault --dataset fan_blade_fault -p ./already_exists
```

downloads 56,595,859 bytes and *then* exits 2 with "Project directory already exists". The
same happens for an incompatible `--task`. Both are pure argument errors detectable in
microseconds. No test covers this ordering.

**Fix:** Hoist the two cheap validations ahead of the fetch, in `main()` before the policy
call:

```python
if args.dataset in DATASET_REGISTRY:
    meta = DATASET_REGISTRY[args.dataset]
    if args.task and args.task not in meta.get("task_types", []):
        print(f"ERROR: Dataset '{args.dataset}' is not compatible with task "
              f"'{args.task}'. Compatible tasks: {', '.join(meta['task_types'])}",
              file=sys.stderr)
        sys.exit(2)
    if os.path.exists(os.path.abspath(args.project)):
        print(f"ERROR: Project directory already exists: "
              f"{os.path.abspath(args.project)}", file=sys.stderr)
        sys.exit(2)
    if _resolve_dataset_zip(args.dataset) is None:
        _apply_init_fetch_policy(args.dataset, args)
```

### WR-02: `release_preflight.py`'s mirror check is a verbatim copy of `release.yml`'s, with a docstring claiming otherwise and no drift guard

**File:** `scripts/release_preflight.py:14-20, 48-112`; `.github/workflows/release.yml:49-112`;
`docs/RELEASING.md:110-113`
**Issue:** The two implementations are line-for-line identical (tag construction, `expected`
comprehension, `gh release view --json tagName,assets` argv, `tagName` comparison, missing
and zero-size asset checks, FATAL wording). Both the script docstring ("reused here rather
than reimplemented, so a maintainer running this locally sees the identical failure CI
would report") and `docs/RELEASING.md` §5 ("reused rather than reimplemented") assert the
opposite of what the source shows. Nothing guards the two copies against divergence, in a
repo that built `tests/test_ci_workflows.py` — an entire file whose stated purpose is a
drift guard between these same two workflows — for exactly this failure mode.

**Fix:** Make the claim true. Move the check into an importable module (e.g.
`mmcli/_mirror_check.py` or a `check_mirror(repo) -> bool` in `scripts/`), have
`release_preflight.py` call it, and replace the `release.yml` heredoc with:

```yaml
run: python3 -c "import sys; from scripts.release_preflight import check_mirror_tag_and_assets as c; sys.exit(0 if c() else 1)"
```

If they must stay separate, add a test to `tests/test_ci_workflows.py` asserting the
`gh release view` argv list and the FATAL message strings match between the two files.

### WR-03: `release_preflight.py` crashes with a traceback when `gh` is absent, and resolves the digest script relative to CWD

**File:** `scripts/release_preflight.py:67-71, 122`
**Issue:** Two robustness defects in the newest file of the phase:

1. `subprocess.run(["gh", ...])` raises `FileNotFoundError` — not a non-zero return code —
   when `gh` is not installed or not on `PATH`. The docstring (line 19-20) documents `gh`
   as a requirement, but the code has no handler, so the documented-missing-prerequisite
   case produces a bare Python traceback instead of the `FATAL:` line every other failure
   path produces. The same applies to `from mmcli.datasets import ...` (line 54) when
   `mmcli` is not importable. Both fail *closed* (exit 1), so this is legibility, not a
   false PASS — but a preflight whose job is producing actionable failures should not have
   a traceback among its outputs.
2. `subprocess.run([sys.executable, "scripts/verify_dataset_digests.py"])` (line 122) uses
   a CWD-relative path. Run from anywhere but the repo root, step 1 can still pass (if
   `mmcli` is pip-installed) while step 2 fails on a missing file, producing a confusing
   "PREFLIGHT FAILED at step 2/2 (digest verification)" that has nothing to do with digests.

**Fix:**

```python
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent

# check_mirror_tag_and_assets:
try:
    result = subprocess.run([...], capture_output=True, text=True)
except FileNotFoundError:
    print("FATAL: the `gh` CLI is not on PATH. Install it and run "
          "`gh auth status` — see docs/RELEASING.md §5.", file=sys.stderr)
    return False

# check_digests:
script = REPO_ROOT / "scripts" / "verify_dataset_digests.py"
if not script.is_file():
    print(f"FATAL: {script} not found", file=sys.stderr)
    return False
result = subprocess.run([sys.executable, str(script)], cwd=str(REPO_ROOT))
```

### WR-04: The digest gate has zero test coverage and runs in no CI job

**File:** `scripts/release_preflight.py` (whole file); `scripts/verify_dataset_digests.py`
(whole file); `.github/workflows/release.yml:27`
**Issue:** `grep -rn "release_preflight\|verify_dataset_digests" tests/ .github/` returns
exactly one hit — a *comment* in `release.yml:27`. Neither script is imported by any test,
neither is invoked by any workflow. `docs/RELEASING.md:236` states the preflight "is what
actually enforces the ordering rule above, rather than relying on a maintainer to
remember it" — but running the preflight itself relies entirely on a maintainer
remembering. The failure the doc calls "the one release mistake that is both easy to make
and expensive to discover" is guarded by a script whose own correctness is unverified and
whose execution is unenforced.

`release.yml`'s `mirror-healthcheck` job does gate `build` on tag/asset presence, which
covers the deleted/mis-tagged-release case. The byte-level digest gate — the one that
catches a re-mirror serving different bytes at the same tag — is fully manual.

**Fix:** At minimum, unit-test the two scripts' decision logic with `gh` and the digest
subprocess stubbed (`gh` returning a wrong `tagName`, a missing asset, a zero-size asset, a
non-zero child exit), asserting each returns `False`/non-zero. Ideally add a scheduled or
`workflow_dispatch` CI job that runs `python3 scripts/verify_dataset_digests.py` against
the live mirror, so a silent re-mirror is caught without a release being cut.

### WR-05: The redirect handler locks the host but not the scheme — same-host `http://` and `ftp://` redirects are followed

**File:** `mmcli/datasets.py:475-487`
**Issue:** `redirect_request` compares only `urlparse(newurl).hostname`. `urllib`'s
`HTTPRedirectHandler.http_error_302` permits `http`, `https` *and* `ftp` targets, and
`urllib.request.build_opener` installs `FTPHandler` and `FileHandler` by default. So a
302 from `https://github.com/...` to `http://github.com/...` or `ftp://github.com/...`
passes the host check and is followed — a silent transport downgrade to plaintext.

Impact is bounded: sha256 verification of the streamed bytes is mandatory and runs after
this, so content substitution is still caught, and the requests carry no credentials. But
`ALLOWED_CROSS_HOST_REDIRECTS`'s own comment block (lines 440-452) positions the host lock
as the defence-in-depth layer *around* the digest, and this hole is invisible in it.
`fetch_dataset`'s HTTPS-only check (`datasets.py:693`) applies to the initial URL only.

**Fix:**

```python
def redirect_request(self, req, fp, code, msg, headers, newurl):
    parts = urllib.parse.urlparse(newurl)
    if parts.scheme != "https":
        raise RuntimeError(
            f"Refusing non-HTTPS redirect while fetching "
            f"'{self._dataset_name}': {req.full_url} -> {newurl}"
        )
    new_host = parts.hostname
    ...
```

### WR-06: Test fixtures write synthetic zips into the package source directory, cleaned up only in a teardown that an interrupt skips

**File:** `tests/test_datasets_cli.py:88-119`; `tests/test_datasets_download.py:62-91`
**Issue:** Both autouse fixtures materialise stand-in zips directly into
`mmcli/example_datasets/` (the live package directory,
`_REAL_BUNDLED_DIR` / `os.path.dirname(datasets.__file__) + "/example_datasets"`) and
delete them in the `yield`-teardown. `.gitignore:11` ignores `mmcli/example_datasets/*.zip`,
so leftovers are invisible to `git status`.

Any hard interrupt (Ctrl-C, `pytest -x` on a crashing collection, an OOM kill, a segfault
in a `torch` import) skips the teardown and leaves nine two-file fake zips permanently in
the developer's package tree. From then on:

* `mmcli init --dataset fan_blade_fault -p ./p` silently succeeds and produces a project
  containing a `README.txt`, because `_resolve_dataset_zip` finds the fake at step 2 and
  bundled files are not digest-verified;
* `_REAL_ZIPS_PRESENT` (`tests/test_datasets_cli.py:122-125`, evaluated at import) flips to
  `True`, so the `@_needs_real_zips` subprocess tests stop skipping and start running
  against the fakes — with a subprocess registry whose digests do *not* match, so they fail
  confusingly rather than skipping cleanly.

**Fix:** Do not write into the package directory. Create the stand-ins in `tmp_path` and
redirect resolution there, the way `tests/test_datasets_cli.py`'s own `hide_bundled` fixture
already does:

```python
@pytest.fixture(autouse=True)
def dataset_zips_present(tmp_path_factory, monkeypatch):
    stage = tmp_path_factory.mktemp("bundled")
    for name, meta in DATASET_REGISTRY.items():
        src = os.path.join(_REAL_BUNDLED_DIR, meta["filename"])
        dst = stage / meta["filename"]
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            continue
        with zipfile.ZipFile(dst, "w") as z:
            z.writestr(f"{name}/README.txt", f"stand-in for {name}\n")
        monkeypatch.setitem(DATASET_REGISTRY[name], "sha256", _sha256_of(dst))
        monkeypatch.setitem(DATASET_REGISTRY[name], "bytes", dst.stat().st_size)
    monkeypatch.setattr(datasets_mod, "_datasets_dir", lambda: str(stage))
```

### WR-07: `TestRegistryInvariants` opts out of the stand-in fixture by class-name string, so a rename makes its assertions vacuous

**File:** `tests/test_datasets_download.py:66-71`
**Issue:**

```python
if request.cls is not None and request.cls.__name__ == "TestRegistryInvariants":
    yield
    return
```

The fixture's own comment states the reason correctly: overriding the registry digests for
a class that asserts on those digests "would make it assert against the stand-in and pass
vacuously." That correctness depends on a string literal matching a class name in the same
file. Rename the class (or split a test out of it) and
`test_every_ti_name_entry_has_valid_sha256_and_bytes` (line 193) begins asserting that the
*fixture's own freshly computed* sha256 is 64 hex characters — which it always is. The
REQ-DATA-02 invariant would then be unguarded, silently.

**Fix:** Invert the coupling — mark the class instead of naming it:

```python
# on the class:
@pytest.mark.no_dataset_standins
class TestRegistryInvariants: ...

# in the fixture:
if request.node.get_closest_marker("no_dataset_standins"):
    yield
    return
```

Or, better, move `TestRegistryInvariants` into its own module that does not define the
fixture at all.

### WR-08: `fetch_dataset`'s stale-cache branch is untested — its tests call `_download_to_cache` and do the `unlink` themselves

**File:** `tests/test_datasets_download.py:719-761`; `mmcli/datasets.py:702-707`
**Issue:** `test_corrupted_cache_entry_is_redownloaded` writes a bad cache file, then at
line 731 performs `os.unlink(dest)` itself with the comment "mirrors fetch_dataset's
stale-cache-entry handling", then calls `_download_to_cache` directly. It exercises the
downloader, not the branch it names. `test_force_redownloads_even_when_cached` (743) does
the same for `force=True`. This is precisely the "helper resembling the real code path
instead of the path itself" pattern.

The untested branch is:

```python
if not force and os.path.isfile(dest_path):
    if _sha256_of(dest_path) == meta["sha256"]:
        return dest_path
    os.unlink(dest_path)          # <-- unguarded
```

`os.unlink` here can raise `OSError` (read-only cache, permissions, Windows sharing
violation), escaping `fetch_dataset` as a traceback rather than the `RuntimeError` its
docstring (`datasets.py:663-673`) promises. `_handle_datasets_pull` only catches
`(KeyError, RuntimeError)`, and `_do_init_fetch` only `RuntimeError`, so the user sees a
raw traceback.

**Fix:** Test the branch through `fetch_dataset` with `_download_to_cache` stubbed and
`dataset_url` monkeypatched to an https URL — the pattern
`test_force_flag_bypasses_fetch_dataset_cache_short_circuit` (763) already uses correctly:

```python
def test_stale_cache_entry_is_unlinked_and_redownloaded(...):
    dest.write_bytes(b"stale, wrong content")
    calls = []
    monkeypatch.setattr(datasets, "_download_to_cache",
                        lambda *a: calls.append(a) or str(dest))
    monkeypatch.setattr(datasets, "dataset_url", lambda n: "https://x.invalid/x.zip")
    fetch_dataset(name)
    assert len(calls) == 1
```

And guard the unlink:

```python
try:
    os.unlink(dest_path)
except OSError as exc:
    raise RuntimeError(
        f"Cached copy of '{name}' at {dest_path} failed verification and "
        f"could not be removed: {exc}"
    ) from exc
```

### WR-09: A subprocess guard test never checks the exit code, so it passes if the CLI crashes

**File:** `tests/test_datasets_cli.py:832-842`
**Issue:** `test_datasets_path_does_not_create_the_cache_directory` runs
`python -m mmcli datasets path fan_blade_fault` and asserts only
`not (tmp_path / "mmcli").exists()`. `proc` is never inspected. If the CLI fails to start
at all — an import error, a missing `__main__`, a broken argparse wiring — nothing is
created and the test passes. Its sibling three lines above
(`test_datasets_list_json_does_not_create_the_cache_directory`, line 826) asserts
`proc.returncode == 0`; the omission here looks accidental.

**Fix:**

```python
proc = subprocess.run(...)
assert proc.returncode in (0, 1), proc.stderr   # 1 = "not available locally"
assert "Traceback" not in proc.stderr, proc.stderr
assert not (tmp_path / "mmcli").exists()
```

### WR-10: The exclude-list guard early-returns on a bare substring, so no build script's flags are actually verified

**File:** `tests/test_build_config.py:212-220`
**Issue:**

```python
if _script_reads_shared_list(text):
    assert "--exclude-module" in _strip_comment_lines(text), (...)
    return
```

All three build scripts reference `pyinstaller_excludes` (verified: `build_macos.sh:70`,
`build_linux.sh:47`, `build_windows.ps1:48`), so **every** parametrisation takes this
branch. The assertion is satisfied by the literal string `--exclude-module` appearing
anywhere in non-comment source — including inside an `echo`, a variable name, or a
PowerShell string that never reaches the `pyinstaller` argv. The `shared_modules` set
computed on line 210 is then discarded unused, and the documented fallback (lines 222-229)
is dead code for the current script set.

The scripts are correct today, so this is a latent gate weakness rather than a live bug:
a refactor that stops feeding `EXCLUDE_ARGS` into the `pyinstaller` invocation (deleting
`"${EXCLUDE_ARGS[@]}"` from the command line while leaving the loop above it) passes this
test and silently restores the 260 MB binary.

**Fix:** Assert the flags reach the invocation, not that the token exists:

```python
if _script_reads_shared_list(text):
    code = _strip_comment_lines(text)
    # the array/variable the loop fills must be spliced into the pyinstaller call
    assert re.search(r'pyinstaller[\s\S]{0,800}(\$\{EXCLUDE_ARGS\[@\]\}|\$ExcludeArgs)', code), (
        f"{script_path.name} builds --exclude-module flags but never passes them "
        f"to pyinstaller"
    )
    return
```

### WR-11: `MMCLI_AUTO_FETCH` has no test coverage and silently ignores every value but "1"/"0"

**File:** `mmcli/cli.py:1735-1750`
**Issue:** `_resolve_explicit_fetch` reads `MMCLI_AUTO_FETCH` and returns `True` for `"1"`,
`False` for `"0"`, and `None` for everything else. `MMCLI_AUTO_FETCH=false`,
`MMCLI_AUTO_FETCH=no`, `MMCLI_AUTO_FETCH=off` and `MMCLI_AUTO_FETCH=0 ` (trailing space)
all fall through to rule 4 — "fetch iff stderr is a tty" — meaning a user who set the
variable to disable fetching gets a multi-megabyte download instead, the exact opposite of
their stated intent, with no warning.

The variable is also completely untested: the only reference in `tests/` is
`monkeypatch.delenv("MMCLI_AUTO_FETCH", ...)` at `test_datasets_cli.py:151`. Neither the
`"1"` nor the `"0"` branch, nor the documented "CLI flag beats env var" precedence
(`cli.py:1739-1744`), has a single assertion. It is also absent from `mmcli init --help`
except as a parenthetical inside the `--fetch`/`--no-fetch` text.

**Fix:** Normalise, reject unrecognised values loudly, and test all four cases:

```python
env = os.environ.get("MMCLI_AUTO_FETCH")
if env is not None:
    normalised = env.strip().lower()
    if normalised in ("1", "true", "yes", "on"):
        return True
    if normalised in ("0", "false", "no", "off"):
        return False
    print(f"WARNING: ignoring unrecognised MMCLI_AUTO_FETCH={env!r} "
          f"(expected 1/0); falling back to the TTY rule.", file=sys.stderr)
return None
```

### WR-12: `_validate_args` never runs for `init` or `datasets`, so `init --project` bypasses the path-traversal guard

**File:** `mmcli/cli.py:2135-2183` vs `mmcli/cli.py:2336`, `1877-1889`
**Issue:** `main()` handles `init` and exits at line 2172, and handles `datasets` and exits
at line 2183. `_validate_args(args)` — which applies `_sanitize_input` to module/task/device/
model and `_is_safe_path` to `config`/`onnx`/`project` — is not reached until line 2336.
So `train --project ../../x` is rejected while `init --project ../../x` is accepted and
`extract_dataset` happily `os.makedirs`es it (`datasets.py:818, 843-844`).

No privilege boundary is crossed here — the path comes from the user running the CLI, on
their own filesystem — so this is an inconsistency in the project's declared security
posture rather than an exploitable traversal. But `init` is the one command in the phase
that creates directories, and it is the only one whose `--project` is unchecked. The
`init` path also skips `_sanitize_input` on `--task`, which is then embedded in the printed
"Next steps" command (`datasets.py:872-874`).

**Fix:** Apply the path guard in the `init` branch before extraction:

```python
if args.project and not os.path.isabs(args.project) and not _is_safe_path(args.project):
    print(f"ERROR: --project/-p contains an unsafe path traversal sequence: "
          f"{args.project!r}", file=sys.stderr)
    sys.exit(2)
```

or move the `init`/`datasets` dispatch to after the `_validate_args(args)` call.

### WR-13: A failed extraction leaves a half-created project directory that blocks the retry

**File:** `mmcli/datasets.py:842-853`
**Issue:** `extract_dataset` creates `<project>/dataset/` (line 843-844) and *then* opens
the zip. On `BadZipFile` it prints an error and `sys.exit(2)` with the directory left
behind. The next attempt hits the "Project directory already exists" check (line 819) and
refuses, so the user must manually `rm -rf` a directory mmcli created and then abandoned.
Combined with WR-01, a truncated-but-digest-passing local zip under `MMCLI_DATASETS`
produces exactly this dead end.

The `except` clause is also too narrow: `zipfile.LargeZipFile`, `RuntimeError` (encrypted
member), and `OSError` (disk full, permission denied mid-extract) all escape as raw
tracebacks from a function whose docstring (`datasets.py:791-795`) promises `SystemExit`
"on any error".

**Fix:**

```python
try:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dataset_dir)
except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, RuntimeError) as exc:
    shutil.rmtree(project_path, ignore_errors=True)
    print(f"ERROR: failed to extract '{zip_path}' into {dataset_dir}: {exc}\n"
          f"The partially created project directory has been removed.",
          file=sys.stderr)
    sys.exit(2)
```

### WR-14: Windows test failures do not block the release build or the published Windows binary

**File:** `.github/workflows/release.yml:124, 170`
**Issue:** The `test` job carries `continue-on-error: ${{ matrix.os == 'windows-latest' }}`,
and `build` declares `needs: [test, mirror-healthcheck]`. `continue-on-error` marks the job
successful for dependency purposes, so a completely red Windows test matrix still lets
`build` run and `release` publish `mmcli-<tag>-windows-x86_64.exe`.

This is load-bearing for this phase specifically: `test_build_config.py`'s
`EXPECTED_ADD_DATA_SEPARATOR["build_windows.ps1"] = ";"` guard exists because "PyInstaller
accepts a wrong `--add-data` separator ... producing a binary with an empty bundle and a
successful build" (`release.yml:236-241`). That guard runs in the `test` job — the one
whose Windows result is discarded.

**Fix:** Either drop `continue-on-error` for the release workflow (keeping it in
`test-cli.yml` if Windows flakiness on PRs is the real motivation), or narrow it to the
specific known-flaky tests via `-k`/`-m` deselection so the phase-10 guard files still gate
the build.

## Info

### IN-01: `verify_dataset_digests.py --only <bundled-only>` reports a registry-wide message

**File:** `scripts/verify_dataset_digests.py:113-115`
**Issue:** `--only generic_audio_classification` passes the name validation (it *is* in the
registry), skips at the `url is None` continue, and then reports
`"No fetchable datasets found in DATASET_REGISTRY."` with exit 1 — a message about the whole
registry for a single-name query, when nine fetchable entries plainly exist.
**Fix:** Special-case the `--only` path: `print(f"'{args.only}' has no mirror asset and is
not fetchable (bundled-only).")` and return 2.

### IN-02: `--skip-digests` returns exit 0 for a partial gate, and the step labels still say "1/2"

**File:** `scripts/release_preflight.py:121, 145-151`
**Issue:** `PREFLIGHT PARTIAL` is printed to stderr but `main()` returns `0`, so any wrapper
that checks only the exit status treats a skipped ~131 MB digest gate as a full pass. The
`[1/2]`/`[2/2]` labels are also unconditional.
**Fix:** Return a distinct non-zero code (e.g. `3`) for the partial run and document it, or
require `--i-know-this-is-not-a-release-preflight`. Compute the labels from whether digests
are enabled.

### IN-03: `gh` JSON parsing is unguarded

**File:** `scripts/release_preflight.py:81, 90`; `.github/workflows/release.yml:83, 92`
**Issue:** `json.loads(result.stdout)` raises `JSONDecodeError` on non-JSON output, and
`{a["name"]: a.get("size", 0) for a in data.get("assets", [])}` raises `KeyError` on an
asset without `name` and `TypeError` if `gh` emits `"assets": null`. All produce tracebacks
rather than the file's uniform `FATAL:` lines.
**Fix:** Wrap the parse in `try/except (ValueError, TypeError, KeyError)` and return
`False` with a `FATAL: could not parse gh output` message including `result.stdout[:200]`.

### IN-04: A cache hit for an entry with no `sha256` is reported as a digest mismatch

**File:** `mmcli/datasets.py:383-393`
**Issue:** `expected = meta.get("sha256")`; when it is absent or empty the `if expected
and ...` guard is false and the code falls into the warning branch that says the file "does
not match the recorded sha256" and suggests `mmcli datasets pull <name> --force` — which
will itself fail for a bundled-only entry. `_validate_registry` only requires `sha256` for
`ti_name` entries, so this is reachable for a legitimately digest-less local entry (the
shape `tests/test_datasets_download.py:431-436` constructs).
**Fix:** Distinguish the cases — if `expected` is falsy, either treat the cache hit as
unverifiable-and-absent with a message saying so, or drop the entry from cache
consideration entirely.

### IN-05: Direct `os.environ` mutation and a `finally` that depends on an import inside `try`

**File:** `tests/test_datasets_download.py:442-459`
**Issue:** `import os as _os` sits inside the `try` (line 443) while the `finally` (line 458)
calls `_os.environ.pop`. The test also sets `MMCLI_DATASETS` via raw `os.environ` assignment
and mutates `DATASET_REGISTRY` directly (line 431) instead of using `monkeypatch.setenv` /
`monkeypatch.setitem`, so an assertion failure before the `try` is entered leaves global
state modified for the rest of the session.
**Fix:** Use `monkeypatch.setenv("MMCLI_DATASETS", str(env_dir))` and
`monkeypatch.setitem(DATASET_REGISTRY, "_test_only_zip_slip", {...})`; drop the try/finally.

### IN-06: Only 6 of ~38 test files run in either CI workflow

**File:** `.github/workflows/test-cli.yml:55-63`; `.github/workflows/release.yml:155-164`
**Issue:** Both invocations name six files explicitly. `tests/` contains ~38, including
`test_security.py`, `test_security_fixes.py`, `test_attack_surface.py`,
`test_integration_security.py`, `test_fuzz_path_validation.py` and
`test_fuzz_sanitization.py` — none of which run on any push, PR, or release. Phase 10's
own `test_ci_workflows.py` locks in the six-file *superset* requirement, which does not
prevent this but does make the current scope look deliberate.
**Fix:** Run `python -m pytest tests/ -k "not TestInitDatasetExtractReal"` and deselect the
genuinely environment-dependent files by marker instead of by omission, so a new test file
is in CI by default rather than by remembering to add it in two places.

### IN-07: `_download_to_cache`'s cleanup handler can mask the original exception

**File:** `mmcli/datasets.py:650-653`
**Issue:** `except BaseException: if os.path.exists(tmp_path): os.unlink(tmp_path); raise`
— if the `unlink` itself raises (a race with another process, a read-only mount), that
`OSError` replaces the checksum-mismatch or oversize-body error the user needed to see.
**Fix:**

```python
except BaseException:
    try:
        os.unlink(tmp_path)
    except OSError:
        pass
    raise
```

---

_Reviewed: 2026-08-02T22:13:15Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
