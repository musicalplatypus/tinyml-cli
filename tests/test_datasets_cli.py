"""Tests for the `mmcli datasets` subcommand and the `init --dataset`
auto-fetch policy (decision D-5), added in Phase 10 Plan 06, plus
`mmcli datasets remove` and the ENOSPC-legibility fix added in 10-09-PLAN.md
Task 1.

Covers:

1. `mmcli datasets list/pull/path` — the JSON contract PlatypusStudio (plan
   10-04) decodes, plus pull/path CLI behaviour.
2. The D-5 auto-fetch policy gating `init --dataset`'s implicit fetch.
3. `mmcli datasets remove` — cache-only deletion, guarded so it can never
   touch the primary (bundled / MMCLI_DATASETS) datasets directory
   (10-09-PLAN.md), plus the `cache_bytes` field 10-09/CONTEXT.md D-10 adds
   to the JSON contract.
4. The out-of-space (ENOSPC) download failure now surfacing as a legible
   `RuntimeError` instead of a raw traceback (10-09-PLAN.md Task 1).

Nothing here reimplements download, verification, cache-resolution, or
TTY-detection logic — every network- or cache-related behaviour is exercised
through the real `mmcli.datasets` functions (10-02), with `_download_to_cache`
monkeypatched to prove no test ever contacts the real network. `_download_to
_cache` is replaced either with a function that fails the test if called (to
prove a refusal never reaches the network), or with one that copies an
already-verified bundled zip into the target cache directory (to simulate a
successful fetch without a network round-trip: since the bundled file is
byte-identical to what a real fetch would produce, its digest already
matches DATASET_REGISTRY, so no fake bytes need to be hand-crafted).

Datasets are still bundled at this point in the phase (10-03 unbundles them
next), so tests that need a `downloadable`/`cached`/`unavailable` state must
not rely on that — they hide the bundled copy via the `hide_bundled` fixture
below, which works regardless of whether 10-03 has landed.
"""

import errno
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

import mmcli.cli as cli_mod
import mmcli.datasets as datasets_mod
from mmcli.cli import main as cli_main
from mmcli.datasets import (
    DATASET_REGISTRY,
    DATASETS_DEFAULT_VERSION,
    _cache_dir,
    cache_entry_path,
)

# The real bundled directory, captured once via the same construction
# `_datasets_dir()` uses (package dir / example_datasets), independent of
# any MMCLI_DATASETS monkeypatching later tests apply. Used to fabricate a
# "successful fetch" without any network access: copying this byte-identical
# file is indistinguishable from a real download as far as sha256 is
# concerned.
_REAL_BUNDLED_DIR = os.path.join(os.path.dirname(datasets_mod.__file__), "example_datasets")

# A small, real TI-fetchable entry, used wherever a test needs "some TI
# dataset" without paying for a 45-56 MB copy.
_SMALL_TI_DATASET = "generic_timeseries_forecasting"


# ---------------------------------------------------------------------------
# CI parity: the mirrored zips are not in a fresh checkout
# ---------------------------------------------------------------------------
#
# `.gitignore` ignores `mmcli/example_datasets/*.zip` and only
# `generic_audio_classification.zip` is tracked, so a CI checkout has ONE of the
# ten. Nine tests below fabricate a "successful fetch" by copying a real zip out
# of `_REAL_BUNDLED_DIR`, which on a runner is simply absent — they failed with
# FileNotFoundError once 10-08 wired this file into CI.
#
# Rather than deselect them (10-08's own must-have forbids adding a test to CI
# and then excluding it), synthesise a stand-in for any missing zip and point
# that entry's recorded sha256 at the stand-in's real digest. The tests exercise
# resolution order, removal and the fetch policy — none of them depend on the
# *contents* of a dataset, only on a file existing whose digest matches the
# registry. On a developer machine every zip is present and nothing is
# substituted, so this changes nothing locally.
#
# 10-REVIEW.md WR-06: stand-ins are staged in a pytest tmp_path_factory
# directory and `_datasets_dir()` is redirected there — never written into the
# real `mmcli/example_datasets/` package tree. `.gitignore` hid the previous
# on-disk leftovers from `git status`, so any hard interrupt (Ctrl-C, a
# crashing collection, an OOM kill) that skipped the old teardown's `os.unlink`
# left up to nine fake zips permanently in the developer's package tree —
# silently changing `mmcli init --dataset` behaviour and flipping
# `_REAL_ZIPS_PRESENT` below, from then on, for every future test run. A
# pytest-managed tmp_path_factory directory needs no manual cleanup and is
# never mistaken for a real bundled dataset by out-of-process code.


def _ensure_dataset_zips_available(tmp_path_factory, monkeypatch):
    """Stage a complete set of registry zips (real files copied, missing ones
    synthesised as stand-ins) in a throwaway directory, and point
    `_datasets_dir()` at it — never at the real package directory."""
    import zipfile
    stage = tmp_path_factory.mktemp("bundled")
    for name, meta in DATASET_REGISTRY.items():
        src = os.path.join(_REAL_BUNDLED_DIR, meta["filename"])
        dst = os.path.join(str(stage), meta["filename"])
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            continue
        with zipfile.ZipFile(dst, "w") as z:
            z.writestr(f"{name}/README.txt", f"stand-in for {name}\n")
        monkeypatch.setitem(
            DATASET_REGISTRY[name], "sha256",
            hashlib.sha256(open(dst, "rb").read()).hexdigest(),
        )
        monkeypatch.setitem(
            DATASET_REGISTRY[name], "bytes", os.path.getsize(dst),
        )

    def _redirected_datasets_dir():
        # Preserve _datasets_dir()'s real MMCLI_DATASETS-first priority
        # (see mmcli/datasets.py) — only the *fallback* (bundled) directory
        # is redirected to the staged copy, not the env-var override, so
        # tests that set MMCLI_DATASETS to exercise the air-gap path still
        # see their own directory, not the stand-in stage.
        env = os.environ.get("MMCLI_DATASETS")
        if env and os.path.isdir(env):
            return env
        return str(stage)

    monkeypatch.setattr(datasets_mod, "_datasets_dir", _redirected_datasets_dir)
    return str(stage)


@pytest.fixture(autouse=True)
def dataset_zips_present(tmp_path_factory, monkeypatch):
    """Autouse so every test in this module sees a complete set of zips,
    staged outside the real package tree (10-REVIEW.md WR-06).

    Yields the staged directory path itself (not just `_datasets_dir()`'s
    return value), for tests that need "a valid digest-matching source zip
    for the currently active registry state" independent of whatever a test
    does to `_datasets_dir()` (e.g. the `hide_bundled` fixture, which
    overrides it to an empty directory — the staged zips still need to be
    reachable as a copy source in that case).
    """
    stage = _ensure_dataset_zips_available(tmp_path_factory, monkeypatch)
    yield stage


_REAL_ZIPS_PRESENT = all(
    os.path.exists(os.path.join(_REAL_BUNDLED_DIR, m["filename"]))
    for m in DATASET_REGISTRY.values()
)

# These two drive `mmcli init` in a SUBPROCESS, which re-imports the registry
# and therefore never sees the in-process sha256 override the stand-in fixture
# installs. They need a genuinely digest-matching zip on disk, which a checkout
# does not have. Skipped rather than left failing, and narrowly: every other
# test in this file runs in CI via the stand-in above.
_needs_real_zips = pytest.mark.skipif(
    not _REAL_ZIPS_PRESENT,
    reason="drives mmcli init in a subprocess; needs real dataset zips, which "
           "are gitignored and absent from a fresh checkout",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point XDG_CACHE_HOME at a throwaway directory for the duration of a
    test, so no test ever touches the developer's real ~/.cache, and ensure
    MMCLI_DATASETS is unset unless a test sets it explicitly."""
    cache_home = tmp_path / "cache_home"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    monkeypatch.delenv("MMCLI_DATASETS", raising=False)
    monkeypatch.delenv("MMCLI_AUTO_FETCH", raising=False)
    return str(cache_home)


@pytest.fixture
def hide_bundled(tmp_path, monkeypatch):
    """Point `_datasets_dir()` at an empty directory so no dataset ever
    resolves as bundled, regardless of whether 10-03 has physically
    unbundled the real zips yet. Every caller inside mmcli.datasets calls
    `_datasets_dir()` by its bare (module-global) name, so patching the
    module attribute redirects them all.
    """
    empty_dir = tmp_path / "empty_bundled"
    empty_dir.mkdir()
    monkeypatch.setattr(datasets_mod, "_datasets_dir", lambda: str(empty_dir))
    return str(empty_dir)


@pytest.fixture
def fake_local_only_entry(monkeypatch):
    """Register a temporary registry entry with no `ti_name` (bundled-only
    shape, like `generic_audio_classification`) whose file exists nowhere —
    used to exercise the `unavailable` state.
    """
    name = "_test_only_local_unavailable"
    DATASET_REGISTRY[name] = {
        "filename": "_test_only_local_unavailable.zip",
        "task_types": ["generic_timeseries_classification"],
        "module": "timeseries",
        "description": "Not a real dataset - test fixture only",
        "sha256": "0" * 64,
        "bytes": 100,
    }
    yield name
    DATASET_REGISTRY.pop(name, None)


def _forbid_download(monkeypatch):
    """Make any accidental real-download attempt fail the test loudly,
    instead of hanging or reaching the real network."""
    monkeypatch.setattr(
        datasets_mod, "_download_to_cache",
        lambda *a, **k: pytest.fail("must not attempt a network download"),
    )


def _install_fake_successful_download(monkeypatch):
    """Make `fetch_dataset()` succeed for any ti-fetchable dataset with zero
    network access, by copying the real bundled zip (byte-identical, so its
    digest already matches DATASET_REGISTRY) into the cache directory
    `_download_to_cache` would have written to. Same technique 10-02's suite
    uses in `test_force_flag_bypasses_fetch_dataset_cache_short_circuit`.
    """
    def _fake_download(url, cache_dir, filename, sha256, nbytes, name, *, on_event=None):
        src = os.path.join(_REAL_BUNDLED_DIR, filename)
        dest = os.path.join(cache_dir, filename)
        shutil.copyfile(src, dest)
        return dest

    monkeypatch.setattr(
        datasets_mod, "dataset_url", lambda n: "https://example.invalid/fake.zip"
    )
    monkeypatch.setattr(datasets_mod, "_download_to_cache", _fake_download)


def _run(monkeypatch, capsys, argv):
    """Invoke mmcli.cli.main() in-process with *argv* and return
    (exit_code, stdout, stderr). Every command this file exercises always
    ends in an explicit sys.exit(), so SystemExit is expected on every path.
    """
    monkeypatch.setattr(sys, "argv", ["mmcli"] + list(argv))
    try:
        cli_main()
        code = 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (1 if exc.code else 0)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ---------------------------------------------------------------------------
# Task 1 — `datasets list --format json` contract
# ---------------------------------------------------------------------------

class TestDatasetsListJson:
    def test_json_contract_all_ten_present_with_required_keys(self, monkeypatch, capsys):
        code, out, err = _run(monkeypatch, capsys, ["datasets", "list", "--format", "json"])
        assert code == 0, err

        import json
        payload = json.loads(out)
        records = payload["datasets"] if isinstance(payload, dict) else payload
        assert len(records) == 10

        for record in records:
            for key in ("name", "version", "state", "bytes", "cache_bytes"):
                assert key in record, (record.get("name"), key)
            assert record["state"] in ("bundled", "cached", "downloadable", "unavailable")
            assert isinstance(record["bytes"], int) and record["bytes"] > 0
            # cache_bytes (CONTEXT.md D-10, additive field, 10-09-PLAN.md):
            # independent of state, so it must be present and either null or
            # a non-negative int, regardless of what `state` says.
            assert record["cache_bytes"] is None or (
                isinstance(record["cache_bytes"], int) and record["cache_bytes"] >= 0
            )

    def test_descriptive_fields_present(self, monkeypatch, capsys):
        code, out, err = _run(monkeypatch, capsys, ["datasets", "list", "--format", "json"])
        assert code == 0, err
        import json
        records = json.loads(out)["datasets"]
        for record in records:
            assert "task_types" in record
            assert "module" in record
            assert "description" in record

    def test_generic_audio_reports_bundled(self, monkeypatch, capsys):
        # Real environment (nothing hidden): the locally-authored audio set
        # is always bundled and has no ti_name.
        code, out, err = _run(monkeypatch, capsys, ["datasets", "list", "--format", "json"])
        assert code == 0, err
        import json
        records = {r["name"]: r for r in json.loads(out)["datasets"]}
        assert records["generic_audio_classification"]["state"] == "bundled"
        assert records["generic_audio_classification"]["version"] is None

    def test_ti_dataset_with_empty_cache_reports_downloadable(
        self, hide_bundled, isolated_cache, monkeypatch, capsys
    ):
        _forbid_download(monkeypatch)
        code, out, err = _run(monkeypatch, capsys, ["datasets", "list", "--format", "json"])
        assert code == 0, err
        import json
        records = {r["name"]: r for r in json.loads(out)["datasets"]}
        assert records[_SMALL_TI_DATASET]["state"] == "downloadable"
        assert records[_SMALL_TI_DATASET]["version"] == DATASETS_DEFAULT_VERSION

    def test_mmcli_datasets_env_reports_bundled_and_no_network(
        self, tmp_path, monkeypatch, capsys
    ):
        meta = DATASET_REGISTRY["pir_detection"]
        src = os.path.join(datasets_mod._datasets_dir(), meta["filename"])
        env_dir = tmp_path / "air_gapped"
        env_dir.mkdir()
        shutil.copyfile(src, env_dir / meta["filename"])
        monkeypatch.setenv("MMCLI_DATASETS", str(env_dir))
        _forbid_download(monkeypatch)

        code, out, err = _run(monkeypatch, capsys, ["datasets", "list", "--format", "json"])
        assert code == 0, err
        import json
        records = {r["name"]: r for r in json.loads(out)["datasets"]}
        assert records["pir_detection"]["state"] == "bundled"

    def test_state_cached_when_resolved_from_cache(
        self, dataset_zips_present, hide_bundled, isolated_cache, monkeypatch, capsys
    ):
        meta = DATASET_REGISTRY[_SMALL_TI_DATASET]
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)
        shutil.copyfile(
            os.path.join(dataset_zips_present, meta["filename"]),
            os.path.join(cache_dir, meta["filename"]),
        )
        _forbid_download(monkeypatch)

        code, out, err = _run(monkeypatch, capsys, ["datasets", "list", "--format", "json"])
        assert code == 0, err
        import json
        records = {r["name"]: r for r in json.loads(out)["datasets"]}
        assert records[_SMALL_TI_DATASET]["state"] == "cached"

    def test_state_unavailable_for_local_only_entry_when_absent(
        self, hide_bundled, isolated_cache, fake_local_only_entry, monkeypatch, capsys
    ):
        _forbid_download(monkeypatch)
        code, out, err = _run(monkeypatch, capsys, ["datasets", "list", "--format", "json"])
        assert code == 0, err
        import json
        records = {r["name"]: r for r in json.loads(out)["datasets"]}
        assert records[fake_local_only_entry]["state"] == "unavailable"

    def test_cache_bytes_none_when_no_cache_entry(
        self, hide_bundled, isolated_cache, monkeypatch, capsys
    ):
        _forbid_download(monkeypatch)
        code, out, err = _run(monkeypatch, capsys, ["datasets", "list", "--format", "json"])
        assert code == 0, err
        import json
        records = {r["name"]: r for r in json.loads(out)["datasets"]}
        assert records[_SMALL_TI_DATASET]["cache_bytes"] is None

    def test_cache_bytes_reports_stale_entry_even_when_bundled_wins_resolution(
        self, isolated_cache, monkeypatch, capsys
    ):
        """CONTEXT.md D-10's core scenario: a dataset can be 'bundled' (the
        primary directory is NOT hidden here) while a stale cache entry from
        an earlier download still sits on disk, since the packaged copy
        wins resolution over the cache. `cache_bytes` must still report that
        entry's size, independent of `state` reporting 'bundled'.
        """
        meta = DATASET_REGISTRY[_SMALL_TI_DATASET]
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)
        cache_path = os.path.join(cache_dir, meta["filename"])
        shutil.copyfile(os.path.join(datasets_mod._datasets_dir(), meta["filename"]), cache_path)

        code, out, err = _run(monkeypatch, capsys, ["datasets", "list", "--format", "json"])
        assert code == 0, err
        import json
        records = {r["name"]: r for r in json.loads(out)["datasets"]}
        record = records[_SMALL_TI_DATASET]
        # Primary directory is not hidden in this test, so resolution
        # still finds the bundled copy first.
        assert record["state"] == "bundled"
        assert record["cache_bytes"] == os.path.getsize(cache_path)

    def test_list_text_format_default(self, monkeypatch, capsys):
        code, out, err = _run(monkeypatch, capsys, ["datasets", "list"])
        assert code == 0, err
        assert "fan_blade_fault" in out
        assert "dataset(s)" in out

    def test_list_filters_by_task(self, monkeypatch, capsys):
        code, out, err = _run(
            monkeypatch, capsys, ["datasets", "list", "--format", "json", "-t", "motor_fault"]
        )
        assert code == 0, err
        import json
        records = json.loads(out)["datasets"]
        assert [r["name"] for r in records] == ["fan_blade_fault"]

    def test_list_filters_by_module(self, monkeypatch, capsys):
        code, out, err = _run(
            monkeypatch, capsys, ["datasets", "list", "--format", "json", "-m", "vision"]
        )
        assert code == 0, err
        import json
        records = json.loads(out)["datasets"]
        assert [r["name"] for r in records] == ["mnist_image_classification"]


# ---------------------------------------------------------------------------
# Task 1 — `datasets pull`
# ---------------------------------------------------------------------------

class TestDatasetsPull:
    def test_pull_unknown_name_exits_nonzero_with_valid_names(self, monkeypatch, capsys):
        code, out, err = _run(monkeypatch, capsys, ["datasets", "pull", "totally_not_real"])
        assert code != 0
        assert "Unknown dataset" in err
        for name in DATASET_REGISTRY:
            assert name in err

    def test_pull_already_cached_issues_no_request_and_exits_zero(
        self, dataset_zips_present, hide_bundled, isolated_cache, monkeypatch, capsys
    ):
        meta = DATASET_REGISTRY[_SMALL_TI_DATASET]
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)
        shutil.copyfile(
            os.path.join(dataset_zips_present, meta["filename"]),
            os.path.join(cache_dir, meta["filename"]),
        )
        _forbid_download(monkeypatch)

        code, out, err = _run(monkeypatch, capsys, ["datasets", "pull", _SMALL_TI_DATASET])
        assert code == 0, err
        assert "available at" in out

    def test_pull_bundled_only_dataset_refuses_cleanly(self, monkeypatch, capsys):
        # generic_audio_classification has no ti_name — nothing to pull.
        code, out, err = _run(
            monkeypatch, capsys, ["datasets", "pull", "generic_audio_classification"]
        )
        assert code != 0
        assert "no upstream source" in err

    def test_pull_force_flag_forwarded_to_fetch_dataset(self, monkeypatch, capsys):
        # 10-11-PLAN.md Task 1: _handle_datasets_pull calls
        # fetch_dataset_detailed (not the fetch_dataset str-wrapper) so it can
        # see the outcome. Monkeypatch that function instead.
        calls = {}

        def fake_fetch_dataset_detailed(name, *, force=False, on_event=None):
            calls["name"] = name
            calls["force"] = force
            return datasets_mod.FetchResult("/fake/path", "forced-redownload")

        monkeypatch.setattr(
            datasets_mod, "fetch_dataset_detailed", fake_fetch_dataset_detailed
        )
        code, out, err = _run(
            monkeypatch, capsys, ["datasets", "pull", _SMALL_TI_DATASET, "--force"]
        )
        assert code == 0, err
        assert calls == {"name": _SMALL_TI_DATASET, "force": True}
        assert "available at" in out

    def test_pull_out_of_space_yields_legible_error_not_traceback(
        self, hide_bundled, isolated_cache, monkeypatch, capsys
    ):
        """10-09-PLAN.md Task 1: an ENOSPC from the download write loop must
        surface as a legible ERROR line naming the free-space problem, never
        an escaping Python traceback. Simulated with a fake opener (so no
        real network is touched) and a real temp file whose `.write` is
        monkeypatched to raise the actual OSError a full disk produces.
        """
        meta = DATASET_REGISTRY[_SMALL_TI_DATASET]

        class _FakeResponse:
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

            def read(self, n):
                # Always has more bytes to give; the write-loop's ENOSPC
                # fires on the very first chunk, so the exact content
                # doesn't matter.
                return b"x" * min(n, meta["bytes"])

        class _FakeOpener:
            def open(self, request, timeout=None):
                return _FakeResponse()

        monkeypatch.setattr(
            datasets_mod.urllib.request, "build_opener",
            lambda *a, **k: _FakeOpener(),
        )

        real_open = open

        def fake_open(path, mode="r", *args, **kwargs):
            f = real_open(path, mode, *args, **kwargs)
            if mode == "wb":
                def _enospc_write(_data):
                    raise OSError(errno.ENOSPC, "No space left on device")
                f.write = _enospc_write
            return f

        monkeypatch.setattr(datasets_mod, "open", fake_open, raising=False)

        code, out, err = _run(monkeypatch, capsys, ["datasets", "pull", _SMALL_TI_DATASET])
        assert code == 1, err
        assert "ERROR" in err
        assert "space" in err.lower()
        assert "Traceback" not in err
        # No partial/temp file left behind in the cache directory.
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)
        assert os.listdir(cache_dir) == []


# ---------------------------------------------------------------------------
# Task 1 — `datasets path`
# ---------------------------------------------------------------------------

class TestDatasetsPath:
    def test_path_unknown_name_exits_nonzero(self, monkeypatch, capsys):
        code, out, err = _run(monkeypatch, capsys, ["datasets", "path", "totally_not_real"])
        assert code != 0
        assert "Unknown dataset" in err

    def test_path_generic_audio_works_no_network_no_cache(self, monkeypatch, capsys):
        code, out, err = _run(
            monkeypatch, capsys, ["datasets", "path", "generic_audio_classification"]
        )
        assert code == 0, err
        printed_path = out.strip()
        assert os.path.isfile(printed_path)

    def test_path_unavailable_exits_nonzero_with_pull_hint(
        self, hide_bundled, isolated_cache, monkeypatch, capsys
    ):
        code, out, err = _run(monkeypatch, capsys, ["datasets", "path", _SMALL_TI_DATASET])
        assert code != 0
        assert f"mmcli datasets pull {_SMALL_TI_DATASET}" in err


# ---------------------------------------------------------------------------
# 10-09-PLAN.md Task 1 — `datasets remove`
# ---------------------------------------------------------------------------

class TestDatasetsRemove:
    def test_cache_entry_path_answers_where_not_where_it_resolves_from(
        self, isolated_cache
    ):
        """`cache_entry_path` always computes the version-scoped cache
        location, independent of whether a file exists there or anywhere
        else — the safety property `remove` depends on.
        """
        meta = DATASET_REGISTRY[_SMALL_TI_DATASET]
        expected = os.path.join(_cache_dir(DATASETS_DEFAULT_VERSION), meta["filename"])
        assert cache_entry_path(_SMALL_TI_DATASET) == expected

    def test_remove_unknown_name_exits_2_with_valid_names(self, monkeypatch, capsys):
        code, out, err = _run(monkeypatch, capsys, ["datasets", "remove", "totally_not_real"])
        assert code == 2
        assert "Unknown dataset" in err
        for name in DATASET_REGISTRY:
            assert name in err

    def test_remove_absent_cache_entry_is_idempotent_exit_0(
        self, hide_bundled, isolated_cache, monkeypatch, capsys
    ):
        code, out, err = _run(monkeypatch, capsys, ["datasets", "remove", _SMALL_TI_DATASET])
        assert code == 0, err
        assert "nothing to remove" in out
        assert err == ""

    def test_remove_present_cache_entry_deletes_it_and_reports_bytes_freed(
        self, dataset_zips_present, hide_bundled, isolated_cache, monkeypatch, capsys
    ):
        meta = DATASET_REGISTRY[_SMALL_TI_DATASET]
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)
        cache_path = os.path.join(cache_dir, meta["filename"])
        shutil.copyfile(os.path.join(dataset_zips_present, meta["filename"]), cache_path)
        expected_bytes = os.path.getsize(cache_path)

        code, out, err = _run(monkeypatch, capsys, ["datasets", "remove", _SMALL_TI_DATASET])
        assert code == 0, err
        assert "Removed" in out
        assert str(expected_bytes) in out
        assert not os.path.isfile(cache_path)

        # Afterwards the dataset reports itself as no longer local, and its
        # cache_bytes field (D-10) reverts to null.
        code, out, err = _run(monkeypatch, capsys, ["datasets", "list", "--format", "json"])
        assert code == 0, err
        import json
        record = {r["name"]: r for r in json.loads(out)["datasets"]}[_SMALL_TI_DATASET]
        assert record["state"] == "downloadable"
        assert record["cache_bytes"] is None

    def test_remove_never_touches_the_primary_bundled_directory(
        self, isolated_cache, monkeypatch, capsys
    ):
        """The guard test the threat model calls out (T-10-09-01 / Task 4
        step 9): with the dataset resolvable from the PRIMARY directory
        (bundled — not hidden here) and the cache empty, `remove` must
        report idempotent absence and must never touch the primary copy.
        `remove` operates only on `cache_entry_path`, never on whatever
        `_resolve_dataset_zip` returns.
        """
        meta = DATASET_REGISTRY[_SMALL_TI_DATASET]
        primary_path = os.path.join(datasets_mod._datasets_dir(), meta["filename"])
        assert os.path.isfile(primary_path)
        original_bytes = open(primary_path, "rb").read()

        code, out, err = _run(monkeypatch, capsys, ["datasets", "remove", _SMALL_TI_DATASET])
        assert code == 0, err

        assert os.path.isfile(primary_path)
        assert open(primary_path, "rb").read() == original_bytes

    def test_remove_under_mmcli_datasets_prints_note_and_does_not_refuse(
        self, dataset_zips_present, hide_bundled, isolated_cache, tmp_path, monkeypatch, capsys
    ):
        """Removal must not refuse under MMCLI_DATASETS (10-09-PLAN.md);
        instead it prints one informational line and proceeds, since the
        cache is not what resolves in that environment.
        """
        meta = DATASET_REGISTRY[_SMALL_TI_DATASET]
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)
        cache_path = os.path.join(cache_dir, meta["filename"])
        shutil.copyfile(os.path.join(dataset_zips_present, meta["filename"]), cache_path)

        env_dir = tmp_path / "air_gapped"
        env_dir.mkdir()
        monkeypatch.setenv("MMCLI_DATASETS", str(env_dir))

        code, out, err = _run(monkeypatch, capsys, ["datasets", "remove", _SMALL_TI_DATASET])
        assert code == 0, err
        assert "MMCLI_DATASETS" in out
        assert not os.path.isfile(cache_path)

    def test_remove_guard_refuses_if_target_escapes_cache_dir(
        self, hide_bundled, isolated_cache, monkeypatch, capsys, tmp_path
    ):
        """Directly exercises the re-assertion guard: if `cache_entry_path`
        were ever to compute a path outside `_cache_dir(version)` (e.g. a
        future refactor bug), `remove` must refuse rather than unlink it.
        """
        outside_dir = tmp_path / "definitely_not_the_cache"
        outside_dir.mkdir()
        meta = DATASET_REGISTRY[_SMALL_TI_DATASET]
        escaped_path = os.path.join(str(outside_dir), meta["filename"])
        with open(escaped_path, "wb") as f:
            f.write(b"not a real zip")

        monkeypatch.setattr(
            datasets_mod, "cache_entry_path", lambda name: escaped_path
        )

        code, out, err = _run(monkeypatch, capsys, ["datasets", "remove", _SMALL_TI_DATASET])
        assert code == 1, err
        assert "Refusing" in err
        assert os.path.isfile(escaped_path)

    def test_remove_unlink_oserror_reports_error_and_exits_1(
        self, dataset_zips_present, hide_bundled, isolated_cache, monkeypatch, capsys
    ):
        meta = DATASET_REGISTRY[_SMALL_TI_DATASET]
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)
        cache_path = os.path.join(cache_dir, meta["filename"])
        shutil.copyfile(os.path.join(dataset_zips_present, meta["filename"]), cache_path)

        def fake_unlink(path):
            raise OSError("permission denied (simulated)")

        monkeypatch.setattr(os, "unlink", fake_unlink)

        code, out, err = _run(monkeypatch, capsys, ["datasets", "remove", _SMALL_TI_DATASET])
        assert code == 1, err
        assert "ERROR" in err


# ---------------------------------------------------------------------------
# 10-REVIEW.md WR-11 — MMCLI_AUTO_FETCH normalisation and precedence
# ---------------------------------------------------------------------------

class TestResolveExplicitFetch:
    """`_resolve_explicit_fetch` was previously completely untested — the
    only reference in this file was an unrelated `monkeypatch.delenv`. Before
    WR-11's fix, only the literal strings "1"/"0" were recognised; anything
    else (including "false", "no", "off", or "0 " with trailing whitespace)
    silently fell through to the TTY-check default — the exact opposite of
    a user's stated intent to disable fetching, with no warning.
    """

    def _args(self, fetch=None):
        import argparse
        return argparse.Namespace(fetch=fetch)

    def test_env_absent_returns_none(self, monkeypatch):
        monkeypatch.delenv("MMCLI_AUTO_FETCH", raising=False)
        assert cli_mod._resolve_explicit_fetch(self._args()) is None

    @pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "on", " 1 "])
    def test_recognised_true_values(self, monkeypatch, value):
        monkeypatch.setenv("MMCLI_AUTO_FETCH", value)
        assert cli_mod._resolve_explicit_fetch(self._args()) is True

    @pytest.mark.parametrize("value", ["0", "false", "False", "FALSE", "no", "off", " 0 "])
    def test_recognised_false_values(self, monkeypatch, value):
        monkeypatch.setenv("MMCLI_AUTO_FETCH", value)
        assert cli_mod._resolve_explicit_fetch(self._args()) is False

    @pytest.mark.parametrize("value", ["maybe", "disable", "2", ""])
    def test_unrecognised_value_falls_back_to_none_with_warning(
        self, monkeypatch, capsys, value
    ):
        monkeypatch.setenv("MMCLI_AUTO_FETCH", value)
        result = cli_mod._resolve_explicit_fetch(self._args())
        assert result is None
        err = capsys.readouterr().err
        assert "WARNING" in err
        assert "MMCLI_AUTO_FETCH" in err

    def test_cli_flag_true_beats_env_false(self, monkeypatch):
        monkeypatch.setenv("MMCLI_AUTO_FETCH", "0")
        assert cli_mod._resolve_explicit_fetch(self._args(fetch=True)) is True

    def test_cli_flag_false_beats_env_true(self, monkeypatch):
        monkeypatch.setenv("MMCLI_AUTO_FETCH", "1")
        assert cli_mod._resolve_explicit_fetch(self._args(fetch=False)) is False

    def test_neither_cli_flag_nor_env_returns_none(self, monkeypatch):
        monkeypatch.delenv("MMCLI_AUTO_FETCH", raising=False)
        assert cli_mod._resolve_explicit_fetch(self._args(fetch=None)) is None


# ---------------------------------------------------------------------------
# Task 2 — D-5 auto-fetch policy for `init --dataset`
# ---------------------------------------------------------------------------

class TestInitAutoFetchPolicy:
    """The seven behaviour cases from 10-06-PLAN.md Task 2, tested by
    injecting the TTY predicate (monkeypatching `stderr_is_tty`) rather than
    allocating a pty, per the plan's own instruction.
    """

    def _init_argv(self, project_dir, extra=()):
        return [
            "init",
            "--task", "generic_timeseries_forecasting",
            "--dataset", _SMALL_TI_DATASET,
            "--project", str(project_dir),
            *extra,
        ]

    @_needs_real_zips
    def test_tty_missing_no_env_fetches_and_creates_project(
        self, hide_bundled, isolated_cache, monkeypatch, capsys, tmp_path
    ):
        monkeypatch.setattr(datasets_mod, "stderr_is_tty", lambda: True)
        _install_fake_successful_download(monkeypatch)

        project = tmp_path / "proj_tty"
        code, out, err = _run(monkeypatch, capsys, self._init_argv(project))
        assert code == 0, err
        assert os.path.isdir(project / "dataset")

    def test_non_tty_missing_refuses_no_request_message_has_pull_cmd_and_size(
        self, hide_bundled, isolated_cache, monkeypatch, capsys, tmp_path
    ):
        monkeypatch.setattr(datasets_mod, "stderr_is_tty", lambda: False)
        _forbid_download(monkeypatch)

        project = tmp_path / "proj_nontty"
        code, out, err = _run(monkeypatch, capsys, self._init_argv(project))
        assert code != 0
        assert f"mmcli datasets pull {_SMALL_TI_DATASET}" in err
        assert "MB" in err
        assert not project.exists()

    def test_no_fetch_with_tty_refuses_no_request(
        self, hide_bundled, isolated_cache, monkeypatch, capsys, tmp_path
    ):
        monkeypatch.setattr(datasets_mod, "stderr_is_tty", lambda: True)
        _forbid_download(monkeypatch)

        project = tmp_path / "proj_no_fetch"
        code, out, err = _run(
            monkeypatch, capsys, self._init_argv(project, extra=["--no-fetch"])
        )
        assert code != 0
        assert f"mmcli datasets pull {_SMALL_TI_DATASET}" in err
        assert not project.exists()

    @_needs_real_zips
    def test_fetch_without_tty_fetches(
        self, hide_bundled, isolated_cache, monkeypatch, capsys, tmp_path
    ):
        monkeypatch.setattr(datasets_mod, "stderr_is_tty", lambda: False)
        _install_fake_successful_download(monkeypatch)

        project = tmp_path / "proj_fetch_flag"
        code, out, err = _run(
            monkeypatch, capsys, self._init_argv(project, extra=["--fetch"])
        )
        assert code == 0, err
        assert os.path.isdir(project / "dataset")

    def test_mmcli_datasets_set_absent_refuses_even_with_fetch(
        self, tmp_path, monkeypatch, capsys
    ):
        env_dir = tmp_path / "air_gapped_empty"
        env_dir.mkdir()
        monkeypatch.setenv("MMCLI_DATASETS", str(env_dir))
        monkeypatch.setattr(datasets_mod, "stderr_is_tty", lambda: True)
        _forbid_download(monkeypatch)

        project = tmp_path / "proj_env_absent"
        code, out, err = _run(
            monkeypatch, capsys, self._init_argv(project, extra=["--fetch"])
        )
        assert code != 0
        assert "MMCLI_DATASETS" in err
        assert not project.exists()

    def test_dataset_already_present_no_policy_involvement(
        self, monkeypatch, capsys, tmp_path
    ):
        # Real environment: nothing hidden, so the bundled zip resolves
        # normally. The policy helper must never even be consulted.
        import mmcli.cli as cli_mod

        def _fail_if_called(name, args):
            pytest.fail("policy must not be consulted when already present")

        monkeypatch.setattr(cli_mod, "_apply_init_fetch_policy", _fail_if_called)

        project = tmp_path / "proj_already_present"
        code, out, err = _run(monkeypatch, capsys, self._init_argv(project))
        assert code == 0, err
        assert os.path.isdir(project / "dataset")

    def test_fetch_and_no_fetch_together_argparse_rejects(
        self, monkeypatch, capsys, tmp_path
    ):
        project = tmp_path / "proj_conflict"
        code, out, err = _run(
            monkeypatch, capsys,
            self._init_argv(project, extra=["--fetch", "--no-fetch"]),
        )
        assert code == 2
        assert not project.exists()

    def test_incompatible_task_rejected_before_any_download_attempt(
        self, hide_bundled, isolated_cache, monkeypatch, capsys, tmp_path
    ):
        # 10-REVIEW.md WR-01: a pure argument error (incompatible --task) must
        # be caught before fetch_dataset() is ever invoked, not after tens of
        # MB have already been downloaded.
        _forbid_download(monkeypatch)

        project = tmp_path / "proj_bad_task"
        code, out, err = _run(
            monkeypatch, capsys,
            [
                "init",
                "--task", "mnist_image_classification",  # incompatible with _SMALL_TI_DATASET
                "--dataset", _SMALL_TI_DATASET,
                "--project", str(project),
            ],
        )
        assert code == 2
        assert "not compatible with task" in err
        assert not project.exists()

    def test_existing_project_dir_rejected_before_any_download_attempt(
        self, hide_bundled, isolated_cache, monkeypatch, capsys, tmp_path
    ):
        # 10-REVIEW.md WR-01: "project directory already exists" must be
        # caught before fetch_dataset() is ever invoked.
        _forbid_download(monkeypatch)

        project = tmp_path / "proj_exists"
        project.mkdir()

        code, out, err = _run(monkeypatch, capsys, self._init_argv(project))
        assert code == 2
        assert "already exists" in err

    def test_project_path_traversal_rejected_before_any_download_attempt(
        self, hide_bundled, isolated_cache, monkeypatch, capsys
    ):
        # 10-REVIEW.md WR-12: _validate_args() (which applies _is_safe_path
        # to --project for every other command) is never reached for `init`
        # — this branch exits before it. `init` is the one command in the
        # phase that creates directories, so it must apply the same guard
        # itself rather than silently accepting a relative traversal
        # sequence that `extract_dataset()` would then os.makedirs().
        #
        # The target is deliberately NOT under /tmp. `_is_safe_path` permits
        # the OS temp dir by design, and on Linux `tempfile.gettempdir()` IS
        # "/tmp" — so a /tmp target is *correctly* accepted there and the
        # guard never fires. macOS hides that: its gettempdir() is a private
        # /var/folders/.../T, so /tmp looks "outside" and the test passed
        # locally while failing on CI (exit 1 from the auto-fetch policy,
        # reached because the guard had allowed the path through). Use a
        # location outside both cwd and the temp dir on every platform —
        # same rationale as test_regression.py's /etc/passwd case.
        _forbid_download(monkeypatch)

        code, out, err = _run(
            monkeypatch, capsys,
            [
                "init",
                "--task", "generic_timeseries_forecasting",
                "--dataset", _SMALL_TI_DATASET,
                "--project", "../../../../../../etc/mmcli_wr12_traversal_test",
            ],
        )
        assert code == 2
        assert "unsafe path traversal" in err
        assert not os.path.exists("/etc/mmcli_wr12_traversal_test")


class TestCacheInspectionHasNoSideEffects:
    """Asking where a cache entry would be must not create anything.

    Found by CodeRabbit review of 10-09 Task 1: `cache_entry_path` went through
    `_cache_dir`, which calls `os.makedirs`. Every `datasets list` therefore
    created `<cache-home>/mmcli/datasets/<version>/` as a side effect of being
    asked a read-only question, and would raise on an unwritable cache home
    even when no download had been requested.
    """

    def test_cache_entry_path_does_not_create_the_cache_directory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        from mmcli.datasets import cache_entry_path

        path = cache_entry_path("arc_fault_classification")
        assert str(tmp_path) in path
        assert not (tmp_path / "mmcli").exists(), (
            "cache_entry_path created the cache directory; a path query must "
            "not touch the filesystem"
        )

    def test_cache_entry_size_does_not_create_the_cache_directory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        from mmcli.datasets import cache_entry_size

        assert cache_entry_size("arc_fault_classification") is None
        assert not (tmp_path / "mmcli").exists(), (
            "cache_entry_size created the cache directory while reporting no entry"
        )

    def test_datasets_list_json_does_not_create_the_cache_directory(self, tmp_path):
        """The user-visible consequence: a pure listing wrote to disk.

        Runs the REAL CLI in a subprocess rather than calling the helpers
        directly. An earlier version of this test called cache_entry_size() in a
        loop and passed while the shipped binary still created the directory —
        because `datasets list` also goes through _resolve_dataset_zip(), which
        had its own _cache_dir() call. Simulating the code path is not the same
        as exercising it.
        """
        env = dict(os.environ, XDG_CACHE_HOME=str(tmp_path))
        env.pop("MMCLI_DATASETS", None)
        proc = subprocess.run(
            [sys.executable, "-m", "mmcli", "datasets", "list", "--format", "json"],
            capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
        )
        assert proc.returncode == 0, proc.stderr
        assert not (tmp_path / "mmcli").exists(), (
            "`mmcli datasets list` created the cache directory; a read-only "
            "listing must not write to the filesystem"
        )

    def test_datasets_path_does_not_create_the_cache_directory(self, tmp_path):
        """`datasets path` is also a pure query and must not write either."""
        env = dict(os.environ, XDG_CACHE_HOME=str(tmp_path))
        env.pop("MMCLI_DATASETS", None)
        proc = subprocess.run(
            [sys.executable, "-m", "mmcli", "datasets", "path", "fan_blade_fault"],
            capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
        )
        # 10-REVIEW.md WR-09: proc was never inspected here — if the CLI
        # failed to start at all (import error, missing __main__, broken
        # argparse wiring), nothing would be created and the test would pass
        # vacuously. 0 = found locally, 1 = "not available locally" (a real,
        # expected outcome depending on whether this dev checkout has the
        # real dataset zips present) — anything else, or a traceback in
        # stderr, means the CLI itself broke.
        assert proc.returncode in (0, 1), proc.stderr
        assert "Traceback" not in proc.stderr, proc.stderr
        assert not (tmp_path / "mmcli").exists(), (
            "`mmcli datasets path` created the cache directory"
        )

    def test_cache_dir_still_creates_when_downloading(self, tmp_path, monkeypatch):
        """The download flow legitimately needs the directory to exist."""
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        from mmcli.datasets import _cache_dir, _cache_dir_path, DATASETS_DEFAULT_VERSION

        pure = _cache_dir_path(DATASETS_DEFAULT_VERSION)
        assert not os.path.exists(pure)
        created = _cache_dir(DATASETS_DEFAULT_VERSION)
        assert created == pure
        assert os.path.isdir(created), "_cache_dir must still create for the download flow"

    def test_resolution_reaching_the_cache_branch_creates_nothing(self, tmp_path, monkeypatch):
        """Directly exercise step 3 of _resolve_dataset_zip.

        The subprocess tests above cannot reach this branch: in a source tree all
        ten zips sit in the package directory, so resolution returns at step 1.
        Only an unbundled install (the shipped binary) falls through to the
        cache — which is exactly where the shipped binary was still creating the
        directory after the first fix. Simulate that by pointing the primary
        directory at an empty one.
        """
        empty_primary = tmp_path / "primary"
        empty_primary.mkdir()
        cache_home = tmp_path / "cache"
        monkeypatch.delenv("MMCLI_DATASETS", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
        monkeypatch.setattr(datasets_mod, "_datasets_dir", lambda: str(empty_primary))

        assert datasets_mod._resolve_dataset_zip("fan_blade_fault") is None
        assert not cache_home.exists(), (
            "_resolve_dataset_zip created the cache directory while answering "
            "a read-only resolution question"
        )


# ---------------------------------------------------------------------------
# 10-11-PLAN.md Task 1 — the integrity-repair fix, proven through the REAL
# CLI entry point (10-UAT.md Gap 1)
# ---------------------------------------------------------------------------

@_needs_real_zips
class TestDatasetsPullIntegrityRepairSubprocess:
    """Drives the real CLI entry point in a subprocess — `[sys.executable,
    "-m", "mmcli", "datasets", "pull", ...]`, following the pattern at
    `TestCacheInspectionHasNoSideEffects` above — rather than the in-process
    `_run` helper. This is the test that proves the fix reaches a user: the
    in-process tests only prove the handler function works.

    Fetches a real, small dataset (`generic_timeseries_forecasting`, ~71 KB)
    from the real GitHub release mirror. There is no way to redirect the
    downloader to a local test server from inside a subprocess — a fresh
    process re-imports `mmcli.datasets` with no in-process monkeypatching
    applied — so this genuinely exercises the network path, unlike every
    other test in this file. Skipped (via `_needs_real_zips`) on a checkout
    without the real dataset zips, same as the other subprocess tests that
    need a genuinely digest-matching source.
    """

    def test_corrupted_cache_entry_prints_warning_and_repaired_line_exit_0(
        self, tmp_path
    ):
        name = "generic_timeseries_forecasting"
        meta = DATASET_REGISTRY[name]

        env = dict(os.environ, XDG_CACHE_HOME=str(tmp_path))
        env.pop("MMCLI_DATASETS", None)
        env.pop("MMCLI_AUTO_FETCH", None)

        cache_dir = tmp_path / "mmcli" / "datasets" / DATASETS_DEFAULT_VERSION
        cache_dir.mkdir(parents=True)
        corrupted_path = cache_dir / meta["filename"]
        corrupted_path.write_bytes(
            b"deliberately corrupted cache entry, does not match the "
            b"recorded sha256 at all"
        )
        corrupted_digest = hashlib.sha256(corrupted_path.read_bytes()).hexdigest()

        proc = subprocess.run(
            [sys.executable, "-m", "mmcli", "datasets", "pull", name],
            capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
            timeout=60,
        )

        assert proc.returncode == 0, (
            f"an integrity repair must still exit 0 — stdout={proc.stdout!r} "
            f"stderr={proc.stderr!r}"
        )
        assert "WARNING" in proc.stderr, proc.stderr
        assert meta["sha256"] in proc.stderr, "expected digest must be named"
        assert corrupted_digest in proc.stderr, "actual (bad) digest must be named"
        assert "REPAIRED" in proc.stdout, proc.stdout
        assert "available at:" in proc.stdout, proc.stdout
        assert hashlib.sha256(corrupted_path.read_bytes()).hexdigest() == meta["sha256"], (
            "the repaired file on disk must match the registry digest again"
        )

    def test_clean_cache_hit_line_is_visibly_different_from_repair(self, tmp_path):
        """A clean cache hit must print a distinctly different line from a
        repair — the four outcomes must be mutually distinguishable from
        stdout alone (10-11-PLAN.md success criteria)."""
        name = "generic_timeseries_forecasting"
        meta = DATASET_REGISTRY[name]

        env = dict(os.environ, XDG_CACHE_HOME=str(tmp_path))
        env.pop("MMCLI_DATASETS", None)
        env.pop("MMCLI_AUTO_FETCH", None)

        cache_dir = tmp_path / "mmcli" / "datasets" / DATASETS_DEFAULT_VERSION
        cache_dir.mkdir(parents=True)
        good_path = cache_dir / meta["filename"]
        shutil.copyfile(
            os.path.join(_REAL_BUNDLED_DIR, meta["filename"]), good_path
        )
        assert hashlib.sha256(good_path.read_bytes()).hexdigest() == meta["sha256"]

        proc = subprocess.run(
            [sys.executable, "-m", "mmcli", "datasets", "pull", name],
            capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
            timeout=60,
        )

        assert proc.returncode == 0, proc.stderr
        assert proc.stderr == "", (
            "a clean cache hit must not print any WARNING: " + proc.stderr
        )
        assert "already cached" in proc.stdout
        assert "REPAIRED" not in proc.stdout
        assert "available at:" in proc.stdout


# ---------------------------------------------------------------------------
# 10-11-PLAN.md Task 2 — the opt-in --progress-json event stream, driven
# through the real CLI entry point (D-5 is LOCKED: see 10-CONTEXT.md)
# ---------------------------------------------------------------------------

@_needs_real_zips
class TestProgressJsonCli:
    """Subprocess-level tests, same pattern as
    TestDatasetsPullIntegrityRepairSubprocess above. All three use a planted,
    already-verified cache entry (the cache-hit path) so no real network
    request is made — the point here is the JSON stream and its absence
    without the flag, not the transfer itself (that path is covered by
    TestProgressEvents in test_datasets_download.py, which redirects the
    real downloader to a local http_server fixture).
    """

    def _env_with_cached_entry(self, tmp_path, name):
        meta = DATASET_REGISTRY[name]
        env = dict(os.environ, XDG_CACHE_HOME=str(tmp_path))
        env.pop("MMCLI_DATASETS", None)
        env.pop("MMCLI_AUTO_FETCH", None)
        cache_dir = tmp_path / "mmcli" / "datasets" / DATASETS_DEFAULT_VERSION
        cache_dir.mkdir(parents=True)
        good_path = cache_dir / meta["filename"]
        shutil.copyfile(os.path.join(_REAL_BUNDLED_DIR, meta["filename"]), good_path)
        assert hashlib.sha256(good_path.read_bytes()).hexdigest() == meta["sha256"]
        return env

    def test_progress_json_on_cache_hit_emits_only_valid_ndjson_result(self, tmp_path):
        import json as _json
        name = "generic_timeseries_forecasting"
        meta = DATASET_REGISTRY[name]
        env = self._env_with_cached_entry(tmp_path, name)

        proc = subprocess.run(
            [sys.executable, "-m", "mmcli", "datasets", "pull",
             "--progress-json", name],
            capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
            timeout=60,
        )

        assert proc.returncode == 0, proc.stderr
        lines = [ln for ln in proc.stderr.splitlines() if ln.strip()]
        assert lines, "expected at least the result event on stderr"
        events = [_json.loads(ln) for ln in lines]  # every line must parse
        assert all(e["v"] == 1 for e in events)
        results = [e for e in events if e["event"] == "result"]
        assert len(results) == 1
        assert results[0]["outcome"] == "cache-hit"
        assert results[0]["dataset"] == name
        assert not any(e["event"] == "start" for e in events)
        assert not any(e["event"] == "progress" for e in events)
        # available-at line still prints on stdout, unaffected by the flag.
        assert "available at:" in proc.stdout

    def test_without_flag_emits_no_json_on_stderr(self, tmp_path):
        """D-5 regression guard: fails if mmcli ever becomes chattier by
        default. Same input as the previous test, minus --progress-json."""
        import json as _json
        name = "generic_timeseries_forecasting"
        env = self._env_with_cached_entry(tmp_path, name)

        proc = subprocess.run(
            [sys.executable, "-m", "mmcli", "datasets", "pull", name],
            capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
            timeout=60,
        )

        assert proc.returncode == 0, proc.stderr
        for ln in proc.stderr.splitlines():
            if not ln.strip():
                continue
            with pytest.raises(_json.JSONDecodeError):
                _json.loads(ln)
        assert "available at:" in proc.stdout

    def test_progress_json_flag_not_recognized_by_init(self, tmp_path):
        """10-CONTEXT.md D-5 is LOCKED: --progress-json lands on `datasets
        pull` only. `init --dataset` must not accept it at all — proving
        this at the argparse level is stronger than proving init's TTY
        refusal message is merely unchanged, since it shows the flag has no
        way to reach the D-5 decision path in the first place."""
        env = dict(os.environ, XDG_CACHE_HOME=str(tmp_path))
        env.pop("MMCLI_DATASETS", None)
        project = tmp_path / "proj"

        proc = subprocess.run(
            [sys.executable, "-m", "mmcli", "init",
             "--task", "generic_timeseries_forecasting",
             "--dataset", "generic_timeseries_forecasting",
             "--project", str(project),
             "--progress-json"],
            capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
            timeout=30,
        )

        assert proc.returncode != 0
        assert "unrecognized argument" in proc.stderr.lower(), proc.stderr
        assert not project.exists()


class TestAnnotationsDirIsNotRequired:
    """dataset/annotations/ is an OUTPUT of modelmaker, never an input.

    mmcli/cli.py used to reject any project whose dataset/ lacked annotations/.
    That inverted modelmaker's contract: when split-list files are absent it calls
    remove_if_exists(annotations_dir) then makedirs() and generates
    instances_{train,val,test}_list.txt itself. The check rejected datasets this
    project ships (anomaly detection's classes/Normal + classes/Anomaly, MNIST,
    audio) and blocked 16 of 75 model x task combinations — two of which train
    successfully once it is lifted. See .planning/FINDINGS-training-matrix.md F-1.

    These drive the REAL CLI in a subprocess. Asserting on _validate_args() alone
    would not prove the shipped command accepts such a project.
    """

    def _project_without_annotations(self, tmp_path, data_subdir="classes"):
        proj = tmp_path / "proj"
        (proj / "dataset" / data_subdir / "class_a").mkdir(parents=True)
        (proj / "dataset" / data_subdir / "class_a" / "sample.csv").write_text("1,2,3\n")
        return proj

    @pytest.mark.parametrize("data_subdir", ["classes", "files", "images"])
    def test_train_accepts_project_without_annotations(self, tmp_path, data_subdir):
        """No 'missing annotations' error for any of the three data layouts."""
        proj = self._project_without_annotations(tmp_path, data_subdir)
        proc = subprocess.run(
            [sys.executable, "-m", "mmcli", "--dry-run", "train",
             "-m", "timeseries", "-t", "generic_timeseries_classification",
             "-d", "F28P55", "-n", "CLS_100_NPU", "-i", str(proj)],
            capture_output=True, text=True,
            env=dict(os.environ, MMCLI_PYTHON=os.environ.get("MMCLI_PYTHON", sys.executable)),
        )
        combined = proc.stdout + proc.stderr
        assert "annotations" not in combined.lower(), (
            "mmcli rejected a project for missing dataset/annotations/, which "
            f"modelmaker generates itself. Output:\n{combined[:600]}"
        )

    def test_missing_data_subdir_is_still_rejected(self, tmp_path):
        """The sibling check is legitimate and must NOT be removed with it."""
        proj = tmp_path / "proj"
        (proj / "dataset").mkdir(parents=True)  # dataset/ exists but is empty
        proc = subprocess.run(
            [sys.executable, "-m", "mmcli", "--dry-run", "train",
             "-m", "timeseries", "-t", "generic_timeseries_classification",
             "-d", "F28P55", "-n", "CLS_100_NPU", "-i", str(proj)],
            capture_output=True, text=True,
            env=dict(os.environ, MMCLI_PYTHON=os.environ.get("MMCLI_PYTHON", sys.executable)),
        )
        combined = proc.stdout + proc.stderr
        assert proc.returncode != 0, "a dataset/ with no data subdirectory must be rejected"
        assert "data subdirectory" in combined.lower(), (
            f"expected the data-subdirectory error, got:\n{combined[:600]}"
        )
