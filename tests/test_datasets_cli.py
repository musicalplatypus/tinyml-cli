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
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

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
    def _fake_download(url, cache_dir, filename, sha256, nbytes, name):
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
        src = os.path.join(_REAL_BUNDLED_DIR, meta["filename"])
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
        self, hide_bundled, isolated_cache, monkeypatch, capsys
    ):
        meta = DATASET_REGISTRY[_SMALL_TI_DATASET]
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)
        shutil.copyfile(
            os.path.join(_REAL_BUNDLED_DIR, meta["filename"]),
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
        shutil.copyfile(os.path.join(_REAL_BUNDLED_DIR, meta["filename"]), cache_path)

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
        self, hide_bundled, isolated_cache, monkeypatch, capsys
    ):
        meta = DATASET_REGISTRY[_SMALL_TI_DATASET]
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)
        shutil.copyfile(
            os.path.join(_REAL_BUNDLED_DIR, meta["filename"]),
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
        calls = {}

        def fake_fetch_dataset(name, *, force=False):
            calls["name"] = name
            calls["force"] = force
            return "/fake/path"

        monkeypatch.setattr(datasets_mod, "fetch_dataset", fake_fetch_dataset)
        code, out, err = _run(
            monkeypatch, capsys, ["datasets", "pull", _SMALL_TI_DATASET, "--force"]
        )
        assert code == 0, err
        assert calls == {"name": _SMALL_TI_DATASET, "force": True}

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
        self, hide_bundled, isolated_cache, monkeypatch, capsys
    ):
        meta = DATASET_REGISTRY[_SMALL_TI_DATASET]
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)
        cache_path = os.path.join(cache_dir, meta["filename"])
        shutil.copyfile(os.path.join(_REAL_BUNDLED_DIR, meta["filename"]), cache_path)
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
        primary_path = os.path.join(_REAL_BUNDLED_DIR, meta["filename"])
        assert os.path.isfile(primary_path)
        original_bytes = open(primary_path, "rb").read()

        code, out, err = _run(monkeypatch, capsys, ["datasets", "remove", _SMALL_TI_DATASET])
        assert code == 0, err

        assert os.path.isfile(primary_path)
        assert open(primary_path, "rb").read() == original_bytes

    def test_remove_under_mmcli_datasets_prints_note_and_does_not_refuse(
        self, hide_bundled, isolated_cache, tmp_path, monkeypatch, capsys
    ):
        """Removal must not refuse under MMCLI_DATASETS (10-09-PLAN.md);
        instead it prints one informational line and proceeds, since the
        cache is not what resolves in that environment.
        """
        meta = DATASET_REGISTRY[_SMALL_TI_DATASET]
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)
        cache_path = os.path.join(cache_dir, meta["filename"])
        shutil.copyfile(os.path.join(_REAL_BUNDLED_DIR, meta["filename"]), cache_path)

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
        self, hide_bundled, isolated_cache, monkeypatch, capsys
    ):
        meta = DATASET_REGISTRY[_SMALL_TI_DATASET]
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)
        cache_path = os.path.join(cache_dir, meta["filename"])
        shutil.copyfile(os.path.join(_REAL_BUNDLED_DIR, meta["filename"]), cache_path)

        def fake_unlink(path):
            raise OSError("permission denied (simulated)")

        monkeypatch.setattr(os, "unlink", fake_unlink)

        code, out, err = _run(monkeypatch, capsys, ["datasets", "remove", _SMALL_TI_DATASET])
        assert code == 1, err
        assert "ERROR" in err


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
        subprocess.run(
            [sys.executable, "-m", "mmcli", "datasets", "path", "fan_blade_fault"],
            capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
        )
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
