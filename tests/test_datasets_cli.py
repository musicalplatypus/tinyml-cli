"""Tests for the `mmcli datasets` subcommand (Phase 10 Plan 06, Task 1):
`list --format json`'s cross-repo JSON contract, `pull`, and `path`.

(Task 2 — the D-5 auto-fetch policy gating `init --dataset` — adds its own
test class to this file in the next commit.)

Nothing here reimplements download, verification, or cache-resolution
logic — every network- or cache-related behaviour is exercised through the
real `mmcli.datasets` functions (10-02), with `_download_to_cache`
monkeypatched to fail the test if it is ever called, proving no test
contacts the real network.

Datasets are still bundled at this point in the phase (10-03 unbundles them
next), so tests that need a `downloadable`/`cached`/`unavailable` state must
not rely on that — they hide the bundled copy via the `hide_bundled` fixture
below, which works regardless of whether 10-03 has landed.
"""

import os
import shutil
import sys

import pytest

import mmcli.datasets as datasets_mod
from mmcli.cli import main as cli_main
from mmcli.datasets import (
    DATASET_REGISTRY,
    DATASETS_DEFAULT_VERSION,
    _cache_dir,
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
            for key in ("name", "version", "state", "bytes"):
                assert key in record, (record.get("name"), key)
            assert record["state"] in ("bundled", "cached", "downloadable", "unavailable")
            assert isinstance(record["bytes"], int) and record["bytes"] > 0

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
