"""Tests for the on-demand dataset fetch mechanism in mmcli/datasets.py.

Task 1 scope: registry invariants (ti_name/sha256/bytes) and the
version-pathed TI URL builder (`dataset_url`).
Task 2 scope (this commit): version-scoped cache (`_cache_dir`) and the
`_resolve_dataset_zip` read-resolution order (MMCLI_DATASETS -> bundled ->
cache), plus a zip-slip regression test for `extract_dataset` (T-10-02-06).
Task 3 (fetch_dataset/_download_to_cache against a local http.server) lands
in a later commit of this same plan.
"""

import hashlib
import os

import pytest

import mmcli.datasets as datasets
from mmcli.datasets import (
    DATASET_REGISTRY,
    DATASETS_DEFAULT_VERSION,
    _cache_dir,
    _resolve_dataset_zip,
    _validate_registry,
    dataset_url,
    extract_dataset,
)


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point XDG_CACHE_HOME at a throwaway directory for the duration of a
    test, so no test ever touches the developer's real ~/.cache."""
    cache_home = tmp_path / "cache_home"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    monkeypatch.delenv("MMCLI_DATASETS", raising=False)
    return str(cache_home)


@pytest.fixture
def fake_ti_entry(monkeypatch):
    """Register a temporary DATASET_REGISTRY entry that behaves like a TI
    dataset (ti_name/sha256/bytes) but whose filename never exists in the
    bundled example_datasets/ directory — needed to exercise the cache-only
    and download-only resolution paths, since all nine real TI datasets are
    still bundled at this point in the phase (10-03 unbundles them).
    """
    body = b"fake dataset payload for phase 10 plan 02 tests\n" * 100
    sha256 = hashlib.sha256(body).hexdigest()
    name = "_test_only_fake_dataset"
    DATASET_REGISTRY[name] = {
        "filename": "_test_only_fake_dataset.zip",
        "task_types": ["generic_timeseries_classification"],
        "module": "timeseries",
        "description": "Not a real dataset - test fixture only",
        "ti_name": "_test_only_fake_dataset.zip",
        "sha256": sha256,
        "bytes": len(body),
    }
    yield {"name": name, "body": body, "sha256": sha256}
    DATASET_REGISTRY.pop(name, None)

class TestRegistryInvariants:
    def test_nine_entries_have_ti_name(self):
        with_ti_name = [n for n, m in DATASET_REGISTRY.items() if m.get("ti_name")]
        assert len(with_ti_name) == 9
        assert "generic_audio_classification" not in with_ti_name

    def test_every_ti_name_entry_has_valid_sha256_and_bytes(self):
        for name, meta in DATASET_REGISTRY.items():
            if not meta.get("ti_name"):
                continue
            sha256 = meta.get("sha256", "")
            assert len(sha256) == 64, name
            assert all(c in "0123456789abcdef" for c in sha256.lower()), name
            assert isinstance(meta.get("bytes"), int) and meta["bytes"] > 0, name

    def test_generic_audio_has_digest_but_no_ti_name(self):
        meta = DATASET_REGISTRY["generic_audio_classification"]
        assert meta.get("ti_name") is None
        assert len(meta["sha256"]) == 64
        assert meta["bytes"] == 18371

    def test_fan_blade_fault_measured_values(self):
        meta = DATASET_REGISTRY["fan_blade_fault"]
        assert meta["ti_name"] == "fan_blade_fault_dsi.zip"
        assert meta["bytes"] == 56595859
        assert meta["sha256"] == (
            "5194925e0f97387a54be989923ec34bef8e65e03fe21652552d7bbcdc21a959e"
        )

    def test_validate_registry_raises_on_missing_sha256(self):
        bad_registry = {
            "broken": {
                "filename": "broken.zip",
                "ti_name": "broken.zip",
                "bytes": 123,
                # sha256 deliberately absent
            }
        }
        with pytest.raises(ValueError, match="broken"):
            _validate_registry(bad_registry)

    def test_validate_registry_raises_on_short_sha256(self):
        bad_registry = {
            "broken": {
                "filename": "broken.zip",
                "ti_name": "broken.zip",
                "sha256": "deadbeef",
                "bytes": 123,
            }
        }
        with pytest.raises(ValueError, match="broken"):
            _validate_registry(bad_registry)

    def test_validate_registry_raises_on_zero_bytes(self):
        bad_registry = {
            "broken": {
                "filename": "broken.zip",
                "ti_name": "broken.zip",
                "sha256": "a" * 64,
                "bytes": 0,
            }
        }
        with pytest.raises(ValueError, match="broken"):
            _validate_registry(bad_registry)

    def test_validate_registry_ignores_entries_without_ti_name(self):
        # An entry with no ti_name is not fetchable, so it needs no digest.
        registry = {"local_only": {"filename": "local.zip"}}
        _validate_registry(registry)  # must not raise

    def test_real_registry_passes_validation(self):
        _validate_registry(DATASET_REGISTRY)  # must not raise (also ran at import)


class TestDatasetUrl:
    def test_fan_blade_fault_url(self):
        url = dataset_url("fan_blade_fault")
        assert url == (
            "https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/"
            "fan_blade_fault_dsi.zip"
        )

    def test_generic_audio_classification_returns_none(self):
        assert dataset_url("generic_audio_classification") is None

    def test_unknown_name_raises_keyerror(self):
        with pytest.raises(KeyError):
            dataset_url("does_not_exist_at_all")

    def test_url_uses_default_version(self):
        for name, meta in DATASET_REGISTRY.items():
            if not meta.get("ti_name"):
                continue
            url = dataset_url(name)
            assert f"/{DATASETS_DEFAULT_VERSION}/datasets/" in url

    def test_per_entry_ti_version_override(self, monkeypatch):
        monkeypatch.setitem(
            DATASET_REGISTRY["fan_blade_fault"], "ti_version", "01_04_00"
        )
        try:
            url = dataset_url("fan_blade_fault")
            assert "/01_04_00/datasets/" in url
            assert "/01_03_00/" not in url
        finally:
            DATASET_REGISTRY["fan_blade_fault"].pop("ti_version", None)

    def test_url_always_https(self):
        for name, meta in DATASET_REGISTRY.items():
            if not meta.get("ti_name"):
                continue
            assert dataset_url(name).startswith("https://software-dl.ti.com/")



class TestCacheDir:
    def test_cache_dir_is_version_scoped(self, isolated_cache):
        path = _cache_dir("01_03_00")
        assert path.endswith(os.path.join("mmcli", "datasets", "01_03_00"))
        assert os.path.isdir(path)

    def test_cache_dir_honours_xdg_cache_home(self, isolated_cache):
        path = _cache_dir("01_03_00")
        assert path.startswith(isolated_cache)

    def test_two_versions_cache_independently(self, isolated_cache):
        p1 = _cache_dir("01_03_00")
        p2 = _cache_dir("01_04_00")
        assert p1 != p2
        assert os.path.isdir(p1)
        assert os.path.isdir(p2)


class TestResolutionOrder:
    def test_env_var_wins_over_cache(self, isolated_cache, tmp_path, monkeypatch, fake_ti_entry):
        name = fake_ti_entry["name"]
        meta = DATASET_REGISTRY[name]

        # Plant a correctly-verified copy in the cache.
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)
        cache_path = os.path.join(cache_dir, meta["filename"])
        with open(cache_path, "wb") as f:
            f.write(fake_ti_entry["body"])

        # Plant a *different* file at the same name under MMCLI_DATASETS.
        env_dir = tmp_path / "env_datasets"
        env_dir.mkdir()
        (env_dir / meta["filename"]).write_bytes(b"env-provided bytes, not cache")
        monkeypatch.setenv("MMCLI_DATASETS", str(env_dir))

        resolved = _resolve_dataset_zip(name)
        assert resolved == str(env_dir / meta["filename"])

    def test_bundled_wins_over_cache(self, isolated_cache):
        # Use a real bundled dataset (still bundled pre-10-03) and plant a
        # bogus cache entry with the same filename; resolution must still
        # prefer the bundled copy.
        name = "generic_timeseries_regression"
        meta = DATASET_REGISTRY[name]
        version = meta.get("ti_version") or DATASETS_DEFAULT_VERSION
        cache_dir = _cache_dir(version)
        with open(os.path.join(cache_dir, meta["filename"]), "wb") as f:
            f.write(b"bogus cache content that must never be preferred")

        resolved = _resolve_dataset_zip(name)
        bundled_dir = os.path.join(os.path.dirname(datasets.__file__), "example_datasets")
        assert resolved == os.path.join(bundled_dir, meta["filename"])

    def test_cache_used_when_neither_env_nor_bundled(self, isolated_cache, fake_ti_entry):
        name = fake_ti_entry["name"]
        meta = DATASET_REGISTRY[name]
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)
        cache_path = os.path.join(cache_dir, meta["filename"])
        with open(cache_path, "wb") as f:
            f.write(fake_ti_entry["body"])

        resolved = _resolve_dataset_zip(name)
        assert resolved == cache_path

    def test_not_present_anywhere_returns_none(self, isolated_cache, fake_ti_entry):
        assert _resolve_dataset_zip(fake_ti_entry["name"]) is None

    def test_corrupted_cache_entry_treated_as_absent(self, isolated_cache, fake_ti_entry, capsys):
        name = fake_ti_entry["name"]
        meta = DATASET_REGISTRY[name]
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)
        cache_path = os.path.join(cache_dir, meta["filename"])
        with open(cache_path, "wb") as f:
            f.write(b"corrupted, does not match the recorded sha256 at all")

        resolved = _resolve_dataset_zip(name)
        assert resolved is None
        captured = capsys.readouterr()
        assert "sha256" in captured.err.lower() or "does not" in captured.err.lower()

    def test_unknown_name_returns_none(self, isolated_cache):
        assert _resolve_dataset_zip("nonexistent_dataset_xyz") is None

    def test_extract_dataset_still_works_when_zip_present_locally(self, isolated_cache):
        # REQ: mmcli init --dataset X behaves identically to before this
        # plan when the zip is already present via MMCLI_DATASETS/bundled.
        name = "generic_timeseries_regression"
        task_type = DATASET_REGISTRY[name]["task_types"][0]
        import tempfile as _tempfile
        with _tempfile.TemporaryDirectory() as d:
            project_path = os.path.join(d, "proj")
            extract_dataset(name, project_path, task_type=task_type)
            assert os.path.isdir(os.path.join(project_path, "dataset"))


class TestZipSlipProtection:
    """T-10-02-06: confirm extract_dataset() cannot be used to write outside
    the target project directory via a malicious zip member path. Python's
    zipfile has stripped '..'/absolute member paths since 3.6, but this
    plan's threat model calls for an explicit test rather than an assumption.
    """

    def test_parent_traversal_member_stays_inside_project(self, isolated_cache, tmp_path):
        import zipfile as _zipfile

        evil_zip = tmp_path / "evil.zip"
        with _zipfile.ZipFile(evil_zip, "w") as zf:
            zf.writestr("classes/class_a/ok.csv", "1,2,3\n")
            zf.writestr("../../../../tmp/evil_zip_slip_marker.txt", "pwned\n")

        DATASET_REGISTRY["_test_only_zip_slip"] = {
            "filename": "evil.zip",
            "task_types": ["generic_timeseries_classification"],
            "module": "timeseries",
            "description": "zip-slip regression fixture",
        }
        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "evil.zip").write_bytes(evil_zip.read_bytes())

        marker = tmp_path / "tmp" / "evil_zip_slip_marker.txt"
        try:
            import os as _os
            _os.environ["MMCLI_DATASETS"] = str(env_dir)
            project_path = str(tmp_path / "proj")
            extract_dataset(
                "_test_only_zip_slip", project_path,
                task_type="generic_timeseries_classification",
            )
            assert not marker.exists(), (
                "zip-slip member escaped the project directory"
            )
            # The benign member must still have landed correctly.
            assert os.path.isfile(
                os.path.join(project_path, "dataset", "classes", "class_a", "ok.csv")
            )
        finally:
            _os.environ.pop("MMCLI_DATASETS", None)
            DATASET_REGISTRY.pop("_test_only_zip_slip", None)

