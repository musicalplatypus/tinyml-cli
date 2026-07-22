"""Tests for the on-demand dataset fetch mechanism in mmcli/datasets.py.

Task 1 scope: registry invariants (ti_name/sha256/bytes) and the
version-pathed TI URL builder (`dataset_url`). Task 2 (cache/resolution) and
Task 3 (fetch_dataset/_download_to_cache) tests land in later commits of
this same plan.
"""

import pytest

from mmcli.datasets import (
    DATASET_REGISTRY,
    DATASETS_DEFAULT_VERSION,
    _validate_registry,
    dataset_url,
)


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

