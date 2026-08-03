"""Unit tests for scripts/release_preflight.py and scripts/verify_dataset_digests.py's
decision logic (10-REVIEW.md WR-04): before this file, neither script was imported by any
test, so their correctness — and, after 10-REVIEW.md WR-03, whether they fail *closed*
with a legible `FATAL:` message rather than a raw traceback when `gh` is absent or the
digest script is missing — was entirely unverified.

`gh` and the digest subprocess are stubbed throughout: this exercises decision logic only
(wrong tagName, missing/zero-size asset, non-zero child exit), never the real network or a
real `gh` invocation.

Not part of the CI-collected six files (10-REVIEW.md IN-06 tracks that scope gap
separately, as an Info finding); this file exists so the logic itself has coverage
regardless of what CI wires in.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def release_preflight():
    return _load_module("_test_release_preflight", SCRIPTS_DIR / "release_preflight.py")


@pytest.fixture
def verify_dataset_digests():
    return _load_module(
        "_test_verify_dataset_digests", SCRIPTS_DIR / "verify_dataset_digests.py"
    )


def _mirror_tag():
    from mmcli.datasets import DATASETS_DEFAULT_VERSION, DATASETS_MIRROR_TAG_PREFIX

    return f"{DATASETS_MIRROR_TAG_PREFIX}{DATASETS_DEFAULT_VERSION}"


def _expected_filenames():
    from mmcli.datasets import DATASET_REGISTRY

    return sorted(
        entry["filename"] for entry in DATASET_REGISTRY.values() if entry.get("ti_name")
    )


class TestCheckMirrorTagAndAssets:
    def test_gh_missing_fails_closed_with_fatal_message_not_traceback(
        self, release_preflight, capsys, monkeypatch
    ):
        def _raise_not_found(*a, **k):
            raise FileNotFoundError("gh: command not found")

        monkeypatch.setattr(release_preflight.subprocess, "run", _raise_not_found)
        result = release_preflight.check_mirror_tag_and_assets()
        assert result is False
        err = capsys.readouterr().err
        assert "FATAL:" in err
        assert "gh" in err.lower()

    def test_gh_non_zero_return_code_fails(self, release_preflight, capsys, monkeypatch):
        fake = mock.Mock(returncode=1, stdout="", stderr="release not found")
        monkeypatch.setattr(release_preflight.subprocess, "run", lambda *a, **k: fake)
        result = release_preflight.check_mirror_tag_and_assets()
        assert result is False
        assert "not reachable" in capsys.readouterr().err

    def test_gh_wrong_tag_name_fails(self, release_preflight, capsys, monkeypatch):
        fake = mock.Mock(
            returncode=0,
            stdout=json.dumps({"tagName": "wrong-tag", "assets": []}),
            stderr="",
        )
        monkeypatch.setattr(release_preflight.subprocess, "run", lambda *a, **k: fake)
        result = release_preflight.check_mirror_tag_and_assets()
        assert result is False
        assert "mis-tagged" in capsys.readouterr().err

    def test_gh_missing_asset_fails(self, release_preflight, capsys, monkeypatch):
        tag = _mirror_tag()
        fake = mock.Mock(
            returncode=0, stdout=json.dumps({"tagName": tag, "assets": []}), stderr=""
        )
        monkeypatch.setattr(release_preflight.subprocess, "run", lambda *a, **k: fake)
        result = release_preflight.check_mirror_tag_and_assets()
        assert result is False
        assert "missing expected asset" in capsys.readouterr().err

    def test_gh_zero_size_asset_fails(self, release_preflight, capsys, monkeypatch):
        tag = _mirror_tag()
        assets = [{"name": name, "size": 0} for name in _expected_filenames()]
        fake = mock.Mock(
            returncode=0,
            stdout=json.dumps({"tagName": tag, "assets": assets}),
            stderr="",
        )
        monkeypatch.setattr(release_preflight.subprocess, "run", lambda *a, **k: fake)
        result = release_preflight.check_mirror_tag_and_assets()
        assert result is False
        assert "zero-byte asset" in capsys.readouterr().err

    def test_gh_success_passes(self, release_preflight, monkeypatch):
        tag = _mirror_tag()
        assets = [{"name": name, "size": 12345} for name in _expected_filenames()]
        fake = mock.Mock(
            returncode=0,
            stdout=json.dumps({"tagName": tag, "assets": assets}),
            stderr="",
        )
        monkeypatch.setattr(release_preflight.subprocess, "run", lambda *a, **k: fake)
        assert release_preflight.check_mirror_tag_and_assets() is True

    def test_default_progress_label_is_1_of_2(
        self, release_preflight, capsys, monkeypatch
    ):
        tag = _mirror_tag()
        assets = [{"name": name, "size": 12345} for name in _expected_filenames()]
        fake = mock.Mock(
            returncode=0,
            stdout=json.dumps({"tagName": tag, "assets": assets}),
            stderr="",
        )
        monkeypatch.setattr(release_preflight.subprocess, "run", lambda *a, **k: fake)
        release_preflight.check_mirror_tag_and_assets()
        assert "[1/2]" in capsys.readouterr().err

    def test_total_steps_one_progress_label_is_1_of_1(
        self, release_preflight, capsys, monkeypatch
    ):
        # 10-REVIEW.md IN-02: under --skip-digests, main() passes
        # total_steps=1 since the digest step never runs in this
        # invocation -- the label must say so, not a hardcoded "[1/2]".
        tag = _mirror_tag()
        assets = [{"name": name, "size": 12345} for name in _expected_filenames()]
        fake = mock.Mock(
            returncode=0,
            stdout=json.dumps({"tagName": tag, "assets": assets}),
            stderr="",
        )
        monkeypatch.setattr(release_preflight.subprocess, "run", lambda *a, **k: fake)
        release_preflight.check_mirror_tag_and_assets(total_steps=1)
        err = capsys.readouterr().err
        assert "[1/1]" in err
        assert "[1/2]" not in err

    def test_gh_non_json_stdout_fails_closed_with_fatal_message(
        self, release_preflight, capsys, monkeypatch
    ):
        # 10-REVIEW.md IN-03: json.loads(result.stdout) previously raised an
        # unguarded JSONDecodeError on non-JSON `gh` output.
        fake = mock.Mock(returncode=0, stdout="not json at all {{{", stderr="")
        monkeypatch.setattr(release_preflight.subprocess, "run", lambda *a, **k: fake)
        result = release_preflight.check_mirror_tag_and_assets()
        assert result is False
        err = capsys.readouterr().err
        assert "FATAL:" in err
        assert "not json at all" in err

    def test_gh_non_object_json_stdout_fails_closed_with_fatal_message(
        self, release_preflight, capsys, monkeypatch
    ):
        # Valid JSON, but not the expected object shape (e.g. a bare list) --
        # data.get("tagName") would raise AttributeError unguarded.
        fake = mock.Mock(returncode=0, stdout=json.dumps([1, 2, 3]), stderr="")
        monkeypatch.setattr(release_preflight.subprocess, "run", lambda *a, **k: fake)
        result = release_preflight.check_mirror_tag_and_assets()
        assert result is False
        assert "FATAL:" in capsys.readouterr().err

    def test_gh_null_assets_fails_closed_with_fatal_message(
        self, release_preflight, capsys, monkeypatch
    ):
        # 10-REVIEW.md IN-03: `"assets": null` makes data.get("assets", [])
        # return None (the key is present), and iterating None in the dict
        # comprehension previously raised an unguarded TypeError.
        tag = _mirror_tag()
        fake = mock.Mock(
            returncode=0,
            stdout=json.dumps({"tagName": tag, "assets": None}),
            stderr="",
        )
        monkeypatch.setattr(release_preflight.subprocess, "run", lambda *a, **k: fake)
        result = release_preflight.check_mirror_tag_and_assets()
        assert result is False
        assert "FATAL:" in capsys.readouterr().err

    def test_gh_asset_missing_name_key_fails_closed_with_fatal_message(
        self, release_preflight, capsys, monkeypatch
    ):
        # 10-REVIEW.md IN-03: an asset object without "name" previously
        # raised an unguarded KeyError building the assets dict.
        tag = _mirror_tag()
        fake = mock.Mock(
            returncode=0,
            stdout=json.dumps({"tagName": tag, "assets": [{"size": 100}]}),
            stderr="",
        )
        monkeypatch.setattr(release_preflight.subprocess, "run", lambda *a, **k: fake)
        result = release_preflight.check_mirror_tag_and_assets()
        assert result is False
        assert "FATAL:" in capsys.readouterr().err


class TestCheckDigests:
    def test_missing_script_fails_closed_with_fatal_message(
        self, release_preflight, capsys, monkeypatch, tmp_path
    ):
        # 10-REVIEW.md WR-03: a missing digest script must fail with a FATAL:
        # message, not an unguarded FileNotFoundError from subprocess.run.
        monkeypatch.setattr(release_preflight, "REPO_ROOT", tmp_path)
        result = release_preflight.check_digests()
        assert result is False
        assert "FATAL:" in capsys.readouterr().err

    def test_nonzero_exit_fails(self, release_preflight, monkeypatch):
        fake = mock.Mock(returncode=1)
        monkeypatch.setattr(release_preflight.subprocess, "run", lambda *a, **k: fake)
        assert release_preflight.check_digests() is False

    def test_zero_exit_passes(self, release_preflight, monkeypatch):
        fake = mock.Mock(returncode=0)
        monkeypatch.setattr(release_preflight.subprocess, "run", lambda *a, **k: fake)
        assert release_preflight.check_digests() is True

    def test_resolves_script_relative_to_repo_root_not_cwd(
        self, release_preflight, monkeypatch, tmp_path
    ):
        # 10-REVIEW.md WR-03: the digest script's path must not depend on the
        # caller's CWD.
        captured = {}

        def _fake_run(argv, **kwargs):
            captured["argv"] = argv
            captured["cwd"] = kwargs.get("cwd")
            return mock.Mock(returncode=0)

        monkeypatch.setattr(release_preflight.subprocess, "run", _fake_run)
        monkeypatch.chdir(tmp_path)
        assert release_preflight.check_digests() is True
        assert str(release_preflight.REPO_ROOT) in captured["argv"][1]
        assert captured["cwd"] == str(release_preflight.REPO_ROOT)


class TestReleasePreflightMain:
    def test_mirror_check_failure_short_circuits_before_digests(
        self, release_preflight, monkeypatch
    ):
        monkeypatch.setattr(
            release_preflight, "check_mirror_tag_and_assets", lambda **kw: False
        )
        digests_called = []
        monkeypatch.setattr(
            release_preflight, "check_digests", lambda: digests_called.append(1) or True
        )
        code = release_preflight.main([])
        assert code == 1
        assert digests_called == []

    def test_skip_digests_returns_partial_exit_code_without_running_digest_gate(
        self, release_preflight, monkeypatch
    ):
        # 10-REVIEW.md IN-02: --skip-digests must NOT exit 0 -- that would
        # let a wrapper checking only $? mistake a skipped ~131 MB digest
        # gate for a full preflight pass. PARTIAL_EXIT_CODE (3) is
        # deliberately distinct from both 0 (full PASS) and 1 (a check
        # actually failed).
        monkeypatch.setattr(
            release_preflight, "check_mirror_tag_and_assets", lambda **kw: True
        )
        digests_called = []
        monkeypatch.setattr(
            release_preflight, "check_digests", lambda: digests_called.append(1) or True
        )
        code = release_preflight.main(["--skip-digests"])
        assert code == release_preflight.PARTIAL_EXIT_CODE
        assert code != 0
        assert digests_called == []

    def test_skip_digests_passes_total_steps_one_to_mirror_check(
        self, release_preflight, monkeypatch
    ):
        # 10-REVIEW.md IN-02: the "[1/N]" progress label must reflect that
        # only one step will run under --skip-digests, not a hardcoded
        # "[1/2]" that implies a second step is coming.
        captured = {}

        def _fake_check(**kw):
            captured.update(kw)
            return True

        monkeypatch.setattr(release_preflight, "check_mirror_tag_and_assets", _fake_check)
        monkeypatch.setattr(release_preflight, "check_digests", lambda: True)
        release_preflight.main(["--skip-digests"])
        assert captured.get("total_steps") == 1

    def test_without_skip_digests_passes_total_steps_two_to_mirror_check(
        self, release_preflight, monkeypatch
    ):
        captured = {}

        def _fake_check(**kw):
            captured.update(kw)
            return True

        monkeypatch.setattr(release_preflight, "check_mirror_tag_and_assets", _fake_check)
        monkeypatch.setattr(release_preflight, "check_digests", lambda: True)
        release_preflight.main([])
        assert captured.get("total_steps") == 2

    def test_both_checks_passing_returns_zero(self, release_preflight, monkeypatch):
        monkeypatch.setattr(
            release_preflight, "check_mirror_tag_and_assets", lambda **kw: True
        )
        monkeypatch.setattr(release_preflight, "check_digests", lambda: True)
        assert release_preflight.main([]) == 0

    def test_digest_failure_after_mirror_success_returns_nonzero(
        self, release_preflight, monkeypatch
    ):
        monkeypatch.setattr(
            release_preflight, "check_mirror_tag_and_assets", lambda **kw: True
        )
        monkeypatch.setattr(release_preflight, "check_digests", lambda: False)
        assert release_preflight.main([]) == 1


@pytest.fixture
def _restore_xdg_cache_home():
    """verify_dataset_digests.main() mutates os.environ["XDG_CACHE_HOME"]
    directly (not via monkeypatch, since it's the real script under test),
    so restore it manually to avoid leaking into later tests."""
    import os

    before = os.environ.get("XDG_CACHE_HOME")
    yield
    if before is None:
        os.environ.pop("XDG_CACHE_HOME", None)
    else:
        os.environ["XDG_CACHE_HOME"] = before


class TestVerifyDatasetDigestsMain:
    def test_all_pass_returns_zero(
        self, verify_dataset_digests, monkeypatch, capsys, _restore_xdg_cache_home
    ):
        import mmcli.datasets as datasets_mod

        monkeypatch.setattr(datasets_mod, "fetch_dataset", lambda name, force=True: "/tmp/x")
        code = verify_dataset_digests.main([])
        assert code == 0
        assert "PASSED" in capsys.readouterr().out

    def test_one_failure_returns_nonzero_and_names_it(
        self, verify_dataset_digests, monkeypatch, capsys, _restore_xdg_cache_home
    ):
        import mmcli.datasets as datasets_mod

        bad_name = sorted(
            n for n, m in datasets_mod.DATASET_REGISTRY.items()
            if datasets_mod.dataset_url(n) is not None
        )[0]

        def _fake_fetch(name, force=True):
            if name == bad_name:
                raise RuntimeError("sha256 mismatch")
            return "/tmp/x"

        monkeypatch.setattr(datasets_mod, "fetch_dataset", _fake_fetch)
        code = verify_dataset_digests.main([])
        assert code == 1
        out = capsys.readouterr().out
        assert bad_name in out
        assert "FAIL" in out

    def test_unknown_only_name_returns_two(
        self, verify_dataset_digests, capsys, _restore_xdg_cache_home
    ):
        code = verify_dataset_digests.main(["--only", "_not_a_real_dataset_"])
        assert code == 2

    def test_only_bundled_only_name_reports_specific_message_not_registry_wide(
        self, verify_dataset_digests, capsys, _restore_xdg_cache_home
    ):
        # 10-REVIEW.md IN-01: --only naming a real registry entry with no
        # mirror asset (bundled-only, e.g. generic_audio_classification)
        # used to fall through to the generic "No fetchable datasets found
        # in DATASET_REGISTRY" message -- true of the whole registry, false
        # and misleading for this single-name query.
        import mmcli.datasets as datasets_mod

        bundled_only = sorted(
            n for n in datasets_mod.DATASET_REGISTRY
            if datasets_mod.dataset_url(n) is None
        )[0]
        code = verify_dataset_digests.main(["--only", bundled_only])
        assert code == 2
        err = capsys.readouterr().err
        assert bundled_only in err
        assert "not fetchable" in err
        assert "No fetchable datasets found in DATASET_REGISTRY" not in err
