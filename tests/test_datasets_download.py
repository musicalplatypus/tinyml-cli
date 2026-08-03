"""Tests for the on-demand dataset fetch mechanism in mmcli/datasets.py.

Covers four things, corresponding to the tasks of Phase 10 Plans 02 and 03:

1. Registry invariants and the version-pathed GitHub release-mirror URL
   (`dataset_url`) — repointed from TI to the mirror in 10-03-PLAN.md D-A/D-B.
2. Version-scoped cache and the read-resolution order
   (`_resolve_dataset_zip`, `_cache_dir`).
3. `fetch_dataset` / `_download_to_cache`: mandatory sha256 verification,
   atomic replace, and every documented failure mode.
4. The narrowed cross-host redirect allowlist (10-03-PLAN.md D-C): the one
   github.com -> release-assets.githubusercontent.com hop is followed, and
   every other cross-host redirect is still refused.

Everything here runs against a local `http.server` fixture, never against
github.com or software-dl.ti.com — a unit suite that hits the real network is
a unit suite that fails on a train (and in CI). `fetch_dataset` itself
refuses any non-HTTPS URL, so the low-level downloader (`_download_to_cache`)
is tested directly with plain `http://127.0.0.1:<port>` URLs from the local
server, and `fetch_dataset`'s HTTPS-only policy is tested separately via
monkeypatching `dataset_url` to prove the refusal happens before any
request is attempted.
"""

import hashlib
import http.server
import os
import threading
import time
from pathlib import Path

import pytest

import mmcli.datasets as datasets
from mmcli.datasets import (
    DATASET_REGISTRY,
    DATASETS_DEFAULT_VERSION,
    _cache_dir,
    _download_to_cache,
    _resolve_dataset_zip,
    _sha256_of,
    _validate_registry,
    dataset_url,
    extract_dataset,
    fetch_dataset,
)


# ---------------------------------------------------------------------------
# Local HTTP server fixture (Task 3) — never touches the real network
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CI parity — see the matching note in tests/test_datasets_cli.py
# ---------------------------------------------------------------------------
# `.gitignore` ignores mmcli/example_datasets/*.zip; only
# generic_audio_classification.zip is tracked, so a checkout has one of ten and
# the two TestResolutionOrder cases below fail with FileNotFoundError. Synthesise
# a valid stand-in for any missing zip and point that entry's recorded digest and
# size at it — these tests assert resolution order and extraction, not contents.
#
# 10-REVIEW.md WR-06: stand-ins are staged in a pytest tmp_path_factory
# directory and `_datasets_dir()` is redirected there — never written into the
# real `mmcli/example_datasets/` package tree, which `.gitignore` hid from
# `git status`. Any hard interrupt that skipped the old teardown's
# `os.unlink` left fake zips permanently in the developer's package tree,
# silently changing `mmcli init --dataset` behaviour from then on. A
# pytest-managed tmp_path_factory directory needs no manual cleanup.

@pytest.fixture(autouse=True)
def _dataset_zips_present(monkeypatch, request, tmp_path_factory):
    import hashlib as _h, os as _os, shutil as _sh, zipfile as _zf
    # TestRegistryInvariants asserts the registry's *recorded* sha256 and byte
    # counts. Overriding them for a stand-in would make it assert against the
    # stand-in and pass vacuously — it needs the real values, and it reads no
    # zip, so it opts out entirely.
    #
    # 10-REVIEW.md WR-07: opt out via an explicit marker on the class, not a
    # class-name string comparison here. A rename or a test split that moved
    # a sha256/bytes assertion into a differently-named class would silently
    # stop opting out under the string-match approach, leaving
    # REQ-DATA-02's invariant unguarded without any test failure to notice —
    # a marker travels with the class through a rename.
    if request.node.get_closest_marker("no_dataset_standins") is not None:
        yield
        return
    bundled = _os.path.join(_os.path.dirname(datasets.__file__), "example_datasets")
    stage = tmp_path_factory.mktemp("bundled")
    for _name, _meta in DATASET_REGISTRY.items():
        _src = _os.path.join(bundled, _meta["filename"])
        _dst = _os.path.join(str(stage), _meta["filename"])
        if _os.path.exists(_src):
            _sh.copyfile(_src, _dst)
            continue
        with _zf.ZipFile(_dst, "w") as _z:
            _z.writestr(f"{_name}/classes/a/sample.csv", "t,v\n0,1\n")
            _z.writestr(f"{_name}/classes/b/sample.csv", "t,v\n0,2\n")
        monkeypatch.setitem(DATASET_REGISTRY[_name], "sha256",
                            _h.sha256(open(_dst, "rb").read()).hexdigest())
        monkeypatch.setitem(DATASET_REGISTRY[_name], "bytes", _os.path.getsize(_dst))

    def _redirected_datasets_dir():
        # Preserve _datasets_dir()'s real MMCLI_DATASETS-first priority (see
        # mmcli/datasets.py) — only the *fallback* (bundled) directory is
        # redirected to the staged copy, not the env-var override.
        _env = _os.environ.get("MMCLI_DATASETS")
        if _env and _os.path.isdir(_env):
            return _env
        return str(stage)

    monkeypatch.setattr(datasets, "_datasets_dir", _redirected_datasets_dir)
    yield


class _RoutedHandler(http.server.BaseHTTPRequestHandler):
    """Serves whatever `routes[path]` says. Routes are set per-test on the
    class before starting the server, and cleared by the fixture teardown.
    """

    routes = {}
    request_log = []

    def do_GET(self):
        type(self).request_log.append(self.path)
        route = self.routes.get(self.path)
        if route is None:
            self.send_response(404)
            self.end_headers()
            return

        kind = route.get("kind", "ok")
        if kind == "redirect":
            self.send_response(302)
            self.send_header("Location", route["location"])
            self.end_headers()
            return
        if kind == "hang":
            time.sleep(route.get("delay", 2.0))
            self.send_response(200)
            self.end_headers()
            return

        body = route["body"]
        send_bytes = route.get("send_bytes", body)
        declared_length = route.get("content_length", len(body))
        self.send_response(200)
        self.send_header("Content-Length", str(declared_length))
        self.end_headers()
        self.wfile.write(send_bytes)

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        pass  # keep test output quiet


@pytest.fixture
def http_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RoutedHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _RoutedHandler.routes = {}
    _RoutedHandler.request_log = []
    port = server.server_address[1]
    try:
        yield f"http://127.0.0.1:{port}", _RoutedHandler
    finally:
        server.shutdown()
        server.server_close()


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


# ---------------------------------------------------------------------------
# Task 1 — registry invariants and dataset_url()
# ---------------------------------------------------------------------------

@pytest.mark.no_dataset_standins
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
            "https://github.com/musicalplatypus/tinyml-cli/releases/download/"
            "datasets-01_03_00/fan_blade_fault.zip"
        )

    def test_url_named_by_local_filename_not_ti_name(self):
        # D-A: the asset/URL name is the LOCAL filename (on-disk zip / cache
        # name), never the upstream ti_name.
        meta = DATASET_REGISTRY["fan_blade_fault"]
        assert meta["ti_name"] == "fan_blade_fault_dsi.zip"
        url = dataset_url("fan_blade_fault")
        assert url.endswith("/fan_blade_fault.zip")
        assert "fan_blade_fault_dsi" not in url

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
            assert f"/datasets-{DATASETS_DEFAULT_VERSION}/{meta['filename']}" in url

    def test_per_entry_ti_version_override(self, monkeypatch):
        # Guards the per-entry `ti_version` override (D-B): it now overrides
        # the mirror payload/release version, not a TI engine version.
        monkeypatch.setitem(
            DATASET_REGISTRY["fan_blade_fault"], "ti_version", "01_04_00"
        )
        try:
            url = dataset_url("fan_blade_fault")
            assert url == (
                "https://github.com/musicalplatypus/tinyml-cli/releases/"
                "download/datasets-01_04_00/fan_blade_fault.zip"
            )
            assert "/datasets-01_03_00/" not in url
        finally:
            DATASET_REGISTRY["fan_blade_fault"].pop("ti_version", None)

    def test_url_always_https(self):
        for name, meta in DATASET_REGISTRY.items():
            if not meta.get("ti_name"):
                continue
            assert dataset_url(name).startswith(
                "https://github.com/musicalplatypus/tinyml-cli/releases/download/"
            )


# ---------------------------------------------------------------------------
# Task 2 — version-scoped cache and resolution order
# ---------------------------------------------------------------------------

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
        # Resolved dynamically via the (possibly test-staged — see
        # _dataset_zips_present, 10-REVIEW.md WR-06) _datasets_dir(), not a
        # hardcoded real-package-tree path, so this stays correct regardless
        # of where bundled zips are staged for the test run.
        assert resolved == os.path.join(datasets._datasets_dir(), meta["filename"])

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

    def test_cache_hit_with_no_sha256_reported_as_unverifiable_not_mismatch(
        self, isolated_cache, monkeypatch, capsys
    ):
        # 10-REVIEW.md IN-04: an entry with no recorded sha256 (legal for a
        # non-ti_name local entry) hitting the cache used to fall into the
        # "does not match the recorded sha256" branch -- false (nothing was
        # compared) and its suggested `--force` remedy fails for an entry
        # with no fetchable URL. The wording must say "no ... sha256", not
        # "does not match".
        name = "_test_only_no_sha256"
        monkeypatch.setitem(DATASET_REGISTRY, name, {
            "filename": "_test_only_no_sha256.zip",
            "task_types": ["generic_timeseries_classification"],
            "module": "timeseries",
            "description": "no-sha256 local entry regression fixture",
        })
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)
        cache_path = os.path.join(cache_dir, "_test_only_no_sha256.zip")
        with open(cache_path, "wb") as f:
            f.write(b"unverifiable payload, no sha256 recorded to check it against")

        resolved = _resolve_dataset_zip(name)
        assert resolved is None
        err = capsys.readouterr().err.lower()
        assert "no" in err and "sha256" in err
        assert "does not" not in err
        assert "--force" not in err

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

    def test_parent_traversal_member_stays_inside_project(self, isolated_cache, tmp_path, monkeypatch):
        import zipfile as _zipfile

        evil_zip = tmp_path / "evil.zip"
        with _zipfile.ZipFile(evil_zip, "w") as zf:
            zf.writestr("classes/class_a/ok.csv", "1,2,3\n")
            zf.writestr("../../../../tmp/evil_zip_slip_marker.txt", "pwned\n")

        monkeypatch.setitem(DATASET_REGISTRY, "_test_only_zip_slip", {
            "filename": "evil.zip",
            "task_types": ["generic_timeseries_classification"],
            "module": "timeseries",
            "description": "zip-slip regression fixture",
        })
        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "evil.zip").write_bytes(evil_zip.read_bytes())

        # The malicious member is "../../../../tmp/evil_zip_slip_marker.txt". A
        # successful escape resolves relative to <project_path>/dataset/ (where
        # extract_dataset() actually extracts), NOT relative to tmp_path itself —
        # asserting on the wrong path made this test pass unconditionally, even
        # against an extract_dataset() with zero protection (10-REVIEW.md CR-02).
        project_path_obj = tmp_path / "proj"
        escape_root = (project_path_obj / "dataset" / "../../../../tmp").resolve()
        marker = escape_root / "evil_zip_slip_marker.txt"

        # 10-REVIEW.md IN-05: use monkeypatch.setenv (auto-restored on
        # teardown, including on assertion failure) instead of a raw
        # os.environ mutation guarded by a try/finally whose cleanup depended
        # on an import performed inside the try block.
        monkeypatch.setenv("MMCLI_DATASETS", str(env_dir))
        project_path = str(project_path_obj)
        extract_dataset(
            "_test_only_zip_slip", project_path,
            task_type="generic_timeseries_classification",
        )
        assert not marker.exists(), (
            f"zip-slip member escaped to {marker}"
        )
        # Positive containment: every extracted file must stay under the
        # project directory — non-vacuous even if the marker path above is
        # ever wrong again.
        resolved_project = Path(project_path).resolve()
        extracted = [p.resolve() for p in resolved_project.rglob("*") if p.is_file()]
        assert extracted, "nothing was extracted — the test proved nothing"
        for p in extracted:
            assert p.is_relative_to(resolved_project), f"{p} escaped {resolved_project}"
        # The benign member must still have landed correctly.
        assert os.path.isfile(
            os.path.join(project_path, "dataset", "classes", "class_a", "ok.csv")
        )


class TestExtractDatasetFailureCleanup:
    """10-REVIEW.md WR-13: a failed extraction must not leave a half-created
    <project>/dataset/ directory behind — the next attempt would otherwise
    hit the "Project directory already exists" refusal and require a manual
    `rm -rf` of a directory mmcli itself created and abandoned. Also widens
    the except clause: the docstring promises SystemExit "on any error", but
    previously only zipfile.BadZipFile was caught.
    """

    def test_bad_zip_file_removes_the_half_created_project_directory(
        self, isolated_cache, tmp_path
    ):
        DATASET_REGISTRY["_test_only_bad_zip"] = {
            "filename": "not_a_zip.zip",
            "task_types": ["generic_timeseries_classification"],
            "module": "timeseries",
            "description": "corrupt-zip regression fixture",
        }
        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "not_a_zip.zip").write_bytes(b"this is not a valid zip file at all")

        project_path = str(tmp_path / "proj")
        try:
            os.environ["MMCLI_DATASETS"] = str(env_dir)
            with pytest.raises(SystemExit):
                extract_dataset(
                    "_test_only_bad_zip", project_path,
                    task_type="generic_timeseries_classification",
                )
            assert not os.path.exists(project_path), (
                "a failed extraction left a half-created project directory "
                "behind — the next attempt would hit 'already exists'"
            )
        finally:
            os.environ.pop("MMCLI_DATASETS", None)
            DATASET_REGISTRY.pop("_test_only_bad_zip", None)

    def test_oserror_mid_extract_is_caught_and_cleans_up(
        self, isolated_cache, tmp_path, monkeypatch
    ):
        # Widened except clause (10-REVIEW.md WR-13): OSError (disk full,
        # permission denied mid-extract) must be caught, not escape as a raw
        # traceback the docstring's "SystemExit on any error" promise
        # contradicts.
        import zipfile as _zipfile

        real_zip = tmp_path / "real.zip"
        with _zipfile.ZipFile(real_zip, "w") as zf:
            zf.writestr("classes/class_a/ok.csv", "1,2,3\n")

        DATASET_REGISTRY["_test_only_oserror"] = {
            "filename": "real.zip",
            "task_types": ["generic_timeseries_classification"],
            "module": "timeseries",
            "description": "OSError-mid-extract regression fixture",
        }
        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "real.zip").write_bytes(real_zip.read_bytes())

        def _raise_oserror(self, member, path=None, pwd=None):
            raise OSError("disk full (simulated)")

        monkeypatch.setattr(_zipfile.ZipFile, "extract", _raise_oserror)

        project_path = str(tmp_path / "proj")
        try:
            os.environ["MMCLI_DATASETS"] = str(env_dir)
            with pytest.raises(SystemExit):
                extract_dataset(
                    "_test_only_oserror", project_path,
                    task_type="generic_timeseries_classification",
                )
            assert not os.path.exists(project_path), (
                "an OSError mid-extract left a half-created project "
                "directory behind"
            )
        finally:
            os.environ.pop("MMCLI_DATASETS", None)
            DATASET_REGISTRY.pop("_test_only_oserror", None)


# ---------------------------------------------------------------------------
# Task 3 — fetch_dataset / _download_to_cache
# ---------------------------------------------------------------------------

class TestDownloadToCacheHappyPath:
    def test_downloads_verifies_and_returns_path(self, isolated_cache, http_server):
        base_url, handler = http_server
        body = b"hello world " * 500
        sha256 = hashlib.sha256(body).hexdigest()
        handler.routes["/ok.zip"] = {"body": body}

        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)
        dest = _download_to_cache(
            f"{base_url}/ok.zip", cache_dir, "ok.zip", sha256, len(body), "ok"
        )
        assert dest == os.path.join(cache_dir, "ok.zip")
        assert os.path.isfile(dest)
        assert _sha256_of(dest) == sha256
        # No leftover temp files.
        leftovers = [f for f in os.listdir(cache_dir) if f.startswith(".fetch-")]
        assert leftovers == []

    def test_no_temp_file_left_on_success(self, isolated_cache, http_server):
        base_url, handler = http_server
        body = b"clean run\n" * 200
        sha256 = hashlib.sha256(body).hexdigest()
        handler.routes["/clean.zip"] = {"body": body}
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)

        _download_to_cache(
            f"{base_url}/clean.zip", cache_dir, "clean.zip", sha256, len(body), "clean"
        )
        assert set(os.listdir(cache_dir)) == {"clean.zip"}


class TestDownloadToCacheFailureModes:
    def test_checksum_mismatch_leaves_no_cache_entry(self, isolated_cache, http_server):
        base_url, handler = http_server
        body = b"actual bytes served"
        wrong_sha256 = hashlib.sha256(b"different bytes entirely").hexdigest()
        handler.routes["/bad.zip"] = {"body": body}
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)

        with pytest.raises(RuntimeError, match="Checksum mismatch"):
            _download_to_cache(
                f"{base_url}/bad.zip", cache_dir, "bad.zip",
                wrong_sha256, len(body), "bad",
            )
        assert os.listdir(cache_dir) == []

    def test_truncated_body_raises_and_leaves_no_cache_entry(self, isolated_cache, http_server):
        base_url, handler = http_server
        full_body = b"x" * 5000
        sent_body = full_body[:1000]  # server closes early
        sha256 = hashlib.sha256(full_body).hexdigest()
        handler.routes["/truncated.zip"] = {
            "body": sent_body,
            "send_bytes": sent_body,
            "content_length": len(full_body),  # lies about the real length
        }
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)

        with pytest.raises(RuntimeError):
            _download_to_cache(
                f"{base_url}/truncated.zip", cache_dir, "truncated.zip",
                sha256, len(full_body), "truncated",
            )
        assert os.listdir(cache_dir) == []

    def test_oversized_body_aborted_mid_stream(self, isolated_cache, http_server):
        base_url, handler = http_server
        expected_bytes = 1000
        # Body is far larger than declared registry size + tolerance.
        oversized_body = b"y" * 50000
        sha256 = hashlib.sha256(oversized_body).hexdigest()
        handler.routes["/huge.zip"] = {
            "body": oversized_body,
            "content_length": len(oversized_body),
        }
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)

        with pytest.raises(RuntimeError, match="Aborting fetch|exceeds the registry size"):
            _download_to_cache(
                f"{base_url}/huge.zip", cache_dir, "huge.zip",
                sha256, expected_bytes, "huge",
            )
        assert os.listdir(cache_dir) == []

    def test_cross_host_redirect_refused(self, isolated_cache, http_server):
        base_url, handler = http_server
        handler.routes["/redirect.zip"] = {
            "kind": "redirect",
            "location": "http://evil.example.invalid/payload.zip",
        }
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)

        with pytest.raises(RuntimeError, match="cross-host redirect"):
            _download_to_cache(
                f"{base_url}/redirect.zip", cache_dir, "redirected.zip",
                "a" * 64, 100, "redirected",
            )
        assert os.listdir(cache_dir) == []


class TestAllowedCrossHostRedirect:
    """10-03-PLAN.md D-C: the redirect handler's allowlist is a closed,
    exact-host mapping. It must follow the one verified
    github.com -> release-assets.githubusercontent.com pair and still refuse
    every other cross-host redirect, including lookalike hosts that a
    suffix/wildcard match would wrongly admit. These drive
    `_HostLockedRedirectHandler` directly (not through a real network
    request) since "github.com" cannot be the host of the local test server.
    """

    def test_github_to_release_assets_redirect_is_followed(self):
        handler = datasets._HostLockedRedirectHandler("github.com", "some_dataset")
        req = datasets.urllib.request.Request(
            "https://github.com/musicalplatypus/tinyml-cli/releases/"
            "download/datasets-01_03_00/fan_blade_fault.zip"
        )
        new_req = handler.redirect_request(
            req, None, 302, "Found", None,
            "https://release-assets.githubusercontent.com/signed/path?se=1",
        )
        assert new_req.full_url == (
            "https://release-assets.githubusercontent.com/signed/path?se=1"
        )

    def test_github_to_unrelated_host_still_refused(self):
        handler = datasets._HostLockedRedirectHandler("github.com", "some_dataset")
        req = datasets.urllib.request.Request(
            "https://github.com/musicalplatypus/tinyml-cli/releases/"
            "download/datasets-01_03_00/fan_blade_fault.zip"
        )
        with pytest.raises(RuntimeError, match="cross-host redirect"):
            handler.redirect_request(
                req, None, 302, "Found", None,
                "https://evil.example.com/payload.zip",
            )

    def test_github_to_lookalike_suffix_host_still_refused(self):
        # A suffix/endswith match on "githubusercontent.com" would wrongly
        # allow this; exact host-string equality must refuse it.
        handler = datasets._HostLockedRedirectHandler("github.com", "some_dataset")
        req = datasets.urllib.request.Request(
            "https://github.com/musicalplatypus/tinyml-cli/releases/"
            "download/datasets-01_03_00/fan_blade_fault.zip"
        )
        with pytest.raises(RuntimeError, match="cross-host redirect"):
            handler.redirect_request(
                req, None, 302, "Found", None,
                "https://release-assets.githubusercontent.com.evil.com/x.zip",
            )

    def test_github_to_lookalike_prefix_host_still_refused(self):
        handler = datasets._HostLockedRedirectHandler("github.com", "some_dataset")
        req = datasets.urllib.request.Request(
            "https://github.com/musicalplatypus/tinyml-cli/releases/"
            "download/datasets-01_03_00/fan_blade_fault.zip"
        )
        with pytest.raises(RuntimeError, match="cross-host redirect"):
            handler.redirect_request(
                req, None, 302, "Found", None,
                "https://evil-githubusercontent.com/x.zip",
            )

    def test_allowlist_does_not_apply_to_non_github_original_host(self):
        # The allowlist is keyed on the ORIGINAL request host. A redirect
        # away from some other host to release-assets.githubusercontent.com
        # is not in ALLOWED_CROSS_HOST_REDIRECTS and must still be refused.
        handler = datasets._HostLockedRedirectHandler(
            "software-dl.ti.com", "some_dataset"
        )
        req = datasets.urllib.request.Request(
            "https://software-dl.ti.com/some/path/asset.zip"
        )
        with pytest.raises(RuntimeError, match="cross-host redirect"):
            handler.redirect_request(
                req, None, 302, "Found", None,
                "https://release-assets.githubusercontent.com/signed/path",
            )

    def test_same_host_http_downgrade_refused(self):
        # 10-REVIEW.md WR-05: the host lock alone does not check scheme, and
        # HTTPRedirectHandler permits http/https/ftp targets by default. A
        # same-host redirect to plain http:// must still be refused — a
        # silent transport downgrade to plaintext, even without a host change.
        handler = datasets._HostLockedRedirectHandler("github.com", "some_dataset")
        req = datasets.urllib.request.Request(
            "https://github.com/musicalplatypus/tinyml-cli/releases/"
            "download/datasets-01_03_00/fan_blade_fault.zip"
        )
        with pytest.raises(RuntimeError, match="non-HTTPS redirect"):
            handler.redirect_request(
                req, None, 302, "Found", None,
                "http://github.com/musicalplatypus/tinyml-cli/payload.zip",
            )

    def test_same_host_ftp_downgrade_refused(self):
        # urllib.request.build_opener() installs FTPHandler by default, so an
        # ftp:// redirect target is not merely unsupported — it would be
        # followed if not explicitly refused here.
        handler = datasets._HostLockedRedirectHandler("github.com", "some_dataset")
        req = datasets.urllib.request.Request(
            "https://github.com/musicalplatypus/tinyml-cli/releases/"
            "download/datasets-01_03_00/fan_blade_fault.zip"
        )
        with pytest.raises(RuntimeError, match="non-HTTPS redirect"):
            handler.redirect_request(
                req, None, 302, "Found", None,
                "ftp://github.com/musicalplatypus/tinyml-cli/payload.zip",
            )

    def test_allowlisted_target_host_over_http_still_refused(self):
        # Even the one allowlisted cross-host pair must not bypass the
        # scheme check — the scheme guard runs before the host allowlist.
        handler = datasets._HostLockedRedirectHandler("github.com", "some_dataset")
        req = datasets.urllib.request.Request(
            "https://github.com/musicalplatypus/tinyml-cli/releases/"
            "download/datasets-01_03_00/fan_blade_fault.zip"
        )
        with pytest.raises(RuntimeError, match="non-HTTPS redirect"):
            handler.redirect_request(
                req, None, 302, "Found", None,
                "http://release-assets.githubusercontent.com/signed/path",
            )

    def test_http_404_names_url(self, isolated_cache, http_server):
        base_url, _handler = http_server
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)

        with pytest.raises(RuntimeError, match="404"):
            _download_to_cache(
                f"{base_url}/does-not-exist.zip", cache_dir, "missing.zip",
                "a" * 64, 100, "missing",
            )
        assert os.listdir(cache_dir) == []

    def test_hung_server_times_out(self, isolated_cache, http_server, monkeypatch):
        base_url, handler = http_server
        monkeypatch.setattr(datasets, "DOWNLOAD_TIMEOUT_SECONDS", 0.3)
        handler.routes["/hang.zip"] = {"kind": "hang", "delay": 3.0}
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)

        with pytest.raises(RuntimeError):
            _download_to_cache(
                f"{base_url}/hang.zip", cache_dir, "hang.zip",
                "a" * 64, 100, "hang",
            )
        assert os.listdir(cache_dir) == []


class TestFetchDatasetPolicy:
    """Glue-level behaviour of fetch_dataset(): MMCLI_DATASETS refusal,
    HTTPS-only enforcement, no-ti_name refusal, unknown-name KeyError, and
    the cache short-circuit — all provable without a real network call by
    monkeypatching `_download_to_cache` to fail the test if it is invoked
    when it should not be.
    """

    def test_unknown_name_raises_keyerror(self, isolated_cache):
        with pytest.raises(KeyError):
            fetch_dataset("does_not_exist_at_all")

    def test_mmcli_datasets_set_refuses_without_any_request(self, isolated_cache, monkeypatch):
        monkeypatch.setenv("MMCLI_DATASETS", "/some/air/gapped/dir")
        monkeypatch.setattr(
            datasets, "_download_to_cache",
            lambda *a, **k: pytest.fail("must not attempt a download"),
        )
        with pytest.raises(RuntimeError, match="MMCLI_DATASETS"):
            fetch_dataset("fan_blade_fault")

    def test_no_ti_name_refuses(self, isolated_cache):
        with pytest.raises(RuntimeError, match="no upstream source"):
            fetch_dataset("generic_audio_classification")

    def test_non_https_url_refused_before_any_request(self, isolated_cache, monkeypatch, fake_ti_entry):
        name = fake_ti_entry["name"]
        monkeypatch.setattr(datasets, "dataset_url", lambda n: "http://example.com/x.zip")
        monkeypatch.setattr(
            datasets, "_download_to_cache",
            lambda *a, **k: pytest.fail("must not attempt a download over http"),
        )
        with pytest.raises(RuntimeError, match="HTTPS"):
            fetch_dataset(name)

    def test_second_call_on_verified_cache_issues_no_request(self, isolated_cache, monkeypatch, fake_ti_entry):
        name = fake_ti_entry["name"]
        meta = DATASET_REGISTRY[name]
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)
        dest = os.path.join(cache_dir, meta["filename"])
        with open(dest, "wb") as f:
            f.write(fake_ti_entry["body"])

        monkeypatch.setattr(
            datasets, "_download_to_cache",
            lambda *a, **k: pytest.fail("cached copy is valid; must not fetch"),
        )
        result = fetch_dataset(name)
        assert result == dest

    def test_corrupted_cache_entry_is_redownloaded(self, isolated_cache, http_server, fake_ti_entry):
        # Exercises the same _download_to_cache path fetch_dataset() falls
        # through to once it finds a cached file whose digest does not
        # match (see fetch_dataset's os.unlink-then-redownload branch).
        # Tested at the _download_to_cache level directly, per this test
        # file's HTTPS-testability note above.
        name = fake_ti_entry["name"]
        meta = DATASET_REGISTRY[name]
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)
        dest = os.path.join(cache_dir, meta["filename"])
        with open(dest, "wb") as f:
            f.write(b"stale, wrong content")
        os.unlink(dest)  # mirrors fetch_dataset's stale-cache-entry handling

        base_url, handler = http_server
        handler.routes[f"/{meta['ti_name']}"] = {"body": fake_ti_entry["body"]}

        result = _download_to_cache(
            f"{base_url}/{meta['ti_name']}", cache_dir, meta["filename"],
            meta["sha256"], meta["bytes"], name,
        )
        assert result == dest
        assert _sha256_of(dest) == meta["sha256"]

    def test_force_redownloads_even_when_cached(self, isolated_cache, http_server, fake_ti_entry):
        name = fake_ti_entry["name"]
        meta = DATASET_REGISTRY[name]
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)
        dest = os.path.join(cache_dir, meta["filename"])
        with open(dest, "wb") as f:
            f.write(fake_ti_entry["body"])  # already correct

        base_url, handler = http_server
        handler.routes[f"/{meta['ti_name']}"] = {"body": fake_ti_entry["body"]}

        # Directly exercises the same code path fetch_dataset(force=True)
        # would take once past the https gate (see class docstring).
        result = _download_to_cache(
            f"{base_url}/{meta['ti_name']}", cache_dir, meta["filename"],
            meta["sha256"], meta["bytes"], name,
        )
        assert os.path.getsize(result) == meta["bytes"]
        assert len(handler.request_log) >= 1

    def test_force_flag_bypasses_fetch_dataset_cache_short_circuit(self, isolated_cache, monkeypatch, fake_ti_entry):
        """force=True on fetch_dataset() itself must call the downloader
        even when a correctly-verified cache entry already exists."""
        name = fake_ti_entry["name"]
        meta = DATASET_REGISTRY[name]
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)
        dest = os.path.join(cache_dir, meta["filename"])
        with open(dest, "wb") as f:
            f.write(fake_ti_entry["body"])  # already valid

        calls = []

        def fake_download(url, cdir, filename, sha256, nbytes, dsname):
            calls.append(dsname)
            return dest

        monkeypatch.setattr(datasets, "_download_to_cache", fake_download)
        monkeypatch.setattr(datasets, "dataset_url", lambda n: "https://example.invalid/x.zip")

        result = fetch_dataset(name, force=True)
        assert calls == [name]
        assert result == dest

        # Without force, the same setup must NOT call the downloader.
        calls.clear()
        result2 = fetch_dataset(name)
        assert calls == []
        assert result2 == dest

    def test_fetch_dataset_stale_cache_entry_is_unlinked_and_redownloaded(
        self, isolated_cache, monkeypatch, fake_ti_entry
    ):
        # 10-REVIEW.md WR-08: exercises fetch_dataset()'s own stale-cache
        # branch directly (unlike test_corrupted_cache_entry_is_redownloaded
        # above, which does the os.unlink itself and calls
        # _download_to_cache directly — testing the helper, not the branch
        # it claims to cover).
        name = fake_ti_entry["name"]
        meta = DATASET_REGISTRY[name]
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)
        dest = os.path.join(cache_dir, meta["filename"])
        with open(dest, "wb") as f:
            f.write(b"stale, wrong content")  # sha256 mismatch

        calls = []

        def fake_download(url, cdir, filename, sha256, nbytes, dsname):
            calls.append(dsname)
            return dest

        monkeypatch.setattr(datasets, "_download_to_cache", fake_download)
        monkeypatch.setattr(datasets, "dataset_url", lambda n: "https://example.invalid/x.zip")

        result = fetch_dataset(name)
        assert calls == [name], (
            "fetch_dataset() must unlink the stale entry and call the "
            "downloader itself, not silently keep serving the stale bytes"
        )
        assert result == dest

    def test_fetch_dataset_stale_cache_unlink_failure_raises_runtimeerror(
        self, isolated_cache, monkeypatch, fake_ti_entry
    ):
        # 10-REVIEW.md WR-08: os.unlink can raise OSError (read-only cache,
        # permissions, a Windows sharing violation). fetch_dataset()'s
        # docstring promises RuntimeError on every failure mode, and callers
        # only catch (KeyError, RuntimeError) — an unguarded OSError would
        # escape as a raw traceback instead.
        name = fake_ti_entry["name"]
        meta = DATASET_REGISTRY[name]
        cache_dir = _cache_dir(DATASETS_DEFAULT_VERSION)
        dest = os.path.join(cache_dir, meta["filename"])
        with open(dest, "wb") as f:
            f.write(b"stale, wrong content")

        monkeypatch.setattr(datasets, "dataset_url", lambda n: "https://example.invalid/x.zip")

        def _raise_permission_denied(path):
            raise OSError("Permission denied (simulated)")

        monkeypatch.setattr(datasets.os, "unlink", _raise_permission_denied)

        with pytest.raises(RuntimeError, match="could not be removed"):
            fetch_dataset(name)


class TestFetchDatasetMmcliDatasetsOverridesForce:
    def test_mmcli_datasets_blocks_even_with_force(self, isolated_cache, monkeypatch, fake_ti_entry):
        monkeypatch.setenv("MMCLI_DATASETS", "/air/gapped")
        monkeypatch.setattr(
            datasets, "_download_to_cache",
            lambda *a, **k: pytest.fail("must not attempt a download"),
        )
        with pytest.raises(RuntimeError, match="MMCLI_DATASETS"):
            fetch_dataset(fake_ti_entry["name"], force=True)


class TestStderrIsTty:
    def test_stderr_is_tty_reflects_isatty(self, monkeypatch):
        monkeypatch.setattr(datasets.sys.stderr, "isatty", lambda: True)
        assert datasets.stderr_is_tty() is True
        monkeypatch.setattr(datasets.sys.stderr, "isatty", lambda: False)
        assert datasets.stderr_is_tty() is False
