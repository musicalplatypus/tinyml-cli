"""
Dataset registry and project initialisation for mmcli.

Provides:
  - DATASET_REGISTRY — mapping of dataset names → metadata
  - list_datasets()  — query available datasets, optionally filtered by task
  - extract_dataset() — unzip an example dataset into a new project directory
  - dataset_url()    — compose the version-pinned GitHub release-mirror URL
                        for a dataset (see mirror_facts/D-A in 10-03-PLAN.md)
  - fetch_dataset()  — download, verify (sha256) and cache a mirrored dataset zip
  - stderr_is_tty()  — shared TTY predicate: gates the tqdm progress bar here
                        and the `init --dataset` auto-fetch policy in the CLI
                        (see 10-06-PLAN.md decision D-5) — one predicate, not two

REQ-DATA-02 invariant, enforced at import time (see _validate_registry below):
every entry that carries a ``ti_name`` (i.e. every dataset fetchable from the
mirror) MUST also carry a 64-hex-character ``sha256`` and a positive
``bytes``. A fetchable dataset without a recorded digest is a configuration
error and must fail at import, not halfway through a multi-megabyte download.
If you add a new fetchable entry, you must add both fields or the module will
refuse to import.
"""

import errno
import hashlib
import os
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile

try:
    # tqdm is a declared dependency (requirements.txt); guarded for the rare
    # environment where it is absent so importing this module never breaks.
    from tqdm import tqdm
except ImportError:  # pragma: no cover - should not happen after adding tqdm
    tqdm = None

# ---------------------------------------------------------------------------
# GitHub release-asset mirror (10-03-PLAN.md D-A/D-B; supersedes the TI
# fetch source from D-1/D-3/D-4 in 10-RESEARCH.md, which is no longer
# reachable — see 10-03-SUMMARY-attempt1-blocked.md)
# ---------------------------------------------------------------------------

# Base for this project's own public GitHub Release assets. dataset_url()
# composes {DATASETS_MIRROR_BASE}/{DATASETS_MIRROR_TAG_PREFIX}{version}/{filename}.
DATASETS_MIRROR_BASE = "https://github.com/musicalplatypus/tinyml-cli/releases/download"
DATASETS_MIRROR_TAG_PREFIX = "datasets-"

# Labels the mirror release/payload version (the release tag is
# datasets-<version>), NOT a TI engine version any more (D-B). This is still
# the version axis datasets are pinned to, and it is still part of the
# on-disk cache path (see _cache_dir): changing it changes the cache key, so
# bumping the pinned version can never silently reuse a dataset fetched under
# an older mirror release. An individual registry entry may override this
# with its own `ti_version` key (kept as the field name; it now overrides the
# mirror payload version, not a TI version).
DATASETS_DEFAULT_VERSION = "01_03_00"

# Socket timeout (seconds), applied to both connect and each read, so a hung
# or slow-drip server fails instead of stalling forever. Tests override this
# via monkeypatch to keep the timeout test fast.
DOWNLOAD_TIMEOUT_SECONDS = 30


def stderr_is_tty() -> bool:
    """Shared predicate: is stderr an interactive terminal right now?

    Used here to decide whether `fetch_dataset` shows a tqdm progress bar,
    and reused verbatim (imported, not reimplemented) by the CLI's
    `init --dataset` auto-fetch policy (10-06-PLAN.md decision D-5): if we
    cannot show progress, we must not start an unnarrated multi-megabyte
    transfer. Keeping this in one place means progress and permission are
    answers to the same question rather than two separate heuristics that
    could drift apart.
    """
    return sys.stderr.isatty()


# ---------------------------------------------------------------------------
# Where example zips are stored
# ---------------------------------------------------------------------------

def _datasets_dir() -> str:
    """Return the directory that holds example dataset zips.

    Priority:
      1. MMCLI_DATASETS env var
      2. mmcli/example_datasets/ (bundled with the package)

    This function's behaviour is unchanged by the fetch/cache mechanism added
    alongside it — `_resolve_dataset_zip()` wraps this rather than replacing
    it, so any existing caller of `_datasets_dir()` keeps working exactly as
    before.
    """
    env = os.environ.get("MMCLI_DATASETS")
    if env and os.path.isdir(env):
        return env
    return os.path.join(os.path.dirname(__file__), "example_datasets")


def _cache_dir_path(version: str) -> str:
    """Return the version-scoped cache directory path **without touching the
    filesystem**.

    Split out from `_cache_dir` so that answering "where would this live"
    stays a pure computation. Inspection paths — `cache_entry_path`,
    `cache_entry_size`, and therefore every `datasets list` — must not create
    a directory as a side effect of being asked a question, and must not fail
    on a read-only or unwritable cache home when no download was requested.
    """
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(base, "mmcli", "datasets", version)


def _cache_dir(version: str) -> str:
    """Return the version-scoped cache directory for downloaded datasets,
    creating it (mode 0700) if it does not already exist.

    Use `_cache_dir_path` instead when only the path is needed: this function
    is for the download flow, which is the one caller that legitimately needs
    the directory to exist.

    Honours XDG_CACHE_HOME, falling back to ~/.cache. Resolves to
    ``<cache-home>/mmcli/datasets/<version>/``.

    The version is part of the path *deliberately*: a flat, version-less
    cache would let bumping DATASETS_DEFAULT_VERSION (or an entry's
    ti_version override) silently reuse a dataset downloaded under an older
    mirror release — exactly the failure D-3 (10-RESEARCH.md, and D-B in
    10-03-PLAN.md) exists to prevent. Two versions therefore always cache
    independently.
    """
    path = _cache_dir_path(version)
    os.makedirs(path, exist_ok=True)
    try:
        # os.makedirs's mode= argument is masked by umask, so set the
        # permission explicitly rather than relying on it.
        os.chmod(path, 0o700)
    except OSError:  # pragma: no cover - best effort, e.g. on some CI runners
        pass
    return path


def _sha256_of(path: str) -> str:
    """Return the hex sha256 digest of the file at *path*, read in chunks."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Registry — add entries as you add zips to example_datasets/
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, dict] = {
    "generic_timeseries_classification": {
        "filename": "generic_timeseries_classification.zip",
        "task_types": ["generic_timeseries_classification"],
        "module": "timeseries",
        "description": "Synthetic waveform classification (sawtooth, sine, square)",
        "ti_name": "generic_timeseries_classification.zip",
        "sha256": "7b2c0980bb30c3bc661004d66373d7ea35ea13ab5b6f8b74f5182c3bc6bc09c1",
        "bytes": 2579940,
    },
    "generic_timeseries_regression": {
        "filename": "generic_timeseries_regression.zip",
        "task_types": ["generic_timeseries_regression"],
        "module": "timeseries",
        "description": "Synthetic timeseries regression dataset",
        "ti_name": "generic_timeseries_regression.zip",
        "sha256": "078d212b00112bcaca4b1bb68b871e8c24eb3ed809b610d64642a74a7854cc23",
        "bytes": 906660,
    },
    "generic_timeseries_anomalydetection": {
        "filename": "generic_timeseries_anomalydetection.zip",
        "task_types": ["generic_timeseries_anomalydetection"],
        "module": "timeseries",
        "description": "Synthetic anomaly detection (amplitude/frequency shifts)",
        "ti_name": "generic_timeseries_anomalydetection.zip",
        "sha256": "7cb2f9fd183fa5c6730abdd0a144e1ce57f7ece9ed93d8663b19a983cde6d6b5",
        "bytes": 4242845,
    },
    "generic_timeseries_forecasting": {
        "filename": "generic_timeseries_forecasting.zip",
        "task_types": ["generic_timeseries_forecasting"],
        "module": "timeseries",
        "description": "Simulated thermostat temperature forecasting",
        "ti_name": "generic_timeseries_forecasting.zip",
        "sha256": "4ae6e7e436817a8ee5f3e528e70741b9c6fabfeb6c19a9fdb321dabad0a804ce",
        "bytes": 71053,
    },
    "arc_fault_classification": {
        "filename": "arc_fault_classification.zip",
        "task_types": ["arc_fault"],
        "module": "timeseries",
        "description": "DC arc fault current classification (DSI sensor)",
        "ti_name": "arc_fault_classification_dsi.zip",
        "sha256": "bcee7b54fb42079bfac1f4a39266fb836c2ef73c3f8fffd8fa04c41671f7656e",
        "bytes": 13290076,
    },
    "ecg_classification": {
        "filename": "ecg_classification.zip",
        "task_types": ["ecg_classification", "generic_timeseries_classification"],
        "module": "timeseries",
        "description": "ECG 2-class heartbeat classification (normal vs abnormal)",
        "ti_name": "ecg_classification_2class.zip",
        "sha256": "881ac26e95378eca9c1979cf1c70a8d1b8f2cb73da65e264a03bf1849c6addc6",
        "bytes": 4651662,
    },
    "fan_blade_fault": {
        "filename": "fan_blade_fault.zip",
        "task_types": ["motor_fault"],
        "module": "timeseries",
        "description": "Fan blade fault classification (vibration data, 3-axis)",
        "ti_name": "fan_blade_fault_dsi.zip",
        "sha256": "5194925e0f97387a54be989923ec34bef8e65e03fe21652552d7bbcdc21a959e",
        "bytes": 56595859,
    },
    "pir_detection": {
        "filename": "pir_detection.zip",
        "task_types": ["pir_detection"],
        "module": "timeseries",
        "description": "PIR motion detection classification (human vs non-human)",
        "ti_name": "pir_detection_classification_dsk.zip",
        "sha256": "d75470c9ba7f56fd4e8801c9f10424262e9935513b9011f55f5f5ed406ae0b0e",
        "bytes": 1579936,
    },
    "mnist_image_classification": {
        "filename": "mnist_image_classification.zip",
        "task_types": ["image_classification"],
        "module": "vision",
        "description": "MNIST handwritten digit classification (28×28 images)",
        "ti_name": "mnist_classes.zip",
        "sha256": "7fa4be9944a364074dc796d5d802dad8f1636f2f4daa6fd735d15f5fe05f3db8",
        "bytes": 46993516,
    },
    "generic_audio_classification": {
        "filename": "generic_audio_classification.zip",
        "task_types": ["audio_classification"],
        "module": "audio",
        "description": "Synthetic 2-class audio (yes/no) — 16kHz sine-wave WAV files",
        # No ti_name: this set is locally authored (be06559) and has no
        # upstream mirror asset, so it stays bundled and is never fetched
        # (D-2, 10-03-PLAN.md). It still carries sha256/bytes for
        # completeness and future integrity checks.
        "sha256": "dfc463e6a0aac80b2db36770e9fc56090f319d400d416b391d160d70382dbc5d",
        "bytes": 18371,
    },
}


def _validate_registry(registry: dict) -> None:
    """Enforce REQ-DATA-02 at import time: every entry with a ``ti_name``
    (i.e. every dataset that can be fetched from the GitHub release mirror)
    must carry a valid 64-hex-character ``sha256`` and a positive ``bytes``.
    Raises ValueError naming the offending entry rather than letting a
    fetchable-but-undigested dataset surface as a runtime surprise partway
    through a download.
    """
    hex_digits = set("0123456789abcdef")
    for name, meta in registry.items():
        if not meta.get("ti_name"):
            continue
        sha256 = meta.get("sha256", "")
        size = meta.get("bytes", 0)
        valid_sha256 = (
            isinstance(sha256, str)
            and len(sha256) == 64
            and set(sha256.lower()) <= hex_digits
        )
        valid_bytes = isinstance(size, int) and size > 0
        if not (valid_sha256 and valid_bytes):
            raise ValueError(
                f"DATASET_REGISTRY['{name}'] has a ti_name "
                f"({meta['ti_name']!r}) but is missing a valid sha256/bytes "
                f"pair (REQ-DATA-02). Every entry that can be fetched from "
                f"the mirror must carry a 64-hex-character sha256 and a "
                f"positive byte count, checked at import so a missing "
                f"digest is a configuration error, not a runtime surprise."
            )


_validate_registry(DATASET_REGISTRY)


def dataset_url(name: str) -> str | None:
    """Return the version-pathed GitHub release-mirror download URL for
    *name*, or ``None`` when the entry has no ``ti_name`` (i.e. it is locally
    authored and bundled only, such as ``generic_audio_classification``).

    Looks the name up through ``DATASET_REGISTRY`` — never composes a URL
    from a caller-supplied string directly — so an unknown name raises
    ``KeyError`` instead of ever reaching URL construction.

    Composes ``{DATASETS_MIRROR_BASE}/{DATASETS_MIRROR_TAG_PREFIX}{version}/
    {meta['filename']}`` (10-03-PLAN.md D-A): the asset is named by the
    entry's LOCAL ``filename`` — the on-disk zip name and the cache filename
    — not by ``ti_name``. ``ti_name`` is no longer the URL source; it is kept
    purely as the fetchable-sentinel (an entry with no ``ti_name`` has no
    mirror asset and is bundled-only) and as provenance recording which
    original TI asset the mirrored bytes came from (D-D). The GitHub release
    download URL is stable and pinnable; it 302-redirects to a signed,
    time-limited ``release-assets.githubusercontent.com`` URL that cannot be
    pinned (see ``_HostLockedRedirectHandler`` / ``ALLOWED_CROSS_HOST_REDIRECTS``
    below for the one redirect hop this module follows).
    """
    meta = DATASET_REGISTRY[name]  # KeyError on unknown name, deliberately
    ti_name = meta.get("ti_name")
    if ti_name is None:
        return None
    version = meta.get("ti_version") or DATASETS_DEFAULT_VERSION
    return (
        f"{DATASETS_MIRROR_BASE}/{DATASETS_MIRROR_TAG_PREFIX}{version}/"
        f"{meta['filename']}"
    )


def _resolve_dataset_zip(name: str) -> str | None:
    """Resolve the on-disk path for *name*'s zip, or ``None`` if it is not
    available anywhere (the caller — `extract_dataset` or the CLI — decides
    whether to fetch it).

    Resolution order:
      1. MMCLI_DATASETS env var (existing offline/air-gap escape hatch)
      2. bundled example_datasets/ (existing, wraps `_datasets_dir()`)
      3. ~/.cache/mmcli/datasets/<version>/ (new — previously downloaded)

    The cache sits *below* the env var and the bundled directory
    deliberately: a user who pointed MMCLI_DATASETS at an air-gapped or
    reproducible dataset directory must never be silently overridden by a
    cached copy, and a bundled zip (shipped with the package) is preferred
    over a cache entry for the same reason existing behaviour did not
    consult a cache at all.

    Digest verification is asymmetric on purpose: files resolved through
    MMCLI_DATASETS are **not** digest-checked, because that directory is
    explicitly user-managed and may legitimately hold a locally prepared
    substitute dataset (REQ-DATA-03). Cache hits **are** re-verified against
    the registry sha256 on every resolution, not only at download time,
    because the cache directory may be writable by another process or user
    that can plant a file there — re-hashing costs tens of milliseconds
    against a download that costs seconds. A cache file that fails
    verification is treated as absent (not returned) and reported to
    stderr, never used.
    """
    meta = DATASET_REGISTRY.get(name)
    if meta is None:
        return None
    filename = meta["filename"]

    # Steps 1-2: MMCLI_DATASETS env var, then bundled example_datasets/.
    # _datasets_dir() already implements exactly this precedence as a single
    # directory choice; we look inside it for the specific file rather than
    # replacing its logic.
    primary_dir = _datasets_dir()
    candidate = os.path.join(primary_dir, filename)
    if os.path.isfile(candidate):
        return candidate

    env = os.environ.get("MMCLI_DATASETS")
    if env and os.path.isdir(env):
        # MMCLI_DATASETS is set and is a real directory, but does not hold
        # this file. It is the user's authoritative, explicitly configured
        # source — do not fall through to the cache, which would silently
        # override an air-gapped setup with a network-fetched copy.
        return None

    # Step 3: version-scoped cache, digest-verified on every hit.
    # `_cache_dir_path`, not `_cache_dir`: resolution is a read-only question and
    # runs for every dataset on every `datasets list`. Creating the cache
    # directory here made a pure listing write to disk, and made it fail on an
    # unwritable cache home even when no download had been requested. Only the
    # download flow creates.
    version = meta.get("ti_version") or DATASETS_DEFAULT_VERSION
    cache_path = os.path.join(_cache_dir_path(version), filename)
    if os.path.isfile(cache_path):
        expected = meta.get("sha256")
        if expected and _sha256_of(cache_path) == expected:
            return cache_path
        print(
            f"WARNING: cached copy of '{name}' at {cache_path} does not "
            f"match the recorded sha256; treating it as absent. Run "
            f"'mmcli datasets pull {name} --force' to redownload.",
            file=sys.stderr,
        )
        return None

    return None


def cache_entry_path(name: str) -> str:
    """Return the version-scoped cache path *name*'s zip would occupy,
    regardless of whether a file currently exists there.

    Deliberately **not** `_resolve_dataset_zip`: that function answers "where
    does this dataset currently resolve from" and can return a path inside
    the primary directory (bundled, or the user's own `MMCLI_DATASETS`
    directory) — precisely the paths `datasets remove` (10-09-PLAN.md) must
    never touch. This function only ever answers "where would the cache
    entry be", so the source keeps those two questions structurally separate
    rather than relying on a caller to filter the resolved path correctly.

    Raises ``KeyError`` on an unknown name, same as `dataset_url`.
    """
    meta = DATASET_REGISTRY[name]  # KeyError on unknown name, deliberately
    version = meta.get("ti_version") or DATASETS_DEFAULT_VERSION
    # `_cache_dir_path`, not `_cache_dir`: asking where a cache entry would be
    # must not create the directory. `datasets list` calls this for every
    # dataset, and a pure listing has no business writing to the filesystem.
    return os.path.join(_cache_dir_path(version), meta["filename"])


def cache_entry_size(name: str) -> int | None:
    """Return the on-disk size in bytes of *name*'s cache entry if one
    exists, or ``None`` if it does not.

    This is independent of `_dataset_state` / resolution: a dataset can
    resolve as ``bundled`` (packaged copy, or a file in the user's own
    `MMCLI_DATASETS` directory) while *also* holding a stale cache entry from
    an earlier download, since the packaged copy wins resolution. That disk
    usage is otherwise invisible and unreclaimable — see CONTEXT.md D-10.
    Uses the actual file size on disk, not the registry's recorded `bytes`,
    so a partially-written or since-corrupted entry still reports a truthful
    reclaimable size.
    """
    path = cache_entry_path(name)
    try:
        return os.path.getsize(path)
    except OSError:
        return None


# Closed allowlist of permitted cross-host redirect PAIRS, keyed on the
# original request host (10-03-PLAN.md D-C; amends 10-02's T-10-02-01/05).
# GitHub release-asset downloads at github.com/.../releases/download/...
# 302-redirect to a signed, time-limited release-assets.githubusercontent.com
# URL — that one hop is verified and deliberately allowed. Every other
# cross-host redirect, from any host, still raises. Exact host-string
# equality only: NEVER a suffix/endswith/wildcard match on
# "githubusercontent.com", so a lookalike host such as
# "release-assets.githubusercontent.com.evil.com" or
# "evil-githubusercontent.com" is still refused. sha256 verification of the
# downloaded bytes remains mandatory and is the real integrity guarantee;
# this allowlist only relaxes the defence-in-depth host lock for one
# verified redirect pair.
ALLOWED_CROSS_HOST_REDIRECTS: dict[str, frozenset[str]] = {
    "github.com": frozenset({"release-assets.githubusercontent.com"}),
}


class _HostLockedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow redirects only when the target host matches the original host,
    or is explicitly allowlisted for that original host.

    A cross-host redirect is refused rather than followed silently: the
    sha256 check would still catch substituted content, but an unexplained
    redirect to a different host is worth surfacing as an error in its own
    right (T-10-02-01/05). The one exception is the closed, exact-host pair
    in ALLOWED_CROSS_HOST_REDIRECTS (D-C, 10-03-PLAN.md): the GitHub
    release-asset redirect to its signed-URL host. Every other cross-host
    redirect is still refused, unchanged from 10-02.
    """

    def __init__(self, allowed_host: str, dataset_name: str):
        self._allowed_host = allowed_host
        self._dataset_name = dataset_name

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parts = urllib.parse.urlparse(newurl)
        original_scheme = urllib.parse.urlparse(req.full_url).scheme
        # 10-REVIEW.md WR-05: the host lock alone permits a scheme downgrade
        # (http/ftp) — urllib's build_opener() installs FTPHandler/FileHandler
        # by default and HTTPRedirectHandler itself permits http/https/ftp
        # targets. sha256 verification still catches substituted content, but
        # a silent transport downgrade to plaintext (or ftp) should not be
        # followed at all. ftp is never permitted regardless of the original
        # scheme. An http *original* scheme only occurs via direct low-level
        # test calls to _download_to_cache against the local http test
        # server — fetch_dataset() enforces HTTPS-only on the initial URL for
        # every real production call — so http->http is tolerated here (it
        # is not a downgrade) while https->http/ftp is always refused.
        if parts.scheme not in ("http", "https") or (
            original_scheme == "https" and parts.scheme != "https"
        ):
            raise RuntimeError(
                f"Refusing non-HTTPS redirect while fetching "
                f"'{self._dataset_name}': {req.full_url} -> {newurl}"
            )
        new_host = parts.hostname
        if new_host == self._allowed_host:
            return super().redirect_request(req, fp, code, msg, headers, newurl)
        allowed_targets = ALLOWED_CROSS_HOST_REDIRECTS.get(
            self._allowed_host, frozenset()
        )
        if new_host in allowed_targets:
            return super().redirect_request(req, fp, code, msg, headers, newurl)
        raise RuntimeError(
            f"Refusing cross-host redirect while fetching "
            f"'{self._dataset_name}': {req.full_url} -> {newurl}"
        )


def _download_to_cache(url: str, cache_dir: str, filename: str,
                       expected_sha256: str, expected_bytes: int,
                       name: str) -> str:
    """Low-level atomic download + verify, shared by `fetch_dataset`.

    Does not enforce a URL scheme itself — `fetch_dataset` is the only place
    that decides HTTPS is mandatory, which keeps this function testable
    against a plain local `http.server` fixture instead of a real mirror TLS
    endpoint.

    Downloads to a temp file **inside** *cache_dir*, hashing while
    streaming, then verifies before an atomic `os.replace()` onto the final
    path. Same-directory placement keeps the rename atomic on the same
    filesystem; a partial file landing at the final path would become a
    poisoned cache hit on the very next run, with no way for that run to
    tell the difference.

    Raises RuntimeError on: oversized/truncated body, checksum mismatch,
    cross-host redirect, non-2xx HTTP status (404 names the URL and this
    dataset), an out-of-space write (ENOSPC, named plainly rather than
    echoing the raw errno string), or any lower-level urllib/socket failure.
    The temp file is always removed on any failure path; nothing is left on
    disk "for debugging".
    """
    dest_path = os.path.join(cache_dir, filename)
    # 1% of the expected size or 1 KiB, whichever is larger — enough slack
    # for legitimate framing overhead, not enough to let a hostile or
    # misconfigured server fill the disk before the digest check runs.
    tolerance = max(1024, int(expected_bytes * 0.01))
    allowed_host = urllib.parse.urlparse(url).hostname

    fd, tmp_path = tempfile.mkstemp(prefix=".fetch-", suffix=".part", dir=cache_dir)
    os.close(fd)
    try:
        opener = urllib.request.build_opener(
            _HostLockedRedirectHandler(allowed_host, name)
        )
        request = urllib.request.Request(
            url, headers={"User-Agent": "mmcli-datasets/1"}
        )
        try:
            response = opener.open(request, timeout=DOWNLOAD_TIMEOUT_SECONDS)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise RuntimeError(
                    f"HTTP 404 fetching '{name}': {url}"
                ) from exc
            raise RuntimeError(
                f"HTTP {exc.code} fetching '{name}': {url}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Failed to fetch '{name}' from {url}: {exc}"
            ) from exc
        except OSError as exc:
            # Connect/read timeouts (socket.timeout is OSError, and is not
            # always wrapped as URLError by urllib -- e.g. a timeout while
            # reading the response status line surfaces as a raw OSError)
            # so a hung server fails loudly here rather than hanging the
            # caller or escaping as an unrelated exception type.
            raise RuntimeError(
                f"Failed to fetch '{name}' from {url}: {exc}"
            ) from exc

        with response:
            content_length_hdr = response.headers.get("Content-Length")
            declared_length = None
            if content_length_hdr is not None:
                try:
                    declared_length = int(content_length_hdr)
                except ValueError:
                    declared_length = None
            if declared_length is not None and declared_length > expected_bytes + tolerance:
                raise RuntimeError(
                    f"Refusing to fetch '{name}': server-advertised "
                    f"Content-Length {declared_length} exceeds the "
                    f"registry size {expected_bytes} by more than the "
                    f"{tolerance}-byte tolerance."
                )

            hasher = hashlib.sha256()
            total = 0
            show_progress = tqdm is not None and stderr_is_tty()
            bar = (
                tqdm(total=expected_bytes, unit="B", unit_scale=True, desc=name)
                if show_progress else None
            )
            try:
                # The `with` is inside its own try/except OSError as well as the
                # per-write guard below: writes are buffered, so the final flush
                # happens when the block exits, and an ENOSPC there escapes the
                # inner handler entirely. Without this, a disk that fills on the
                # last partial buffer produces a bare traceback — the exact
                # failure the inner handler was added to prevent.
                with open(tmp_path, "wb") as out:
                    while True:
                        chunk = response.read(65536)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > expected_bytes + tolerance:
                            raise RuntimeError(
                                f"Aborting fetch of '{name}': streamed "
                                f"{total} bytes, more than the registry "
                                f"size {expected_bytes} plus the "
                                f"{tolerance}-byte tolerance. The server "
                                f"may be hostile or misconfigured."
                            )
                        try:
                            out.write(chunk)
                        except OSError as exc:
                            if exc.errno == errno.ENOSPC:
                                raise RuntimeError(
                                    f"Not enough free disk space to write "
                                    f"'{name}' to {cache_dir}: the disk is "
                                    f"full. Free up space and retry "
                                    f"'mmcli datasets pull {name}'."
                                ) from exc
                            raise RuntimeError(
                                f"Failed writing '{name}' to {cache_dir}: "
                                f"{exc}"
                            ) from exc
                        hasher.update(chunk)
                        if bar is not None:
                            bar.update(len(chunk))
            except OSError as exc:
                # Only reachable for an OSError raised outside the per-write
                # guard — in practice the buffered flush on close. Same
                # translation, so the user sees one message whichever byte the
                # disk ran out on.
                if exc.errno == errno.ENOSPC:
                    raise RuntimeError(
                        f"Not enough free disk space to write '{name}' to "
                        f"{cache_dir}: the disk is full. Free up space and "
                        f"retry 'mmcli datasets pull {name}'."
                    ) from exc
                raise RuntimeError(
                    f"Failed writing '{name}' to {cache_dir}: {exc}"
                ) from exc
            finally:
                if bar is not None:
                    bar.close()

            if declared_length is not None and total < declared_length:
                raise RuntimeError(
                    f"Truncated download of '{name}': server declared "
                    f"Content-Length {declared_length} but only {total} "
                    f"bytes were received."
                )

        actual_sha256 = hasher.hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Checksum mismatch for '{name}': expected "
                f"{expected_sha256}, got {actual_sha256}. URL: {url}. "
                f"The temp file has been removed; nothing was cached."
            )

        os.replace(tmp_path, dest_path)
        return dest_path
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def fetch_dataset(name: str, *, force: bool = False) -> str:
    """Download, verify (sha256) and cache the mirrored dataset zip for *name*.

    Returns the path to the verified, cached file. If a correctly-verified
    copy is already cached and *force* is False, returns it immediately
    without issuing any network request.

    Raises
    ------
    KeyError
        *name* is not in DATASET_REGISTRY.
    RuntimeError
        MMCLI_DATASETS is set (REQ-DATA-03 — this variable signals a
        managed/air-gapped environment and disables fetching unconditionally,
        even on this explicit call), *name* has no ti_name (nothing to fetch
        — it is bundled-only), the composed URL is not HTTPS, or the
        download/verification fails for any of the reasons documented on
        `_download_to_cache`.
    """
    meta = DATASET_REGISTRY[name]  # KeyError on unknown name, deliberately

    env = os.environ.get("MMCLI_DATASETS")
    if env:
        raise RuntimeError(
            f"Refusing to fetch '{name}': MMCLI_DATASETS is set to "
            f"'{env}'. That variable signals a managed or air-gapped "
            f"environment; a tool that fetches anyway would silently "
            f"defeat it. Unset MMCLI_DATASETS to allow fetching, or place "
            f"'{meta['filename']}' in that directory yourself."
        )

    url = dataset_url(name)
    if url is None:
        raise RuntimeError(
            f"'{name}' has no upstream source and cannot be fetched — it "
            f"is bundled with mmcli. See `mmcli datasets path {name}`."
        )
    if not url.startswith("https://"):
        raise RuntimeError(
            f"Refusing to fetch non-HTTPS dataset URL for '{name}': {url}"
        )

    version = meta.get("ti_version") or DATASETS_DEFAULT_VERSION
    cache_dir = _cache_dir(version)
    dest_path = os.path.join(cache_dir, meta["filename"])

    if not force and os.path.isfile(dest_path):
        if _sha256_of(dest_path) == meta["sha256"]:
            return dest_path
        # Stale or corrupted cache entry: fall through and redownload rather
        # than serving bad bytes.
        try:
            os.unlink(dest_path)
        except OSError as exc:
            # 10-REVIEW.md WR-08: os.unlink can raise (read-only cache,
            # permissions, a Windows sharing violation) — this docstring
            # promises RuntimeError on every failure mode, and the CLI
            # callers only catch (KeyError, RuntimeError), not OSError.
            raise RuntimeError(
                f"Cached copy of '{name}' at {dest_path} failed verification "
                f"and could not be removed: {exc}"
            ) from exc

    return _download_to_cache(
        url, cache_dir, meta["filename"], meta["sha256"], meta["bytes"], name
    )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def list_datasets(task_type: str | None = None,
                  module: str | None = None) -> list[dict]:
    """Return datasets matching the optional *task_type* and *module* filters.

    Each returned dict has keys: name, filename, task_types, module, description.
    """
    results = []
    for name, meta in DATASET_REGISTRY.items():
        if task_type and task_type not in meta.get("task_types", []):
            continue
        if module and meta.get("module") != module:
            continue
        results.append({"name": name, **meta})
    return results


def print_dataset_list(task_type: str | None = None,
                       module: str | None = None) -> None:
    """Print a formatted table of available datasets.

    Used by ``mmcli init --list`` to show the user what datasets are
    available before they create a project.
    """
    datasets = list_datasets(task_type=task_type, module=module)

    if not datasets:
        filters = []
        if task_type:
            filters.append(f"task={task_type}")
        if module:
            filters.append(f"module={module}")
        print(f"No datasets found matching: {', '.join(filters)}")
        return

    # Column widths (minimum padding)
    max_name = max(len(d["name"]) for d in datasets)
    max_tasks = max(len(", ".join(d["task_types"])) for d in datasets)
    max_mod = max(len(d.get("module", "")) for d in datasets)

    # Header
    print("\nAvailable example datasets:\n")
    hdr = (f" {'Dataset':<{max_name}}  {'Task Types':<{max_tasks}}  "
           f"{'Module':<{max_mod}}  Description")
    print(hdr)
    print("─" * len(hdr))

    for d in datasets:
        tasks_str = ", ".join(d["task_types"])
        print(f" {d['name']:<{max_name}}  {tasks_str:<{max_tasks}}  "
              f"{d.get('module', ''):<{max_mod}}  {d.get('description', '')}")

    print(f"\n{len(datasets)} dataset(s) available. Create a project with:")
    print("  mmcli init -t TASK_TYPE --dataset DATASET -p ./my_project\n")


def get_dataset(name: str) -> dict | None:
    """Look up a single dataset by name. Returns None if not found."""
    meta = DATASET_REGISTRY.get(name)
    if meta is None:
        return None
    return {"name": name, **meta}


def extract_dataset(dataset_name: str, project_path: str,
                    task_type: str | None = None) -> None:
    """Create *project_path* and extract the named dataset into it.

    The TI example dataset zips contain ``classes/`` (or ``files/``) and
    ``annotations/`` at the zip root.  We extract into
    ``<project_path>/dataset/`` so that the resulting project matches the
    layout expected by ``mmcli train``.

    Raises
    ------
    SystemExit
        On any error (unknown dataset, incompatible task, zip not found,
        target directory already exists).
    """
    meta = DATASET_REGISTRY.get(dataset_name)
    if meta is None:
        available = ", ".join(sorted(DATASET_REGISTRY.keys())) or "(none)"
        print(
            f"ERROR: Unknown dataset '{dataset_name}'.\n"
            f"Available datasets: {available}",
            file=sys.stderr,
        )
        sys.exit(2)

    # Validate task compatibility
    if task_type and task_type not in meta.get("task_types", []):
        compatible = ", ".join(meta["task_types"])
        print(
            f"ERROR: Dataset '{dataset_name}' is not compatible with "
            f"task '{task_type}'.\n"
            f"Compatible tasks: {compatible}",
            file=sys.stderr,
        )
        sys.exit(2)

    # Resolve paths
    project_path = os.path.abspath(project_path)
    if os.path.exists(project_path):
        print(
            f"ERROR: Project directory already exists: {project_path}\n"
            "Choose a different name or remove the existing directory.",
            file=sys.stderr,
        )
        sys.exit(2)

    zip_path = _resolve_dataset_zip(dataset_name)
    if zip_path is None:
        hint = (
            f"Run `mmcli datasets pull {dataset_name}` to fetch it, "
            if meta.get("ti_name")
            else ""
        )
        print(
            f"ERROR: Dataset zip not found for '{dataset_name}'.\n"
            f"{hint}or place '{meta['filename']}' in the datasets "
            f"directory:\n  {_datasets_dir()}",
            file=sys.stderr,
        )
        sys.exit(2)

    # Create project directory and extract into dataset/ subdirectory
    dataset_dir = os.path.join(project_path, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Explicit containment guard: extract member-by-member (rather than
            # a single extractall()) and confirm each resulting real path stays
            # under dataset_dir, instead of relying solely on zipfile's own
            # member-path sanitisation (correct in current CPython but
            # undocumented as a stability guarantee — 10-REVIEW.md CR-02). This
            # checks zf.extract()'s own return value (the path it actually wrote
            # to) rather than re-deriving the target with a different path
            # resolution, which would disagree with zipfile's arcname filtering
            # and false-positive on members zipfile already neutralises safely
            # (e.g. leading '../' segments are stripped, not resolved).
            root = os.path.realpath(dataset_dir)
            for member in zf.namelist():
                extracted_path = zf.extract(member, dataset_dir)
                real_extracted = os.path.realpath(extracted_path)
                if not (real_extracted == root or real_extracted.startswith(root + os.sep)):
                    if os.path.isfile(real_extracted):
                        os.remove(real_extracted)
                    print(
                        f"ERROR: refusing to extract '{member}' from "
                        f"'{zip_path}': it escaped to {real_extracted}",
                        file=sys.stderr,
                    )
                    sys.exit(2)
    except zipfile.BadZipFile:
        print(
            f"ERROR: '{zip_path}' is not a valid zip file.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Verify expected subdirectories
    data_subdirs = ["classes", "files", "images"]
    has_data = any(
        os.path.isdir(os.path.join(dataset_dir, d)) for d in data_subdirs
    )
    if not has_data:
        print(
            f"WARNING: The extracted dataset does not contain any of "
            f"{', '.join(data_subdirs)}.  The project may not be usable "
            f"with 'mmcli train' as-is.",
            file=sys.stderr,
        )

    print(f"✓ Project created: {project_path}")
    print(f"  Dataset: {dataset_name} — {meta.get('description', '')}")
    print()
    print("Next steps:")
    print(f"  mmcli train -m {meta.get('module', 'MODULE')} "
          f"-t {(task_type or meta['task_types'][0])} "
          f"-d DEVICE -n MODEL -i {project_path}")

