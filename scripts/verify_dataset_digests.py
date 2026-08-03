#!/usr/bin/env python3
"""verify_dataset_digests.py — GET-and-hash gate over every fetchable dataset.

This is a *content* check, not a reachability check (review finding F-3): a HEAD
request only proves a URL resolves, it says nothing about the bytes the
GitHub release mirror actually serves at that path. This script downloads the
full body of every registry entry that has a mirror asset (``dataset_url(name)
is not None``) and verifies the downloaded bytes against the registry's
recorded ``sha256``/``bytes`` — the one failure mode that breaks every user is
the mirror serving different bytes at the pinned release/version path than
whatever was hashed when the registry entry was written, and that is exactly
what a HEAD check cannot see.

It is driven through ``mmcli.datasets.fetch_dataset(name, force=True)`` rather
than a second, parallel downloader: that function is the exact code every
`mmcli datasets pull` and `init --dataset` invocation runs (timeouts, redirect
handling, digest comparison, atomic cache replace), so this gate proves the
code users actually run, not an approximation of it. ``fetch_dataset`` itself
raises ``RuntimeError`` naming both the expected and actual digest on a
mismatch, which is surfaced verbatim below as the FAIL line's detail.

Usage:
    python3 scripts/verify_dataset_digests.py
    python3 scripts/verify_dataset_digests.py --only fan_blade_fault

Exit status: 0 if every fetchable dataset's downloaded bytes match its
registry digest, non-zero otherwise (the specific FAIL lines name which
dataset(s) failed and why).

This script downloads real data from this project's own GitHub release
mirror (github.com/musicalplatypus/tinyml-cli/releases, tag
datasets-<version>; roughly 131 MB across all nine fetchable datasets, a few
minutes on a typical connection) into a throwaway cache directory — never the
user's real ``~/.cache/mmcli`` — and is safe to re-run at any time, e.g. after
a ``DATASETS_DEFAULT_VERSION`` bump / a re-mirror to a new release tag (see
docs/RELEASING.md).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        metavar="NAME",
        help="verify only this one registry entry, instead of every "
             "fetchable dataset (useful for iterating without "
             "re-downloading the other ~131 MB)",
    )
    args = parser.parse_args(argv)

    # The air-gap rule (REQ-DATA-03): fetch_dataset() refuses unconditionally
    # when MMCLI_DATASETS is set, even on an explicit call. If it were left
    # set here (e.g. inherited from a developer's shell), every dataset would
    # "fail" with a refusal message instead of being verified, and the gate
    # would report nothing meaningful. Unset it before importing mmcli.datasets
    # so the registry/URL logic below is unaffected by the caller's environment.
    os.environ.pop("MMCLI_DATASETS", None)

    # Route the version-scoped cache to a throwaway temp directory for the
    # duration of this run, so the gate never reads from or writes to the
    # user's real ~/.cache/mmcli — every dataset is downloaded fresh via
    # force=True regardless of what (if anything) is already cached there.
    tmp_cache_home = tempfile.mkdtemp(prefix="mmcli-digest-gate-")
    os.environ["XDG_CACHE_HOME"] = tmp_cache_home

    try:
        # Imported only after the environment above is set, so any
        # module-level cache-path computation sees the throwaway directory.
        from mmcli.datasets import DATASET_REGISTRY, dataset_url, fetch_dataset

        if args.only is not None and args.only not in DATASET_REGISTRY:
            available = ", ".join(sorted(DATASET_REGISTRY))
            print(
                f"Unknown dataset: {args.only!r}. Available: {available}",
                file=sys.stderr,
            )
            return 2

        names = [args.only] if args.only is not None else sorted(DATASET_REGISTRY)

        checked = 0
        failures: list[str] = []
        for name in names:
            url = dataset_url(name)
            if url is None:
                # No TI upstream (generic_audio_classification) — nothing to
                # fetch, and no network request should be made for it.
                continue
            checked += 1
            print(f"{name}: GET {url} ...", file=sys.stderr, flush=True)
            try:
                fetch_dataset(name, force=True)
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"{name}: {url}: FAIL — {exc}")
                failures.append(name)
                continue
            print(f"{name}: {url}: PASS")

        print()
        if failures:
            print(
                f"{len(failures)}/{checked} fetchable dataset(s) FAILED: "
                f"{', '.join(failures)}"
            )
            return 1
        if checked == 0:
            if args.only is not None:
                # --only named a real registry entry (validated above), but
                # it has no mirror asset (dataset_url() is None — bundled-only,
                # e.g. generic_audio_classification). Reporting the
                # whole-registry message here (10-REVIEW.md IN-01) is
                # misleading when nine other entries plainly are fetchable.
                print(
                    f"{args.only!r} has no mirror asset and is not fetchable "
                    f"(bundled-only).",
                    file=sys.stderr,
                )
                return 2
            print("No fetchable datasets found in DATASET_REGISTRY.", file=sys.stderr)
            return 1
        print(f"All {checked} fetchable dataset(s) PASSED.")
        return 0
    finally:
        shutil.rmtree(tmp_cache_home, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
