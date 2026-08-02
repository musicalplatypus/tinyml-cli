#!/usr/bin/env python3
"""release_preflight.py — D-05 scripted release preflight.

docs/RELEASING.md's ordering rule — publish the mirror release, verify 9/9
digests, *then* build and ship — has a real failure mode if it is reversed:
a release built against an unpublished or wrongly-tagged mirror ships a
binary whose every `mmcli init --dataset`/`datasets pull` 404s for its
users. This script is the enforcement, not the description of it. Run it
before `bash build_macos.sh` / `bash build_linux.sh` / `build_windows.ps1`
on a release build.

Two checks, run in the order a release actually depends on:

  1. **Mirror reachability + tag correctness** — `gh release view <tag>
     --json tagName,assets` against this project's own GitHub release. Cheap
     (no payload downloaded); the same check release.yml's
     `mirror-healthcheck` CI job runs, reused here rather than
     reimplemented, so a maintainer running this locally sees the identical
     failure CI would report. Requires the `gh` CLI on PATH and
     authenticated (`gh auth status`).
  2. **Digest verification** — invokes `scripts/verify_dataset_digests.py`
     as a subprocess (not reimplemented inline), which performs the real
     GET-and-hash gate over all nine fetchable datasets via
     `mmcli.datasets.fetch_dataset(name, force=True)` — the same function
     every real `mmcli datasets pull` runs. Downloads ~131 MB into a
     throwaway cache; several minutes on a typical connection.

Either check failing means "do not build yet" — see docs/RELEASING.md for
what each failure means and how to fix it.

Usage:
    python3 scripts/release_preflight.py
    python3 scripts/release_preflight.py --skip-digests   # tag/asset check only, fast iteration

Exit status: 0 if both checks pass (or the requested subset does), non-zero
otherwise.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

REPO = "musicalplatypus/tinyml-cli"


def check_mirror_tag_and_assets() -> bool:
    """Verify the mirror release named by DATASETS_DEFAULT_VERSION exists,
    is tagged exactly as expected, and carries every fetchable dataset's
    asset at a non-zero size. No payload is downloaded — see
    release.yml's `mirror-healthcheck` job, which this mirrors.
    """
    from mmcli.datasets import (
        DATASET_REGISTRY,
        DATASETS_DEFAULT_VERSION,
        DATASETS_MIRROR_TAG_PREFIX,
    )

    tag = f"{DATASETS_MIRROR_TAG_PREFIX}{DATASETS_DEFAULT_VERSION}"
    expected = sorted(
        entry["filename"] for entry in DATASET_REGISTRY.values() if entry.get("ti_name")
    )

    print(f"[1/2] Checking mirror release '{tag}' in {REPO} ...", file=sys.stderr)

    result = subprocess.run(
        ["gh", "release", "view", tag, "--repo", REPO, "--json", "tagName,assets"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            f"FATAL: mirror release '{tag}' not reachable via `gh release view` "
            f"(missing release, or a mis-tagged/forgotten --latest=false publish?): "
            f"{result.stderr.strip()}",
            file=sys.stderr,
        )
        return False

    data = json.loads(result.stdout)
    if data.get("tagName") != tag:
        print(
            f"FATAL: gh reported tagName {data.get('tagName')!r}, expected {tag!r} — "
            f"the mirror release appears mis-tagged",
            file=sys.stderr,
        )
        return False

    assets = {a["name"]: a.get("size", 0) for a in data.get("assets", [])}
    missing = [name for name in expected if name not in assets]
    if missing:
        print(
            f"FATAL: mirror release '{tag}' is missing expected asset(s): {missing}",
            file=sys.stderr,
        )
        return False

    zero_size = [name for name in expected if assets[name] <= 0]
    if zero_size:
        print(
            f"FATAL: mirror release '{tag}' has zero-byte asset(s): {zero_size}",
            file=sys.stderr,
        )
        return False

    print(
        f"OK: mirror release '{tag}' has all {len(expected)} expected assets, "
        f"all non-zero size (no payload downloaded).",
        file=sys.stderr,
    )
    return True


def check_digests() -> bool:
    """Run scripts/verify_dataset_digests.py as a subprocess (the real
    GET-and-hash gate), streaming its output live rather than capturing it,
    since a multi-minute, multi-hundred-megabyte download with no visible
    progress looks identical to a hang.
    """
    print("[2/2] Running scripts/verify_dataset_digests.py (full digest gate) ...", file=sys.stderr)
    result = subprocess.run([sys.executable, "scripts/verify_dataset_digests.py"])
    return result.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--skip-digests",
        action="store_true",
        help="Only check mirror reachability/tagging, skip the ~131 MB digest gate "
             "(fast iteration; do NOT skip this before an actual release build).",
    )
    args = parser.parse_args(argv)

    tag_ok = check_mirror_tag_and_assets()
    if not tag_ok:
        print(
            "\nPREFLIGHT FAILED at step 1/2 (mirror tag/assets). Do not proceed to build. "
            "See docs/RELEASING.md.",
            file=sys.stderr,
        )
        return 1

    if args.skip_digests:
        print(
            "\nPREFLIGHT PARTIAL: mirror tag/assets OK, digest gate skipped (--skip-digests). "
            "Run without --skip-digests before an actual release build.",
            file=sys.stderr,
        )
        return 0

    digests_ok = check_digests()
    if not digests_ok:
        print(
            "\nPREFLIGHT FAILED at step 2/2 (digest verification). Do not proceed to build. "
            "See docs/RELEASING.md.",
            file=sys.stderr,
        )
        return 1

    print("\nPREFLIGHT PASSED: mirror tag/assets OK, all fetchable digests verified. Safe to build.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
