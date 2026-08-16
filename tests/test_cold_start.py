"""
Pins the cold-start fix for `_detect_training_device()` (Phase 14, REQ-COLD-01/02).

_detect_training_device() shells out to `system_profiler SPDisplaysDataType`
on macOS (~0.87s per call, uncached) to pick a training backend (mps/cuda/
cpu). Before this fix it ran unconditionally while building the argparse
tree -- once in main() and once per subparser that calls
_add_training_args() (train, run) -- so a single `mmcli --version` invocation
triggered it 3 times for a value `--version` never even looks at: 2.60s
spent detecting a training device just to print a version string.

These tests assert *call count*, not wall-clock time, on purpose: a timing
assertion is flaky on a loaded CI/dev machine and becomes exactly the kind
of test people learn to ignore or skip. Counting calls to
_detect_training_device() is a deterministic, machine-independent way to
catch a regression to eager/repeated detection -- e.g. someone reintroducing
`detected = _detect_training_device()` at module-import time or inside an
argparse default= expression, which is invisible in a code review that only
skims for "does this still work" and not "does this still run 3 times".

Non-vacuousness: verified manually (not part of this file, since the point
of this file is to detect *this specific* regression, not to reintroduce
it) by swapping in the pre-fix implementation of `mmcli/cli.py` (which
called `_detect_training_device()` unconditionally in `main()` and in
`_add_training_args()`, invoked from both the `train` and `run`
subparsers) and confirming `test_version_never_detects_training_device`
failed with `actual_calls == 3`, matching the measured figure in the plan
objective. See the plan's SUMMARY for the exact procedure and output.
"""
import sys

import pytest

from mmcli import cli


def _install_counting_wrapper(monkeypatch):
    """Wrap cli._detect_training_device() to record how many times it is
    actually invoked, while still delegating to the real implementation
    (so its own internal process-lifetime cache and return value are
    unaffected -- we're only instrumenting the call site, not replacing
    the detection logic itself)."""
    calls = {"n": 0}
    real_detect = cli._detect_training_device

    def counting_wrapper():
        calls["n"] += 1
        return real_detect()

    monkeypatch.setattr(cli, "_detect_training_device", counting_wrapper)
    return calls


class TestColdStartDeviceDetection:
    """Pins REQ-COLD-01/02: detection must not run eagerly for commands
    that never consume a training device."""

    def test_version_never_detects_training_device(self, monkeypatch):
        """`mmcli --version` must not call _detect_training_device() at
        all: argparse's built-in --version action exits during parsing,
        before main() would ever reach the point where a training device
        is resolved (train/run only). This is the exact invocation the
        plan's objective measured at 3 calls / 2.60s before this fix."""
        calls = _install_counting_wrapper(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["mmcli", "--version"])

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 0
        assert calls["n"] == 0, (
            f"_detect_training_device() was called {calls['n']} time(s) for "
            "`mmcli --version`; expected 0 (fully lazy -- --version never "
            "needs a training device)."
        )

    def test_train_subcommand_help_never_detects_training_device(self, monkeypatch):
        """`mmcli train --help` must not detect either: it prints the
        `train` subparser's own help (built once, eagerly, alongside every
        other subparser regardless of which command runs) and exits before
        any command body executes. Detection is display-text-only in the
        top-level description (see next test) and consumption-only in
        build_config() -- neither is reached here."""
        calls = _install_counting_wrapper(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["mmcli", "train", "--help"])

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 0
        assert calls["n"] == 0, (
            f"_detect_training_device() was called {calls['n']} time(s) for "
            "`mmcli train --help`; expected 0."
        )

    def test_top_level_help_detects_at_most_once(self, monkeypatch):
        """`mmcli --help` is the one path that legitimately needs the
        value (it prints a "Detected training backend: <device>" line in
        the top-level description), so it is allowed exactly one call --
        never the pre-fix 3."""
        calls = _install_counting_wrapper(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["mmcli", "--help"])

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 0
        assert calls["n"] <= 1, (
            f"_detect_training_device() was called {calls['n']} time(s) for "
            "`mmcli --help`; expected at most 1."
        )
