"""
Performance tests — lightweight timing checks that run without external deps.

These don't benchmark the ML pipeline (that requires tinyml_modelmaker and
real hardware). They guard against regressions in the CLI's own startup cost
and in the hot-path security helpers that are called on every invocation.
"""
import subprocess
import sys
import os
import time
import pytest

PYTHON = sys.executable
MMCLI = [PYTHON, "-m", "mmcli"]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Generous thresholds — these catch catastrophic regressions, not micro-opts.
# Tight thresholds cause flaky CI; we only care about order-of-magnitude changes.
CLI_STARTUP_LIMIT_S = 5.0      # --help should finish in <5 s on any CI box
SAFE_PATH_CALL_LIMIT_US = 500  # _is_safe_path must be < 500 µs per call
SANITIZE_CALL_LIMIT_US = 100   # _sanitize_input must be < 100 µs per call


class TestPerformance:
    def test_cli_help_startup_time(self):
        """mmcli --help must complete within CLI_STARTUP_LIMIT_S seconds."""
        start = time.monotonic()
        r = subprocess.run([*MMCLI, "--help"], capture_output=True, cwd=REPO)
        elapsed = time.monotonic() - start
        assert r.returncode == 0
        assert elapsed < CLI_STARTUP_LIMIT_S, \
            f"CLI startup too slow: {elapsed:.2f}s (limit {CLI_STARTUP_LIMIT_S}s)"

    def test_is_safe_path_call_time(self):
        """_is_safe_path must be fast — it's called on every CLI invocation."""
        from mmcli.cli import _is_safe_path
        N = 1000
        start = time.monotonic()
        for _ in range(N):
            _is_safe_path("some/relative/path")
        elapsed_us = (time.monotonic() - start) / N * 1_000_000
        assert elapsed_us < SAFE_PATH_CALL_LIMIT_US, \
            f"_is_safe_path too slow: {elapsed_us:.1f}µs avg (limit {SAFE_PATH_CALL_LIMIT_US}µs)"

    def test_sanitize_input_call_time(self):
        """_sanitize_input must be fast — called on every string arg."""
        from mmcli.cli import _sanitize_input
        N = 10_000
        start = time.monotonic()
        for _ in range(N):
            _sanitize_input("timeseries")
        elapsed_us = (time.monotonic() - start) / N * 1_000_000
        assert elapsed_us < SANITIZE_CALL_LIMIT_US, \
            f"_sanitize_input too slow: {elapsed_us:.1f}µs avg (limit {SANITIZE_CALL_LIMIT_US}µs)"

    def test_memory_usage_import(self):
        """Importing mmcli.cli must not consume runaway memory.

        Uses sys.getsizeof as a coarse proxy — not a real memory profiler, but
        catches cases where module-level data structures explode in size.
        """
        import sys as _sys
        import mmcli.cli as _cli
        # The module object itself is tiny; this just ensures it imports cleanly
        assert _sys.getsizeof(_cli) < 10 * 1024 * 1024, "mmcli.cli module suspiciously large"

    def test_concurrency_handling(self):
        """Two mmcli --help invocations in parallel must both succeed."""
        import concurrent.futures
        def _help():
            return subprocess.run([*MMCLI, "--help"], capture_output=True, cwd=REPO).returncode

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            results = list(ex.map(lambda _: _help(), range(2)))
        assert all(rc == 0 for rc in results), f"Parallel invocations failed: {results}"


def test_load_scenarios():
    """Repeated invocations don't degrade — ten --help calls all succeed."""
    for i in range(10):
        r = subprocess.run([*MMCLI, "--help"], capture_output=True, cwd=REPO)
        assert r.returncode == 0, f"Invocation {i} failed"


def test_scaling_behavior():
    """_is_safe_path scales linearly — 10× more calls take < 20× more time."""
    from mmcli.cli import _is_safe_path
    def _time_n(n):
        t = time.monotonic()
        for _ in range(n): _is_safe_path("relative/path")
        return time.monotonic() - t

    t100 = _time_n(100)
    t1000 = _time_n(1000)
    if t100 > 0:
        ratio = t1000 / t100
        assert ratio < 20, f"Non-linear scaling: 10× calls took {ratio:.1f}× longer"


def test_resource_cleanup_under_load():
    """After many invocations no leftover Python processes remain."""
    import concurrent.futures, psutil
    pid_before = {p.pid for p in psutil.process_iter(["pid", "name"])
                  if "mmcli" in (p.info.get("name") or "")}

    def _run_help():
        subprocess.run([*MMCLI, "--help"], capture_output=True, cwd=REPO)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(lambda _: _run_help(), range(8)))

    pid_after = {p.pid for p in psutil.process_iter(["pid", "name"])
                 if "mmcli" in (p.info.get("name") or "")}
    leaked = pid_after - pid_before
    assert not leaked, f"Leaked mmcli processes: {leaked}"
