"""
Progress utilities for mmcli commands.

Provides a simple wrapper to execute a subprocess while displaying a tqdm
progress bar. The progress is line‑based (each stdout/stderr line increments the
bar). This is sufficient for long‑running commands such as ``mmcli train`` or
``mmcli compile`` where output streams continuously during the operation.
"""

import os
import subprocess
from typing import List

try:
    # tqdm may not be installed at runtime; the plan adds it as a dependency.
    from tqdm import tqdm
except ImportError:  # pragma: no cover – should never happen after adding tqdm
    tqdm = None


def run_with_progress(command: List[str], description: str = "Running") -> int:
    """Execute *command* showing a terminal progress bar.

    Parameters
    ----------
    command: list of str
        The executable and its arguments, e.g. ``[python, script.py, cfg]``.
    description: str, optional
        Text displayed next to the tqdm bar.

    Returns
    -------
    int
        Return code from the subprocess.
    """
    if tqdm is None:
        # Fallback – just run without a progress bar.
        result = subprocess.run(command)
        return result.returncode

    # Start the process with stdout+stderr merged so we can count lines uniformly.
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line buffered
    )

    # Use tqdm without a known total – it will grow indefinitely.
    with tqdm(total=None, desc=description, unit="line", dynamic_ncols=True) as pbar:
        if proc.stdout is None:
            return 1
        for line in proc.stdout:
            # Echo the subprocess output so the user sees normal logs.
            print(line, end="", flush=True)
            pbar.update(1)
    proc.wait()
    return proc.returncode
