import subprocess
import sys

def test_shell_exit():
    """Run ``mmcli shell`` and ensure it exits cleanly when given ``exit``.

    The interactive shell prints a prompt ``(mmcli) ``. We feed the command
    ``exit`` via stdin and verify that the process terminates with exit code 0
    and that the prompt appears in the captured stdout.
    """
    # Use the same Python interpreter that launched this test runner.
    cmd = [sys.executable, "-m", "mmcli", "shell"]
    result = subprocess.run(
        cmd,
        input=b"exit\n",
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0
    # The prompt should be printed before the ``exit`` command is processed.
    assert b"(mmcli) " in result.stdout
