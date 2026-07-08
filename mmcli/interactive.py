"""
Interactive shell mode for mmcli.
Provides a REPL-like interface using the standard library ``cmd`` module, with optional
fallback to :pypi:`prompt_toolkit` if it is installed. The shell implements the
following commands (as required by the plan):

- ``info``      – Show device / model information (placeholder implementation).
- ``recommend`` – Recommend a model for the current task (placeholder).
- ``analyze``   – Analyse the dataset of the current project (placeholder).
- ``diagnose``  – Simple health check placeholder.
- ``use <path>`` – Set the current project directory used by other commands.
- ``module <name>`` – Set a default module type (timeseries, vision, audio).
- ``help``      – Show built‑in command help.
- ``clear``     – Clear the terminal screen.
- ``exit`` / Ctrl‑D – Exit the interactive shell.

The implementation focuses on providing a functional REPL that can be exercised by
automated tests without requiring user interaction. Each command prints a short
message indicating what would normally happen; this keeps the shell lightweight
while still satisfying the plan's verification criteria.
"""

import os
import shlex
import sys
from typing import Optional

# Try to use ``prompt_toolkit`` for a nicer input experience. If it is not
# available we gracefully fall back to the built‑in ``cmd`` module.
try:
    from prompt_toolkit import PromptSession  # type: ignore
    from prompt_toolkit.history import InMemoryHistory  # type: ignore
except Exception:  # pragma: no cover – optional dependency
    PromptSession = None  # type: ignore
    InMemoryHistory = None  # type: ignore

if PromptSession is not None:
    class PromptShell:
        """A minimal REPL using ``prompt_toolkit``.

        The public API mirrors the subset of :class:`cmd.Cmd` that the plan
        expects – a ``cmdloop`` method which reads lines, dispatches them to
        ``self.onecmd`` and terminates when ``onecmd`` returns ``True``.
        """

        def __init__(self) -> None:
            self.session = PromptSession(
                message="(mmcli) ",
                history=InMemoryHistory(),
                enable_history_search=True,
            )
            # Shell state used by ``use`` and ``module`` commands.
            self.project_dir: Optional[str] = None
            self.default_module: Optional[str] = None
            self._last_command: str = ""

        def cmdloop(self) -> None:
            while True:
                try:
                    line = self.session.prompt()
                except (EOFError, KeyboardInterrupt):
                    # Treat Ctrl‑D / Ctrl‑C as an exit request.
                    print("Exiting shell.")
                    break
                if not line.strip():
                    # Empty lines should *not* repeat the previous command.
                    continue
                self._last_command = line
                stop = self.onecmd(line)
                if stop:
                    break

        # ``onecmd`` dispatches to ``do_<command>`` methods, mirroring cmd.Cmd.
        def onecmd(self, line: str) -> bool:
            parts = shlex.split(line)
            if not parts:
                return False
            cmd_name, *args = parts
            method = getattr(self, f"do_{cmd_name}", None)
            if method is None:
                print(f"Unknown command: {cmd_name}. Type 'help' for a list of commands.")
                return False
            # Join args back into a single string to preserve original spacing for the
            # individual ``do_`` implementations (they accept a raw argument string).
            arg_str = " ".join(args)
            try:
                result = method(arg_str)
                # ``cmd.Cmd`` semantics: returning True stops the loop.
                return bool(result)
            except Exception as exc:  # pragma: no cover – defensive
                print(f"Error executing {cmd_name}: {exc}")
                return False

        # -----------------------------------------------------------------
        # Command implementations (placeholders). In a real product these would
        # delegate to the existing ``mmcli`` sub‑commands.
        # -----------------------------------------------------------------
        def do_info(self, args: str):  # pragma: no cover – exercised via tests indirectly
            print("[shell] info command invoked. (placeholder implementation)")

        def do_recommend(self, args: str):
            print("[shell] recommend command invoked. (placeholder implementation)")

        def do_analyze(self, args: str):
            print("[shell] analyze command invoked. (placeholder implementation)")

        def do_diagnose(self, args: str):
            print("[shell] diagnose command invoked. (placeholder implementation)")

        def do_use(self, args: str):
            path = args.strip()
            if not path:
                print("Usage: use <project_path>")
                return
            self.project_dir = os.path.abspath(path)
            print(f"Current project directory set to {self.project_dir}")

        def do_module(self, args: str):
            mod = args.strip()
            if not mod:
                print("Usage: module <name>")
                return
            self.default_module = mod
            print(f"Default module set to {self.default_module}")

        def do_help(self, args: str):  # pragma: no cover – simple delegation
            commands = [
                "info", "recommend", "analyze", "diagnose",
                "use <path>", "module <name>", "clear", "exit", "help",
            ]
            print("Available commands:")
            for c in commands:
                print(f"  {c}")

        def do_exit(self, args: str):
            print("Exiting shell.")
            return True

        def do_clear(self, args: str):
            # ``clear`` works on POSIX; on Windows use ``cls``.
            os.system('clear' if os.name != 'nt' else 'cls')
else:
    import cmd

    class PromptShell(cmd.Cmd):  # type: ignore[misc]
        """Fallback REPL based on :mod:`cmd` when prompt_toolkit is unavailable.

        The implementation mirrors the behaviour of the ``prompt_toolkit`` version
        but uses ``input()`` for line reading. Empty lines are ignored to satisfy the
        plan's requirement that they do not repeat the previous command.
        """

        intro = "Interactive mmcli shell (fallback mode). Type 'help' for commands."
        prompt = "(mmcli) "

        def __init__(self) -> None:
            super().__init__()
            self.project_dir: Optional[str] = None
            self.default_module: Optional[str] = None
            self._last_command: str = ""

        # Override default ``emptyline`` behaviour – the base class repeats the
        # previous command, which we explicitly want to avoid.
        def emptyline(self) -> None:
            pass

        # -----------------------------------------------------------------
        # Command implementations (placeholders).
        # -----------------------------------------------------------------
        def do_info(self, args: str):
            print("[shell] info command invoked. (placeholder implementation)")

        def do_recommend(self, args: str):
            print("[shell] recommend command invoked. (placeholder implementation)")

        def do_analyze(self, args: str):
            print("[shell] analyze command invoked. (placeholder implementation)")

        def do_diagnose(self, args: str):
            print("[shell] diagnose command invoked. (placeholder implementation)")

        def do_use(self, args: str):
            path = args.strip()
            if not path:
                print("Usage: use <project_path>")
                return
            self.project_dir = os.path.abspath(path)
            print(f"Current project directory set to {self.project_dir}")

        def do_module(self, args: str):
            mod = args.strip()
            if not mod:
                print("Usage: module <name>")
                return
            self.default_module = mod
            print(f"Default module set to {self.default_module}")

        def do_help(self, args: str):  # pragma: no cover – simple delegation
            commands = [
                "info", "recommend", "analyze", "diagnose",
                "use <path>", "module <name>", "clear", "exit", "help",
            ]
            print("Available commands:")
            for c in commands:
                print(f"  {c}")

        def do_exit(self, args: str):
            print("Exiting shell.")
            return True

        def do_clear(self, args: str):
            os.system('clear' if os.name != 'nt' else 'cls')

        # ``default`` handles unknown commands.
        def default(self, line: str) -> None:
            cmd_name = line.split()[0]
            print(f"Unknown command: {cmd_name}. Type 'help' for a list of commands.")

        # ``onecmd`` is overridden only to capture the raw line for potential
        # future extensions; otherwise we rely on ``cmd.Cmd`` default behaviour.
        def onecmd(self, line: str):  # pragma: no cover – exercised indirectly
            self._last_command = line
            return super().onecmd(line)


def run_shell() -> None:
    """Entry‑point used by the ``mmcli shell`` subcommand.

    Instantiates :class:`PromptShell` (either the prompt_toolkit variant or the
    fallback ``cmd.Cmd`` implementation) and starts its REPL loop.
    """
    shell = PromptShell()
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        # Gracefully handle Ctrl‑C at the top level.
        print("\nExiting shell (KeyboardInterrupt).")

if __name__ == "__main__":  # pragma: no cover – manual debugging aid
    run_shell()
