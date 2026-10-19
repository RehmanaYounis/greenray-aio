"""Minimal terminal UI: coloured output and line input (works on Windows and Linux)."""

from __future__ import annotations

import os
import sys


class UI:
    def __init__(self, color: bool = True):
        if os.name == "nt":
            os.system("")  # enables ANSI escape codes in the Windows console
        self.color = color and sys.stdout.isatty() and not os.environ.get("NO_COLOR")

    def _c(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.color else s

    def bold(self, s: str) -> str:
        return self._c("1", s)

    def dim(self, s: str) -> str:
        return self._c("2", s)

    def red(self, s: str) -> str:
        return self._c("31", s)

    def green(self, s: str) -> str:
        return self._c("32", s)

    def yellow(self, s: str) -> str:
        return self._c("33", s)

    def cyan(self, s: str) -> str:
        return self._c("36", s)

    def out(self, s: str = "") -> None:
        print(s)

    def ask(self, prompt: str) -> str:
        """Read a line; EOF (Ctrl-D / end of piped input) reads as ``quit``."""
        try:
            return input(prompt)
        except EOFError:
            return "quit"


def enable_line_editing(words: list[str]) -> None:
    """Arrow-key history and tab completion where GNU readline exists (Linux, macOS, WSL)."""
    try:
        import readline
    except ImportError:
        return

    def complete(text: str, i: int):
        matches = [w for w in words if w.startswith(text)]
        return matches[i] if i < len(matches) else None

    readline.set_completer(complete)
    readline.parse_and_bind("tab: complete")
