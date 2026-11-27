"""Terminal UI: theme colours, panels, tables and progress bars, plus line input.

Rich rendering is used only for the app's own screens (menus, briefings, results,
debriefs) and only on a colour terminal. Simulated tool output is never styled
here; it stays exactly as the real tools print it. Without colour (pipes, tests,
NO_COLOR) every helper returns plain text.
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..core.shell import Context
from .console import LabCompleter, LineReader, TimeUp

ACCENT = "green"  # GreenRay brand colour
# softer than the terminal's own green, yellow and red, which glare on a dark background
LEVEL_STYLE = {"easy": "#7ee787", "medium": "#e3b341", "hard": "#f47067"}
PASSED, STARTED, OPEN = "✔", "◐", "·"
MARK_STYLE = {PASSED: "#7ee787", STARTED: "#e3b341"}


class UI:
    def __init__(self, color: bool = True, history_file: Path | None = None):
        if os.name == "nt":
            os.system("")  # enables ANSI escape codes in the Windows console
        self.interactive = sys.stdin.isatty() and sys.stdout.isatty()
        self.color = color and self.interactive and not os.environ.get("NO_COLOR")
        self.history_file = history_file
        self._reader: LineReader | None = None
        self._prompt_timeout: float | None = None  # set by ask_until while its prompt is open

    def set_completer(self, words: Iterable[str], context: Callable[[], Context | None]) -> None:
        """Tab completion for the lab shell; no-op without a terminal."""
        if self.interactive:
            self._line_reader().completer = LabCompleter(words, context)

    @contextmanager
    def private_history(self):
        """Keep exam answers out of the persistent history."""
        saved, self._reader = self._reader, LineReader(None) if self.interactive else None
        try:
            yield
        finally:
            self._reader = saved

    def _line_reader(self) -> LineReader:
        if self._reader is None:
            self._reader = LineReader(self.history_file)
        return self._reader

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

    def level(self, difficulty: str, s: str) -> str:
        """Easy green, medium yellow, hard red: the one colour code used for labs everywhere."""
        return {"easy": self.green, "medium": self.yellow, "hard": self.red}.get(difficulty, self.bold)(s)

    def blue(self, s: str) -> str:
        return self._c("94", s)

    def cyan(self, s: str) -> str:
        return self._c("36", s)

    # ------------------------------------------------------------ rich screens (strings)

    def _render(self, renderable) -> str:
        width = min(100, shutil.get_terminal_size((100, 24)).columns)
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, width=width, highlight=False, legacy_windows=False)
        console.print(renderable)
        return buffer.getvalue().rstrip("\n")

    def banner(self, version: str, tagline: str) -> str:
        if not self.color:
            return f"GreenRay AIO {version}: {tagline}"
        mark = Text()
        mark.append(" ◢◤ ", style=f"bold {ACCENT}")
        mark.append("GreenRay", style=f"bold {ACCENT}")
        mark.append(" AIO", style="bold")
        mark.append(f"   v{version}\n", style="dim")
        mark.append(f"    {tagline}", style="italic")
        return self._render(Panel(mark, border_style=ACCENT, box=box.ROUNDED, expand=False, padding=(0, 2)))

    def panel(self, body: str, title: str = "", subtitle: str = "", style: str = ACCENT) -> str:
        if not self.color:
            head = f"=== {title} ===" + (f"   {subtitle}" if subtitle else "")
            return (head + "\n" if title else "") + body
        return self._render(
            Panel(
                Text.from_ansi(body),
                title=f"[bold]{title}[/]" if title else None,
                subtitle=subtitle or None,
                border_style=style,
                box=box.ROUNDED,
                expand=True,
                padding=(0, 1),
            )
        )

    def tiles(self, items: list[tuple[str, str, str]]) -> str:
        """A row of small panels, each a label over a figure: (label, figure, colour of the figure)."""
        if not self.color:
            return "   ".join(f"{label}: {figure}" for label, figure, _ in items)
        grid = Table.grid(expand=True, padding=(0, 1))
        cells = []
        for label, figure, colour in items:
            grid.add_column(ratio=1)
            body = Text()
            body.append(label + "\n", style="dim")
            body.append_text(Text.from_ansi(figure, style=f"bold {colour}".strip()))
            cells.append(Panel(body, box=box.ROUNDED, border_style="dim", padding=(0, 1)))
        grid.add_row(*cells)
        return self._render(grid)

    def lab_panel(self, title: str, note: str, rows: list[tuple[str, str, str, str]]) -> str:
        """One exam domain as a panel of labs: (mark, code, name, level) per row."""
        if not self.color:
            lines = [f"{title}  ({note})"]
            lines += [f"  {mark} {code:<7} {name:<36} {level}" for mark, code, name, level in rows]
            return "\n".join(lines)
        t = Table(box=None, expand=True, show_edge=False, pad_edge=False, header_style="dim", padding=(0, 1))
        t.add_column("ID", width=7, no_wrap=True, style="dim")
        t.add_column("LAB", ratio=1, no_wrap=True, overflow="ellipsis")
        t.add_column("LEVEL", width=7, no_wrap=True)
        t.add_column("", width=1)
        for mark, code, name, level in rows:
            done = Text(mark, style=MARK_STYLE.get(mark, "dim"))
            t.add_row(code, name, Text(level, style=LEVEL_STYLE[level]), done)
        return self._render(
            Panel(
                t,
                title=f"[bold]{title}[/]",
                title_align="left",
                subtitle=f"[dim]{note}[/]",
                subtitle_align="right",
                border_style="dim",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

    def table(self, headers: list[str], rows: list[list[str]], styles: list[str] | None = None) -> str:
        if not self.color:
            return "\n".join("  " + "  ".join(r) for r in rows)
        t = Table(*headers, box=box.SIMPLE_HEAD, header_style="bold", show_edge=False, pad_edge=False)
        for i, r in enumerate(rows):
            t.add_row(*(Text.from_ansi(c) for c in r), style=(styles[i] if styles else None))
        return self._render(t)

    def bar(self, done: int, total: int, width: int = 20) -> str:
        filled = round(width * done / total) if total else 0
        if not self.color:
            return "#" * filled + "-" * (width - filled)
        return f"\033[32m{'█' * filled}\033[2m{'░' * (width - filled)}\033[0m"

    def out(self, s: str = "") -> None:
        print(s)

    def ask_until(self, prompt: str, deadline: float, typed: bool = False) -> str:
        """Like ``ask``, for a prompt on an exam clock: raises TimeUp instead of returning a late line.

        On a terminal the prompt closes by itself when the deadline passes. Where input cannot be
        interrupted (a pipe), the line is refused once it arrives, so nothing typed late ever counts.
        """
        left = deadline - time.time()
        if left <= 0:
            raise TimeUp
        # the line still comes from ask(), so a UI that replaces ask() (a test's scripted input) is honoured
        self._prompt_timeout = left
        try:
            line = self.ask(prompt, typed)
        finally:
            self._prompt_timeout = None
        if time.time() >= deadline:
            raise TimeUp
        return line

    def ask(self, prompt: str, typed: bool = False) -> str:
        """Read a line; EOF (Ctrl-D / end of piped input) reads as ``quit``. ``typed`` colours the input."""
        try:
            if self.interactive:
                return self._line_reader().read(prompt, typed, timeout=self._prompt_timeout)
            return input(prompt)
        except EOFError:
            return "quit"
