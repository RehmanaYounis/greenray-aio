"""Line editing for the lab prompt: history, Ctrl-C/Ctrl-D as in bash, tab completion.

Built on prompt_toolkit so it behaves the same on Windows, WSL, macOS and Linux.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory, InMemoryHistory

from ..core.shell import Context

# Commands whose first argument is a host rather than a file.
HOST_COMMANDS = {"ssh", "scp", "ping"}


class LabCompleter(Completer):
    """Completes commands at the start of a line, then hosts or simulated paths.

    Paths come from the simulated node the learner is on, relative to its working
    directory, so ``cat gr<TAB>`` after ``cd /etc/slurm`` works as on a real box.
    """

    def __init__(self, words: Iterable[str], context: Callable[[], Context | None]):
        self.words = sorted(set(words))
        self.context = context

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        text = document.text_before_cursor
        word = document.get_word_before_cursor(WORD=True)
        tokens = text.split()
        at_start = len(tokens) == 0 or (len(tokens) == 1 and not text.endswith(" "))
        if at_start:
            yield from self._matches(self.words, word)
            return
        ctx = self.context()
        if ctx is None:
            return
        if tokens[0] in HOST_COMMANDS and (len(tokens) == 1 or (len(tokens) == 2 and not text.endswith(" "))):
            yield from self._matches(sorted(ctx.state.nodes), word)
            return
        yield from self._paths(ctx, word)

    @staticmethod
    def _matches(candidates: Iterable[str], word: str):
        for c in candidates:
            if c.startswith(word):
                yield Completion(c, start_position=-len(word))

    def _paths(self, ctx: Context, word: str):
        from ..commands.system import known_dirs

        base, slash, partial = word.rpartition("/")
        directory = ctx.abspath(base or "/") if slash else ctx.cwd
        prefix = directory.rstrip("/") + "/"
        dirs = known_dirs(ctx)
        names: set[str] = set()
        for path in [*ctx.node.files, *dirs]:
            if not path.startswith(prefix) or path == prefix:
                continue
            head, _, rest = path[len(prefix) :].partition("/")
            names.add(head + "/" if rest or path in dirs else head)
        for name in sorted(names):
            if name.startswith(partial):
                yield Completion(base + slash + name, start_position=-len(word), display=name)


class LineReader:
    """A prompt_toolkit session shared by one UI; swap the completer per lab."""

    def __init__(self, history_file: Path | None):
        history = FileHistory(str(history_file)) if history_file else InMemoryHistory()
        self.session: PromptSession = PromptSession(history=history, enable_history_search=True)
        self.completer: Completer | None = None

    def read(self, prompt: str) -> str:
        return self.session.prompt(ANSI(prompt), completer=self.completer, complete_while_typing=False)
