"""A small bash-like shell that runs simulated commands against the cluster state.

Supports: `;` and `&&` chaining, `|` pipes, `>` / `>>` redirects, `cat > f <<EOF`
heredocs, `export VAR=val`, `sudo` (ignored, you are root), `ssh <host>` / `exit`.
"""

from __future__ import annotations

import pickle
import posixpath
from collections.abc import Callable
from dataclasses import dataclass, field

from .. import commands
from .errors import CommandError
from .state import HEAD, AttrDict

HEREDOC_PROMPT = "> "
INTERNAL_ERROR_CODE = 70  # EX_SOFTWARE
INCONSISTENT_FLAG = "_internal_error"  # set in state.flags when a failed command could not be rolled back


@dataclass
class Context:
    state: AttrDict
    host: str = HEAD
    host_stack: list = field(default_factory=list)
    saved_cwd: dict = field(default_factory=dict)  # ssh depth -> working directory to restore on exit
    stdin: str = ""
    cwd: str = "/root"
    history: list = field(default_factory=list)  # raw command lines entered by the user
    executed: list = field(default_factory=list)  # (host, argv) for every simple command run
    reader: Callable[[str], str | None] | None = (
        None  # prompt -> line; lets interactive commands (editors) read from the learner
    )
    color: bool = False  # the learner's terminal supports colour (set by the UI, never in tests)
    tty: bool = False  # the current command writes straight to that terminal (not a pipe or redirect)

    @property
    def node(self):
        return self.state.nodes[self.host]

    @property
    def colorize(self) -> bool:
        """True when a command should colour its output, like real tools do on a terminal."""
        return self.color and self.tty

    def abspath(self, path: str) -> str:
        """Resolve ``path`` like the shell would: ``~`` is /root, relative paths start at the cwd."""
        if path == "~" or path.startswith("~/"):
            path = "/root" + path[1:]
        if not path.startswith("/"):
            path = posixpath.join(self.cwd, path)
        return posixpath.normpath(path)

    def ran(self, *prefix: str, host: str | None = None) -> bool:
        """True if a command starting with ``prefix`` was executed (optionally on ``host``)."""
        for h, argv in self.executed:
            if host and h != host:
                continue
            if tuple(argv[: len(prefix)]) == prefix:
                return True
        return False


def _split_top(line: str, seps: tuple[str, ...]) -> list[tuple[str, str]]:
    """Split on separators outside quotes and ``$(...)``. Returns [(segment, separator_before)]."""
    out, buf, quote, prev_sep, depth = [], [], None, "", 0
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            elif ch == "\\" and quote == '"' and i + 1 < len(line):
                buf.append(line[i + 1])
                i += 1
            i += 1
            continue
        if ch == "\\" and i + 1 < len(line):
            buf.append(line[i : i + 2])
            i += 2
            continue
        if ch in "'\"":
            quote = ch
        elif line.startswith("$(", i):
            depth += 1
            buf.append("$(")
            i += 2
            continue
        elif ch == ")" and depth:
            depth -= 1
        elif not depth:
            matched = next((s for s in seps if line.startswith(s, i)), None)
            if matched:
                out.append(("".join(buf).strip(), prev_sep))
                buf, prev_sep = [], matched
                i += len(matched)
                continue
        buf.append(ch)
        i += 1
    out.append(("".join(buf).strip(), prev_sep))
    return out


_NAME_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"


def tokenize(text: str, lookup, substitute) -> list[str]:
    """Split ``text`` into words the way bash does for simple commands.

    Single quotes are literal. Outside them, ``$VAR``/``${VAR}`` expand through
    ``lookup(name)`` (unset -> empty) and ``$(cmd)`` through ``substitute(cmd)``.
    A word that is only an unquoted expansion which comes out empty is dropped.
    """
    words: list[str] = []
    buf: list[str] = []
    in_word = False
    quoted_part = False
    quote = None
    i, n = 0, len(text)

    def expand_at(j: int) -> tuple[str, int]:
        if text.startswith("$(", j):
            depth, k = 1, j + 2
            while k < n and depth:
                if text.startswith("$(", k):
                    depth += 1
                    k += 1
                elif text[k] == ")":
                    depth -= 1
                k += 1
            if depth:
                raise ValueError("unexpected EOF while looking for matching `)'")
            return substitute(text[j + 2 : k - 1]).rstrip("\n"), k
        if text.startswith("${", j):
            end = text.find("}", j)
            if end < 0:
                raise ValueError("bad substitution")
            return lookup(text[j + 2 : end]), end + 1
        k = j + 1
        while k < n and text[k] in _NAME_CHARS:
            k += 1
        if k == j + 1:
            return "$", k
        return lookup(text[j + 1 : k]), k

    while i < n:
        ch = text[i]
        if quote == "'":
            if ch == "'":
                quote = None
            else:
                buf.append(ch)
            i += 1
            continue
        if quote == '"':
            if ch == '"':
                quote = None
                i += 1
            elif ch == "\\" and i + 1 < n and text[i + 1] in '"\\$`':
                buf.append(text[i + 1])
                i += 2
            elif ch == "$":
                val, i = expand_at(i)
                buf.append(val)
            else:
                buf.append(ch)
                i += 1
            continue
        if ch in " \t\n":
            if in_word and ("".join(buf) or quoted_part):
                words.append("".join(buf))
            buf, in_word, quoted_part = [], False, False
            i += 1
            continue
        in_word = True
        if ch in "'\"":
            quote = ch
            quoted_part = True
            i += 1
        elif ch == "\\" and i + 1 < n:
            buf.append(text[i + 1])
            quoted_part = True
            i += 2
        elif ch == "$":
            val, i = expand_at(i)
            parts = val.split()  # unquoted expansions are word-split
            if val[:1].isspace() and (buf or quoted_part):
                words.append("".join(buf))
                buf, quoted_part = [], False
            for k, part in enumerate(parts):
                if k:
                    words.append("".join(buf))
                    buf, quoted_part = [], False
                buf.append(part)
            if parts and val[-1:].isspace():
                words.append("".join(buf))
                buf, in_word, quoted_part = [], False, False
        else:
            buf.append(ch)
            i += 1
    if quote:
        raise ValueError(f"unexpected EOF while looking for matching `{quote}'")
    if in_word and ("".join(buf) or quoted_part):
        words.append("".join(buf))
    return words


class Shell:
    def __init__(self, state, host: str = HEAD, color: bool = False):
        self.ctx = Context(state=state, host=host, color=color)

    def _lookup(self, name: str) -> str:
        if name == "HOSTNAME":
            return self.ctx.host
        if name == "?":
            return "0"
        return self.ctx.node.env.get(name, "")

    def _substitute(self, command: str) -> str:
        stdin = self.ctx.stdin
        out = self.run(command, record=False)
        self.ctx.stdin = stdin
        return out

    def _display_cwd(self) -> str:
        cwd = self.ctx.cwd
        return "~" + cwd[len("/root") :] if cwd == "/root" or cwd.startswith("/root/") else cwd

    @property
    def prompt(self) -> str:
        return f"root@{self.ctx.host}:{self._display_cwd()}# "

    def styled_prompt(self) -> str:
        """Ubuntu's default root prompt colours: bold green user@host, bold blue path."""
        return f"\033[01;32mroot@{self.ctx.host}\033[00m:\033[01;34m{self._display_cwd()}\033[00m# "

    def _resolve_paths(self, argv: list[str]) -> list[str]:
        """Make relative or ``~`` file arguments absolute when they name an existing file or directory.

        Commands work with absolute paths internally; this lets learners type
        ``cat gres.conf`` after ``cd /etc/slurm``, as on a real system.
        """
        from ..commands.system import known_dirs

        files, dirs = self.ctx.node.files, None
        out = [argv[0]]
        for arg in argv[1:]:
            prefix, value = "", arg
            if arg.startswith("-"):
                if "=" not in arg:
                    out.append(arg)
                    continue
                prefix, _, value = arg.partition("=")
                prefix += "="
            if not value or value.startswith("/") or "://" in value or (":" in value and not value.startswith("~")):
                out.append(arg)
                continue
            candidate = self.ctx.abspath(value)
            if dirs is None:
                dirs = known_dirs(self.ctx)
            if value.startswith("~") or candidate in files or candidate in dirs:
                out.append(prefix + candidate)
            else:
                out.append(arg)
        return out

    def run(self, line: str, heredoc_reader=None, record: bool = True) -> str:
        """Run one input line and return its combined output."""
        line = line.strip()
        if not line or line.startswith("#"):
            return ""
        if record:
            self.ctx.history.append(line)
        if heredoc_reader is not None:
            self.ctx.reader = heredoc_reader
        heredoc = None
        if "<<" in line:
            line, heredoc = self._read_heredoc(line, heredoc_reader)
        outputs = []
        last_code = 0
        for segment, sep in _split_top(line, ("&&", "||", ";")):
            if not segment:
                continue
            if sep == "&&" and last_code != 0:
                continue
            if sep == "||" and last_code == 0:
                continue
            out, last_code = self._run_pipeline(segment, heredoc)
            heredoc = None
            if out:
                outputs.append(out.rstrip("\n"))
        return "\n".join(outputs)

    # ------------------------------------------------------------ internal-error recovery

    def _checkpoint(self):
        """Snapshot the cluster and shell position so a crashing command can be undone."""
        try:
            state = pickle.dumps(self.ctx.state, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            state = None
        return state, self.ctx.host, list(self.ctx.host_stack), self.ctx.cwd

    def _recover(self, argv: list[str], error: Exception, checkpoint, executed_before: int) -> str:
        """Undo a command that raised an unexpected exception and explain what happened.

        The command (and anything it ran via ``ssh host cmd``) is dropped from the
        executed-command log, so evidence checks cannot credit a command that crashed.
        """
        state_blob, host, host_stack, cwd = checkpoint
        del self.ctx.executed[executed_before:]
        self.ctx.host, self.ctx.host_stack, self.ctx.cwd = host, host_stack, cwd
        detail = f"{type(error).__name__}: {error}"
        if state_blob is not None:
            restored = pickle.loads(state_blob)
            self.ctx.state.clear()
            dict.update(self.ctx.state, restored)
            outcome = "The simulated cluster was restored to its state before this command."
        else:
            self.ctx.state.setdefault("flags", {})[INCONSISTENT_FLAG] = f"{argv[0]}: {detail}"
            outcome = "The cluster state could not be restored; type `reset` before continuing (grading is blocked)."
        return (
            f"greenray: internal simulator error while running '{argv[0]}' ({detail}).\n"
            f"{outcome} This is a bug in the simulator, not in your command; please report it."
        )

    def _read_heredoc(self, line: str, reader):
        head, _, marker = line.partition("<<")
        marker = marker.strip().strip("'\"").lstrip("-")
        rest = ""
        if " " in marker:  # e.g. `cat <<EOF > file`
            marker, rest = marker.split(" ", 1)
        body = []
        if reader is None:
            raise CommandError("heredoc needs interactive input")
        while True:
            nxt = reader(HEREDOC_PROMPT)
            if nxt is None or nxt.strip() == marker:
                break
            body.append(nxt)
        return (head + " " + rest).strip(), "\n".join(body) + "\n"

    def _run_pipeline(self, segment: str, heredoc: str | None) -> tuple[str, int]:
        stages = [s for s, _ in _split_top(segment, ("|",))]
        data = heredoc or ""
        code = 0
        for i, stage in enumerate(stages):
            redirect, target = None, None
            parts = _split_top(stage, (">>", ">"))
            if len(parts) > 1:
                stage = parts[0][0]
                redirect, target = parts[1][1], self.ctx.abspath(parts[1][0].strip())
            # real tools only colour output that goes straight to the terminal
            self.ctx.tty = i == len(stages) - 1 and not redirect
            try:
                argv = tokenize(stage, self._lookup, self._substitute)
            except ValueError as e:
                return f"bash: syntax error: {e}", 2
            if argv and argv[0] == "sudo":
                argv = argv[1:]
            if not argv:
                continue
            argv = self._resolve_paths(argv)
            self.ctx.stdin = data
            executed_before = len(self.ctx.executed)
            self.ctx.executed.append((self.ctx.host, argv))
            checkpoint = self._checkpoint()
            try:
                data = commands.dispatch(self.ctx, argv)
                code = 0
            except CommandError as e:
                data, code = str(e), e.code
            except Exception as e:  # a simulator bug, not a user error
                return self._recover(argv, e, checkpoint, executed_before), INTERNAL_ERROR_CODE
            if data is None:
                data = ""
            if redirect and target:
                files = self.ctx.node.files
                if redirect == ">>":
                    files[target] = files.get(target, "") + data
                else:
                    files[target] = data
                data = ""
        return data, code
