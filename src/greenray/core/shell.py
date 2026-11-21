"""A small bash-like shell that runs simulated commands against the cluster state.

Supports: `;` and `&&` chaining, `|` pipes, `>` / `>>` redirects, `cat > f <<EOF`
heredocs, `export VAR=val`, `sudo` (ignored, you are root), `ssh <host>` / `exit`.
"""

from __future__ import annotations

import pickle
import posixpath
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .. import commands
from . import clock
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
    outputs: list = field(default_factory=list)  # (host, argv, text) for every command that completed without error
    # (position in executed, host, argv, text, completed) for every command that ended, with its output or its error
    # message; lets a check ask what a command showed and whether that came before another command
    results: list = field(default_factory=list)
    # a shell started by another command (ssh host cmd, pdsh, bash -c, sudo, salloc cmd, cmsh pexec) shares these
    # lists with its parent, so evidence is found on the host where the command really ran
    reader: Callable[[str], str | None] | None = (
        None  # prompt -> line; lets interactive commands (editors) read from the learner
    )
    color: bool = False  # the learner's terminal supports colour (set by the UI, never in tests)
    tty: bool = False  # the current command writes straight to that terminal (not a pipe or redirect)
    emit: Callable[[str], None] | None = None  # print immediately; lets watch/dmon stream on a terminal
    live_delay: float = 0.0  # real seconds between streamed samples (0 in tests and pipes)
    last_code: int = 0  # exit status of the last pipeline, what $? expands to
    oldcwd: str = "/root"  # for `cd -` and $OLDPWD
    alloc_stack: list = field(default_factory=list)  # salloc shells: job ids released by `exit`

    @property
    def node(self):
        return self.state.nodes[self.host]

    @property
    def streaming(self) -> bool:
        """True when a command may print samples as they happen instead of returning one blob."""
        return self.tty and self.emit is not None

    @property
    def colorize(self) -> bool:
        """True when a command should colour its output, like real tools do on a terminal."""
        return self.color and self.tty

    def abspath(self, path: str) -> str:
        """Resolve ``path`` like the shell would: ``~`` is /root, relative paths start at the cwd."""
        if path == "~" or path.startswith("~/"):
            path = self.home + path[1:]
        if not path.startswith("/"):
            path = posixpath.join(self.cwd, path)
        return posixpath.normpath(path)

    @property
    def user(self) -> str:
        from ..commands.slurm.common import _USER

        return _USER[-1]

    @property
    def home(self) -> str:
        return "/root" if self.user == "root" else f"/home/{self.user}"

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
    return [w for w, _ in tokenize_ex(text, lookup, substitute)]


def _arith(expr: str, lookup) -> str:
    """``$((...))`` on integers: + - * / % and parentheses; bare names are variables."""
    import ast

    src = re.sub(r"\$?\b([A-Za-z_]\w*)\b", lambda m: str(int(lookup(m.group(1)) or 0)), expr)
    try:
        node = ast.parse(src.strip() or "0", mode="eval")
    except SyntaxError:
        raise ValueError(f"{expr}: syntax error in expression") from None
    ops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a // b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
    }

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -ev(n.operand) if isinstance(n.op, ast.USub) else ev(n.operand)
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            b = ev(n.right)
            if b == 0 and isinstance(n.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                raise ValueError(f"{expr}: division by 0")
            return ops[type(n.op)](ev(n.left), b)
        raise ValueError(f"{expr}: syntax error in expression")

    return str(ev(node))


def tokenize_ex(text: str, lookup, substitute) -> list[tuple[str, bool]]:
    """Like ``tokenize`` but each word carries whether any part of it was quoted or escaped.

    The shell globs only words that were not; ``echo "*"`` stays a star.
    """
    words: list[tuple[str, bool]] = []
    buf: list[str] = []
    in_word = False
    quoted_part = False
    quote = None
    i, n = 0, len(text)

    def expand_at(j: int) -> tuple[str, int]:
        if text.startswith("$((", j):
            depth, k = 2, j + 3
            while k < n and depth:
                if text[k] == "(":
                    depth += 1
                elif text[k] == ")":
                    depth -= 1
                k += 1
            if depth:
                raise ValueError("unexpected EOF while looking for matching `)'")
            return _arith(text[j + 3 : k - 2], lookup), k
        if text.startswith("$?", j):
            return lookup("?"), j + 2
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
            inner = text[j + 2 : end]
            m = re.fullmatch(r"([A-Za-z_]\w*|\?)(:?[-=+?])(.*)", inner, re.S)
            if m:
                name, op, word = m.groups()
                cur = lookup(name)
                if op in (":-", "-"):
                    return (cur if cur else word), end + 1
                if op in (":=", "="):
                    if not cur:
                        lookup(f"={name}={word}")  # assignment through the lookup hook
                        return word, end + 1
                    return cur, end + 1
                if op in (":+", "+"):
                    return (word if cur else ""), end + 1
                if not cur:
                    raise ValueError(f"{name}: {word or 'parameter null or not set'}")
                return cur, end + 1
            if not re.fullmatch(r"[A-Za-z_]\w*|\?|#\w*", inner):
                raise ValueError(f"${{{inner}}}: bad substitution")
            return lookup(inner), end + 1
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
                words.append(("".join(buf), quoted_part))
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
                words.append(("".join(buf), quoted_part))
                buf, quoted_part = [], False
            for k, part in enumerate(parts):
                if k:
                    words.append(("".join(buf), quoted_part))
                    buf, quoted_part = [], False
                buf.append(part)
            if parts and val[-1:].isspace():
                words.append(("".join(buf), quoted_part))
                buf, in_word, quoted_part = [], False, False
        else:
            buf.append(ch)
            i += 1
    if quote:
        raise ValueError(f"unexpected EOF while looking for matching `{quote}'")
    if in_word and ("".join(buf) or quoted_part):
        words.append(("".join(buf), quoted_part))
    return words


def _glob_match(pattern: str, path: str) -> bool:
    """fnmatch per path component: ``*`` never crosses a ``/``, as in bash."""
    import fnmatch

    pp, cp = pattern.split("/"), path.split("/")
    if len(pp) != len(cp):
        return False
    return all(fnmatch.fnmatchcase(c, p) for p, c in zip(pp, cp))


# commands whose bare relative arguments name files; for anything else a word is just a word
PATH_COMMANDS = frozenset(
    [
        "cat",
        "less",
        "more",
        "head",
        "tail",
        "wc",
        "grep",
        "egrep",
        "sed",
        "awk",
        "sort",
        "uniq",
        "cut",
        "tr",
        "diff",
        "cmp",
        "cp",
        "mv",
        "rm",
        "ln",
        "ls",
        "stat",
        "file",
        "du",
        "md5sum",
        "sha256sum",
        "tar",
        "gzip",
        "gunzip",
        "nano",
        "vi",
        "vim",
        "sbatch",
        "srun",
        "bash",
        "sh",
        "source",
        "chmod",
        "chown",
        "touch",
        "mkdir",
        "rmdir",
        "find",
        "tee",
        "realpath",
        "basename",
        "dirname",
        "xargs",
        "scp",
        "cd",
        "python",
        "python3",
    ]
)
REDIRECTS = ("2>&1", "2>>", "2>", "&>>", "&>", ">>", ">", "<")
ASSIGNMENT = re.compile(r"^([A-Za-z_]\w*)=(.*)$", re.S)


class Shell:
    def __init__(self, state, host: str = HEAD, color: bool = False):
        self.ctx = Context(state=state, host=host, color=color)
        for fn in commands.HOOKS_STATE:
            fn(state)

    def _lookup(self, name: str) -> str:
        if name.startswith("="):  # ${X:=word} assigns through this hook
            var, _, value = name[1:].partition("=")
            self.ctx.node.env[var] = value
            return value
        special = {
            "HOSTNAME": self.ctx.host,
            "?": str(self.ctx.last_code),
            "HOME": self.ctx.home,
            "USER": self.ctx.user,
            "LOGNAME": self.ctx.user,
            "PWD": self.ctx.cwd,
            "OLDPWD": self.ctx.oldcwd,
            "SHELL": "/bin/bash",
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/cuda/bin",
        }
        if name in self.ctx.node.env:
            return self.ctx.node.env[name]
        return special.get(name, "")

    def _substitute(self, command: str) -> str:
        stdin = self.ctx.stdin
        out, _err = self._run_line(command)  # stderr stays on the terminal, not in the substitution
        self.ctx.stdin = stdin
        return out

    def _display_cwd(self) -> str:
        cwd, home = self.ctx.cwd, self.ctx.home
        return "~" + cwd[len(home) :] if cwd == home or cwd.startswith(home + "/") else cwd

    @property
    def prompt(self) -> str:
        return f"{self.ctx.user}@{self.ctx.host}:{self._display_cwd()}{'#' if self.ctx.user == 'root' else '$'} "

    def styled_prompt(self) -> str:
        """Ubuntu's default root prompt colours: bold green user@host, bold blue path."""
        mark = "#" if self.ctx.user == "root" else "$"
        return f"\033[01;32m{self.ctx.user}@{self.ctx.host}\033[00m:\033[01;34m{self._display_cwd()}\033[00m{mark} "

    def _resolve_paths(self, argv: list[str]) -> list[str]:
        """Make relative or ``~`` file arguments absolute when they name an existing file or directory.

        Commands work with absolute paths internally; this lets learners type
        ``cat gres.conf`` after ``cd /etc/slurm``. Only commands that take files get this,
        plus any word that is written like a path (``./x``, ``~/x``, ``a/b``): for ``echo data``
        or ``kubectl get ns slurm`` a word is just a word, as in bash.
        """
        from ..commands.system import known_dirs

        files, dirs = self.ctx.node.files, None
        takes_files = argv[0].rsplit("/", 1)[-1] in PATH_COMMANDS
        out = [argv[0]]
        for arg in argv[1:]:
            prefix, value = "", arg
            if arg.startswith("-"):
                if "=" not in arg:
                    out.append(arg)
                    continue
                prefix, _, value = arg.partition("=")
                prefix += "="
            pathlike = value.startswith(("~", "./", "../")) or ("/" in value and not value.startswith("/"))
            if not value or value.startswith("/") or "://" in value or (":" in value and not value.startswith("~")):
                out.append(arg)
                continue
            if not (takes_files or pathlike):
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

    def _glob(self, words: list[tuple[str, bool]]) -> list[str]:
        """Expand unquoted ``*``, ``?`` and ``[...]`` against the node's files and directories.

        A pattern that matches nothing is left as typed, which is what bash does by default.
        """
        from ..commands.system import known_dirs

        out: list[str] = []
        names: set[str] | None = None
        for word, quoted in words:
            if quoted or not any(c in word for c in "*?["):
                out.append(word)
                continue
            if names is None:
                from ..commands.system import virtual_files

                names = set(self.ctx.node.files) | set(virtual_files(self.ctx)) | known_dirs(self.ctx)
            absolute = word.startswith("/")
            pattern = self.ctx.abspath(word)
            hits = sorted(n for n in names if _glob_match(pattern, n))
            if not hits:
                out.append(word)
                continue
            base = self.ctx.cwd.rstrip("/") + "/"
            out.extend(h if absolute else (h[len(base) :] if h.startswith(base) else h) for h in hits)
        return out

    def run(self, line: str, heredoc_reader=None, record: bool = True) -> str:
        """Run one input line and return what the terminal shows: output and error text together."""
        out, err = self._run_line(line, heredoc_reader, record)
        return "\n".join(part for part in (err, out) if part)

    def _run_line(self, line: str, heredoc_reader=None, record: bool = True) -> tuple[str, str]:
        """Run one input line; returns (stdout, stderr) so a substitution or pipe sees only stdout."""
        line = line.strip()
        if not line or line.startswith("#"):
            return "", ""
        if record:
            self.ctx.history.append(line)
        if heredoc_reader is not None:
            self.ctx.reader = heredoc_reader
        heredoc = None
        if "<<" in line and _split_top(line, ("<<",))[1:]:
            try:
                line, heredoc = self._read_heredoc(line, heredoc_reader or self.ctx.reader)
            except CommandError as e:
                self.ctx.last_code = e.code
                return "", str(e)
        outputs, errors = [], []
        last_code = self.ctx.last_code
        for segment, sep in _split_top(line, ("&&", "||", ";")):
            if not segment:
                continue
            if sep == "&&" and last_code != 0:
                continue
            if sep == "||" and last_code == 0:
                continue
            out, err, last_code = self._run_pipeline(segment, heredoc)
            heredoc = None
            self.ctx.last_code = last_code
            if out:
                outputs.append(out.rstrip("\n"))
            if err:
                errors.append(err.rstrip("\n"))
        return "\n".join(outputs), "\n".join(errors)

    # ------------------------------------------------------------ internal-error recovery

    def _checkpoint(self):
        """Snapshot the cluster and shell position so a crashing command can be undone."""
        try:
            state = pickle.dumps(self.ctx.state, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            state = None
        return state, self.ctx.host, list(self.ctx.host_stack), self.ctx.cwd

    def _recover(self, argv: list[str], error: Exception, checkpoint, executed_before: int, outputs_before: int) -> str:
        """Undo a command that raised an unexpected exception and explain what happened.

        The command (and anything it ran via ``ssh host cmd``) is dropped from the
        executed-command log, so evidence checks cannot credit a command that crashed.
        """
        state_blob, host, host_stack, cwd = checkpoint
        del self.ctx.executed[executed_before:]
        del self.ctx.outputs[outputs_before:]
        self.ctx.results[:] = [r for r in self.ctx.results if r[0] < executed_before]
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
        head, _, tail = line.partition("<<")
        strip_tabs = tail.startswith("-")
        tail = tail.lstrip("-").lstrip()
        m = re.match(r"""('([^']*)'|"([^"]*)"|(\S+))(.*)$""", tail, re.S)
        if not m:
            raise CommandError("bash: syntax error near unexpected token `newline'", 2)
        marker = m.group(2) or m.group(3) or m.group(4) or ""
        rest = m.group(5)
        body = []
        if reader is None:
            raise CommandError("heredoc needs interactive input")
        while True:
            nxt = reader(HEREDOC_PROMPT)
            if nxt is None or nxt.strip() == marker:
                break
            body.append(nxt.lstrip("\t") if strip_tabs else nxt)
        return (head + " " + rest).strip(), "\n".join(body) + "\n"

    def _redirects(self, stage: str) -> tuple[str, dict]:
        """Split ``cmd > out 2> err < in`` into the command and its redirections."""
        parts = _split_top(stage, REDIRECTS)
        redir: dict = {}
        cmd = parts[0][0]
        for text, sep in parts[1:]:
            if sep == "2>&1":
                redir["err_to_out"] = True
                cmd = (cmd + " " + text).strip()
                continue
            words = text.strip().split(None, 1)
            if not words:
                raise CommandError("bash: syntax error near unexpected token `newline'", 2)
            target = self.ctx.abspath(words[0].strip("'\""))
            if len(words) > 1:
                cmd = (cmd + " " + words[1]).strip()
            if sep == "<":
                redir["stdin"] = target
            elif sep in ("2>", "2>>"):
                redir["err"] = (target, sep == "2>>")
            elif sep in ("&>", "&>>"):
                redir["out"] = (target, sep == "&>>")
                redir["err"] = (target, sep == "&>>")
            else:
                redir["out"] = (target, sep == ">>")
        return cmd, redir

    def _write(self, target: str, text: str, append: bool) -> None:
        files = self.ctx.node.files
        files[target] = (files.get(target, "") if append else "") + text

    def _run_pipeline(self, segment: str, heredoc: str | None) -> tuple[str, str, int]:
        stages = [s for s, _ in _split_top(segment, ("|",))]
        data = heredoc or ""
        code = 0
        errors: list[str] = []
        for i, stage in enumerate(stages):
            try:
                stage, redir = self._redirects(stage)
            except CommandError as e:
                return "", str(e), e.code
            # real tools only colour output that goes straight to the terminal
            self.ctx.tty = i == len(stages) - 1 and "out" not in redir
            try:
                words = tokenize_ex(stage, self._lookup, self._substitute)
            except ValueError as e:
                return "", f"bash: syntax error: {e}", 2
            argv = self._glob(words)
            # VAR=value [VAR2=value2] [command]: assignments stick when there is no command
            temp_env: dict = {}
            while argv and (m := ASSIGNMENT.match(argv[0])):
                name, value = m.groups()
                temp_env[name] = value
                argv = argv[1:]
            if not argv:
                self.ctx.node.env.update(temp_env)
                continue
            try:
                argv = self._sudo(argv)
            except CommandError as e:
                errors.append(str(e))
                code = e.code
                data = ""
                continue
            argv = self._resolve_paths(argv)
            if "stdin" in redir:
                src = redir["stdin"]
                if src not in self.ctx.node.files:
                    errors.append(f"bash: {src}: No such file or directory")
                    code = 1
                    data = ""
                    continue
                data = self.ctx.node.files[src]
            self.ctx.stdin = data
            executed_before, outputs_before = len(self.ctx.executed), len(self.ctx.outputs)
            ran_on = self.ctx.host
            self.ctx.executed.append((ran_on, argv))
            checkpoint = self._checkpoint()
            clock.advance(self.ctx.state)  # inside the checkpoint so a rolled-back crash takes no time
            saved_env = dict(self.ctx.node.env) if temp_env else None
            if temp_env:
                self.ctx.node.env.update(temp_env)
            err = ""
            try:
                data = commands.dispatch(self.ctx, argv)
                code = 0
                self.ctx.outputs.append((self.ctx.host, argv, data or ""))
                self.ctx.results.append((executed_before, ran_on, argv, data or "", True))
            except CommandError as e:
                data, err, code = "", str(e), e.code
                self.ctx.results.append((executed_before, ran_on, argv, err, False))
            except Exception as e:  # a simulator bug, not a user error
                return "", self._recover(argv, e, checkpoint, executed_before, outputs_before), INTERNAL_ERROR_CODE
            finally:
                if saved_env is not None:
                    self.ctx.node.env.clear()
                    self.ctx.node.env.update(saved_env)
            for fn in commands.HOOKS_TICK:
                fn(self.ctx.state)
            if data is None:
                data = ""
            if err and "err" in redir:
                target, append = redir["err"]
                self._write(target, err + ("\n" if not err.endswith("\n") else ""), append)
                err = ""
            if err and redir.get("err_to_out"):
                data = data + ("\n" if data and not data.endswith("\n") else "") + err + "\n"
                err = ""
            if err:
                errors.append(err)
            if "out" in redir:
                target, append = redir["out"]
                self._write(target, data, append or redir.get("err") == (target, False) and bool(errors) and False)
                data = ""
        return data, "\n".join(errors), code

    def _sudo(self, argv: list[str]) -> list[str]:
        """``sudo [-u user] cmd``: root runs it directly, another user through runuser."""
        if not argv or argv[0] != "sudo" or any(a in ("-h", "--help") for a in argv[1:]):
            return argv
        user, rest = None, argv[1:]
        while rest and rest[0].startswith("-"):
            if rest[0] in ("-u", "--user") and len(rest) > 1:
                user, rest = rest[1], rest[2:]
            elif rest[0].startswith("--user="):
                user, rest = rest[0].split("=", 1)[1], rest[1:]
            elif rest[0] == "--":
                rest = rest[1:]
                break
            else:
                rest = rest[1:]  # -i -E -n -s and friends change nothing here
        if not rest:
            raise CommandError("usage: sudo [-u user] command", 1)
        if user and user != "root":
            return ["runuser", "-u", user, "--", *rest]
        return rest
