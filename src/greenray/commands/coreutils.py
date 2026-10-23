"""The everyday text, file and shell tools that are on every node: less, cut, uniq, tr, find, mv, diff, date, ..."""

from __future__ import annotations

import fnmatch
import re
import shlex
import time
from datetime import datetime, timedelta

from ..core import clock
from ..core.errors import CommandError
from . import command, dispatch
from .common import opt
from .system import known_dirs


def _file(ctx, path: str, tool: str) -> str:
    full = ctx.abspath(path)
    if full in ctx.node.files:
        return ctx.node.files[full]
    if full in known_dirs(ctx):
        raise CommandError(f"{tool}: {path}: Is a directory", 1)
    raise CommandError(f"{tool}: {path}: No such file or directory", 1)


def _inputs(ctx, args: list[str], tool: str) -> str:
    """The text a filter works on: the named files, else stdin."""
    files = [a for a in args if not a.startswith("-")]
    if not files:
        return ctx.stdin
    return "".join(_file(ctx, f, tool) for f in files)


@command("less", "more", help="less FILE: show a file (the pager prints everything here)")
def less(ctx, args):
    return _inputs(ctx, args, "less")


@command("cut", help="cut -d DELIM -f LIST | -c LIST")
def cut(ctx, args):
    delim = opt(args, "-d", "--delimiter", default="\t")
    fields = opt(args, "-f", "--fields")
    chars = opt(args, "-c", "--characters")
    if not fields and not chars:
        raise CommandError(
            "cut: you must specify a list of bytes, characters, or fields\nTry 'cut --help' for more information.", 1
        )
    text = _inputs(ctx, [a for a in args if a not in (delim, fields, chars)], "cut")
    ranges = []
    for part in (fields or chars).split(","):
        lo, _, hi = part.partition("-")
        if not (lo or hi) or (lo and not lo.isdigit()) or (hi and not hi.isdigit()):
            raise CommandError(f"cut: invalid field value '{part}'", 1)
        ranges.append((int(lo) if lo else 1, int(hi) if hi else (int(lo) if not _ else 10**6)))
    out = []
    for line in text.splitlines():
        if chars:
            out.append("".join(line[lo - 1 : hi] for lo, hi in ranges))
            continue
        if delim not in line:
            out.append(line)
            continue
        parts = line.split(delim)
        picked = [p for i, p in enumerate(parts, 1) if any(lo <= i <= hi for lo, hi in ranges)]
        out.append(delim.join(picked))
    return "\n".join(out) + ("\n" if out else "")


@command("uniq", help="uniq [-c] [-d]: collapse adjacent duplicate lines")
def uniq(ctx, args):
    text = _inputs(ctx, args, "uniq")
    count, dups_only = "-c" in args, "-d" in args
    out, prev, n = [], None, 0

    def flush():
        if prev is not None and (not dups_only or n > 1):
            out.append(f"{n:>7} {prev}" if count else prev)

    for line in text.splitlines():
        if line == prev:
            n += 1
        else:
            flush()
            prev, n = line, 1
    flush()
    return "\n".join(out) + ("\n" if out else "")


@command("tr", help="tr SET1 SET2 | tr -d SET | tr -s SET")
def tr(ctx, args):
    delete, squeeze = "-d" in args, "-s" in args
    sets = [a for a in args if not a.startswith("-")]
    if not sets:
        raise CommandError("tr: missing operand\nTry 'tr --help' for more information.", 1)

    def expand(s):
        s = s.encode().decode("unicode_escape")
        return re.sub(r"(.)-(.)", lambda m: "".join(chr(c) for c in range(ord(m.group(1)), ord(m.group(2)) + 1)), s)

    a = expand(sets[0])
    text = ctx.stdin
    if delete:
        text = "".join(c for c in text if c not in a)
    elif len(sets) > 1:
        b = expand(sets[1])
        b = b + b[-1] * (len(a) - len(b)) if b else b
        text = text.translate(str.maketrans(a, b[: len(a)]))
    if squeeze:
        text = re.sub(f"([{re.escape(a)}])\\1+", r"\1", text)
    return text


@command("printf", help="printf FORMAT [ARG...]")
def printf(ctx, args):
    if not args:
        raise CommandError("printf: usage: printf [-v var] format [arguments]", 2)
    fmt = args[0].encode().decode("unicode_escape")
    values = list(args[1:])
    specs = re.findall(r"%[-+ 0#]*\d*(?:\.\d+)?[sdifgxX%]", fmt)
    per_round = sum(1 for s in specs if s != "%%")
    out = []
    while True:
        chunk = values[:per_round] if per_round else []
        values = values[per_round:]
        try:
            out.append(
                fmt % tuple(_coerce(v, s) for v, s in zip(chunk + [""] * per_round, [s for s in specs if s != "%%"]))
            )
        except (TypeError, ValueError) as e:
            raise CommandError(f"printf: {e}", 1)
        if not values or not per_round:
            break
    return "".join(out)


def _coerce(value, spec):
    if spec[-1] in "dixX":
        return int(value or 0)
    if spec[-1] in "fg":
        return float(value or 0)
    return value


@command("mv", help="mv SRC DST")
def mv(ctx, args):
    pos = [a for a in args if not a.startswith("-")]
    if len(pos) < 2:
        raise CommandError("mv: missing file operand\nTry 'mv --help' for more information.", 1)
    *srcs, dst = pos
    dst_full = ctx.abspath(dst)
    dst_is_dir = dst_full in known_dirs(ctx) or dst.endswith("/")
    if len(srcs) > 1 and not dst_is_dir:
        raise CommandError(f"mv: target '{dst}' is not a directory", 1)
    files, dirs = ctx.node.files, ctx.node.setdefault("dirs", [])
    for src in srcs:
        full = ctx.abspath(src)
        target = dst_full.rstrip("/") + "/" + full.rsplit("/", 1)[-1] if dst_is_dir else dst_full
        if full in files:
            files[target] = files.pop(full)
        elif full in known_dirs(ctx):
            for f in [f for f in files if f.startswith(full + "/")]:
                files[target + f[len(full) :]] = files.pop(f)
            if full in dirs:
                dirs.remove(full)
            dirs.append(target)
        else:
            raise CommandError(f"mv: cannot stat '{src}': No such file or directory", 1)
    return ""


@command("ln", help="ln -s TARGET LINK (a copy stands in for the link)")
def ln(ctx, args):
    pos = [a for a in args if not a.startswith("-")]
    if len(pos) < 2:
        raise CommandError("ln: missing file operand\nTry 'ln --help' for more information.", 1)
    target, link = ctx.abspath(pos[0]), ctx.abspath(pos[1])
    if "-s" not in args and target not in ctx.node.files:
        raise CommandError(f"ln: failed to access '{pos[0]}': No such file or directory", 1)
    if link in ctx.node.files and "-f" not in args:
        raise CommandError(f"ln: failed to create symbolic link '{pos[1]}': File exists", 1)
    ctx.node.files[link] = ctx.node.files.get(target, "")
    ctx.node.setdefault("links", {})[link] = target
    return ""


@command("rmdir", help="rmdir DIR: remove an empty directory")
def rmdir(ctx, args):
    for a in args:
        if a.startswith("-"):
            continue
        full = ctx.abspath(a)
        if full not in known_dirs(ctx):
            raise CommandError(f"rmdir: failed to remove '{a}': No such file or directory", 1)
        if any(f.startswith(full + "/") for f in ctx.node.files):
            raise CommandError(f"rmdir: failed to remove '{a}': Directory not empty", 1)
        if full in ctx.node.get("dirs", []):
            ctx.node.dirs.remove(full)
    return ""


@command("find", help="find DIR [-name PATTERN] [-type f|d] [-newer FILE] [-mmin N]")
def find(ctx, args):
    roots, i, name, ftype = [], 0, None, None
    while i < len(args):
        a = args[i]
        if a in ("-name", "-iname"):
            name = args[i + 1] if i + 1 < len(args) else None
            i += 2
            continue
        if a == "-type":
            ftype = args[i + 1] if i + 1 < len(args) else None
            if ftype not in ("f", "d"):
                raise CommandError(f"find: Unknown argument to -type: {ftype}", 1)
            i += 2
            continue
        if a in ("-newer", "-mmin", "-mtime", "-size", "-maxdepth", "-user"):
            i += 2
            continue
        if a.startswith("-"):
            raise CommandError(f"find: unknown predicate `{a}'", 1)
        roots.append(a)
        i += 1
    roots = roots or ["."]
    dirs = known_dirs(ctx)
    out = []
    for root in roots:
        full = ctx.abspath(root)
        if full not in dirs and full not in ctx.node.files:
            raise CommandError(f"find: ‘{root}’: No such file or directory", 1)
        prefix = full.rstrip("/") + "/"
        entries = sorted({d for d in dirs if d == full or d.startswith(prefix)}) + sorted(
            f for f in ctx.node.files if f == full or f.startswith(prefix)
        )
        for e in entries:
            kind = "d" if e in dirs else "f"
            if ftype and kind != ftype:
                continue
            if name and not fnmatch.fnmatch(e.rsplit("/", 1)[-1], name):
                continue
            shown = e if root.startswith("/") else root.rstrip("/") + e[len(full) :]
            out.append(shown)
    return "\n".join(out) + ("\n" if out else "")


@command("which", "whereis", help="which COMMAND")
def which(ctx, args):
    from . import names

    out, missing = [], False
    for a in args:
        if a.startswith("-"):
            continue
        if a in names():
            out.append(f"/usr/bin/{a}" if a not in ("nvidia-smi", "dcgmi") else f"/usr/bin/{a}")
        else:
            missing = True
    if missing and not out:
        raise CommandError("", 1)
    return "\n".join(out) + ("\n" if out else "")


@command("file", help="file PATH: what kind of file it is")
def file_cmd(ctx, args):
    out = []
    for a in args:
        if a.startswith("-"):
            continue
        full = ctx.abspath(a)
        if full in known_dirs(ctx):
            out.append(f"{a}: directory")
        elif full in ctx.node.files:
            text = ctx.node.files[full]
            if text.startswith("#!"):
                interp = text.splitlines()[0][2:].strip()
                out.append(f"{a}: {interp.rsplit('/', 1)[-1]} script, ASCII text executable")
            elif not text:
                out.append(f"{a}: empty")
            elif full.endswith((".sqsh", ".bfb")):
                out.append(f"{a}: Squashfs filesystem, little endian, version 4.0")
            else:
                out.append(f"{a}: ASCII text")
        else:
            out.append(f"{a}: cannot open `{a}' (No such file or directory)")
    return "\n".join(out) + ("\n" if out else "")


@command("stat", help="stat PATH")
def stat(ctx, args):
    out = []
    when = clock.now(ctx.state).strftime("%Y-%m-%d %H:%M:%S.000000000 +0000")
    for a in args:
        if a.startswith("-"):
            continue
        full = ctx.abspath(a)
        if full in ctx.node.files:
            size, kind, mode = len(ctx.node.files[full]), "regular file", "-rw-r--r--"
            if ctx.node.files[full].startswith("#!"):
                mode = "-rwxr-xr-x"
        elif full in known_dirs(ctx):
            size, kind, mode = 4096, "directory", "drwxr-xr-x"
        else:
            raise CommandError(f"stat: cannot statx '{a}': No such file or directory", 1)
        out += [
            f"  File: {a}",
            f"  Size: {size:<15} Blocks: {(size + 511) // 512 * 8:<10} IO Block: 4096   {kind}",
            f"Device: 10302h/66306d\tInode: {abs(hash(full)) % 9000000 + 1000000}     Links: 1",
            f"Access: ({'0755' if 'x' in mode else ('0644' if kind != 'directory' else '0755')}/{mode})  Uid: (    0/    root)   Gid: (    0/    root)",
            f"Access: {when}",
            f"Modify: {when}",
            f"Change: {when}",
            f" Birth: {when}",
        ]
    return "\n".join(out) + ("\n" if out else "")


@command("du", help="du [-sh] PATH")
def du(ctx, args):
    human, summary = (
        any(a.startswith("-") and "h" in a for a in args),
        any(a.startswith("-") and "s" in a for a in args),
    )
    paths = [a for a in args if not a.startswith("-")] or ["."]
    out = []
    for p in paths:
        full = ctx.abspath(p)
        files = {f: len(t) for f, t in ctx.node.files.items() if f == full or f.startswith(full.rstrip("/") + "/")}
        if not files and full not in known_dirs(ctx):
            raise CommandError(f"du: cannot access '{p}': No such file or directory", 1)
        total = sum((s + 4095) // 4096 * 4 for s in files.values()) or 4
        if not summary:
            for f, s in sorted(files.items()):
                out.append(f"{_du_size((s + 4095) // 4096 * 4, human)}\t{f}")
        out.append(f"{_du_size(total, human)}\t{p}")
    return "\n".join(out) + "\n"


def _du_size(kib: int, human: bool) -> str:
    if not human:
        return str(kib)
    if kib >= 1024 * 1024:
        return f"{kib / 1024 / 1024:.1f}G"
    if kib >= 1024:
        return f"{kib / 1024:.1f}M"
    return f"{kib}K"


@command("basename", help="basename PATH [SUFFIX]")
def basename(ctx, args):
    pos = [a for a in args if not a.startswith("-")]
    if not pos:
        raise CommandError("basename: missing operand\nTry 'basename --help' for more information.", 1)
    name = pos[0].rstrip("/").rsplit("/", 1)[-1]
    if len(pos) > 1 and name.endswith(pos[1]) and name != pos[1]:
        name = name[: -len(pos[1])]
    return name + "\n"


@command("dirname", help="dirname PATH")
def dirname(ctx, args):
    pos = [a for a in args if not a.startswith("-")]
    if not pos:
        raise CommandError("dirname: missing operand\nTry 'dirname --help' for more information.", 1)
    head = pos[0].rstrip("/").rsplit("/", 1)[0] if "/" in pos[0].rstrip("/") else "."
    return (head or "/") + "\n"


@command("realpath", "readlink", help="realpath PATH: the absolute path")
def realpath(ctx, args):
    pos = [a for a in args if not a.startswith("-")]
    if not pos:
        raise CommandError("realpath: missing operand", 1)
    out = []
    for p in pos:
        full = ctx.abspath(p)
        links = ctx.node.get("links", {})
        out.append(links.get(full, full))
    return "\n".join(out) + "\n"


@command("date", help="date [+FORMAT] [-u] [-d STRING]")
def date(ctx, args):
    now = clock.now(ctx.state)
    spec = opt(args, "-d", "--date")
    if spec:
        m = re.fullmatch(r"(\d+) (second|minute|hour|day)s? ago", spec)
        if m:
            now -= timedelta(**{m.group(2) + "s": int(m.group(1))})
        elif spec.startswith("@") and spec[1:].isdigit():
            now = datetime.fromtimestamp(int(spec[1:]))
        else:
            try:
                now = datetime.fromisoformat(spec)
            except ValueError:
                raise CommandError(f"date: invalid date ‘{spec}’", 1)
    fmt = next((a[1:] for a in args if a.startswith("+")), None)
    if fmt:
        return now.strftime(fmt.replace("%N", "000000000")) + "\n"
    return now.strftime("%a %b %e %H:%M:%S UTC %Y") + "\n"


@command("sleep", help="sleep SECONDS (time passes on the simulated clock)")
def sleep(ctx, args):
    if not args:
        raise CommandError("sleep: missing operand\nTry 'sleep --help' for more information.", 1)
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd]?)", args[0])
    if not m:
        raise CommandError(f"sleep: invalid time interval ‘{args[0]}’", 1)
    seconds = float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]
    clock.advance(ctx.state, seconds)
    if ctx.streaming:
        time.sleep(min(seconds, 2.0))
    return ""


@command("true", help="true")
def true(ctx, args):
    return ""


@command("false", help="false")
def false(ctx, args):
    raise CommandError("", 1)


@command("seq", help="seq [FIRST [STEP]] LAST")
def seq(ctx, args):
    nums = [a for a in args if not a.startswith("-") or a[1:].isdigit()]
    try:
        values = [int(n) for n in nums]
    except ValueError:
        raise CommandError(f"seq: invalid floating point argument: ‘{nums[-1]}’", 1)
    if not values:
        raise CommandError("seq: missing operand\nTry 'seq --help' for more information.", 1)
    first, step, last = (
        (1, 1, values[0]) if len(values) == 1 else (values[0], 1, values[1]) if len(values) == 2 else values[:3]
    )
    if step == 0:
        raise CommandError("seq: invalid Zero increment value: ‘0’", 1)
    return "".join(f"{v}\n" for v in range(first, last + (1 if step > 0 else -1), step))


@command("xargs", help="xargs [-n N] COMMAND: run a command with stdin as arguments")
def xargs(ctx, args):
    per = opt(args, "-n")
    rest = [a for a in args if a not in ("-n", per)]
    words = ctx.stdin.split()
    cmd = rest or ["echo"]
    if not words:
        return ""
    chunk = int(per) if per and per.isdigit() else len(words)
    out = []
    for i in range(0, len(words), chunk):
        out.append(dispatch(ctx, cmd + words[i : i + chunk]))
    return "".join(out)


@command("test", "[", help="test EXPRESSION: -f FILE, -d DIR, -e PATH, -z STR, -n STR, STR = STR, N -eq N")
def test(ctx, args):
    args = [a for a in args if a != "]"]
    ok = _evaluate(ctx, args)
    if not ok:
        raise CommandError("", 1)
    return ""


def _evaluate(ctx, args: list[str]) -> bool:
    if not args:
        return False
    if args[0] == "!":
        return not _evaluate(ctx, args[1:])
    if len(args) == 2:
        flag, v = args
        full = ctx.abspath(v)
        return {
            "-f": full in ctx.node.files,
            "-d": full in known_dirs(ctx),
            "-e": full in ctx.node.files or full in known_dirs(ctx),
            "-r": full in ctx.node.files,
            "-s": bool(ctx.node.files.get(full)),
            "-x": ctx.node.files.get(full, "").startswith("#!") or full.startswith("/usr/bin"),
            "-z": v == "",
            "-n": v != "",
        }.get(flag, bool(v))
    if len(args) == 3:
        a, op, b = args
        if op in ("=", "==", "!="):
            return (a == b) == (op != "!=")
        try:
            x, y = int(a), int(b)
        except ValueError:
            raise CommandError(f"bash: test: {a}: integer expression expected", 2)
        return {"-eq": x == y, "-ne": x != y, "-lt": x < y, "-le": x <= y, "-gt": x > y, "-ge": x >= y}[op]
    return bool(args[0])


@command("expr", help="expr A + B | A \\* B | STRING : REGEX")
def expr(ctx, args):
    if len(args) == 3 and args[1] in ("+", "-", "*", "/", "%"):
        try:
            x, y = int(args[0]), int(args[2])
        except ValueError:
            raise CommandError("expr: non-integer argument", 2)
        if args[1] in ("/", "%") and y == 0:
            raise CommandError("expr: division by zero", 2)
        return str({"+": x + y, "-": x - y, "*": x * y, "/": x // y, "%": x % y}[args[1]]) + "\n"
    if len(args) == 3 and args[1] == ":":
        m = re.match(args[2], args[0])
        return (str(len(m.group(0))) if m else "0") + "\n"
    raise CommandError("expr: syntax error: unexpected argument", 2)


@command("diff", help="diff [-u] FILE1 FILE2")
def diff(ctx, args):
    import difflib

    pos = [a for a in args if not a.startswith("-")]
    if len(pos) != 2:
        raise CommandError("diff: missing operand after 'diff'\ndiff: Try 'diff --help' for more information.", 2)
    a, b = (_file(ctx, p, "diff").splitlines(keepends=True) for p in pos)
    if "-u" in args:
        lines = list(difflib.unified_diff(a, b, fromfile=pos[0], tofile=pos[1]))
    else:
        lines = []
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
            if tag == "equal":
                continue
            span = lambda x, y: str(x + 1) if y - x <= 1 else f"{x + 1},{y}"  # noqa: E731
            lines.append(
                f"{span(i1, i2)}{'acd'[['replace', 'equal', 'delete'].index(tag) if tag in ('replace', 'delete') else 0] if tag != 'insert' else 'a'}{span(j1, j2)}\n"
            )
            lines += [f"< {l}" for l in a[i1:i2]]
            if tag == "replace":
                lines.append("---\n")
            lines += [f"> {l}" for l in b[j1:j2]]
    if not lines:
        return ""
    raise CommandError("".join(lines).rstrip("\n"), 1)


@command("cmp", help="cmp FILE1 FILE2")
def cmp(ctx, args):
    pos = [a for a in args if not a.startswith("-")]
    if len(pos) != 2:
        raise CommandError("cmp: missing operand after 'cmp'\ncmp: Try 'cmp --help' for more information.", 2)
    a, b = (_file(ctx, p, "cmp") for p in pos)
    if a == b:
        return ""
    for i, (x, y) in enumerate(zip(a, b), 1):
        if x != y:
            raise CommandError(f"{pos[0]} {pos[1]} differ: byte {i}, line {a[:i].count(chr(10)) + 1}", 1)
    short = pos[0] if len(a) < len(b) else pos[1]
    raise CommandError(f"cmp: EOF on {short} after byte {min(len(a), len(b))}", 1)


@command("history", help="history: the commands typed in this session")
def history(ctx, args):
    return "".join(f"{i:>5}  {line}\n" for i, line in enumerate(ctx.history, 1))


@command("alias", "set", "umask", help="alias / set / umask: accepted, nothing to configure here")
def shell_builtin(ctx, args):
    return ""


@command("bash", "sh", help="bash SCRIPT | bash -c 'COMMANDS': run a script on this node")
def bash(ctx, args):
    from ..core.shell import Shell

    if not args:
        raise CommandError("bash: an interactive shell is already running; give a script or -c 'commands'", 1)
    if args[0] == "-c":
        lines = [" ".join(args[1:])]
    else:
        text = _file(ctx, args[0], "bash")
        lines = [l for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
        if lines and lines[0].startswith("#!"):
            lines = lines[1:]
    sub = Shell(ctx.state, ctx.host)
    sub.ctx.executed = ctx.executed
    sub.ctx.cwd = ctx.cwd
    out = []
    for line in lines:
        try:
            res = sub.run(line, record=False)
        except CommandError as e:
            out.append(str(e))
            if "set -e" in text if args[0] != "-c" else False:
                break
            continue
        if res:
            out.append(res.rstrip("\n"))
    return "\n".join(out) + ("\n" if out else "")


@command("sudo", help="sudo COMMAND: you are already root here")
def sudo(ctx, args):
    rest = [a for a in args if not a.startswith("-")]
    if not rest:
        raise CommandError("usage: sudo -h | -K | -k | -V\nusage: sudo [-u user] command", 1)
    return dispatch(ctx, rest)


@command("tar", help="tar -czf ARCHIVE PATH... | tar -xzf ARCHIVE [-C DIR] | tar -tf ARCHIVE")
def tar(ctx, args):
    flags = "".join(a.lstrip("-") for a in args if a.startswith("-") and not a.startswith("--") and a not in ("-C",))
    pos = [a for a in args if not a.startswith("-")]
    into = opt(args, "-C", default=None)
    if into:
        pos = [p for p in pos if p != into]
    if not pos:
        raise CommandError("tar: Refusing to read archive contents from terminal (missing -f option?)", 2)
    archive, members = ctx.abspath(pos[0]), pos[1:]
    files = ctx.node.files
    if "c" in flags:
        if not members:
            raise CommandError("tar: Cowardly refusing to create an empty archive", 2)
        bundle = {}
        for m in members:
            full = ctx.abspath(m)
            matched = {f: t for f, t in files.items() if f == full or f.startswith(full.rstrip("/") + "/")}
            if not matched and full not in known_dirs(ctx):
                raise CommandError(
                    f"tar: {m}: Cannot stat: No such file or directory\ntar: Exiting with failure status due to previous errors",
                    2,
                )
            bundle.update({f.lstrip("/"): t for f, t in matched.items()})
        files[archive] = "TAR:" + repr(bundle)
        return "".join(f"{n}\n" for n in bundle) if "v" in flags else ""
    if archive not in files or not files[archive].startswith("TAR:"):
        raise CommandError(
            f"tar: {pos[0]}: Cannot open: No such file or directory\ntar: Error is not recoverable: exiting now", 2
        )
    bundle = eval(files[archive][4:], {"__builtins__": {}})  # the archive is our own repr, see above
    if "t" in flags:
        return "".join(f"{n}\n" for n in bundle)
    if "x" in flags:
        base = ctx.abspath(into) if into else ctx.cwd
        for n, t in bundle.items():
            files[base.rstrip("/") + "/" + n] = t
        return "".join(f"{n}\n" for n in bundle) if "v" in flags else ""
    raise CommandError("tar: You must specify one of the '-Acdtrux', '--delete' or '--test-label' options", 2)


@command("gzip", "gunzip", help="gzip FILE | gunzip FILE.gz")
def gzip(ctx, args):
    pos = [a for a in args if not a.startswith("-")]
    if not pos:
        raise CommandError("gzip: compressed data not written to a terminal. Use -f to force compression.", 1)
    files = ctx.node.files
    for p in pos:
        full = ctx.abspath(p)
        if full not in files:
            raise CommandError(f"gzip: {p}: No such file or directory", 1)
        if "-d" in args or full.endswith(".gz"):
            if not full.endswith(".gz"):
                raise CommandError(f"gzip: {p}: unknown suffix -- ignored", 2)
            files[full[:-3]] = files.pop(full)
        else:
            files[full + ".gz"] = files.pop(full)
    return ""


@command("jq", help="jq FILTER [FILE]: . and .key paths")
def jq(ctx, args):
    import json

    pos = [a for a in args if not a.startswith("-")]
    if not pos:
        raise CommandError("jq - commandline JSON processor\nUsage:\tjq [OPTIONS] FILTER [FILE...]", 2)
    text = _file(ctx, pos[1], "jq") if len(pos) > 1 else ctx.stdin
    try:
        data = json.loads(text)
    except ValueError as e:
        raise CommandError(f"jq: error (at <stdin>:0): {e}", 5)
    for key in [k for k in pos[0].strip().split(".") if k]:
        m = re.fullmatch(r"(\w*)\[(\d*)\]", key)
        name, index = (m.group(1), m.group(2)) if m else (key, None)
        if name:
            if not isinstance(data, dict):
                raise CommandError(f'jq: error (at <stdin>:0): Cannot index {type(data).__name__} with "{name}"', 5)
            data = data.get(name)
        if index is not None and isinstance(data, list):
            data = data[int(index)] if index else data
    compact = "-c" in args
    return (json.dumps(data) if compact else json.dumps(data, indent=2)) + "\n"


@command("nproc", help="nproc: number of CPUs")
def nproc(ctx, args):
    return f"{int(ctx.node.get('cpus', 224) or 224)}\n"


def quote(args: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in args)
