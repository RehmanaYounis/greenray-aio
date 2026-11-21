"""Generic Linux commands: files, text filters, services, logs, modules, ssh."""

from __future__ import annotations

import contextlib
import fnmatch
import re
import shlex
import time
from datetime import timedelta

from ..core import audit, clock
from ..core.errors import CommandError
from . import command
from .common import int_arg, opt

# ---------------------------------------------------------------- files


def virtual_files(ctx) -> dict[str, str]:
    """Files the kernel presents from live state: driver version, uptime, the GPU device nodes."""
    from ..core import clock

    node = ctx.node
    out: dict[str, str] = {}
    age = clock.now(ctx.state) - clock.booted(ctx.state)
    out["/proc/uptime"] = f"{age.total_seconds():.2f} {age.total_seconds() * 180:.2f}\n"
    out["/proc/loadavg"] = "0.08 0.05 0.01 1/2210 48213\n"
    if node.get("gpus") and node.get("driver_loaded", True):
        out["/proc/driver/nvidia/version"] = (
            f"NVRM version: NVIDIA UNIX x86_64 Kernel Module  {node.driver_version}  Thu Jun  6 15:21:04 UTC 2024\n"
            "GCC version:  gcc version 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04)\n"
        )
        for g in node.gpus:
            out[f"/dev/nvidia{g.index}"] = ""
        out["/dev/nvidiactl"] = ""
        out["/dev/nvidia-uvm"] = ""
        out["/dev/nvidia-uvm-tools"] = ""
        out["/dev/nvidia-modeset"] = ""
        for i in range(node.get("nvswitches", 0)):
            out[f"/dev/nvidia-nvswitch{i}"] = ""
    return out


def _read(ctx, path, tool="cat"):
    files = ctx.node.files
    if path not in files:
        virtual = virtual_files(ctx)
        if path in virtual:
            return virtual[path]
        raise CommandError(f"{tool}: {path}: No such file or directory", 2 if tool == "grep" else 1)
    return files[path]


@command("cat", help="print files (also: cat > file <<EOF ... EOF)")
def cat(ctx, args):
    text = (
        "".join(_read(ctx, a) for a in args if not a.startswith("-"))
        if [a for a in args if not a.startswith("-")]
        else ctx.stdin
    )
    if "-n" in args or "--number" in args:
        return "".join(f"{i:6}\t{line}" for i, line in enumerate(text.splitlines(True), 1))
    return text


def _ls_row(ctx, path, name):
    """One ``ls -l`` line; sizes come from the simulated file, dates from the simulated clock."""
    dirs = known_dirs(ctx)
    is_dir = path.rstrip("/") in dirs and path not in ctx.node.files
    size = 4096 if is_dir else len(ctx.node.files.get(path, ""))
    mode = "drwxr-xr-x" if is_dir else ("-rwxr-xr-x" if name.endswith(".sh") else "-rw-r--r--")
    stamp = clock.stamp(ctx.state, "%b %e %H:%M")
    if path.startswith("/dev/nvidia"):  # character devices: major 195 for the GPUs, 234 for the UVM driver
        tail = path[len("/dev/nvidia") :]
        major, minor = (195, int(tail)) if tail.isdigit() else (195, 255) if tail == "ctl" else (234, 0)
        return f"crw-rw-rw- 1 root root {major:>3}, {minor:>3} {stamp} {name}"
    return f"{mode} 1 root root {size:>8} {stamp} {name}"


@command("ls", help="list files (-a shows dotfiles, -l long form)")
def ls(ctx, args):
    flags = "".join(a[1:] for a in args if a.startswith("-") and not a.startswith("--"))
    show_all, long = "a" in flags or "--all" in args, "l" in flags
    paths = [a for a in args if not a.startswith("-")] or [ctx.cwd]
    all_files = {**virtual_files(ctx), **ctx.node.files}
    files = sorted(all_files)
    dirs = known_dirs(ctx)
    out = []
    base = ctx.cwd.rstrip("/") + "/"
    for p in paths:
        if p in all_files:
            shown = p[len(base) :] if p.startswith(base) else p  # a file under the cwd is listed by name
            out.append(_ls_row(ctx, p, shown) if long else shown)
            continue
        prefix = p.rstrip("/") + "/"
        children = {f[len(prefix) :].split("/")[0] for f in files if f.startswith(prefix)}
        children |= {d[len(prefix) :].split("/")[0] for d in dirs if d.startswith(prefix) and d != prefix}
        globbed = [f for f in files if fnmatch.fnmatch(f, p)]
        if not children and not globbed and p.rstrip("/") not in dirs:
            raise CommandError(f"ls: cannot access '{p}': No such file or directory", 2)
        if not show_all:
            children = {c for c in children if not c.startswith(".")}
        names = sorted(children) or globbed
        if long:
            out.extend(_ls_row(ctx, prefix + n if n not in files else n, n) for n in names)
            continue
        if ctx.colorize:  # ls --color=auto: directories bold blue, scripts bold green
            names = [
                f"\033[01;34m{n}\033[0m"
                if prefix + n in dirs
                else (f"\033[01;32m{n}\033[0m" if n.endswith(".sh") else n)
                for n in names
            ]
        out.extend(names)
    return "\n".join(out) + ("\n" if out else "")


# Directories that exist on every simulated host even when empty.
DEFAULT_DIRS = {"/", "/root", "/etc", "/tmp", "/home", "/var/log", "/etc/modules-load.d", "/etc/profile.d"}


def known_dirs(ctx) -> set[str]:
    """Explicit directories (mkdir, scenario setup) plus every parent of an existing file."""
    dirs = set(DEFAULT_DIRS) | {d.rstrip("/") or "/" for d in ctx.node.get("dirs", [])}
    for f in list(ctx.node.files) + list(virtual_files(ctx)):
        parts = f.split("/")[:-1]
        for i in range(2, len(parts) + 1):
            dirs.add("/".join(parts[:i]))
    return dirs


@command("echo", help="print text (use > or >> to write files)")
def echo(ctx, args):
    if args and args[0] == "-e":
        args = args[1:]
        return " ".join(args).encode().decode("unicode_escape") + "\n"
    if args and args[0] == "-n":
        return " ".join(args[1:])
    return " ".join(args) + "\n"


@command("tee", help="write stdin to a file")
def tee(ctx, args):
    append = "-a" in args
    targets = [ctx.abspath(a) for a in args if not a.startswith("-")]
    for t in targets:
        ctx.node.files[t] = (ctx.node.files.get(t, "") if append else "") + ctx.stdin
    return ctx.stdin


@command("touch")
def touch(ctx, args):
    for a in args:
        if not a.startswith("-"):
            ctx.node.files.setdefault(ctx.abspath(a), "")
    return ""


@command("rm")
def rm(ctx, args):
    force = any(a.startswith("-") and "f" in a for a in args)
    recursive = any(a.startswith("-") and ("r" in a or "R" in a) for a in args)
    for a in args:
        if a.startswith("-"):
            continue
        path = ctx.abspath(a)
        inside = [f for f in ctx.node.files if f.startswith(path.rstrip("/") + "/")]
        if path in ctx.node.files:
            del ctx.node.files[path]
        elif inside and recursive:
            for f in inside:
                del ctx.node.files[f]
        elif inside:
            raise CommandError(f"rm: cannot remove '{a}': Is a directory", 1)
        elif not force:
            raise CommandError(f"rm: cannot remove '{a}': No such file or directory", 1)
    return ""


@command("cp")
def cp(ctx, args):
    operands = [a for a in args if a and not a.startswith("-")]
    if not operands:
        raise CommandError("cp: missing file operand\nTry 'cp --help' for more information.")
    if len(operands) == 1:
        raise CommandError(
            f"cp: missing destination file operand after '{operands[0]}'\nTry 'cp --help' for more information."
        )
    src, dst = operands[:2]
    dst = ctx.abspath(dst)
    if dst in known_dirs(ctx):
        dst = dst.rstrip("/") + "/" + src.rsplit("/", 1)[-1]
    ctx.node.files[dst] = _read(ctx, src)
    return ""


@command("mkdir")
def mkdir(ctx, args):
    parents = any(a in ("-p", "--parents") for a in args)
    dirs = ctx.node.setdefault("dirs", [])
    for a in args:
        if a.startswith("-"):
            continue
        path = ctx.abspath(a)
        existing = known_dirs(ctx)
        if path in existing or path in ctx.node.files:
            if not parents:
                raise CommandError(f"mkdir: cannot create directory ‘{a}’: File exists", 1)
            continue
        parent = path.rsplit("/", 1)[0] or "/"
        if parent not in existing and not parents:
            raise CommandError(f"mkdir: cannot create directory ‘{a}’: No such file or directory", 1)
        dirs.append(path)
    return ""


@command("cd", help="cd [DIR]: change the working directory (default: home)")
def cd(ctx, args):
    if args == ["-"]:  # back to the previous directory, which bash prints
        ctx.cwd, ctx.oldcwd = ctx.oldcwd, ctx.cwd
        return ctx.cwd + "\n"
    target = next((a for a in args if a != "--" and not a.startswith("-")), "~")
    path = ctx.abspath(target)
    if path in ctx.node.files:
        raise CommandError(f"bash: cd: {target}: Not a directory")
    if path not in known_dirs(ctx):
        raise CommandError(f"bash: cd: {target}: No such file or directory")
    ctx.oldcwd, ctx.cwd = ctx.cwd, path
    return ""


@command("pwd", help="pwd: print the working directory")
def pwd(ctx, args):
    return ctx.cwd + "\n"


@command("chmod", "chown", "sync")
def noop(ctx, args):
    return ""


@command("sed", help="sed [-n] [-i] [-E] 's/old/new/[g]' | '/re/d' | 'Np' | 'Nd' [file]  (-i edits in place)")
def sed(ctx, args):
    inplace = quiet = ere = False
    exprs, files = [], []
    it = iter(args)
    for a in it:
        if a == "-i" or a.startswith("-i"):
            inplace = True
        elif a == "-n":
            quiet = True
        elif a in ("-E", "-r", "--regexp-extended"):
            ere = True
        elif a == "-e":
            e = next(it, None)
            if e is None:
                raise CommandError(
                    "sed: option requires an argument -- 'e'\nUsage: sed [OPTION]... {script-only-if-no-other-script} [input-file]..."
                )
            exprs.append(e)
        elif a.startswith("-") and len(a) > 1 and not exprs:
            raise CommandError(f"sed: invalid option -- '{a[1:]}'")
        elif not exprs and not a.startswith("-"):
            exprs.append(a)
        else:
            files.append(a)
    if not exprs:
        raise CommandError("Usage: sed [OPTION]... {script-only-if-no-other-script} [input-file]...")
    text = _read(ctx, files[0]) if files else ctx.stdin
    for e in exprs:
        text = _apply_sed(e, text, quiet=quiet, ere=ere)
    if inplace and files:
        ctx.node.files[files[0]] = text
        return ""
    return text


def _bre_to_python(pat: str) -> str:
    r"""POSIX basic regex (sed's default) -> Python: \( \) \{ \} \| \+ \? are special, the bare ones literal."""
    out, i = [], 0
    while i < len(pat):
        c = pat[i]
        if c == "\\" and i + 1 < len(pat):
            n = pat[i + 1]
            out.append(n if n in "(){}|+?" else c + n)
            i += 2
            continue
        out.append("\\" + c if c in "(){}|+?" else c)
        i += 1
    return "".join(out)


def _split_unescaped(s: str, delim: str) -> list[str]:
    parts, cur, i = [], [], 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            # an escaped delimiter is the delimiter itself; other escapes stay for the regex engine
            cur.append(s[i + 1] if s[i + 1] == delim else c + s[i + 1])
            i += 2
            continue
        if c == delim:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    parts.append("".join(cur))
    return parts


def _apply_sed(expr: str, text: str, quiet: bool = False, ere: bool = False) -> str:
    m = re.match(
        r"^(?:/(?P<addr>(?:[^/\\]|\\.)*)/|(?P<line>\d+)(?:,(?P<upto>\d+|\$))?|(?P<last>\$))?(?P<cmd>[sdp])(?P<rest>.*)$",
        expr,
    )
    if not m:
        raise CommandError(f"sed: -e expression #1, char {len(expr)}: unknown command: `{expr[:1]}'")
    conv = (lambda s: s) if ere else _bre_to_python
    addr, line_no, last, upto = m.group("addr"), m.group("line"), m.group("last"), m.group("upto")
    lines = text.splitlines(True)

    def selected(i: int, line: str) -> bool:
        if addr is not None:
            return re.search(conv(addr), line) is not None
        if line_no is not None:
            if upto is not None:
                return int(line_no) <= i <= (len(lines) if upto == "$" else int(upto))
            return i == int(line_no)
        if last:
            return i == len(lines)
        return True

    try:
        if m.group("cmd") == "d":
            return "".join(l for i, l in enumerate(lines, 1) if not selected(i, l))
        if m.group("cmd") == "p":
            picked = [l for i, l in enumerate(lines, 1) if selected(i, l)]
            return "".join(picked) if quiet else "".join(l + l if selected(i, l) else l for i, l in enumerate(lines, 1))
        rest = m.group("rest")
        if not rest:
            raise CommandError("sed: -e expression #1, char 1: unterminated `s' command")
        parts = _split_unescaped(rest[1:], rest[0])
        if len(parts) < 3:
            raise CommandError(f"sed: -e expression #1, char {len(expr)}: unterminated `s' command")
        pat, rep, flags = conv(parts[0]), parts[1], parts[2]
        rep = re.sub(r"\\(\d)", r"\\g<\1>", rep).replace("&", r"\g<0>")
        count = 0 if "g" in flags else 1
        out = []
        for i, line in enumerate(lines, 1):
            if selected(i, line):
                line = re.sub(pat, rep, line, count=count)
            if not quiet or ("p" in flags and selected(i, line)):
                out.append(line)
        return "".join(out)
    except re.error as e:
        raise CommandError(f"sed: -e expression #1, char {len(expr)}: {e}") from None


# ---------------------------------------------------------------- text filters


@command("grep", "egrep", help="filter lines (-i -v -c -n -o -l -w -E -e -A/-B/-C supported)")
def grep(ctx, args):
    flags, patterns, files = set(), [], []
    after = before = 0
    explicit = []  # -e PATTERN, wherever it appears: every positional is then a file
    rest, it = [], iter(args)
    for a in it:
        if a in ("-e", "--regexp"):
            explicit.append(next(it, ""))
        elif a.startswith("--regexp="):
            explicit.append(a.split("=", 1)[1])
        else:
            rest.append(a)
    patterns.extend(explicit)
    it = iter(rest)
    for a in it:
        ctx_opt = re.fullmatch(r"-([ABC])(\d*)|--(after|before)-context=(\d+)|--context=(\d+)", a)
        if a == "--":
            files.extend(it)
        elif ctx_opt and not patterns:
            which = ctx_opt.group(1) or {"after": "A", "before": "B"}.get(ctx_opt.group(3) or "", "C")
            raw = ctx_opt.group(2) or ctx_opt.group(4) or ctx_opt.group(5) or next(it, "")
            if not raw.isdigit():
                raise CommandError(f"grep: {raw}: invalid context length argument", 2)
            n = int(raw)
            after = n if which in ("A", "C") else after
            before = n if which in ("B", "C") else before
        elif a.startswith("--") and len(a) > 2:
            flags.add(
                {
                    "ignore-case": "i",
                    "invert-match": "v",
                    "count": "c",
                    "line-number": "n",
                    "only-matching": "o",
                    "files-with-matches": "l",
                    "word-regexp": "w",
                    "extended-regexp": "E",
                    "fixed-strings": "F",
                    "no-filename": "h",
                    "with-filename": "H",
                }.get(a[2:], a[2:])
            )
        elif a.startswith("-") and len(a) > 1:
            flags.update(a[1:])
        elif not patterns:
            patterns.append(a)
        else:
            files.append(a)
    if not patterns:
        raise CommandError("Usage: grep [OPTION]... PATTERNS [FILE]...\nTry 'grep --help' for more information.", 2)
    pats = [re.escape(p) if "F" in flags else p for p in patterns]
    if "w" in flags:
        pats = [rf"\b(?:{p})\b" for p in pats]
    try:
        rx = re.compile("|".join(f"(?:{p})" for p in pats), re.I if "i" in flags else 0)
    except re.error as e:
        raise CommandError(f"grep: {e}", 2) from None
    sources = [(f, _read(ctx, f, "grep")) for f in files] if files else [(None, ctx.stdin)]
    with_name = ("H" in flags or len(sources) > 1) and "h" not in flags
    out, total, matched_files = [], 0, []
    for name, text in sources:
        lines = text.splitlines()
        hits = [i for i, l in enumerate(lines) if bool(rx.search(l)) != ("v" in flags)]
        total += len(hits)
        if hits:
            matched_files.append(name or "(standard input)")
        if "c" in flags or "l" in flags:
            if "c" in flags:
                out.append(f"{name}:{len(hits)}" if with_name else str(len(hits)))
            continue
        prefix = f"{name}:" if with_name else ""
        if "o" in flags:
            for i in hits:
                for m in rx.finditer(lines[i]):
                    out.append(prefix + (f"{i + 1}:" if "n" in flags else "") + m.group(0))
            continue
        keep = sorted({j for i in hits for j in range(max(0, i - before), min(len(lines), i + after + 1))})
        for j in keep:
            line = lines[j]
            if ctx.colorize and "v" not in flags:  # Ubuntu aliases grep to grep --color=auto
                line = rx.sub(lambda m: f"\033[01;31m\033[K{m.group(0)}\033[m\033[K", line)
            sep = ":" if j in hits else "-"
            out.append(prefix + (f"{j + 1}{sep}" if "n" in flags else "") + line)
    if "l" in flags:
        out = matched_files
    if total == 0:  # grep prints its counts but still exits 1 when nothing matched
        raise CommandError("\n".join(out), 1)
    return "\n".join(out) + "\n"


@command("head")
def head(ctx, args):
    n = _num_arg(args, 10, "head")
    return "\n".join(_input(ctx, args).splitlines()[:n]) + "\n"


@command("tail")
def tail(ctx, args):
    lines = _input(ctx, args).splitlines()
    for i, a in enumerate(args):
        if a == "-n" and i + 1 < len(args) and args[i + 1].startswith("+") and args[i + 1][1:].isdigit():
            return "\n".join(lines[int(args[i + 1][1:]) - 1 :]) + "\n"
    n = _num_arg(args, 10, "tail")
    return "\n".join(lines[-n:]) + "\n"


@command("wc")
def wc(ctx, args):
    text = _input(ctx, args, "wc")
    counts = {"l": len(text.splitlines()), "w": len(text.split()), "c": len(text.encode()), "m": len(text)}
    want = [c for a in args if a.startswith("-") and not a.startswith("--") for c in a[1:] if c in counts]
    want += [a[2:][0] for a in args if a in ("--lines", "--words", "--bytes", "--chars")]
    if not want:
        want = ["l", "w", "c"]
    return " ".join(str(counts[c]) for c in want) + "\n"


@command("sort")
def sort_(ctx, args):
    flags = "".join(a[1:] for a in args if a.startswith("-") and not a.startswith("--") and a not in ("-k", "-t"))
    key_spec, sep = opt(args, "-k", "--key"), opt(args, "-t", "--field-separator")
    numeric, unique, rev = "n" in flags or "--numeric-sort" in args, "u" in flags or "--unique" in args, "r" in flags
    if key_spec and not re.fullmatch(r"\d+(,\d+)?[nr]*", key_spec):
        raise CommandError(f"sort: invalid key {key_spec!r}", 2)
    lines = _input(ctx, args, "sort", value_flags=("-k", "-t")).splitlines()
    field = int(key_spec.split(",")[0]) if key_spec else None
    numeric = numeric or bool(key_spec and "n" in key_spec)
    rev = rev or bool(key_spec and "r" in key_spec)

    def num(s):
        m = re.match(r"\s*(-?\d+(?:\.\d+)?)", s)
        return float(m.group(1)) if m else 0.0

    def key(line):
        part = line
        if field:
            cols = line.split(sep) if sep else line.split()
            part = cols[field - 1] if field <= len(cols) else ""
        return num(part) if numeric else part

    lines.sort(key=key, reverse=rev)
    if unique:
        seen, kept = set(), []
        for line in lines:
            k = key(line)
            if k not in seen:
                seen.add(k)
                kept.append(line)
        lines = kept
    return "\n".join(lines) + "\n"


@command("awk", help="awk [-F SEP] 'PATTERN {print $N, $NF, NR} END {print ...}' [FILE]")
def awk(ctx, args):
    return _awk(ctx, args)


def _awk(ctx, args):
    """The subset administrators type at a prompt: field printing with a regex or comparison guard, NR and END."""
    sep = opt(args, "-F")
    rest = [a for a in args if a not in ("-F", sep)]
    rest = [a for a in rest if not (a.startswith("-F") and len(a) > 2)]
    if sep is None:
        packed = next((a for a in args if a.startswith("-F") and len(a) > 2), None)
        sep = packed[2:] if packed else None
    prog = rest[0] if rest else ""
    files = rest[1:]
    text = "".join(_read(ctx, f, "awk") for f in files) if files else ctx.stdin
    lines = text.splitlines()
    m = re.fullmatch(
        r"\s*(?:BEGIN\s*\{(?P<begin>[^}]*)\}\s*)?(?:(?P<guard>[^{}]+?)\s*)?(?:\{(?P<body>[^}]*)\})?\s*(?:END\s*\{(?P<end>[^}]*)\}\s*)?",
        prog,
    )
    if not m or not prog.strip():
        raise CommandError(f"awk: cmd. line:1: {prog}\nawk: cmd. line:1: ^ syntax error", 2)
    begin, guard, body, end = m.group("begin"), (m.group("guard") or "").strip(), m.group("body"), m.group("end")
    if body is None and guard:
        body = "print"
    total = {"sum": 0.0}
    out = []

    def fields_of(line):
        return line.split(sep) if sep else line.split()

    def value(tok, cols, nr):
        tok = tok.strip()
        if tok == "NR":
            return str(nr)
        if tok == "NF":
            return str(len(cols))
        if tok == "$0":
            return line_text[0]
        if tok == "$NF":
            return cols[-1] if cols else ""
        mm = re.fullmatch(r"\$(\d+)", tok)
        if mm:
            i = int(mm.group(1))
            return cols[i - 1] if 0 < i <= len(cols) else (line_text[0] if i == 0 else "")
        if tok.startswith('"') and tok.endswith('"'):
            return tok[1:-1]
        if tok == "sum":
            return str(int(total["sum"]) if total["sum"] == int(total["sum"]) else total["sum"])
        return tok

    def guard_ok(line, cols, nr):
        if not guard:
            return True
        if guard.startswith("/") and guard.endswith("/"):
            return re.search(guard[1:-1], line) is not None
        if guard.startswith("!/") and guard.endswith("/"):
            return re.search(guard[2:-1], line) is None
        mm = re.fullmatch(r"(.+?)\s*(==|!=|>=|<=|>|<|~|!~)\s*(.+)", guard)
        if not mm:
            raise CommandError(f"awk: cmd. line:1: {guard}\nawk: cmd. line:1: ^ unsupported pattern", 2)
        a, op, b = value(mm.group(1), cols, nr), mm.group(2), value(mm.group(3), cols, nr)
        if op in ("~", "!~"):
            hit = re.search(b.strip("/"), a) is not None
            return hit if op == "~" else not hit
        try:
            fa, fb = float(a), float(b)
            a, b = fa, fb
        except ValueError:
            pass
        return {"==": a == b, "!=": a != b, ">=": a >= b, "<=": a <= b, ">": a > b, "<": a < b}[op]

    def run_body(code, cols, nr):
        for stmt in [s.strip() for s in code.split(";") if s.strip()]:
            if stmt == "print":
                out.append(line_text[0])
            elif stmt.startswith("print"):
                expr = stmt[5:].strip()
                parts = [p for p in re.split(r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", expr)] if expr else ["$0"]
                out.append(" ".join(value(p, cols, nr) for p in parts))
            elif re.fullmatch(r"sum\s*\+=\s*\S+", stmt):
                with contextlib.suppress(ValueError):
                    total["sum"] += float(value(stmt.split("+=")[1], cols, nr) or 0)
            else:
                raise CommandError(f"awk: cmd. line:1: {stmt}\nawk: cmd. line:1: ^ unsupported statement", 2)

    line_text = [""]
    if begin:
        run_body(begin, [], 0)
    for nr, line in enumerate(lines, 1):
        line_text[0] = line
        cols = fields_of(line)
        if guard_ok(line, cols, nr) and body:
            run_body(body, cols, nr)
    if end:
        line_text[0] = lines[-1] if lines else ""
        run_body(end, fields_of(line_text[0]), len(lines))
    return "\n".join(out) + "\n" if out else ""


def _awk_old(ctx, args):
    prog = next((a for a in args if not a.startswith("-")), "")
    fields = [int(x) for x in re.findall(r"\$(\d+)", prog)]
    out = []
    for line in ctx.stdin.splitlines():
        cols = line.split()
        out.append(" ".join(cols[f - 1] if f and f <= len(cols) else (line if f == 0 else "") for f in fields))
    return "\n".join(out) + "\n"


def _num_arg(args, default, tool="head"):
    for i, a in enumerate(args):
        if a == "-n" and i + 1 < len(args):
            v = args[i + 1]
            if not v.isdigit():
                raise CommandError(f"{tool}: invalid number of lines: '{v}'")
            return int(v)
        if re.fullmatch(r"-\d+", a):
            return int(a[1:])
    return default


def _input(ctx, args, tool="cat", value_flags=("-n",)):
    files, it = [], iter(args)
    for a in it:
        if a in value_flags:  # an option value, not a file
            next(it, None)
        elif not a.startswith("-") and not a.isdigit():
            files.append(a)
    return "".join(_read(ctx, f, tool) for f in files) if files else ctx.stdin


# ---------------------------------------------------------------- hosts


@command("hostname")
def hostname(ctx, args):
    return ctx.host + "\n"


@command("ssh", help="ssh <host> — move your shell to another node")
def ssh(ctx, args):
    # skip ssh options before the host; everything after the host is the remote command
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i] in ("-p", "-i", "-l", "-o", "-J") else 1
    if i >= len(args):
        raise CommandError("usage: ssh <host> [command]", 255)
    target = args[i].split("@")[-1]
    remote = args[i + 1 :]
    if target not in ctx.state.nodes:
        raise CommandError(f"ssh: Could not resolve hostname {target}: Name or service not known", 255)
    node = ctx.state.nodes[target]
    if not node.up or node.power != "on":
        raise CommandError(f"ssh: connect to host {target} port 22: No route to host", 255)
    if remote:  # ssh host cmd ...
        import shlex

        from ..core.shell import Shell

        sub = Shell(ctx.state, target)
        sub.ctx.executed = ctx.executed
        sub.ctx.outputs = ctx.outputs
        # one quoted argument is a command line for the remote shell (redirects and pipes included)
        line = remote[0] if len(remote) == 1 else " ".join(shlex.quote(a) for a in remote)
        return sub.run(line)
    source = ctx.state.nodes[ctx.host]
    ctx.saved_cwd[len(ctx.host_stack)] = ctx.cwd
    ctx.host_stack.append(ctx.host)
    ctx.host = target
    ctx.cwd = "/root"
    return motd(ctx.state, node, source)


def motd(state, node, source) -> str:
    """Login banner in the style of Ubuntu / DGX OS (generic text, no vendor artwork)."""
    gpus = node.get("gpus") or []
    lines = [f"Welcome to {node.os} (GNU/Linux {node.kernel} x86_64)", ""]
    lines.append(f" * System:      {node.get('model', 'server')}")
    if gpus:
        lines.append(f" * GPUs:        {len(gpus)} x {gpus[0].name}  (driver {node.get('driver_version', '?')})")
    lines.append(" * Management:  Base Command Manager")
    lines += ["", f"Last login: {clock.stamp(state)} from {source.get('ip', source.hostname)}"]
    return "\n".join(lines) + "\n"


@command("exit", "logout")
def exit_(ctx, args):
    from .slurm.jobs import leave_allocation

    released = leave_allocation(ctx)
    if released is not None:
        return released
    if ctx.host_stack:
        ctx.host = ctx.host_stack.pop()
        ctx.cwd = ctx.saved_cwd.pop(len(ctx.host_stack), "/root")
        return "logout\n"
    raise CommandError("(you are on the head node; use 'quit' to leave the lab)", 0)


@command("uptime")
def uptime(ctx, args):
    busy = sum(g.utilization for g in ctx.node.gpus) / 100 if ctx.node.gpus else 0.0
    load = f"{busy + 0.08:.2f}, {busy * 0.8 + 0.05:.2f}, {busy * 0.5 + 0.01:.2f}"
    return f" {clock.stamp(ctx.state, '%H:%M:%S')} up {clock.uptime_text(ctx.state)},  1 user,  load average: {load}\n"


@command("uname")
def uname(ctx, args):
    fields = {
        "s": "Linux",
        "n": ctx.host,
        "r": ctx.node.kernel,
        "v": "#42-Ubuntu SMP PREEMPT_DYNAMIC Tue Sep 10 11:07:16 UTC 2024",
        "m": "x86_64",
        "o": "GNU/Linux",
    }
    want = "".join(a[1:] for a in args if a.startswith("-") and not a.startswith("--"))
    if "a" in want or "--all" in args:
        want = "snrvmo"
    bad = set(want) - set(fields)
    if bad:
        raise CommandError(f"uname: invalid option -- '{bad.pop()}'\nTry 'uname --help' for more information.", 1)
    return " ".join(fields[k] for k in "snrvmo" if k in (want or "s")) + "\n"


@command("export", help="export VAR=value | export VAR | export -p")
def export(ctx, args):
    if not args or args == ["-p"]:
        return "".join(f'declare -x {k}="{v}"\n' for k, v in sorted(ctx.node.env.items()))
    for a in args:
        if "=" in a:
            k, v = a.split("=", 1)
            ctx.node.env[k] = v
        elif a not in ctx.node.env:
            ctx.node.env[a] = ""  # exporting an unset name marks it; its value stays empty
    return ""


@command("unset")
def unset(ctx, args):
    for a in args:
        ctx.node.env.pop(a, None)
    return ""


def _shell_env(ctx) -> dict:
    """What a process on the node sees: the node's variables plus the ones bash sets itself."""
    base = {
        "HOME": ctx.home,
        "USER": ctx.user,
        "LOGNAME": ctx.user,
        "HOSTNAME": ctx.host,
        "PWD": ctx.cwd,
        "SHELL": "/bin/bash",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/cuda/bin",
        "TERM": "xterm-256color",
        "LANG": "C.UTF-8",
    }
    base.update(ctx.node.env)
    return base


@command("env", "printenv", help="env | env VAR=value command | printenv [NAME]")
def env(ctx, args):
    from . import dispatch

    assignments: dict = {}
    rest = list(args)
    while rest and "=" in rest[0] and not rest[0].startswith("-"):
        k, v = rest.pop(0).split("=", 1)
        assignments[k] = v
    if assignments and rest:  # env VAR=x command: the variable lives only for that command
        saved = dict(ctx.node.env)
        ctx.node.env.update(assignments)
        try:
            return dispatch(ctx, rest)
        finally:
            ctx.node.env.clear()
            ctx.node.env.update(saved)
    table = _shell_env(ctx)
    table.update(assignments)
    if rest:
        missing = [a for a in rest if a not in table]
        text = "".join(table[a] + "\n" for a in rest if a in table)
        if missing:
            raise CommandError(text.rstrip("\n"), 1)
        return text
    return "".join(f"{k}={v}\n" for k, v in sorted(table.items()))


# ---------------------------------------------------------------- services


def _svc(ctx, name):
    name = name.removesuffix(".service")
    if name not in ctx.node.services:
        raise CommandError(f"Unit {name}.service could not be found.", 4)
    return name


def _service_hook(ctx, name, action):
    """Let other modules react to service restarts (e.g. slurmd re-reading gres.conf)."""
    from . import HOOKS_SERVICE

    for fn in HOOKS_SERVICE:
        fn(ctx, name, action)


@command("systemctl", help="systemctl status|start|stop|restart|enable|disable|is-active <unit>")
def systemctl(ctx, args):
    if not args:
        raise CommandError("usage: systemctl <action> <unit>")
    action, units = args[0], [a for a in args[1:] if not a.startswith("-")]
    if action == "list-units" or action == "list-unit-files":
        return "".join(f"{s}.service  loaded {st} running\n" for s, st in sorted(ctx.node.services.items()))
    if action == "daemon-reload":
        return ""
    out = []
    for u in units:
        name = _svc(ctx, u)
        st = ctx.node.services[name]
        if action == "status":
            sub = "running" if st == "active" else ("failed" if st == "failed" else "dead")
            enabled = "enabled" if ctx.node.services_enabled.get(name, True) else "disabled"
            dot, active = "●", f"{st} ({sub})"
            if ctx.colorize:  # systemctl colours the unit dot and the Active state
                colour = {"active": "1;32", "failed": "1;31"}.get(st)
                if colour:
                    dot, active = f"\033[{colour}m●\033[0m", f"\033[{colour}m{active}\033[0m"
            out.append(
                f"{dot} {name}.service\n     Loaded: loaded (/lib/systemd/system/{name}.service; {enabled})\n"
                f"     Active: {active}"
            )
            log = ctx.node.journal.get(name, [])
            if log:
                out.append("\n".join(log[-5:]))
        elif action == "is-active":
            out.append(st)
        elif action == "is-enabled":
            out.append("enabled" if ctx.node.services_enabled.get(name, True) else "disabled")
        elif action in ("start", "restart", "reload"):
            if action == "restart" and name in ("docker", "containerd") and ctx.node.get("containers"):
                audit.record(ctx.state, audit.RUNTIME_RESTART_WITH_CONTAINERS, ctx.host, f"systemctl restart {name}")
            ctx.node.services[name] = "active"
            _service_hook(ctx, name, action)
            if ctx.node.services[name] != "active":
                raise CommandError(
                    f"Job for {name}.service failed because the control process exited with error code.\n"
                    f'See "systemctl status {name}.service" and "journalctl -xeu {name}.service" for details.'
                )
        elif action == "stop":
            if name in ("docker", "containerd") and ctx.node.get("containers"):
                audit.record(ctx.state, audit.RUNTIME_RESTART_WITH_CONTAINERS, ctx.host, f"systemctl stop {name}")
            ctx.node.services[name] = "inactive"
            _service_hook(ctx, name, action)
        elif action in ("enable", "disable"):
            ctx.node.services_enabled[name] = action == "enable"
            if "--now" in args:
                ctx.node.services[name] = "active" if action == "enable" else "inactive"
                _service_hook(ctx, name, "start" if action == "enable" else "stop")
        else:
            raise CommandError(f"systemctl: unknown action {action}")
    return "\n".join(out) + ("\n" if out else "")


@command("service")
def service(ctx, args):
    if len(args) >= 2:
        return systemctl(ctx, [args[1], args[0]])
    raise CommandError("usage: service <name> <action>")


@command("journalctl", help="journalctl [-u UNIT] [-k] [-b] [-n N] [-p PRIO] [--since WHEN] [-f]")
def journalctl(ctx, args):
    unit, tail, prio, kernel, since = None, None, None, False, None
    it = iter(args)

    def unit_arg(flag):
        value = next(it, None)
        if value is None:  # getopt's complaint when the value is missing
            short = flag.lstrip("-")[-1]
            raise CommandError(
                f"journalctl: option '{flag}' requires an argument"
                if flag.startswith("--")
                else f"journalctl: option requires an argument -- '{short}'"
            )
        return value.removesuffix(".service")

    for a in it:
        if a in ("-u", "--unit"):
            unit = unit_arg(a)
        elif a.startswith("--unit="):
            unit = a.split("=", 1)[1].removesuffix(".service")
        elif a.startswith("-u") and len(a) > 2 and not a.startswith("--"):
            unit = a[2:].removesuffix(".service")
        elif a.startswith("-xeu") or a.startswith("-eu"):
            unit = unit_arg(a)
        elif a in ("-n", "--lines"):
            tail = int_arg(next(it, ""), "Failed to parse lines", 1, minimum=0)
        elif a.startswith("--lines="):
            tail = int_arg(a.split("=", 1)[1], "Failed to parse lines", 1, minimum=0)
        elif a.startswith("-n") and a[2:].isdigit():
            tail = int(a[2:])
        elif a in ("-p", "--priority"):
            prio = next(it, "")
        elif a.startswith("--priority="):
            prio = a.split("=", 1)[1]
        elif a in ("-k", "--dmesg"):
            kernel = True
        elif a in ("--since", "-S"):
            since = next(it, "")
        elif a.startswith("--since="):
            since = a.split("=", 1)[1]
        elif a in ("-b", "--boot", "-x", "-e", "-f", "--follow", "--no-pager", "-r", "--reverse", "-a", "--all"):
            continue
        elif a.startswith("-"):
            raise CommandError(f"journalctl: unrecognized option '{a}'", 1)
    if kernel:
        lines = [l for l in ctx.node.dmesg]
    elif unit:
        lines = list(ctx.node.journal.get(unit, []))
        if not lines and ctx.node.services.get(unit) == "active":
            # a unit that has run since boot has at least its start in the journal
            booted = clock.booted(ctx.state).strftime("%b %d %H:%M:%S")
            lines = [
                f"{booted} {ctx.host} systemd[1]: Starting {unit}.service...",
                f"{booted} {ctx.host} systemd[1]: Started {unit}.service.",
            ]
    else:
        lines = [l for log in ctx.node.journal.values() for l in log]
    if prio:
        levels = {
            "emerg": 0,
            "alert": 1,
            "crit": 2,
            "err": 3,
            "error": 3,
            "warning": 4,
            "warn": 4,
            "notice": 5,
            "info": 6,
            "debug": 7,
        }
        top = levels.get(prio.split("..")[-1].lower(), int(prio) if prio.isdigit() else 7)
        if top <= 4:
            lines = [
                l
                for l in lines
                if re.search(r"error|fail|fatal|warn|Xid|refused|denied|not responding|killed", l, re.I)
            ]
    if since:
        m = re.search(r"(\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2})?)", since)
        if m:
            lines = [
                l
                for l in lines
                if (lm := re.match(r"(\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2})?)", l)) is None or lm.group(1) >= m.group(1)
            ]
    if tail is not None:
        lines = lines[-tail:] if tail else []
    if "-r" in args or "--reverse" in args:
        lines = lines[::-1]
    return ("\n".join(lines) + "\n") if lines else "-- No entries --\n"


@command("dmesg", help="kernel ring buffer (look for NVRM: Xid); -T human times, -l LEVEL filters")
def dmesg(ctx, args):
    lines = list(ctx.node.dmesg)
    level = opt(args, "-l", "--level")
    if level:
        top = {"emerg": 0, "alert": 1, "crit": 2, "err": 3, "warn": 4, "notice": 5, "info": 6, "debug": 7}.get(
            level.split(",")[-1], 7
        )
        if top <= 4:
            lines = [l for l in lines if re.search(r"error|fail|fatal|warn|Xid|fallen off|AER|taint|timeout", l, re.I)]
    if "-T" in args or "--ctime" in args:
        boot = clock.booted(ctx.state)

        def human(m):
            return "[" + (boot + timedelta(seconds=float(m.group(1)))).strftime("%a %b %d %H:%M:%S %Y") + "]"

        lines = [re.sub(r"^\[\s*([\d.]+)\]", human, line) for line in lines]
    if ctx.colorize:  # dmesg --color=auto shows timestamps in green
        lines = [re.sub(r"^(\[\s*[\d.]+\])", "\033[32m\\1\033[0m", line) for line in lines]
    return "\n".join(lines) + "\n"


@command("lsmod")
def lsmod(ctx, args):
    rows = ["Module                  Size  Used by"]
    rows += [f"{m:<22} 65536  0" for m in ctx.node.modules]
    return "\n".join(rows) + "\n"


# Modules the DGX OS image ships; anything else fails the way modprobe does on a real node.
KNOWN_MODULES = frozenset(
    {
        "nvidia",
        "nvidia_uvm",
        "nvidia_modeset",
        "nvidia_drm",
        "nvidia_peermem",
        "nvidia_fs",
        "mlx5_core",
        "mlx5_ib",
        "ib_core",
        "ib_uverbs",
        "ib_umad",
        "rdma_ucm",
        "rdma_cm",
        "ib_ipoib",
        "nvme",
        "nvme_tcp",
        "overlay",
        "br_netfilter",
        "ipmi_devintf",
        "ipmi_si",
        "gdrdrv",
        "nfs",
        "lustre",
    }
)


# which loaded modules hold a reference to which (lsmod's "Used by" column, and what -r refuses)
MODULE_DEPS = {
    "nvidia_uvm": ["nvidia"],
    "nvidia_modeset": ["nvidia"],
    "nvidia_drm": ["nvidia_modeset"],
    "nvidia_peermem": ["nvidia", "ib_core"],
    "mlx5_ib": ["mlx5_core", "ib_core"],
    "ib_uverbs": ["ib_core"],
    "rdma_ucm": ["ib_core"],
}


@command("modprobe", help="modprobe [-r] <module>")
def modprobe(ctx, args):
    remove = "-r" in args or "--remove" in args
    # the kernel treats '-' and '_' in module names as the same character
    mods = [a.replace("-", "_") for a in args if not a.startswith("-")]
    if not mods:
        raise CommandError("modprobe: ERROR: missing parameters. See -h.", 1)
    for m in mods:
        if m not in KNOWN_MODULES and m not in ctx.node.modules:
            raise CommandError(f"modprobe: FATAL: Module {m} not found in directory /lib/modules/{ctx.node.kernel}", 1)
        if remove:
            if m == "nvidia" and any(g.processes for g in ctx.node.gpus):
                raise CommandError("modprobe: FATAL: Module nvidia is in use.")
            holders = [d for d, deps in MODULE_DEPS.items() if m in deps and d in ctx.node.modules]
            if holders:
                raise CommandError(f"modprobe: FATAL: Module {m} is in use.")
            if m in ctx.node.modules:
                ctx.node.modules.remove(m)
            if m == "nvidia":
                ctx.node.driver_loaded = False
        else:
            if m not in ctx.node.modules:
                ctx.node.modules.append(m)
            if m == "nvidia":
                ctx.node.driver_loaded = True
    return ""


@command("lspci", help="lspci | grep -i nvidia")
def lspci(ctx, args):
    rows = []
    for g in ctx.node.gpus:
        if g.fallen_off_bus:
            continue
        rows.append(f"{g.bus_id[9:]} 3D controller: NVIDIA Corporation GH100 [H100 SXM5 80GB] (rev a1)")
    for i in range(ctx.node.nvswitches):
        rows.append(f"{0x05 + i:02x}:00.0 Bridge: NVIDIA Corporation Device 22a3 (rev a1)")
    for p in ctx.node.ib:
        rows.append(f"{p.ca}: Infiniband controller: Mellanox Technologies MT2910 Family [ConnectX-7]")
    return "\n".join(rows) + "\n"


@command("reboot", help="reboot the current node (clears transient GPU faults)")
def reboot(ctx, args):
    from . import HOOKS_REBOOT

    node = ctx.node
    jobs = audit.running_jobs_on(ctx.state, ctx.host)
    if jobs:
        audit.record(ctx.state, audit.REBOOT_WITH_JOBS, ctx.host, "reboot", ", ".join(str(j.id) for j in jobs))
    for fn in HOOKS_REBOOT:
        fn(ctx, node)
    msg = f"Connection to {ctx.host} closed by remote host. (node rebooted and is back up)\n"
    if ctx.host_stack:
        ctx.host = ctx.host_stack.pop()
        ctx.cwd = ctx.saved_cwd.pop(len(ctx.host_stack), "/root")
    return msg


@command("pdsh", help="pdsh -w <hostlist> [-R ssh] command: run a command on several nodes (BCM ships pdsh)")
def pdsh(ctx, args):
    """Run one command line on each node of a hostlist; output lines carry the node name, as pdsh prints them."""
    from .cmsh.model import bcm_init
    from .slurm.common import expand_hostlist

    hosts: list[str] = []
    excluded: list[str] = []
    rest: list[str] = []
    it = iter(args)
    for a in it:
        if a in ("-w", "--w"):
            hosts += expand_hostlist(next(it, ""))
        elif a.startswith("-w") and len(a) > 2:
            hosts += expand_hostlist(a[2:])
        elif a in ("-g",):  # a BCM category, through genders-style groups
            cat = next(it, "")
            bcm = bcm_init(ctx.state)
            hosts += [d for d, v in bcm.devices.items() if v.get("category") == cat]
        elif a in ("-R", "-t", "-f", "-l"):
            next(it, None)
        elif a == "-x":
            excluded += expand_hostlist(next(it, ""))
        elif a in ("-S", "-b", "-N"):
            pass
        elif a == "-a":
            bcm = bcm_init(ctx.state)
            hosts += [d for d, v in bcm.devices.items() if v.type != "HeadNode"]
        else:
            rest.append(a)
    hosts = [h for h in dict.fromkeys(hosts) if h not in excluded]
    if not hosts:
        raise CommandError("pdsh@bcm-head: no remote hosts specified", 1)
    if not rest:
        raise CommandError("pdsh@bcm-head: interactive mode is not available in the lab; give a command", 1)
    from ..core.shell import Shell

    line = rest[0] if len(rest) == 1 else " ".join(shlex.quote(a) for a in rest)
    out = []
    for h in hosts:
        node = ctx.state.nodes.get(h)
        if node is None:
            out.append(f"pdsh@{ctx.host}: {h}: ssh exited with exit code 255")
            continue
        if not node.get("up", True) or node.get("power", "on") != "on":
            out.append(f"pdsh@{ctx.host}: {h}: ssh: connect to host {h} port 22: No route to host")
            continue
        sub = Shell(ctx.state, h)
        sub.ctx.executed = ctx.executed
        sub.ctx.outputs = ctx.outputs
        text = sub.run(line)
        out.extend(f"{h}: {ln}" for ln in (text.splitlines() or [""]))
    return "\n".join(out) + "\n"


@command("last", help="last [-n N]: recent logins")
def last(ctx, args):
    from ..core import clock

    n = _num_arg(args, 10, "last")
    now = clock.now(ctx.state)
    booted = clock.booted(ctx.state)
    rows = [
        f"root     pts/0        10.141.255.254   {now:%a %b %e %H:%M}   still logged in",
        f"root     pts/0        10.141.255.254   {now.replace(hour=max(0, now.hour - 1)):%a %b %e %H:%M} - {now:%H:%M}  (01:00)",
        f"reboot   system boot  {ctx.node.kernel:<16} {booted:%a %b %e %H:%M}   still running",
    ]
    return "\n".join(rows[:n]) + f"\n\nwtmp begins {booted:%a %b %e %H:%M:%S %Y}\n"


@command("ipmitool", help="ipmitool sel list | sdr | power status|on|off|cycle | chassis status")
def ipmitool(ctx, args):
    node = ctx.node
    if "chassis" in args:
        on = node.get("power", "on") == "on"
        return (
            f"System Power         : {'on' if on else 'off'}\nPower Overload       : false\n"
            "Power Interlock      : inactive\nMain Power Fault     : false\nPower Control Fault  : false\n"
            "Power Restore Policy : previous\nLast Power Event     : \nChassis Intrusion    : inactive\n"
            "Front-Panel Lockout  : inactive\nDrive Fault          : false\nCooling/Fan Fault    : "
            f"{'true' if any(f.get('status') != 'ok' for f in node.get('fans', [])) else 'false'}\n"
        )
    if "sel" in args:
        sel = node.get("sel", [])
        return ("\n".join(sel) + "\n") if sel else "SEL has no entries\n"
    if "sdr" in args:
        hot = max((g.temp_c for g in node.gpus), default=40)
        return (
            f"Inlet Temp       | 24 degrees C      | ok\n"
            f"GPU Max Temp     | {hot} degrees C      | {'cr' if hot >= 87 else 'ok'}\n"
            f"PSU1 Status      | 0x01              | ok\n"
            + "".join(
                f"{f['name']:<16} | {f['rpm']} RPM{' ' * max(1, 10 - len(str(f['rpm'])))}| {f['status']}\n"
                for f in node.get("fans", [{"name": "FAN1", "rpm": 9800, "status": "ok"}])
            )
        )
    if "power" in args:
        return f"Chassis Power is {node.power}\n"
    return "usage: ipmitool [sel list|sdr|power status]\n"


@command("ping")
def ping(ctx, args):
    host = next((a for a in args if not a.startswith("-") and not a.isdigit()), None)
    count = 1
    for i, a in enumerate(args):
        if a == "-c" and i + 1 < len(args):
            if not args[i + 1].isdigit() or int(args[i + 1]) < 1:
                raise CommandError(
                    f"ping: invalid argument: '{args[i + 1]}': out of range: 1 <= value <= 2147483647", 1
                )
            count = int(args[i + 1])
    node = ctx.state.nodes.get(host)
    if node is None:
        raise CommandError(f"ping: {host}: Name or service not known", 2)
    if not node.up:
        raise CommandError(f"PING {host}: Destination Host Unreachable", 1)
    lines = [f"PING {host} ({node.ip}) 56(84) bytes of data."]
    lines += [
        f"64 bytes from {host} ({node.ip}): icmp_seq={i} ttl=64 time=0.{112 + 7 * i % 40:03d} ms"
        for i in range(1, count + 1)
    ]
    lines += [
        "",
        f"--- {host} ping statistics ---",
        f"{count} packets transmitted, {count} received, 0% packet loss, time {(count - 1) * 1000}ms",
    ]
    return "\n".join(lines) + "\n"


@command("free")
def free(ctx, args):
    return (
        "               total        used        free\n"
        f"Mem:      {ctx.node.real_memory_mb * 1024:>12} {12 * 1024 * 1024:>11} {ctx.node.real_memory_mb * 1000:>11}\n"
    )


# ---------------------------------------------------------------- processes


@command("ps", help="ps aux — includes GPU processes")
def ps(ctx, args):
    rows = ["USER         PID %CPU %MEM COMMAND"]
    for svc, st in ctx.node.services.items():
        if st == "active":
            rows.append(f"root        {1000 + abs(hash(svc)) % 900:>4}  0.1  0.0 /usr/sbin/{svc}")
    for g in ctx.node.gpus:
        for p in g.processes:
            rows.append(f"{p.get('user', 'alice'):<10} {p.pid:>5} 98.0  1.2 {p.name}")
    return "\n".join(rows) + "\n"


@command("top", help="top: one snapshot of the busiest processes (batch mode)")
def top(ctx, args):
    node = ctx.node
    procs = [
        (1000 + abs(hash(svc)) % 900, "root", 0.1, 0.0, f"/usr/sbin/{svc}")
        for svc, st in node.services.items()
        if st == "active"
    ]
    procs += [(p.pid, p.get("user", "alice"), 98.0, 1.2, p.name) for g in node.gpus for p in g.processes]
    procs.sort(key=lambda r: -r[2])
    total_mib = node.real_memory_mb
    used_mib = 12 * 1024 + sum(int(g.memory_used_mib) for g in node.gpus) // 8
    busy = sum(g.utilization for g in node.gpus) / 100 if node.gpus else 0.0
    load = f"{busy + 0.08:.2f}, {busy * 0.8 + 0.05:.2f}, {busy * 0.5 + 0.01:.2f}"
    cpu_us = min(99.0, 0.3 + 12.0 * busy)
    lines = [
        f"top - {clock.stamp(ctx.state, '%H:%M:%S')} up {clock.uptime_text(ctx.state)},  1 user,  load average: {load}",
        f"Tasks: {len(procs) + 180} total,   {1 + sum(1 for r in procs if r[2] > 50)} running, {len(procs) + 179} sleeping,   0 stopped,   0 zombie",
        f"%Cpu(s): {cpu_us:4.1f} us,  0.2 sy,  0.0 ni, {99.8 - cpu_us:4.1f} id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st",
        f"MiB Mem : {total_mib:9.1f} total, {total_mib - used_mib:9.1f} free, {used_mib:9.1f} used, {2048:9.1f} buff/cache",
        f"MiB Swap: {0:9.1f} total, {0:9.1f} free, {0:9.1f} used. {total_mib - used_mib:9.1f} avail Mem",
        "",
        "    PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND",
    ]
    for pid, user, cpu, mem, name in procs:
        state = "R" if cpu > 50 else "S"
        lines.append(
            f"{pid:>7} {user:<9} 20   0 {'24.1g' if cpu > 50 else '1.2g':>7} {'18.3g' if cpu > 50 else '48m':>6} {'1.1g' if cpu > 50 else '12m':>6} {state} {cpu:5.1f} {mem:5.1f} {'14:21.07' if cpu > 50 else '0:01.12':>9} {name.rsplit('/', 1)[-1]}"
        )
    return "\n".join(lines) + "\n"


@command("kill", "pkill", help="kill [-9] <pid>")
def kill(ctx, args):
    targets = [a for a in args if not a.startswith("-")]
    found = False
    for g in ctx.node.gpus:
        keep = []
        for p in g.processes:
            if str(p.pid) in targets or p.name in targets:
                found = True
                g.memory_used_mib = max(0, g.memory_used_mib - p.used_mib)
            else:
                keep.append(p)
        g.processes = keep
        if not keep:
            g.utilization = 0
    if not found:
        raise CommandError(f"kill: ({' '.join(targets)}) - No such process")
    return ""


@command("man", help="man COMMAND: manual page")
def man(ctx, args):
    from ..content import helppages

    names = [a for a in args if not a.startswith("-") and not a.isdigit()]
    if not names:
        raise CommandError("What manual page do you want?\nFor example, try 'man man'.")
    text = helppages.manual(names[0])
    if text is None:
        raise CommandError(f"No manual entry for {names[0]}", 16)
    return text


@command("clear", help="clear the terminal screen")
def clear(ctx, args):
    return "\033[H\033[2J" if ctx.streaming else ""


@command("watch", help="watch [-n SECONDS] COMMAND: rerun a command and show its output")
def watch(ctx, args):
    from ..core.shell import Shell

    interval, rest = 2.0, list(args)
    if rest and rest[0] in ("-n", "--interval"):
        if len(rest) < 2:
            raise CommandError("watch: option requires an argument -- 'n'")
        try:
            interval = max(0.1, float(rest[1]))
        except ValueError:
            raise CommandError(f"watch: failed to parse argument: '{rest[1]}'")
        rest = rest[2:]
    elif rest and rest[0].startswith("-n"):
        interval, rest = float(rest[0][2:]), rest[1:]
    if not rest:
        raise CommandError("watch: no command given\nUsage: watch [-n seconds] command")
    command_line = " ".join(rest)
    sub = Shell(ctx.state, ctx.host, color=ctx.color)
    sub.ctx.cwd, sub.ctx.executed, sub.ctx.reader = ctx.cwd, ctx.executed, ctx.reader
    # Real watch runs forever; here it stops after a few refreshes or Ctrl-C so the lab never hangs.
    rounds = 5 if ctx.streaming else 1
    try:
        for i in range(rounds):
            header = f"Every {interval:.1f}s: {command_line}".ljust(60) + f"{ctx.host}: {clock.stamp(ctx.state)}"
            screen = ("\033[H\033[2J" if ctx.streaming else "") + header + "\n\n" + sub.run(command_line) + "\n"
            if not ctx.streaming:
                return screen
            ctx.emit(screen)
            if i < rounds - 1:
                time.sleep(interval if ctx.live_delay else 0)
                clock.advance(ctx.state, interval)
    except KeyboardInterrupt:
        pass
    return ""
