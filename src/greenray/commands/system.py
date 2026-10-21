"""Generic Linux commands: files, text filters, services, logs, modules, ssh."""

from __future__ import annotations

import fnmatch
import re
import time
from datetime import timedelta

from ..core import clock
from ..core.errors import CommandError
from . import command

# ---------------------------------------------------------------- files


def _read(ctx, path):
    files = ctx.node.files
    if path not in files:
        raise CommandError(f"cat: {path}: No such file or directory")
    return files[path]


@command("cat", help="print files (also: cat > file <<EOF ... EOF)")
def cat(ctx, args):
    if not args:
        return ctx.stdin
    return "".join(_read(ctx, a) for a in args if not a.startswith("-"))


@command("ls", help="list files")
def ls(ctx, args):
    paths = [a for a in args if not a.startswith("-")] or [ctx.cwd]
    files = sorted(ctx.node.files)
    dirs = known_dirs(ctx)
    out = []
    for p in paths:
        if p in ctx.node.files:
            out.append(p)
            continue
        prefix = p.rstrip("/") + "/"
        children = {f[len(prefix) :].split("/")[0] for f in files if f.startswith(prefix)}
        children |= {d[len(prefix) :].split("/")[0] for d in dirs if d.startswith(prefix) and d != prefix}
        globbed = [f for f in files if fnmatch.fnmatch(f, p)]
        if not children and not globbed and p.rstrip("/") not in dirs:
            raise CommandError(f"ls: cannot access '{p}': No such file or directory", 2)
        names = sorted(children) or globbed
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
    for f in ctx.node.files:
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
    for a in args:
        if a.startswith("-"):
            continue
        ctx.node.files.pop(a, None)
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
    dirs = ctx.node.setdefault("dirs", [])
    for a in args:
        path = ctx.abspath(a)
        if not a.startswith("-") and path not in dirs:
            dirs.append(path)
    return ""


@command("cd", help="cd [DIR]: change the working directory (default: home)")
def cd(ctx, args):
    target = next((a for a in args if a != "--" and not a.startswith("-")), "~")
    path = ctx.abspath(target)
    if path in ctx.node.files:
        raise CommandError(f"bash: cd: {target}: Not a directory")
    if path not in known_dirs(ctx):
        raise CommandError(f"bash: cd: {target}: No such file or directory")
    ctx.cwd = path
    return ""


@command("pwd", help="pwd: print the working directory")
def pwd(ctx, args):
    return ctx.cwd + "\n"


@command("chmod", "chown", "sync")
def noop(ctx, args):
    return ""


@command("sed", help="sed -i 's/old/new/[g]' file  (in-place edit)")
def sed(ctx, args):
    inplace = False
    exprs, files = [], []
    it = iter(args)
    for a in it:
        if a == "-i" or a.startswith("-i"):
            inplace = True
        elif a == "-e":
            exprs.append(next(it))
        elif not exprs and not a.startswith("-"):
            exprs.append(a)
        else:
            files.append(a)
    text = _read(ctx, files[0]) if files else ctx.stdin
    for e in exprs:
        text = _apply_sed(e, text)
    if inplace and files:
        ctx.node.files[files[0]] = text
        return ""
    return text


def _apply_sed(expr: str, text: str) -> str:
    m = re.match(r"^(/(?P<addr>[^/]*)/)?(?P<cmd>[sd])(?P<rest>.*)$", expr)
    if not m:
        raise CommandError(f"sed: unsupported expression: {expr}")
    addr = m.group("addr")
    if m.group("cmd") == "d":
        return "".join(l for l in text.splitlines(True) if not (addr and re.search(addr, l)))
    rest = m.group("rest")
    delim = rest[0]
    parts = rest[1:].split(delim)
    if len(parts) < 3:
        raise CommandError("sed: unterminated `s' command")
    pat, rep, flags = parts[0], parts[1], parts[2]
    rep = re.sub(r"\\(\d)", r"\\\1", rep).replace("&", r"\g<0>")
    count = 0 if "g" in flags else 1
    out = []
    for line in text.splitlines(True):
        if addr is None or re.search(addr, line):
            line = re.sub(pat, rep, line, count=count)
        out.append(line)
    return "".join(out)


# ---------------------------------------------------------------- text filters


@command("grep", "egrep", help="filter lines (-i -v -c -E -A/-B supported)")
def grep(ctx, args):
    flags, pattern, files = set(), None, []
    after = before = 0
    it = iter(args)
    for a in it:
        ctx_opt = re.fullmatch(r"-([ABC])(\d*)|--(after|before)-context=(\d+)|--context=(\d+)", a)
        if ctx_opt and pattern is None:
            which = ctx_opt.group(1) or {"after": "A", "before": "B"}.get(ctx_opt.group(3) or "", "C")
            raw = ctx_opt.group(2) or ctx_opt.group(4) or ctx_opt.group(5) or next(it, "")
            if not raw.isdigit():
                raise CommandError(f"grep: {raw}: invalid context length argument", 2)
            n = int(raw)
            after = n if which in ("A", "C") else after
            before = n if which in ("B", "C") else before
        elif a.startswith("-") and pattern is None and len(a) > 1:
            flags.update(a[1:])
        elif pattern is None:
            pattern = a
        else:
            files.append(a)
    if pattern is None:
        raise CommandError("usage: grep [-ivcE] PATTERN [FILE]", 2)
    text = "".join(_read(ctx, f) for f in files) if files else ctx.stdin
    rx = re.compile(pattern, re.I if "i" in flags else 0)
    lines = text.splitlines()
    hits = [i for i, l in enumerate(lines) if bool(rx.search(l)) != ("v" in flags)]
    if "c" in flags:
        return f"{len(hits)}\n"
    if not hits:
        raise CommandError("", 1)
    keep = sorted({j for i in hits for j in range(max(0, i - before), min(len(lines), i + after + 1))})
    if ctx.colorize and "v" not in flags:  # Ubuntu aliases grep to grep --color=auto
        return "\n".join(rx.sub(lambda m: f"\033[01;31m\033[K{m.group(0)}\033[m\033[K", lines[j]) for j in keep) + "\n"
    return "\n".join(lines[j] for j in keep) + "\n"


@command("head")
def head(ctx, args):
    n = _num_arg(args, 10)
    return "\n".join(_input(ctx, args).splitlines()[:n]) + "\n"


@command("tail")
def tail(ctx, args):
    n = _num_arg(args, 10)
    lines = _input(ctx, args).splitlines()
    return "\n".join(lines[-n:]) + "\n"


@command("wc")
def wc(ctx, args):
    text = _input(ctx, args)
    if "-l" in args:
        return f"{len(text.splitlines())}\n"
    return f"{len(text.splitlines())} {len(text.split())} {len(text)}\n"


@command("sort")
def sort_(ctx, args):
    lines = sorted(_input(ctx, args).splitlines(), reverse="-r" in args)
    return "\n".join(lines) + "\n"


@command("awk", help="limited: awk '{print $N}'")
def awk(ctx, args):
    prog = next((a for a in args if not a.startswith("-")), "")
    fields = [int(x) for x in re.findall(r"\$(\d+)", prog)]
    out = []
    for line in ctx.stdin.splitlines():
        cols = line.split()
        out.append(" ".join(cols[f - 1] if f and f <= len(cols) else (line if f == 0 else "") for f in fields))
    return "\n".join(out) + "\n"


def _num_arg(args, default):
    for i, a in enumerate(args):
        if a == "-n" and i + 1 < len(args):
            return int(args[i + 1])
        if re.fullmatch(r"-\d+", a):
            return int(a[1:])
    return default


def _input(ctx, args):
    files = [a for a in args if not a.startswith("-") and not a.isdigit()]
    return "".join(_read(ctx, f) for f in files) if files else ctx.stdin


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
        return sub.run(" ".join(shlex.quote(a) for a in remote))
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
    if "-r" in args:
        return ctx.node.kernel + "\n"
    return f"Linux {ctx.host} {ctx.node.kernel} x86_64 GNU/Linux\n"


@command("export", help="export VAR=value")
def export(ctx, args):
    for a in args:
        if "=" in a:
            k, v = a.split("=", 1)
            ctx.node.env[k] = v
    return ""


@command("unset")
def unset(ctx, args):
    for a in args:
        ctx.node.env.pop(a, None)
    return ""


@command("env", "printenv")
def env(ctx, args):
    if args:
        return "".join(ctx.node.env.get(a, "") + "\n" for a in args)
    return "".join(f"{k}={v}\n" for k, v in sorted(ctx.node.env.items()))


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
            ctx.node.services[name] = "active"
            _service_hook(ctx, name, action)
            if ctx.node.services[name] != "active":
                raise CommandError(
                    f"Job for {name}.service failed because the control process exited with error code.\n"
                    f'See "systemctl status {name}.service" and "journalctl -xeu {name}.service" for details.'
                )
        elif action == "stop":
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


@command("journalctl", help="journalctl -u <unit>")
def journalctl(ctx, args):
    unit = None
    for i, a in enumerate(args):
        if a in ("-u", "--unit") and i + 1 < len(args):
            unit = args[i + 1].removesuffix(".service")
        elif a.startswith("-u") and len(a) > 2 and not a.startswith("--"):
            unit = a[2:].removesuffix(".service")
        elif a.startswith("-xeu"):
            unit = args[i + 1].removesuffix(".service") if i + 1 < len(args) else None
    if unit:
        lines = ctx.node.journal.get(unit, [])
        return ("\n".join(lines) + "\n") if lines else "-- No entries --\n"
    lines = [l for log in ctx.node.journal.values() for l in log]
    return "\n".join(lines) + "\n" if lines else "-- No entries --\n"


@command("dmesg", help="kernel ring buffer (look for NVRM: Xid)")
def dmesg(ctx, args):
    lines = list(ctx.node.dmesg)
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


@command("modprobe", help="modprobe [-r] <module>")
def modprobe(ctx, args):
    remove = "-r" in args
    # the kernel treats '-' and '_' in module names as the same character
    mods = [a.replace("-", "_") for a in args if not a.startswith("-")]
    for m in mods:
        if remove:
            if m == "nvidia" and any(g.processes for g in ctx.node.gpus):
                raise CommandError("modprobe: FATAL: Module nvidia is in use.")
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
    for fn in HOOKS_REBOOT:
        fn(ctx, node)
    msg = f"Connection to {ctx.host} closed by remote host. (node rebooted and is back up)\n"
    if ctx.host_stack:
        ctx.host = ctx.host_stack.pop()
        ctx.cwd = ctx.saved_cwd.pop(len(ctx.host_stack), "/root")
    return msg


@command("ipmitool", help="ipmitool sel list | sdr | power status|on|off|cycle")
def ipmitool(ctx, args):
    node = ctx.node
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
    node = ctx.state.nodes.get(host)
    if node is None:
        raise CommandError(f"ping: {host}: Name or service not known", 2)
    if not node.up:
        raise CommandError(f"PING {host}: Destination Host Unreachable", 1)
    return f"64 bytes from {host} ({node.ip}): icmp_seq=1 ttl=64 time=0.112 ms\n"


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
