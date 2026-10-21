"""Run commands as another user: su, runuser and whoami."""

from __future__ import annotations

from .. import command
from .common import _USER, _err


def _known_user(state, name) -> bool:
    if name == "root":
        return True
    if name in state.slurm.users:
        return True
    bcm = state.get("bcm") or {}
    return name in (bcm.get("users") or {})


def _run_as(ctx, user, line, prog):
    from ...core.shell import Shell

    if not _known_user(ctx.state, user):
        raise _err(f"{prog}: user {user} does not exist or the user entry does not contain all the required fields", 1)
    sub = Shell(ctx.state, ctx.host)
    sub.ctx.executed = ctx.executed
    sub.ctx.host_stack = list(ctx.host_stack)
    sub.ctx.cwd = "/root" if user == "root" else f"/home/{user}"
    _USER.append(user)
    try:
        out = sub.run(line, record=False)
    finally:
        _USER.pop()
    return out + ("\n" if out and not out.endswith("\n") else "")


@command("su", help='su - <user> -c "<command>" — run one command as another user (e.g. to test sbatch as alice)')
def su_cmd(ctx, args):
    user, cmd, it = None, None, iter(args)
    for a in it:
        if a in ("-", "-l", "--login"):
            continue
        if a in ("-c", "--command"):
            cmd = next(it, None)
        elif a.startswith("--command="):
            cmd = a.split("=", 1)[1]
        elif a.startswith("-"):
            continue
        elif user is None:
            user = a
    user = user or "root"
    if cmd is None:
        raise _err(
            "su: interactive shells are not supported in the lab; run one command at a time:\n"
            f'  su - {user} -c "sbatch --wrap=hostname"',
            1,
        )
    return _run_as(ctx, user, cmd, "su")


@command("runuser", help="runuser -u <user> -- <command> — run a command as another user")
def runuser_cmd(ctx, args):
    import shlex as _shlex

    user, rest = None, list(args)
    if "-u" in rest:
        i = rest.index("-u")
        user = rest[i + 1] if i + 1 < len(rest) else None
        del rest[i : i + 2]
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not user or not rest:
        raise _err("runuser: usage: runuser -u <user> -- <command> [args...]", 1)
    return _run_as(ctx, user, " ".join(_shlex.quote(a) for a in rest), "runuser")


@command("whoami")
def whoami(ctx, args):
    return _USER[-1] + "\n"
