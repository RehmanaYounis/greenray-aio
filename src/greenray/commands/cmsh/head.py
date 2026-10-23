"""Head-node services seen from every node: cmha, and the LDAP users BCM serves (id, getent)."""

from __future__ import annotations

from ...core.errors import CommandError
from ...core.state import HEAD
from .. import command
from .model import bcm_init


@command("cmha", help="cmha status — BCM head node high-availability status")
def cmha(ctx, args):
    if ctx.host != HEAD:
        raise CommandError("bash: cmha: command not found", 127)
    if args[:1] == ["status"] or not args:
        return (
            "Node Status: running in active master mode\n\n"
            "Failover status:\n"
            f"{HEAD}* -> (no passive head node configured)\n"
            "  HA is not configured on this cluster. Run cmha-setup to add a passive head node.\n"
        )
    raise CommandError("usage: cmha status")


# ---------------------------------------------------------------- LDAP-backed identity lookups


def _ldap_user(state, name):
    bcm = bcm_init(state)
    return bcm.users.get(name)


def _groups_of(state, name):
    bcm = bcm_init(state)
    return sorted((g for g in bcm.groups.values() if name in g.get("members", [])), key=lambda g: g.get("gid", 0))


@command("id", help="id [user] — uid, gid and groups (BCM users come from LDAP on every node)")
def id_cmd(ctx, args):
    from ..slurm import _USER

    names = [a for a in args if not a.startswith("-")]
    name = names[0] if names else _USER[-1]
    if name == "root":
        return "uid=0(root) gid=0(root) groups=0(root)\n"
    u = _ldap_user(ctx.state, name)
    if u is None:
        raise CommandError(f"id: '{name}': no such user", 1)
    uid = u.get("uid", 1000)
    prim = ctx.state.bcm.groups.get(name, {}).get("gid", uid)
    groups = [f"{prim}({name})"] + [f"{g.get('gid')}({g.name})" for g in _groups_of(ctx.state, name) if g.name != name]
    return f"uid={uid}({name}) gid={prim}({name}) groups={','.join(groups)}\n"


@command("getent", help="getent passwd|group [name] — look up users/groups (LDAP)")
def getent_cmd(ctx, args):
    if not args or args[0] not in ("passwd", "group"):
        raise CommandError("Usage: getent passwd|group [key]", 1)
    bcm = bcm_init(ctx.state)
    keys = args[1:]
    out = []
    if args[0] == "passwd":
        users = [("root", {"uid": 0, "home": "/root", "shell": "/bin/bash"})] + [
            (n, u) for n, u in bcm.users.items() if n != "root"
        ]
        for n, u in users:
            if keys and n not in keys:
                continue
            uid = u.get("uid", 0)
            gid = bcm.groups.get(n, {}).get("gid", uid)
            out.append(
                f"{n}:x:{uid}:{gid}:{u.get('surname', '') or n}:{u.get('home', '/home/' + n)}:{u.get('shell', '/bin/bash')}"
            )
    else:
        for n, g in bcm.groups.items():
            if keys and n not in keys:
                continue
            out.append(f"{n}:x:{g.get('gid', 0)}:{','.join(g.get('members', []))}")
    if keys and not out:
        raise CommandError("", 2)
    return "\n".join(out) + "\n"
