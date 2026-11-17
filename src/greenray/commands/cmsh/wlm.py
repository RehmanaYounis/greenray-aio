"""BCM-managed Slurm: the ``wlm`` and ``configurationoverlay`` objects and the slurm.conf they generate.

On a BCM cluster the queues and the node list of ``slurm.conf`` come from cmd's own model: a
hand edit of the file lasts until cmd regenerates it. Here the model is ``state.bcm.wlm``
(one instance, ``slurm``, with its job queues) and ``state.bcm.overlays`` (which nodes carry
the slurmclient, slurmserver, slurmsubmit and slurmaccounting roles). Every commit in those
modes and every restart of cmd rewrites the NodeName and PartitionName lines.
"""

from __future__ import annotations

import re

from ...core.state import HEAD
from .. import HOOKS_SERVICE, HOOKS_STATE, command
from ..slurm.common import CONF, compress_hostlist, expand_hostlist, parse_time
from ..slurm.conf import parse_slurm_conf
from ..slurm.nodes import apply_config

JOBQUEUE_PROPS = ("name", "nodes", "maxtime", "defaulttime", "default", "state", "qos", "allowqos", "options")
QUEUE_STATES = ("UP", "DOWN", "DRAIN", "INACTIVE")
OVERLAY_ROLES = {
    "slurm-server": ["slurmserver"],
    "slurm-client": ["slurmclient"],
    "slurm-client-gpu": ["slurmclient"],
    "slurm-submit": ["slurmsubmit"],
    "slurm-accounting": ["slurmaccounting"],
}
NODE_LINE_DEFAULTS = "CPUs=224 Sockets=2 CoresPerSocket=56 ThreadsPerCore=2 RealMemory=2063000"


def wlm_init(state) -> None:
    """Derive the model from the current slurm.conf once; from then on the model is the source."""
    bcm = state.bcm
    if "wlm" not in bcm:
        conf = parse_slurm_conf(state.nodes[HEAD].files.get(CONF, ""))
        queues = {}
        for name in conf["part_order"]:
            a = conf["partitions"][name]
            queues[name] = {
                "name": name,
                "nodes": a.get("nodes", ""),
                "maxtime": a.get("maxtime", "INFINITE"),
                "defaulttime": a.get("defaulttime", ""),
                "default": a.get("default", "NO").upper() == "YES",
                "state": a.get("state", "UP").upper(),
                "qos": a.get("qos", ""),
                "allowqos": a.get("allowqos", ""),
                "options": "",
            }
        bcm.wlm = {
            "slurm": {
                "name": "slurm",
                "type": "Slurm",
                "version": "23.11.6",
                "jobqueues": queues,
                "cgroups": {"constraindevices": True, "constraincores": True, "constrainramspace": True},
            }
        }
    if "overlays" not in bcm:
        gpu_cats = sorted(
            {d.get("category") for d in bcm.devices.values() if d.type != "HeadNode" and d.get("category")}
        )
        bcm.overlays = {
            "slurm-server": {
                "name": "slurm-server",
                "priority": 500,
                "categories": [],
                "nodes": [HEAD],
                "roles": ["slurmserver"],
            },
            "slurm-client": {
                "name": "slurm-client",
                "priority": 500,
                "categories": list(gpu_cats),
                "nodes": [],
                "roles": ["slurmclient"],
            },
            "slurm-client-gpu": {
                "name": "slurm-client-gpu",
                "priority": 450,
                "categories": list(gpu_cats),
                "nodes": [],
                "roles": ["slurmclient"],
            },
            "slurm-submit": {
                "name": "slurm-submit",
                "priority": 500,
                "categories": list(gpu_cats),
                "nodes": [HEAD],
                "roles": ["slurmsubmit"],
            },
            "slurm-accounting": {
                "name": "slurm-accounting",
                "priority": 500,
                "categories": [],
                "nodes": [HEAD],
                "roles": ["slurmaccounting"],
            },
        }


def overlay_members(state, overlay: str) -> list[str]:
    """Nodes an overlay covers: those listed by name plus every node of its categories."""
    ov = state.bcm.overlays.get(overlay)
    if ov is None:
        return []
    out = set(ov.get("nodes", []))
    for name, dev in state.bcm.devices.items():
        if dev.get("category") in ov.get("categories", []):
            out.add(name)
    return sorted(out)


def slurm_clients(state) -> list[str]:
    """Compute nodes that get a NodeName line: covered by an overlay carrying the slurmclient role."""
    nodes: set[str] = set()
    for name, ov in state.bcm.overlays.items():
        if "slurmclient" in ov.get("roles", []):
            nodes.update(n for n in overlay_members(state, name) if n != HEAD)
    return sorted(nodes)


def gpu_clients(state) -> set[str]:
    return set(overlay_members(state, "slurm-client-gpu")) - {HEAD}


def render_slurm_conf(state) -> str:
    """The file cmd writes: every other line kept, NodeName and PartitionName lines from the model."""
    wlm_init(state)
    old = state.nodes[HEAD].files.get(CONF, "")
    conf = parse_slurm_conf(old)
    kept = [line for line in old.splitlines() if not re.match(r"\s*(NodeName|PartitionName)\s*=", line, re.I)]
    while kept and not kept[-1].strip():
        kept.pop()
    clients = slurm_clients(state)
    with_gpu = gpu_clients(state)
    node_lines = []
    for group_has_gpu in (True, False):
        names = [n for n in clients if (n in with_gpu) == group_has_gpu]
        if not names:
            continue
        sample = conf["nodes"].get(names[0], {})
        attrs = " ".join(f"{k}={v}" for k, v in _node_attrs(sample).items())
        gres = " Gres=gpu:h100:8" if group_has_gpu else ""
        node_lines.append(f"NodeName={compress_hostlist(names)} {attrs}{gres} State=UNKNOWN")
    part_lines = []
    for q in state.bcm.wlm.slurm.jobqueues.values():
        line = f"PartitionName={q.name} Nodes={q.nodes or 'ALL'} Default={'YES' if q.default else 'NO'}"
        line += f" MaxTime={q.maxtime or 'INFINITE'}"
        if q.get("defaulttime"):
            line += f" DefaultTime={q.defaulttime}"
        if q.get("qos"):
            line += f" QOS={q.qos}"
        if q.get("allowqos"):
            line += f" AllowQos={q.allowqos}"
        if q.get("options"):
            line += f" {q.options}"
        line += f" State={q.get('state', 'UP')}"
        part_lines.append(line)
    return "\n".join([*kept, *node_lines, *part_lines]) + "\n"


def _node_attrs(sample: dict) -> dict:
    keys = {
        "cpus": "CPUs",
        "sockets": "Sockets",
        "corespersocket": "CoresPerSocket",
        "threadspercore": "ThreadsPerCore",
        "realmemory": "RealMemory",
    }
    out = {}
    for k, name in keys.items():
        if k in sample:
            out[name] = sample[k]
    if not out:
        for tok in NODE_LINE_DEFAULTS.split():
            k, v = tok.split("=")
            out[k] = v
    return out


def regenerate(state, reason: str) -> None:
    """What cmd does after a wlm or overlay commit and on its own restart: rewrite the file, reconfigure."""
    head = state.nodes[HEAD]
    head.files[CONF] = render_slurm_conf(state)
    from .model import NOW

    state.bcm.setdefault("events", []).append(
        f"{NOW} [notice] {HEAD}: {reason}; slurm.conf regenerated, slurmctld reconfigured"
    )
    head.journal.setdefault("cmd", []).append(f"{NOW} {HEAD} cmd[1127]: Info: Slurm: wrote {CONF} ({reason})")
    # cmd also applies the roles: a node that just became a Slurm client gets its slurmd started
    for n in slurm_clients(state):
        node = state.nodes.get(n)
        if node is not None and node.get("up", True) and node.services.get("slurmd") != "active":
            node.services["slurmd"] = "active"
    if head.services.get("slurmctld") == "active":
        from ..slurm.scheduler import _ensure

        _ensure(state)
        apply_config(state)


def validate_queue(bcm, name: str, prop: str, value: str, err) -> object:
    """cmsh's own checks on job queue properties, before the value is staged."""
    if prop == "nodes":
        try:
            nodes = expand_hostlist(value)
        except Exception:
            err(f"Error: invalid node list '{value}'")
        unknown = [n for n in nodes if n not in bcm.devices or bcm.devices[n].type == "HeadNode"]
        if unknown:
            err(f"Error: unknown compute node(s): {', '.join(unknown)}")
        return compress_hostlist(nodes)
    if prop in ("maxtime", "defaulttime"):
        if value.upper() in ("INFINITE", "UNLIMITED"):
            return "INFINITE"
        try:
            ok = parse_time(value) is not None
        except ValueError:
            ok = False
        if not ok:
            err(f"Error: invalid time '{value}' (use minutes, HH:MM:SS or D-HH:MM:SS, or INFINITE)")
        return value
    if prop == "default":
        if value.lower() not in ("yes", "no", "true", "false"):
            err("Error: default must be yes or no")
        return value.lower() in ("yes", "true")
    if prop == "state":
        if value.upper() not in QUEUE_STATES:
            err(f"Error: state must be one of {', '.join(QUEUE_STATES)}")
        return value.upper()
    if prop == "name":
        err("Error: property 'name' is read-only")
    return value


# ---------------------------------------------------------------- cmd restart rewrites the file


def _on_service(ctx, name, action):
    if ctx.host == HEAD and name == "cmd" and action in ("start", "restart"):
        from .model import bcm_init

        bcm_init(ctx.state)
        regenerate(ctx.state, "cmd started")


HOOKS_SERVICE.append(_on_service)


def _snapshot(state) -> None:
    """The model is what BCM holds when the lab opens; anything typed into slurm.conf afterwards is a hand edit."""
    from .model import bcm_init

    bcm_init(state)
    wlm_init(state)


HOOKS_STATE.append(_snapshot)


# ---------------------------------------------------------------- the setup wizards


@command("cm-wlm-setup", help="cm-wlm-setup: BCM's workload-manager wizard (interactive; its result lives in cmsh)")
def cm_wlm_setup(ctx, args):
    from ...core.errors import CommandError

    if ctx.host != HEAD:
        raise CommandError("bash: cm-wlm-setup: command not found", 127)
    raise CommandError(
        "cm-wlm-setup: this wizard is interactive (ncurses) and is not available in the lab shell.\n"
        'What it produces is already here: the Slurm instance and its queues in  cmsh -c "wlm; use slurm; jobqueue; list"\n'
        'and the role assignments in  cmsh -c "configurationoverlay; list".',
        1,
    )


@command(
    "cm-kubernetes-setup",
    help="cm-kubernetes-setup: BCM's Kubernetes wizard (interactive; its result lives in cmsh and kubectl)",
)
def cm_kubernetes_setup(ctx, args):
    from ...core.errors import CommandError

    if ctx.host != HEAD:
        raise CommandError("bash: cm-kubernetes-setup: command not found", 127)
    if "--list-users" in args:
        return "\n".join(f"{u} : {u}-restricted" for u in sorted(ctx.state.bcm.get("k8s_users", []))) + "\n"
    if "--add-user" in args:
        i = args.index("--add-user")
        user = args[i + 1] if i + 1 < len(args) else ""
        if not user or user not in ctx.state.bcm.users:
            raise CommandError(
                f"cm-kubernetes-setup: user '{user}' does not exist in BCM (add it with cmsh user mode first)", 1
            )
        users = ctx.state.bcm.setdefault("k8s_users", [])
        if user not in users:
            users.append(user)
        return f"Kubernetes user {user} created with role {user}-restricted (kubeconfig in /home/{user}/.kube/config)\n"
    raise CommandError(
        "cm-kubernetes-setup: this wizard is interactive (ncurses) and is not available in the lab shell.\n"
        "The cluster it deploys is already here (kubectl get nodes); --add-user <name> and --list-users work.",
        1,
    )
