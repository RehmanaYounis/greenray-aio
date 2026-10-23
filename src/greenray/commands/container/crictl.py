"""crictl: the containerd view of the pods on a node."""

from __future__ import annotations

import json

from ...core.errors import CommandError
from .. import command
from ..common import columns
from .docker import _short_id
from .images import norm_image
from .runtime import container_init


@command("crictl", help="crictl ps [-a] | pods | images | info | logs <id>  (containerd view of k8s pods on this node)")
def crictl(ctx, args):
    container_init(ctx.state)
    node = ctx.node
    if node.services.get("containerd") != "active":
        raise CommandError(
            'E1019 10:42:17.000000   1 remote_runtime.go:277] "ListContainers with filter from runtime service failed" '
            'err="rpc error: code = Unavailable desc = connection error: desc = \\"transport: Error while dialing: '
            'dial unix /run/containerd/containerd.sock: connect: no such file or directory\\""',
            1,
        )
    from .. import kubectl as k

    k.k8s_init(ctx.state)
    k.reconcile(ctx.state)
    pods = [p for p in ctx.state.k8s.pods if p.node == ctx.host]
    sub = args[0] if args else "ps"
    if sub == "info":
        rt = node.runtime_loaded.containerd
        runtimes = {
            h: {
                "runtimeType": "io.containerd.runc.v2",
                "options": {
                    "BinaryName": "/usr/bin/nvidia-container-runtime" if h == "nvidia" else "",
                    "SystemdCgroup": True,
                },
            }
            for h in rt["handlers"]
        }
        info = {
            "status": {
                "conditions": [{"type": "RuntimeReady", "status": True}, {"type": "NetworkReady", "status": True}]
            },
            "config": {
                "containerd": {"defaultRuntimeName": rt["default"], "runtimes": runtimes},
                "sandboxImage": "registry.k8s.io/pause:3.9",
            },
        }
        return json.dumps(info, indent=2) + "\n"
    if sub == "pods":
        rows = [("POD ID", "CREATED", "STATE", "NAME", "NAMESPACE", "ATTEMPT", "RUNTIME")]
        for p in pods:
            if p.status in ("Pending",):
                continue
            rows.append(
                (
                    _short_id(p.uid, n=13),
                    "12 minutes ago",
                    "Ready",
                    p.name,
                    p.namespace,
                    "0",
                    p.spec.get("runtimeClassName") or "(default)",
                )
            )
        return columns(rows)
    if sub == "images":
        rows = [("IMAGE", "TAG", "IMAGE ID", "SIZE")]
        seen = set()
        for p in pods:
            for c in p.spec.get("containers", []):
                ref = norm_image(c.get("image", ""))
                if ref in seen or p.status in ("ErrImagePull", "ImagePullBackOff"):
                    continue
                seen.add(ref)
                name, _, tag = ref.rpartition(":")
                rows.append((name, tag, _short_id(ref, n=13), "1.1GB"))
        return columns(rows)
    if sub == "logs":
        cid = next((a for a in args[1:] if not a.startswith("-")), "")
        for p in pods:
            if _short_id(p.uid, "c", n=13).startswith(cid):
                return "\n".join(p.get("logs", [])) + "\n"
        raise CommandError(
            f'rpc error: code = NotFound desc = an error occurred when try to find container "{cid}": not found', 1
        )
    if sub == "ps":
        show_all = "-a" in args or "--all" in args
        name_f = None
        if "--name" in args:
            name_f = args[args.index("--name") + 1]
        rows = [("CONTAINER", "IMAGE", "CREATED", "STATE", "NAME", "ATTEMPT", "POD ID", "POD")]
        for p in pods:
            running = p.status == "Running"
            exited = p.status in ("CrashLoopBackOff", "Completed", "Error")
            if not (running or (show_all and exited)):
                continue
            for c in p.spec.get("containers", []):
                if name_f and name_f not in c.get("name", ""):
                    continue
                rows.append(
                    (
                        _short_id(p.uid, "c", n=13),
                        _short_id(norm_image(c.get("image", "")), n=13),
                        "12 minutes ago",
                        "Running" if running else "Exited",
                        c.get("name", ""),
                        str(k.pod_restarts(ctx.state, p)),
                        _short_id(p.uid, n=13),
                        p.name,
                    )
                )
        return columns(rows)
    raise CommandError(f"No help topic for '{sub}'", 1)
