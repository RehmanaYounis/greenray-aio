"""``kubectl logs``, ``exec`` and ``run``: the commands that look inside a container."""

from __future__ import annotations

import json

from ...core.errors import CommandError
from ...core.state import AttrDict
from .. import container as ct
from .apply import with_sa_secrets
from .edit import merge_patch
from .model import new_pod, ns_of, pod_restarts, pods_of, remove_pod, resolve_kind, validate_pod_spec
from .output import list_objects
from .scheduler import PodCtx, reconcile, visible_devices


def find_pod(state, opts: dict, ref: str) -> tuple:
    k = state.k8s
    ns = ns_of(opts)
    if "/" in ref:
        kw, name = ref.split("/", 1)
        kind = resolve_kind(kw)
        if kind == "pods":
            ref = name
        elif kind in ("daemonsets", "deployments", "jobs"):
            if kind == "daemonsets":
                pods = pods_of(state, "DaemonSet", name, ns)
            elif kind == "jobs":
                pods = pods_of(state, "Job", name, ns)
            else:
                pods = [p for p in k.pods if p.get("owner") and p.owner.get("deployment") == name and p.namespace == ns]
            if not pods:
                raise CommandError(
                    "error: timed out waiting for the condition"
                    if kind != "daemonsets"
                    else f'Error from server (NotFound): daemonsets.apps "{name}" not found',
                    1,
                )
            pods.sort(key=lambda p: p.name)
            return pods[0], len(pods)
        else:
            raise CommandError(f"error: cannot get logs from {kw}", 1)
    p = next((p for p in k.pods if p.namespace == ns and p.name == ref), None)
    if p is None:
        raise CommandError(f'Error from server (NotFound): pods "{ref}" not found', 1)
    return p, 1


def cmd_logs(state, pos: list[str], opts: dict) -> str:
    if not pos and opts.get("selector"):
        pods = [p for p in list_objects(state, "pods", opts) if p.status not in ("Pending",)]
        if not pos and not pods:
            return "No resources found in " + ns_of(opts) + " namespace.\n"
        out = []
        for p in pods[:5]:
            try:
                out.append(
                    cmd_logs(state, [p.name], {k_: v for k_, v in opts.items() if k_ != "selector"}).rstrip("\n")
                )
            except CommandError as e:
                out.append(str(e))
        if len(pods) > 5:
            out.append(
                f"error: you are attempting to follow {len(pods)} log streams, but maximum allowed concurrency is 5, "
                "use --max-log-requests to increase the limit"
            )
        return "\n".join(out) + "\n"
    if not pos:
        raise CommandError("error: expected 'logs [-f] [-p] (POD | TYPE/NAME) [-c CONTAINER]'.", 1)
    p, n = find_pod(state, opts, pos[0])
    out = ""
    if n > 1:
        out = f"Found {n} pods, using pod/{p.name}\n"
    if p.node is None or p.status in ("Pending", "ContainerCreating") or p.status.startswith("Init"):
        c = (p.spec.get("containers") or [{}])[0].get("name")
        if p.status.startswith("Init") and p.get("component"):
            return (
                out
                + f'Defaulted container "{c}" out of: {c}, driver-validation (init)\n'
                + f'Error from server (BadRequest): container "{c}" in pod "{p.name}" is waiting to start: PodInitializing\n'
            )
        raise CommandError(
            out
            + f'Error from server (BadRequest): container "{c}" in pod "{p.name}" is waiting to start: '
            + ("ContainerCreating" if p.node else "PodInitializing"),
            1,
        )
    if p.status in ("ErrImagePull", "ImagePullBackOff"):
        c = (p.spec.get("containers") or [{}])[0].get("name")
        raise CommandError(
            f'Error from server (BadRequest): container "{c}" in pod "{p.name}" is waiting to start: '
            f"trying and failing to pull image",
            1,
        )
    lines = list(p.get("logs") or [])
    if opts.get("tail") not in (None, "-1"):
        lines = lines[-int(opts["tail"]) :]
    if opts.get("previous") and pod_restarts(state, p) == 0:
        c = (p.spec.get("containers") or [{}])[0].get("name")
        raise CommandError(
            f'Error from server (BadRequest): previous terminated container "{c}" in pod "{p.name}" not found', 1
        )
    return out + ("\n".join(lines) + "\n" if lines else "")


def cmd_exec(ctx, pos: list[str], opts: dict, cmd: list[str]) -> str:
    state = ctx.state
    if not pos:
        raise CommandError("error: expected 'exec (POD | TYPE/NAME) [-c CONTAINER] [flags] -- COMMAND [args...]'.", 1)
    p, _ = find_pod(state, opts, pos[0])
    if not cmd:
        raise CommandError("error: you must specify at least one command for the container", 1)
    if p.node is None:
        raise CommandError(f"Error from server (BadRequest): pod {p.name} does not have a host assigned", 1)
    if p.status == "Completed":
        raise CommandError("error: cannot exec into a container in a completed pod; current phase is Succeeded", 1)
    if p.status != "Running":
        c = (p.spec.get("containers") or [{}])[0].get("name")
        raise CommandError(f'error: unable to upgrade connection: container not found ("{c}")', 1)
    image = (p.spec.get("containers") or [{}])[0].get("image", "")
    env = {e.get("name"): str(e.get("value")) for c in p.spec.get("containers", []) for e in (c.get("env") or [])}
    rdma = any(r.startswith("rdma/") for r in p.requests)
    node = state.nodes[p.node]
    if p.get("gpu_fraction"):  # Run:ai fractional GPU: the container sees a slice of the framebuffer
        node = AttrDict(dict(node))
        node["gpus"] = [
            AttrDict({**dict(g), "memory_total_mib": int(g.memory_total_mib * p.gpu_fraction)}) for g in node.gpus
        ]
    try:
        return ct.exec_in_container(
            PodCtx(state, p.node),
            node,
            visible_devices(state, p),
            cmd,
            image,
            hostname=p.name,
            env=env,
            rdma=rdma,
        )
    except CommandError as e:
        raise CommandError(f"{e}\ncommand terminated with exit code {e.code if e.code != 1 else 1}", e.code)


def cmd_run(ctx, pos: list[str], opts: dict, cmd: list[str]) -> str:
    state = ctx.state
    for bad in ("limits", "requests"):
        if bad in opts:
            raise CommandError(f"error: unknown flag: --{bad}\nSee 'kubectl run --help' for usage.", 1)
    if not pos:
        raise CommandError("error: NAME is required for run", 1)
    name = pos[0]
    image = opts.get("image")
    if not image:
        raise CommandError('error: required flag(s) "image" not set', 1)
    ns = ns_of(opts)
    if ns not in state.k8s.namespaces:
        raise CommandError(f'Error from server (NotFound): namespaces "{ns}" not found', 1)
    if any(p.namespace == ns and p.name == name for p in state.k8s.pods):
        raise CommandError(f'Error from server (AlreadyExists): pods "{name}" already exists', 1)
    c = {"name": name, "image": image}
    if cmd:
        c["command" if opts.get("command") else "args"] = cmd
    spec = {"containers": [c], "restartPolicy": opts.get("restart", "Always")}
    if opts.get("overrides"):
        try:
            ov = json.loads(opts["overrides"])
        except ValueError as e:
            raise CommandError(f"error: Invalid JSON Patch: {e}", 1)
        spec = merge_patch(spec, ov.get("spec", {}), strategic=True)
    validate_pod_spec("Pod", name, spec)
    labels = {"run": name}
    if opts.get("labels"):
        labels = dict(x.split("=", 1) for x in opts["labels"].split(","))
    p = new_pod(state, ns, name, with_sa_secrets(state, ns, spec), labels=labels)
    reconcile(state)
    if opts.get("rm") and opts.get("stdin"):
        out = (
            "\n".join(p.logs) + "\n"
            if p.status in ("Completed", "Failed")
            else ("error: timed out waiting for the condition\n" if p.node is None else "")
        )
        remove_pod(state, p)
        return out + f'pod "{name}" deleted\n'
    return f"pod/{name} created\n"
