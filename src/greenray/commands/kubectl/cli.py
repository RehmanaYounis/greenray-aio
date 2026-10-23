"""Argument parsing and verb dispatch for the ``kubectl`` command."""

from __future__ import annotations

import re

from ...core.errors import CommandError
from ...core.state import HEAD
from .. import command
from .apply import apply_doc, cmd_create, delete_obj, load_docs
from .edit import cmd_cordon, cmd_drain, cmd_label, cmd_patch, cmd_rollout, cmd_scale, cmd_taint, cmd_wait
from .exec_logs import cmd_exec, cmd_logs, cmd_run
from .model import (
    CLUSTER_SCOPED,
    EXTRA_KINDS,
    K8S_VERSION,
    KINDS,
    SINGULAR,
    advance,
    ns_of,
    remove_pod,
    resolve_kind,
    split_targets,
    table,
)
from .output import cmd_describe, cmd_get, crd_installed, list_objects
from .scheduler import k8s_init, reconcile

VALUE_FLAGS = {
    "-n",
    "--namespace",
    "-o",
    "--output",
    "-l",
    "--selector",
    "-f",
    "--filename",
    "-L",
    "--label-columns",
    "--type",
    "-p",
    "--patch",
    "-c",
    "--container",
    "--tail",
    "--image",
    "--restart",
    "--replicas",
    "--sort-by",
    "--field-selector",
    "--from-literal",
    "--from-file",
    "--docker-server",
    "--docker-username",
    "--docker-password",
    "--docker-email",
    "--timeout",
    "--grace-period",
    "--overrides",
    "--for",
    "--since",
    "--labels",
    "--dry-run",
    "--kubeconfig",
    "--context",
    "--env",
    "--port",
}

MULTI = {"-f", "--filename", "--from-literal", "--from-file", "-L", "--label-columns", "--env"}

ALIASES = {
    "-n": "namespace",
    "-o": "output",
    "-l": "selector",
    "-f": "filename",
    "-L": "label-columns",
    "-p": "patch",
    "-c": "container",
    "-A": "all-namespaces",
    "-i": "stdin",
    "-t": "tty",
    "-w": "watch",
}


def reject_names_with_selection(names: list[str], opts: dict) -> None:
    """kubectl refuses explicit names next to --all or a selector; both pick the set of objects."""
    if names and opts.get("selector"):
        raise CommandError("error: name cannot be provided when a selector is specified", 1)
    if names and opts.get("all"):
        raise CommandError("error: cannot use --all together with a resource name", 1)


def parse_args(args: list[str], verb: str) -> tuple[list[str], dict, list[str]]:
    pos, opts, cmd = [], {}, []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            cmd = args[i + 1 :]
            break
        if verb == "logs" and a in ("-p", "--previous"):
            opts["previous"] = True
        elif verb == "logs" and a in ("-f", "--follow"):
            opts["follow"] = True
        elif verb in ("exec", "run", "attach") and re.fullmatch(r"-[it]+", a):
            opts["stdin"] = True
        elif a.startswith("--") and "=" in a:
            key, val = a[2:].split("=", 1)
            if "--" + key in MULTI:
                opts.setdefault(key, []).append(val)
            else:
                opts[key] = val
        elif len(a) > 3 and a[2] == "=" and a[:2] in VALUE_FLAGS:  # -n=kube-system, as pflag allows
            key, val = ALIASES.get(a[:2], a[1]), a[3:]
            if a[:2] in MULTI:
                opts.setdefault(key, []).append(val)
            else:
                opts[key] = val
        elif a in VALUE_FLAGS:
            key = ALIASES.get(a, a.lstrip("-"))
            val = args[i + 1] if i + 1 < len(args) else ""
            i += 1
            if a in MULTI:
                opts.setdefault(key, []).append(val)
            else:
                opts[key] = val
        elif a.startswith("-n") and len(a) > 2 and not a.startswith("--"):
            opts["namespace"] = a[2:]
        elif a.startswith("-o") and len(a) > 2 and not a.startswith("--"):
            opts["output"] = a[2:]
        elif a.startswith("--"):
            opts[a[2:]] = True
        elif a.startswith("-") and len(a) > 1:
            for ch in a[1:]:
                opts[ALIASES.get("-" + ch, ch)] = True
        else:
            pos.append(a)
        i += 1
    if verb == "exec" and not cmd and len(pos) > 1:
        cmd = pos[1:]
        pos = pos[:1]
    return pos, opts, cmd


HELP_TEXT = """kubectl controls the Kubernetes cluster manager.

 get        Display one or many resources (pods, nodes, ds, deploy, ns, events, clusterpolicy, ...)
 describe   Show details of a specific resource
 logs       Print the logs for a container in a pod (-p for the previous instance)
 apply      Apply a configuration to a resource by file name or stdin (-f file.yaml / -f -)
 create     Create a resource (namespace, secret docker-registry|generic, deployment, -f)
 delete     Delete resources by file names, stdin, resources and names, or by label selector
 patch      Update fields of a resource (--type merge|json|strategic -p '...')
 label      Update the labels on a resource (key=value, key-, --overwrite)
 taint      Update the taints on one or more nodes (key=value:Effect, trailing - removes)
 cordon / uncordon / drain    Mark node unschedulable / schedulable / evict its pods
 exec       Execute a command in a container (kubectl exec -it POD -- nvidia-smi)
 run        Run a particular image on the cluster
 rollout    restart|status for daemonsets and deployments
 scale / wait / version / cluster-info / api-resources
"""


@command(
    "kubectl", "k", help="kubectl get|describe|logs|apply|create|delete|patch|label|taint|cordon|drain|exec|run|rollout"
)
def kubectl(ctx, args: list[str]) -> str:
    state = ctx.state
    if ctx.host != HEAD:
        raise CommandError(
            "E1019 10:42:17.317224   48211 memcache.go:265] couldn't get current server API group list: Get "
            '"http://localhost:8080/api?timeout=32s": dial tcp 127.0.0.1:8080: connect: connection refused\n'
            "The connection to the server localhost:8080 was refused - did you specify the right host or port?",
            1,
        )
    if state.nodes[HEAD].services.get("kube-apiserver", "active") != "active":
        raise CommandError(
            "The connection to the server 10.141.255.254:6443 was refused - did you specify the right host or port?", 1
        )
    k8s_init(state)
    if not args or args[0] in ("-h", "--help", "help"):
        return HELP_TEXT
    verb, rest = args[0], args[1:]
    pos, opts, cmd = parse_args(rest, verb)
    if isinstance(opts.get("filename"), str):
        opts["filename"] = [opts["filename"]]
    ctx._kns = opts.get("namespace")
    advance(state)
    reconcile(state)
    if verb == "version":
        return (
            f"Client Version: {K8S_VERSION}\nKustomize Version: v5.0.4-0.20230601165947-6ce0bf390ce3\n"
            f"Server Version: {K8S_VERSION}\n"
        )
    if verb == "cluster-info":
        return (
            "Kubernetes control plane is running at https://10.141.255.254:6443\n"
            "CoreDNS is running at https://10.141.255.254:6443/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy\n\n"
            "To further debug and diagnose cluster problems, use 'kubectl cluster-info dump'.\n"
        )
    if verb == "api-resources":
        rows = [("NAME", "SHORTNAMES", "APIVERSION", "NAMESPACED", "KIND")]
        for kind, names in KINDS.items():
            if kind in ("clusterpolicies", "nicclusterpolicies") and not crd_installed(state, kind):
                continue
            short = ",".join(n for n in names if len(n) <= 3 and n != kind)
            api = {
                "daemonsets": "apps/v1",
                "deployments": "apps/v1",
                "jobs": "batch/v1",
                "clusterpolicies": "nvidia.com/v1",
                "nicclusterpolicies": "mellanox.com/v1alpha1",
                "runtimeclasses": "node.k8s.io/v1",
            }.get(kind, EXTRA_KINDS[kind].api_version if kind in EXTRA_KINDS else "v1")
            rows.append((kind, short, api, str(kind not in CLUSTER_SCOPED).lower(), SINGULAR[kind].title()))
        return table(rows)
    if verb == "top":
        raise CommandError("error: Metrics API not available", 1)
    if verb == "get":
        return cmd_get(state, pos, opts)
    if verb == "describe":
        return cmd_describe(state, pos, opts)
    if verb == "logs":
        return cmd_logs(state, pos, opts)
    if verb in ("apply", "replace"):
        out = []
        for src, d in load_docs(ctx, opts):
            if verb == "replace" and opts.get("force"):
                meta = d.get("metadata") or {}
                ns = meta.get("namespace") or ctx._kns or "default"
                kind = d.get("kind", "")
                if kind == "Pod":
                    p = next((p for p in state.k8s.pods if p.namespace == ns and p.name == meta.get("name")), None)
                    if p:
                        remove_pod(state, p)
                        out.append(f'pod "{meta.get("name")}" deleted')
                        out.append(apply_doc(ctx, src, d).replace("created", "replaced"))
                        continue
            out.append(apply_doc(ctx, src, d))
        reconcile(state)
        return "\n".join(out) + "\n"
    if verb == "create":
        res = cmd_create(ctx, pos, opts)
        reconcile(state)
        return res
    if verb == "delete":
        out = []
        if opts.get("filename"):
            for _src, d in load_docs(ctx, opts):
                meta = d.get("metadata") or {}
                kind = resolve_kind(d.get("kind", "").lower())
                try:
                    out.append(delete_obj(state, kind, meta.get("namespace") or ns_of(opts), meta.get("name"), opts))
                except CommandError as e:
                    if not opts.get("ignore-not-found"):
                        raise
                    del e
        else:
            targets = split_targets(pos)
            for kind, names in targets:
                reject_names_with_selection(names, opts)
                if not names and (opts.get("all") or opts.get("selector")):
                    names = [i.get("name") for i in list_objects(state, kind, opts)]
                    if not names:
                        out.append("No resources found")
                if not names and not opts.get("all") and not opts.get("selector"):
                    raise CommandError("error: resource(s) were provided, but no name was specified", 1)
                for nm in names:
                    try:
                        out.append(delete_obj(state, kind, ns_of(opts), nm, opts))
                    except CommandError:
                        if not opts.get("ignore-not-found"):
                            raise
        reconcile(state)
        return "\n".join(out) + "\n"
    if verb == "patch":
        res = cmd_patch(ctx, pos, opts)
        reconcile(state)
        return res
    if verb == "label":
        res = cmd_label(state, pos, opts)
        reconcile(state)
        return res
    if verb == "annotate":
        return "".join(f"{p} annotated\n" for p in pos[:2] if "=" not in p)
    if verb == "taint":
        res = cmd_taint(state, pos, opts)
        reconcile(state)
        return res
    if verb in ("cordon", "uncordon"):
        res = cmd_cordon(state, pos, opts, on=(verb == "cordon"))
        reconcile(state)
        return res
    if verb == "drain":
        res = cmd_drain(state, pos, opts)
        reconcile(state)
        return res
    if verb == "exec":
        return cmd_exec(ctx, pos, opts, cmd)
    if verb == "run":
        return cmd_run(ctx, pos, opts, cmd)
    if verb == "rollout":
        res = cmd_rollout(state, pos, opts)
        reconcile(state)
        return res
    if verb == "scale":
        res = cmd_scale(state, pos, opts)
        reconcile(state)
        return res
    if verb == "wait":
        return cmd_wait(state, pos, opts)
    if verb == "edit":
        raise CommandError(
            "error: kubectl edit needs an interactive editor, which this lab does not provide.\n"
            "Use `kubectl patch ... -p '<json>'`, or change the manifest file and `kubectl apply -f` it.",
            1,
        )
    raise CommandError(f'error: unknown command "{verb}" for "kubectl"\n\nDid you mean this?\n\tget\n\tapply', 1)
