"""Argument parsing and verb dispatch for the ``kubectl`` command."""

from __future__ import annotations

import json
import re

import yaml

from ...core.errors import CommandError
from ...core.state import HEAD
from .. import command
from .apply import apply_doc, cmd_create, delete_obj, load_docs
from .edit import (
    cmd_cordon,
    cmd_drain,
    cmd_expose,
    cmd_label,
    cmd_patch,
    cmd_rollout,
    cmd_scale,
    cmd_set,
    cmd_taint,
    cmd_wait,
)
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
    "--target-port",
    "--protocol",
    "--name",
    "--to-revision",
    "--revision",
    "--api-version",
    "--cluster",
    "--user",
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
 set        set image deploy/NAME CONTAINER=IMAGE: a new revision of a deployment
 expose     Create a Service for a deployment or pod (--port, --type)
 explain    Documentation of a resource and its fields
 config     get-contexts | current-context | view: the kubeconfig in use
 scale / wait / version / cluster-info / api-resources
"""


# kind -> (Kind, apiVersion, description, {field path: description}); the fields a candidate reads in the lab
EXPLAIN = {
    "pods": (
        "Pod",
        "v1",
        "Pod is a collection of containers that can run on a host. This resource is created by clients and scheduled onto hosts.",
        {
            "spec": "Specification of the desired behavior of the pod.",
            "spec.containers": "List of containers belonging to the pod. Cannot be updated.",
            "spec.containers.resources": "Compute resources required by the container, limits and requests. GPUs are requested as the extended resource nvidia.com/gpu under limits.",
            "spec.containers.resources.limits": "Maximum compute resources allowed; an extended resource such as nvidia.com/gpu goes here and the request is set to the same value.",
            "spec.nodeSelector": "Selector which must be true for the pod to fit on a node: key/value pairs the node's labels must carry.",
            "spec.tolerations": "The pod's tolerations; a pod needs one for every taint on the node it should land on.",
            "spec.runtimeClassName": "RuntimeClass to run this pod with (nvidia for the NVIDIA container runtime).",
            "status": "Most recently observed status of the pod, populated by the system.",
            "status.phase": "Pending, Running, Succeeded, Failed or Unknown.",
        },
    ),
    "deployments": (
        "Deployment",
        "apps/v1",
        "Deployment enables declarative updates for Pods and ReplicaSets.",
        {
            "spec": "Specification of the desired behavior of the Deployment.",
            "spec.replicas": "Number of desired pods. Defaults to 1.",
            "spec.selector": "Label selector for pods; it must match the pod template's labels.",
            "spec.template": "Template describing the pods that will be created.",
            "spec.strategy": "The deployment strategy used to replace existing pods with new ones (RollingUpdate or Recreate).",
            "status": "Most recently observed status of the Deployment.",
        },
    ),
    "daemonsets": (
        "DaemonSet",
        "apps/v1",
        "DaemonSet represents the configuration of a daemon set: one copy of the pod on every node that matches its selector and tolerates the node's taints.",
        {
            "spec": "The desired behavior of this daemon set.",
            "spec.template": "An object that describes the pod that will be created.",
            "spec.updateStrategy": "An update strategy to replace existing DaemonSet pods with new pods.",
        },
    ),
    "nodes": (
        "Node",
        "v1",
        "Node is a worker node in Kubernetes. Each node will have a unique identifier in the cache (i.e. in etcd).",
        {
            "spec": "Spec defines the behavior of a node.",
            "spec.taints": "If specified, the node's taints.",
            "spec.unschedulable": "Unschedulable controls node schedulability of new pods. By default, node is schedulable.",
            "status": "Most recently observed status of the node.",
            "status.allocatable": "Allocatable represents the resources of a node that are available for scheduling; nvidia.com/gpu appears here once the device plugin has registered.",
            "status.capacity": "Capacity represents the total resources of a node.",
            "status.conditions": "Conditions is an array of current observed node conditions (Ready, MemoryPressure, DiskPressure, PIDPressure, NetworkUnavailable).",
        },
    ),
    "namespaces": (
        "Namespace",
        "v1",
        "Namespace provides a scope for Names. Use of multiple namespaces is optional.",
        {},
    ),
    "services": (
        "Service",
        "v1",
        "Service is a named abstraction of software service consisting of local port that the proxy listens on, and the selector that determines which pods will answer requests sent through the proxy.",
        {
            "spec": "Spec defines the behavior of a service.",
            "spec.type": "ClusterIP, NodePort, LoadBalancer or ExternalName.",
            "spec.ports": "The list of ports that are exposed by this service.",
            "spec.selector": "Route service traffic to pods with label keys and values matching this selector.",
        },
    ),
    "clusterpolicies": (
        "ClusterPolicy",
        "nvidia.com/v1",
        "ClusterPolicy is the Schema for the clusterpolicies API of the NVIDIA GPU Operator: one object that configures every operand (driver, toolkit, device plugin, DCGM exporter, MIG manager, GFD).",
        {
            "spec": "ClusterPolicySpec defines the desired state of ClusterPolicy.",
            "spec.driver": "Driver component spec: enabled, version, repository; enabled false when the driver is preinstalled on the hosts.",
            "spec.toolkit": "Toolkit component spec (nvidia-container-toolkit).",
            "spec.devicePlugin": "DevicePlugin component spec.",
            "spec.mig": "MIG strategy: none, single or mixed.",
            "spec.migManager": "MIGManager component spec.",
            "spec.dcgmExporter": "DCGMExporter component spec.",
            "spec.gfd": "GPU Feature Discovery component spec.",
            "status": "ClusterPolicyStatus defines the observed state of ClusterPolicy.",
            "status.state": "ready, notReady or ignored.",
        },
    ),
    "replicasets": (
        "ReplicaSet",
        "apps/v1",
        "ReplicaSet ensures that a specified number of pod replicas are running at any given time. Deployments manage ReplicaSets; one is kept per revision.",
        {
            "spec": "Spec defines the specification of the desired behavior of the ReplicaSet.",
            "spec.replicas": "Replicas is the number of desired replicas.",
        },
    ),
}


def cmd_explain(pos: list[str], opts: dict) -> str:
    if not pos:
        raise CommandError(
            'error: You must specify the type of resource to explain. Use "kubectl api-resources" for a complete list of supported resources.',
            1,
        )
    word, _, path = pos[0].partition(".")
    kind = resolve_kind(word)
    if kind not in EXPLAIN:
        raise CommandError(
            f'error: no documentation is available for {kind} in this lab; see "kubectl api-resources"', 1
        )
    name, api, desc, fields = EXPLAIN[kind]
    group, _, version = api.rpartition("/")
    head = [f"GROUP:      {group}" if group else "GROUP:      ", f"KIND:       {name}", f"VERSION:    {version}", ""]
    if path:
        if path not in fields:
            raise CommandError(f'error: field "{path}" does not exist', 1)
        out = head + [f"FIELD: {path.rsplit('.', 1)[-1]} <Object>", "", "DESCRIPTION:", f"    {fields[path]}"]
        children = sorted(f for f in fields if f.startswith(path + ".") and f.count(".") == path.count(".") + 1)
        if children:
            out += ["", "FIELDS:"] + [f"  {c.rsplit('.', 1)[-1]}\t<Object>" for c in children]
        return "\n".join(out) + "\n"
    out = head + [
        "DESCRIPTION:",
        f"    {desc}",
        "",
        "FIELDS:",
        "  apiVersion\t<string>",
        "  kind\t<string>",
        "  metadata\t<ObjectMeta>",
    ]
    out += [f"  {f}\t<Object>" for f in sorted(fields) if "." not in f]
    if opts.get("recursive"):
        out += [f"    {f.split('.', 1)[1]}\t<Object>" for f in sorted(fields) if "." in f]
    return "\n".join(out) + "\n"


def cmd_config(state, pos: list[str], opts: dict) -> str:
    sub = pos[0] if pos else ""
    ctx_name = "kubernetes-admin@kubernetes"
    if sub == "current-context":
        return ctx_name + "\n"
    if sub == "get-contexts":
        if opts.get("output") == "name":
            return ctx_name + "\n"
        return table(
            [
                ("CURRENT", "NAME", "CLUSTER", "AUTHINFO", "NAMESPACE"),
                ("*", ctx_name, "kubernetes", "kubernetes-admin", ""),
            ]
        )
    if sub == "view":
        doc = {
            "apiVersion": "v1",
            "clusters": [
                {
                    "cluster": {"certificate-authority-data": "DATA+OMITTED", "server": "https://10.141.255.254:6443"},
                    "name": "kubernetes",
                }
            ],
            "contexts": [{"context": {"cluster": "kubernetes", "user": "kubernetes-admin"}, "name": ctx_name}],
            "current-context": ctx_name,
            "kind": "Config",
            "preferences": {},
            "users": [
                {
                    "name": "kubernetes-admin",
                    "user": {"client-certificate-data": "DATA+OMITTED", "client-key-data": "DATA+OMITTED"},
                }
            ],
        }
        if opts.get("raw"):
            raise CommandError(
                "error: --raw is not available in this lab: the admin credentials are not part of the simulation", 1
            )
        if opts.get("output") == "json":
            return json.dumps(doc, indent=2) + "\n"
        return yaml.safe_dump(doc, sort_keys=False)
    if sub in ("use-context", "set-context", "set-cluster", "set-credentials", "delete-context", "rename-context"):
        raise CommandError(
            f"error: kubectl config {sub} is not supported in this lab: the cluster has one context, {ctx_name}", 1
        )
    raise CommandError("error: kubectl config needs a subcommand: current-context, get-contexts or view", 1)


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
    leading: list[str] = []
    while args and args[0].startswith("-"):  # kubectl -n llm describe pod ...
        if "=" in args[0] or args[0] not in ("-n", "--namespace", "--context", "--kubeconfig", "-s", "--server"):
            leading += args[:1]
            args = args[1:]
        else:
            leading += args[:2]
            args = args[2:]
    if not args:
        return HELP_TEXT
    verb, rest = args[0], args[1:] + leading
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
    if verb == "set":
        res = cmd_set(state, pos, opts)
        reconcile(state)
        return res
    if verb == "expose":
        return cmd_expose(state, pos, opts)
    if verb == "explain":
        return cmd_explain(pos, opts)
    if verb == "config":
        return cmd_config(state, pos, opts)
    if verb == "wait":
        return cmd_wait(state, pos, opts)
    if verb == "edit":
        raise CommandError(
            "error: kubectl edit needs an interactive editor, which this lab does not provide.\n"
            "Use `kubectl patch ... -p '<json>'`, or change the manifest file and `kubectl apply -f` it.",
            1,
        )
    raise CommandError(f'error: unknown command "{verb}" for "kubectl"\n\nDid you mean this?\n\tget\n\tapply', 1)
