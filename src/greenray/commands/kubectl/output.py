"""``kubectl get`` and ``kubectl describe``: object listing, table printers and output formats."""

from __future__ import annotations

import datetime
import hashlib
import json
import re

import yaml

from ...core.errors import CommandError
from ...core.state import AttrDict
from .jsonpath import custom_columns, jsonpath
from .model import (
    CLUSTER_SCOPED,
    CRI,
    EPOCH,
    EXTRA_KINDS,
    K8S_VERSION,
    KIND_PREFIX,
    SINGULAR,
    age,
    events_for,
    fmt_taint,
    is_extended,
    node_taints,
    ns_of,
    pod_restarts,
    pod_tolerations,
    restarts_str,
    selector_matches,
    split_targets,
    table,
    ts,
)
from .scheduler import node_used
from .views import cp_obj, deploy_obj, ds_obj, job_obj, ncp_obj, node_obj, pod_obj, secret_obj


def list_objects(state, kind: str, opts: dict, names=()) -> list:
    k = state.k8s
    all_ns = opts.get("all-namespaces")
    ns = ns_of(opts)
    sel = opts.get("selector")
    fsel = opts.get("field-selector")

    def ns_ok(o):
        return kind in CLUSTER_SCOPED or all_ns or o.get("namespace") == ns

    if kind == "pods":
        items = [p for p in k.pods if ns_ok(p)]
        if sel:
            items = [p for p in items if selector_matches(sel, p.labels)]
        if fsel:
            for cond in fsel.split(","):
                f, op, v = re.match(r"([\w.]+)(!=|==|=)(.*)", cond).groups()
                getter = {
                    "spec.nodeName": lambda p: p.node or "",
                    "status.phase": lambda p: p.phase,
                    "metadata.name": lambda p: p.name,
                    "metadata.namespace": lambda p: p.namespace,
                }.get(f)
                if getter:
                    items = [p for p in items if (getter(p) == v) != (op == "!=")]
        items.sort(key=lambda p: (p.namespace, p.name))
    elif kind == "nodes":
        items = [AttrDict({"name": n, **{}}) for n in sorted(k.nodes)]
        if sel:
            items = [i for i in items if selector_matches(sel, k.nodes[i.name].labels)]
    elif kind == "daemonsets":
        items = sorted([d for d in k.daemonsets if ns_ok(d)], key=lambda d: (d.namespace, d.name))
    elif kind == "deployments":
        items = sorted([d for d in k.deployments if ns_ok(d)], key=lambda d: (d.namespace, d.name))
        if sel:
            items = [d for d in items if selector_matches(sel, d.labels)]
    elif kind == "jobs":
        items = sorted([d for d in k.jobs if ns_ok(d)], key=lambda d: (d.namespace, d.name))
    elif kind == "namespaces":
        items = [AttrDict({"name": n}) for n in sorted(k.namespaces)]
    elif kind == "events":
        items = [e for e in k.events if all_ns or e.namespace == ns]
        if fsel:
            for cond in fsel.split(","):
                f, op, v = re.match(r"([\w.]+)(!=|==|=)(.*)", cond).groups()
                key = {
                    "involvedObject.name": "name",
                    "involvedObject.kind": "kind",
                    "type": "type",
                    "reason": "reason",
                }.get(f)
                if key:
                    items = [e for e in items if (str(e.get(key)) == v) != (op == "!=")]
        items = sorted(items, key=lambda e: e.last)
    elif kind == "clusterpolicies":
        items = [k.clusterpolicy] if k.get("clusterpolicy") else []
        if not crd_installed(state, "clusterpolicies"):
            raise CommandError('error: the server doesn\'t have a resource type "clusterpolicies"', 1)
    elif kind == "nicclusterpolicies":
        if not crd_installed(state, "nicclusterpolicies"):
            raise CommandError('error: the server doesn\'t have a resource type "nicclusterpolicies"', 1)
        items = [k.nicclusterpolicy] if k.get("nicclusterpolicy") else []
    elif kind == "secrets":
        items = sorted([s for s in k.secrets if ns_ok(s)], key=lambda s: (s.namespace, s.name))
        sa_tok = []
        items = sa_tok + items
    elif kind == "configmaps":
        items = sorted([s for s in k.configmaps if ns_ok(s)], key=lambda s: (s.namespace, s.name))
    elif kind == "runtimeclasses":
        items = sorted(k.runtimeclasses, key=lambda r: r.name)
    elif kind == "serviceaccounts":
        nss = k.namespaces if all_ns else [ns]
        items = [
            AttrDict(
                {
                    "name": "default",
                    "namespace": n_,
                    "created": k.ns_created.get(n_, 0),
                    "imagePullSecrets": (k.serviceaccounts.get(n_, {}).get("default") or {}).get(
                        "imagePullSecrets", []
                    ),
                }
            )
            for n_ in nss
            if n_ in k.namespaces
        ]
    elif kind == "services":
        items = (
            [AttrDict({"name": "kubernetes", "namespace": "default", "created": -41 * 86400})]
            if (all_ns or ns == "default")
            else []
        )
    elif kind in EXTRA_KINDS:
        items = [i for i in EXTRA_KINDS[kind].list(state, opts) if ns_ok(i)]
        if sel:
            items = [i for i in items if selector_matches(sel, i.get("labels") or {})]
    else:
        items = []
    if names:
        found = []
        for nm in names:
            hit = [i for i in items if i.get("name") == nm]
            if not hit:
                if kind == "nodes" and nm in k.nodes:
                    hit = [AttrDict({"name": nm})]
                else:
                    scope = "" if kind in CLUSTER_SCOPED else ""
                    if kind in EXTRA_KINDS:
                        scope = f"{kind}.{EXTRA_KINDS[kind].api_version.split('/')[0]}"
                        raise CommandError(f'Error from server (NotFound): {scope} "{nm}" not found', 1)
                    raise CommandError(
                        f'Error from server (NotFound): {kind if kind != "clusterpolicies" else "clusterpolicies.nvidia.com"} "{nm}" not found{scope}',
                        1,
                    )
            found.extend(hit)
        items = found
    return items


def crd_installed(state, kind: str) -> bool:
    k = state.k8s
    chart = {"clusterpolicies": "gpu-operator", "nicclusterpolicies": "network-operator"}[kind]
    return (
        any(r.get("chart") == chart for r in k.helm_releases)
        or (kind == "clusterpolicies" and k.get("clusterpolicy"))
        or (kind == "nicclusterpolicies" and k.get("nicclusterpolicy"))
    )


def api_obj(state, kind: str, item) -> dict:
    if kind in EXTRA_KINDS:
        return EXTRA_KINDS[kind].api(state, item)
    if kind == "pods":
        return pod_obj(state, item)
    if kind == "nodes":
        return node_obj(state, item.name)
    if kind == "daemonsets":
        return ds_obj(state, item)
    if kind == "deployments":
        return deploy_obj(state, item)
    if kind == "jobs":
        return job_obj(state, item)
    if kind == "clusterpolicies":
        return cp_obj(state)
    if kind == "nicclusterpolicies":
        return ncp_obj(state)
    if kind == "secrets":
        return secret_obj(state, item)
    if kind == "namespaces":
        return {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": item.name}, "status": {"phase": "Active"}}
    if kind == "runtimeclasses":
        return {
            "apiVersion": "node.k8s.io/v1",
            "kind": "RuntimeClass",
            "metadata": {"name": item.name},
            "handler": item.handler,
        }
    if kind == "configmaps":
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": item.name, "namespace": item.namespace},
            "data": json.loads(json.dumps(item.get("data") or {})),
        }
    if kind == "serviceaccounts":
        o = {"apiVersion": "v1", "kind": "ServiceAccount", "metadata": {"name": item.name, "namespace": item.namespace}}
        if item.get("imagePullSecrets"):
            o["imagePullSecrets"] = json.loads(json.dumps(item.imagePullSecrets))
        return o
    if kind == "events":
        return {
            "apiVersion": "v1",
            "kind": "Event",
            "type": item.type,
            "reason": item.reason,
            "message": item.message,
            "count": item.count,
            "involvedObject": {"kind": item.kind, "name": item.name, "namespace": item.namespace},
            "source": {"component": item.source},
            "lastTimestamp": ts(item.last),
            "firstTimestamp": ts(item.first),
        }
    return {"apiVersion": "v1", "kind": SINGULAR[kind].title(), "metadata": {"name": item.get("name")}}


def node_status_str(kn) -> str:
    s = "Ready" if kn.get("ready", True) else "NotReady"
    if not kn.get("schedulable", True):
        s += ",SchedulingDisabled"
    return s


def labels_str(labels) -> str:
    return ",".join(f"{k_}={v}" for k_, v in sorted((labels or {}).items())) or "<none>"


def print_table(state, kind: str, items: list, opts: dict, prefix: str = "") -> str:
    k = state.k8s
    wide = opts.get("output") == "wide"
    all_ns = opts.get("all-namespaces") and kind not in CLUSTER_SCOPED
    show_labels = opts.get("show-labels")
    lcols = [c for spec in opts.get("label-columns", []) for c in spec.split(",")]
    rows = []
    if kind == "pods":
        hdr = ["NAME", "READY", "STATUS", "RESTARTS", "AGE"] + (
            ["IP", "NODE", "NOMINATED NODE", "READINESS GATES"] if wide else []
        )
        for p in items:
            r = [p.name, p.ready, p.status, restarts_str(state, p), age(state, p.created)]
            if wide:
                r += [p.get("ip") or "<none>", p.node or "<none>", "<none>", "<none>"]
            rows.append((p, r))
    elif kind == "nodes":
        hdr = ["NAME", "STATUS", "ROLES", "AGE", "VERSION"]
        if wide:
            hdr += ["INTERNAL-IP", "EXTERNAL-IP", "OS-IMAGE", "KERNEL-VERSION", "CONTAINER-RUNTIME"]
        for i in items:
            kn, sn = k.nodes[i.name], state.nodes.get(i.name)
            r = [
                i.name,
                node_status_str(kn),
                kn.get("roles", "<none>"),
                age(state, kn.created),
                kn.get("version", K8S_VERSION),
            ]
            if wide:
                r += [sn.ip, "<none>", sn.os.split(" (")[0], sn.kernel, CRI]
            rows.append((AttrDict({"labels": kn.labels, "namespace": ""}), r))
    elif kind == "daemonsets":
        hdr = ["NAME", "DESIRED", "CURRENT", "READY", "UP-TO-DATE", "AVAILABLE", "NODE SELECTOR", "AGE"]
        if wide:
            hdr += ["CONTAINERS", "IMAGES", "SELECTOR"]
        for d in items:
            o = ds_obj(state, d)["status"]
            r = [
                d.name,
                o["desiredNumberScheduled"],
                o["currentNumberScheduled"],
                o["numberReady"],
                o["updatedNumberScheduled"],
                o["numberAvailable"],
                ",".join(f"{a}={b}" for a, b in d.node_selector.items()) or "<none>",
                age(state, d.created),
            ]
            if wide:
                r += [d.container, d.image, f"app={d.name}"]
            rows.append((d, r))
    elif kind == "deployments":
        hdr = ["NAME", "READY", "UP-TO-DATE", "AVAILABLE", "AGE"]
        for d in items:
            st = deploy_obj(state, d)["status"]
            rows.append(
                (
                    d,
                    [
                        d.name,
                        f"{st['readyReplicas']}/{d.replicas}",
                        st["updatedReplicas"],
                        st["availableReplicas"],
                        age(state, d.created),
                    ],
                )
            )
    elif kind == "jobs":
        hdr = ["NAME", "COMPLETIONS", "DURATION", "AGE"]
        for j in items:
            st = job_obj(state, j)["status"]
            rows.append(
                (
                    j,
                    [
                        j.name,
                        f"{st['succeeded']}/{j.get('completions', 1)}",
                        age(state, j.created),
                        age(state, j.created),
                    ],
                )
            )
    elif kind == "namespaces":
        hdr = ["NAME", "STATUS", "AGE"]
        for i in items:
            rows.append(
                (AttrDict({"namespace": ""}), [i.name, "Active", age(state, k.ns_created.get(i.name, -41 * 86400))])
            )
    elif kind == "events":
        hdr = ["LAST SEEN", "TYPE", "REASON", "OBJECT", "MESSAGE"]
        for e in items:
            rows.append((e, [age(state, e.last), e.type, e.reason, f"{e.kind.lower()}/{e.name}", e.message]))
    elif kind == "clusterpolicies" or kind == "nicclusterpolicies":
        hdr = ["NAME", "STATUS", "AGE"]
        for c in items:
            rows.append(
                (
                    AttrDict({"namespace": ""}),
                    [c.name, (c.get("status") or {}).get("state", "notReady"), age(state, c.created)],
                )
            )
    elif kind == "secrets":
        hdr = ["NAME", "TYPE", "DATA", "AGE"]
        for s in items:
            rows.append((s, [s.name, s.get("type", "Opaque"), len(s.get("data") or {}), age(state, s.created)]))
    elif kind == "configmaps":
        hdr = ["NAME", "DATA", "AGE"]
        for s in items:
            rows.append((s, [s.name, len(s.get("data") or {}), age(state, s.created)]))
    elif kind == "runtimeclasses":
        hdr = ["NAME", "HANDLER", "AGE"]
        for r_ in items:
            rows.append((AttrDict({"namespace": ""}), [r_.name, r_.handler, age(state, r_.created)]))
    elif kind == "serviceaccounts":
        hdr = ["NAME", "SECRETS", "AGE"]
        for s in items:
            rows.append((s, [s.name, "0", age(state, s.created)]))
    elif kind == "services":
        hdr = ["NAME", "TYPE", "CLUSTER-IP", "EXTERNAL-IP", "PORT(S)", "AGE"]
        for s in items:
            rows.append((s, [s.name, "ClusterIP", "10.96.0.1", "<none>", "443/TCP", age(state, s.created)]))
    elif kind in EXTRA_KINDS:
        hdr, raw = EXTRA_KINDS[kind].table(state, items)
        rows = [(i, list(r)) for i, r in zip(items, raw, strict=True)]
    else:
        hdr = ["NAME"]
        rows = [(i, [i.get("name")]) for i in items]
    if prefix:
        rows = [(o, [prefix + str(r[0])] + r[1:]) for o, r in rows]
    if lcols:
        hdr += [c.split("/")[-1].upper() for c in lcols]
        for obj, r in rows:
            r += [(obj.get("labels") or {}).get(c, "") for c in lcols]
    if show_labels:
        hdr.append("LABELS")
        for obj, r in rows:
            r.append(labels_str(obj.get("labels")))
    if all_ns:
        hdr = ["NAMESPACE"] + hdr
        rows = [(o, [o.get("namespace", "")] + r) for o, r in rows]
    return table([hdr] + [r for _, r in rows])


def output(state, kind: str, items: list, opts: dict, single: bool = False) -> str:
    fmt = opts.get("output")
    if fmt in (None, "wide"):
        if not items:
            if kind in CLUSTER_SCOPED or opts.get("all-namespaces"):
                return "No resources found\n"
            return f"No resources found in {ns_of(opts)} namespace.\n"
        return print_table(state, kind, items, opts)
    objs = [api_obj(state, kind, i) for i in items]
    doc = (
        objs[0]
        if single and len(objs) == 1
        else {"apiVersion": "v1", "items": objs, "kind": "List", "metadata": {"resourceVersion": ""}}
    )
    if fmt == "yaml":
        return yaml.safe_dump(doc, default_flow_style=False, sort_keys=True)
    if fmt == "json":
        return json.dumps(doc, indent=4) + "\n"
    if fmt == "name":
        return "".join(f"{KIND_PREFIX[kind]}/{i.get('name')}\n" for i in items)
    if fmt.startswith("jsonpath=") or fmt.startswith("jsonpath-as-json=") or fmt.startswith("jsonpath="):
        return jsonpath(doc, fmt.split("=", 1)[1])
    if fmt.startswith("custom-columns="):
        return custom_columns(objs, fmt.split("=", 1)[1])
    raise CommandError(
        f'error: unable to match a printer suitable for the output format "{fmt}", allowed formats are: '
        "custom-columns,go-template,json,jsonpath,name,wide,yaml",
        1,
    )


def cmd_get(state, pos: list[str], opts: dict) -> str:
    targets = split_targets(pos)
    multi = len(targets) > 1
    out = []
    for kind, names in targets:
        items = list_objects(state, kind, opts, names)
        if multi and opts.get("output") in (None, "wide"):
            if items:
                out.append(print_table(state, kind, items, opts, prefix=KIND_PREFIX[kind] + "/").rstrip("\n"))
            continue
        out.append(output(state, kind, items, opts, single=bool(names) and len(names) == 1 and not multi).rstrip("\n"))
    if not out:
        return f"No resources found in {ns_of(opts)} namespace.\n"
    return "\n\n".join(out) + "\n"


def _kv_block(d, indent=2, width=None):
    if not d:
        return " " * indent + "<none>"
    width = width or max(len(k_) for k_ in d) + 3
    return "\n".join(" " * indent + f"{k_ + ':':<{width}}{v}" for k_, v in d.items())


def describe_events(state, uid: str, indent: str = "") -> str:
    evs = events_for(state, uid)
    if not evs:
        return "Events:              <none>"
    rows = [("  Type", "Reason", "Age", "From", "Message"), ("  ----", "------", "----", "----", "-------")]
    for e in sorted(evs, key=lambda e: e.last):
        a = age(state, e.last) if e.count == 1 else f"{age(state, e.last)} (x{e.count} over {age(state, e.first)})"
        rows.append(("  " + e.type, e.reason, a, e.source, e.message))
    return "Events:\n" + table(rows, gap=2).rstrip("\n")


def describe_node(state, n: str) -> str:
    k = state.k8s
    kn, sn = k.nodes[n], state.nodes[n]
    lines = [f"Name:               {n}", f"Roles:              {kn.get('roles', '<none>')}"]
    labs = [f"{a}={b}" for a, b in sorted(kn.labels.items())]
    lines.append("Labels:             " + ("\n                    ".join(labs) if labs else "<none>"))
    lines.append(
        "Annotations:        node.alpha.kubernetes.io/ttl: 0\n                    volumes.kubernetes.io/controller-managed-attach-detach: true"
    )
    lines.append(f"CreationTimestamp:  {(EPOCH + datetime.timedelta(seconds=kn.created)):%a, %d %b %Y %H:%M:%S +0000}")
    taints = [fmt_taint(t) for t in node_taints(state, n)]
    lines.append("Taints:             " + ("\n                    ".join(taints) if taints else "<none>"))
    lines.append(f"Unschedulable:      {'false' if kn.get('schedulable', True) else 'true'}")
    ready = kn.get("ready", True)
    lines.append(
        "Conditions:\n"
        "  Type             Status  Reason                       Message\n"
        "  ----             ------  ------                       -------\n"
        "  MemoryPressure   False   KubeletHasSufficientMemory   kubelet has sufficient memory available\n"
        "  DiskPressure     False   KubeletHasNoDiskPressure     kubelet has no disk pressure\n"
        "  PIDPressure      False   KubeletHasSufficientPID      kubelet has sufficient PID available\n"
        + (
            "  Ready            True    KubeletReady                 kubelet is posting ready status"
            if ready
            else "  Ready            Unknown NodeStatusUnknown            Kubelet stopped posting node status."
        )
    )
    lines.append(f"Addresses:\n  InternalIP:  {sn.ip}\n  Hostname:    {n}")
    w = max(len(r) for r in list(kn.capacity) + list(kn.allocatable)) + 3
    lines.append("Capacity:\n" + _kv_block(dict(sorted(kn.capacity.items())), width=w))
    lines.append("Allocatable:\n" + _kv_block(dict(sorted(kn.allocatable.items())), width=w))
    lines.append(
        "System Info:\n"
        + _kv_block(
            {
                "Machine ID": hashlib.md5(n.encode()).hexdigest(),
                "Kernel Version": sn.kernel,
                "OS Image": sn.os.split(" (")[0],
                "Operating System": "linux",
                "Architecture": "amd64",
                "Container Runtime Version": CRI,
                "Kubelet Version": kn.get("version", K8S_VERSION),
                "Kube-Proxy Version": kn.get("version", K8S_VERSION),
            },
            indent=2,
            width=28,
        )
    )
    lines.append(f"PodCIDR:                      10.244.{sorted(k.nodes).index(n)}.0/24")
    pods = [p for p in k.pods if p.node == n and p.phase not in ("Succeeded", "Failed")]
    rows = [
        ("  Namespace", "Name", "CPU Requests", "CPU Limits", "Memory Requests", "Memory Limits", "Age"),
        ("  ---------", "----", "------------", "----------", "---------------", "-------------", "---"),
    ]
    for p in sorted(pods, key=lambda p: (p.namespace, p.name)):
        rows.append(("  " + p.namespace, p.name, "0 (0%)", "0 (0%)", "0 (0%)", "0 (0%)", age(state, p.created)))
    lines.append(f"Non-terminated Pods:          ({len(pods)} in total)\n" + table(rows, gap=2).rstrip("\n"))
    used = node_used(state, n)
    ar = [
        ("  Resource", "Requests", "Limits"),
        ("  --------", "--------", "------"),
        ("  cpu", "100m (0%)", "0 (0%)"),
        ("  memory", "70Mi (0%)", "170Mi (0%)"),
        ("  ephemeral-storage", "0 (0%)", "0 (0%)"),
        ("  hugepages-1Gi", "0 (0%)", "0 (0%)"),
        ("  hugepages-2Mi", "0 (0%)", "0 (0%)"),
    ]
    for r in sorted(set(kn.capacity) | set(used)):
        if is_extended(r):
            v = used.get(r, 0)
            ar.append(("  " + r, str(v), str(v)))
    lines.append(
        "Allocated resources:\n  (Total limits may be over 100 percent, i.e., overcommitted.)\n"
        + table(ar, gap=2).rstrip("\n")
    )
    lines.append(describe_events(state, "node-" + n))
    return "\n".join(lines) + "\n"


def describe_pod(state, p) -> str:
    lines = [
        f"Name:             {p.name}",
        f"Namespace:        {p.namespace}",
        "Priority:         0",
        "Service Account:  default",
        f"Node:             {p.node + '/' + state.nodes[p.node].ip if p.node else '<none>'}",
    ]
    if p.node:
        lines.append(f"Start Time:       {(EPOCH + datetime.timedelta(seconds=p.created)):%a, %d %b %Y %H:%M:%S +0000}")
    labs = [f"{a}={b}" for a, b in sorted((p.labels or {}).items())]
    lines.append("Labels:           " + ("\n                  ".join(labs) if labs else "<none>"))
    lines.append("Annotations:      <none>")
    lines.append(f"Status:           {p.phase}")
    lines.append(f"IP:               {p.get('ip') or ''}")
    if p.get("owner"):
        lines.append(f"Controlled By:    {p.owner.kind}/{p.owner.name}")
    lines.append("Containers:")
    for c in p.spec.get("containers", []):
        lines.append(f"  {c.get('name')}:")
        lines.append(f"    Image:      {c.get('image')}")
        if c.get("command"):
            lines.append("    Command:\n" + "\n".join(f"      {x}" for x in c.command))
        if c.get("args"):
            lines.append("    Args:\n" + "\n".join(f"      {x}" for x in c.args))
        if p.status == "Running":
            lines.append(
                f"    State:          Running\n      Started:      {(EPOCH + datetime.timedelta(seconds=p.get('started') or 0)):%a, %d %b %Y %H:%M:%S +0000}"
            )
            lines.append("    Ready:          True")
        elif p.status == "Completed":
            lines.append("    State:          Terminated\n      Reason:       Completed\n      Exit Code:    0")
            lines.append("    Ready:          False")
        elif p.node:
            reason = p.status if not p.status.startswith("Init") else "PodInitializing"
            lines.append(f"    State:          Waiting\n      Reason:       {reason}")
            if p.status in ("CrashLoopBackOff", "Error"):
                lines.append("    Last State:     Terminated\n      Reason:       Error\n      Exit Code:    1")
            lines.append("    Ready:          False")
        lines.append(f"    Restart Count:  {pod_restarts(state, p)}")
        res = c.get("resources") or {}
        for label in ("limits", "requests"):
            vals = dict(res.get(label) or {})
            if label == "requests":
                for r, v in (res.get("limits") or {}).items():
                    vals.setdefault(r, v)
            if vals:
                lines.append(f"    {label.title()}:\n" + "\n".join(f"      {r}:  {v}" for r, v in vals.items()))
        env = c.get("env") or []
        lines.append(
            "    Environment:"
            + ("  <none>" if not env else "\n" + "\n".join(f"      {e.get('name')}:  {e.get('value')}" for e in env))
        )
        lines.append("    Mounts:\n      /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access (ro)")
    conds = [("  Type", "Status")]
    if p.node:
        conds += [
            ("  PodReadyToStartContainers", "True"),
            ("  Initialized", "True" if not p.status.startswith("Init") else "False"),
            ("  Ready", "True" if p.status == "Running" else "False"),
            ("  ContainersReady", "True" if p.status == "Running" else "False"),
            ("  PodScheduled", "True"),
        ]
    else:
        conds.append(("  PodScheduled", "False"))
    lines.append("Conditions:\n" + table(conds, gap=1).rstrip("\n"))
    lines.append(
        "Volumes:\n  kube-api-access:\n    Type:                    Projected (a volume that contains injected data from multiple sources)"
    )
    has_res = any((c.get("resources") or {}) for c in p.spec.get("containers", []))
    lines.append(f"QoS Class:                   {'Guaranteed' if has_res else 'BestEffort'}")
    ns_sel = p.spec.get("nodeSelector") or {}
    lines.append("Node-Selectors:              " + (",".join(f"{a}={b}" for a, b in ns_sel.items()) or "<none>"))
    tl = []
    for t in pod_tolerations(p):
        s = t.get("key") or ""
        if t.get("operator") == "Exists" and not t.get("key"):
            s = "op=Exists"
        else:
            if t.get("value"):
                s += f"={t.get('value')}"
            if t.get("effect"):
                s += f":{t.get('effect')}"
            if t.get("operator") == "Exists":
                s += " op=Exists"
        if t.get("tolerationSeconds"):
            s += f" for {t.get('tolerationSeconds')}s"
        tl.append(s)
    lines.append("Tolerations:                 " + "\n                             ".join(tl))
    lines.append(describe_events(state, p.uid))
    return "\n".join(lines) + "\n"


def cmd_describe(state, pos: list[str], opts: dict) -> str:
    targets = split_targets(pos)
    out = []
    for kind, names in targets:
        items = list_objects(state, kind, opts, names) if names else list_objects(state, kind, opts)
        if names == [] and kind == "pods" and not items:
            return f"No resources found in {ns_of(opts)} namespace.\n"
        for i in items:
            if kind == "nodes":
                out.append(describe_node(state, i.name))
            elif kind == "pods":
                out.append(describe_pod(state, i))
            else:
                obj = api_obj(state, kind, i)
                meta = obj.pop("metadata", {})
                text = [f"Name:         {meta.get('name')}"]
                if meta.get("namespace"):
                    text.append(f"Namespace:    {meta.get('namespace')}")
                text.append(f"API Version:  {obj.pop('apiVersion', '')}\nKind:         {obj.pop('kind', '')}")
                body = yaml.safe_dump(obj, default_flow_style=False, sort_keys=False)
                text.append("\n".join(_yaml_to_describe(body)))
                text.append("Events:       <none>")
                out.append("\n".join(text) + "\n")
    return "\n\n".join(o.rstrip("\n") for o in out) + "\n"


def _yaml_to_describe(body):
    out = []
    for line in body.splitlines():
        stripped = line.lstrip()
        ind = len(line) - len(stripped)
        if ":" in stripped and not stripped.startswith("- "):
            k_, _, v = stripped.partition(":")
            k_ = k_[:1].upper() + k_[1:]
            out.append(" " * ind + f"{k_}:" + (f"  {v.strip()}" if v.strip() else ""))
        else:
            out.append(line)
    return out
