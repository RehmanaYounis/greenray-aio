"""Cluster state model: object constructors, pod status bookkeeping, taints/selectors and the kinds registry."""

from __future__ import annotations

import copy
import datetime
import hashlib
import re

from ...core.errors import CommandError
from ...core.state import wrap

# state.k8s layout (scenarios grade the end state, so every field here is stable):
#   nodes[name]      labels, taints, schedulable (cordon), capacity / allocatable, ready
#   pods             one record per pod (see ``new_pod``); ``component`` marks pods owned by system
#                    controllers (GPU Operator operands, NFD, ...), user pods have none
#   deployments      user + system Deployments (pods are named <deploy>-<rs hash>-<5>)
#   daemonsets       DaemonSets created by helm charts, the GPU/Network Operator or user manifests
#   clusterpolicy    the GPU Operator ClusterPolicy {name, namespace, spec, status}
#   nicclusterpolicy the Network Operator NicClusterPolicy
#   secrets / configmaps / runtimeclasses / jobs / serviceaccounts / events
#   _tick            simulated seconds since EPOCH; every kubectl/helm command advances it by TICK

TICK = 15  # simulated seconds that pass per kubectl/helm command
EPOCH = datetime.datetime(2026, 10, 19, 9, 30, 0)
ALPH = "bcdfghjklmnpqrstvwxz2456789"
K8S_VERSION = "v1.29.4"
CRI = "containerd://1.7.13"
SUFFIX_GPU = "nvidia.com/"

DS_AUTO_TOLERATIONS = [
    {"key": "node.kubernetes.io/not-ready", "operator": "Exists"},
    {"key": "node.kubernetes.io/unreachable", "operator": "Exists"},
    {"key": "node.kubernetes.io/unschedulable", "operator": "Exists"},
    {"key": "node.kubernetes.io/disk-pressure", "operator": "Exists"},
]

DEFAULT_POD_TOLERATIONS = [
    {"key": "node.kubernetes.io/not-ready", "operator": "Exists", "effect": "NoExecute", "tolerationSeconds": 300},
    {"key": "node.kubernetes.io/unreachable", "operator": "Exists", "effect": "NoExecute", "tolerationSeconds": 300},
]


# fn(state) run in every reconcile pass, right before the kube-scheduler. Pods whose ``component``
# starts with "runai" are managed by such a hook (runai.py), not by kubelet_sync.
RECONCILE_HOOKS: list = []


def sfx(*parts, n: int = 5) -> str:
    h = hashlib.md5("|".join(map(str, parts)).encode()).digest()
    return "".join(ALPH[b % len(ALPH)] for b in h[:n])


def now(state) -> int:
    return state.k8s.get("_tick", 0)


def advance(state, sec: int = TICK) -> None:
    state.k8s["_tick"] = now(state) + sec


def ts(sec: int) -> str:
    return (EPOCH + datetime.timedelta(seconds=sec)).strftime("%Y-%m-%dT%H:%M:%SZ")


def klog(state, level: str = "I") -> str:
    t = EPOCH + datetime.timedelta(seconds=now(state))
    return f"{level}{t:%m%d %H:%M:%S}.{(now(state) * 7919) % 1000000:06d}       1 "


def human(sec: int) -> str:
    sec = max(0, int(sec))
    if sec < 120:
        return f"{sec}s"
    m = sec // 60
    if m < 10:
        return f"{m}m{sec % 60}s" if sec % 60 else f"{m}m"
    if m < 180:
        return f"{m}m"
    h = m // 60
    if h < 8:
        return f"{h}h{m % 60}m" if m % 60 else f"{h}h"
    if h < 48:
        return f"{h}h"
    d = h // 24
    if d < 8:
        return f"{d}d{h % 24}h" if h % 24 else f"{d}d"
    return f"{d}d"


def age(state, created: int | None) -> str:
    return human(now(state) - (created if created is not None else now(state)))


def table(rows, gap: int = 3) -> str:
    if not rows:
        return ""
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    out = []
    for r in rows:
        cells = [str(c).ljust(widths[i] + gap) if i < len(r) - 1 else str(c) for i, c in enumerate(r)]
        out.append("".join(cells).rstrip())
    return "\n".join(out) + "\n"


def truthy(v, default: bool = True) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() not in ("false", "0", "no", "off", "")


def qty(v) -> int | float:
    """Parse a k8s quantity for countable resources. Returns int or float."""
    if v is None:
        return 0
    s = str(v).strip()
    if s.endswith("m") and s[:-1].isdigit():
        return int(s[:-1]) / 1000
    try:
        f = float(s)
    except ValueError:
        return 0
    return int(f) if f.is_integer() else f


def is_extended(res: str) -> bool:
    return "/" in res and not res.startswith(("kubernetes.io/", "requests.", "hugepages"))


def new_uid(state) -> str:
    k = state.k8s
    k["_uid"] = k.get("_uid", 0) + 1
    h = hashlib.md5(f"uid{k._uid}".encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def pod_spec(name: str, image: str, extra: dict | None = None) -> dict:
    spec = {"containers": [{"name": name, "image": image}]}
    spec.update(extra or {})
    return spec


def new_pod(
    state,
    ns: str,
    name: str,
    spec: dict,
    labels: dict | None = None,
    owner: dict | None = None,
    component: str | None = None,
    node: str | None = None,
    created: int | None = None,
    release: str | None = None,
):
    k = state.k8s
    spec = wrap(copy.deepcopy(dict(spec)))
    pod = {
        "namespace": ns,
        "name": name,
        "uid": new_uid(state),
        "labels": dict(labels or {}),
        "owner": owner,
        "component": component,
        "release": release,
        "spec": spec,
        "node": node,
        "phase": "Pending",
        "status": "Pending",
        "ready": f"0/{len(spec.get('containers', [])) or 1}",
        "restart_base": 0,
        "crash_since": None,
        "last_restart": None,
        "requests": pod_requests(spec),
        "devices": [],
        "logs": [],
        "created": now(state) if created is None else created,
        "started": None,
        "ip": None,
        "sched_msg": "",
    }
    k.pods.append(pod)
    return k.pods[-1]


def add_daemonset(
    state,
    ns: str,
    name: str,
    image: str,
    component: str,
    node_selector: dict | None = None,
    tolerations: list | None = None,
    created: int | None = None,
    release: str | None = None,
    owner: str = "helm",
    container: str | None = None,
    runtime_class: str | None = None,
    extra: dict | None = None,
):
    k = state.k8s
    ds = {
        "namespace": ns,
        "name": name,
        "image": image,
        "component": component,
        "node_selector": dict(node_selector or {}),
        "tolerations": list(tolerations or []),
        "created": now(state) if created is None else created,
        "release": release,
        "owner": owner,
        "container": container or name,
        "runtime_class": runtime_class,
    }
    ds.update(extra or {})
    existing = next((d for d in k.daemonsets if d.namespace == ns and d.name == name), None)
    if existing:
        existing.update({kk: v for kk, v in ds.items() if kk != "created"})
        return existing
    k.daemonsets.append(ds)
    return k.daemonsets[-1]


def add_deployment(
    state,
    ns: str,
    name: str,
    image: str,
    replicas: int = 1,
    component: str | None = None,
    created: int | None = None,
    release: str | None = None,
    template: dict | None = None,
    labels: dict | None = None,
):
    k = state.k8s
    if template is None:
        template = {"metadata": {"labels": {"app": name}}, "spec": pod_spec(name, image)}
    d = {
        "namespace": ns,
        "name": name,
        "replicas": replicas,
        "component": component,
        "release": release,
        "labels": dict(labels or {"app": name}),
        "selector": {"matchLabels": dict(template["metadata"].get("labels", {}))},
        "template": template,
        "created": now(state) if created is None else created,
        "generation": 1,
    }
    k.deployments.append(d)
    return k.deployments[-1]


def ensure_namespace(state, ns: str) -> None:
    k = state.k8s
    if ns not in k.namespaces:
        k.namespaces.append(ns)
        k.ns_created[ns] = now(state)


def remove_release(state, release: str) -> None:
    """Delete everything a helm release created."""
    k = state.k8s
    k.daemonsets = [d for d in k.daemonsets if d.get("release") != release]
    k.deployments = [d for d in k.deployments if d.get("release") != release]
    k.pods = [p for p in k.pods if p.get("release") != release]


def pod_requests(spec: dict) -> dict:
    total = {}
    for c in spec.get("containers", []) or []:
        res = c.get("resources") or {}
        lim = res.get("limits") or {}
        req = res.get("requests") or {}
        for r in set(lim) | set(req):
            if not is_extended(r):
                continue
            v = qty(lim.get(r, req.get(r)))
            total[r] = total.get(r, 0) + v
    return total


def validate_pod_spec(kind: str, name: str, spec: dict, path: str = "spec") -> None:
    errs = []
    containers = spec.get("containers") or []
    if not containers:
        errs.append(f"{path}.containers: Required value")
    for i, c in enumerate(containers):
        p = f"{path}.containers[{i}]"
        if not c.get("name"):
            errs.append(f"{p}.name: Required value")
        if not c.get("image"):
            errs.append(f"{p}.image: Required value")
        res = c.get("resources") or {}
        lim = res.get("limits") or {}
        req = res.get("requests") or {}
        for r, v in req.items():
            if is_extended(r) and r not in lim:
                errs.append(f"{p}.resources.limits: Required value: Limit must be set for non overcommitable resources")
            elif is_extended(r) and qty(v) != qty(lim[r]):
                errs.append(
                    f'{p}.resources.requests[{r}]: Invalid value: "{v}": must be equal to {r} limit of {lim[r]}'
                )
        for r, v in lim.items():
            if is_extended(r) and not float(qty(v)).is_integer():
                errs.append(f'{p}.resources.limits[{r}]: Invalid value: "{v}": must be an integer')
            if r.startswith("nvidia.com/mig.") or r == "nvidia.com/gpus":
                pass
    for t in spec.get("tolerations") or []:
        if t.get("operator") == "Exists" and t.get("value"):
            errs.append(
                f"{path}.tolerations[0].operator: Invalid value: \"{t.get('value')}\": value must be empty when `operator` is 'Exists'"
            )
        if t.get("effect") and t.get("effect") not in ("NoSchedule", "PreferNoSchedule", "NoExecute"):
            errs.append(
                f'{path}.tolerations[0].effect: Unsupported value: "{t.get("effect")}": supported values: "NoExecute", "NoSchedule", "PreferNoSchedule"'
            )
    if errs:
        raise CommandError(
            f'The {kind} "{name}" is invalid: \n* ' + "\n* ".join(errs)
            if len(errs) > 1
            else f'The {kind} "{name}" is invalid: {errs[0]}'
        )


def fmt_taint(t: dict) -> str:
    v = t.get("value")
    return f"{t['key']}={v}:{t['effect']}" if v else f"{t['key']}:{t['effect']}"


def node_taints(state, name: str) -> list:
    kn = state.k8s.nodes[name]
    taints = list(kn.get("taints") or [])
    if not kn.get("schedulable", True):
        taints.append(wrap({"key": "node.kubernetes.io/unschedulable", "effect": "NoSchedule", "value": ""}))
    if not kn.get("ready", True):
        taints.append(wrap({"key": "node.kubernetes.io/unreachable", "effect": "NoSchedule", "value": ""}))
        taints.append(wrap({"key": "node.kubernetes.io/unreachable", "effect": "NoExecute", "value": ""}))
    return taints


def tolerates(tols, taint) -> bool:
    for t in tols or []:
        eff = t.get("effect")
        if eff and eff != taint.get("effect"):
            continue
        op = t.get("operator") or "Equal"
        key = t.get("key")
        if op == "Exists":
            if not key or key == taint.get("key"):
                return True
        elif key == taint.get("key") and str(t.get("value") or "") == str(taint.get("value") or ""):
            return True
    return False


def untolerated(state, name: str, tols, ignore_unschedulable: bool = False):
    for t in node_taints(state, name):
        if t.get("effect") not in ("NoSchedule", "NoExecute"):
            continue
        if ignore_unschedulable and t.key == "node.kubernetes.io/unschedulable":
            continue
        if not tolerates(tols, t):
            return t
    return None


def parse_selector(sel: str | None) -> list:
    out = []
    for part in re.split(r",(?![^()]*\))", sel or ""):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^([^!=\s]+)\s+(in|notin)\s+\(([^)]*)\)$", part)
        if m:
            out.append((m.group(1), m.group(2), [v.strip() for v in m.group(3).split(",")]))
        elif "!=" in part:
            k_, v = part.split("!=", 1)
            out.append((k_.strip(), "!=", [v.strip()]))
        elif "==" in part:
            k_, v = part.split("==", 1)
            out.append((k_.strip(), "=", [v.strip()]))
        elif "=" in part:
            k_, v = part.split("=", 1)
            out.append((k_.strip(), "=", [v.strip()]))
        elif part.startswith("!"):
            out.append((part[1:], "!exists", []))
        else:
            out.append((part, "exists", []))
    return out


def selector_matches(sel, labels) -> bool:
    labels = labels or {}
    for key, op, vals in parse_selector(sel) if isinstance(sel, str) else sel:
        has = key in labels
        v = str(labels.get(key, ""))
        if op == "=" and not (has and v == vals[0]):
            return False
        if op == "!=" and has and v == vals[0]:
            return False
        if op == "exists" and not has:
            return False
        if op == "!exists" and has:
            return False
        if op == "in" and not (has and v in vals):
            return False
        if op == "notin" and has and v in vals:
            return False
    return True


def node_selector_ok(spec: dict, labels: dict) -> bool:
    for key, val in (spec.get("nodeSelector") or {}).items():
        if str(labels.get(key)) != str(val):
            return False
    aff = ((spec.get("affinity") or {}).get("nodeAffinity") or {}).get(
        "requiredDuringSchedulingIgnoredDuringExecution"
    ) or {}
    terms = aff.get("nodeSelectorTerms") or []
    if terms:

        def term_ok(term):
            for e in term.get("matchExpressions") or []:
                key, op, vals = e.get("key"), e.get("operator"), [str(x) for x in e.get("values") or []]
                has, v = key in labels, str(labels.get(key, ""))
                if op == "In" and not (has and v in vals):
                    return False
                if op == "NotIn" and has and v in vals:
                    return False
                if op == "Exists" and not has:
                    return False
                if op == "DoesNotExist" and has:
                    return False
            return True

        if not any(term_ok(t) for t in terms):
            return False
    return True


def event(state, obj_kind: str, obj, etype: str, reason: str, msg: str, source: str = "kubelet") -> None:
    k = state.k8s
    uid = obj.get("uid", obj.get("name"))
    for e in k.events:
        if e.get("uid") == uid and e.reason == reason and e.message == msg:
            if now(state) - e.last >= 60:  # repeated events are aggregated, like the event recorder does
                e.count += 1
                e.last = now(state)
            return
    k.events.append(
        {
            "uid": uid,
            "namespace": obj.get("namespace", "default"),
            "kind": obj_kind,
            "name": obj["name"],
            "type": etype,
            "reason": reason,
            "message": msg,
            "source": source,
            "first": now(state),
            "last": now(state),
            "count": 1,
        }
    )


def events_for(state, uid: str) -> list:
    return [e for e in state.k8s.events if e.get("uid") == uid]


def crash_count(elapsed: int) -> tuple[int, int]:
    t, n, back = 0, 0, 10
    while True:
        t += back
        if t > elapsed:
            return n, t - back
        n += 1
        back = min(back * 2, 300)


def pod_restarts(state, pod) -> int:
    n = pod.get("restart_base", 0)
    if pod.get("crash_since") is not None:
        n += crash_count(now(state) - pod.crash_since)[0]
    return n


def restarts_str(state, pod) -> str:
    n = pod_restarts(state, pod)
    if not n:
        return "0"
    if pod.get("crash_since") is not None:
        _, last = crash_count(now(state) - pod.crash_since)
        when = pod.crash_since + last
    else:
        when = pod.get("last_restart") or pod.created
    return f"{n} ({human(now(state) - when)} ago)"


def set_status(
    state, pod, status: str, ready: str | None = None, logs: list | None = None, crash: bool = False
) -> None:
    if crash:
        if pod.get("crash_since") is None:
            pod.crash_since = now(state)
        if pod_restarts(state, pod) == 0 and now(state) - pod.crash_since < 10:
            status = "Error"
    elif pod.get("crash_since") is not None:
        pod.restart_base = pod_restarts(state, pod) + 1
        pod.last_restart = now(state)
        pod.crash_since = None
    pod.status = status
    if status in ("Running", "CrashLoopBackOff", "Error") or crash:
        pod.phase = "Running"
    elif status == "Completed":
        pod.phase = "Succeeded"
    elif status == "Failed":
        pod.phase = "Failed"
    else:
        pod.phase = "Pending"
    n = len(pod.spec.get("containers", [])) or 1
    pod.ready = ready if ready is not None else (f"{n}/{n}" if status == "Running" else f"0/{n}")
    if status in ("Running", "Completed") and pod.get("started") is None:
        pod.started = now(state)
    if pod.node and not pod.get("ip"):
        nodes = sorted(state.k8s.nodes)
        k = state.k8s
        k["_ip"] = k.get("_ip", 10) + 1
        pod.ip = f"10.244.{nodes.index(pod.node)}.{k._ip}" if pod.node in nodes else "10.244.9.9"
    if logs is not None:
        pod.logs = list(logs)


def pods_of(state, kind: str, name: str, ns: str) -> list:
    return [
        p
        for p in state.k8s.pods
        if p.get("owner") and p.owner.get("kind") == kind and p.owner.get("name") == name and p.namespace == ns
    ]


def remove_pod(state, p) -> None:
    k = state.k8s
    if p.get("owner") and p.owner.get("kind") == "DaemonSet":
        key = f"{p.namespace}/{p.owner.name}/{p.node}"
        k._gen[key] = k._gen.get(key, 0) + 1
    if p.get("owner") and p.owner.get("kind") == "ReplicaSet":
        key = f"{p.namespace}/{p.owner.name}"
        k._gen[key] = k._gen.get(key, 0) + 1
    if p.component == "migManager":
        k.mig_applied.pop(p.node, None)
    k.pods.remove(p)


def cp_spec(state):
    cp = state.k8s.get("clusterpolicy")
    return cp.spec if cp else None


def pod_tolerations(pod) -> list:
    return list(pod.spec.get("tolerations") or []) + DEFAULT_POD_TOLERATIONS


KINDS = {
    "pods": ("po", "pod", "pods"),
    "nodes": ("no", "node", "nodes"),
    "daemonsets": ("ds", "daemonset", "daemonsets", "daemonset.apps", "daemonsets.apps"),
    "deployments": ("deploy", "deployment", "deployments", "deployment.apps", "deployments.apps"),
    "namespaces": ("ns", "namespace", "namespaces"),
    "events": ("ev", "event", "events"),
    "clusterpolicies": ("clusterpolicy", "clusterpolicies", "clusterpolicy.nvidia.com", "clusterpolicies.nvidia.com"),
    "nicclusterpolicies": (
        "nicclusterpolicy",
        "nicclusterpolicies",
        "nicclusterpolicy.mellanox.com",
        "nicclusterpolicies.mellanox.com",
    ),
    "secrets": ("secret", "secrets"),
    "runtimeclasses": ("runtimeclass", "runtimeclasses", "runtimeclass.node.k8s.io"),
    "jobs": ("job", "jobs", "job.batch"),
    "configmaps": ("cm", "configmap", "configmaps"),
    "serviceaccounts": ("sa", "serviceaccount", "serviceaccounts"),
    "services": ("svc", "service", "services"),
}

CLUSTER_SCOPED = {"nodes", "namespaces", "clusterpolicies", "nicclusterpolicies", "runtimeclasses"}

KIND_PREFIX = {
    "pods": "pod",
    "nodes": "node",
    "daemonsets": "daemonset.apps",
    "deployments": "deployment.apps",
    "namespaces": "namespace",
    "events": "event",
    "clusterpolicies": "clusterpolicy.nvidia.com",
    "nicclusterpolicies": "nicclusterpolicy.mellanox.com",
    "secrets": "secret",
    "runtimeclasses": "runtimeclass.node.k8s.io",
    "jobs": "job.batch",
    "configmaps": "configmap",
    "serviceaccounts": "serviceaccount",
    "services": "service",
}

SINGULAR = {
    "pods": "pod",
    "nodes": "node",
    "daemonsets": "daemonset",
    "deployments": "deployment",
    "namespaces": "namespace",
    "events": "event",
    "clusterpolicies": "clusterpolicy",
    "nicclusterpolicies": "nicclusterpolicy",
    "secrets": "secret",
    "runtimeclasses": "runtimeclass",
    "jobs": "job",
    "configmaps": "configmap",
    "serviceaccounts": "serviceaccount",
    "services": "service",
}


# Kinds contributed by other command modules (runai.py): plural -> handler with ``names`` (aliases),
# ``api_version``, ``kind``, ``namespaced``, ``list(state, opts)``, ``table(state, items) -> (header, rows)``,
# ``api(state, item)``, ``apply(state, doc, src) -> str`` and ``delete(state, name) -> str``.
EXTRA_KINDS: dict = {}


def register_kind(plural: str, handler) -> None:
    EXTRA_KINDS[plural] = handler
    KINDS[plural] = tuple(handler.names)
    if not handler.namespaced:
        CLUSTER_SCOPED.add(plural)
    KIND_PREFIX[plural] = f"{handler.kind.lower()}.{handler.api_version.split('/')[0]}"
    SINGULAR[plural] = handler.kind.lower()


def resolve_kind(word: str) -> str:
    w = word.lower()
    for kind, names in KINDS.items():
        if w in names:
            return kind
    raise CommandError(f'error: the server doesn\'t have a resource type "{word}"', 1)


def split_targets(pos: list[str]) -> list[tuple[str, list[str]]]:
    """['pod', 'a', 'b'] or ['pod/a', 'node/b'] or ['pods,ds'] -> [(kind, [names])]"""
    if not pos:
        raise CommandError(
            'You must specify the type of resource to get. Use "kubectl api-resources" for a complete list of supported resources.',
            1,
        )
    if pos[0] == "all":
        return [(kd, pos[1:]) for kd in ("pods", "services", "daemonsets", "deployments", "jobs")]
    if "/" in pos[0]:
        out = []
        for p in pos:
            kw, _, name = p.partition("/")
            out.append((resolve_kind(kw), [name]))
        return out
    kinds = [resolve_kind(w) for w in pos[0].split(",")]
    return [(kd, pos[1:]) for kd in kinds]


def ns_of(opts: dict) -> str:
    return opts.get("namespace") or "default"
