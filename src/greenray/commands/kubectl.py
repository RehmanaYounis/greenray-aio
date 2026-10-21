"""Kubernetes emulation: API objects, controllers, scheduler, kubelet and the kubectl CLI.

Everything lives in ``state.k8s`` so scenarios can grade the end state:

  k8s.nodes[name]       labels, taints, schedulable (cordon), capacity / allocatable, ready
  k8s.pods              one record per pod (see ``new_pod``); ``component`` is set for
                        pods owned by system controllers (GPU Operator operands, NFD, ...)
  k8s.deployments       user + system Deployments (pods are named <deploy>-<rs hash>-<5>)
  k8s.daemonsets        DaemonSets (created by helm charts / the GPU Operator controller)
  k8s.clusterpolicy     the GPU Operator ClusterPolicy {name, namespace, spec, status}
  k8s.nicclusterpolicy the Network Operator NicClusterPolicy
  k8s.secrets / configmaps / runtimeclasses / jobs / serviceaccounts / events

``reconcile(state)`` plays every controller once: NFD labels, GPU Operator operands
(driver, toolkit, validator, device plugin, GFD, DCGM exporter, MIG manager), the
device-plugin resource advertisement, the scheduler and the kubelet. Every kubectl/helm
command calls it, and so do service restarts and reboots.

GPU sharing: ClusterPolicy ``devicePlugin.config.{name,default}`` + the node label
``nvidia.com/device-plugin.config`` select a key of a ConfigMap in the operator namespace;
``sharing.timeSlicing`` / ``sharing.mps`` replicate the advertised devices (see ``dp_desired_config``).
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import re
import shlex

import yaml

from ..core.errors import CommandError
from ..core.state import HEAD, AttrDict, wrap
from . import HOOKS_REBOOT, HOOKS_SERVICE, command
from . import container as ct

TICK = 15  # simulated seconds that pass per kubectl/helm command
EPOCH = datetime.datetime(2026, 10, 19, 9, 30, 0)
ALPH = "bcdfghjklmnpqrstvwxz2456789"
K8S_VERSION = "v1.29.4"
CRI = "containerd://1.7.13"
SUFFIX_GPU = "nvidia.com/"

MIG_CONFIGS = {
    "all-disabled": None,
    "all-enabled": [],
    "all-1g.10gb": ["1g.10gb"] * 7,
    "all-1g.10gb.me": ["1g.10gb+me"],
    "all-1g.20gb": ["1g.20gb"] * 4,
    "all-2g.20gb": ["2g.20gb"] * 3,
    "all-3g.40gb": ["3g.40gb"] * 2,
    "all-4g.40gb": ["4g.40gb"],
    "all-7g.80gb": ["7g.80gb"],
    "all-balanced": ["3g.40gb", "2g.20gb", "1g.10gb", "1g.10gb"],
}
MIG_SM = {"1g.10gb": 16, "1g.10gb+me": 16, "1g.20gb": 16, "2g.20gb": 32, "3g.40gb": 60, "4g.40gb": 64, "7g.80gb": 132}

# GPU Operator operands: (daemonset name, ClusterPolicy key, deploy-label suffix, uses runtimeClass, image, container)
GPU_OPERANDS = [
    (
        "nvidia-driver-daemonset",
        "driver",
        "driver",
        False,
        "nvcr.io/nvidia/driver:{drv}-ubuntu22.04",
        "nvidia-driver-ctr",
    ),
    (
        "nvidia-container-toolkit-daemonset",
        "toolkit",
        "container-toolkit",
        False,
        "nvcr.io/nvidia/k8s/container-toolkit:v1.16.1-ubuntu20.04",
        "nvidia-container-toolkit-ctr",
    ),
    (
        "nvidia-operator-validator",
        "validator",
        "operator-validator",
        True,
        "nvcr.io/nvidia/cloud-native/gpu-operator-validator:{ver}",
        "nvidia-operator-validator",
    ),
    (
        "nvidia-device-plugin-daemonset",
        "devicePlugin",
        "device-plugin",
        True,
        "nvcr.io/nvidia/k8s-device-plugin:v0.16.2-ubi8",
        "nvidia-device-plugin",
    ),
    (
        "gpu-feature-discovery",
        "gfd",
        "gpu-feature-discovery",
        True,
        "nvcr.io/nvidia/k8s-device-plugin:v0.16.2-ubi8",
        "gpu-feature-discovery",
    ),
    ("nvidia-dcgm", "dcgm", "dcgm", True, "nvcr.io/nvidia/cloud-native/dcgm:3.3.7-1-ubuntu22.04", "nvidia-dcgm-ctr"),
    (
        "nvidia-dcgm-exporter",
        "dcgmExporter",
        "dcgm-exporter",
        True,
        "nvcr.io/nvidia/k8s/dcgm-exporter:3.3.7-3.5.0-ubuntu22.04",
        "nvidia-dcgm-exporter",
    ),
    (
        "nvidia-mig-manager",
        "migManager",
        "mig-manager",
        True,
        "nvcr.io/nvidia/cloud-native/k8s-mig-manager:v0.8.0-ubuntu20.04",
        "nvidia-mig-manager",
    ),
    (
        "nvidia-node-status-exporter",
        "nodeStatusExporter",
        "node-status-exporter",
        True,
        "nvcr.io/nvidia/cloud-native/gpu-operator-validator:{ver}",
        "nvidia-node-status-exporter",
    ),
]
OPERAND_DEFAULT_OFF = {"dcgm", "nodeStatusExporter"}
OPERATOR_LABEL_PREFIXES = (
    "nvidia.com/gpu.",
    "nvidia.com/cuda.",
    "nvidia.com/gfd.",
    "nvidia.com/mig.capable",
    "nvidia.com/mig.strategy",
    "nvidia.com/mig-",
    "nvidia.com/mps.",
    "nvidia.com/vgpu.",
)
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


# Extension points for other command modules (e.g. runai.py):
#   RECONCILE_HOOKS: fn(state) run in every reconcile pass, right before the kube-scheduler.
#     Pods whose ``component`` starts with "runai" are managed by such a hook, not by kubelet_sync.
#   EXTRA_KINDS: plural -> handler with ``names`` (aliases), ``api_version``, ``kind``, ``namespaced``,
#     ``list(state, opts)``, ``table(state, items) -> (header, rows)``, ``api(state, item)``,
#     ``apply(state, doc, src) -> str`` and ``delete(state, name) -> str``.
RECONCILE_HOOKS: list = []
EXTRA_KINDS: dict = {}


def register_kind(plural, handler):
    EXTRA_KINDS[plural] = handler
    KINDS[plural] = tuple(handler.names)
    if not handler.namespaced:
        CLUSTER_SCOPED.add(plural)
    KIND_PREFIX[plural] = f"{handler.kind.lower()}.{handler.api_version.split('/')[0]}"
    SINGULAR[plural] = handler.kind.lower()


# ====================================================================== small helpers


def sfx(*parts, n=5):
    h = hashlib.md5("|".join(map(str, parts)).encode()).digest()
    return "".join(ALPH[b % len(ALPH)] for b in h[:n])


def now(state):
    return state.k8s.get("_tick", 0)


def advance(state, sec=TICK):
    state.k8s["_tick"] = now(state) + sec


def ts(sec):
    return (EPOCH + datetime.timedelta(seconds=sec)).strftime("%Y-%m-%dT%H:%M:%SZ")


def klog(state, level="I"):
    t = EPOCH + datetime.timedelta(seconds=now(state))
    return f"{level}{t:%m%d %H:%M:%S}.{(now(state) * 7919) % 1000000:06d}       1 "


def human(sec):
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


def age(state, created):
    return human(now(state) - (created if created is not None else now(state)))


def table(rows, gap=3):
    if not rows:
        return ""
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    out = []
    for r in rows:
        cells = [str(c).ljust(widths[i] + gap) if i < len(r) - 1 else str(c) for i, c in enumerate(r)]
        out.append("".join(cells).rstrip())
    return "\n".join(out) + "\n"


def truthy(v, default=True):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() not in ("false", "0", "no", "off", "")


def qty(v):
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


def is_extended(res):
    return "/" in res and not res.startswith(("kubernetes.io/", "requests.", "hugepages"))


def new_uid(state):
    k = state.k8s
    k["_uid"] = k.get("_uid", 0) + 1
    h = hashlib.md5(f"uid{k._uid}".encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


# ====================================================================== init


def k8s_init(state):
    k = state.k8s
    if k.get("_init"):
        return k
    ct.container_init(state)
    k["_init"] = True
    k.setdefault("_tick", 0)
    for key in ("daemonsets", "secrets", "jobs", "configmaps", "runtimeclasses", "services"):
        if k.get(key) is None:
            k[key] = []
    k.setdefault("nicclusterpolicy", None)
    k.setdefault("serviceaccounts", {})
    k.setdefault("_gen", {})
    k.setdefault("mig_applied", {})
    k.setdefault("ns_created", {})
    k.setdefault("_ctr_driver", {})
    for ns in k.namespaces:
        k.ns_created.setdefault(ns, -41 * 86400)
    for name, kn in k.nodes.items():
        kn.setdefault("created", -41 * 86400)
        kn.setdefault("taints", [])
        labels = kn.setdefault("labels", {})
        for key, val in (
            ("beta.kubernetes.io/arch", "amd64"),
            ("beta.kubernetes.io/os", "linux"),
            ("kubernetes.io/arch", "amd64"),
            ("kubernetes.io/hostname", name),
            ("kubernetes.io/os", "linux"),
        ):
            labels.setdefault(key, val)
        cap = kn.setdefault("capacity", {})
        cap.setdefault("ephemeral-storage", "1844640752Ki")
        cap.setdefault("hugepages-1Gi", "0")
        cap.setdefault("hugepages-2Mi", "0")
        cap.setdefault("pods", "110")
        if "allocatable" not in kn:
            kn["allocatable"] = dict(cap)
    # control plane + CNI
    for comp in ("etcd", "kube-apiserver", "kube-controller-manager", "kube-scheduler"):
        new_pod(
            state,
            "kube-system",
            f"{comp}-{HEAD}",
            _spec(comp, f"registry.k8s.io/{comp}:{K8S_VERSION}" if comp != "etcd" else "registry.k8s.io/etcd:3.5.12-0"),
            component="static",
            node=HEAD,
            created=-41 * 86400,
        )
    add_deployment(
        state,
        "kube-system",
        "coredns",
        "registry.k8s.io/coredns/coredns:v1.11.1",
        replicas=2,
        component="system",
        created=-41 * 86400,
    )
    add_daemonset(
        state,
        "kube-system",
        "kube-proxy",
        "registry.k8s.io/kube-proxy:" + K8S_VERSION,
        component="system",
        tolerations=[{"operator": "Exists"}],
        created=-41 * 86400,
    )
    add_daemonset(
        state,
        "kube-flannel",
        "kube-flannel-ds",
        "docker.io/flannel/flannel:v0.25.1",
        component="system",
        tolerations=[{"operator": "Exists"}],
        created=-41 * 86400,
    )
    reconcile(state)
    return k


def _spec(name, image, extra=None):
    spec = {"containers": [{"name": name, "image": image}]}
    spec.update(extra or {})
    return spec


# ====================================================================== object constructors


def new_pod(state, ns, name, spec, labels=None, owner=None, component=None, node=None, created=None, release=None):
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
    ns,
    name,
    image,
    component,
    node_selector=None,
    tolerations=None,
    created=None,
    release=None,
    owner="helm",
    container=None,
    runtime_class=None,
    extra=None,
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
    state, ns, name, image, replicas=1, component=None, created=None, release=None, template=None, labels=None
):
    k = state.k8s
    if template is None:
        template = {"metadata": {"labels": {"app": name}}, "spec": _spec(name, image)}
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


def ensure_namespace(state, ns):
    k = state.k8s
    if ns not in k.namespaces:
        k.namespaces.append(ns)
        k.ns_created[ns] = now(state)


def remove_release(state, release):
    """Delete everything a helm release created."""
    k = state.k8s
    k.daemonsets = [d for d in k.daemonsets if d.get("release") != release]
    k.deployments = [d for d in k.deployments if d.get("release") != release]
    k.pods = [p for p in k.pods if p.get("release") != release]


# ====================================================================== pods: requests, validation


def pod_requests(spec):
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


def validate_pod_spec(kind, name, spec, path="spec"):
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


# ====================================================================== taints, selectors


def fmt_taint(t):
    v = t.get("value")
    return f"{t['key']}={v}:{t['effect']}" if v else f"{t['key']}:{t['effect']}"


def node_taints(state, name):
    kn = state.k8s.nodes[name]
    taints = list(kn.get("taints") or [])
    if not kn.get("schedulable", True):
        taints.append(wrap({"key": "node.kubernetes.io/unschedulable", "effect": "NoSchedule", "value": ""}))
    if not kn.get("ready", True):
        taints.append(wrap({"key": "node.kubernetes.io/unreachable", "effect": "NoSchedule", "value": ""}))
        taints.append(wrap({"key": "node.kubernetes.io/unreachable", "effect": "NoExecute", "value": ""}))
    return taints


def tolerates(tols, taint):
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


def untolerated(state, name, tols, ignore_unschedulable=False):
    for t in node_taints(state, name):
        if t.get("effect") not in ("NoSchedule", "NoExecute"):
            continue
        if ignore_unschedulable and t.key == "node.kubernetes.io/unschedulable":
            continue
        if not tolerates(tols, t):
            return t
    return None


def parse_selector(sel):
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


def selector_matches(sel, labels):
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


def node_selector_ok(spec, labels):
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


# ====================================================================== events


def event(state, obj_kind, obj, etype, reason, msg, source="kubelet"):
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


def events_for(state, uid):
    return [e for e in state.k8s.events if e.get("uid") == uid]


# ====================================================================== pod status bookkeeping


def crash_count(elapsed):
    t, n, back = 0, 0, 10
    while True:
        t += back
        if t > elapsed:
            return n, t - back
        n += 1
        back = min(back * 2, 300)


def pod_restarts(state, pod):
    n = pod.get("restart_base", 0)
    if pod.get("crash_since") is not None:
        n += crash_count(now(state) - pod.crash_since)[0]
    return n


def restarts_str(state, pod):
    n = pod_restarts(state, pod)
    if not n:
        return "0"
    if pod.get("crash_since") is not None:
        _, last = crash_count(now(state) - pod.crash_since)
        when = pod.crash_since + last
    else:
        when = pod.get("last_restart") or pod.created
    return f"{n} ({human(now(state) - when)} ago)"


def set_status(state, pod, status, ready=None, logs=None, crash=False):
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


# ====================================================================== controllers


def refresh_nodes(state):
    for name, kn in state.k8s.nodes.items():
        sn = state.nodes.get(name)
        if sn is None:
            continue
        kn.ready = bool(
            sn.up
            and sn.get("power", "on") == "on"
            and sn.services.get("kubelet") == "active"
            and sn.services.get("containerd") == "active"
        )


def pods_of(state, kind, name, ns):
    return [
        p
        for p in state.k8s.pods
        if p.get("owner") and p.owner.get("kind") == kind and p.owner.get("name") == name and p.namespace == ns
    ]


def ds_eligible(state, ds):
    out = []
    for n, kn in sorted(state.k8s.nodes.items()):
        if not selector_matches([(kk, "=", [v]) for kk, v in ds.node_selector.items()], kn.labels):
            continue
        if untolerated(state, n, list(ds.tolerations) + DS_AUTO_TOLERATIONS):
            continue
        out.append(n)
    return out


def sync_daemonsets(state):
    k = state.k8s
    names = {(d.namespace, d.name) for d in k.daemonsets}
    k.pods = [
        p
        for p in k.pods
        if not (p.get("owner") and p.owner.get("kind") == "DaemonSet" and (p.namespace, p.owner.name) not in names)
    ]
    for ds in k.daemonsets:
        eligible = ds_eligible(state, ds)
        pods = pods_of(state, "DaemonSet", ds.name, ds.namespace)
        for p in pods:
            if p.node not in eligible:
                k.pods.remove(p)
        have = {p.node for p in pods_of(state, "DaemonSet", ds.name, ds.namespace)}
        for n in eligible:
            if n in have:
                continue
            gen = k._gen.get(f"{ds.namespace}/{ds.name}/{n}", 0)
            extra = {"runtimeClassName": ds.runtime_class} if ds.get("runtime_class") else {}
            extra["tolerations"] = list(ds.tolerations)
            new_pod(
                state,
                ds.namespace,
                f"{ds.name}-{sfx(ds.namespace, ds.name, n, gen)}",
                _spec(ds.container, ds.image, extra),
                labels={"app": ds.name},
                owner={"kind": "DaemonSet", "name": ds.name},
                component=ds.component,
                node=n,
                created=ds.created if ds.created < 0 else None,
                release=ds.get("release"),
            )


def template_hash(template):
    return sfx(json.dumps(template, sort_keys=True, default=str), n=10)


def sync_deployments(state):
    k = state.k8s
    for d in k.deployments:
        h = template_hash(d.template)
        rs = f"{d.name}-{h}"
        for p in list(k.pods):
            o = p.get("owner")
            if (
                o
                and o.get("kind") == "ReplicaSet"
                and o.get("deployment") == d.name
                and p.namespace == d.namespace
                and o.name != rs
            ):
                k.pods.remove(p)
        pods = pods_of(state, "ReplicaSet", rs, d.namespace)
        want = int(d.replicas)
        i = 0
        while len(pods) < want:
            gen = k._gen.get(f"{d.namespace}/{rs}", 0)
            name = f"{rs}-{sfx(rs, i, gen, len(pods))}"
            i += 1
            if any(p.name == name and p.namespace == d.namespace for p in k.pods):
                continue
            labels = dict(d.template.get("metadata", {}).get("labels", {}))
            labels["pod-template-hash"] = h
            node = HEAD if d.get("component") else None
            new_pod(
                state,
                d.namespace,
                name,
                d.template.get("spec", {}),
                labels=labels,
                owner={"kind": "ReplicaSet", "name": rs, "deployment": d.name},
                component=d.get("component"),
                node=node,
                created=d.created if d.created < 0 else None,
                release=d.get("release"),
            )
            pods = pods_of(state, "ReplicaSet", rs, d.namespace)
        if len(pods) > want:
            pods.sort(key=lambda p: (p.node is not None, -p.created))
            for p in pods[: len(pods) - want]:
                k.pods.remove(p)


def sync_jobs(state):
    k = state.k8s
    for j in k.jobs:
        pods = pods_of(state, "Job", j.name, j.namespace)
        want = int(j.get("parallelism", 1) or 1)
        done = sum(1 for p in pods if p.status == "Completed")
        completions = int(j.get("completions", 1) or 1)
        while len(pods) < want and done < completions:
            name = f"{j.name}-{sfx(j.namespace, j.name, len(pods))}"
            labels = dict(j.template.get("metadata", {}).get("labels", {}))
            labels.update({"job-name": j.name, "batch.kubernetes.io/job-name": j.name})
            new_pod(
                state,
                j.namespace,
                name,
                j.template.get("spec", {}),
                labels=labels,
                owner={"kind": "Job", "name": j.name},
            )
            pods = pods_of(state, "Job", j.name, j.namespace)


# ---------------------------------------------------------------- NFD


def nfd_sync(state):
    k = state.k8s
    for n, kn in k.nodes.items():
        if not any(p.node == n and p.component == "nfd-worker" and p.status == "Running" for p in k.pods):
            continue
        sn = state.nodes[n]
        lab = kn.labels
        lab["feature.node.kubernetes.io/kernel-version.full"] = sn.kernel
        lab["feature.node.kubernetes.io/system-os_release.ID"] = "ubuntu"
        lab["feature.node.kubernetes.io/system-os_release.VERSION_ID"] = "22.04"
        lab["feature.node.kubernetes.io/cpu-model.vendor_id"] = "Intel"
        if sn.gpus:
            lab["feature.node.kubernetes.io/pci-10de.present"] = "true"
        if sn.ib:
            lab["feature.node.kubernetes.io/pci-15b3.present"] = "true"
            lab["feature.node.kubernetes.io/rdma.available"] = "true"


# ---------------------------------------------------------------- GPU operator


def cp_spec(state):
    cp = state.k8s.get("clusterpolicy")
    return cp.spec if cp else None


def operand_enabled(spec, key):
    sect = spec.get(key) or {}
    return truthy(sect.get("enabled"), key not in OPERAND_DEFAULT_OFF)


def gpu_operator_sync(state):
    k = state.k8s
    cp = k.get("clusterpolicy")
    if not cp:
        k.daemonsets = [d for d in k.daemonsets if d.get("owner") != "clusterpolicy"]
        return
    spec = cp.spec
    ns = cp.namespace
    operator_up = any(p.component == "gpu-operator" and p.status == "Running" for p in k.pods)
    if not operator_up:
        return
    for _n, kn in k.nodes.items():
        lab = kn.labels
        if lab.get("feature.node.kubernetes.io/pci-10de.present") != "true":
            continue
        lab.setdefault("nvidia.com/gpu.present", "true")
        for _, key, suffix, *_ in GPU_OPERANDS:
            if key == "migManager":
                if lab.get("nvidia.com/mig.capable") == "true":
                    lab.setdefault("nvidia.com/gpu.deploy.mig-manager", "true")
                continue
            lab.setdefault(f"nvidia.com/gpu.deploy.{suffix}", "true")
        if operand_enabled(spec, "migManager") and lab.get("nvidia.com/mig.capable") == "true":
            default = ((spec.get("migManager") or {}).get("config") or {}).get("default", "all-disabled")
            if default:
                lab.setdefault("nvidia.com/mig.config", default)
    ver = cp.get("version", "v24.6.2")
    drv = (spec.get("driver") or {}).get("version", "550.90.07")
    tols = list((spec.get("daemonsets") or {}).get("tolerations") or [])
    want = set()
    for name, key, suffix, uses_rc, image, cname in GPU_OPERANDS:
        if key != "validator" and not operand_enabled(spec, key):
            continue
        want.add(name)
        add_daemonset(
            state,
            ns,
            name,
            image.format(drv=drv, ver=ver),
            component=key,
            node_selector={f"nvidia.com/gpu.deploy.{suffix}": "true"},
            tolerations=tols,
            created=cp.created,
            release=cp.get("release"),
            owner="clusterpolicy",
            container=cname,
            runtime_class="nvidia" if uses_rc else None,
        )
    k.daemonsets = [d for d in k.daemonsets if d.get("owner") != "clusterpolicy" or d.name in want]
    # a changed devicePlugin.config makes the operator roll the device-plugin and GFD pods
    sig = json.dumps((spec.get("devicePlugin") or {}).get("config") or {}, sort_keys=True, default=str)
    if k.get("_dp_cfg_sig") is not None and k._dp_cfg_sig != sig:
        for p in [p for p in k.pods if p.component in ("devicePlugin", "gfd")]:
            _remove_pod(state, p)
    k["_dp_cfg_sig"] = sig
    if not any(rc.name == "nvidia" for rc in k.runtimeclasses):
        k.runtimeclasses.append(
            {"name": "nvidia", "handler": "nvidia", "created": cp.created, "release": cp.get("release")}
        )


def strip_operator_labels(state):
    for kn in state.k8s.nodes.values():
        for key in list(kn.labels):
            if key.startswith(OPERATOR_LABEL_PREFIXES) or key.startswith("nvidia.com/mig.config.state"):
                del kn.labels[key]


def _host_drv(state, n):
    return state.nodes[n].driver_loaded and not state.k8s._ctr_driver.get(n)


def node_gpu_conditions(state, n):
    """-> (driver_ok, runtime_ok, sandbox_ok) for operator-managed node n."""
    spec = cp_spec(state)
    node = state.nodes[n]
    k = state.k8s
    driver_on = operand_enabled(spec, "driver")
    toolkit_on = operand_enabled(spec, "toolkit")
    if driver_on:
        drv_ok = not _host_drv(state, n)
        if drv_ok and not k._ctr_driver.get(n):
            k._ctr_driver[n] = True
            node.driver_loaded = True
    else:
        if k._ctr_driver.get(n):  # containerized driver removed -> modules unloaded
            k._ctr_driver[n] = False
            node.driver_loaded = False
        drv_ok = bool(node.driver_loaded)
    has_handler = ct.containerd_has(state, n, "nvidia")
    rt_ok = drv_ok if toolkit_on else has_handler
    sandbox_ok = toolkit_on or has_handler
    return drv_ok, rt_ok, sandbox_ok


def gpu_operand_status(state, pod):
    n, comp = pod.node, pod.component
    spec = cp_spec(state)
    if spec is None:
        return
    drv_ok, rt_ok, sandbox_ok = node_gpu_conditions(state, n)
    sn = state.nodes[n]
    if comp == "driver":
        if drv_ok:
            set_status(state, pod, "Running", logs=driver_ok_log(state, n))
        else:
            set_status(state, pod, "CrashLoopBackOff", crash=True, logs=driver_crash_log(state, n))
            event(
                state,
                "Pod",
                pod,
                "Warning",
                "BackOff",
                f"Back-off restarting failed container nvidia-driver-ctr in pod {pod.name}_{pod.namespace}({pod.uid})",
            )
        return
    if comp == "toolkit":
        if not drv_ok:
            set_status(state, pod, "Init:0/1", logs=[])
            return
        node = state.nodes[n]
        loaded = node.runtime_loaded.containerd
        if "nvidia" not in loaded.handlers or loaded.default != "nvidia":
            node.files[ct.CONTAINERD_CONF] = ct.containerd_config(nvidia=True, default="nvidia")
            ct.reload_runtime(state, n, "containerd")
        set_status(state, pod, "Running", logs=toolkit_log(state))
        return
    if pod.spec.get("runtimeClassName") and not sandbox_ok:
        set_status(state, pod, "ContainerCreating")
        event(
            state,
            "Pod",
            pod,
            "Warning",
            "FailedCreatePodSandBox",
            "Failed to create pod sandbox: rpc error: code = Unknown desc = failed to get sandbox runtime: "
            'no runtime for "nvidia" is configured',
        )
        return
    if not drv_ok or not rt_ok:
        if comp == "validator":
            set_status(
                state,
                pod,
                "Init:0/4" if not drv_ok else "Init:1/4",
                logs=[
                    f'Error from server (BadRequest): container "nvidia-operator-validator" in pod "{pod.name}" is waiting to start: PodInitializing'
                ],
            )
        else:
            set_status(
                state,
                pod,
                "Init:0/1",
                logs=[
                    f'Error from server (BadRequest): container "{pod.spec.containers[0].name}" in pod "{pod.name}" is waiting to start: PodInitializing'
                ],
            )
        return
    strategy = (spec.get("mig") or {}).get("strategy", "single")
    if comp == "devicePlugin":
        info = dp_refresh_config(state, pod)
        if info["err"] == "missing-cm":
            set_status(state, pod, "ContainerCreating", logs=[])
            pod.logs = [
                f'Error from server (BadRequest): container "nvidia-device-plugin" in pod "{pod.name}" is waiting '
                "to start: ContainerCreating"
            ]
            event(
                state,
                "Pod",
                pod,
                "Warning",
                "FailedMount",
                f'MountVolume.SetUp failed for volume "{info["source"][0]}" : configmap "{info["source"][0]}" not found',
            )
            return
        if info["err"]:
            set_status(state, pod, "Init:CrashLoopBackOff", crash=True)
            pod.status = "Init:CrashLoopBackOff"
            pod.phase = "Pending"
            pod.logs = [
                'Defaulted container "nvidia-device-plugin" out of: nvidia-device-plugin, config-manager, '
                "toolkit-validation (init), config-manager-init (init)",
                f'Error from server (BadRequest): container "nvidia-device-plugin" in pod "{pod.name}" is waiting '
                "to start: PodInitializing",
                f"(config-manager-init) {klog(state, 'E')}main.go:178] Error: {dp_config_error_text(state, pod, info)}",
            ]
            event(
                state,
                "Pod",
                pod,
                "Warning",
                "BackOff",
                f"Back-off restarting failed container config-manager-init in pod {pod.name}_{pod.namespace}({pod.uid})",
            )
            return
        inv, err = gpu_inventory(state, n, strategy)
        if err:
            set_status(state, pod, "CrashLoopBackOff", crash=True, logs=device_plugin_log(state, n, strategy, inv, err))
            event(
                state,
                "Pod",
                pod,
                "Warning",
                "BackOff",
                f"Back-off restarting failed container nvidia-device-plugin in pod {pod.name}_{pod.namespace}({pod.uid})",
            )
        else:
            cfg = info.get("cfg")
            set_status(
                state, pod, "Running", logs=device_plugin_log(state, n, strategy, apply_sharing(inv, cfg), None, cfg)
            )
        return
    logs = {
        "validator": ["all validations are successful"],
        "gfd": [
            f"{klog(state)}main.go:126] Starting OS watcher.",
            f"{klog(state)}main.go:131] Loading configuration.",
            f"{klog(state)}main.go:243] Labels: ...",
            f"{klog(state)}main.go:262] Sleeping for 60000000000",
        ],
        "dcgmExporter": [
            'time="2026-10-19T09:31:02Z" level=info msg="Starting dcgm-exporter"',
            'time="2026-10-19T09:31:02Z" level=info msg="DCGM successfully initialized!"',
            'time="2026-10-19T09:31:03Z" level=info msg="Collecting DCP Metrics"',
            'time="2026-10-19T09:31:03Z" level=info msg="Starting webserver"',
            'time="2026-10-19T09:31:03Z" level=info msg="Listening on" address=":9400"',
        ],
        "migManager": pod.get("logs") or ["Waiting for change to 'nvidia.com/mig.config' label"],
    }.get(comp, ["running"])
    set_status(state, pod, "Running", logs=logs)
    del sn


def cuda_validator_sync(state):
    k = state.k8s
    cp = k.get("clusterpolicy")
    ns = cp.namespace if cp else None
    have = {p.node: p for p in k.pods if p.component == "cuda-validator"}
    for p in list(k.pods):
        if p.component == "validator" and p.status == "Running" and p.node not in have and cp:
            v = new_pod(
                state,
                ns,
                f"nvidia-cuda-validator-{sfx('cuda', p.node, p.uid)}",
                _spec(
                    "nvidia-cuda-validator",
                    f"nvcr.io/nvidia/cloud-native/gpu-operator-validator:{cp.get('version', 'v24.6.2')}",
                    {"runtimeClassName": "nvidia", "restartPolicy": "OnFailure"},
                ),
                component="cuda-validator",
                node=p.node,
                release=cp.get("release"),
            )
            set_status(state, v, "Completed", ready="0/1", logs=["cuda workload validation is successful"])
    if not cp:
        k.pods = [p for p in k.pods if p.component != "cuda-validator"]


# ---------------------------------------------------------------- device inventory / resources


def mig_devices(g):
    return [(g.index, gi.id, gi.profile) for gi in g.gpu_instances if any(c.gi == gi.id for c in g.compute_instances)]


def gpu_inventory(state, n, strategy):
    """-> ({resource: [(gpu_index, gi_id|None), ...]}, error|None)"""
    node = state.nodes[n]
    gpus = [g for g in node.gpus if not g.fallen_off_bus]
    inv = {"nvidia.com/gpu": []}
    if strategy in ("none", None) or not any(g.mig_mode for g in gpus):
        inv["nvidia.com/gpu"] = [(g.index, None) for g in gpus]
        return inv, None
    if strategy == "single":
        devs = [d for g in gpus for d in mig_devices(g)]
        profiles = {d[2] for d in devs}
        bad = [g for g in gpus if not g.mig_mode or not mig_devices(g)]
        if bad or len(profiles) != 1:
            return inv, (
                f"invalid MIG configuration: At least one device with migEnabled=true was not configured "
                f"correctly: error visiting device: device {bad[0].index if bad else 0} has an invalid MIG configuration"
            )
        inv["nvidia.com/gpu"] = [(d[0], d[1]) for d in devs]
        return inv, None
    for g in gpus:
        if not g.mig_mode:
            inv["nvidia.com/gpu"].append((g.index, None))
            continue
        for idx, gi, prof in mig_devices(g):
            inv.setdefault(f"nvidia.com/mig-{prof}", []).append((idx, gi))
    return inv, None


def device_plugin_sync(state):
    k = state.k8s
    for n, kn in k.nodes.items():
        sn = state.nodes.get(n)
        if sn is None or not sn.gpus:
            continue
        dp = next(
            (
                p
                for p in k.pods
                if p.node == n and p.status == "Running" and p.component in ("devicePlugin", "standalone-device-plugin")
            ),
            None,
        )
        current = {r for r in kn.capacity if r.startswith(SUFFIX_GPU)}
        if dp is None:
            for r in current:
                kn.capacity[r] = "0"
                kn.allocatable[r] = "0"
            continue
        if dp.component == "devicePlugin":
            strategy = (cp_spec(state).get("mig") or {}).get("strategy", "single")
        else:
            strategy = next((d.get("mig_strategy", "none") for d in k.daemonsets if d.name == dp.owner.name), "none")
        inv, err = gpu_inventory(state, n, strategy)
        if dp.component == "devicePlugin":
            inv = apply_sharing(inv, (dp.get("dp_cfg") or {}).get("cfg"))
        healthy = {g.index for g in sn.gpus if g.healthy and not g.reset_required and not g.fallen_off_bus}
        for r in current | set(inv):
            devs = inv.get(r, [])
            kn.capacity[r] = str(len(devs))
            kn.allocatable[r] = str(sum(1 for d in devs if d[0] in healthy))


# ---------------------------------------------------------------- GPU sharing (time-slicing / MPS)

DP_CONFIG_LABEL = "nvidia.com/device-plugin.config"


def _parse_sharing(text):
    """Parse one device-plugin config file. -> (cfg | None, error | None).

    cfg = {"strategy": "time-slicing"|"mps", "resources": [{"name", "replicas", "rename"}],
           "failRequestsGreaterThanOne": bool}
    """
    try:
        doc = yaml.safe_load(text or "") or {}
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        return None, (
            f"unable to parse config file: error converting YAML to JSON: yaml: line {mark.line + 1 if mark else 1}: "
            f"{getattr(e, 'problem', 'could not parse')}"
        )
    if not isinstance(doc, dict):
        return None, "unable to parse config file: json: cannot unmarshal string into Go value of type v1.Config"
    if doc.get("version") not in ("v1", None):
        return None, f"unable to parse config file: unknown version: {doc.get('version')}"
    sharing = doc.get("sharing") or {}
    if not isinstance(sharing, dict) or not sharing:
        return None, None
    strategy = "mps" if sharing.get("mps") else ("time-slicing" if sharing.get("timeSlicing") else None)
    if strategy is None:
        return None, None
    block = sharing.get("mps" if strategy == "mps" else "timeSlicing") or {}
    rename_all = truthy(block.get("renameByDefault"), False)
    out = []
    for r in block.get("resources") or []:
        if not isinstance(r, dict) or not r.get("name"):
            return None, "unable to finalize config: invalid sharing config: resource name must be set"
        try:
            reps = int(r.get("replicas"))
        except (TypeError, ValueError):
            return (
                None,
                f"unable to parse config file: invalid replicas value for {r.get('name')}: {r.get('replicas')!r}",
            )
        if reps < 1 or (strategy == "mps" and reps < 2):
            return None, (
                f"unable to finalize config: invalid {strategy} config: number of replicas must be "
                f"{'greater than 1' if strategy == 'mps' else 'at least 1'}, got {reps}"
            )
        name = str(r["name"])
        rename = r.get("rename") or (name + ".shared" if rename_all else None)
        out.append({"name": name, "replicas": reps, "rename": rename})
    if not out:
        return None, None
    return {
        "strategy": strategy,
        "resources": out,
        "failRequestsGreaterThanOne": truthy(block.get("failRequestsGreaterThanOne"), False),
    }, None


def dp_desired_config(state, n):
    """What the device plugin's config-manager on node n would load right now.

    -> {"source": [cm, key], "cfg": cfg|None, "err": None|"missing-cm"|"missing-key"|parse error}
    """
    k = state.k8s
    cp = k.get("clusterpolicy")
    conf = ((cp.spec.get("devicePlugin") or {}).get("config") or {}) if cp else {}
    name = conf.get("name") or ""
    if not name:
        return {"source": ["", ""], "cfg": None, "err": None}
    label = k.nodes[n].labels.get(DP_CONFIG_LABEL)
    key = label or conf.get("default") or ""
    out = {"source": [name, key], "cfg": None, "err": None}
    cm = next((c for c in k.configmaps if c.namespace == cp.namespace and c.name == name), None)
    if cm is None:
        out["err"] = "missing-cm"
        return out
    if not key:
        return out  # no label and no default: the plugin runs with its built-in (unshared) config
    data = cm.get("data") or {}
    if key not in data:
        out["err"] = "missing-key"
        return out
    out["cfg"], out["err"] = _parse_sharing(data[key])
    return out


def dp_running_config(state, n):
    """Sharing config the running operator device plugin on n actually uses (None = no sharing)."""
    dp = next(
        (p for p in state.k8s.pods if p.node == n and p.component == "devicePlugin" and p.status == "Running"), None
    )
    if dp is None or not dp.get("dp_cfg"):
        return None
    return dp.dp_cfg.get("cfg")


def dp_refresh_config(state, pod):
    """Config-manager behaviour: re-read the config when the selected (ConfigMap, key) changes,
    when the pod (re)starts, or while the previous read failed. Editing the data of the same
    key in place is NOT picked up until the device-plugin pods are restarted."""
    want = dp_desired_config(state, pod.node)
    cur = pod.get("dp_cfg")
    if cur is None or list(cur.get("source") or []) != list(want["source"]) or cur.get("err"):
        pod["dp_cfg"] = want
    return pod.dp_cfg


def apply_sharing(inv, cfg):
    """Replicate devices for shared resources: {res: [(gpu, gi, replica), ...]} (replica-major order)."""
    if not cfg:
        return inv
    out = {}
    for r, devs in inv.items():
        rule = next((x for x in cfg["resources"] if x["name"] == r), None)
        if rule is None:
            out.setdefault(r, []).extend(devs)
            continue
        name = rule.get("rename") or r
        if name != r:
            out.setdefault(r, [])
        out.setdefault(name, []).extend((d[0], d[1], i) for i in range(int(rule["replicas"])) for d in devs)
    return out


def dp_config_error_text(state, pod, info):
    cp = state.k8s.clusterpolicy
    name, key = info["source"]
    if info["err"] == "missing-key":
        return f'unable to get config: specified config "{key}" does not exist in ConfigMap {cp.namespace}/{name}'
    return f"unable to load config: {info['err']}"


def rdma_sync(state):
    k = state.k8s
    ncp = k.get("nicclusterpolicy")
    resources = []
    if ncp:
        rsdp = ncp.spec.get("rdmaSharedDevicePlugin") or {}
        cfg = rsdp.get("config")
        try:
            cfgd = json.loads(cfg) if isinstance(cfg, str) else (cfg or {})
        except ValueError:
            cfgd = {}
        for r in cfgd.get("configList", []) or []:
            resources.append(
                (
                    r.get("resourcePrefix", "rdma") + "/" + r.get("resourceName", "rdma_shared_device_a"),
                    int(r.get("rdmaHcaMax", 63)),
                )
            )
    for n, kn in k.nodes.items():
        running = any(p.node == n and p.component == "rdma-shared-dp" and p.status == "Running" for p in k.pods)
        for r in [r for r in kn.capacity if r.startswith("rdma/")]:
            if not running or r not in dict(resources):
                kn.capacity[r] = "0"
                kn.allocatable[r] = "0"
        if running:
            for r, cnt in resources:
                kn.capacity[r] = str(cnt)
                kn.allocatable[r] = str(cnt)


# ---------------------------------------------------------------- GFD labels


def gfd_sync(state):
    k = state.k8s
    spec = cp_spec(state)
    if spec is None:
        return
    strategy = (spec.get("mig") or {}).get("strategy", "single")
    for n, kn in k.nodes.items():
        if not any(p.node == n and p.component == "gfd" and p.status == "Running" for p in k.pods):
            continue
        sn = state.nodes[n]
        lab = kn.labels
        maj, mnr, rev = (sn.driver_version.split(".") + ["0", "0"])[:3]
        cmaj, cmin = (sn.cuda_version.split(".") + ["0"])[:2]
        g0 = sn.gpus[0]
        product = g0.name.replace("NVIDIA ", "NVIDIA-").replace(" ", "-")
        lab.update(
            {
                "nvidia.com/cuda.driver.major": maj,
                "nvidia.com/cuda.driver.minor": mnr,
                "nvidia.com/cuda.driver.rev": rev,
                "nvidia.com/cuda.driver-version.full": sn.driver_version,
                "nvidia.com/cuda.runtime.major": cmaj,
                "nvidia.com/cuda.runtime.minor": cmin,
                "nvidia.com/cuda.runtime-version.full": sn.cuda_version,
                "nvidia.com/gfd.timestamp": str(1792399800 + now(state)),
                "nvidia.com/gpu.compute.major": "9",
                "nvidia.com/gpu.compute.minor": "0",
                "nvidia.com/gpu.family": "hopper",
                "nvidia.com/gpu.machine": sn.model.replace(" ", "-"),
                "nvidia.com/gpu.memory": str(g0.memory_total_mib),
                "nvidia.com/gpu.product": product,
                "nvidia.com/gpu.replicas": "1",
                "nvidia.com/gpu.sharing-strategy": "none",
                "nvidia.com/gpu.count": str(len(sn.gpus)),
                "nvidia.com/mig.capable": "true" if g0.mig_capable else "false",
                "nvidia.com/mig.strategy": strategy,
                "nvidia.com/mps.capable": "false",
            }
        )
        for key in [kk for kk in lab if kk.startswith("nvidia.com/mig-")]:
            del lab[key]
        devs = [d for g in sn.gpus for d in mig_devices(g)]
        if any(g.mig_mode for g in sn.gpus) and devs:
            if strategy == "single":
                prof = devs[0][2]
                from .nvidia_smi import PROFILE_MEM_MIB

                lab["nvidia.com/gpu.product"] = f"{product}-MIG-{prof}"
                lab["nvidia.com/gpu.count"] = str(len(devs))
                lab["nvidia.com/gpu.memory"] = str(PROFILE_MEM_MIB.get(prof, 0))
                lab["nvidia.com/gpu.multiprocessors"] = str(MIG_SM.get(prof, 0))
            elif strategy == "mixed":
                from .nvidia_smi import MIG_PROFILES, PROFILE_MEM_MIB

                for prof in sorted({d[2] for d in devs}):
                    cnt = sum(1 for d in devs if d[2] == prof)
                    base = f"nvidia.com/mig-{prof}"
                    lab[f"{base}.count"] = str(cnt)
                    lab[f"{base}.memory"] = str(PROFILE_MEM_MIB.get(prof, 0))
                    lab[f"{base}.multiprocessors"] = str(MIG_SM.get(prof, 0))
                    lab[f"{base}.slices.gi"] = str(MIG_PROFILES[prof][1])
                    lab[f"{base}.slices.ci"] = str(MIG_PROFILES[prof][1])
                    lab[f"{base}.engines.copy"] = str(MIG_PROFILES[prof][1])
        else:
            lab.pop("nvidia.com/gpu.multiprocessors", None)
        sharing = dp_running_config(state, n)
        rule = next((r for r in (sharing or {}).get("resources", []) if r["name"] == "nvidia.com/gpu"), None)
        if rule:
            lab["nvidia.com/gpu.replicas"] = str(rule["replicas"])
            lab["nvidia.com/gpu.sharing-strategy"] = sharing["strategy"]
            if not rule.get("rename"):
                lab["nvidia.com/gpu.product"] += "-SHARED"
        for r in (sharing or {}).get("resources", []):
            if r["name"].startswith("nvidia.com/mig-") and f"{r['name']}.count" in lab:
                lab[f"{r['name']}.replicas"] = str(r["replicas"])
                lab[f"{r['name']}.sharing-strategy"] = sharing["strategy"]


# ---------------------------------------------------------------- MIG manager


def apply_mig_layout(node, layout):
    from .nvidia_smi import MIG_PROFILES

    for g in node.gpus:
        g.compute_instances = []
        g.gpu_instances = []
        g.mig_pending = None
        if layout is None:
            g.mig_mode = False
            continue
        g.mig_mode = True
        placement = 0
        for i, prof in enumerate(layout, start=1):
            pid, cs = MIG_PROFILES[prof][0], MIG_PROFILES[prof][1]
            g.gpu_instances.append({"id": i, "profile": prof, "profile_id": pid, "placement": placement})
            g.compute_instances.append({"gi": i, "id": 0, "profile": prof})
            placement += cs


def mig_manager_sync(state):
    k = state.k8s
    for p in k.pods:
        if p.component != "migManager" or p.status != "Running":
            continue
        n = p.node
        kn, sn = k.nodes[n], state.nodes[n]
        want = kn.labels.get("nvidia.com/mig.config")
        if not want or k.mig_applied.get(n) == want:
            continue
        t = f'time="{ts(now(state))}"'
        log = [
            f"Updating to MIG config: {want}",
            "Getting current value of the 'nvidia.com/gpu.deploy.device-plugin' node label",
            "Shutting down all GPU clients in Kubernetes by disabling their component-specific nodeSelector labels",
            "Shutting down all GPU clients on the host by stopping their systemd services",
            "Stopping nvsm.service (active, will-restart)",
            "Stopping nvidia-dcgm.service (active, will-restart)",
            "Applying the selected MIG config to the node",
            f'{t} level=info msg="Parsing config file..."',
            f'{t} level=info msg="Selecting specific MIG config..."',
        ]
        if want not in MIG_CONFIGS:
            kn.labels["nvidia.com/mig.config.state"] = "failed"
            k.mig_applied[n] = want
            p.logs = log + [
                f'{t} level=error msg="Error parsing config: config \\"{want}\\" not found in /mig-parted-config/config.yaml"',
                "Changing the 'nvidia.com/mig.config.state' node label to 'failed'",
            ]
            event(
                state,
                "Node",
                {"name": n, "uid": "node-" + n, "namespace": ""},
                "Warning",
                "MigConfigFailed",
                f"Failed to apply MIG config {want}",
                source="nvidia-mig-manager",
            )
            continue
        busy = [g.index for g in sn.gpus if g.processes]
        if busy:
            kn.labels["nvidia.com/mig.config.state"] = "failed"
            k.mig_applied[n] = want
            p.logs = log + [
                f'{t} level=error msg="Error applying MIG configuration: error enabling MIG mode on GPU {busy[0]}: In use by another client"',
                "Restarting all GPU clients previously shutdown on the host by restarting their systemd services",
                "Changing the 'nvidia.com/mig.config.state' node label to 'failed'",
            ]
            continue
        apply_mig_layout(sn, MIG_CONFIGS[want])
        kn.labels["nvidia.com/mig.config.state"] = "success"
        k.mig_applied[n] = want
        p.logs = log + [
            f'{t} level=info msg="Asserting MIG mode: {"Disabled" if MIG_CONFIGS[want] is None else "Enabled"}"',
            f'{t} level=info msg="MIG configuration applied successfully"',
            "Restarting validator pod to re-run all validations",
            "Restarting all GPU clients previously shutdown in Kubernetes by reenabling their component-specific nodeSelector labels",
            "Restarting all GPU clients previously shutdown on the host by restarting their systemd services",
            "Changing the 'nvidia.com/mig.config.state' node label to 'success'",
        ]


# ---------------------------------------------------------------- other system components


def system_status(state, pod):
    comp = pod.component
    if comp in (
        "static",
        "system",
        "gpu-operator",
        "nfd-master",
        "nfd-worker",
        "nfd-gc",
        "network-operator",
        "generic",
        "multus",
        "whereabouts",
        "nv-ipam",
    ):
        set_status(state, pod, "Running", logs=pod.get("logs") or [f"{klog(state)}main.go:52] {pod.name} started"])
        if comp == "gpu-operator":
            pod.logs = operator_log(state)
        return
    if comp == "standalone-device-plugin":
        n = pod.node
        sn = state.nodes[n]
        rt = sn.runtime_loaded.containerd
        runtime = pod.spec.get("runtimeClassName") or rt.default
        if not sn.gpus:
            set_status(state, pod, "Running", logs=device_plugin_nonvml_log(state, crash=False))
        elif pod.spec.get("runtimeClassName") and runtime not in rt.handlers:
            set_status(state, pod, "ContainerCreating")
            event(
                state,
                "Pod",
                pod,
                "Warning",
                "FailedCreatePodSandBox",
                f'Failed to create pod sandbox: rpc error: code = Unknown desc = failed to get sandbox runtime: no runtime for "{runtime}" is configured',
            )
        elif runtime != "nvidia" or "nvidia" not in rt.handlers or not sn.driver_loaded:
            set_status(state, pod, "CrashLoopBackOff", crash=True, logs=device_plugin_nonvml_log(state, crash=True))
            event(
                state,
                "Pod",
                pod,
                "Warning",
                "BackOff",
                f"Back-off restarting failed container nvidia-device-plugin-ctr in pod {pod.name}_{pod.namespace}({pod.uid})",
            )
        else:
            strategy = next(
                (d.get("mig_strategy", "none") for d in state.k8s.daemonsets if d.name == pod.owner.name), "none"
            )
            inv, err = gpu_inventory(state, n, strategy)
            if err:
                set_status(
                    state, pod, "CrashLoopBackOff", crash=True, logs=device_plugin_log(state, n, strategy, inv, err)
                )
            else:
                set_status(state, pod, "Running", logs=device_plugin_log(state, n, strategy, inv, None))
        return
    if comp == "rdma-shared-dp":
        ncp = state.k8s.get("nicclusterpolicy")
        set_status(
            state,
            pod,
            "Running",
            logs=[
                "2026/10/19 09:31:12 Starting K8s RDMA Shared Device Plugin version= master",
                "2026/10/19 09:31:12 resource manager reading configs",
                "2026/10/19 09:31:12 Reading /k8s-rdma-shared-dev-plugin/config.json",
                "2026/10/19 09:31:12 loaded config: "
                + str((ncp.spec.get("rdmaSharedDevicePlugin") or {}).get("config", "") if ncp else ""),
                "2026/10/19 09:31:12 Discovering host devices",
                "2026/10/19 09:31:12 discovering host network devices",
                "2026/10/19 09:31:12 Initializing resource servers",
                "2026/10/19 09:31:12 Starting all servers...",
                "2026/10/19 09:31:12 all servers started.",
                "2026/10/19 09:31:12 Listening for term signals",
            ],
        )
        return
    if comp == "mofed":
        sn = state.nodes[pod.node]
        if sn.services.get("openibd") == "active":
            set_status(
                state,
                pod,
                "CrashLoopBackOff",
                crash=True,
                logs=[
                    "[INFO] Starting MOFED driver container",
                    "[INFO] Unloading host inbox/OFED drivers",
                    "rmmod: ERROR: Module mlx5_core is in use by: mlx5_ib nvidia_peermem",
                    "[ERROR] Failed to unload host RDMA modules: a host DOCA/MLNX_OFED stack (openibd) owns the ConnectX devices",
                    "[ERROR] driver container exiting with error",
                ],
            )
        else:
            set_status(state, pod, "Running", logs=["[INFO] MOFED driver loaded"])
        return
    set_status(state, pod, "Running")


# ---------------------------------------------------------------- scheduler + kubelet


def node_used(state, n, exclude=None):
    used = {}
    for p in state.k8s.pods:
        if p.node != n or p is exclude or p.phase in ("Succeeded", "Failed"):
            continue
        for r, v in (p.get("requests") or {}).items():
            used[r] = used.get(r, 0) + v
    return used


def free_of(state, n, res):
    kn = state.k8s.nodes[n]
    return qty(kn.allocatable.get(res, 0)) - node_used(state, n).get(res, 0)


def pod_tolerations(pod):
    return list(pod.spec.get("tolerations") or []) + DEFAULT_POD_TOLERATIONS


def schedule(state):
    k = state.k8s
    nodes = sorted(k.nodes)
    for pod in sorted([p for p in k.pods if p.node is None and not p.get("component")], key=lambda p: p.created):
        if pod.phase != "Pending":
            continue
        spec = pod.spec
        if spec.get("nodeName") and spec.nodeName in k.nodes:
            bind(state, pod, spec.nodeName)
            continue
        reasons, fits, preempt = {}, [], {"np": 0, "victims": 0}
        tols = pod_tolerations(pod)
        for n in nodes:
            kn = k.nodes[n]
            why = []
            if not kn.get("schedulable", True) and not tolerates(
                tols, {"key": "node.kubernetes.io/unschedulable", "effect": "NoSchedule"}
            ):
                why = ["node(s) were unschedulable"]
            else:
                t = untolerated(state, n, tols, ignore_unschedulable=True)
                if t is not None:
                    why = [f"node(s) had untolerated taint {{{t.key}: {t.get('value') or ''}}}"]
                elif not node_selector_ok(spec, kn.labels):
                    why = ["node(s) didn't match Pod's node affinity/selector"]
                else:
                    why = [f"Insufficient {r}" for r, v in sorted(pod.requests.items()) if free_of(state, n, r) < v]
            if why:
                for w in why:
                    reasons[w] = reasons.get(w, 0) + 1
                preempt["victims" if why[0].startswith("Insufficient") else "np"] += 1
            else:
                fits.append(n)
        if fits:

            def score(n):
                alloc = sum(qty(v) for r, v in k.nodes[n].allocatable.items() if r.startswith(SUFFIX_GPU)) or 1
                used = sum(v for r, v in node_used(state, n).items() if r.startswith(SUFFIX_GPU))
                return (used / alloc, n)

            bind(state, pod, min(fits, key=score))
            continue
        parts = sorted(f"{c} {r}" for r, c in reasons.items())
        pre = []
        if preempt["victims"]:
            pre.append(f"{preempt['victims']} No preemption victims found for incoming pod")
        if preempt["np"]:
            pre.append(f"{preempt['np']} Preemption is not helpful for scheduling")
        msg = (
            f"0/{len(nodes)} nodes are available: {', '.join(parts)}. preemption: 0/{len(nodes)} nodes are available: "
            f"{', '.join(sorted(pre))}.."
        )
        if pod.get("sched_msg") != msg:
            pod.sched_msg = msg
        set_status(state, pod, "Pending")
        event(state, "Pod", pod, "Warning", "FailedScheduling", msg, source="default-scheduler")


def bind(state, pod, n):
    pod.node = n
    pod.sched_msg = ""
    pod.devices = allocate_devices(state, pod, n)
    event(
        state,
        "Pod",
        pod,
        "Normal",
        "Scheduled",
        f"Successfully assigned {pod.namespace}/{pod.name} to {n}",
        source="default-scheduler",
    )


def node_strategy(state, n):
    k = state.k8s
    dp = next((p for p in k.pods if p.node == n and p.component in ("devicePlugin", "standalone-device-plugin")), None)
    if dp is None:
        return "none"
    if dp.component == "devicePlugin":
        return (cp_spec(state).get("mig") or {}).get("strategy", "single")
    return next((d.get("mig_strategy", "none") for d in k.daemonsets if d.name == dp.owner.name), "none")


def allocate_devices(state, pod, n):
    if not state.nodes[n].gpus:
        return []
    inv, _ = gpu_inventory(state, n, node_strategy(state, n))
    inv = apply_sharing(inv, dp_running_config(state, n))
    taken = {tuple(d) for p in state.k8s.pods if p.node == n and p is not pod for d in (p.get("devices") or [])}
    out = []
    for r, v in pod.requests.items():
        if r not in inv:
            continue
        free = [d for d in inv[r] if tuple(d) not in taken]
        for d in free[: int(v)]:
            out.append(list(d))
            taken.add(tuple(d))
    return out


def pull_creds(state, pod, registry):
    """-> (creds | None, missing_secret_names)"""
    k = state.k8s
    names = [s.get("name") for s in pod.spec.get("imagePullSecrets") or []]
    missing, creds = [], None
    for nm in names:
        sec = next((s for s in k.secrets if s.namespace == pod.namespace and s.name == nm), None)
        if sec is None:
            missing.append(nm)
            continue
        raw = (sec.get("data") or {}).get(".dockerconfigjson") or (sec.get("data") or {}).get(".dockercfg")
        try:
            cfg = json.loads(raw or "{}")
        except ValueError:
            continue
        auths = cfg.get("auths", cfg)
        for key, ent in auths.items():
            if key.replace("https://", "").split("/")[0] != registry:
                continue
            user, pw = ent.get("username"), ent.get("password")
            if ent.get("auth") and not user:
                import base64

                user, _, pw = base64.b64decode(ent["auth"]).decode().partition(":")
            creds = (user, pw)
    return creds, missing


def kubelet_sync(state):
    k = state.k8s
    for pod in k.pods:
        if pod.get("component") or pod.node is None:
            continue
        n = pod.node
        kn = k.nodes.get(n)
        if kn is None or not kn.get("ready", True):
            continue
        if pod.status in ("Completed", "Running") and pod.get("started") is not None:
            continue
        rc = pod.spec.get("runtimeClassName")
        if rc:
            rco = next((r for r in k.runtimeclasses if r.name == rc), None)
            handler = rco.handler if rco else rc
            if not ct.containerd_has(state, n, handler):
                set_status(state, pod, "ContainerCreating")
                event(
                    state,
                    "Pod",
                    pod,
                    "Warning",
                    "FailedCreatePodSandBox",
                    f'Failed to create pod sandbox: rpc error: code = Unknown desc = failed to get sandbox runtime: no runtime for "{handler}" is configured',
                )
                continue
        failed = False
        for c in pod.spec.get("containers", []):
            image = c.get("image", "")
            reg = ct.parse_image(image)[0]
            creds, missing = pull_creds(state, pod, reg)
            if missing:
                event(
                    state,
                    "Pod",
                    pod,
                    "Warning",
                    "FailedToRetrieveImagePullSecret",
                    f"Unable to retrieve some image pull secrets ({', '.join(missing)}); attempting to pull the image may not succeed.",
                )
            err = ct.registry_check(state, image, creds)
            if err:
                event(state, "Pod", pod, "Normal", "Pulling", f'Pulling image "{image}"')
                event(
                    state,
                    "Pod",
                    pod,
                    "Warning",
                    "Failed",
                    f'Failed to pull image "{image}": ' + ct.pull_error_text(err, image, tool="containerd"),
                )
                first = pod.status not in ("ErrImagePull", "ImagePullBackOff")
                event(state, "Pod", pod, "Warning", "Failed", "Error: ErrImagePull")
                if not first:
                    event(state, "Pod", pod, "Normal", "BackOff", f'Back-off pulling image "{image}"')
                    event(state, "Pod", pod, "Warning", "Failed", "Error: ImagePullBackOff")
                set_status(state, pod, "ErrImagePull" if first else "ImagePullBackOff")
                pod.logs = [
                    f'Error from server (BadRequest): container "{c.get("name")}" in pod "{pod.name}" is waiting to start: '
                    f"trying and failing to pull image"
                ]
                failed = True
                break
        if failed:
            continue
        for c in pod.spec.get("containers", []):
            event(
                state,
                "Pod",
                pod,
                "Normal",
                "Pulled",
                f'Container image "{c.get("image")}" already present on machine'
                if ct.parse_image(c.get("image", ""))[0] != "nvcr.io"
                else f'Successfully pulled image "{c.get("image")}" in 1m12.517s (1m12.517s including waiting)',
            )
            event(state, "Pod", pod, "Normal", "Created", f"Created container {c.get('name')}")
            event(state, "Pod", pod, "Normal", "Started", f"Started container {c.get('name')}")
        status, logs = user_workload(state, pod)
        set_status(state, pod, status, logs=logs, ready="0/1" if status == "Completed" else None)


def visible_devices(state, pod):
    """GPUs a container in ``pod`` really sees (depends on runtime + device plugin allocation)."""
    n = pod.node
    sn = state.nodes[n]
    rc = pod.spec.get("runtimeClassName")
    runtime = rc or ct.containerd_default(state, n)
    if runtime != "nvidia" or not sn.driver_loaded:
        return []
    if pod.get("devices"):
        seen = []
        for d in pod.devices:  # time-sliced replicas of one GPU are the same physical device
            if tuple(d[:2]) not in seen:
                seen.append(tuple(d[:2]))
        return seen
    image = (pod.spec.get("containers") or [{}])[0].get("image", "")
    env = {e.get("name"): e.get("value") for c in pod.spec.get("containers", []) for e in (c.get("env") or [])}
    nvd = env.get("NVIDIA_VISIBLE_DEVICES", "all" if image.startswith("nvcr.io/nvidia/") else None)
    if nvd == "all":
        return [(g.index, None) for g in sn.gpus if not g.fallen_off_bus]
    return []


def user_workload(state, pod):
    c = (pod.spec.get("containers") or [{}])[0]
    image = c.get("image", "")
    argv = [str(x) for x in (c.get("command") or []) + (c.get("args") or [])]
    if "cuda-sample" in image and "vectoradd" in image:
        devs = visible_devices(state, pod)
        if not devs:
            return "Completed", [
                "Failed to allocate device vector A (error code CUDA driver version is insufficient for CUDA runtime version)!"
            ]
        return "Completed", [
            "[Vector addition of 50000 elements]",
            "Copy input data from the host memory to the CUDA device",
            "CUDA kernel launch with 196 blocks of 256 threads",
            "Copy output data from the CUDA device to the host memory",
            "Test PASSED",
            "Done",
        ]
    flat = argv
    if len(argv) >= 3 and argv[0] in ("sh", "bash", "/bin/sh", "/bin/bash") and argv[1] == "-c":
        try:
            flat = shlex.split(argv[2])
        except ValueError:
            flat = argv
    if flat and flat[0] in ("nvidia-smi", "/usr/bin/nvidia-smi"):
        try:
            out = ct.exec_in_container(
                _FakeCtx(state, pod.node),
                state.nodes[pod.node],
                visible_devices(state, pod),
                flat,
                image,
                hostname=pod.name,
            )
            return "Completed", out.rstrip("\n").split("\n")
        except CommandError as e:
            return "Failed", [str(e)]
    gpus = sum(v for r, v in pod.requests.items() if r.startswith(SUFFIX_GPU))
    if "pytorch" in image or "nemo" in image or "torchrun" in " ".join(argv):
        return "Running", [
            f"[rank0]: world_size={max(1, int(gpus))} backend=nccl",
            "NCCL INFO Bootstrap : Using eth0:10.244.1.17<0>",
            "NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB [2]mlx5_2:1/IB [3]mlx5_3:1/IB ; OOB eth0:10.244.1.17<0>",
            "epoch 0 step 50/5000 loss 7.912 throughput 1840 samples/s",
        ]
    if "tritonserver" in image:
        return "Running", [
            "I1019 09:40:01.000000 1 server.cc:676] +-------+---------+--------+",
            "I1019 09:40:01.000000 1 grpc_server.cc:2463] Started GRPCInferenceService at 0.0.0.0:8001",
            "I1019 09:40:01.000000 1 http_server.cc:4692] Started HTTPService at 0.0.0.0:8000",
        ]
    return "Running", []


class _FakeCtx:
    def __init__(self, state, host):
        self.state, self.host, self.stdin, self.history, self.executed = state, host, "", [], []

    @property
    def node(self):
        return self.state.nodes[self.host]


def cp_status_sync(state):
    k = state.k8s
    cp = k.get("clusterpolicy")
    if not cp:
        return
    ops = [
        p
        for p in k.pods
        if p.get("owner")
        and p.owner.get("kind") == "DaemonSet"
        and any(d.name == p.owner.name and d.get("owner") == "clusterpolicy" for d in k.daemonsets)
    ]
    ready = bool(ops) and all(p.status in ("Running", "Completed") for p in ops)
    cp["status"] = {"state": "ready" if ready else "notReady", "namespace": cp.namespace}
    ncp = k.get("nicclusterpolicy")
    if ncp:
        ops = [p for p in k.pods if p.component in ("rdma-shared-dp", "mofed", "multus")]
        ncp["status"] = {"state": "ready" if ops and all(p.status == "Running" for p in ops) else "notReady"}


def network_operator_sync(state):
    k = state.k8s
    ncp = k.get("nicclusterpolicy")
    owned = [d for d in k.daemonsets if d.get("owner") == "nicclusterpolicy"]
    if not ncp or not any(p.component == "network-operator" and p.status == "Running" for p in k.pods):
        if not ncp:
            k.daemonsets = [d for d in k.daemonsets if d.get("owner") != "nicclusterpolicy"]
        return
    spec, ns = ncp.spec, ncp.namespace
    sel = {"feature.node.kubernetes.io/pci-15b3.present": "true"}
    tols = list(spec.get("tolerations") or []) + [
        {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
    ]
    want = set()
    if spec.get("ofedDriver"):
        want.add("mofed-ubuntu22.04-ds")
        add_daemonset(
            state,
            ns,
            "mofed-ubuntu22.04-ds",
            "nvcr.io/nvidia/mellanox/doca-driver:24.04-0.6.6.0-0",
            "mofed",
            node_selector=sel,
            tolerations=tols,
            owner="nicclusterpolicy",
            release=ncp.get("release"),
            container="mofed-container",
        )
    if spec.get("rdmaSharedDevicePlugin"):
        want.add("rdma-shared-dp-ds")
        add_daemonset(
            state,
            ns,
            "rdma-shared-dp-ds",
            "nvcr.io/nvidia/cloud-native/k8s-rdma-shared-dev-plugin:v1.4.0",
            "rdma-shared-dp",
            node_selector=sel,
            tolerations=tols,
            owner="nicclusterpolicy",
            release=ncp.get("release"),
            container="rdma-shared-dp",
        )
    if spec.get("secondaryNetwork"):
        want.add("kube-multus-ds")
        add_daemonset(
            state,
            ns,
            "kube-multus-ds",
            "ghcr.io/k8snetworkplumbingwg/multus-cni:v3.9.3",
            "multus",
            tolerations=[{"operator": "Exists"}],
            owner="nicclusterpolicy",
            release=ncp.get("release"),
        )
    k.daemonsets = [d for d in k.daemonsets if d.get("owner") != "nicclusterpolicy" or d.name in want]
    del owned


def reconcile(state):
    k8s_init(state)
    for _ in range(3):
        refresh_nodes(state)
        nfd_sync(state)
        gpu_operator_sync(state)
        network_operator_sync(state)
        sync_daemonsets(state)
        sync_deployments(state)
        sync_jobs(state)
        for p in state.k8s.pods:
            if not p.get("component") or p.node is None:
                continue
            kn = state.k8s.nodes.get(p.node)
            if kn is not None and not kn.get("ready", True):
                continue
            if p.component in (
                "driver",
                "toolkit",
                "validator",
                "devicePlugin",
                "gfd",
                "dcgm",
                "dcgmExporter",
                "migManager",
                "nodeStatusExporter",
            ):
                gpu_operand_status(state, p)
            elif p.component == "cuda-validator" or p.component.startswith("runai"):
                continue
            else:
                system_status(state, p)
        cuda_validator_sync(state)
        mig_manager_sync(state)
        device_plugin_sync(state)
        rdma_sync(state)
        gfd_sync(state)
        for hook in RECONCILE_HOOKS:
            hook(state)
        schedule(state)
        kubelet_sync(state)
    cp_status_sync(state)


def _on_service(ctx, name, action):
    state = ctx.state
    if not state.k8s.get("_init"):
        return
    if name in ("containerd", "docker") and action in ("start", "restart", "reload"):
        ct.reload_runtime(state, ctx.host, name)
    reconcile(state)


def _on_reboot(ctx, node):
    state = ctx.state
    if not state.k8s.get("_init"):
        return
    host = node.hostname
    state.k8s.mig_applied.pop(host, None)
    for p in state.k8s.pods:
        if p.node == host and p.get("component"):
            p.restart_base = p.get("restart_base", 0) + 1
            p.last_restart = now(state)
    reconcile(state)


HOOKS_SERVICE.append(_on_service)
HOOKS_REBOOT.append(_on_reboot)


# ====================================================================== logs text


def driver_crash_log(state, n):
    sn = state.nodes[n]
    drv = (cp_spec(state).get("driver") or {}).get("version", "550.90.07")
    return [
        "DRIVER_ARCH is x86_64",
        f"Creating directory NVIDIA-Linux-x86_64-{drv}",
        "Verifying archive integrity... OK",
        f"Uncompressing NVIDIA Accelerated Graphics Driver for Linux-x86_64 {drv}" + "." * 40,
        "",
        "WARNING: Unable to determine the default X library path. The path /tmp/null/lib will be used, but this path was not detected in the ldconfig(8) cache, and no directory exists at this path, so it is likely that libraries installed there will not be found by the loader.",
        "",
        "========== NVIDIA Software Installer ==========",
        "",
        f"Starting installation of NVIDIA driver version {drv} for Linux kernel version {sn.kernel}",
        "",
        "Stopping NVIDIA persistence daemon...",
        "Unloading NVIDIA driver kernel modules...",
        "nvidia_modeset       1306624  0",
        "nvidia_uvm           4648960  0",
        f"nvidia              56811520  {9 + len(sn.gpus)} nvidia_uvm,nvidia_peermem,nvidia_modeset",
        "Could not unload NVIDIA driver kernel modules, driver is in use",
        "Stopping NVIDIA persistence daemon...",
        "Unloading NVIDIA driver kernel modules...",
        "Unmounting NVIDIA driver rootfs...",
    ]


def driver_ok_log(state, n):
    drv = (cp_spec(state).get("driver") or {}).get("version", "550.90.07")
    return [
        f"Starting installation of NVIDIA driver version {drv}",
        "Loading NVIDIA driver kernel modules...",
        "+ modprobe nvidia",
        "+ modprobe nvidia-uvm",
        "+ modprobe nvidia-modeset",
        "Starting NVIDIA persistence daemon...",
        "Mounting NVIDIA driver rootfs...",
        "Done, now waiting for signal",
    ]


def toolkit_log(state):
    t = f'time="{ts(now(state))}"'
    return [
        f'{t} level=info msg="Starting nvidia-toolkit"',
        f"{t} level=info msg=\"Installing NVIDIA container toolkit to '/usr/local/nvidia/toolkit'\"",
        f'{t} level=info msg="Setting up runtime"',
        f'{t} level=info msg="Loading config from /runtime/config-dir/config.toml"',
        f'{t} level=info msg="Flushing config to /runtime/config-dir/config.toml"',
        f'{t} level=info msg="Sending SIGHUP signal to containerd"',
        f'{t} level=info msg="Successfully signaled containerd"',
        f"{t} level=info msg=\"Completed 'setup' for containerd\"",
        f'{t} level=info msg="Waiting for signal"',
    ]


def device_plugin_log(state, n, strategy, inv, err, sharing=None):
    L = klog(state)
    E = klog(state, "E")
    share = ""
    if sharing:
        key = "mps" if sharing["strategy"] == "mps" else "timeSlicing"
        res = ",\n".join(
            f'        {{"name": "{r["name"]}", '
            + (f'"rename": "{r["rename"]}", ' if r.get("rename") else "")
            + f'"devices": "all", "replicas": {r["replicas"]}}}'
            for r in sharing["resources"]
        )
        share = (
            f',\n  "sharing": {{\n    "{key}": {{\n      "failRequestsGreaterThanOne": '
            f'{str(sharing["failRequestsGreaterThanOne"]).lower()},\n      "resources": [\n{res}\n      ]\n    }}\n  }}'
        )
    out = [
        f"{L}main.go:199] Starting FS watcher.",
        f"{L}main.go:206] Starting OS watcher.",
        f"{L}main.go:221] Starting Plugins.",
        f"{L}main.go:278] Loading configuration.",
        f"{L}main.go:303] Updating config with default resource matching patterns.",
        f'{L}main.go:314] \nRunning with config:\n{{\n  "version": "v1",\n  "flags": {{\n    "migStrategy": "{strategy}",\n'
        f'    "failOnInitError": true,\n    "nvidiaDriverRoot": "/",\n    "plugin": {{\n      "passDeviceSpecs": true,\n'
        f'      "deviceListStrategy": ["envvar"],\n      "deviceIDStrategy": "uuid"\n    }}\n  }},\n'
        f'  "resources": {{\n    "gpus": [{{"pattern": "*", "name": "nvidia.com/gpu"}}]'
        + (',\n    "mig": [{"pattern": "*", "name": "nvidia.com/mig-*"}]' if strategy == "mixed" else "")
        + "\n  }"
        + share
        + "\n}",
        f"{L}main.go:317] Retrieving plugins.",
        f"{L}factory.go:104] Detected NVML platform: found NVML library",
        f"{L}factory.go:104] Detected non-Tegra platform: /sys/devices/soc0/family file not found",
    ]
    if err:
        return out + [
            f"{E}main.go:132] error starting plugins: error getting plugins: unable to load resource managers to manage "
            f"plugin devices: failed to construct NVML resource managers: error building device map: error building "
            f"device map from config.resources: {err}"
        ]
    for r, devs in sorted(inv.items()):
        if not devs:
            continue
        sock = r.replace("nvidia.com/", "nvidia-")
        out += [
            f"{L}server.go:216] Starting GRPC server for '{r}'",
            f"{L}server.go:147] Starting to serve '{r}' on /var/lib/kubelet/device-plugins/{sock}.sock",
            f"{L}server.go:154] Registered device plugin for '{r}' with Kubelet",
        ]
    return out


def device_plugin_nonvml_log(state, crash=True):
    L, W, E = klog(state), klog(state, "W"), klog(state, "E")
    out = [
        f"{L}main.go:199] Starting FS watcher.",
        f"{L}main.go:206] Starting OS watcher.",
        f"{L}main.go:221] Starting Plugins.",
        f"{L}main.go:278] Loading configuration.",
        f"{L}main.go:303] Updating config with default resource matching patterns.",
        f'{L}main.go:314] \nRunning with config:\n{{\n  "version": "v1",\n  "flags": {{\n    "migStrategy": "none",\n'
        f'    "failOnInitError": true,\n    "nvidiaDriverRoot": "/"\n  }}\n}}',
        f"{L}main.go:317] Retrieving plugins.",
        f"{W}factory.go:31] No valid resources detected, creating a null CDI handler",
        f"{L}factory.go:104] Detected non-NVML platform: could not load NVML library: libnvidia-ml.so.1: cannot open shared object file: No such file or directory",
        f"{L}factory.go:104] Detected non-Tegra platform: /sys/devices/soc0/family file not found",
        f"{E}factory.go:112] Incompatible platform detected",
        f"{E}factory.go:113] If this is a GPU node, did you configure the NVIDIA Container Toolkit?",
        f"{E}factory.go:114] You can check the prerequisites at: https://github.com/NVIDIA/k8s-device-plugin#prerequisites",
        f"{E}factory.go:115] You can learn how to set the runtime at: https://github.com/NVIDIA/k8s-device-plugin#quick-start",
        f"{E}factory.go:116] If this is not a GPU node, you should set up a toleration or nodeSelector to only deploy this plugin on GPU nodes",
    ]
    if crash:
        out.append(
            f"{E}main.go:132] error starting plugins: error creating plugin manager: unable to create plugin manager: platform detection failed"
        )
    else:
        out.append(f"{L}main.go:346] No devices found. Waiting indefinitely.")
    return out


def operator_log(state):
    cp = state.k8s.get("clusterpolicy")
    if not cp:
        return ['{"level":"info","msg":"No ClusterPolicy found, waiting"}']
    out = []
    for name, key, *_ in GPU_OPERANDS:
        if key != "validator" and not operand_enabled(cp.spec, key):
            out.append(
                f'{{"level":"info","ts":{1792399800 + now(state)},"logger":"controllers.ClusterPolicy",'
                f'"msg":"ClusterPolicy step completed","state:":"state-{_state_name(key)}","status":"disabled"}}'
            )
            continue
        pods = [p for p in state.k8s.pods if p.get("owner") and p.owner.get("name") == name]
        ok = pods and all(p.status in ("Running", "Completed") for p in pods)
        out.append(
            f'{{"level":"info","ts":{1792399800 + now(state)},"logger":"controllers.ClusterPolicy",'
            f'"msg":"ClusterPolicy step completed","state:":"state-{_state_name(key)}","status":"{"ready" if ok else "notReady"}"}}'
        )
    return out


def _state_name(key):
    return {
        "driver": "driver",
        "toolkit": "container-toolkit",
        "validator": "operator-validation",
        "devicePlugin": "device-plugin",
        "gfd": "gpu-feature-discovery",
        "dcgm": "dcgm",
        "dcgmExporter": "dcgm-exporter",
        "migManager": "mig-manager",
        "nodeStatusExporter": "node-status-exporter",
    }[key]


# ====================================================================== API object views


def node_obj(state, n):
    kn, sn = state.k8s.nodes[n], state.nodes.get(n)
    meta = {
        "name": n,
        "labels": dict(kn.labels),
        "creationTimestamp": ts(kn.created),
        "annotations": {
            "node.alpha.kubernetes.io/ttl": "0",
            "volumes.kubernetes.io/controller-managed-attach-detach": "true",
        },
    }
    spec = {"podCIDR": f"10.244.{sorted(state.k8s.nodes).index(n)}.0/24"}
    if kn.get("taints"):
        spec["taints"] = [dict(t) for t in kn.taints]
        for t in spec["taints"]:
            if not t.get("value"):
                t.pop("value", None)
    if not kn.get("schedulable", True):
        spec["unschedulable"] = True
        spec.setdefault("taints", []).append(
            {"key": "node.kubernetes.io/unschedulable", "effect": "NoSchedule", "timeAdded": ts(now(state))}
        )
    ready = kn.get("ready", True)
    status = {
        "capacity": dict(kn.capacity),
        "allocatable": dict(kn.allocatable),
        "conditions": [
            {
                "type": "MemoryPressure",
                "status": "False",
                "reason": "KubeletHasSufficientMemory",
                "message": "kubelet has sufficient memory available",
            },
            {
                "type": "DiskPressure",
                "status": "False",
                "reason": "KubeletHasNoDiskPressure",
                "message": "kubelet has no disk pressure",
            },
            {
                "type": "PIDPressure",
                "status": "False",
                "reason": "KubeletHasSufficientPID",
                "message": "kubelet has sufficient PID available",
            },
            {
                "type": "Ready",
                "status": "True" if ready else "Unknown",
                "reason": "KubeletReady" if ready else "NodeStatusUnknown",
                "message": "kubelet is posting ready status" if ready else "Kubelet stopped posting node status.",
            },
        ],
        "addresses": [{"type": "InternalIP", "address": sn.ip if sn else ""}, {"type": "Hostname", "address": n}],
        "nodeInfo": {
            "architecture": "amd64",
            "containerRuntimeVersion": CRI,
            "kernelVersion": sn.kernel if sn else "",
            "kubeletVersion": kn.get("version", K8S_VERSION),
            "kubeProxyVersion": kn.get("version", K8S_VERSION),
            "operatingSystem": "linux",
            "osImage": (sn.os.split(" (")[0] if sn else ""),
        },
    }
    return {"apiVersion": "v1", "kind": "Node", "metadata": meta, "spec": spec, "status": status}


def pod_obj(state, p):
    meta = {
        "name": p.name,
        "namespace": p.namespace,
        "uid": p.uid,
        "creationTimestamp": ts(p.created),
        "labels": dict(p.labels or {}),
    }
    if p.get("owner"):
        meta["ownerReferences"] = [
            {
                "apiVersion": "apps/v1" if p.owner.kind in ("ReplicaSet", "DaemonSet") else "batch/v1",
                "kind": p.owner.kind,
                "name": p.owner.name,
                "controller": True,
            }
        ]
    spec = copy.deepcopy(dict(p.spec))
    spec = json.loads(json.dumps(spec, default=str))
    if p.node:
        spec["nodeName"] = p.node
    spec.setdefault("restartPolicy", "Always")
    spec.setdefault("schedulerName", "default-scheduler")
    spec.setdefault("serviceAccountName", "default")
    spec["tolerations"] = list(spec.get("tolerations") or []) + DEFAULT_POD_TOLERATIONS
    cstat = []
    for c in spec.get("containers", []):
        st = {
            "name": c.get("name"),
            "image": c.get("image"),
            "ready": p.status == "Running",
            "restartCount": pod_restarts(state, p),
            "started": p.status == "Running",
        }
        if p.status == "Running":
            st["state"] = {"running": {"startedAt": ts(p.get("started") or p.created)}}
        elif p.status == "Completed":
            st["state"] = {"terminated": {"exitCode": 0, "reason": "Completed"}}
        elif p.node:
            st["state"] = {"waiting": {"reason": p.status if not p.status.startswith("Init") else "PodInitializing"}}
        cstat.append(st)
    conds = [{"type": "PodScheduled", "status": "True" if p.node else "False"}]
    if not p.node and p.get("sched_msg"):
        conds[0].update({"reason": "Unschedulable", "message": p.sched_msg})
    if p.node:
        conds.append({"type": "Ready", "status": "True" if p.status == "Running" else "False"})
    status = {"phase": p.phase, "conditions": conds, "qosClass": "BestEffort"}
    if p.node:
        status.update(
            {
                "hostIP": state.nodes[p.node].ip if p.node in state.nodes else "",
                "podIP": p.get("ip") or "",
                "containerStatuses": cstat,
                "startTime": ts(p.created),
            }
        )
    return {"apiVersion": "v1", "kind": "Pod", "metadata": meta, "spec": spec, "status": status}


def ds_obj(state, d):
    pods = pods_of(state, "DaemonSet", d.name, d.namespace)
    desired = len(ds_eligible(state, d))
    ready = sum(1 for p in pods if p.status == "Running")
    tmpl_spec = {"containers": [{"name": d.container, "image": d.image}], "tolerations": list(d.tolerations)}
    if d.node_selector:
        tmpl_spec["nodeSelector"] = dict(d.node_selector)
    if d.get("runtime_class"):
        tmpl_spec["runtimeClassName"] = d.runtime_class
    return {
        "apiVersion": "apps/v1",
        "kind": "DaemonSet",
        "metadata": {"name": d.name, "namespace": d.namespace, "creationTimestamp": ts(d.created)},
        "spec": {
            "selector": {"matchLabels": {"app": d.name}},
            "template": {"metadata": {"labels": {"app": d.name}}, "spec": tmpl_spec},
        },
        "status": {
            "desiredNumberScheduled": desired,
            "currentNumberScheduled": len(pods),
            "numberReady": ready,
            "updatedNumberScheduled": len(pods),
            "numberAvailable": ready,
        },
    }


def deploy_obj(state, d):
    pods = [
        p
        for p in state.k8s.pods
        if p.get("owner") and p.owner.get("deployment") == d.name and p.namespace == d.namespace
    ]
    ready = sum(1 for p in pods if p.status == "Running")
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": d.name,
            "namespace": d.namespace,
            "labels": dict(d.labels),
            "creationTimestamp": ts(d.created),
            "generation": d.get("generation", 1),
        },
        "spec": {
            "replicas": int(d.replicas),
            "selector": json.loads(json.dumps(d.selector)),
            "template": json.loads(json.dumps(d.template, default=str)),
        },
        "status": {
            "replicas": len(pods),
            "readyReplicas": ready,
            "availableReplicas": ready,
            "updatedReplicas": len(pods),
        },
    }


def job_obj(state, j):
    pods = pods_of(state, "Job", j.name, j.namespace)
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": j.name, "namespace": j.namespace, "creationTimestamp": ts(j.created)},
        "spec": {
            "parallelism": j.get("parallelism", 1),
            "completions": j.get("completions", 1),
            "template": json.loads(json.dumps(j.template, default=str)),
        },
        "status": {
            "active": sum(1 for p in pods if p.phase in ("Pending", "Running")),
            "succeeded": sum(1 for p in pods if p.status == "Completed"),
        },
    }


def cp_obj(state):
    cp = state.k8s.clusterpolicy
    return {
        "apiVersion": "nvidia.com/v1",
        "kind": "ClusterPolicy",
        "metadata": {
            "name": cp.name,
            "creationTimestamp": ts(cp.created),
            "labels": {"app.kubernetes.io/managed-by": "Helm"},
            "annotations": {
                "meta.helm.sh/release-name": (cp.get("release") or "/").split("/")[-1],
                "meta.helm.sh/release-namespace": cp.namespace,
            },
        },
        "spec": json.loads(json.dumps(cp.spec)),
        "status": json.loads(json.dumps(cp.get("status") or {})),
    }


def ncp_obj(state):
    ncp = state.k8s.nicclusterpolicy
    return {
        "apiVersion": "mellanox.com/v1alpha1",
        "kind": "NicClusterPolicy",
        "metadata": {"name": ncp.name, "creationTimestamp": ts(ncp.created)},
        "spec": json.loads(json.dumps(ncp.spec)),
        "status": json.loads(json.dumps(ncp.get("status") or {})),
    }


def secret_obj(state, s):
    import base64

    data = {k_: base64.b64encode(str(v).encode()).decode() for k_, v in (s.get("data") or {}).items()}
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "type": s.get("type", "Opaque"),
        "metadata": {"name": s.name, "namespace": s.namespace, "creationTimestamp": ts(s.created)},
        "data": data,
    }


# ====================================================================== jsonpath


def _jp_steps(path):
    path = path.strip()
    if path.startswith("$"):
        path = path[1:]
    steps, i = [], 0
    while i < len(path):
        ch = path[i]
        if ch == ".":
            i += 1
            key = ""
            while i < len(path) and path[i] not in ".[":
                if path[i] == "\\" and i + 1 < len(path):
                    key += path[i + 1]
                    i += 2
                    continue
                key += path[i]
                i += 1
            if key == "*":
                steps.append(("all",))
            elif key:
                steps.append(("key", key))
        elif ch == "[":
            j = path.index("]", i) if "?(" not in path[i : i + 3] else path.index(")]", i) + 1
            inner = path[i + 1 : j].strip()
            i = j + 1
            if inner == "*":
                steps.append(("all",))
            elif inner.startswith("?("):
                m = re.match(r"\?\(@\.([\w.\\/-]+)\s*(==|!=)\s*['\"]?([^'\"]*)['\"]?\)", inner)
                if m:
                    steps.append(("filter", _jp_steps("." + m.group(1)), m.group(2), m.group(3)))
            elif inner[:1] in "'\"":
                steps.append(("key", inner.strip("'\"")))
            else:
                try:
                    steps.append(("idx", int(inner.split(":")[0] or 0)))
                except ValueError:
                    steps.append(("key", inner))
        elif ch == "@":
            i += 1
        else:
            key = ""
            while i < len(path) and path[i] not in ".[":
                key += path[i]
                i += 1
            steps.append(("key", key))
    return steps


def jp_eval(obj, path):
    cur = [obj]
    for st in _jp_steps(path):
        nxt = []
        for c in cur:
            if (
                st[0] == "key"
                and isinstance(c, dict)
                and st[1] in c
                or st[0] == "idx"
                and isinstance(c, list)
                and -len(c) <= st[1] < len(c)
            ):
                nxt.append(c[st[1]])
            elif st[0] == "all":
                nxt.extend(c if isinstance(c, list) else list(c.values()) if isinstance(c, dict) else [])
            elif st[0] == "filter" and isinstance(c, list):
                for item in c:
                    vals = jp_eval(item, "".join("." + s[1] for s in st[1] if s[0] == "key"))
                    v = str(vals[0]) if vals else None
                    if (st[2] == "==" and v == st[3]) or (st[2] == "!=" and v != st[3]):
                        nxt.append(item)
        cur = nxt
    return cur


def _jp_fmt(v):
    if isinstance(v, (dict, list)):
        return json.dumps(v, separators=(",", ":"))
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def jsonpath(obj, tmpl):
    tmpl = tmpl.strip()
    if "{" not in tmpl:
        tmpl = "{" + tmpl + "}"
    tokens = re.split(r"(\{[^{}]*\})", tmpl)

    def render(toks, cur):
        out, i = [], 0
        while i < len(toks):
            t = toks[i]
            if t.startswith("{") and t.endswith("}"):
                expr = t[1:-1].strip()
                if expr.startswith("range "):
                    depth, j = 1, i + 1
                    while j < len(toks):
                        e = toks[j][1:-1].strip() if toks[j].startswith("{") else ""
                        if e.startswith("range "):
                            depth += 1
                        elif e == "end":
                            depth -= 1
                            if depth == 0:
                                break
                        j += 1
                    body = toks[i + 1 : j]
                    for item in jp_eval(cur if not expr[6:].strip().startswith("$") else obj, expr[6:].strip()):
                        out.append(render(body, item))
                    i = j + 1
                    continue
                if expr == "end":
                    i += 1
                    continue
                if expr[:1] in "'\"":
                    out.append(expr[1:-1].encode().decode("unicode_escape"))
                else:
                    out.append(" ".join(_jp_fmt(v) for v in jp_eval(obj if expr.startswith("$") else cur, expr)))
            else:
                out.append(t)
            i += 1
        return "".join(out)

    return render(tokens, obj)


def custom_columns(objs, spec):
    cols = []
    for part in re.split(r",(?![^\[]*\])", spec):
        if ":" in part:
            h, p = part.split(":", 1)
            cols.append((h, p))
    rows = [[h for h, _ in cols]]
    for o in objs:
        row = []
        for _, p in cols:
            vals = jp_eval(o, p)
            row.append(",".join(_jp_fmt(v) for v in vals) if vals else "<none>")
        rows.append(row)
    return table(rows)


# ====================================================================== argument parsing

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


def parse_args(args, verb):
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


def resolve_kind(word):
    w = word.lower()
    for kind, names in KINDS.items():
        if w in names:
            return kind
    raise CommandError(f'error: the server doesn\'t have a resource type "{word}"', 1)


def split_targets(pos):
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


def ns_of(opts):
    return opts.get("namespace") or "default"


# ====================================================================== object listing


def list_objects(state, kind, opts, names=()):
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
        if not _crd_installed(state, "clusterpolicies"):
            raise CommandError('error: the server doesn\'t have a resource type "clusterpolicies"', 1)
    elif kind == "nicclusterpolicies":
        if not _crd_installed(state, "nicclusterpolicies"):
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


def _crd_installed(state, kind):
    k = state.k8s
    chart = {"clusterpolicies": "gpu-operator", "nicclusterpolicies": "network-operator"}[kind]
    return (
        any(r.get("chart") == chart for r in k.helm_releases)
        or (kind == "clusterpolicies" and k.get("clusterpolicy"))
        or (kind == "nicclusterpolicies" and k.get("nicclusterpolicy"))
    )


def api_obj(state, kind, item):
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


# ====================================================================== get


def node_status_str(kn):
    s = "Ready" if kn.get("ready", True) else "NotReady"
    if not kn.get("schedulable", True):
        s += ",SchedulingDisabled"
    return s


def labels_str(labels):
    return ",".join(f"{k_}={v}" for k_, v in sorted((labels or {}).items())) or "<none>"


def print_table(state, kind, items, opts, prefix=""):
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


def output(state, kind, items, opts, single=False):
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


def cmd_get(state, pos, opts):
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


# ====================================================================== describe


def _kv_block(d, indent=2, width=None):
    if not d:
        return " " * indent + "<none>"
    width = width or max(len(k_) for k_ in d) + 3
    return "\n".join(" " * indent + f"{k_ + ':':<{width}}{v}" for k_, v in d.items())


def describe_events(state, uid, indent=""):
    evs = events_for(state, uid)
    if not evs:
        return "Events:              <none>"
    rows = [("  Type", "Reason", "Age", "From", "Message"), ("  ----", "------", "----", "----", "-------")]
    for e in sorted(evs, key=lambda e: e.last):
        a = age(state, e.last) if e.count == 1 else f"{age(state, e.last)} (x{e.count} over {age(state, e.first)})"
        rows.append(("  " + e.type, e.reason, a, e.source, e.message))
    return "Events:\n" + table(rows, gap=2).rstrip("\n")


def describe_node(state, n):
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


def describe_pod(state, p):
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


def cmd_describe(state, pos, opts):
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


# ====================================================================== apply / create / delete


def _read_manifest(ctx, path):
    if path == "-":
        return ctx.stdin
    if path.startswith(("http://", "https://")):
        if "k8s-device-plugin" in path and "nvidia-device-plugin.yml" in path:
            return STATIC_DEVICE_PLUGIN
        raise CommandError(f'error: unable to read URL "{path}", server reported 404 Not Found, status code=404', 1)
    full = path if path.startswith("/") else f"{ctx.cwd.rstrip('/')}/{path}"
    files = ctx.node.files
    if full not in files and path not in files:
        raise CommandError(f'error: the path "{path}" does not exist', 1)
    return files.get(full, files.get(path))


STATIC_DEVICE_PLUGIN = """\
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: nvidia-device-plugin-daemonset
  namespace: kube-system
spec:
  selector:
    matchLabels:
      name: nvidia-device-plugin-ds
  template:
    metadata:
      labels:
        name: nvidia-device-plugin-ds
    spec:
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      priorityClassName: "system-node-critical"
      containers:
      - image: nvcr.io/nvidia/k8s-device-plugin:v0.15.0
        name: nvidia-device-plugin-ctr
        env:
          - name: FAIL_ON_INIT_ERROR
            value: "false"
"""


def load_docs(ctx, opts):
    files = opts.get("filename") or []
    if not files:
        raise CommandError("error: must specify one of -f and -k", 1)
    docs = []
    for f in files:
        text = _read_manifest(ctx, f)
        try:
            for d in yaml.safe_load_all(text):
                if d is None:
                    continue
                if not isinstance(d, dict):
                    raise CommandError(
                        f"error: error parsing {f}: error unmarshaling JSON: json: cannot unmarshal string into Go value of type map[string]interface {{}}",
                        1,
                    )
                if d.get("kind") == "List":
                    docs.extend((f, x) for x in d.get("items", []))
                else:
                    docs.append((f, d))
        except yaml.YAMLError as e:
            mark = getattr(e, "problem_mark", None)
            line = mark.line + 1 if mark else 1
            raise CommandError(
                f"error: error parsing {f}: error converting YAML to JSON: yaml: line {line}: "
                f"{getattr(e, 'problem', 'could not parse')}",
                1,
            )
    return docs


ALLOWED_POD_UPDATE = ("image", "activeDeadlineSeconds", "terminationGracePeriodSeconds")


def _norm(o):
    return json.loads(json.dumps(o, sort_keys=True, default=str))


def pod_update_allowed(old_spec, new_spec):
    o, n = _norm(old_spec), _norm(new_spec)
    for spec in (o, n):
        spec.pop("activeDeadlineSeconds", None)
        spec.pop("terminationGracePeriodSeconds", None)
        for c in spec.get("containers", []):
            c.pop("image", None)
    ot, nt = o.pop("tolerations", None) or [], n.pop("tolerations", None) or []
    if any(t not in nt for t in ot):
        return False
    return o == n


def apply_doc(ctx, src, doc, mode="apply"):
    state = ctx.state
    k = state.k8s
    kind = doc.get("kind")
    api = doc.get("apiVersion", "")
    meta = doc.get("metadata") or {}
    name = meta.get("name")
    if not kind or not api:
        raise CommandError(
            f'error: error validating "{src}": error validating data: [apiVersion not set, kind not set]; '
            "if you choose to ignore these errors, turn validation off with --validate=false",
            1,
        )
    if not name and kind != "List":
        raise CommandError(
            f'error: error when retrieving current configuration of:\nResource: "{kind.lower()}s", '
            f'GroupVersionKind: "{api}, Kind={kind}"\nName: "", Namespace: "default"\nfrom server for: '
            f'"{src}": resource name may not be empty',
            1,
        )
    ns = meta.get("namespace") or ctx_ns(ctx) or "default"
    labels = dict(meta.get("labels") or {})
    verb_new = "created"
    exists_err = f'Error from server (AlreadyExists): error when creating "{src}": {{kind}} "{name}" already exists'

    def need_ns():
        if ns not in k.namespaces:
            raise CommandError(
                f'Error from server (NotFound): error when creating "{src}": namespaces "{ns}" not found', 1
            )

    if kind == "Namespace":
        if name in k.namespaces:
            if mode == "create":
                raise CommandError(exists_err.format(kind="namespaces"), 1)
            return f"namespace/{name} unchanged"
        ensure_namespace(state, name)
        return f"namespace/{name} {verb_new}"
    if kind == "Pod":
        need_ns()
        spec = doc.get("spec") or {}
        validate_pod_spec("Pod", name, spec)
        if spec.get("runtimeClassName") and not any(r.name == spec["runtimeClassName"] for r in k.runtimeclasses):
            raise CommandError(
                f'Error from server (Forbidden): error when creating "{src}": pods "{name}" is forbidden: '
                f'pod rejected: RuntimeClass "{spec["runtimeClassName"]}" not found',
                1,
            )
        cur = next((p for p in k.pods if p.namespace == ns and p.name == name), None)
        if cur:
            if mode == "create":
                raise CommandError(exists_err.format(kind="pods"), 1)
            if _norm(cur.spec) == _norm(_with_sa_secrets(state, ns, spec, cur.spec)) and dict(cur.labels) == labels:
                return f"pod/{name} unchanged"
            if not pod_update_allowed(cur.spec, _with_sa_secrets(state, ns, spec, cur.spec)):
                diff = _spec_diff(cur.spec, spec)
                raise CommandError(
                    f'The Pod "{name}" is invalid: spec: Forbidden: pod updates may not change fields other than '
                    "`spec.containers[*].image`,`spec.initContainers[*].image`,`spec.activeDeadlineSeconds`,"
                    "`spec.tolerations` (only additions to existing tolerations),`spec.terminationGracePeriodSeconds` "
                    "(allow it to be set to 1 if it was previously negative)\n" + diff,
                    1,
                )
            cur.spec = wrap(_with_sa_secrets(state, ns, spec, cur.spec))
            cur.labels = labels
            cur.requests = pod_requests(cur.spec)
            return f"pod/{name} configured"
        spec = _with_sa_secrets(state, ns, spec)
        new_pod(state, ns, name, spec, labels=labels)
        return f"pod/{name} {verb_new}"
    if kind in ("Deployment", "Job", "DaemonSet") and api.startswith(("apps/", "batch/")):
        need_ns()
        spec = doc.get("spec") or {}
        tmpl = spec.get("template") or {}
        validate_pod_spec(kind, name, tmpl.get("spec") or {}, "spec.template.spec")
        tl = (tmpl.get("metadata") or {}).get("labels") or {}
        ml = (spec.get("selector") or {}).get("matchLabels")
        if kind != "Job" and (ml is None or not all(tl.get(a) == b for a, b in ml.items())):
            raise CommandError(
                f'The {kind} "{name}" is invalid: spec.template.metadata.labels: Invalid value: '
                f"map[string]string{{{', '.join(f'{a!r}:{b!r}' for a, b in tl.items())}}}: "
                "`selector` does not match template `labels`",
                1,
            )
        tmpl = _norm(tmpl)
        tmpl["spec"] = _with_sa_secrets(state, ns, tmpl.get("spec") or {})
        if kind == "Deployment":
            cur = next((d for d in k.deployments if d.namespace == ns and d.name == name), None)
            if cur:
                if mode == "create":
                    raise CommandError(exists_err.format(kind="deployments.apps"), 1)
                if _norm(cur.template) == tmpl and int(cur.replicas) == int(spec.get("replicas", 1)):
                    return f"deployment.apps/{name} unchanged"
                cur.template = tmpl
                cur.replicas = int(spec.get("replicas", 1))
                cur.generation = cur.get("generation", 1) + 1
                cur.selector = {"matchLabels": dict(ml)}
                return f"deployment.apps/{name} configured"
            d = add_deployment(
                state, ns, name, "", replicas=int(spec.get("replicas", 1)), template=tmpl, labels=labels or None
            )
            d.selector = {"matchLabels": dict(ml)}
            return f"deployment.apps/{name} {verb_new}"
        if kind == "Job":
            cur = next((j for j in k.jobs if j.namespace == ns and j.name == name), None)
            if cur:
                if _norm(cur.template) == tmpl:
                    return f"job.batch/{name} unchanged"
                raise CommandError(
                    f'The Job "{name}" is invalid: spec.template: Invalid value: core.PodTemplateSpec{{...}}: field is immutable',
                    1,
                )
            k.jobs.append(
                {
                    "namespace": ns,
                    "name": name,
                    "template": tmpl,
                    "parallelism": spec.get("parallelism", 1),
                    "completions": spec.get("completions", 1),
                    "created": now(state),
                }
            )
            return f"job.batch/{name} {verb_new}"
        # DaemonSet
        cspec = tmpl["spec"]["containers"][0]
        image = cspec.get("image", "")
        comp = "standalone-device-plugin" if "k8s-device-plugin" in image else "generic"
        env = {e.get("name"): e.get("value") for e in cspec.get("env") or []}
        existed = any(d.namespace == ns and d.name == name for d in k.daemonsets)
        add_daemonset(
            state,
            ns,
            name,
            image,
            comp,
            node_selector=tmpl["spec"].get("nodeSelector") or {},
            tolerations=tmpl["spec"].get("tolerations") or [],
            owner="user",
            container=cspec.get("name"),
            runtime_class=tmpl["spec"].get("runtimeClassName"),
            extra={"mig_strategy": env.get("MIG_STRATEGY", "none")},
        )
        return f"daemonset.apps/{name} {'configured' if existed else verb_new}"
    if kind == "Secret":
        need_ns()
        import base64

        data = {a: base64.b64decode(str(b)).decode(errors="replace") for a, b in (doc.get("data") or {}).items()}
        data.update({a: str(b) for a, b in (doc.get("stringData") or {}).items()})
        return _upsert(
            k.secrets, ns, name, {"type": doc.get("type", "Opaque"), "data": data}, state, "secret", mode, src
        )
    if kind == "ConfigMap":
        need_ns()
        return _upsert(k.configmaps, ns, name, {"data": dict(doc.get("data") or {})}, state, "configmap", mode, src)
    if kind == "ServiceAccount":
        need_ns()
        k.serviceaccounts.setdefault(ns, {})[name] = {"imagePullSecrets": list(doc.get("imagePullSecrets") or [])}
        return f"serviceaccount/{name} configured"
    if kind == "Service":
        need_ns()
        return f"service/{name} {verb_new}"
    if kind == "RuntimeClass":
        cur = next((r for r in k.runtimeclasses if r.name == name), None)
        if cur:
            cur.handler = doc.get("handler", name)
            return f"runtimeclass.node.k8s.io/{name} configured"
        k.runtimeclasses.append({"name": name, "handler": doc.get("handler", name), "created": now(state)})
        return f"runtimeclass.node.k8s.io/{name} {verb_new}"
    if kind == "ClusterPolicy":
        if not _crd_installed(state, "clusterpolicies"):
            raise CommandError(
                f'error: resource mapping not found for name: "{name}" namespace: "" from "{src}": no matches for kind '
                f'"ClusterPolicy" in version "{api}"\nensure CRDs are installed first',
                1,
            )
        if k.get("clusterpolicy"):
            if _norm(k.clusterpolicy.spec) == _norm(doc.get("spec") or {}):
                return f"clusterpolicy.nvidia.com/{name} unchanged"
            k.clusterpolicy.spec = doc.get("spec") or {}
            return f"clusterpolicy.nvidia.com/{name} configured"
        rel = next((r for r in k.helm_releases if r.chart == "gpu-operator"), None)
        k.clusterpolicy = {
            "name": name,
            "namespace": rel.namespace if rel else "gpu-operator",
            "spec": doc.get("spec") or {},
            "created": now(state),
            "version": rel.version if rel else "v24.6.2",
            "release": None,
        }
        return f"clusterpolicy.nvidia.com/{name} {verb_new}"
    ext = next((h for h in EXTRA_KINDS.values() if h.kind == kind), None)
    if ext is not None:
        return ext.apply(state, doc, src)
    if kind == "NicClusterPolicy":
        if not _crd_installed(state, "nicclusterpolicies"):
            raise CommandError(
                f'error: resource mapping not found for name: "{name}" namespace: "" from "{src}": no matches for kind '
                f'"NicClusterPolicy" in version "{api}"\nensure CRDs are installed first',
                1,
            )
        rel = next((r for r in k.helm_releases if r.chart == "network-operator"), None)
        if k.get("nicclusterpolicy"):
            k.nicclusterpolicy.spec = doc.get("spec") or {}
            return f"nicclusterpolicy.mellanox.com/{name} configured"
        k.nicclusterpolicy = {
            "name": name,
            "namespace": rel.namespace if rel else "nvidia-network-operator",
            "spec": doc.get("spec") or {},
            "created": now(state),
            "release": None,
        }
        return f"nicclusterpolicy.mellanox.com/{name} {verb_new}"
    raise CommandError(
        f'error: resource mapping not found for name: "{name}" namespace: "{ns}" from "{src}": no matches for kind '
        f'"{kind}" in version "{api}"\nensure CRDs are installed first',
        1,
    )


def _upsert(lst, ns, name, fields, state, label, mode, src):
    cur = next((s for s in lst if s.namespace == ns and s.name == name), None)
    if cur:
        if mode == "create":
            raise CommandError(
                f'Error from server (AlreadyExists): error when creating "{src}": {label}s "{name}" already exists', 1
            )
        cur.update(fields)
        return f"{label}/{name} configured"
    rec = {"namespace": ns, "name": name, "created": now(state)}
    rec.update(fields)
    lst.append(rec)
    return f"{label}/{name} created"


def _with_sa_secrets(state, ns, spec, previous=None):
    spec = _norm(spec)
    if previous is not None and previous.get("imagePullSecrets") and not spec.get("imagePullSecrets"):
        spec["imagePullSecrets"] = _norm(previous.imagePullSecrets)
    if not spec.get("imagePullSecrets"):
        sa = (state.k8s.serviceaccounts.get(ns, {}) or {}).get(spec.get("serviceAccountName", "default")) or {}
        if sa.get("imagePullSecrets"):
            spec["imagePullSecrets"] = _norm(sa["imagePullSecrets"])
    return spec


def _spec_diff(old, new):
    o = yaml.safe_dump(_norm(old), default_flow_style=False).splitlines()
    n = yaml.safe_dump(_norm(new), default_flow_style=False).splitlines()
    out = ["  core.PodSpec{"]
    for line in o:
        if line not in n:
            out.append("- " + line)
    for line in n:
        if line not in o:
            out.append("+ " + line)
    out.append("  }")
    return "\n".join(out)


def ctx_ns(ctx):
    return getattr(ctx, "_kns", None)


def delete_obj(state, kind, ns, name, opts):
    k = state.k8s
    if kind == "pods":
        p = next((p for p in k.pods if p.namespace == ns and p.name == name), None)
        if p is None:
            raise CommandError(f'Error from server (NotFound): pods "{name}" not found', 1)
        _remove_pod(state, p)
        return f'pod "{name}" deleted'
    if kind == "deployments":
        d = next((d for d in k.deployments if d.namespace == ns and d.name == name), None)
        if d is None:
            raise CommandError(f'Error from server (NotFound): deployments.apps "{name}" not found', 1)
        k.deployments.remove(d)
        k.pods = [p for p in k.pods if not (p.get("owner") and p.owner.get("deployment") == name and p.namespace == ns)]
        return f'deployment.apps "{name}" deleted'
    if kind == "jobs":
        j = next((j for j in k.jobs if j.namespace == ns and j.name == name), None)
        if j is None:
            raise CommandError(f'Error from server (NotFound): jobs.batch "{name}" not found', 1)
        k.jobs.remove(j)
        k.pods = [p for p in k.pods if not (p.get("owner") and p.owner.get("kind") == "Job" and p.owner.name == name)]
        return f'job.batch "{name}" deleted'
    if kind == "daemonsets":
        d = next((d for d in k.daemonsets if d.namespace == ns and d.name == name), None)
        if d is None:
            raise CommandError(f'Error from server (NotFound): daemonsets.apps "{name}" not found', 1)
        if d.get("owner") == "clusterpolicy":
            return f'daemonset.apps "{name}" deleted'  # the operator recreates it immediately
        k.daemonsets.remove(d)
        return f'daemonset.apps "{name}" deleted'
    if kind == "namespaces":
        if name not in k.namespaces:
            raise CommandError(f'Error from server (NotFound): namespaces "{name}" not found', 1)
        if name in ("default", "kube-system"):
            raise CommandError(
                f'Error from server (Forbidden): namespaces "{name}" is forbidden: this namespace may not be deleted', 1
            )
        k.namespaces.remove(name)
        for lst in ("pods", "deployments", "jobs", "secrets", "configmaps", "daemonsets"):
            k[lst] = [o for o in k[lst] if o.get("namespace") != name]
        return f'namespace "{name}" deleted'
    if kind in ("secrets", "configmaps"):
        lst = k[kind]
        o = next((s for s in lst if s.namespace == ns and s.name == name), None)
        if o is None:
            raise CommandError(f'Error from server (NotFound): {kind} "{name}" not found', 1)
        lst.remove(o)
        return f'{SINGULAR[kind]} "{name}" deleted'
    if kind == "clusterpolicies":
        if not k.get("clusterpolicy") or k.clusterpolicy.name != name:
            raise CommandError(f'Error from server (NotFound): clusterpolicies.nvidia.com "{name}" not found', 1)
        k.clusterpolicy = None
        strip_operator_labels(state)
        return f'clusterpolicy.nvidia.com "{name}" deleted'
    if kind == "nicclusterpolicies":
        if not k.get("nicclusterpolicy"):
            raise CommandError(f'Error from server (NotFound): nicclusterpolicies.mellanox.com "{name}" not found', 1)
        k.nicclusterpolicy = None
        return f'nicclusterpolicy.mellanox.com "{name}" deleted'
    if kind == "runtimeclasses":
        k.runtimeclasses = [r for r in k.runtimeclasses if r.name != name]
        return f'runtimeclass.node.k8s.io "{name}" deleted'
    if kind in EXTRA_KINDS:
        return EXTRA_KINDS[kind].delete(state, name)
    if kind == "nodes":
        raise CommandError("error: deleting nodes is not supported in this lab (use cordon/drain)", 1)
    raise CommandError(f"error: deleting {kind} is not supported in this lab", 1)


def _remove_pod(state, p):
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


# ====================================================================== patch


def merge_patch(target, patch, strategic=False):
    if not isinstance(patch, dict) or not isinstance(target, dict):
        return copy.deepcopy(patch)
    out = dict(target)
    for key, v in patch.items():
        if v is None:
            out.pop(key, None)
        elif isinstance(v, dict) and isinstance(out.get(key), dict):
            out[key] = merge_patch(out[key], v, strategic)
        elif (
            strategic
            and isinstance(v, list)
            and isinstance(out.get(key), list)
            and key in ("containers", "env", "volumes")
            and all(isinstance(x, dict) and "name" in x for x in v + out[key])
        ):
            merged = {x["name"]: x for x in out[key]}
            for x in v:
                merged[x["name"]] = merge_patch(merged.get(x["name"], {}), x, strategic)
            out[key] = list(merged.values())
        else:
            out[key] = copy.deepcopy(v)
    return out


def json_patch(doc, ops):
    doc = copy.deepcopy(doc)
    for op in ops:
        path = [p.replace("~1", "/").replace("~0", "~") for p in op["path"].split("/")[1:]]
        parent = doc
        for p in path[:-1]:
            parent = parent[int(p)] if isinstance(parent, list) else parent.setdefault(p, {})
        last = path[-1]
        kind = op["op"]
        if kind in ("add", "replace"):
            if isinstance(parent, list):
                if last == "-":
                    parent.append(op["value"])
                elif kind == "add":
                    parent.insert(int(last), op["value"])
                else:
                    parent[int(last)] = op["value"]
            else:
                if kind == "replace" and last not in parent:
                    raise CommandError(
                        "The request is invalid: the server rejected our request due to an error in our request", 1
                    )
                parent[last] = op["value"]
        elif kind == "remove":
            if isinstance(parent, list):
                parent.pop(int(last))
            elif last in parent:
                del parent[last]
            else:
                raise CommandError(
                    "The request is invalid: the server rejected our request due to an error in our request", 1
                )
    return doc


def cmd_patch(ctx, pos, opts):
    state = ctx.state
    k = state.k8s
    targets = split_targets(pos)
    kind, names = targets[0]
    if not names:
        raise CommandError("error: resource(s) were provided, but no name was specified", 1)
    name = names[0]
    raw = opts.get("patch")
    if not raw:
        raise CommandError("error: must specify -p to patch", 1)
    try:
        patch = json.loads(raw)
    except ValueError:
        try:
            patch = yaml.safe_load(raw)
        except yaml.YAMLError:
            raise CommandError(f'error: unable to parse "{raw}": yaml: invalid', 1)
    ptype = opts.get("type", "strategic")
    ns = ns_of(opts)
    items = list_objects(state, kind, opts, [name])
    obj = api_obj(state, kind, items[0])
    before = _norm(obj)
    if ptype == "json":
        new = json_patch(obj, patch)
    else:
        new = merge_patch(obj, patch, strategic=(ptype == "strategic"))
    if _norm(new) == before:
        return f"{KIND_PREFIX[kind]}/{name} patched (no change)\n"
    if kind == "nodes":
        kn = k.nodes[name]
        kn.labels = dict(new.get("metadata", {}).get("labels") or {})
        sp = new.get("spec") or {}
        kn.schedulable = not sp.get("unschedulable", False)
        kn.taints = [t for t in sp.get("taints") or [] if t.get("key") != "node.kubernetes.io/unschedulable"]
    elif kind == "pods":
        p = items[0]
        spec = new.get("spec") or {}
        spec.pop("nodeName", None)
        spec["tolerations"] = [t for t in spec.get("tolerations") or [] if t not in DEFAULT_POD_TOLERATIONS]
        for key_ in ("restartPolicy", "schedulerName", "serviceAccountName"):
            if key_ not in p.spec:
                spec.pop(key_, None)
        if not pod_update_allowed(p.spec, spec):
            raise CommandError(
                f'The Pod "{name}" is invalid: spec: Forbidden: pod updates may not change fields other than '
                "`spec.containers[*].image`,`spec.initContainers[*].image`,`spec.activeDeadlineSeconds`,"
                "`spec.tolerations` (only additions to existing tolerations),`spec.terminationGracePeriodSeconds` "
                "(allow it to be set to 1 if it was previously negative)",
                1,
            )
        p.spec = wrap(spec)
        p.labels = dict(new.get("metadata", {}).get("labels") or {})
    elif kind == "deployments":
        d = items[0]
        sp = new.get("spec") or {}
        validate_pod_spec("Deployment", name, (sp.get("template") or {}).get("spec") or {}, "spec.template.spec")
        d.replicas = int(sp.get("replicas", d.replicas))
        d.template = sp.get("template", d.template)
        d.generation = d.get("generation", 1) + 1
    elif kind == "clusterpolicies":
        k.clusterpolicy.spec = new.get("spec") or {}
    elif kind == "nicclusterpolicies":
        k.nicclusterpolicy.spec = new.get("spec") or {}
    elif kind == "serviceaccounts":
        k.serviceaccounts.setdefault(ns, {})[name] = {"imagePullSecrets": new.get("imagePullSecrets") or []}
    elif kind == "configmaps":
        items[0].data = {a: str(b) for a, b in (new.get("data") or {}).items()}
    elif kind == "daemonsets":
        d = items[0]
        tspec = ((new.get("spec") or {}).get("template") or {}).get("spec") or {}
        d.tolerations = tspec.get("tolerations") or []
        d.node_selector = tspec.get("nodeSelector") or {}
        if tspec.get("containers"):
            d.image = tspec["containers"][0].get("image", d.image)
    else:
        raise CommandError(f"error: patching {kind} is not supported in this lab", 1)
    return f"{KIND_PREFIX[kind]}/{name} patched\n"


# ====================================================================== label / taint / cordon / drain


def _target_nodes(state, pos, opts, verb):
    k = state.k8s
    if pos and "/" in pos[0]:
        names = [p.split("/", 1)[1] for p in pos if "/" in p]
        rest = [p for p in pos if "/" not in p]
    else:
        if not pos:
            raise CommandError("error: You must provide one or more resources by argument or filename.", 1)
        resolve_kind(pos[0])
        names = []
        rest = pos[1:]
        if opts.get("all"):
            names = sorted(k.nodes)
        elif opts.get("selector"):
            names = [n for n in sorted(k.nodes) if selector_matches(opts["selector"], k.nodes[n].labels)]
        else:
            while rest and "=" not in rest[0] and not rest[0].endswith("-") and ":" not in rest[0]:
                names.append(rest.pop(0))
    for n in names:
        if n not in k.nodes:
            raise CommandError(f'Error from server (NotFound): nodes "{n}" not found', 1)
    return names, rest


def cmd_label(state, pos, opts):
    k = state.k8s
    kind = resolve_kind(pos[0].split("/")[0]) if pos else None
    if kind not in ("nodes", "pods", "namespaces"):
        raise CommandError("error: this lab supports labeling nodes, pods and namespaces", 1)
    if kind == "nodes":
        names, changes = _target_nodes(state, pos, opts, "label")
        targets = [(f"node/{n}", k.nodes[n].labels) for n in names]
    elif kind == "pods":
        if "/" in pos[0]:
            names, changes = [pos[0].split("/", 1)[1]], pos[1:]
        else:
            names, changes = [pos[1]], pos[2:]
        targets = []
        for nm in names:
            p = next((p for p in k.pods if p.namespace == ns_of(opts) and p.name == nm), None)
            if p is None:
                raise CommandError(f'Error from server (NotFound): pods "{nm}" not found', 1)
            targets.append((f"pod/{nm}", p.labels))
    else:
        raise CommandError("error: namespace labels are not modelled in this lab", 1)
    if not changes:
        raise CommandError("error: at least one label update is required", 1)
    out = []
    for ref, labels in targets:
        changed, removed = False, False
        for ch in changes:
            if ch.endswith("-") and "=" not in ch:
                key = ch[:-1]
                if key in labels:
                    del labels[key]
                    removed = True
                continue
            if "=" not in ch:
                raise CommandError(f"error: invalid label spec: {ch}", 1)
            key, val = ch.split("=", 1)
            if (
                not re.fullmatch(r"([a-z0-9.-]+/)?[A-Za-z0-9]([-A-Za-z0-9_.]*[A-Za-z0-9])?", key)
                or len(val) > 63
                or (val and not re.fullmatch(r"[A-Za-z0-9]([-A-Za-z0-9_.]*[A-Za-z0-9])?", val))
            ):
                raise CommandError(
                    f'error: invalid label value: "{ch}": a valid label must be an empty string or consist of '
                    "alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character",
                    1,
                )
            if key in labels and labels[key] != val and not opts.get("overwrite"):
                raise CommandError(f"error: '{key}' already has a value ({labels[key]}), and --overwrite is false", 1)
            if labels.get(key) != val:
                labels[key] = val
                changed = True
        out.append(f"{ref} {'labeled' if changed else 'unlabeled' if removed else 'not labeled'}")
    return "\n".join(out) + "\n"


def cmd_taint(state, pos, opts):
    k = state.k8s
    names, specs = _target_nodes(state, pos, opts, "taint")
    if not specs:
        raise CommandError("error: at least one taint update is required", 1)
    out = []
    for n in names:
        kn = k.nodes[n]
        changed = False
        removed = False
        for s in specs:
            remove = s.endswith("-")
            s2 = s[:-1] if remove else s
            key, eff = s2, None
            if ":" in s2:
                key, eff = s2.rsplit(":", 1)
            val = ""
            if "=" in key:
                key, val = key.split("=", 1)
            if remove:
                before = len(kn.taints)
                kn.taints = [t for t in kn.taints if not (t.key == key and (eff is None or t.effect == eff))]
                if len(kn.taints) == before:
                    raise CommandError(f'error: taint "{s2}" not found', 1)
                removed = True
                continue
            if eff not in ("NoSchedule", "PreferNoSchedule", "NoExecute"):
                raise CommandError(f"error: invalid taint effect: {eff}, unsupported taint effect", 1)
            same = [t for t in kn.taints if t.key == key and t.effect == eff]
            if same and not opts.get("overwrite"):
                raise CommandError(
                    f"error: node {n} already has {key} taint(s) with same effect(s) and --overwrite is false", 1
                )
            kn.taints = [t for t in kn.taints if not (t.key == key and t.effect == eff)]
            kn.taints.append({"key": key, "value": val, "effect": eff})
            changed = True
            if eff == "NoExecute":
                for p in list(k.pods):
                    if p.node == n and not p.get("component") and not tolerates(pod_tolerations(p), kn.taints[-1]):
                        _remove_pod(state, p)
        out.append(f"node/{n} {'tainted' if changed else 'untainted' if removed else 'modified'}")
    return "\n".join(out) + "\n"


def cmd_cordon(state, pos, opts, on=True):
    k = state.k8s
    names = [p.split("/", 1)[-1] for p in pos] or (sorted(k.nodes) if opts.get("all") else [])
    if opts.get("selector"):
        names = [n for n in sorted(k.nodes) if selector_matches(opts["selector"], k.nodes[n].labels)]
    out = []
    for n in names:
        if n not in k.nodes:
            raise CommandError(f'Error from server (NotFound): nodes "{n}" not found', 1)
        kn = k.nodes[n]
        if kn.get("schedulable", True) == (not on):
            out.append(f"node/{n} already {'cordoned' if on else 'uncordoned'}")
            continue
        kn.schedulable = not on
        out.append(f"node/{n} {'cordoned' if on else 'uncordoned'}")
    return "\n".join(out) + "\n"


def cmd_drain(state, pos, opts):
    k = state.k8s
    names = [p.split("/", 1)[-1] for p in pos]
    out = []
    for n in names:
        if n not in k.nodes:
            raise CommandError(f'Error from server (NotFound): nodes "{n}" not found', 1)
        pods = [
            p
            for p in k.pods
            if p.node == n and p.phase not in ("Succeeded", "Failed") and p.get("component") != "static"
        ]
        ds_pods = [p for p in pods if p.get("owner") and p.owner.get("kind") == "DaemonSet"]
        bare = [p for p in pods if not p.get("owner")]
        errs = []
        if ds_pods and not opts.get("ignore-daemonsets"):
            errs.append(
                "cannot delete DaemonSet-managed Pods (use --ignore-daemonsets to ignore): "
                + ", ".join(f"{p.namespace}/{p.name}" for p in ds_pods)
            )
        if bare and not opts.get("force"):
            errs.append(
                "cannot delete Pods declare no controller (use --force to override): "
                + ", ".join(f"{p.namespace}/{p.name}" for p in bare)
            )
        lines = []
        if k.nodes[n].get("schedulable", True):
            k.nodes[n].schedulable = False
            lines.append(f"node/{n} cordoned")
        else:
            lines.append(f"node/{n} already cordoned")
        if errs:
            raise CommandError(
                "\n".join(
                    lines
                    + [
                        f'error: unable to drain node "{n}" due to error:'
                        + ", ".join(errs)
                        + ", continuing command...",
                        "There are pending nodes to be drained:",
                        f" {n}",
                    ]
                    + [f"error: {e}" for e in errs]
                ),
                1,
            )
        if ds_pods:
            lines.append(
                "Warning: ignoring DaemonSet-managed Pods: " + ", ".join(f"{p.namespace}/{p.name}" for p in ds_pods)
            )
        if bare:
            lines.append(
                "Warning: deleting Pods that declare no controller: "
                + ", ".join(f"{p.namespace}/{p.name}" for p in bare)
            )
        victims = [p for p in pods if p not in ds_pods]
        for p in victims:
            lines.append(f"evicting pod {p.namespace}/{p.name}")
        for p in victims:
            _remove_pod(state, p)
            lines.append(f"pod/{p.name} evicted")
        lines.append(f"node/{n} drained")
        out.extend(lines)
    return "\n".join(out) + "\n"


# ====================================================================== logs / exec / run


def find_pod(state, opts, ref):
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


def cmd_logs(state, pos, opts):
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


def cmd_exec(ctx, pos, opts, cmd):
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
            _FakeCtx(state, p.node),
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


def cmd_run(ctx, pos, opts, cmd):
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
    p = new_pod(state, ns, name, _with_sa_secrets(state, ns, spec), labels=labels)
    reconcile(state)
    if opts.get("rm") and opts.get("stdin"):
        out = (
            "\n".join(p.logs) + "\n"
            if p.status in ("Completed", "Failed")
            else ("error: timed out waiting for the condition\n" if p.node is None else "")
        )
        _remove_pod(state, p)
        return out + f'pod "{name}" deleted\n'
    return f"pod/{name} created\n"


# ====================================================================== create


def cmd_create(ctx, pos, opts):
    state = ctx.state
    k = state.k8s
    if opts.get("filename"):
        out = [apply_doc(ctx, src, d, mode="create") for src, d in load_docs(ctx, opts)]
        return "\n".join(out) + "\n"
    if not pos:
        raise CommandError("error: must specify one of -f and -k", 1)
    what = pos[0]
    ns = ns_of(opts)
    if what in ("namespace", "ns"):
        name = pos[1]
        if name in k.namespaces:
            raise CommandError(f'Error from server (AlreadyExists): namespaces "{name}" already exists', 1)
        ensure_namespace(state, name)
        return f"namespace/{name} created\n"
    if ns not in k.namespaces:
        raise CommandError(f'error: failed to create {what}: namespaces "{ns}" not found', 1)
    if what == "secret":
        stype, name = pos[1], pos[2] if len(pos) > 2 else None
        if not name:
            raise CommandError("error: exactly one NAME is required, got 0", 1)
        if any(s.namespace == ns and s.name == name for s in k.secrets):
            raise CommandError(f'error: failed to create secret secrets "{name}" already exists', 1)
        if stype == "docker-registry":
            server = opts.get("docker-server", "https://index.docker.io/v1/")
            user = opts.get("docker-username")
            pw = opts.get("docker-password")
            how = ct.unquoted_oauthtoken(ctx)
            if user == ct.OAUTH_USER and how in ("unquoted", "double"):
                user = ""
            if not user or not pw:
                raise CommandError(
                    "error: either --from-file or the combination of --docker-username, --docker-password "
                    "and --docker-server is required",
                    1,
                )
            import base64

            auth = base64.b64encode(f"{user}:{pw}".encode()).decode()
            ent = {"username": user, "password": pw, "auth": auth}
            if opts.get("docker-email"):
                ent["email"] = opts["docker-email"]
            data = {".dockerconfigjson": json.dumps({"auths": {server: ent}})}
            k.secrets.append(
                {
                    "namespace": ns,
                    "name": name,
                    "type": "kubernetes.io/dockerconfigjson",
                    "data": data,
                    "created": now(state),
                }
            )
            return f"secret/{name} created\n"
        if stype in ("generic", "tls"):
            data = {}
            for lit in opts.get("from-literal", []):
                a, _, b = lit.partition("=")
                data[a] = b
            for ff in opts.get("from-file", []):
                key, _, path = ff.rpartition("=")
                content = _read_manifest(ctx, path)
                data[key or path.rsplit("/", 1)[-1]] = content
            k.secrets.append(
                {"namespace": ns, "name": name, "type": opts.get("type", "Opaque"), "data": data, "created": now(state)}
            )
            return f"secret/{name} created\n"
        raise CommandError(f"error: unknown secret type {stype}", 1)
    if what in ("configmap", "cm"):
        name = pos[1]
        if any(c.namespace == ns and c.name == name for c in k.configmaps):
            raise CommandError(f'error: failed to create configmap: configmaps "{name}" already exists', 1)
        data = {}
        for lit in opts.get("from-literal", []):
            a, _, b = lit.partition("=")
            data[a] = b
        for ff in opts.get("from-file", []):
            key, _, path = ff.rpartition("=")
            data[key or path.rsplit("/", 1)[-1]] = _read_manifest(ctx, path)
        k.configmaps.append({"namespace": ns, "name": name, "data": data, "created": now(state)})
        return f"configmap/{name} created\n"
    if what in ("deployment", "deploy"):
        name = pos[1]
        image = opts.get("image")
        if not image:
            raise CommandError('error: required flag(s) "image" not set', 1)
        tmpl = {
            "metadata": {"labels": {"app": name}},
            "spec": {"containers": [{"name": image.rsplit("/", 1)[-1].split(":")[0], "image": image}]},
        }
        add_deployment(state, ns, name, image, replicas=int(opts.get("replicas", 1)), template=tmpl)
        return f"deployment.apps/{name} created\n"
    raise CommandError(f'error: unknown command "{what}" for "kubectl create"', 1)


# ====================================================================== rollout / scale / wait


def cmd_rollout(state, pos, opts):
    if len(pos) < 2:
        raise CommandError("error: required resource not specified", 1)
    action, ref = pos[0], pos[1] if "/" in pos[1] else f"{pos[1]}/{pos[2]}"
    kw, name = ref.split("/", 1)
    kind = resolve_kind(kw)
    ns = ns_of(opts)
    k = state.k8s
    if kind == "daemonsets":
        d = next((d for d in k.daemonsets if d.namespace == ns and d.name == name), None)
        if d is None:
            raise CommandError(f'Error from server (NotFound): daemonsets.apps "{name}" not found', 1)
        pods = pods_of(state, "DaemonSet", name, ns)
        if action == "restart":
            for p in list(pods):
                _remove_pod(state, p)
            return f"daemonset.apps/{name} restarted\n"
        if action == "status":
            reconcile(state)
            pods = pods_of(state, "DaemonSet", name, ns)
            ready = sum(1 for p in pods if p.status == "Running")
            want = len(ds_eligible(state, d))
            if ready == want:
                return f'daemon set "{name}" successfully rolled out\n'
            raise CommandError(
                f'Waiting for daemon set "{name}" rollout to finish: {ready} of {want} updated pods are available...\n'
                "error: timed out waiting for the condition",
                1,
            )
    if kind == "deployments":
        d = next((d for d in k.deployments if d.namespace == ns and d.name == name), None)
        if d is None:
            raise CommandError(f'Error from server (NotFound): deployments.apps "{name}" not found', 1)
        if action == "restart":
            tm = d.template.setdefault("metadata", {})
            ann = tm.setdefault("annotations", {})
            ann["kubectl.kubernetes.io/restartedAt"] = ts(now(state))
            return f"deployment.apps/{name} restarted\n"
        if action == "status":
            reconcile(state)
            st = deploy_obj(state, d)["status"]
            if st["readyReplicas"] == int(d.replicas):
                return f'deployment "{name}" successfully rolled out\n'
            raise CommandError(
                f'Waiting for deployment "{name}" rollout to finish: {st["readyReplicas"]} of {d.replicas} '
                "updated replicas are available...\nerror: timed out waiting for the condition",
                1,
            )
    raise CommandError(f"error: rollout {action} is not supported for {kind} in this lab", 1)


def cmd_scale(state, pos, opts):
    ref = pos[0] if "/" in pos[0] else f"{pos[0]}/{pos[1]}"
    kw, name = ref.split("/", 1)
    kind = resolve_kind(kw)
    if kind != "deployments" or "replicas" not in opts:
        raise CommandError("error: scale supports deployments with --replicas in this lab", 1)
    d = next((d for d in state.k8s.deployments if d.namespace == ns_of(opts) and d.name == name), None)
    if d is None:
        raise CommandError(f'Error from server (NotFound): deployments.apps "{name}" not found', 1)
    d.replicas = int(opts["replicas"])
    return f"deployment.apps/{name} scaled\n"


def cmd_wait(state, pos, opts):
    cond = opts.get("for", "")
    advance(state, 60)
    reconcile(state)
    out = []
    for kind, names in split_targets(pos):
        items = list_objects(state, kind, opts, names) if names else list_objects(state, kind, opts)
        for i in items:
            ok = False
            if kind == "pods":
                if cond.lower() in ("condition=ready", "condition=ready=true"):
                    ok = i.status == "Running"
                elif cond.lower().startswith("delete"):
                    ok = False
                else:
                    ok = i.status in ("Running", "Completed")
            elif kind == "nodes":
                ok = state.k8s.nodes[i.name].get("ready", True)
            else:
                ok = True
            if not ok:
                raise CommandError(f"error: timed out waiting for the condition on {kind}/{i.get('name')}", 1)
            out.append(f"{SINGULAR[kind]}/{i.get('name')} condition met")
    return "\n".join(out) + "\n"


# ====================================================================== entry point

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
def kubectl(ctx, args):
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
            if kind in ("clusterpolicies", "nicclusterpolicies") and not _crd_installed(state, kind):
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
                        _remove_pod(state, p)
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


# ====================================================================== scenario helpers


def apply_yaml(state, text, namespace=None):
    """Apply manifest text as if with `kubectl apply -f` (for scenario setup)."""
    from types import SimpleNamespace

    k8s_init(state)
    ctx = SimpleNamespace(state=state, _kns=namespace)
    out = [apply_doc(ctx, "manifest.yaml", d) for d in yaml.safe_load_all(text) if d]
    reconcile(state)
    return out
