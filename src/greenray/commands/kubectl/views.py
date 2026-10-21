"""API object views: the YAML/JSON representation ``kubectl get -o`` renders for each kind."""

from __future__ import annotations

import base64
import copy
import json

from .controllers import ds_eligible
from .model import CRI, DEFAULT_POD_TOLERATIONS, K8S_VERSION, now, pod_restarts, pods_of, ts


def node_obj(state, n: str) -> dict:
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


def pod_obj(state, p) -> dict:
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


def ds_obj(state, d) -> dict:
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


def deploy_obj(state, d) -> dict:
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


def job_obj(state, j) -> dict:
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


def cp_obj(state) -> dict:
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


def ncp_obj(state) -> dict:
    ncp = state.k8s.nicclusterpolicy
    return {
        "apiVersion": "mellanox.com/v1alpha1",
        "kind": "NicClusterPolicy",
        "metadata": {"name": ncp.name, "creationTimestamp": ts(ncp.created)},
        "spec": json.loads(json.dumps(ncp.spec)),
        "status": json.loads(json.dumps(ncp.get("status") or {})),
    }


def secret_obj(state, s) -> dict:
    data = {k_: base64.b64encode(str(v).encode()).decode() for k_, v in (s.get("data") or {}).items()}
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "type": s.get("type", "Opaque"),
        "metadata": {"name": s.name, "namespace": s.namespace, "creationTimestamp": ts(s.created)},
        "data": data,
    }
