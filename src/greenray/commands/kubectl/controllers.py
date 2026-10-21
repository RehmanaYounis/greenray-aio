"""Core controllers: node readiness, DaemonSet/Deployment/Job replication, NFD, Network Operator."""

from __future__ import annotations

import json

from ...core.state import HEAD
from .devices import device_plugin_log, device_plugin_nonvml_log, gpu_inventory
from .gpu_operator import operator_log
from .model import (
    DS_AUTO_TOLERATIONS,
    add_daemonset,
    event,
    klog,
    new_pod,
    pod_spec,
    pods_of,
    selector_matches,
    set_status,
    sfx,
    untolerated,
)


def refresh_nodes(state) -> None:
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


def ds_eligible(state, ds) -> list[str]:
    out = []
    for n, kn in sorted(state.k8s.nodes.items()):
        if not selector_matches([(kk, "=", [v]) for kk, v in ds.node_selector.items()], kn.labels):
            continue
        if untolerated(state, n, list(ds.tolerations) + DS_AUTO_TOLERATIONS):
            continue
        out.append(n)
    return out


def sync_daemonsets(state) -> None:
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
                pod_spec(ds.container, ds.image, extra),
                labels={"app": ds.name},
                owner={"kind": "DaemonSet", "name": ds.name},
                component=ds.component,
                node=n,
                created=ds.created if ds.created < 0 else None,
                release=ds.get("release"),
            )


def template_hash(template: dict) -> str:
    return sfx(json.dumps(template, sort_keys=True, default=str), n=10)


def sync_deployments(state) -> None:
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


def sync_jobs(state) -> None:
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


def nfd_sync(state) -> None:
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


def system_status(state, pod) -> None:
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


def cp_status_sync(state) -> None:
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


def network_operator_sync(state) -> None:
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
