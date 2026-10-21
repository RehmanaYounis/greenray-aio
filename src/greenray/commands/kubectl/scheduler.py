"""kube-scheduler and kubelet emulation plus ``reconcile``, the single pass that plays every controller."""

from __future__ import annotations

import base64
import json
import shlex

from ...core.errors import CommandError
from ...core.state import HEAD
from .. import HOOKS_REBOOT, HOOKS_SERVICE
from .. import container as ct
from .controllers import (
    cp_status_sync,
    network_operator_sync,
    nfd_sync,
    refresh_nodes,
    sync_daemonsets,
    sync_deployments,
    sync_jobs,
    system_status,
)
from .devices import apply_sharing, device_plugin_sync, dp_running_config, gpu_inventory, rdma_sync
from .gpu_operator import cuda_validator_sync, gfd_sync, gpu_operand_status, gpu_operator_sync, mig_manager_sync
from .model import (
    K8S_VERSION,
    RECONCILE_HOOKS,
    SUFFIX_GPU,
    add_daemonset,
    add_deployment,
    cp_spec,
    event,
    new_pod,
    node_selector_ok,
    now,
    pod_spec,
    pod_tolerations,
    qty,
    set_status,
    tolerates,
    untolerated,
)


def k8s_init(state):
    """Lay out ``state.k8s`` (control plane, CNI, node defaults) once and run a first reconcile."""
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
            pod_spec(
                comp, f"registry.k8s.io/{comp}:{K8S_VERSION}" if comp != "etcd" else "registry.k8s.io/etcd:3.5.12-0"
            ),
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


def node_used(state, n: str, exclude=None) -> dict:
    used = {}
    for p in state.k8s.pods:
        if p.node != n or p is exclude or p.phase in ("Succeeded", "Failed"):
            continue
        for r, v in (p.get("requests") or {}).items():
            used[r] = used.get(r, 0) + v
    return used


def free_of(state, n: str, res: str) -> int | float:
    kn = state.k8s.nodes[n]
    return qty(kn.allocatable.get(res, 0)) - node_used(state, n).get(res, 0)


def schedule(state) -> None:
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


def bind(state, pod, n: str) -> None:
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


def node_strategy(state, n: str) -> str:
    k = state.k8s
    dp = next((p for p in k.pods if p.node == n and p.component in ("devicePlugin", "standalone-device-plugin")), None)
    if dp is None:
        return "none"
    if dp.component == "devicePlugin":
        return (cp_spec(state).get("mig") or {}).get("strategy", "single")
    return next((d.get("mig_strategy", "none") for d in k.daemonsets if d.name == dp.owner.name), "none")


def allocate_devices(state, pod, n: str) -> list:
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


def pull_creds(state, pod, registry: str) -> tuple[tuple | None, list[str]]:
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
                user, _, pw = base64.b64decode(ent["auth"]).decode().partition(":")
            creds = (user, pw)
    return creds, missing


def kubelet_sync(state) -> None:
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


def visible_devices(state, pod) -> list:
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


def user_workload(state, pod) -> tuple[str, list[str]]:
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
                PodCtx(state, pod.node),
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


class PodCtx:
    """Stand-in shell context for ``container.exec_in_container`` when the caller is a pod, not a login."""

    def __init__(self, state, host):
        self.state, self.host, self.stdin, self.history, self.executed = state, host, "", [], []

    @property
    def node(self):
        return self.state.nodes[self.host]


def reconcile(state) -> None:
    """Play every controller, the scheduler and the kubelet until the cluster settles (3 passes suffice)."""
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
