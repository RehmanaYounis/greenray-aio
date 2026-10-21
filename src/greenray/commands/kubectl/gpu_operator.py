"""GPU Operator controller: operand DaemonSets, per-operand pod status, GFD labels and the MIG manager."""

from __future__ import annotations

import json

from .. import container as ct
from ..nvidia_smi import MIG_PROFILES, PROFILE_MEM_MIB
from .devices import (
    apply_sharing,
    device_plugin_log,
    dp_config_error_text,
    dp_refresh_config,
    dp_running_config,
    gpu_inventory,
    mig_devices,
)
from .model import add_daemonset, cp_spec, event, klog, new_pod, now, pod_spec, remove_pod, set_status, sfx, truthy, ts

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


def operand_enabled(spec: dict, key: str) -> bool:
    sect = spec.get(key) or {}
    return truthy(sect.get("enabled"), key not in OPERAND_DEFAULT_OFF)


def gpu_operator_sync(state) -> None:
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
            remove_pod(state, p)
    k["_dp_cfg_sig"] = sig
    if not any(rc.name == "nvidia" for rc in k.runtimeclasses):
        k.runtimeclasses.append(
            {"name": "nvidia", "handler": "nvidia", "created": cp.created, "release": cp.get("release")}
        )


def strip_operator_labels(state) -> None:
    for kn in state.k8s.nodes.values():
        for key in list(kn.labels):
            if key.startswith(OPERATOR_LABEL_PREFIXES) or key.startswith("nvidia.com/mig.config.state"):
                del kn.labels[key]


def _host_drv(state, n):
    return state.nodes[n].driver_loaded and not state.k8s._ctr_driver.get(n)


def node_gpu_conditions(state, n: str) -> tuple[bool, bool, bool]:
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


def gpu_operand_status(state, pod) -> None:
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


def cuda_validator_sync(state) -> None:
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
                pod_spec(
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


def gfd_sync(state) -> None:
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
                lab["nvidia.com/gpu.product"] = f"{product}-MIG-{prof}"
                lab["nvidia.com/gpu.count"] = str(len(devs))
                lab["nvidia.com/gpu.memory"] = str(PROFILE_MEM_MIB.get(prof, 0))
                lab["nvidia.com/gpu.multiprocessors"] = str(MIG_SM.get(prof, 0))
            elif strategy == "mixed":
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


def apply_mig_layout(node, layout: list[str] | None) -> None:
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


def mig_manager_sync(state) -> None:
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


def driver_crash_log(state, n: str) -> list[str]:
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


def driver_ok_log(state, n: str) -> list[str]:
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


def toolkit_log(state) -> list[str]:
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


def operator_log(state) -> list[str]:
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
