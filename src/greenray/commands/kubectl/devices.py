"""NVIDIA device plugin emulation: GPU/MIG inventory, time-slicing/MPS sharing config and RDMA resources."""

from __future__ import annotations

import json

import yaml

from .model import SUFFIX_GPU, cp_spec, klog, truthy


def mig_devices(g):
    return [(g.index, gi.id, gi.profile) for gi in g.gpu_instances if any(c.gi == gi.id for c in g.compute_instances)]


def gpu_inventory(state, n: str, strategy: str | None) -> tuple[dict, str | None]:
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


def device_plugin_sync(state) -> None:
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


def dp_desired_config(state, n: str) -> dict:
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


def dp_running_config(state, n: str) -> dict | None:
    """Sharing config the running operator device plugin on n actually uses (None = no sharing)."""
    dp = next(
        (p for p in state.k8s.pods if p.node == n and p.component == "devicePlugin" and p.status == "Running"), None
    )
    if dp is None or not dp.get("dp_cfg"):
        return None
    return dp.dp_cfg.get("cfg")


def dp_refresh_config(state, pod) -> dict:
    """Config-manager behaviour: re-read the config when the selected (ConfigMap, key) changes,
    when the pod (re)starts, or while the previous read failed. Editing the data of the same
    key in place is NOT picked up until the device-plugin pods are restarted."""
    want = dp_desired_config(state, pod.node)
    cur = pod.get("dp_cfg")
    if cur is None or list(cur.get("source") or []) != list(want["source"]) or cur.get("err"):
        pod["dp_cfg"] = want
    return pod.dp_cfg


def apply_sharing(inv: dict, cfg: dict | None) -> dict:
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


def dp_config_error_text(state, pod, info: dict) -> str:
    cp = state.k8s.clusterpolicy
    name, key = info["source"]
    if info["err"] == "missing-key":
        return f'unable to get config: specified config "{key}" does not exist in ConfigMap {cp.namespace}/{name}'
    return f"unable to load config: {info['err']}"


def rdma_sync(state) -> None:
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


def device_plugin_log(
    state, n: str, strategy: str, inv: dict, err: str | None, sharing: dict | None = None
) -> list[str]:
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


def device_plugin_nonvml_log(state, crash: bool = True) -> list[str]:
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
