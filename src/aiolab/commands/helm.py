"""Helm 3 emulation with the NVIDIA charts used on DGX clusters.

Repositories:  https://helm.ngc.nvidia.com/nvidia  (gpu-operator, network-operator, ...)
               https://nvidia.github.io/k8s-device-plugin  (nvidia-device-plugin, standalone)
Releases are stored in ``state.k8s.helm_releases`` as::

    {name, namespace, chart, repo, version, app_version, values (user-supplied),
     computed (chart defaults merged with user values), status, revision, updated, history}

Installing ``nvidia/gpu-operator`` creates the operator Deployment, NFD and the
``cluster-policy`` ClusterPolicy; the operator controller in kubectl.py then creates the
operand DaemonSets (driver, toolkit, device plugin, ...). Installing
``nvidia/network-operator`` creates the operator, NFD and (with deployCR=true) a NicClusterPolicy.
"""

from __future__ import annotations

import copy
import datetime
import json
import re

import yaml

from ..core.errors import CommandError
from ..core.state import wrap
from . import command
from . import kubectl as kc

NGC_URL = "https://helm.ngc.nvidia.com/nvidia"

REPOS = {
    NGC_URL: {
        "gpu-operator": ["v24.6.2", "v24.3.0", "v23.9.2"],
        "network-operator": ["24.4.1", "24.1.1", "23.10.0"],
        "k8s-nim-operator": ["v1.0.0"],
        "nvidia-device-plugin": [],
    },
    "https://nvidia.github.io/k8s-device-plugin": {
        "nvidia-device-plugin": ["0.16.2", "0.15.0"],
        "gpu-feature-discovery": ["0.8.2"],
    },
    "https://mellanox.github.io/network-operator": {"network-operator": ["24.4.1", "24.1.1"]},
}
CHART_DESC = {
    "gpu-operator": "NVIDIA GPU Operator creates/configures/manages GPUs atop Kubernetes",
    "network-operator": "Nvidia network operator",
    "k8s-nim-operator": "NVIDIA NIM Operator",
    "nvidia-device-plugin": "A Helm chart for the nvidia-device-plugin on Kubernetes",
    "gpu-feature-discovery": "A Helm chart for gpu-feature-discovery on Kubernetes",
}
DRIVER_FOR = {"v24.6.2": "550.90.07", "v24.3.0": "550.54.15", "v23.9.2": "535.129.03"}


# ====================================================================== chart defaults


def gpu_operator_defaults(version):
    return {
        "nfd": {"enabled": True},
        "operator": {"defaultRuntime": "containerd", "runtimeClass": "nvidia", "cleanupCRD": False},
        "mig": {"strategy": "single"},
        "driver": {
            "enabled": True,
            "repository": "nvcr.io/nvidia",
            "image": "driver",
            "version": DRIVER_FOR.get(version, "550.90.07"),
            "rdma": {"enabled": False, "useHostMofed": False},
            "upgradePolicy": {"autoUpgrade": True},
        },
        "toolkit": {
            "enabled": True,
            "repository": "nvcr.io/nvidia/k8s",
            "image": "container-toolkit",
            "version": "v1.16.1-ubuntu20.04",
        },
        "devicePlugin": {
            "enabled": True,
            "repository": "nvcr.io/nvidia",
            "image": "k8s-device-plugin",
            "version": "v0.16.2-ubi8",
            "config": {"create": False, "name": "", "default": "", "data": {}},
        },
        "dcgm": {"enabled": False},
        "dcgmExporter": {"enabled": True, "serviceMonitor": {"enabled": False}},
        "gfd": {"enabled": True},
        "migManager": {"enabled": True, "config": {"name": "default-mig-parted-config", "default": "all-disabled"}},
        "nodeStatusExporter": {"enabled": False},
        "validator": {"plugin": {"env": [{"name": "WITH_WORKLOAD", "value": "false"}]}},
        "cdi": {"enabled": False, "default": False},
        "daemonsets": {
            "priorityClassName": "system-node-critical",
            "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
        },
        "sandboxWorkloads": {"enabled": False},
    }


def network_operator_defaults(version):
    return {
        "nfd": {"enabled": True, "deployNodeFeatureRules": True},
        "sriovNetworkOperator": {"enabled": False},
        "deployCR": False,
        "ofedDriver": {
            "deploy": False,
            "repository": "nvcr.io/nvidia/mellanox",
            "image": "doca-driver",
            "version": "24.04-0.6.6.0-0",
        },
        "rdmaSharedDevicePlugin": {
            "deploy": True,
            "repository": "nvcr.io/nvidia/cloud-native",
            "image": "k8s-rdma-shared-dev-plugin",
            "version": "v1.4.0",
            "resources": [{"name": "rdma_shared_device_a", "vendors": ["15b3"], "rdmaHcaMax": 63}],
        },
        "sriovDevicePlugin": {"deploy": False},
        "secondaryNetwork": {"deploy": False},
        "nvIpam": {"deploy": False},
    }


def device_plugin_defaults(version):
    return {
        "migStrategy": "none",
        "failOnInitError": True,
        "runtimeClassName": None,
        "nfd": {"enabled": False},
        "gfd": {"enabled": False},
    }


DEFAULTS = {
    "gpu-operator": gpu_operator_defaults,
    "network-operator": network_operator_defaults,
    "nvidia-device-plugin": device_plugin_defaults,
}


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ====================================================================== --set parsing (helm strvals)


def _typed(v, as_string=False):
    if as_string:
        return v
    if v.startswith("{") and v.endswith("}"):
        return [_typed(x.strip()) for x in v[1:-1].split(",") if x.strip()]
    low = v.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "null":
        return None
    if re.fullmatch(r"-?\d+", v) and not (len(v) > 1 and v.startswith("0")):
        return int(v)
    return v


def _split_unescaped(s, sep):
    parts, buf, i, depth = [], "", 0, 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s):
            buf += s[i : i + 2]
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append(buf)
            buf = ""
        else:
            buf += ch
        i += 1
    parts.append(buf)
    return parts


def set_value(tree, path, value):
    keys = []
    for part in _split_unescaped(path, "."):
        part = part.replace("\\.", ".")
        m = re.fullmatch(r"(.*?)((\[\d+\])+)", part)
        if m:
            keys.append(m.group(1))
            keys.extend(int(x) for x in re.findall(r"\[(\d+)\]", m.group(2)))
        else:
            keys.append(part)
    cur = tree
    for i, key in enumerate(keys):
        last = i == len(keys) - 1
        nxt = None if last else ([] if isinstance(keys[i + 1], int) else {})
        if isinstance(key, int):
            while len(cur) <= key:
                cur.append({} if not isinstance(nxt, list) else [])
            if last:
                cur[key] = value
            else:
                if not isinstance(cur[key], type(nxt)):
                    cur[key] = nxt
                cur = cur[key]
        else:
            if last:
                cur[key] = value
            else:
                if not isinstance(cur.get(key), type(nxt)):
                    cur[key] = nxt
                cur = cur[key]


def parse_sets(specs, as_string=False):
    out = {}
    for spec in specs:
        for item in _split_unescaped(spec, ","):
            if not item:
                continue
            if "=" not in item:
                raise CommandError(f'Error: failed parsing --set data: key "{item}" has no value', 1)
            key, val = item.split("=", 1)
            set_value(out, key, _typed(val, as_string))
    return out


# ====================================================================== state helpers


def releases(state):
    return state.k8s.helm_releases


def find_release(state, name, ns):
    return next((r for r in releases(state) if r.name == name and r.namespace == ns), None)


def _now_str(state):
    t = kc.EPOCH + datetime.timedelta(seconds=kc.now(state))
    return f"{t:%Y-%m-%d %H:%M:%S}.{(kc.now(state) * 7919) % 1000000:06d}412 +0000 UTC"


def resolve_chart(state, ref, version=None, verb="INSTALLATION"):
    if "/" not in ref:
        raise CommandError(
            f"Error: {verb} FAILED: non-absolute URLs should be in form of repo_name/path_to_chart, got: {ref}", 1
        )
    repo, chart = ref.split("/", 1)
    url = state.k8s.helm_repos.get(repo)
    if url is None:
        raise CommandError(f"Error: {verb} FAILED: repo {repo} not found", 1)
    charts = REPOS.get(url.rstrip("/"), {})
    versions = charts.get(chart)
    if not versions:
        raise CommandError(
            f'Error: {verb} FAILED: chart "{chart}" matching {version or ""} not found in {repo} index. '
            f"(try 'helm repo update'): no chart name found",
            1,
        )
    if version:
        match = [v for v in versions if v.lstrip("v") == version.lstrip("v")]
        if not match:
            raise CommandError(
                f'Error: {verb} FAILED: chart "{chart}" matching {version} not found in {repo} index. '
                f"(try 'helm repo update'): no chart version found for {chart}-{version}",
                1,
            )
        return repo, chart, match[0]
    return repo, chart, versions[0]


def _load_values_files(ctx, files, verb):
    vals = {}
    for f in files:
        path = f if f.startswith("/") else f"{ctx.cwd.rstrip('/')}/{f}"
        text = ctx.node.files.get(path, ctx.node.files.get(f))
        if text is None:
            raise CommandError(f"Error: {verb} FAILED: open {f}: no such file or directory", 1)
        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError as e:
            raise CommandError(f"Error: {verb} FAILED: failed to parse {f}: error converting YAML to JSON: {e}", 1)
        vals = deep_merge(vals, data)
    return vals


# ====================================================================== chart effects


def _upsert_deployment(state, ns, name, image, component, release, replicas=1):
    k = state.k8s
    d = next((d for d in k.deployments if d.namespace == ns and d.name == name), None)
    if d:
        d.template.spec.containers[0].image = image
        return d
    tmpl = {
        "metadata": {"labels": {"app": name}},
        "spec": {"containers": [{"name": name.split("-node-feature")[0] if False else name, "image": image}]},
    }
    return kc.add_deployment(
        state, ns, name, image, replicas=replicas, component=component, release=release, template=tmpl
    )


def _nfd(state, ns, name, release, enabled):
    k = state.k8s
    names = [f"{name}-node-feature-discovery-master", f"{name}-node-feature-discovery-gc"]
    if not enabled:
        k.deployments = [d for d in k.deployments if not (d.get("release") == release and d.name in names)]
        k.daemonsets = [
            d
            for d in k.daemonsets
            if not (d.get("release") == release and d.name.endswith("node-feature-discovery-worker"))
        ]
        k.pods = [
            p for p in k.pods if not (p.get("release") == release and str(p.get("component", "")).startswith("nfd"))
        ]
        return
    _upsert_deployment(state, ns, names[0], "registry.k8s.io/nfd/node-feature-discovery:v0.16.3", "nfd-master", release)
    _upsert_deployment(state, ns, names[1], "registry.k8s.io/nfd/node-feature-discovery:v0.16.3", "nfd-gc", release)
    kc.add_daemonset(
        state,
        ns,
        f"{name}-node-feature-discovery-worker",
        "registry.k8s.io/nfd/node-feature-discovery:v0.16.3",
        "nfd-worker",
        release=release,
        container="worker",
        tolerations=[
            {"key": "node-role.kubernetes.io/master", "operator": "Equal", "effect": "NoSchedule"},
            {"key": "node-role.kubernetes.io/control-plane", "operator": "Equal", "effect": "NoSchedule"},
            {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
        ],
    )


def apply_chart(state, rel, previous=None):
    k = state.k8s
    key = f"{rel.namespace}/{rel.name}"
    vals = rel.computed
    ns = rel.namespace
    if rel.chart == "gpu-operator":
        cp = k.get("clusterpolicy")
        if cp and cp.get("release") not in (key, None):
            other = cp.release.split("/")[-1]
            raise CommandError(
                'Error: INSTALLATION FAILED: Unable to continue with install: ClusterPolicy "cluster-policy" '
                'in namespace "" exists and cannot be imported into the current release: invalid ownership '
                f'metadata; annotation validation error: key "meta.helm.sh/release-name" must equal "{rel.name}": '
                f'current value is "{other}"',
                1,
            )
        _upsert_deployment(state, ns, "gpu-operator", f"nvcr.io/nvidia/gpu-operator:{rel.version}", "gpu-operator", key)
        _nfd(state, ns, rel.name, key, kc.truthy((vals.get("nfd") or {}).get("enabled"), True))
        spec = {kk: copy.deepcopy(v) for kk, v in vals.items() if kk not in ("nfd", "node-feature-discovery")}
        for chart_only in ("create", "data"):  # chart-side values, not ClusterPolicy fields
            ((spec.get("devicePlugin") or {}).get("config") or {}).pop(chart_only, None)
        created = cp.created if cp else kc.now(state)
        k.clusterpolicy = {
            "name": "cluster-policy",
            "namespace": ns,
            "spec": spec,
            "created": created,
            "version": rel.version,
            "release": key,
        }
        if "nvidia" not in [r.name for r in k.runtimeclasses]:
            k.runtimeclasses.append({"name": "nvidia", "handler": "nvidia", "created": kc.now(state), "release": key})
        dpc = (vals.get("devicePlugin") or {}).get("config") or {}
        if kc.truthy(dpc.get("create"), False) and dpc.get("name"):
            # the chart renders devicePlugin.config.data into a ConfigMap owned by the release
            data = {
                a: b if isinstance(b, str) else yaml.safe_dump(b, sort_keys=False)
                for a, b in (dpc.get("data") or {}).items()
            }
            cm = next((c for c in k.configmaps if c.namespace == ns and c.name == dpc["name"]), None)
            if cm is None:
                k.configmaps.append(
                    {"namespace": ns, "name": dpc["name"], "data": data, "created": kc.now(state), "release": key}
                )
            else:
                cm.data = data
    elif rel.chart == "network-operator":
        _upsert_deployment(
            state,
            ns,
            "network-operator",
            f"nvcr.io/nvidia/cloud-native/network-operator:v{rel.version}",
            "network-operator",
            key,
        )
        _nfd(state, ns, rel.name, key, kc.truthy((vals.get("nfd") or {}).get("enabled"), True))
        ncp = k.get("nicclusterpolicy")
        if kc.truthy(vals.get("deployCR"), False):
            spec = {}
            if kc.truthy((vals.get("ofedDriver") or {}).get("deploy"), False):
                o = vals["ofedDriver"]
                spec["ofedDriver"] = {
                    "image": o.get("image", "doca-driver"),
                    "repository": o.get("repository"),
                    "version": o.get("version"),
                }
            r = vals.get("rdmaSharedDevicePlugin") or {}
            if kc.truthy(r.get("deploy"), True):
                cfg = {
                    "configList": [
                        {
                            "resourceName": x.get("name", "rdma_shared_device_a"),
                            "rdmaHcaMax": x.get("rdmaHcaMax", 63),
                            "selectors": {
                                k_: v for k_, v in x.items() if k_ in ("vendors", "deviceIDs", "ifNames", "drivers")
                            },
                        }
                        for x in r.get("resources") or []
                    ]
                }
                spec["rdmaSharedDevicePlugin"] = {
                    "image": r.get("image"),
                    "repository": r.get("repository"),
                    "version": r.get("version"),
                    "config": json.dumps(cfg),
                }
            if kc.truthy((vals.get("secondaryNetwork") or {}).get("deploy"), False):
                spec["secondaryNetwork"] = {"multus": {"image": "multus-cni"}, "cniPlugins": {"image": "plugins"}}
            k.nicclusterpolicy = {
                "name": "nic-cluster-policy",
                "namespace": ns,
                "spec": spec,
                "created": ncp.created if ncp else kc.now(state),
                "release": key,
            }
        elif ncp and ncp.get("release") == key:
            k.nicclusterpolicy = None
    elif rel.chart == "nvidia-device-plugin":
        rc = vals.get("runtimeClassName")
        kc.add_daemonset(
            state,
            ns,
            f"{rel.name}-nvidia-device-plugin",
            "nvcr.io/nvidia/k8s-device-plugin:v" + rel.version.lstrip("v"),
            "standalone-device-plugin",
            release=key,
            container="nvidia-device-plugin-ctr",
            tolerations=[
                {"key": "CriticalAddonsOnly", "operator": "Exists"},
                {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
            ],
            runtime_class=rc,
            extra={"mig_strategy": vals.get("migStrategy", "none")},
        )
    else:
        _upsert_deployment(state, ns, rel.name, f"nvcr.io/nvidia/{rel.chart}:{rel.version}", "generic", key)


def remove_chart(state, rel):
    k = state.k8s
    key = f"{rel.namespace}/{rel.name}"
    kc.remove_release(state, key)
    k.configmaps = [c for c in k.configmaps if c.get("release") != key]
    cp = k.get("clusterpolicy")
    if cp and cp.get("release") == key:
        toolkit_on = kc.operand_enabled(cp.spec, "toolkit")
        k.clusterpolicy = None
        k.daemonsets = [d for d in k.daemonsets if d.get("owner") != "clusterpolicy"]
        kc.sync_daemonsets(state)
        kc.strip_operator_labels(state)
        k.mig_applied = {}
        if toolkit_on:
            # the toolkit container removes the nvidia runtime from containerd when it shuts down
            from . import container as ct

            for n, sn in state.nodes.items():
                if sn.gpus:
                    sn.files[ct.CONTAINERD_CONF] = ct.containerd_config()
                    ct.reload_runtime(state, n, "containerd")
    ncp = k.get("nicclusterpolicy")
    if ncp and ncp.get("release") == key:
        k.nicclusterpolicy = None
        k.daemonsets = [d for d in k.daemonsets if d.get("owner") != "nicclusterpolicy"]


# ====================================================================== install / upgrade / uninstall


def install(
    state, name, chart_ref, namespace="default", values=None, version=None, create_namespace=False, verb="INSTALLATION"
):
    """Install a chart programmatically (used by the CLI and by scenario setup)."""
    kc.k8s_init(state)
    repo, chart, ver = resolve_chart(state, chart_ref, version, verb)
    if find_release(state, name, namespace):
        raise CommandError(f"Error: {verb} FAILED: cannot re-use a name that is still in use", 1)
    if namespace not in state.k8s.namespaces:
        if not create_namespace:
            raise CommandError(f'Error: {verb} FAILED: create: failed to create: namespaces "{namespace}" not found', 1)
        kc.ensure_namespace(state, namespace)
    user = values or {}
    computed = deep_merge(DEFAULTS.get(chart, lambda v: {})(ver), user)
    rel = wrap(
        {
            "name": name,
            "namespace": namespace,
            "chart": chart,
            "repo": repo,
            "version": ver,
            "app_version": ver,
            "values": user,
            "computed": computed,
            "status": "deployed",
            "revision": 1,
            "updated": kc.now(state),
            "history": [
                {
                    "revision": 1,
                    "updated": kc.now(state),
                    "status": "deployed",
                    "chart": f"{chart}-{ver}",
                    "description": "Install complete",
                    "values": copy.deepcopy(user),
                }
            ],
        }
    )
    apply_chart(state, rel)
    state.k8s.helm_releases.append(rel)
    kc.reconcile(state)
    return find_release(state, name, namespace)


def upgrade(state, rel, chart_ref, values, version=None, reuse=False, reset=False):
    repo, chart, ver = resolve_chart(state, chart_ref, version, "UPGRADE")
    if chart != rel.chart:
        pass
    if reuse:
        user = deep_merge(dict(rel["values"]), values or {})
    elif not values and not reset:
        user = copy.deepcopy(dict(rel["values"]))  # helm 3 reuses old values when none are given
    else:
        user = values or {}
    rel.chart, rel.repo, rel.version, rel.app_version = chart, repo, ver, ver
    rel["values"] = user
    rel.computed = deep_merge(DEFAULTS.get(chart, lambda v: {})(ver), user)
    rel.revision += 1
    rel.updated = kc.now(state)
    rel.status = "deployed"
    for h in rel.history:
        if h.status == "deployed":
            h.status = "superseded"
    rel.history.append(
        {
            "revision": rel.revision,
            "updated": kc.now(state),
            "status": "deployed",
            "chart": f"{chart}-{ver}",
            "description": "Upgrade complete",
            "values": copy.deepcopy(user),
        }
    )
    apply_chart(state, rel)
    kc.reconcile(state)
    return rel


def uninstall(state, rel):
    remove_chart(state, rel)
    state.k8s.helm_releases.remove(rel)
    kc.reconcile(state)


def preinstall(
    state, chart_ref="nvidia/gpu-operator", name=None, namespace=None, values=None, version=None, settle=1800
):
    """Scenario helper: add the NVIDIA repo, install a chart and let time pass."""
    kc.k8s_init(state)
    state.k8s.helm_repos.setdefault(chart_ref.split("/")[0], NGC_URL)
    chart = chart_ref.split("/", 1)[1]
    name = name or chart
    namespace = namespace or {"gpu-operator": "gpu-operator", "network-operator": "nvidia-network-operator"}.get(
        chart, "default"
    )
    rel = install(state, name, chart_ref, namespace, values or {}, version, create_namespace=True)
    settle_time(state, settle)
    return rel


def settle_time(state, seconds=600):
    kc.advance(state, seconds)
    kc.reconcile(state)


# ====================================================================== CLI

VAL_FLAGS = {
    "-n",
    "--namespace",
    "--set",
    "--set-string",
    "--set-file",
    "-f",
    "--values",
    "--version",
    "--timeout",
    "-o",
    "--output",
    "--description",
    "--repo",
    "--username",
    "--password",
    "--kube-context",
    "--post-renderer",
    "--revision",
    "--max",
}


def _parse(args):
    pos, opts = [], {"set": [], "set-string": [], "values": []}
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--") and "=" in a:
            key, val = a[2:].split("=", 1)
            if key in ("set", "set-string", "values"):
                opts[key].append(val)
            else:
                opts[key] = val
        elif a in VAL_FLAGS:
            key = {"-n": "namespace", "-f": "values", "-o": "output"}.get(a, a.lstrip("-"))
            val = args[i + 1] if i + 1 < len(args) else ""
            i += 1
            if key in ("set", "set-string", "values"):
                opts[key].append(val)
            else:
                opts[key] = val
        elif a.startswith("--"):
            opts[a[2:]] = True
        elif a.startswith("-") and len(a) > 1:
            for ch in a[1:]:
                opts[
                    {
                        "A": "all-namespaces",
                        "g": "generate-name",
                        "i": "install",
                        "a": "all",
                        "q": "short",
                        "l": "versions",
                    }.get(ch, ch)
                ] = True
        else:
            pos.append(a)
        i += 1
    return pos, opts


def _release_banner(rel, state, notes=""):
    out = [
        f"NAME: {rel.name}",
        f"LAST DEPLOYED: {(kc.EPOCH + datetime.timedelta(seconds=kc.now(state))):%a %b %d %H:%M:%S %Y}",
        f"NAMESPACE: {rel.namespace}",
        f"STATUS: {rel.status}",
        f"REVISION: {rel.revision}",
        "TEST SUITE: None",
    ]
    if notes:
        out += ["NOTES:", notes]
    return "\n".join(out) + "\n"


def _notes(chart, ns):
    if chart == "network-operator":
        return (
            f"Get Network Operator deployed resources by running the following commands:\n\n$ kubectl -n {ns} get pods"
        )
    return ""


def _user_values(ctx, opts, verb):
    vals = _load_values_files(ctx, opts["values"], verb)
    vals = deep_merge(vals, parse_sets(opts["set"]))
    vals = deep_merge(vals, parse_sets(opts["set-string"], as_string=True))
    return vals


def _helm_table(rows):
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("\t".join(str(c).ljust(widths[i]) for i, c in enumerate(r)) for r in rows) + "\n"


@command(
    "helm",
    help="helm repo add|update|list, install/upgrade (--set, -f, -n, --create-namespace, --version, --wait), list -A, uninstall, get values, show values",
)
def helm(ctx, args):
    state = ctx.state
    from ..core.state import HEAD

    if ctx.host != HEAD:
        raise CommandError(
            'Error: Kubernetes cluster unreachable: Get "http://localhost:8080/version": dial tcp 127.0.0.1:8080: '
            "connect: connection refused",
            1,
        )
    kc.k8s_init(state)
    if not args or args[0] in ("-h", "--help", "help"):
        return (
            "The Kubernetes package manager\n\nUsage:\n  helm [command]\n\nAvailable Commands:\n"
            "  get  history  install  list  repo  rollback  search  show  status  uninstall  upgrade  version\n"
        )
    verb, rest = args[0], args[1:]
    pos, opts = _parse(rest)
    ns = opts.get("namespace", "default")
    kc.advance(state)
    kc.reconcile(state)
    k = state.k8s
    if verb == "version":
        return 'version.BuildInfo{Version:"v3.15.2", GitCommit:"1a500d5625419a524fdae4b33de351cc4f58ec35", GitTreeState:"clean", GoVersion:"go1.22.4"}\n'
    if verb == "repo":
        sub = pos[0] if pos else ""
        if sub == "add":
            if len(pos) < 3:
                raise CommandError(
                    'Error: "helm repo add" requires 2 arguments\n\nUsage:  helm repo add [NAME] [URL] [flags]', 1
                )
            name, url = pos[1], pos[2].rstrip("/")
            if url not in REPOS:
                raise CommandError(
                    f'Error: looks like "{pos[2]}" is not a valid chart repository or cannot be reached: '
                    f"failed to fetch {url}/index.yaml : 404 Not Found",
                    1,
                )
            if name in k.helm_repos:
                if k.helm_repos[name].rstrip("/") == url:
                    return f'"{name}" already exists with the same configuration, skipping\n'
                if not opts.get("force-update"):
                    raise CommandError(
                        f"Error: repository name ({name}) already exists, please specify a different name", 1
                    )
            k.helm_repos[name] = url
            return f'"{name}" has been added to your repositories\n'
        if sub in ("list", "ls"):
            if not k.helm_repos:
                raise CommandError("Error: no repositories to show", 1)
            return _helm_table([("NAME", "URL")] + sorted(k.helm_repos.items()))
        if sub in ("update", "up"):
            if not k.helm_repos:
                raise CommandError("Error: no repositories found. You must add one before updating", 1)
            names = pos[1:] or sorted(k.helm_repos)
            out = ["Hang tight while we grab the latest from your chart repositories..."]
            for n in names:
                if n not in k.helm_repos:
                    raise CommandError(f"Error: no repositories found matching '{n}'.  Nothing will be updated", 1)
                out.append(f'...Successfully got an update from the "{n}" chart repository')
            out.append("Update Complete. ⎈Happy Helming!⎈")
            return "\n".join(out) + "\n"
        if sub in ("remove", "rm"):
            for n in pos[1:]:
                if k.helm_repos.pop(n, None) is None:
                    raise CommandError(f'Error: no repo named "{n}" found', 1)
            return "".join(f'"{n}" has been removed from your repositories\n' for n in pos[1:])
        raise CommandError(f'Error: unknown command "{sub}" for "helm repo"', 1)
    if verb == "search":
        if not pos or pos[0] != "repo":
            raise CommandError("Error: this lab supports `helm search repo [keyword]`", 1)
        kw = pos[1] if len(pos) > 1 else ""
        rows = [("NAME", "CHART VERSION", "APP VERSION", "DESCRIPTION")]
        for rname, url in sorted(k.helm_repos.items()):
            for chart, vers in sorted(REPOS.get(url, {}).items()):
                full = f"{rname}/{chart}"
                if kw and kw.lower() not in full.lower():
                    continue
                for v in vers if opts.get("versions") else vers[:1]:
                    rows.append((full, v, v, CHART_DESC.get(chart, "")))
        if len(rows) == 1:
            return "No results found\n"
        return _helm_table(rows)
    if verb in ("show", "inspect"):
        what = pos[0] if pos else ""
        if len(pos) < 2:
            raise CommandError(f'Error: "helm show {what}" requires 1 argument', 1)
        repo, chart, ver = resolve_chart(state, pos[1], opts.get("version"), "SHOW")
        if what == "values":
            return yaml.safe_dump(DEFAULTS.get(chart, lambda v: {})(ver), default_flow_style=False, sort_keys=False)
        if what == "chart":
            return f"apiVersion: v2\nappVersion: {ver}\ndescription: {CHART_DESC.get(chart, '')}\nname: {chart}\nversion: {ver}\n"
        raise CommandError(f'Error: unknown command "{what}" for "helm show"', 1)
    if verb in ("install", "upgrade"):
        vword = "INSTALLATION" if verb == "install" else "UPGRADE"
        if verb == "install":
            if opts.get("generate-name"):
                if not pos:
                    raise CommandError(
                        "Error: INSTALLATION FAILED: must either provide a name or specify --generate-name", 1
                    )
                chart_ref = pos[0]
                name = f"{chart_ref.split('/')[-1]}-{1792399800 + kc.now(state)}"
            else:
                if len(pos) < 2:
                    raise CommandError(
                        "Error: INSTALLATION FAILED: must either provide a name or specify --generate-name", 1
                    )
                name, chart_ref = pos[0], pos[1]
        else:
            if len(pos) < 2:
                raise CommandError(
                    'Error: "helm upgrade" requires 2 arguments\n\nUsage:  helm upgrade [RELEASE] [CHART] [flags]', 1
                )
            name, chart_ref = pos[0], pos[1]
        user = _user_values(ctx, opts, vword)
        if opts.get("dry-run"):
            repo, chart, ver = resolve_chart(state, chart_ref, opts.get("version"), vword)
            computed = deep_merge(DEFAULTS.get(chart, lambda v: {})(ver), user)
            return (
                f"NAME: {name}\nNAMESPACE: {ns}\nSTATUS: pending-{verb}\nREVISION: 1\nHOOKS:\nMANIFEST:\n---\n"
                f"# Source: {chart}/templates/clusterpolicy.yaml (computed values)\n"
                + yaml.safe_dump(computed, default_flow_style=False, sort_keys=False)
            )
        rel = find_release(state, name, ns)
        if verb == "upgrade":
            if rel is None:
                if not opts.get("install"):
                    raise CommandError(f'Error: UPGRADE FAILED: "{name}" has no deployed releases', 1)
                verb = "install"
            else:
                upgrade(
                    state,
                    rel,
                    chart_ref,
                    user,
                    opts.get("version"),
                    reuse=bool(opts.get("reuse-values")),
                    reset=bool(opts.get("reset-values")),
                )
                if opts.get("wait"):
                    settle_time(state, 90)
                head = f'Release "{name}" has been upgraded. Happy Helming!\n'
                return head + _release_banner(rel, state, _notes(rel.chart, ns))
        rel = install(state, name, chart_ref, ns, user, opts.get("version"), bool(opts.get("create-namespace")))
        if opts.get("wait"):
            settle_time(state, 90)
        prefix = f'Release "{name}" does not exist. Installing it now.\n' if args[0] == "upgrade" else ""
        return prefix + _release_banner(rel, state, _notes(rel.chart, ns))
    if verb in ("uninstall", "delete", "del", "un"):
        if not pos:
            raise CommandError('Error: "helm uninstall" requires at least 1 argument', 1)
        out = []
        for name in pos:
            rel = find_release(state, name, ns)
            if rel is None:
                raise CommandError(f"Error: uninstall: Release not loaded: {name}: release: not found", 1)
            uninstall(state, rel)
            out.append(f'release "{name}" uninstalled')
        return "\n".join(out) + "\n"
    if verb in ("list", "ls"):
        rels = [r for r in releases(state) if opts.get("all-namespaces") or r.namespace == ns]
        if opts.get("short"):
            return "".join(r.name + "\n" for r in rels)
        rows = [("NAME", "NAMESPACE", "REVISION", "UPDATED", "STATUS", "CHART", "APP VERSION")]
        for r in sorted(rels, key=lambda r: r.name):
            upd = kc.EPOCH + datetime.timedelta(seconds=r.updated)
            rows.append(
                (
                    r.name,
                    r.namespace,
                    r.revision,
                    f"{upd:%Y-%m-%d %H:%M:%S}.{(r.updated * 7919) % 1000000:06d}412 +0000 UTC",
                    r.status,
                    f"{r.chart}-{r.version}",
                    r.app_version,
                )
            )
        return _helm_table(rows)
    if verb == "get":
        what = pos[0] if pos else ""
        if len(pos) < 2:
            raise CommandError(f'Error: "helm get {what}" requires 1 argument', 1)
        rel = find_release(state, pos[1], ns)
        if rel is None:
            raise CommandError("Error: release: not found", 1)
        if what == "values":
            data = rel.computed if opts.get("all") else rel["values"]
            data = json.loads(json.dumps(data))
            if opts.get("output") == "json":
                return json.dumps(data) + "\n"
            head = "COMPUTED VALUES:\n" if opts.get("all") else "USER-SUPPLIED VALUES:\n"
            return head + (yaml.safe_dump(data, default_flow_style=False, sort_keys=True) if data else "null\n")
        if what in ("manifest", "all"):
            return (
                f"---\n# Source: {rel.chart}/templates/operator.yaml\napiVersion: apps/v1\nkind: Deployment\n"
                f"metadata:\n  name: {rel.chart}\n  namespace: {rel.namespace}\n"
            )
        if what == "notes":
            return "NOTES:\n" + _notes(rel.chart, ns) + "\n"
        raise CommandError(f'Error: unknown command "{what}" for "helm get"', 1)
    if verb == "status":
        rel = find_release(state, pos[0], ns) if pos else None
        if rel is None:
            raise CommandError("Error: release: not found", 1)
        return _release_banner(rel, state, _notes(rel.chart, ns))
    if verb == "history":
        rel = find_release(state, pos[0], ns) if pos else None
        if rel is None:
            raise CommandError("Error: release: not found", 1)
        rows = [("REVISION", "UPDATED", "STATUS", "CHART", "APP VERSION", "DESCRIPTION")]
        for h in rel.history:
            upd = kc.EPOCH + datetime.timedelta(seconds=h.updated)
            rows.append(
                (
                    h.revision,
                    f"{upd:%a %b %d %H:%M:%S %Y}",
                    h.status,
                    h.chart,
                    h.chart.rsplit("-", 1)[-1],
                    h.description,
                )
            )
        return _helm_table(rows)
    if verb == "rollback":
        rel = find_release(state, pos[0], ns) if pos else None
        if rel is None:
            raise CommandError("Error: release: not found", 1)
        target = int(pos[1]) if len(pos) > 1 else rel.revision - 1
        h = next((h for h in rel.history if h.revision == target), None)
        if h is None:
            raise CommandError(f"Error: release has no {target} version", 1)
        upgrade(state, rel, f"{rel.repo}/{rel.chart}", dict(h["values"]), h.chart.rsplit("-", 1)[-1], reset=True)
        rel.history[-1].description = f"Rollback to {target}"
        return "Rollback was a success! Happy Helming!\n"
    raise CommandError(f'Error: unknown command "{verb}" for "helm"\nRun \'helm --help\' for usage.', 1)
