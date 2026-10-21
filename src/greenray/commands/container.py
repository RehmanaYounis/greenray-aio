"""Container tooling on DGX / BCM nodes: docker, ngc, enroot, nvidia-ctk, crictl.

Registry model (nvcr.io):
  nvcr.io/nvidia/...         public NGC catalog (state.ngc.catalog). Anonymous pulls work.
  nvcr.io/<org>/[<team>/]..  private org/team registry (state.ngc.private_repos, set by
                             scenarios). Needs username ``$oauthtoken`` and a valid NGC API key.
An API key is valid if it is listed in ``state.ngc.valid_api_keys``. If a scenario does not
set that list, any key that looks like an NGC key (``nvapi-...``) is accepted.

Container runtime model:
  The *file* /etc/containerd/config.toml (or /etc/docker/daemon.json) is what you edit.
  ``node.runtime_loaded`` is what the running daemon actually uses. It is refreshed only
  when the daemon is (re)started, so editing the config without a restart does nothing.
"""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import json
import re
from types import SimpleNamespace

from ..core.errors import CommandError
from ..core.state import AttrDict, AttrList, wrap
from . import HOOKS_REBOOT, HOOKS_SERVICE, command

CONTAINERD_CONF = "/etc/containerd/config.toml"
DOCKER_CONF = "/etc/docker/daemon.json"
DOCKER_AUTH = "/root/.docker/config.json"
ENROOT_CREDS = "/root/.config/enroot/.credentials"
NGC_CONFIG = "/root/.ngc/config"
CDI_SPEC = "/etc/cdi/nvidia.yaml"
OAUTH_USER = "$oauthtoken"
CTK_VERSION = "1.16.1"

EXTRA_CATALOG = {
    "nvidia/k8s/cuda-sample": ["vectoradd-cuda12.5.0", "nbody-cuda12.5.0"],
    "nvidia/k8s-device-plugin": ["v0.16.2", "v0.15.0"],
    "nvidia/cuda": ["12.5.0-base-ubuntu22.04"],
    "nvidia/cloud-native/k8s-rdma-shared-dev-plugin": ["v1.4.0"],
    "nvidia/mellanox/rping-test": ["latest"],
}

IMAGE_SIZES = {
    "pytorch": "20.5GB",
    "tensorflow": "17.9GB",
    "tritonserver": "15.2GB",
    "nemo": "31.7GB",
    "cuda": "243MB",
    "cuda-sample": "124MB",
    "nemo-finetune": "33.1GB",
}


# ====================================================================== runtime config


def containerd_config(nvidia: bool = False, default: str = "runc") -> str:
    """Render a containerd v2 config.toml like the stock one / the one nvidia-ctk writes."""
    lines = [
        "version = 2",
        "",
        "[plugins]",
        "",
        '  [plugins."io.containerd.grpc.v1.cri"]',
        '    sandbox_image = "registry.k8s.io/pause:3.9"',
        "",
        '    [plugins."io.containerd.grpc.v1.cri".containerd]',
        f'      default_runtime_name = "{default}"',
        "",
        '      [plugins."io.containerd.grpc.v1.cri".containerd.runtimes]',
        "",
        '        [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc]',
        '          runtime_type = "io.containerd.runc.v2"',
        "",
        '          [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc.options]',
        "            SystemdCgroup = true",
    ]
    if nvidia:
        lines += [
            "",
            '        [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia]',
            "          privileged_without_host_devices = false",
            '          runtime_engine = ""',
            '          runtime_root = ""',
            '          runtime_type = "io.containerd.runc.v2"',
            "",
            '          [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia.options]',
            '            BinaryName = "/usr/bin/nvidia-container-runtime"',
            "            SystemdCgroup = true",
        ]
    return "\n".join(lines) + "\n"


def parse_containerd(text: str) -> dict:
    handlers = []
    for m in re.finditer(r"containerd\.runtimes\.([A-Za-z0-9_-]+)\]", text or ""):
        if m.group(1) not in handlers:
            handlers.append(m.group(1))
    m = re.search(r'default_runtime_name\s*=\s*"([^"]*)"', text or "")
    default = m.group(1) if m else "runc"
    if "runc" not in handlers:
        handlers.insert(0, "runc")
    return {"handlers": handlers, "default": default or "runc"}


def go_json_error(text: str, err: ValueError) -> str:
    """Translate a Python JSON error into the message dockerd (Go encoding/json) prints."""
    msg = getattr(err, "msg", str(err))
    pos = getattr(err, "pos", 0) or 0
    ch = (text or "")[pos : pos + 1]
    if "Illegal trailing comma" in msg:  # Python 3.13+ wording; Go reports the character after the comma
        rest = (text or "")[pos + 1 :].lstrip()
        ch = rest[:1]
        if ch == "]":
            return "invalid character ']' looking for beginning of value"
        return (
            f"invalid character '{ch}' looking for beginning of object key string"
            if ch
            else "unexpected end of JSON input"
        )
    if "Expecting property name" in msg:
        return (
            f"invalid character '{ch}' looking for beginning of object key string"
            if ch
            else "unexpected end of JSON input"
        )
    if "Expecting ',' delimiter" in msg:
        return f"invalid character '{ch}' after object key:value pair" if ch else "unexpected end of JSON input"
    if "Expecting ':' delimiter" in msg:
        return f"invalid character '{ch}' after object key" if ch else "unexpected end of JSON input"
    if "Expecting value" in msg:
        return f"invalid character '{ch}' looking for beginning of value" if ch else "unexpected end of JSON input"
    if "Extra data" in msg:
        return f"invalid character '{ch}' after top-level value"
    return msg


def parse_docker(text: str):
    """Return (runtimes, default, error)."""
    try:
        data = json.loads(text) if (text or "").strip() else {}
    except ValueError as e:
        return ["runc", "io.containerd.runc.v2"], "runc", go_json_error(text, e)
    if not isinstance(data, dict):
        return ["runc", "io.containerd.runc.v2"], "runc", "json: cannot unmarshal into Go value of type config.Config"
    runtimes = ["io.containerd.runc.v2", "runc"] + [r for r in data.get("runtimes", {}) if r not in ("runc",)]
    default = data.get("default-runtime", "runc")
    if default not in runtimes:
        return runtimes, default, f"specified default runtime '{default}' does not exist"
    return runtimes, default, None


def runtime_init(state):
    """Lazily give every node a containerd config and a 'loaded' runtime snapshot."""
    for _name, node in state.nodes.items():
        node.files.setdefault(CONTAINERD_CONF, containerd_config())
        if "runtime_loaded" not in node:
            rt, dflt, _ = parse_docker(node.files.get(DOCKER_CONF, ""))
            node.runtime_loaded = {
                "containerd": parse_containerd(node.files[CONTAINERD_CONF]),
                # gpu_driver: dockerd registers its built-in "nvidia" device driver (used by --gpus)
                # only if nvidia-container-runtime-hook was installed when the daemon started
                "docker": {"handlers": rt, "default": dflt, "gpu_driver": toolkit_installed(node)},
            }


def reload_runtime(state, host: str, service: str):
    node = state.nodes[host]
    runtime_init(state)
    if service == "containerd":
        node.runtime_loaded.containerd = parse_containerd(node.files.get(CONTAINERD_CONF, ""))
    elif service == "docker":
        rt, dflt, err = parse_docker(node.files.get(DOCKER_CONF, ""))
        j = node.journal.setdefault("docker", [])
        if err:
            node.services["docker"] = "failed"
            j.append(f"Oct 19 10:31:07 {host} systemd[1]: Starting Docker Application Container Engine...")
            prefix = (
                "failed to start daemon"
                if err.startswith("specified default runtime")
                else f"unable to configure the Docker daemon with file {DOCKER_CONF}"
            )
            j.append(f"Oct 19 10:31:07 {host} dockerd[48812]: {prefix}: {err}")
            j.append(
                f"Oct 19 10:31:07 {host} systemd[1]: docker.service: Main process exited, code=exited, status=1/FAILURE"
            )
            j.append(f"Oct 19 10:31:07 {host} systemd[1]: docker.service: Failed with result 'exit-code'.")
            j.append(f"Oct 19 10:31:07 {host} systemd[1]: Failed to start Docker Application Container Engine.")
            for c in node.containers:
                if c.status.startswith("Up"):
                    c.status = "Exited (255) 3 seconds ago"
            return
        node.runtime_loaded.docker = {"handlers": rt, "default": dflt, "gpu_driver": toolkit_installed(node)}
        j.append(f"Oct 19 10:31:09 {host} systemd[1]: Starting Docker Application Container Engine...")
        j.append(f'Oct 19 10:31:09 {host} dockerd[49117]: time="2026-10-19T10:31:09Z" level=info msg="Starting up"')
        for c in node.containers:
            if c.status.startswith("Up"):
                c.status = "Exited (0) 1 second ago"  # the daemon restart stopped it
            if c.get("restart") in ("always", "unless-stopped") and not c.get("manually_stopped"):
                err = start_container(state, host, c)
                if err:
                    j.append(
                        f'Oct 19 10:31:10 {host} dockerd[49117]: time="2026-10-19T10:31:10Z" level=error '
                        f'msg="failed to start container" container={c.id} error="{err}"'
                    )
        j.append(
            f'Oct 19 10:31:10 {host} dockerd[49117]: time="2026-10-19T10:31:10Z" level=info msg="Daemon has completed '
            'initialization"'
        )
        j.append(f"Oct 19 10:31:10 {host} systemd[1]: Started Docker Application Container Engine.")


def containerd_has(state, host, handler="nvidia") -> bool:
    runtime_init(state)
    return handler in state.nodes[host].runtime_loaded.containerd.handlers


def containerd_default(state, host) -> str:
    runtime_init(state)
    return state.nodes[host].runtime_loaded.containerd.default


def toolkit_installed(node) -> bool:
    """NVIDIA Container Toolkit packages present (``ctk_installed`` is set by apt / scenario setup)."""
    return bool(node.get("ctk_installed", "/etc/nvidia-container-runtime/config.toml" in node.files))


def _on_service(ctx, name, action):
    if name in ("containerd", "docker") and action in ("start", "restart", "reload"):
        reload_runtime(ctx.state, ctx.host, name)


def _on_reboot(ctx, node):
    runtime_init(ctx.state)
    host = node.hostname
    for svc in ("containerd", "docker"):
        reload_runtime(ctx.state, host, svc)


HOOKS_SERVICE.append(_on_service)
HOOKS_REBOOT.append(_on_reboot)


# ====================================================================== registry / credentials


def container_init(state):
    ngc = state.ngc
    for repo, tags in EXTRA_CATALOG.items():
        cur = ngc.catalog.setdefault(repo, [])
        for t in tags:
            if t not in cur:
                cur.append(t)
    ngc.setdefault("private_repos", {})
    ngc.setdefault("valid_api_keys", [])
    ngc.setdefault("format_type", "ascii")
    ngc.setdefault("ace", "no-ace")
    runtime_init(state)


def parse_image(ref: str):
    """-> (registry, repo, tag). docker.io short names get library/."""
    ref = ref.split("@", 1)[0]
    parts = ref.split("/")
    if len(parts) > 1 and ("." in parts[0] or ":" in parts[0] or parts[0] == "localhost"):
        registry, rest = parts[0], "/".join(parts[1:])
    else:
        registry, rest = "docker.io", ref
    tag = "latest"
    last = rest.rsplit("/", 1)[-1]
    if ":" in last:
        rest, tag = rest.rsplit(":", 1)
    if registry == "docker.io" and "/" not in rest:
        rest = "library/" + rest
    return registry, rest, tag


def norm_image(ref: str) -> str:
    reg, repo, tag = parse_image(ref)
    if reg == "docker.io":
        repo = repo.removeprefix("library/")
        return f"{repo}:{tag}"
    return f"{reg}/{repo}:{tag}"


def key_valid(state, key: str) -> bool:
    if not key or key.startswith("$"):
        return False
    keys = state.ngc.get("valid_api_keys") or []
    if keys:
        return key in keys
    return bool(re.fullmatch(r"nvapi-[A-Za-z0-9_-]{20,}", key) or re.fullmatch(r"[A-Za-z0-9+/=]{60,}", key))


def creds_valid(state, user: str, password: str) -> bool:
    return user == OAUTH_USER and key_valid(state, password)


def registry_check(state, ref: str, creds=None):
    """Return None if the image can be pulled with ``creds`` (user, password) else an error kind."""
    container_init(state)
    reg, repo, tag = parse_image(ref)
    if reg != "nvcr.io":
        return None
    authed = bool(creds) and creds_valid(state, *creds)
    if repo in state.ngc.catalog:
        return None if tag in state.ngc.catalog[repo] else "manifest_unknown"
    if repo in state.ngc.private_repos:
        if not authed:
            return "unauthorized"
        return None if tag in state.ngc.private_repos[repo] else "manifest_unknown"
    return "not_found" if authed else "unauthorized"


def docker_creds(node, registry="nvcr.io"):
    try:
        auths = json.loads(node.files.get(DOCKER_AUTH, "") or "{}").get("auths", {})
    except ValueError:
        return None
    for key, ent in auths.items():
        if key.replace("https://", "").rstrip("/") == registry and ent.get("auth"):
            user, _, pw = base64.b64decode(ent["auth"]).decode().partition(":")
            return user, pw
    return None


def unquoted_oauthtoken(ctx) -> str | None:
    """Mimic bash: was ``$oauthtoken`` on the raw line left unprotected by single quotes?

    The lab shell keeps unknown ``$vars`` literally, but real bash expands an unset
    ``$oauthtoken`` to an empty string unless it is in single quotes or escaped.
    Returns 'unquoted', 'double' or None.
    """
    line = ctx.history[-1] if ctx.history else ""
    i, quote = 0, None
    while i < len(line):
        ch = line[i]
        if ch == "\\" and quote != "'":
            i += 2
            continue
        if quote:
            if ch == quote:
                quote = None
            elif ch == "$" and quote == '"' and line.startswith("$oauthtoken", i):
                return "double"
        elif ch in "'\"":
            quote = ch
        elif ch == "$" and line.startswith("$oauthtoken", i):
            return "unquoted"
        i += 1
    return None


def pull_error_text(kind, ref, tool="docker"):
    reg, repo, tag = parse_image(ref)
    if tool == "docker":
        return {
            "unauthorized": f'Error response from daemon: Head "https://{reg}/v2/{repo}/manifests/{tag}": unauthorized: authentication required',
            "not_found": f"Error response from daemon: pull access denied for {reg}/{repo}, repository does not exist or may require 'docker login': denied: requested access to the resource is denied",
            "manifest_unknown": f"Error response from daemon: manifest for {norm_image(ref)} not found: manifest unknown: manifest unknown",
        }[kind]
    # containerd / kubelet wording
    full = norm_image(ref) if reg != "docker.io" else f"docker.io/{repo}:{tag}"
    base = f'failed to pull and unpack image "{full}": failed to resolve reference "{full}"'
    return {
        "unauthorized": f"{base}: pulling from host {reg} failed with status code [manifests {tag}]: 401 Unauthorized",
        "not_found": f"{base}: {full}: not found",
        "manifest_unknown": f"{base}: {full}: not found",
    }[kind]


# ====================================================================== in-container execution


def gpu_proxy_node(node, devices):
    """Build a node-like object exposing only ``devices`` = [(gpu_index, gi_id|None)], renumbered from 0."""
    proxy = AttrDict(dict(node))
    gpus = AttrList()
    by_gpu = {}
    for idx, gi in devices:
        by_gpu.setdefault(idx, []).append(gi)
    for new_idx, (idx, gis) in enumerate(sorted(by_gpu.items())):
        g = node.gpus[idx]
        pg = AttrDict(dict(g))
        pg["index"] = new_idx
        pg["processes"] = AttrList()
        if g.mig_mode and any(x is not None for x in gis):
            pg["gpu_instances"] = wrap([i for i in g.gpu_instances if i.id in gis])
            pg["compute_instances"] = wrap([c for c in g.compute_instances if c.gi in gis])
        gpus.append(pg)
    proxy["gpus"] = gpus
    return proxy


def exec_in_container(ctx, node, devices, argv, image="", hostname="container", env=None, rdma=False):
    """Run a command 'inside' a container that sees ``devices``. Returns output or raises."""
    from .nvidia_smi import nvidia_smi

    if not argv:
        return ""
    cmd = argv[0].rsplit("/", 1)[-1]
    env = dict(env or {})
    if cmd in ("sh", "bash") and len(argv) >= 3 and argv[1] == "-c":
        import shlex

        try:
            return exec_in_container(ctx, node, devices, shlex.split(argv[2]), image, hostname, env, rdma)
        except ValueError:
            return ""
    if cmd == "nvidia-smi":
        if not devices:
            raise CommandError(
                "OCI runtime exec failed: exec failed: unable to start container process: "
                'exec: "nvidia-smi": executable file not found in $PATH: unknown',
                127,
            )
        sub = SimpleNamespace(
            node=gpu_proxy_node(node, devices), state=ctx.state, host=ctx.host, stdin="", history=[], executed=[]
        )
        return nvidia_smi(sub, argv[1:])
    if cmd in ("python", "python3"):
        code = " ".join(argv[1:])
        n = len({d[0] for d in devices}) if not any(d[1] is not None for d in devices) else len(devices)
        out = []
        if "device_count" in code:
            out.append(str(n))
        if "is_available" in code:
            out.append("True" if n else "False")
        if "get_device_name" in code and n:
            out.append(node.gpus[devices[0][0]].name + (" MIG 1g.10gb" if devices[0][1] is not None else ""))
        if not out:
            if "torch" in code and "pytorch" not in image and "nemo" not in image:
                raise CommandError(
                    'Traceback (most recent call last):\n  File "<string>", line 1, in <module>\n'
                    "ModuleNotFoundError: No module named 'torch'",
                    1,
                )
            out.append("")
        return "\n".join(out) + "\n"
    if cmd == "nvcc":
        return (
            "nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c) 2005-2024 NVIDIA Corporation\n"
            "Cuda compilation tools, release 12.4, V12.4.131\nBuild cuda_12.4.r12.4/compiler.34097967_0\n"
        )
    if cmd in ("env", "printenv"):
        uuids = []
        for idx, gi in devices:
            g = node.gpus[idx]
            uuids.append(f"MIG-{g.uuid[4:12]}-{gi:04d}" if gi is not None else g.uuid)
        base = {
            "PATH": "/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin",
            "HOSTNAME": hostname,
            "CUDA_VERSION": "12.4.1",
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
        }
        if uuids:
            base["NVIDIA_VISIBLE_DEVICES"] = ",".join(uuids)
        elif image.startswith("nvcr.io/nvidia/"):
            base["NVIDIA_VISIBLE_DEVICES"] = "all"
        base.update(env)
        return "".join(f"{k}={v}\n" for k, v in sorted(base.items()))
    if cmd == "hostname":
        return hostname + "\n"
    if cmd in ("ibv_devices", "ibv_devinfo", "ibstat"):
        if not rdma:
            if cmd == "ibv_devices":
                return "    device                 node GUID\n    ------              ----------------\n"
            raise CommandError(
                "Failed to get IB devices list: Function not implemented"
                if cmd != "ibstat"
                else "ibpanic: [1] main: stat of IB device 'mlx5_0' failed: No such file or directory",
                1,
            )
        if cmd == "ibv_devices":
            rows = ["    device                 node GUID", "    ------              ----------------"]
            rows += [f"    {p.ca:<20}b83fd2030{i:02x}3a1c{i:01x}" for i, p in enumerate(node.ib)]
            return "\n".join(rows) + "\n"
        return "".join(
            f"hca_id:\t{p.ca}\n\ttransport:\t\t\tInfiniBand (0)\n\tfw_ver:\t\t\t\t{p.fw}\n"
            f"\t\tport:\t1\n\t\t\tstate:\t\t\tPORT_{p.state.upper()} (4)\n\t\t\tactive_mtu:\t\t4096 (5)\n"
            f"\t\t\tlink_layer:\t\t{p.link_layer}\n"
            for p in node.ib
        )
    if cmd == "ls":
        target = " ".join(argv[1:])
        if "/dev/infiniband" in target:
            if not rdma:
                raise CommandError("ls: cannot access '/dev/infiniband': No such file or directory", 2)
            return " ".join(["rdma_cm"] + [f"uverbs{i}" for i in range(len(node.ib))]) + "\n"
        if "nvidia" in target:
            devs = [f"/dev/nvidia{idx}" for idx in sorted({d[0] for d in devices})]
            if not devs:
                raise CommandError("ls: cannot access '/dev/nvidia*': No such file or directory", 2)
            return "\n".join(["/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidiactl"] + devs) + "\n"
        return "bin  boot  dev  etc  home  lib  opt  proc  root  run  sbin  sys  tmp  usr  var  workspace\n"
    if cmd in ("echo",):
        return " ".join(argv[1:]) + "\n"
    if cmd in ("sleep", "true", "bash", "sh", "cat", "tail"):
        return ""
    raise CommandError(
        f"OCI runtime exec failed: exec failed: unable to start container process: "
        f'exec: "{argv[0]}": executable file not found in $PATH: unknown',
        127,
    )


# ====================================================================== docker


def _short_id(*parts, n=12):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:n]


_ADJ = [
    "admiring",
    "brave",
    "clever",
    "dazzling",
    "eager",
    "focused",
    "gifted",
    "happy",
    "jolly",
    "keen",
    "lucid",
    "modest",
    "nifty",
    "optimistic",
    "quirky",
    "relaxed",
    "serene",
    "trusting",
    "vibrant",
    "zealous",
]
_NAMES = ["hopper", "turing", "lovelace", "curie", "noether", "dijkstra", "knuth", "shannon", "babbage", "ritchie"]


def _docker_name(seed):
    h = int(_short_id(seed, n=8), 16)
    return f"{_ADJ[h % len(_ADJ)]}_{_NAMES[(h // 20) % len(_NAMES)]}"


def _docker_running(ctx):
    if ctx.node.services.get("docker") != "active":
        raise CommandError(
            "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?"
        )


def _pull(ctx, ref, quiet=False):
    """Pull into the local docker image store; returns output text."""
    node = ctx.node
    full = norm_image(ref)
    reg, repo, tag = parse_image(ref)
    creds = docker_creds(node, reg)
    err = registry_check(ctx.state, ref, creds)
    if err:
        raise CommandError(pull_error_text(err, ref))
    digest = "sha256:" + _short_id(full, n=64)
    if full in node.docker_images:
        return f"{tag}: Pulling from {repo}\nDigest: {digest}\nStatus: Image is up to date for {full}\n{full}\n"
    node.docker_images.append(full)
    layers = [_short_id(full, i) for i in range(4)]
    out = [f"{tag}: Pulling from {repo.removeprefix('library/')}"]
    out += [f"{l}: Pull complete" for l in layers]
    out += [f"Digest: {digest}", f"Status: Downloaded newer image for {full}", full]
    return "\n".join(out) + "\n"


def _parse_gpus(spec, node):
    spec = spec.strip().strip("'\"")
    gpus = node.gpus
    if spec == "all":
        return [g.index for g in gpus if not g.fallen_off_bus]
    if spec.isdigit():
        n = int(spec)
        return [g.index for g in gpus if not g.fallen_off_bus][:n]
    if spec.startswith("device="):
        out = []
        for tok in spec[len("device=") :].split(","):
            tok = tok.strip().strip("'\"")
            match = [g for g in gpus if str(g.index) == tok or g.uuid == tok]
            if not match:
                raise CommandError(
                    "docker: Error response from daemon: failed to create task for container: failed to "
                    "create shim task: OCI runtime create failed: runc create failed: unable to start container "
                    "process: error during container init: error running hook #0: error running hook: exit status 1, "
                    f"stdout: , stderr: Auto-detected mode as 'legacy'\nnvidia-container-cli: device error: {tok}: unknown device: unknown.",
                    125,
                )
            out.extend(g.index for g in match)
        return out
    raise CommandError(
        f"invalid argument \"{spec}\" for \"--gpus\" flag: unexpected key '{spec.split('=')[0]}' in '{spec}'", 125
    )


def docker_loaded(node):
    if "runtime_loaded" not in node:
        return {"handlers": ["runc"], "default": "runc", "gpu_driver": toolkit_installed(node)}
    d = node.runtime_loaded.docker
    return {"handlers": d["handlers"], "default": d["default"], "gpu_driver": d.get("gpu_driver", True)}


def container_devices(node, runtime, env, image, gpu_request=None):
    """GPUs a docker container gets: via a GPU device request (--gpus / compose deploy.devices),
    or via the nvidia runtime reading NVIDIA_VISIBLE_DEVICES (env or the image default)."""
    loaded = docker_loaded(node)
    if gpu_request is not None:
        if not loaded["gpu_driver"]:
            raise CommandError(
                'Error response from daemon: could not select device driver "" with capabilities: [[gpu]].'
            )
        if not toolkit_installed(node):
            raise CommandError(
                "Error response from daemon: failed to create task for container: failed to create shim task: OCI "
                "runtime create failed: runc create failed: unable to start container process: error during container "
                'init: error running hook #0: error running hook: exec: "nvidia-container-runtime-hook": executable '
                "file not found in $PATH: unknown."
            )
        if not node.driver_loaded:
            raise CommandError(
                "Error response from daemon: failed to create task for container: failed to create shim task: "
                "OCI runtime create failed: runc create failed: unable to start container process: error during "
                "container init: error running hook #0: error running hook: exit status 1, stdout: , stderr: "
                "nvidia-container-cli: initialization error: nvml error: driver not loaded: unknown."
            )
        try:
            return [(i, None) for i in _parse_gpus(str(gpu_request), node)]
        except CommandError as e:
            raise CommandError(str(e).removeprefix("docker: ")) from None
    if runtime == "nvidia" and node.driver_loaded and toolkit_installed(node):
        nvd = (env or {}).get("NVIDIA_VISIBLE_DEVICES", "all" if image.startswith("nvcr.io/nvidia/") else "")
        if nvd and nvd not in ("void", "none"):
            return [(i, None) for i in _parse_gpus("all" if nvd == "all" else "device=" + nvd, node)]
    return []


def start_container(state, host, c):
    """(Re)start an existing container record. Returns an error string or None."""
    node = state.nodes[host]
    loaded = docker_loaded(node)
    rt = c.get("runtime") or "runc"
    if rt not in loaded["handlers"]:
        c.status = "Exited (128) 1 second ago"
        return f"unknown or invalid runtime name: {rt}"
    try:
        devs = container_devices(node, rt, c.get("env") or {}, c.image, c.get("gpu_request"))
    except CommandError as e:
        c.status = "Exited (128) 1 second ago"
        return str(e).removeprefix("Error response from daemon: ")
    c.gpus = [d[0] for d in devs]
    c.status = "Up 1 second"
    c.manually_stopped = False
    return None


DOCKER_VALUE_FLAGS = {
    "--gpus",
    "--restart",
    "--name",
    "-e",
    "--env",
    "-v",
    "--volume",
    "--shm-size",
    "--ipc",
    "--ulimit",
    "-p",
    "--publish",
    "--runtime",
    "--network",
    "--net",
    "-w",
    "--workdir",
    "--entrypoint",
    "--device",
    "-u",
    "--user",
    "--cpus",
    "-m",
    "--memory",
    "--env-file",
    "--label",
    "-l",
    "--mount",
    "--cap-add",
    "--security-opt",
    "--pid",
    "--hostname",
    "-h",
}


def _docker_run(ctx, args):
    node = ctx.node
    opts, i = {"env": {}}, 0
    while i < len(args):
        a = args[i]
        if not a.startswith("-") or a == "-":
            break
        key, val = a, None
        if a.startswith("--") and "=" in a:
            key, val = a.split("=", 1)
        if key in DOCKER_VALUE_FLAGS:
            if val is None:
                i += 1
                val = args[i] if i < len(args) else ""
            if key in ("-e", "--env"):
                k, _, v = val.partition("=")
                opts["env"][k] = v
            else:
                opts[key.lstrip("-")] = val
        elif key.startswith("-") and not key.startswith("--"):
            for ch in key[1:]:
                opts[ch] = True
        else:
            opts[key.lstrip("-")] = True
        i += 1
    if i >= len(args):
        raise CommandError("\"docker run\" requires at least 1 argument.\nSee 'docker run --help'.", 125)
    image, cmd = args[i], args[i + 1 :]
    full = norm_image(image)
    out = []
    if full not in node.docker_images:
        out.append(f"Unable to find image '{full}' locally")
        try:
            out.append(_pull(ctx, image).rstrip("\n"))
        except CommandError as e:
            raise CommandError("\n".join(out + ["docker: " + str(e), "See 'docker run --help'."]), 125)
    # which GPUs does the container see?
    loaded = docker_loaded(node)
    runtime = opts.get("runtime") or loaded["default"]
    if opts.get("runtime") and opts["runtime"] not in loaded["handlers"]:
        raise CommandError(
            f"docker: Error response from daemon: unknown or invalid runtime name: {opts['runtime']}.\n"
            "See 'docker run --help'.",
            125,
        )
    try:
        devices = container_devices(node, runtime, opts["env"], full, opts.get("gpus"))
    except CommandError as e:
        raise CommandError("\n".join(out + ["docker: " + str(e)]), 125) from None
    name = opts.get("name") or _docker_name(f"{ctx.host}{full}{len(node.containers)}")
    if any(c.name == name for c in node.containers):
        raise CommandError(
            f'docker: Error response from daemon: Conflict. The container name "/{name}" is already in use. '
            "You have to remove (or rename) that container to be able to reuse that name.\nSee 'docker run --help'.",
            125,
        )
    cid = _short_id(ctx.host, name, len(node.containers), n=64)
    record = {
        "id": cid[:12],
        "image": full,
        "name": name,
        "command": " ".join(cmd) or "/bin/bash",
        "gpus": [d[0] for d in devices],
        "env": dict(opts["env"]),
        "runtime": runtime,
        "gpu_request": opts.get("gpus"),
        "restart": opts.get("restart") or "no",
    }
    if opts.get("d"):
        node.containers.append({**record, "status": "Up 2 seconds"})
        return "\n".join(out + [cid]) + "\n"
    try:
        res = exec_in_container(ctx, node, devices, cmd, full, hostname=cid[:12], env=opts["env"]) if cmd else ""
        if cmd and cmd[0].rsplit("/", 1)[-1] == "nvidia-smi" and devices:
            ctx.state.flags.setdefault("gpu_containers", []).append(
                {"host": ctx.host, "image": full, "gpus": len({d[0] for d in devices}), "runtime": runtime}
            )
        if not cmd and (opts.get("i") or opts.get("t")):
            res = "(interactive container shells are not simulated; the container exited)\n"
        code_str = "0"
    except CommandError as e:
        msg = str(e)
        if "executable file not found" in msg:
            raise CommandError(
                "\n".join(
                    out
                    + [
                        "docker: Error response from daemon: failed to create task for container: "
                        "failed to create shim task: OCI runtime create failed: runc create failed: "
                        "unable to start container process: error during container init: "
                        f'exec: "{cmd[0]}": executable file not found in $PATH: unknown.'
                    ]
                ),
                127,
            )
        res, code_str = msg + "\n", "1"
    if not opts.get("rm"):
        node.containers.append({**record, "command": " ".join(cmd), "status": f"Exited ({code_str}) 2 seconds ago"})
    return "\n".join(out + [res.rstrip("\n")]) + "\n" if out else res


def _docker_login(ctx, args):
    node = ctx.node
    user, pw, server, stdin_pw = None, None, None, False
    it = iter(range(len(args)))
    i = 0
    pos = []
    while i < len(args):
        a = args[i]
        if a in ("-u", "--username"):
            user = args[i + 1] if i + 1 < len(args) else ""
            i += 2
            continue
        if a in ("-p", "--password"):
            pw = args[i + 1] if i + 1 < len(args) else ""
            i += 2
            continue
        if a.startswith("--username="):
            user = a.split("=", 1)[1]
        elif a.startswith("--password="):
            pw = a.split("=", 1)[1]
        elif a == "--password-stdin":
            stdin_pw = True
        else:
            pos.append(a)
        i += 1
    del it
    server = pos[0] if pos else "docker.io"
    if len(pos) > 1:
        raise CommandError(
            "\"docker login\" requires at most 1 argument.\nSee 'docker login --help'.\n\n"
            "Usage:  docker login [OPTIONS] [SERVER]",
            1,
        )
    how = unquoted_oauthtoken(ctx)
    if user == OAUTH_USER and how == "unquoted":
        # bash expands the unset variable to nothing: `-u -p KEY` -> username "-p", KEY becomes a 2nd argument
        raise CommandError(
            "\"docker login\" requires at most 1 argument.\nSee 'docker login --help'.\n\n"
            "Usage:  docker login [OPTIONS] [SERVER]\n\nLog in to a registry",
            1,
        )
    if user == OAUTH_USER and how == "double":
        user = ""
    if stdin_pw:
        pw = ctx.stdin.strip()
        if not pw:
            raise CommandError("Error: Password Required", 1)
    out = []
    if pw is not None and not stdin_pw:
        out.append("WARNING! Using --password via the CLI is insecure. Use --password-stdin.")
    if not user or pw is None or pw == "" or pw.startswith("$"):
        raise CommandError("\n".join(out + ["Error: Cannot perform an interactive login from a non TTY device"]), 1)
    if server.replace("https://", "").rstrip("/") == "nvcr.io":
        if not creds_valid(ctx.state, user, pw):
            raise CommandError(
                "\n".join(
                    out
                    + ['Error response from daemon: Get "https://nvcr.io/v2/": unauthorized: authentication required']
                ),
                1,
            )
        ctx.state.ngc.registry_logged_in[ctx.host] = True
    try:
        cfg = json.loads(node.files.get(DOCKER_AUTH, "") or "{}")
    except ValueError:
        cfg = {}
    cfg.setdefault("auths", {})[server.replace("https://", "").rstrip("/")] = {
        "auth": base64.b64encode(f"{user}:{pw}".encode()).decode()
    }
    node.files[DOCKER_AUTH] = json.dumps(cfg, indent="\t") + "\n"
    out += [
        f"WARNING! Your password will be stored unencrypted in {DOCKER_AUTH}.",
        "Configure a credential helper to remove this warning. See",
        "https://docs.docker.com/engine/reference/commandline/login/#credentials-store",
        "",
        "Login Succeeded",
    ]
    return "\n".join(out) + "\n"


@command(
    "docker",
    help="docker images|pull|run [--gpus all|'\"device=0,1\"'] [--rm]|ps|exec|inspect|logs|start|login nvcr.io|info|rmi"
    " | docker compose -f FILE up -d|ps|down|config",
)
def docker(ctx, args):
    container_init(ctx.state)
    node = ctx.node
    if not args or args[0] in ("-h", "--help", "help"):
        return (
            "Usage:  docker [OPTIONS] COMMAND\n\nCommon Commands:\n  run  exec  ps  pull  images  login  logout  info\n"
            "  inspect  logs  start  restart  stop  rm  rmi  compose\n"
        )
    sub, rest = args[0], args[1:]
    if sub == "version" or sub == "--version":
        return "Docker version 24.0.7, build afdd53b\n"
    _docker_running(ctx)
    if sub == "login":
        return _docker_login(ctx, rest)
    if sub == "logout":
        server = (rest or ["docker.io"])[0]
        try:
            cfg = json.loads(node.files.get(DOCKER_AUTH, "") or "{}")
        except ValueError:
            cfg = {}
        if cfg.get("auths", {}).pop(server, None) is None:
            return f"Not logged in to {server}\n"
        node.files[DOCKER_AUTH] = json.dumps(cfg, indent="\t") + "\n"
        ctx.state.ngc.registry_logged_in[ctx.host] = False
        return f"Removing login credentials for {server}\n"
    if sub == "pull":
        refs = [a for a in rest if not a.startswith("-")]
        if not refs:
            raise CommandError('"docker pull" requires exactly 1 argument.', 1)
        return _pull(ctx, refs[0])
    if sub in ("images", "image") and (sub == "images" or (rest and rest[0] in ("ls", "list"))):
        rows = [("REPOSITORY", "TAG", "IMAGE ID", "CREATED", "SIZE")]
        for ref in node.docker_images:
            name, _, tag = ref.rpartition(":")
            size = next((v for k, v in IMAGE_SIZES.items() if name.endswith("/" + k) or name == k), "1.2GB")
            rows.append((name, tag, _short_id(ref), "5 months ago", size))
        return _cols(rows)
    if sub == "run":
        return _docker_run(ctx, rest)
    if sub == "ps":
        show_all = "-a" in rest or "--all" in rest
        rows = [("CONTAINER ID", "IMAGE", "COMMAND", "CREATED", "STATUS", "PORTS", "NAMES")]
        for c in node.containers:
            if not show_all and not c.status.startswith("Up"):
                continue
            cmd = (
                '"' + (c.get("command") or "/bin/bash")[:17] + '…"'
                if len(c.get("command") or "") > 17
                else f'"{c.get("command") or "/bin/bash"}"'
            )
            rows.append((c.id, c.image, cmd, "2 minutes ago", c.status, "", c.name))
        return _cols(rows)
    if sub == "exec":
        return _docker_exec(ctx, rest)
    if sub == "inspect" or (sub == "container" and rest[:1] == ["inspect"]):
        return _docker_inspect(ctx, rest[1:] if sub == "container" else rest)
    if sub == "logs":
        return _docker_logs(ctx, rest)
    if sub == "compose":
        return _docker_compose(ctx, rest)
    if sub in ("start", "restart"):
        out = []
        for t in [a for a in rest if not a.startswith("-")]:
            c = _find_container(node, t)
            err = start_container(ctx.state, ctx.host, c)
            if err:
                raise CommandError(f"Error response from daemon: {err}\nError: failed to start containers: {t}", 1)
            out.append(t)
        return "\n".join(out) + "\n"
    if sub in ("rm", "stop", "kill"):
        targets = [a for a in rest if not a.startswith("-")]
        out = []
        for t in targets:
            c = next((c for c in node.containers if c.id.startswith(t) or c.name == t), None)
            if c is None:
                raise CommandError(f"Error response from daemon: No such container: {t}")
            if sub == "rm":
                if c.status.startswith("Up") and "-f" not in rest:
                    raise CommandError(
                        f"Error response from daemon: You cannot remove a running container {c.id}. "
                        "Stop the container before attempting removal or force remove"
                    )
                node.containers.remove(c)
            else:
                c.status = "Exited (137) 1 second ago"
                c["manually_stopped"] = True
            out.append(t)
        return "\n".join(out) + "\n"
    if sub == "rmi":
        for t in [a for a in rest if not a.startswith("-")]:
            full = norm_image(t)
            if full not in node.docker_images:
                raise CommandError(f"Error response from daemon: No such image: {t}")
            node.docker_images.remove(full)
            return f"Untagged: {full}\nDeleted: sha256:{_short_id(full, n=64)}\n"
        return ""
    if sub == "info":
        rt = node.runtime_loaded.docker
        return (
            f"Client: Docker Engine - Community\n Version:    24.0.7\n\nServer:\n Containers: {len(node.containers)}\n"
            f" Images: {len(node.docker_images)}\n Server Version: 24.0.7\n Storage Driver: overlay2\n"
            f" Cgroup Driver: systemd\n Cgroup Version: 2\n Runtimes: {' '.join(sorted(set(rt['handlers'])))}\n"
            f" Default Runtime: {rt['default']}\n Kernel Version: {node.kernel}\n Operating System: {node.os}\n"
            f" CPUs: {node.cpus}\n Name: {ctx.host}\n Docker Root Dir: /var/lib/docker\n"
        )
    raise CommandError(f"docker: '{sub}' is not a docker command.\nSee 'docker --help'", 1)


# ====================================================================== docker exec / inspect / logs / compose


def _find_container(node, ref):
    c = next((c for c in node.containers if c.name == ref or c.id.startswith(ref)), None)
    if c is None:
        raise CommandError(f"Error response from daemon: No such container: {ref}", 1)
    return c


def _docker_exec(ctx, args):
    node = ctx.node
    env, i = {}, 0
    while i < len(args) and args[i].startswith("-"):
        a = args[i]
        if a in ("-e", "--env", "-u", "--user", "-w", "--workdir"):
            if a in ("-e", "--env") and i + 1 < len(args):
                k, _, v = args[i + 1].partition("=")
                env[k] = v
            i += 2
            continue
        i += 1
    if i >= len(args) or i + 1 >= len(args):
        raise CommandError("\"docker exec\" requires at least 2 arguments.\nSee 'docker exec --help'.", 1)
    c = _find_container(node, args[i])
    if not c.status.startswith("Up"):
        raise CommandError(f"Error response from daemon: container {c.id} is not running", 1)
    devices = [(g, None) for g in c.get("gpus") or []]
    return exec_in_container(
        ctx, node, devices, args[i + 1 :], c.image, hostname=c.id, env={**(c.get("env") or {}), **env}
    )


def _inspect_obj(ctx, c):
    req = c.get("gpu_request")
    device_requests = None
    if req is not None:
        spec = str(req).strip("'\"")
        device_requests = [
            {
                "Driver": "",
                "Count": -1 if spec == "all" else (int(spec) if spec.isdigit() else 0),
                "DeviceIDs": spec[len("device=") :].split(",") if spec.startswith("device=") else None,
                "Capabilities": [["gpu"]],
                "Options": {},
            }
        ]
    env = [f"{k}={v}" for k, v in sorted((c.get("env") or {}).items())]
    if c.image.startswith("nvcr.io/nvidia/") and not any(e.startswith("NVIDIA_VISIBLE_DEVICES=") for e in env):
        env.append("NVIDIA_VISIBLE_DEVICES=all")
    env += ["NVIDIA_DRIVER_CAPABILITIES=compute,utility", "PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/bin"]
    up = c.status.startswith("Up")
    labels = {}
    if c.get("compose_project"):
        labels = {
            "com.docker.compose.project": c.compose_project,
            "com.docker.compose.service": c.compose_service,
            "com.docker.compose.project.config_files": c.compose_file,
        }
    return {
        "Id": _short_id(ctx.host, c.name, n=64),
        "Name": "/" + c.name,
        "State": {"Status": "running" if up else "exited", "Running": up, "ExitCode": 0 if up else 1},
        "Image": c.image,
        "Config": {"Image": c.image, "Env": env, "Labels": labels, "Cmd": (c.get("command") or "").split()},
        "HostConfig": {
            "Runtime": c.get("runtime") or "runc",
            "DeviceRequests": device_requests,
            "RestartPolicy": {"Name": c.get("restart") or "no", "MaximumRetryCount": 0},
        },
    }


def _go_template(tmpl, obj):
    """Tiny subset of Go templates: {{.A.B}} and {{json .A.B}}."""

    def repl(m):
        expr = m.group(1).strip()
        as_json = expr.startswith("json ")
        path = expr[5:].strip() if as_json else expr
        cur = obj
        for part in [p for p in path.split(".") if p]:
            cur = cur.get(part) if isinstance(cur, dict) else None
        if as_json:
            return json.dumps(cur)
        if cur is None:
            return "<no value>"
        if isinstance(cur, list):
            return "[" + " ".join(str(x) for x in cur) + "]"
        if isinstance(cur, bool):
            return str(cur).lower()
        if isinstance(cur, dict):
            return "map[" + " ".join(f"{k}:{v}" for k, v in cur.items()) + "]"
        return str(cur)

    return re.sub(r"\{\{(.*?)\}\}", repl, tmpl)


def _docker_inspect(ctx, args):
    fmt, refs, i = None, [], 0
    while i < len(args):
        a = args[i]
        if a in ("-f", "--format"):
            fmt = args[i + 1] if i + 1 < len(args) else ""
            i += 2
            continue
        if a.startswith("--format="):
            fmt = a.split("=", 1)[1]
        elif not a.startswith("-"):
            refs.append(a)
        i += 1
    if not refs:
        raise CommandError("\"docker inspect\" requires at least 1 argument.\nSee 'docker inspect --help'.", 1)
    objs = [_inspect_obj(ctx, _find_container(ctx.node, r)) for r in refs]
    if fmt:
        return "".join(_go_template(fmt, o) + "\n" for o in objs)
    return json.dumps(objs, indent=4) + "\n"


def _docker_logs(ctx, args):
    refs = [a for a in args if not a.startswith("-")]
    if not refs:
        raise CommandError("\"docker logs\" requires exactly 1 argument.\nSee 'docker logs --help'.", 1)
    c = _find_container(ctx.node, refs[0])
    return "\n".join(container_logs(ctx.node, c)) + "\n"


def container_logs(node, c):
    if c.get("logs"):
        return list(c.logs)
    if "tritonserver" in c.image:
        gpus = c.get("gpus") or []
        if not gpus:
            return [
                "W1019 10:31:12.105443 1 pinned_memory_manager.cc:273] Unable to allocate pinned system memory, "
                "pinned memory pool will not be available: CUDA driver version is insufficient for CUDA runtime version",
                "I1019 10:31:12.105577 1 cuda_memory_manager.cc:117] CUDA memory pool disabled",
                "E1019 10:31:12.109421 1 model_lifecycle.cc:638] failed to load 'embedder' version 1: Invalid argument: "
                "instance group embedder_0 of model embedder specifies invalid or unsupported gpu id 0. "
                "GPUs with at least the minimum required CUDA compute compatibility of 6.000000 are:",
                "I1019 10:31:12.109502 1 server.cc:676] +----------+---------+------------------------------------+",
                "error: creating server: Internal - failed to load all models",
            ]
        lines = [
            "I1019 10:31:12.004211 1 pinned_memory_manager.cc:275] Pinned memory pool is created at '0x7f3a40000000' "
            "with size 268435456",
        ]
        lines += [
            f"I1019 10:31:12.0045{i:02d} 1 cuda_memory_manager.cc:107] CUDA memory pool is created on device {i} "
            "with size 67108864"
            for i in range(len(gpus))
        ]
        lines += [
            "I1019 10:31:14.881020 1 model_lifecycle.cc:839] successfully loaded 'embedder'",
            "I1019 10:31:14.882115 1 grpc_server.cc:2463] Started GRPCInferenceService at 0.0.0.0:8001",
            "I1019 10:31:14.882390 1 http_server.cc:4692] Started HTTPService at 0.0.0.0:8000",
        ]
        return lines
    return []


COMPOSE_NAMES = ("compose.yaml", "compose.yml", "docker-compose.yaml", "docker-compose.yml")


def _compose_load(ctx, files):
    node = ctx.node
    if not files:
        for nm in COMPOSE_NAMES:
            path = f"{ctx.cwd.rstrip('/')}/{nm}"
            if path in node.files:
                files = [path]
                break
    if not files:
        raise CommandError(
            "no configuration file provided: not found\n"
            "(hint: in this lab `cd` does not change directory; use docker compose -f /path/to/compose.yaml ...)",
            1,
        )
    path = files[0] if files[0].startswith("/") else f"{ctx.cwd.rstrip('/')}/{files[0]}"
    if path not in node.files:
        raise CommandError(f"open {path}: no such file or directory", 1)
    import yaml

    try:
        doc = yaml.safe_load(node.files[path]) or {}
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        raise CommandError(
            f"yaml: line {mark.line + 1 if mark else 1}: {getattr(e, 'problem', 'could not parse')}", 15
        ) from None
    services = doc.get("services") or {}
    if not isinstance(services, dict) or not services:
        raise CommandError(f"{path}: no service defined", 15)
    return path, doc, services


def _svc_env(svc):
    env = svc.get("environment") or {}
    if isinstance(env, list):
        out = {}
        for e in env:
            k, _, v = str(e).partition("=")
            out[k] = v
        return out
    return {str(k): "" if v is None else str(v) for k, v in env.items()}


def _svc_gpu_request(name, svc):
    devs = (((svc.get("deploy") or {}).get("resources") or {}).get("reservations") or {}).get("devices")
    if not devs:
        return None
    d = devs[0]
    caps = d.get("capabilities")
    if not caps:
        raise CommandError(f"services.{name}.deploy.resources.reservations.devices.0 capabilities is required", 15)
    if "gpu" not in caps:
        return None
    if d.get("device_ids"):
        return "device=" + ",".join(str(x) for x in d["device_ids"])
    count = d.get("count", "all")
    return "all" if str(count) == "all" else str(int(count))


def _docker_compose(ctx, args):
    node = ctx.node
    files, project, i = [], None, 0
    while i < len(args) and args[i].startswith("-"):
        a = args[i]
        if a in ("-f", "--file"):
            files.append(args[i + 1] if i + 1 < len(args) else "")
            i += 2
            continue
        if a in ("-p", "--project-name"):
            project = args[i + 1] if i + 1 < len(args) else ""
            i += 2
            continue
        if a.startswith("--file="):
            files.append(a.split("=", 1)[1])
        i += 1
    if i >= len(args):
        raise CommandError("Usage:  docker compose [OPTIONS] COMMAND\n\nCommands: up down ps config logs restart", 1)
    verb, rest = args[i], args[i + 1 :]
    if verb == "version":
        return "Docker Compose version v2.24.5\n"
    path, doc, services = _compose_load(ctx, files)
    project = (project or doc.get("name") or path.rsplit("/", 2)[-2] or "default").lower()
    if verb == "config":
        import yaml

        return yaml.safe_dump({"name": project, "services": services}, sort_keys=False)
    _docker_running(ctx)
    mine = [c for c in node.containers if c.get("compose_project") == project]
    wanted = [a for a in rest if not a.startswith("-")]
    if verb == "ps":
        rows = [("NAME", "IMAGE", "COMMAND", "SERVICE", "CREATED", "STATUS", "PORTS")]
        for c in mine:
            if not c.status.startswith("Up") and "-a" not in rest and "--all" not in rest:
                continue
            rows.append(
                (c.name, c.image, f'"{(c.get("command") or "")[:20]}"', c.compose_service, "3 hours ago", c.status, "")
            )
        return _cols(rows)
    if verb == "down":
        out = []
        for c in mine:
            node.containers.remove(c)
            out.append(f" ✔ Container {c.name}  Removed")
        out.append(f" ✔ Network {project}_default  Removed")
        return f"[+] Running {len(out)}/{len(out)}\n" + "\n".join(out) + "\n"
    if verb == "logs":
        out = []
        for c in mine:
            if wanted and c.compose_service not in wanted:
                continue
            out += [f"{c.name}  | {ln}" for ln in container_logs(node, c)]
        return "\n".join(out) + "\n"
    if verb in ("restart", "start", "stop"):
        out, errs = [], []
        for c in mine:
            if wanted and c.compose_service not in wanted:
                continue
            if verb == "stop":
                c.status, c.manually_stopped = "Exited (0) 1 second ago", True
                out.append(f" ✔ Container {c.name}  Stopped")
                continue
            err = start_container(ctx.state, ctx.host, c)
            if err:
                errs.append(f"Error response from daemon: {err}")
            else:
                out.append(f" ✔ Container {c.name}  Started")
        if errs:
            raise CommandError("\n".join(out + errs), 1)
        return "\n".join(out) + "\n"
    if verb != "up":
        raise CommandError(f"unknown docker compose command: {verb}", 1)
    loaded = docker_loaded(node)
    out, errs = [], []
    for name, svc in services.items():
        if wanted and name not in wanted:
            continue
        svc = svc or {}
        image = svc.get("image")
        if not image:
            raise CommandError(
                f"service {name} has neither an image nor a build context specified: invalid compose project", 1
            )
        full = norm_image(image)
        cname = svc.get("container_name") or f"{project}-{name}-1"
        sig = json.dumps(svc, sort_keys=True, default=str)
        cur = next((c for c in node.containers if c.name == cname), None)
        if cur is not None and cur.get("compose_sig") == sig:
            if cur.status.startswith("Up"):
                out.append(f" ✔ Container {cname}  Running")
                continue
            err = start_container(ctx.state, ctx.host, cur)
            if err:
                errs.append(f"Error response from daemon: {err}")
                out.append(f" ✘ Container {cname}  Error")
            else:
                out.append(f" ✔ Container {cname}  Started")
            continue
        if full not in node.docker_images:
            try:
                _pull(ctx, image)
            except CommandError as e:
                errs.append(str(e))
                out.append(f" ✘ {name} Error")
                continue
        runtime = svc.get("runtime") or loaded["default"]
        if runtime not in loaded["handlers"]:
            errs.append(f"Error response from daemon: unknown or invalid runtime name: {runtime}")
            out.append(f" ✘ Container {cname}  Error")
            continue
        env = _svc_env(svc)
        try:
            req = _svc_gpu_request(name, svc)
            devs = container_devices(node, runtime, env, full, req)
        except CommandError as e:
            errs.append(str(e))
            out.append(f" ✘ Container {cname}  Error")
            continue
        verb_done = "Started"
        if cur is not None:
            node.containers.remove(cur)
            verb_done = "Recreated"
        cmd = svc.get("command") or ""
        node.containers.append(
            {
                "id": _short_id(ctx.host, cname, sig)[:12],
                "image": full,
                "name": cname,
                "status": "Up 1 second",
                "command": " ".join(cmd) if isinstance(cmd, list) else str(cmd),
                "gpus": [d[0] for d in devs],
                "env": env,
                "runtime": runtime,
                "gpu_request": req,
                "restart": svc.get("restart") or "no",
                "compose_project": project,
                "compose_service": name,
                "compose_file": path,
                "compose_sig": sig,
            }
        )
        out.append(f" ✔ Container {cname}  {verb_done}")
    head = f"[+] Running {sum(1 for o in out if chr(0x2714) in o)}/{len(out)}"
    text = head + "\n" + "\n".join(out)
    if errs:
        raise CommandError(text + "\n" + "\n".join(errs), 1)
    return text + "\n"


def _cols(rows, gap=3):
    if not rows:
        return ""
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    out = []
    for r in rows:
        out.append(
            "".join(str(c).ljust(widths[i] + gap) if i < len(r) - 1 else str(c) for i, c in enumerate(r)).rstrip()
        )
    return "\n".join(out) + "\n"


# ====================================================================== NGC CLI


def _ngc_org_choices(state):
    orgs = sorted({repo.split("/")[0] for repo in state.ngc.private_repos}) or ["nvidia"]
    return orgs


def _ngc_team_choices(state, org):
    teams = sorted(
        {repo.split("/")[1] for repo in state.ngc.private_repos if repo.startswith(org + "/") and repo.count("/") >= 2}
    )
    return teams + ["no-team"]


def _ngc_write_config(ctx):
    ngc = ctx.state.ngc
    text = (
        ";WARNING - This is a machine generated file.  Do not edit manually.\n"
        ';WARNING - To update local config settings, see "ngc config set -h" \n\n'
        "[CURRENT]\n"
        f"apikey = {ngc.api_key}\nformat_type = {ngc.format_type}\norg = {ngc.org or 'no-org'}\n"
        f"team = {ngc.team or 'no-team'}\nace = {ngc.ace}\n"
    )
    ctx.node.files[NGC_CONFIG] = text


def _ngc_config_set(ctx, args):
    state, ngc = ctx.state, ctx.state.ngc
    answers = [l.strip() for l in ctx.stdin.splitlines()]
    env = ctx.node.env
    key = env.get("NGC_CLI_API_KEY") or env.get("NGC_API_KEY") if not answers else None
    prompts = []

    def ask(prompt, default):
        prompts.append(prompt)
        if answers:
            v = answers.pop(0)
            return v or default
        return None

    if key is None:
        key = ask(
            f"Enter API key [{_mask(ngc.api_key) if ngc.api_key else 'no-apikey'}]. Choices: [<VALID_APIKEY>, 'no-apikey']: ",
            ngc.api_key or "no-apikey",
        )
        if key is None:
            raise CommandError(
                prompts[0] + "\nError: no interactive terminal. Feed the answers on stdin, e.g.\n"
                "  ngc config set <<EOF\n  <api key>\n  ascii\n  <org>\n  <team>\n  no-ace\n  EOF",
                1,
            )
    if key != "no-apikey" and not key_valid(state, key):
        raise CommandError("".join(prompts) + "\nInvalid apikey", 1)
    fmt = ask("Enter CLI output format type [ascii]. Choices: ['ascii', 'csv', 'json']: ", "ascii") or "ascii"
    orgs = _ngc_org_choices(state)
    org = (
        ask(f"Enter org [{ngc.org or 'no-org'}]. Choices: {orgs}: ", ngc.org or orgs[0])
        or env.get("NGC_CLI_ORG")
        or orgs[0]
    )
    if org not in orgs:
        raise CommandError(
            "".join(prompts) + f"\nInvalid org. Please re-run 'ngc config set' and choose one of: {orgs}", 1
        )
    teams = _ngc_team_choices(state, org)
    team = (
        ask(f"Enter team [{ngc.team or 'no-team'}]. Choices: {teams}: ", "no-team")
        or env.get("NGC_CLI_TEAM")
        or "no-team"
    )
    if team not in teams:
        raise CommandError(
            "".join(prompts) + f"\nInvalid team. Please re-run 'ngc config set' and choose one of: {teams}", 1
        )
    ace = ask("Enter ace [no-ace]. Choices: ['no-ace']: ", "no-ace") or "no-ace"
    ngc.api_key = "" if key == "no-apikey" else key
    ngc.configured = bool(ngc.api_key)
    ngc.format_type, ngc.org, ngc.team, ngc.ace = fmt, org, ("" if team == "no-team" else team), ace
    _ngc_write_config(ctx)
    return (
        "\n".join(p.rstrip() for p in prompts)
        + "\nValidating configuration...\nSuccessfully validated configuration.\n"
        f"Saving configuration...\nSuccessfully saved NGC configuration to {NGC_CONFIG}\n"
    )


def _mask(key):
    return "*" * 10 + key[-4:] if key else ""


def _ngc_require(ctx):
    ngc = ctx.state.ngc
    env_key = ctx.node.env.get("NGC_CLI_API_KEY", "")
    if env_key and key_valid(ctx.state, env_key):
        return
    if not ngc.configured or not key_valid(ctx.state, ngc.api_key):
        raise CommandError(
            "Missing API key and missing Email Authentication. This command requires an API key or "
            "authentication via browser login. Run 'ngc config set' to set your API key.",
            1,
        )


@command("ngc", help="NGC CLI: ngc config set|current, ngc registry image list|info")
def ngc_cli(ctx, args):
    container_init(ctx.state)
    ngc = ctx.state.ngc
    if not args or args[0] in ("-h", "--help"):
        return "usage: ngc [--version] [-h] {config,registry,org,team,user,diag} ...\n"
    if args[0] in ("--version", "version"):
        return "NGC CLI 3.48.0\n"
    if args[:2] == ["config", "set"]:
        return _ngc_config_set(ctx, args[2:])
    if args[:2] == ["config", "current"]:
        rows = [("key", "value", "source")]
        if ngc.api_key:
            rows.append(("apikey", _mask(ngc.api_key), NGC_CONFIG))
        rows += [("format_type", ngc.format_type, NGC_CONFIG)]
        if ngc.org:
            rows.append(("org", ngc.org, NGC_CONFIG))
        if ngc.team:
            rows.append(("team", ngc.team, NGC_CONFIG))
        return _ngc_table(rows)
    if args[:2] == ["config", "clear"]:
        ngc.api_key, ngc.configured, ngc.org, ngc.team = "", False, "", ""
        ctx.node.files.pop(NGC_CONFIG, None)
        return "Successfully cleared configuration.\n"
    if args[:1] == ["registry"] and len(args) >= 3 and args[1] == "image":
        _ngc_require(ctx)
        action = args[2]
        pattern = next((a for a in args[3:] if not a.startswith("-")), None)
        repos = dict(ngc.catalog)
        repos.update(ngc.private_repos)
        if action == "list":
            rows = [("Name", "Repository", "Latest Tag", "Image Size", "Updated Date", "Permission", "Signed Tag?")]
            for repo, tags in sorted(repos.items()):
                if pattern and not _glob_match(pattern.split(":")[0], repo):
                    continue
                perm = "unlocked" if repo in ngc.catalog else "Accessible"
                rows.append(
                    (
                        repo.rsplit("/", 1)[-1].replace("-", " ").title(),
                        repo,
                        tags[0],
                        next((v for k, v in IMAGE_SIZES.items() if repo.endswith(k)), "1.2 GB"),
                        "May 28, 2024",
                        perm,
                        "False",
                    )
                )
            return _ngc_table(rows)
        if action == "info":
            if not pattern:
                raise CommandError(
                    "ngc registry image info: error: the following arguments are required: <image>[:<tag>]", 2
                )
            repo, _, tag = pattern.partition(":")
            if repo not in repos:
                raise CommandError(f"Error: '{repo}' could not be found.", 1)
            if tag and tag not in repos[repo]:
                raise CommandError(f"Error: Tag '{tag}' could not be found for image '{repo}'.", 1)
            return _ngc_table(
                [
                    ("Image Information", ""),
                    ("Name", f"{repo}:{tag or repos[repo][0]}"),
                    ("Architecture", "amd64, arm64"),
                    ("Schema Version", "2"),
                    ("Tags", ", ".join(repos[repo])),
                ]
            )
        raise CommandError(
            f"ngc registry image: invalid choice: '{action}' (choose from 'info', 'list', 'pull', 'push', 'remove')", 2
        )
    if args[:2] == ["org", "list"] or args[:2] == ["user", "who"]:
        _ngc_require(ctx)
        return _ngc_table([("Name", "Org"), ("aiops-admin", ngc.org or "no-org")])
    raise CommandError(f"ngc: error: argument: invalid choice: '{args[0]}'", 2)


def _glob_match(pattern, text):
    import fnmatch

    return fnmatch.fnmatch(text, pattern) or fnmatch.fnmatch(text, pattern + "*")


def _ngc_table(rows):
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [sep]
    for n, r in enumerate(rows):
        out.append("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)) + " |")
        if n == 0:
            out.append(sep)
    out.append(sep)
    return "\n".join(out) + "\n"


# ====================================================================== enroot


def _enroot_creds(node):
    text = node.files.get(ENROOT_CREDS, "")
    m = re.search(r"machine\s+nvcr\.io\s+login\s+(\S+)\s+password\s+(\S+)", text)
    return (m.group(1), m.group(2)) if m else None


@command("enroot", help="enroot import docker://nvcr.io#nvidia/pytorch:24.05-py3 | create | list | start | remove")
def enroot(ctx, args):
    container_init(ctx.state)
    node = ctx.node
    node.setdefault("enroot_containers", [])
    if not args or args[0] in ("-h", "--help"):
        return "Usage: enroot COMMAND [ARG...]\n\nCommands:\n  batch  bundle  create  exec  export  import  list  remove  start  version\n"
    sub, rest = args[0], args[1:]
    if sub == "version":
        return "3.4.1\n"
    if sub == "import":
        out_name = None
        if "-o" in rest:
            out_name = rest[rest.index("-o") + 1]
        uri = next((a for a in rest if a.startswith(("docker://", "dockerd://"))), None)
        if uri is None:
            raise CommandError("[ERROR] Invalid image reference: expected docker://[USER@][REGISTRY#]IMAGE[:TAG]", 1)
        body = uri.split("://", 1)[1]
        if "@" in body.split("#")[0]:
            body = body.split("@", 1)[1]
        if "#" in body:
            registry, image = body.split("#", 1)
        else:
            registry, image = "registry-1.docker.io", body
        ref = f"{registry}/{image}" if registry != "registry-1.docker.io" else image
        reg, repo, tag = (
            parse_image(ref)
            if registry != "registry-1.docker.io"
            else ("docker.io", image.split(":")[0], (image.split(":") + ["latest"])[1])
        )
        lines = ["[INFO] Querying registry for permission grant"]
        if registry == "registry-1.docker.io" and image.startswith("nvcr.io/"):
            raise CommandError(
                "\n".join(
                    lines
                    + [
                        "[INFO] Authenticating with user: <anonymous>",
                        "[ERROR] URL https://auth.docker.io/token returned error code: 401 Unauthorized",
                        "        (hint: enroot separates the registry with '#': docker://nvcr.io#nvidia/pytorch:24.05-py3)",
                    ]
                ),
                1,
            )
        creds = _enroot_creds(node)
        if reg == "nvcr.io":
            err = registry_check(ctx.state, ref, creds)
            lines.append(f"[INFO] Authenticating with user: {creds[0] if creds else '<anonymous>'}")
            if err == "unauthorized":
                raise CommandError(
                    "\n".join(
                        lines
                        + [
                            f"[ERROR] URL https://nvcr.io/proxy_auth?scope=repository:{repo}:pull "
                            "returned error code: 401 Unauthorized"
                        ]
                    ),
                    1,
                )
            if err:
                raise CommandError(
                    "\n".join(
                        lines
                        + [f"[ERROR] URL https://nvcr.io/v2/{repo}/manifests/{tag} returned error code: 404 Not Found"]
                    ),
                    1,
                )
            lines.append("[INFO] Authentication succeeded")
        name = out_name or (repo.replace("/", "+") + "+" + tag + ".sqsh").removeprefix("library+")
        path = name if name.startswith("/") else f"{ctx.cwd.rstrip('/')}/{name}"
        if path in node.files:
            raise CommandError(f"[ERROR] File already exists: {path}", 1)
        node.files[path] = f"<squashfs image of {ref}>\n"
        if name not in node.enroot_images:
            node.enroot_images.append(name)
        lines += [
            "[INFO] Fetching image manifest list",
            "[INFO] Fetching image manifest",
            "[INFO] Downloading 34 missing layers...",
            "",
            "100% 34:0=0s 3b4ab1f3c6c4d9a6e0d8a1b0f4c0f6b1c9a3e2d4",
            "",
            "[INFO] Extracting image layers...",
            "",
            "100% 33:0=0s 9f2d8a1b",
            "",
            "[INFO] Converting whiteouts...",
            "",
            "[INFO] Creating squashfs filesystem...",
            "",
            f"Parallel mksquashfs: Using {node.cpus} processors",
            f"Creating 4.0 filesystem on {path}, block size 131072.",
            "[=====================================] 100%",
        ]
        return "\n".join(lines) + "\n"
    if sub == "create":
        name = None
        if "--name" in rest:
            name = rest[rest.index("--name") + 1]
        elif "-n" in rest:
            name = rest[rest.index("-n") + 1]
        src = next((a for a in rest if a.endswith(".sqsh")), None)
        if not src:
            raise CommandError("Usage: enroot create [options] [--] IMAGE", 1)
        path = src if src.startswith("/") else f"{ctx.cwd.rstrip('/')}/{src}"
        if path not in node.files:
            raise CommandError(f"[ERROR] No such file or directory: {src}", 1)
        name = name or src.rsplit("/", 1)[-1].removesuffix(".sqsh")
        if name in node.enroot_containers:
            raise CommandError(f"[ERROR] File already exists: /tmp/enroot-data/user-0/{name}", 1)
        node.enroot_containers.append(name)
        return f"[INFO] Extracting squashfs filesystem...\n\nParallel unsquashfs: Using {node.cpus} processors\ncreated 61534 files\n"
    if sub in ("list", "ls"):
        return "".join(c + "\n" for c in node.enroot_containers)
    if sub in ("remove", "rm"):
        for n in [a for a in rest if not a.startswith("-")]:
            if n not in node.enroot_containers:
                raise CommandError(f"[ERROR] No such file or directory: /tmp/enroot-data/user-0/{n}", 1)
            node.enroot_containers.remove(n)
        return ""
    if sub in ("start", "exec"):
        i = 0
        while i < len(rest) and rest[i].startswith("-"):
            i += 2 if rest[i] in ("-e", "--env", "-m", "--mount", "-c", "--conf") else 1
        if i >= len(rest):
            raise CommandError("Usage: enroot start [options] [--] NAME|IMAGE [COMMAND] [ARG...]", 1)
        name, cmd = rest[i], rest[i + 1 :]
        if name not in node.enroot_containers and not name.endswith(".sqsh"):
            raise CommandError(f"[ERROR] No such file or directory: /tmp/enroot-data/user-0/{name}", 1)
        devices = [(g.index, None) for g in node.gpus if not g.fallen_off_bus] if node.driver_loaded else []
        return exec_in_container(ctx, node, devices, cmd or ["bash"], "nvcr.io/nvidia/" + name, hostname=ctx.host)
    raise CommandError(f"[ERROR] Unknown command: {sub}", 1)


# ====================================================================== nvidia-ctk


@command(
    "nvidia-ctk", help="nvidia-ctk runtime configure --runtime=docker|containerd [--set-as-default] | cdi generate|list"
)
def nvidia_ctk(ctx, args):
    container_init(ctx.state)
    node = ctx.node
    if not toolkit_installed(node):
        raise CommandError("bash: nvidia-ctk: command not found", 127)
    if not args or args[0] in ("-h", "--help"):
        return (
            "NAME:\n   NVIDIA Container Toolkit CLI\n\nCOMMANDS:\n   hook     A collection of hooks\n"
            "   runtime  A collection of runtime-related utilities\n   cdi      Provide tools for interacting with CDI\n"
        )
    if args[0] in ("--version", "-v"):
        return f"NVIDIA Container Toolkit CLI version {CTK_VERSION}\ncommit: a470818ba7d9166be282cd0039dd2fc9b0a34d73\n"
    opts = _kv_opts(args)
    if args[:2] == ["runtime", "configure"]:
        rt = opts.get("runtime", "docker")
        default = "set-as-default" in opts
        dry = "dry-run" in opts
        if rt == "docker":
            path = opts.get("config", DOCKER_CONF)
            try:
                data = json.loads(node.files.get(path, "") or "{}")
            except ValueError as e:
                raise CommandError(f"ERRO[0000] unable to load config: {e}", 1)
            data.setdefault("runtimes", {})["nvidia"] = {"args": [], "path": "nvidia-container-runtime"}
            if default:
                data["default-runtime"] = "nvidia"
            text = json.dumps(data, indent=4) + "\n"
            svc = "docker"
        elif rt == "containerd":
            path = opts.get("config", CONTAINERD_CONF)
            cur = parse_containerd(node.files.get(path, ""))
            text = containerd_config(nvidia=True, default="nvidia" if default else cur["default"])
            svc = "containerd"
        elif rt == "crio":
            path, svc, text = (
                opts.get("config", "/etc/crio/crio.conf"),
                "crio",
                '[crio.runtime.runtimes.nvidia]\nruntime_path = "/usr/bin/nvidia-container-runtime"\n',
            )
        else:
            raise CommandError(f"ERRO[0000] unrecognized runtime '{rt}'", 1)
        if dry:
            return text
        existed = path in node.files
        node.files[path] = text
        out = [
            f"INFO[0000] Loading config from {path}"
            if existed
            else "INFO[0000] Config file does not exist; using empty config",
            f"INFO[0000] Wrote updated config to {path}",
            f"INFO[0000] It is recommended that {svc} daemon be restarted.",
        ]
        return "\n".join(out) + "\n"
    if args[:2] == ["cdi", "generate"]:
        if not node.driver_loaded or not node.gpus:
            raise CommandError(
                "ERRO[0000] failed to generate CDI spec: failed to create device CDI specs: failed to initialize NVML: ERROR_LIBRARY_NOT_FOUND",
                1,
            )
        out_path = opts.get("output")
        devs = []
        for g in node.gpus:
            devs.append(f'- name: "{g.index}"\n  containerEdits:\n    deviceNodes:\n    - path: /dev/nvidia{g.index}')
            devs.append(f"- name: {g.uuid}\n  containerEdits:\n    deviceNodes:\n    - path: /dev/nvidia{g.index}")
            for gi in g.gpu_instances:
                devs.append(
                    f'- name: "{g.index}:{gi.id}"\n  containerEdits:\n    deviceNodes:\n    - path: /dev/nvidia-caps/nvidia-cap{gi.id * 9 + 3}'
                )
        devs.append(
            "- name: all\n  containerEdits:\n    deviceNodes:\n"
            + "".join(f"    - path: /dev/nvidia{g.index}\n" for g in node.gpus).rstrip("\n")
        )
        spec = (
            "---\ncdiVersion: 0.5.0\ncontainerEdits:\n  deviceNodes:\n  - path: /dev/nvidiactl\n  - path: /dev/nvidia-uvm\n"
            "  hooks:\n  - hookName: createContainer\n    path: /usr/bin/nvidia-ctk\n    args: [nvidia-ctk, hook, update-ldcache]\n"
            "devices:\n" + "\n".join(devs) + "\nkind: nvidia.com/gpu\n"
        )
        log = [
            "INFO[0000] Using /usr/lib/x86_64-linux-gnu/libnvidia-ml.so." + node.driver_version,
            "INFO[0000] Auto-detected mode as 'nvml'",
            "INFO[0000] Selecting /dev/nvidia-modeset as /dev/nvidia-modeset",
            "INFO[0000] Selecting /dev/nvidia-uvm as /dev/nvidia-uvm",
            "INFO[0000] Selecting /dev/nvidiactl as /dev/nvidiactl",
        ]
        if out_path:
            node.files[out_path] = spec
            log.append("INFO[0000] Generated CDI spec with version 0.5.0")
            return "\n".join(log) + "\n"
        return "\n".join(log) + "\n" + spec
    if args[:2] == ["cdi", "list"]:
        spec = node.files.get(CDI_SPEC, "") or next(
            (v for k, v in node.files.items() if k.startswith(("/etc/cdi/", "/var/run/cdi/"))), ""
        )
        names = re.findall(r"^- name: \"?([^\"\n]+)\"?$", spec, re.M)
        return f"INFO[0000] Found {len(names)} CDI devices\n" + "".join(f"nvidia.com/gpu={n}\n" for n in names)
    raise CommandError(f"Incorrect Usage: nvidia-ctk {' '.join(args[:2])}", 1)


def _kv_opts(args):
    """--key=value / --key value / --flag  ->  dict."""
    out = {}
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--"):
            k, eq, v = a[2:].partition("=")
            if eq:
                out[k] = v
            elif (
                i + 1 < len(args) and not args[i + 1].startswith("-") and k in ("runtime", "config", "output", "format")
            ):
                out[k] = args[i + 1]
                i += 1
            else:
                out[k] = True
        i += 1
    return out


# ====================================================================== crictl


@command("crictl", help="crictl ps [-a] | pods | images | info | logs <id>  (containerd view of k8s pods on this node)")
def crictl(ctx, args):
    container_init(ctx.state)
    node = ctx.node
    if node.services.get("containerd") != "active":
        raise CommandError(
            'E1019 10:42:17.000000   1 remote_runtime.go:277] "ListContainers with filter from runtime service failed" '
            'err="rpc error: code = Unavailable desc = connection error: desc = \\"transport: Error while dialing: '
            'dial unix /run/containerd/containerd.sock: connect: no such file or directory\\""',
            1,
        )
    from . import kubectl as k

    k.k8s_init(ctx.state)
    k.reconcile(ctx.state)
    pods = [p for p in ctx.state.k8s.pods if p.node == ctx.host]
    sub = args[0] if args else "ps"
    if sub == "info":
        rt = node.runtime_loaded.containerd
        runtimes = {
            h: {
                "runtimeType": "io.containerd.runc.v2",
                "options": {
                    "BinaryName": "/usr/bin/nvidia-container-runtime" if h == "nvidia" else "",
                    "SystemdCgroup": True,
                },
            }
            for h in rt["handlers"]
        }
        info = {
            "status": {
                "conditions": [{"type": "RuntimeReady", "status": True}, {"type": "NetworkReady", "status": True}]
            },
            "config": {
                "containerd": {"defaultRuntimeName": rt["default"], "runtimes": runtimes},
                "sandboxImage": "registry.k8s.io/pause:3.9",
            },
        }
        return json.dumps(info, indent=2) + "\n"
    if sub == "pods":
        rows = [("POD ID", "CREATED", "STATE", "NAME", "NAMESPACE", "ATTEMPT", "RUNTIME")]
        for p in pods:
            if p.status in ("Pending",):
                continue
            rows.append(
                (
                    _short_id(p.uid, n=13),
                    "12 minutes ago",
                    "Ready",
                    p.name,
                    p.namespace,
                    "0",
                    p.spec.get("runtimeClassName") or "(default)",
                )
            )
        return _cols(rows)
    if sub == "images":
        rows = [("IMAGE", "TAG", "IMAGE ID", "SIZE")]
        seen = set()
        for p in pods:
            for c in p.spec.get("containers", []):
                ref = norm_image(c.get("image", ""))
                if ref in seen or p.status in ("ErrImagePull", "ImagePullBackOff"):
                    continue
                seen.add(ref)
                name, _, tag = ref.rpartition(":")
                rows.append((name, tag, _short_id(ref, n=13), "1.1GB"))
        return _cols(rows)
    if sub == "logs":
        cid = next((a for a in args[1:] if not a.startswith("-")), "")
        for p in pods:
            if _short_id(p.uid, "c", n=13).startswith(cid):
                return "\n".join(p.get("logs", [])) + "\n"
        raise CommandError(
            f'rpc error: code = NotFound desc = an error occurred when try to find container "{cid}": not found', 1
        )
    if sub == "ps":
        show_all = "-a" in args or "--all" in args
        name_f = None
        if "--name" in args:
            name_f = args[args.index("--name") + 1]
        rows = [("CONTAINER", "IMAGE", "CREATED", "STATE", "NAME", "ATTEMPT", "POD ID", "POD")]
        for p in pods:
            running = p.status == "Running"
            exited = p.status in ("CrashLoopBackOff", "Completed", "Error")
            if not (running or (show_all and exited)):
                continue
            for c in p.spec.get("containers", []):
                if name_f and name_f not in c.get("name", ""):
                    continue
                rows.append(
                    (
                        _short_id(p.uid, "c", n=13),
                        _short_id(norm_image(c.get("image", "")), n=13),
                        "12 minutes ago",
                        "Running" if running else "Exited",
                        c.get("name", ""),
                        str(k.pod_restarts(ctx.state, p)),
                        _short_id(p.uid, n=13),
                        p.name,
                    )
                )
        return _cols(rows)
    raise CommandError(f"No help topic for '{sub}'", 1)


# ====================================================================== apt / apt-get / dpkg (container-related packages only)

CTK_REPO_LIST = "/etc/apt/sources.list.d/nvidia-container-toolkit.list"
CTK_REPO_URL = "https://nvidia.github.io/libnvidia-container/stable/deb/$(ARCH)"
CTK_PKGS = {
    "nvidia-container-toolkit": ("NVIDIA Container toolkit", "1,642 kB"),
    "nvidia-container-toolkit-base": ("NVIDIA Container Toolkit Base", "3,412 kB"),
    "libnvidia-container-tools": ("NVIDIA container runtime library (command-line tools)", "24.1 kB"),
    "libnvidia-container1": ("NVIDIA container runtime library", "925 kB"),
}
BASE_PKGS = {
    "docker-ce": ("5:24.0.7-1~ubuntu.22.04~jammy", "Docker: the open-source application container engine"),
    "docker-ce-cli": ("5:24.0.7-1~ubuntu.22.04~jammy", "Docker CLI: the open-source application container engine"),
    "docker-compose-plugin": ("2.24.5-1~ubuntu.22.04~jammy", "Docker Compose (V2) plugin for the Docker CLI."),
    "containerd.io": ("1.7.13-1", "An open and reliable container runtime"),
}


def _ctk_version():
    return f"{CTK_VERSION}-1"


def installed_packages(node):
    pkgs = {k: v for k, v in BASE_PKGS.items()}
    drv = node.driver_version.split(".")[0]
    if node.gpus:
        pkgs[f"nvidia-driver-{drv}-server"] = (f"{node.driver_version}-0ubuntu1", "NVIDIA Server Driver metapackage")
    if toolkit_installed(node):
        for p, (desc, _size) in CTK_PKGS.items():
            pkgs[p] = (_ctk_version(), desc)
    return pkgs


def ctk_available(node):
    """Can apt find the toolkit packages? DGX OS ships them in its own repo; plain Ubuntu needs
    NVIDIA's repo (CTK_REPO_LIST) and a fresh `apt-get update` after adding it."""
    if toolkit_installed(node) or "DGX OS" in node.os:
        return True
    return "nvidia.github.io/libnvidia-container" in node.files.get(CTK_REPO_LIST, "") and node.get("apt_fresh", True)


def _apt_update(ctx):
    node = ctx.node
    node["apt_fresh"] = True
    out = [
        "Hit:1 http://archive.ubuntu.com/ubuntu jammy InRelease",
        "Get:2 http://archive.ubuntu.com/ubuntu jammy-updates InRelease [128 kB]",
        "Hit:3 http://security.ubuntu.com/ubuntu jammy-security InRelease",
        "Hit:4 https://download.docker.com/linux/ubuntu jammy InRelease",
    ]
    if "nvidia.github.io/libnvidia-container" in node.files.get(CTK_REPO_LIST, ""):
        out += [
            "Get:5 https://nvidia.github.io/libnvidia-container/stable/deb/amd64  InRelease [1,477 B]",
            "Get:6 https://nvidia.github.io/libnvidia-container/stable/deb/amd64  Packages [14.2 kB]",
        ]
    out += ["Fetched 144 kB in 1s (131 kB/s)", "Reading package lists... Done"]
    return "\n".join(out) + "\n"


APT_HEAD = "Reading package lists... Done\nBuilding dependency tree... Done\nReading state information... Done\n"


def _apt_install(ctx, pkgs, assume_yes):
    node = ctx.node
    have = installed_packages(node)
    new, already = [], []
    for p in pkgs:
        p = p.split("=", 1)[0]
        if p in have:
            already.append(f"{p} is already the newest version ({have[p][0]}).")
            continue
        if p == "nvidia-docker2":
            p = "nvidia-container-toolkit"  # deprecated meta-package; pulls in the toolkit
        if p not in CTK_PKGS or not ctk_available(node):
            raise CommandError(APT_HEAD + f"E: Unable to locate package {p}", 100)
        new.append(p)
    out = APT_HEAD
    if already:
        out += "\n".join(already) + "\n"
    if not new:
        return out + "0 upgraded, 0 newly installed, 0 to remove and 12 not upgraded.\n"
    allp = sorted(CTK_PKGS) if "nvidia-container-toolkit" in new else sorted(set(new))
    extra = [p for p in allp if p not in new]
    if extra:
        out += "The following additional packages will be installed:\n  " + " ".join(extra) + "\n"
    out += "The following NEW packages will be installed:\n  " + " ".join(allp) + "\n"
    out += f"0 upgraded, {len(allp)} newly installed, 0 to remove and 12 not upgraded.\n"
    out += "Need to get 6,003 kB of archives.\nAfter this operation, 27.6 MB of additional disk space will be used.\n"
    if not assume_yes and ctx.stdin.strip().lower() not in ("y", "yes"):
        return out + "Do you want to continue? [Y/n] Abort.\n"
    for n, p in enumerate(allp, 1):
        out += f"Get:{n} https://nvidia.github.io/libnvidia-container/stable/deb/amd64  {p} {_ctk_version()} [{CTK_PKGS[p][1]}]\n"
    for p in allp:
        out += f"Selecting previously unselected package {p}.\nUnpacking {p} ({_ctk_version()}) ...\n"
    for p in allp:
        out += f"Setting up {p} ({_ctk_version()}) ...\n"
    out += "Processing triggers for libc-bin (2.35-0ubuntu3.8) ...\n"
    node["ctk_installed"] = True
    node.files.setdefault("/etc/nvidia-container-runtime/config.toml", NVIDIA_CTR_CONFIG)
    return out


NVIDIA_CTR_CONFIG = """\
#accept-nvidia-visible-devices-as-volume-mounts = false
#accept-nvidia-visible-devices-envvar-when-unprivileged = true
disable-require = false
supported-driver-capabilities = "compat32,compute,display,graphics,ngx,utility,video"

[nvidia-container-cli]
environment = []
ldconfig = "@/sbin/ldconfig.real"
load-kmods = true
no-cgroups = false

[nvidia-container-runtime]
log-level = "info"
mode = "auto"
runtimes = ["docker-runc", "runc", "crun"]

[nvidia-container-runtime-hook]
path = "nvidia-container-runtime-hook"
skip-mode-detection = false

[nvidia-ctk]
path = "nvidia-ctk"
"""


def _apt_remove(ctx, pkgs, purge):
    node = ctx.node
    have = installed_packages(node)
    gone = [p for p in pkgs if p in have]
    out = APT_HEAD
    for p in pkgs:
        if p not in have:
            out += f"Package '{p}' is not installed, so not removed\n"
    if not gone:
        return out + "0 upgraded, 0 newly installed, 0 to remove and 12 not upgraded.\n"
    if any(p in CTK_PKGS for p in gone):
        node["ctk_installed"] = False
        if purge:
            node.files.pop("/etc/nvidia-container-runtime/config.toml", None)
    elif any(p.startswith("docker") or p == "containerd.io" for p in gone):
        raise CommandError("E: removing the container engine is not supported in this lab", 100)
    out += "The following packages will be REMOVED:\n  " + " ".join(gone) + "\n"
    out += f"0 upgraded, 0 newly installed, {len(gone)} to remove and 12 not upgraded.\n"
    out += "".join(f"Removing {p} ({have[p][0]}) ...\n" for p in gone)
    return out


@command("apt", "apt-get", help="apt-get update | apt-get install -y <pkg> | apt list --installed (container packages)")
def apt(ctx, args):
    node = ctx.node
    flags = {a for a in args if a.startswith("-")}
    pos = [a for a in args if not a.startswith("-")]
    if not pos:
        return "apt 2.4.12 (amd64)\nUsage: apt [options] command\n"
    sub, pkgs = pos[0], pos[1:]
    yes = bool(flags & {"-y", "--yes", "--assume-yes", "-qy", "-yq"})
    if sub == "update":
        return _apt_update(ctx)
    if sub == "install":
        if not pkgs:
            return APT_HEAD + "0 upgraded, 0 newly installed, 0 to remove and 12 not upgraded.\n"
        return _apt_install(ctx, pkgs, yes)
    if sub in ("remove", "purge"):
        return _apt_remove(ctx, pkgs, purge=sub == "purge" or "--purge" in flags)
    if sub == "list":
        have = installed_packages(node)
        rows = ["Listing... Done"]
        names = sorted(have) if "--installed" in flags else sorted(set(have) | set(CTK_PKGS))
        for p in names:
            if pkgs and not any(fnmatch.fnmatch(p, g) for g in pkgs):
                continue
            if p in have:
                rows.append(f"{p}/now {have[p][0]} amd64 [installed]")
            elif ctk_available(node):
                rows.append(f"{p}/unknown {_ctk_version()} amd64")
        return "\n".join(rows) + "\n"
    if sub in ("show", "policy"):
        return apt_cache(ctx, [sub] + pkgs)
    if sub in ("upgrade", "full-upgrade", "dist-upgrade", "autoremove"):
        return (
            APT_HEAD + "Calculating upgrade... Done\n0 upgraded, 0 newly installed, 0 to remove and 12 not upgraded.\n"
        )
    raise CommandError(f"E: Invalid operation {sub}", 100)


@command("apt-cache", help="apt-cache policy|show <pkg>")
def apt_cache(ctx, args):
    node = ctx.node
    pos = [a for a in args if not a.startswith("-")]
    if not pos:
        raise CommandError("E: No packages found", 100)
    sub, pkgs = pos[0], pos[1:]
    have = installed_packages(node)
    out = []
    for p in pkgs:
        known = p in have or (p in CTK_PKGS and ctk_available(node))
        if sub == "policy":
            if not known:
                out.append(f"N: Unable to locate package {p}")
                continue
            inst = have.get(p, ("(none)",))[0]
            cand = have[p][0] if p in have else _ctk_version()
            out.append(f"{p}:\n  Installed: {inst}\n  Candidate: {cand}\n  Version table:")
            src = (
                "https://nvidia.github.io/libnvidia-container/stable/deb/amd64  Packages"
                if p in CTK_PKGS
                else "http://archive.ubuntu.com/ubuntu jammy/main amd64 Packages"
            )
            out.append(f" {'***' if p in have else '   '} {cand} 500\n        500 {src}")
        else:
            if not known:
                raise CommandError(f"N: Unable to locate package {p}\nE: No packages found", 100)
            ver = have.get(p, (_ctk_version(),))[0]
            desc = (CTK_PKGS.get(p) or ("",))[0] or have.get(p, ("", ""))[1]
            out.append(f"Package: {p}\nVersion: {ver}\nArchitecture: amd64\nDescription: {desc}\n")
    return "\n".join(out) + "\n"


@command("dpkg", "dpkg-query", help="dpkg -l [pattern] — installed packages")
def dpkg(ctx, args):
    node = ctx.node
    if not args or args[0] not in ("-l", "--list", "-s", "--status"):
        raise CommandError("dpkg: need an action option (lab supports -l [pattern] and -s <pkg>)", 2)
    pats = args[1:] or ["*"]
    have = installed_packages(node)
    if args[0] in ("-s", "--status"):
        p = pats[0]
        if p not in have:
            raise CommandError(
                f"dpkg-query: package '{p}' is not installed and no information is available\n"
                "Use dpkg --info (= dpkg-deb --info) to examine archive files.",
                1,
            )
        return f"Package: {p}\nStatus: install ok installed\nArchitecture: amd64\nVersion: {have[p][0]}\n"
    hits = [(p, v) for p, v in sorted(have.items()) if any(fnmatch.fnmatch(p, g) for g in pats)]
    if not hits:
        raise CommandError(f"dpkg-query: no packages found matching {' '.join(pats)}", 1)
    rows = [
        "Desired=Unknown/Install/Remove/Purge/Hold",
        "| Status=Not/Inst/Conf-files/Unpacked/halF-conf/Half-inst/trig-aWait/Trig-pend",
        "|/ Err?=(none)/Reinst-required (Status,Err: uppercase=bad)",
        "||/ Name                          Version                         Architecture Description",
        "+++-=============================-===============================-============-====================",
    ]
    for p, (ver, desc) in hits:
        rows.append(f"ii  {p:<29} {ver:<31} amd64        {desc}")
    return "\n".join(rows) + "\n"
