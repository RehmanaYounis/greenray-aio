"""The container runtime on a node: containerd and docker configuration, what the daemon loaded, and the toolkit."""

from __future__ import annotations

import json
import re

from .. import HOOKS_REBOOT, HOOKS_SERVICE

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
                from .docker import start_container  # docker builds on this module, so the import stays local

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
