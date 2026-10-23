"""nvidia-ctk: configuring the runtimes and generating CDI specs."""

from __future__ import annotations

import json
import re

from ...core.errors import CommandError
from .. import command
from .runtime import (
    CDI_SPEC,
    CONTAINERD_CONF,
    CTK_VERSION,
    DOCKER_CONF,
    container_init,
    containerd_config,
    parse_containerd,
    toolkit_installed,
)


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
