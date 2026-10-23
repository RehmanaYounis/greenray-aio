"""enroot: import, create, list, start and remove squashfs containers."""

from __future__ import annotations

import re

from ...core.errors import CommandError
from .. import command
from .images import parse_image, registry_check
from .runtime import ENROOT_CREDS, container_init
from .sandbox import exec_in_container


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
        elif registry_check(ctx.state, ref, None):
            raise CommandError(
                "\n".join(
                    lines
                    + [
                        "[INFO] Authenticating with user: <anonymous>",
                        f"[ERROR] URL https://registry-1.docker.io/v2/{repo}/manifests/{tag} returned error code: 404 Not Found",
                    ]
                ),
                1,
            )
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
