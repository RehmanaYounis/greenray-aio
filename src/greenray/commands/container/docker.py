"""The docker command: run, exec, login, inspect, logs and compose."""

from __future__ import annotations

import base64
import hashlib
import json
import re

from ...core.errors import CommandError
from .. import command
from ..common import columns
from .images import (
    creds_valid,
    docker_creds,
    norm_image,
    parse_image,
    pull_error_text,
    registry_check,
    unquoted_oauthtoken,
)
from .runtime import DOCKER_AUTH, IMAGE_SIZES, OAUTH_USER, container_init, toolkit_installed
from .sandbox import exec_in_container


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
        return columns(rows)
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
        return columns(rows)
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
        return columns(rows)
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
