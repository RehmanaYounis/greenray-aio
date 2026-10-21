"""Job step execution: batch scripts, srun tasks, the job environment and pyxis container imports."""

from __future__ import annotations

import math
import re
import shlex

from ...core.errors import CommandError
from .common import _GUARD, _TASK, HEAD, PLUGSTACK, compress_hostlist
from .conf import _plugstack_has_pyxis


def _expand_pattern(pat, j):
    return (
        pat.replace("%j", str(j.id))
        .replace("%x", j.name)
        .replace("%u", j.user)
        .replace("%N", j.nodelist[0] if j.nodelist else "node")
        .replace("%A", str(j.id))
    )


def _job_env(j, node_name):
    return {
        "SLURM_JOB_ID": str(j.id),
        "SLURM_JOBID": str(j.id),
        "SLURM_JOB_NAME": j.name,
        "SLURM_JOB_NODELIST": compress_hostlist(j.nodelist),
        "SLURM_NNODES": str(j.num_nodes),
        "SLURM_JOB_NUM_NODES": str(j.num_nodes),
        "SLURM_NTASKS": str(j.num_nodes * j.ntasks_per_node),
        "SLURM_NTASKS_PER_NODE": str(j.ntasks_per_node),
        "SLURM_JOB_PARTITION": j.partition,
        "SLURM_SUBMIT_DIR": j.work_dir,
        "SLURMD_NODENAME": node_name,
        "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(len(j.alloc_gpus.get(node_name, [])))),
        "SLURM_JOB_GPUS": ",".join(str(i) for i in j.alloc_gpus.get(node_name, [])),
        "SLURM_JOB_ACCOUNT": j.account,
        "SLURM_JOB_USER": j.user,
        "SLURM_CPUS_ON_NODE": str(2 * len((j.get("alloc_cores") or {}).get(node_name, [])) or j.ntasks_per_node),
    }


def _run_batch(state, j):
    """Execute the batch script on the first allocated node and write its output file."""
    from ...core.shell import Shell

    host = j.nodelist[0]
    node = state.nodes.get(host)
    if node is None:
        return
    env = _job_env(j, host)
    saved = dict(node.env)
    node.env.update(env)
    out = []
    _GUARD["depth"] += 1
    failed = False
    try:
        from ..storage import path_available

        if not path_available(state, host, j.work_dir):
            out.append(
                f"slurmstepd: error: couldn't chdir to `{j.work_dir}': No such file or directory: going to /tmp instead"
            )
        sh = Shell(state, host)
        sh.ctx.cwd = j.work_dir
        _CURRENT.append(j)
        for line in j.script.splitlines()[1:]:
            st = line.strip()
            if not st or st.startswith("#"):
                continue
            if st.startswith(("set ", "module ", "source ", "cd ")):
                continue
            simple = not any(c in st for c in "|;&><$`") and not st.startswith(("srun", "export"))
            if simple:
                try:
                    argv = shlex.split(st)
                except ValueError:
                    argv = []
                res, code = _exec_task(_FakeCtx(state, host), j, host, argv, bool(j.container_image))
                if code:
                    failed = True
                    j.final_exit = f"{code}:0"
                if argv and argv[0] == "exit":
                    if res:
                        out.append(res)
                    break
            else:
                res = sh.run(st)
                if "Exited with exit code" in res or "command not found" in res:
                    failed = True
            if res:
                out.append(res)
    except Exception as e:  # never let a script crash the simulator
        out.append(f"slurmstepd: error: {e}")
        failed = True
    finally:
        _GUARD["depth"] -= 1
        if _CURRENT and _CURRENT[-1] is j:
            _CURRENT.pop()
        node.env.clear()
        node.env.update(saved)
    if failed:
        j.final_exit = j.get("final_exit") or "1:0"
        j.final_state = "FAILED"
    text = "\n".join(out) + ("\n" if out else "")
    sub = state.nodes.get(j.submit_host) or state.nodes.get(HEAD)
    sub.files[j.stdout] = text
    if host != j.submit_host:
        node.files[j.stdout] = text


_CURRENT: list = []


class _FakeCtx:
    def __init__(self, state, host):
        self.state = state
        self.host = host
        self.host_stack = []


def _container_python(code, j, gpus_here):
    out = []
    for m in re.finditer(r"print\((.*?)\)(?:;|$)", code):
        expr = m.group(1).strip()
        vals = []
        for part in [x.strip() for x in expr.split(",")]:
            if part == "torch.__version__":
                vals.append("2.4.0a0+07cecf4168.nv24.05")
            elif part == "torch.cuda.device_count()":
                vals.append(str(gpus_here))
            elif part == "torch.cuda.is_available()":
                vals.append(str(gpus_here > 0))
            elif part == "torch.version.cuda":
                vals.append("12.4")
            elif part.startswith("torch.cuda.get_device_name"):
                vals.append("NVIDIA H100 80GB HBM3")
            elif part.startswith(("'", '"')):
                vals.append(part.strip("'\""))
            else:
                vals.append(part)
        out.append(" ".join(vals))
    return "\n".join(out)


def _exec_task(ctx, j, host, argv, container):
    """Run one task's command on ``host``. Returns (output, exit_code)."""
    from ...core.shell import Context
    from .. import dispatch

    state = ctx.state
    node = state.nodes[host]
    gidx = (j.alloc_gpus or {}).get(host, [])
    constrain = (
        "ConstrainDevices=yes" in state.nodes[HEAD].files.get("/etc/slurm/cgroup.conf", "")
        if HEAD in state.nodes
        else True
    )
    if not argv:
        return "", 0
    prog = argv[0].rsplit("/", 1)[-1]
    if prog in ("true", ":", "sleep", "wait"):
        return "", 0
    if prog == "false":
        return "", 1
    if prog == "exit":
        return "", int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else 0
    if prog == "nvidia-smi" and argv[1:] in (["-L"], ["--list-gpus"]):
        if not node.get("driver_loaded", True):
            return (
                "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. "
                "Make sure that the latest NVIDIA driver is installed and running."
            ), 9
        gpus = [node.gpus[i] for i in gidx if i < len(node.gpus)] if constrain else list(node.gpus)
        if not gpus and constrain:
            return "No devices found.", 6
        lines = []
        for k, g in enumerate(gpus):
            if g.fallen_off_bus:
                lines.append(f"Unable to determine the device handle for GPU{g.bus_id}: Unknown Error")
                continue
            lines.append(f"GPU {k if constrain else g.index}: {g.name} (UUID: {g.uuid})")
        return "\n".join(lines), 0
    if prog in ("python", "python3") and "-c" in argv:
        code = argv[argv.index("-c") + 1] if argv.index("-c") + 1 < len(argv) else ""
        if "torch" in code and not container:
            return (
                'Traceback (most recent call last):\n  File "<string>", line 1, in <module>\n'
                "ModuleNotFoundError: No module named 'torch'"
            ), 1
        return _container_python(code, j, len(gidx)), 0
    if container and prog == "cat" and argv[1:] == ["/etc/os-release"]:
        return 'PRETTY_NAME="Ubuntu 22.04.4 LTS"\nNAME="Ubuntu"\nVERSION_ID="22.04"', 0
    if prog in ("bash", "sh") and "-c" in argv:
        from ...core.shell import Shell

        sh = Shell(state, host)
        return sh.run(argv[argv.index("-c") + 1]), 0
    if prog == "env" and not argv[1:]:
        env = dict(node.env)
        env.update(_job_env(j, host))
        return "\n".join(f"{k}={v}" for k, v in sorted(env.items())), 0
    if prog == "printenv" and argv[1:]:
        env = _job_env(j, host)
        env.update({k: v for k, v in node.env.items()})
        return "\n".join(env.get(a, "") for a in argv[1:]), 0
    sub = Context(state=state, host=host)
    sub.cwd = j.work_dir
    saved = dict(node.env)
    node.env.update(_job_env(j, host))
    try:
        out = dispatch(sub, argv)
        code = 0
    except CommandError as e:
        out, code = str(e), e.code
        if "command not found" in out:
            out = f"slurmstepd: error: execve(): {argv[0]}: No such file or directory"
            code = 2
    finally:
        node.env.clear()
        node.env.update(saved)
    return (out or "").rstrip("\n"), code


def _pyxis_import(state, host, image):
    m = re.match(r"^(?:(?P<user>[^@#/]+)@)?(?:(?P<reg>[^#]+)#)?(?P<img>[^:#]+)(?::(?P<tag>[^#]+))?$", image or "")
    lines = [f"pyxis: importing docker image: {image}"]
    if image.endswith(".sqsh") or image.startswith("/"):
        node = state.nodes[host]
        if image in node.files or image in node.get("enroot_images", []):
            return True, []
        return False, [
            f"slurmstepd: error: pyxis: unable to open squashfs image {image}: No such file or directory",
            "slurmstepd: error: pyxis: couldn't start container",
            "slurmstepd: error: spank: required plugin spank_pyxis.so: task_init() failed with rc=-1",
            "slurmstepd: error: Failed to invoke spank plugin stack",
        ]
    if not m:
        return False, lines + ["slurmstepd: error: pyxis: invalid image name"]
    reg = m.group("reg") or "registry-1.docker.io"
    img = m.group("img")
    tag = m.group("tag") or "latest"
    catalog = state.get("ngc", {}).get("catalog", {})
    ok = True
    err = None
    if reg == "nvcr.io":
        if img not in catalog:
            ok = False
            err = (
                f"URL https://nvcr.io/proxy_auth?scope=repository:{img}:pull returned error code: 401 Unauthorized"
                if not img.startswith("nvidia/")
                else f"URL https://nvcr.io/v2/{img}/manifests/{tag} returned error code: 404 Not Found"
            )
        elif tag not in catalog[img]:
            ok = False
            err = f"URL https://nvcr.io/v2/{img}/manifests/{tag} returned error code: 404 Not Found"
    elif reg == "registry-1.docker.io":
        first = img.split("/", 1)[0]
        if "." in first:  # e.g. nvcr.io/nvidia/pytorch: registry written with '/' instead of '#'
            ok = False
            err = f"URL https://registry-1.docker.io/v2/{img}/manifests/{tag} returned error code: 401 Unauthorized"
    if ok:
        node = state.nodes[host]
        node.setdefault("enroot_images", [])
        if image not in node.enroot_images:
            node.enroot_images.append(image)
        return True, [lines[0], f"pyxis: imported docker image: {image}"]
    return False, lines + [
        f"slurmstepd: error: pyxis: child {23000 + len(image)} failed with error code: 1",
        "slurmstepd: error: pyxis: failed to import docker image",
        "slurmstepd: error: pyxis: printing enroot log file:",
        "slurmstepd: error: pyxis:     [INFO] Querying registry for permission grant",
        "slurmstepd: error: pyxis:     [INFO] Authenticating with user: <anonymous>",
        "slurmstepd: error: pyxis:     [INFO] Authentication succeeded",
        "slurmstepd: error: pyxis:     [INFO] Fetching image manifest list",
        f"slurmstepd: error: pyxis:     [ERROR] {err}",
        "slurmstepd: error: pyxis: couldn't start container",
        "slurmstepd: error: spank: required plugin spank_pyxis.so: task_init() failed with rc=-1",
        "slurmstepd: error: Failed to invoke spank plugin stack",
    ]


def _run_step(ctx, j, opts, argv):
    """Run a job step on the job's nodes. Returns (output, rc)."""
    state = ctx.state
    label = opts.get("label") or opts.get("l")
    nodes = list(j.nodelist)
    if "nodes" in opts:
        nodes = nodes[: int(str(opts["nodes"]).split("-")[0])]
    tpn = int(opts["ntasks_per_node"]) if "ntasks_per_node" in opts else None
    if "ntasks" in opts:
        total = int(opts["ntasks"])
    else:
        total = len(nodes) * (tpn or j.ntasks_per_node)
    per = tpn or max(1, math.ceil(total / max(1, len(nodes))))
    image = opts.get("container_image") or (j.container_image if not j.batch else "")
    out = []
    rc = 0
    step_id = len(j.steps)
    task = 0
    for host in nodes:
        node = state.nodes[host]
        if image:
            if not _plugstack_has_pyxis(node.files) or node.get("pyxis_installed", True) is False:
                out.append(
                    f'slurmstepd: error: spank: plugin "pyxis" is not loaded on {host} '
                    f"(check {PLUGSTACK}); option --container-image cannot be honoured"
                )
                out.append(f"srun: error: {host}: task {task}: Exited with exit code 1")
                rc = 1
                task += per
                continue
            ok, msgs = _pyxis_import(state, host, image)
            out.extend(msgs)
            if not ok:
                for t in range(per):
                    out.append(f"srun: error: {host}: task {task + t}: Exited with exit code 1")
                rc = 1
                task += per
                continue
        for k in range(per):
            if task >= total:
                break
            _TASK.update({"local": k, "per_node": per})
            try:
                text, code = _exec_task(ctx, j, host, argv, bool(image))
            finally:
                _TASK.update({"local": None, "per_node": 1})
            if text:
                out.extend((f"{task}: {l}" if label else l) for l in text.split("\n"))
            if code:
                out.append(f"srun: error: {host}: task {task}: Exited with exit code {code}")
                rc = code
            task += 1
    j.steps.append(
        {
            "id": str(step_id),
            "name": argv[0] if argv else "",
            "state": "FAILED" if rc else "COMPLETED",
            "exit_code": f"{rc}:0",
            "nodes": compress_hostlist(nodes),
            "container_image": image,
        }
    )
    if image and not j.container_image:
        j.container_image = image
    return "\n".join(out), rc
