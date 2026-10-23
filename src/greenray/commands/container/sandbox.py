"""Running a command inside a container: which GPUs it sees and how the node answers."""

from __future__ import annotations

from types import SimpleNamespace

from ...core.errors import CommandError
from ...core.state import AttrDict, AttrList, wrap


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
    from ..nvidia_smi import nvidia_smi

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
