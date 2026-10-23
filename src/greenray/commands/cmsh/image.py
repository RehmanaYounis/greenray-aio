"""chroot into a software image, with apt working on the image's package list."""

from __future__ import annotations

import re
import shlex

from ...core.errors import CommandError
from ...core.state import HEAD
from .. import command
from .model import REPO, _kernels, bcm_init, image_files


class _ImgNode(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None


class _ImgCtx:
    """A minimal Context whose filesystem is a software image directory."""

    def __init__(self, ctx, files):
        self.state = ctx.state
        self.host = ctx.host
        self.stdin = ctx.stdin
        self.cwd = "/"
        self.executed = ctx.executed
        self.host_stack = []
        self.node = _ImgNode(
            files=files,
            env={},
            up=True,
            gpus=[],
            ib=[],
            services={},
            modules=[],
            kernel=ctx.node.kernel,
            hostname=ctx.host,
        )


def _apt(img, args):
    pkgs = img.packages
    action = next((a for a in args if not a.startswith("-")), "")
    names = [a for a in args if not a.startswith("-")][1:]
    if action in ("update",):
        return (
            "Hit:1 http://archive.ubuntu.com/ubuntu jammy InRelease\n"
            "Hit:2 https://repo.download.nvidia.com/baseos/ubuntu/jammy/x86_64 dgx InRelease\n"
            "Reading package lists... Done\n"
        )
    if action == "list":
        return "".join(f"{p}/jammy,now {v} amd64 [installed]\n" for p, v in sorted(pkgs.items()))
    if action in ("install", "upgrade"):
        out = ["Reading package lists... Done", "Building dependency tree... Done"]
        changes = []
        for spec in names:
            name, _, ver = spec.partition("=")
            if name not in REPO:
                raise CommandError("\n".join(out + [f"E: Unable to locate package {name}"]), 100)
            versions = REPO[name]
            ver = ver or versions[-1]
            if ver not in versions:
                raise CommandError("\n".join(out + [f"E: Version '{ver}' for '{name}' was not found"]), 100)
            changes.append((name, pkgs.get(name), ver))
            if name == "nvidia-container-toolkit" and "nvidia-container-toolkit-base" in pkgs:
                changes.append(("nvidia-container-toolkit-base", pkgs.get("nvidia-container-toolkit-base"), ver))
        if not changes or all(o == n for _, o, n in changes):
            return "\n".join(out + ["0 upgraded, 0 newly installed, 0 to remove and 0 not upgraded."]) + "\n"
        up = [c for c in changes if c[1] and c[1] != c[2]]
        new = [c for c in changes if not c[1]]
        if up:
            out += ["The following packages will be upgraded:", "  " + " ".join(c[0] for c in up)]
        if new:
            out += ["The following NEW packages will be installed:", "  " + " ".join(c[0] for c in new)]
        out.append(f"{len(up)} upgraded, {len(new)} newly installed, 0 to remove and 0 not upgraded.")
        for name, old, ver in changes:
            out.append(f"Unpacking {name} ({ver}){' over (' + old + ')' if old else ''} ...")
        for name, _old, ver in changes:
            pkgs[name] = ver
            out.append(f"Setting up {name} ({ver}) ...")
            if name.startswith("linux-image-"):
                kver = name[len("linux-image-") :]
                out.append(f"update-initramfs: Generating /boot/initrd.img-{kver}")
                out.append(
                    "W: Not updating the boot loader inside a chroot; BCM generates the ramdisk "
                    "when the image's kernelversion is set and committed"
                )
        return "\n".join(out) + "\n"
    if action in ("remove", "purge"):
        for n in names:
            pkgs.pop(n, None)
        return f"Removing {' '.join(names)} ...\n"
    raise CommandError(f"E: Invalid operation {action}", 100)


@command(
    "chroot", "cm-chroot-sw-img", help="chroot /cm/images/<image> <command> — run a command inside a BCM software image"
)
def chroot(ctx, args):
    if not args:
        raise CommandError("chroot: missing operand\nUsage: chroot NEWROOT COMMAND [ARG]...", 125)
    path = args[0].rstrip("/")
    bcm = bcm_init(ctx.state)
    img_name = next((n for n, i in bcm.softwareimages.items() if i.path.rstrip("/") == path), None)
    if ctx.host != HEAD or img_name is None:
        raise CommandError(f"chroot: cannot change root directory to '{args[0]}': No such file or directory", 125)
    img = bcm.softwareimages[img_name]
    cmd = args[1:]
    if not cmd:
        raise CommandError(
            "chroot: interactive shells are not supported in the lab; give the command to run, e.g.\n"
            f"  cm-chroot-sw-img {path} apt-get install -y <package>",
            1,
        )
    name = cmd[0]
    if name in ("bash", "sh") and len(cmd) >= 3 and cmd[1] == "-c":
        cmd = shlex.split(cmd[2])
        name = cmd[0]
    if name in ("apt-get", "apt"):
        return _apt(img, cmd[1:])
    if name in ("dpkg", "dpkg-query"):
        pat = next((a for a in cmd[1:] if not a.startswith("-")), "")
        rows = [
            "Desired=Unknown/Install/Remove/Purge/Hold",
            "||/ Name                              Version                       Architecture Description",
            "+++-=================================-=============================-============-===========",
        ]
        hits = [
            f"ii  {p:<33} {v:<29} amd64        "
            for p, v in sorted(img.packages.items())
            if not pat or re.search(pat.replace("*", ".*"), p)
        ]
        if pat and not hits:
            raise CommandError(f"dpkg-query: no packages found matching {pat}", 1)
        return "\n".join(rows + hits) + "\n"
    if name == "uname":
        # classic trap: uname inside a chroot reports the *running* (head node) kernel
        return (ctx.node.kernel if "-r" in cmd else f"Linux {ctx.host} {ctx.node.kernel} x86_64 GNU/Linux") + "\n"
    if name == "ls" and len(cmd) > 1 and cmd[-1].rstrip("/") == "/lib/modules":
        return "\n".join(_kernels(img.packages)) + "\n"
    if name == "nvidia-ctk" and "--version" in cmd:
        v = img.packages.get("nvidia-container-toolkit", "").split("-")[0]
        if not v:
            raise CommandError("bash: nvidia-ctk: command not found", 127)
        return f"NVIDIA Container Toolkit CLI version {v}\ncommit: 4f6dbd4\n"
    if name == "exit":
        return ""
    # generic file tools operate on the image's files
    from .. import dispatch

    prefix = img.path.rstrip("/")
    head = ctx.state.nodes[HEAD].files
    before = image_files(ctx.state, img_name)
    view = dict(before)
    sub = _ImgCtx(ctx, view)
    out = dispatch(sub, cmd)
    for p in set(before) - set(view):
        head.pop(prefix + p, None)
    for p, c in view.items():
        if before.get(p) != c:
            head[prefix + p] = c
    return out
