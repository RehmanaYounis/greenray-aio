"""apt, apt-get, apt-cache and dpkg for the container toolkit packages."""

from __future__ import annotations

import fnmatch

from ...core.errors import CommandError
from .. import command
from .runtime import CTK_VERSION, toolkit_installed

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
