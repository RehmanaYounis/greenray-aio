"""Munge authentication: key agreement between slurmctld and each slurmd, and the munge CLI tools."""

from __future__ import annotations

import re

from .. import command
from .common import HEAD, MUNGE_KEY, TICK, _err, _ts


def _munge_key(state, host):
    """The key munged on ``host`` is running with. None = munge not modelled on this host."""
    node = state.nodes.get(host)
    if node is None or "munge" not in node.services:
        return None
    if node.services.get("munge") != "active":
        return "<munged down>"
    return node.get("munge_key_loaded", node.files.get(MUNGE_KEY))


def _munge_ok(state, host) -> bool:
    """Can slurmctld (on the head node) and ``host`` decode each other's credentials?"""
    if host == HEAD:
        return True
    hk, nk = _munge_key(state, HEAD), _munge_key(state, host)
    if hk is None or nk is None:
        return True
    return hk == nk and hk != "<munged down>"


def _key_tag(key) -> str:
    import hashlib

    return hashlib.sha256(str(key).encode()).hexdigest()[:12]


def _munge_log(state, host):
    """Journal evidence for a node whose credentials slurmctld cannot decode."""
    node = state.nodes.get(host)
    if node is None:
        return
    ts = _ts(state).replace("T", " ")
    pid = 4100 + (state.slurm.clock // TICK) % 800
    sd = node.journal.setdefault("slurmd", [])
    if node.services.get("munge") != "active":
        sd.append(
            f'{ts} {host} slurmd[{pid}]: slurmd: error: Munge encode failed: Failed to access "/run/munge/munge.socket.2": No such file or directory'
        )
        sd.append(f"{ts} {host} slurmd[{pid}]: slurmd: error: Unable to register: Protocol authentication error")
        return
    sd.append(f"{ts} {host} slurmd[{pid}]: slurmd: error: Munge decode failed: Invalid credential")
    sd.append(
        f"{ts} {host} slurmd[{pid}]: slurmd: error: slurm_unpack_received_msg: auth_g_verify: REQUEST_PING has authentication error: Invalid authentication credential"
    )
    sd.append(f"{ts} {host} slurmd[{pid}]: slurmd: error: Unable to register: Zero Bytes were transmitted or received")
    if HEAD in state.nodes:
        hj = state.nodes[HEAD].journal.setdefault("slurmctld", [])
        hj.append(f"{ts} {HEAD} slurmctld[2211]: error: Munge decode failed: Invalid credential")
        hj.append(
            f"{ts} {HEAD} slurmctld[2211]: error: slurm_unpack_received_msg: auth_g_verify: "
            f"MESSAGE_NODE_REGISTRATION_STATUS has authentication error: Invalid authentication credential"
        )
        hj.append(
            f"{ts} {HEAD} slurmctld[2211]: error: slurm_unpack_received_msg: [[{host}]:{40000 + pid}] "
            f"Protocol authentication error"
        )


@command("munge", help="munge -n — encode an empty credential (pipe into unmunge to test the key)")
def munge_cmd(ctx, args):
    import base64

    if "-V" in args or "--version" in args:
        return "munge-0.5.14 (2020-01-14)\n"
    key = _munge_key(ctx.state, ctx.host)
    if key is None:
        key = ctx.node.files.get(MUNGE_KEY, "default-munge-key")
    if key == "<munged down>":
        raise _err('munge: Error: Unable to access "/run/munge/munge.socket.2": No such file or directory', 6)
    payload = "" if any(a in ("-n", "--no-input") for a in args) else (ctx.stdin or "")
    blob = base64.b64encode(
        f"{_key_tag(key)}|{ctx.host}|{ctx.state.slurm.get('clock', 0)}|{len(payload)}".encode()
    ).decode()
    return f"MUNGE:AwQFAAB{blob}:\n"


@command("unmunge", help="unmunge — decode a credential from stdin (munge -n | unmunge, ssh node munge -n | unmunge)")
def unmunge_cmd(ctx, args):
    import base64

    data = (ctx.stdin or "").strip()
    m = re.search(r"MUNGE:AwQFAAB([A-Za-z0-9+/=]+):", data)
    if not m:
        raise _err("unmunge: Error: No credential specified", 2)
    try:
        tag, src, _, length = base64.b64decode(m.group(1)).decode().split("|", 3)
    except Exception:
        raise _err("unmunge: Error: Invalid credential format", 9) from None
    key = _munge_key(ctx.state, ctx.host)
    if key is None:
        key = ctx.node.files.get(MUNGE_KEY, "default-munge-key")
    if key == "<munged down>":
        raise _err('unmunge: Error: Unable to access "/run/munge/munge.socket.2": No such file or directory', 6)
    src_ip = ctx.state.nodes[src].ip if src in ctx.state.nodes else "unknown"
    if tag != _key_tag(key):
        raise _err("unmunge: Error: Invalid credential", 15)
    ts = _ts(ctx.state).replace("T", " ")
    return (
        "STATUS:          Success (0)\n"
        f"ENCODE_HOST:     {src} ({src_ip})\n"
        f"ENCODE_TIME:     {ts} +0000\n"
        f"DECODE_TIME:     {ts} +0000\n"
        "TTL:             300\n"
        "CIPHER:          aes128 (4)\n"
        "MAC:             sha256 (5)\n"
        "ZIP:             none (0)\n"
        "UID:             root (0)\n"
        "GID:             root (0)\n"
        f"LENGTH:          {length}\n\n"
    )


@command("mungekey", "create-munge-key", help="mungekey --create [--force] — generate a NEW munge key (rotates it!)")
def mungekey_cmd(ctx, args):
    import hashlib

    force = any(a in ("-f", "--force") for a in args)
    if MUNGE_KEY in ctx.node.files and not force:
        raise _err(f'mungekey: Error: Failed to create "{MUNGE_KEY}": File exists', 1)
    seed = f"{ctx.host}-{ctx.state.slurm.get('clock', 0)}-{len(ctx.executed)}"
    ctx.node.files[MUNGE_KEY] = "munge-key:" + hashlib.sha256(seed.encode()).hexdigest() + "\n"
    return ""
