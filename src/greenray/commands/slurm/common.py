"""Constants, hostlist and time-spec parsing, and the output helpers shared by every Slurm command."""

from __future__ import annotations

import contextlib
import datetime as _dt
import re

from ..common import err, has, opt

SLURM_VERSION = "23.11.6"
BASE_TIME = _dt.datetime(2026, 10, 19, 9, 0, 0)
TICK = 30
CONF = "/etc/slurm/slurm.conf"
GRES_CONF = "/etc/slurm/gres.conf"
PLUGSTACK = "/etc/slurm/plugstack.conf"
HEAD = "bcm-head"
BATCH_DEFAULT_DURATION = 90
TERMINAL = ("COMPLETED", "FAILED", "CANCELLED", "NODE_FAIL", "TIMEOUT", "OUT_OF_MEMORY")
STATE_CODES = {
    "PENDING": "PD",
    "RUNNING": "R",
    "COMPLETED": "CD",
    "FAILED": "F",
    "CANCELLED": "CA",
    "NODE_FAIL": "NF",
    "TIMEOUT": "TO",
    "COMPLETING": "CG",
    "OUT_OF_MEMORY": "OOM",
    "SUSPENDED": "S",
    "REQUEUED": "RQ",
}
NOT_AVAIL_TEXT = "Nodes required for job are DOWN, DRAINED or reserved for jobs in higher priority partitions"

_GUARD = {"depth": 0}
# user identity of the current command (``su - alice -c "sbatch ..."``); root by default
_USER: list = ["root"]
# local task id of the job step task being executed (for per-task CPU binding)
_err, _has, _opt = err, has, opt  # the names the slurm modules share

_TASK: dict = {"local": None, "per_node": 1}
MUNGE_KEY = "/etc/munge/munge.key"
CORES_PER_SOCKET = 56


def _split_commas(expr: str) -> list[str]:
    out, buf, depth = [], "", 0
    for ch in expr:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(buf)
            buf = ""
        else:
            buf += ch
    if buf:
        out.append(buf)
    return [x.strip() for x in out if x.strip()]


def expand_hostlist(expr) -> list[str]:
    """dgx-[01-03,05] -> [dgx-01, dgx-02, dgx-03, dgx-05]"""
    if expr is None:
        return []
    if isinstance(expr, (list, tuple)):
        return [h for e in expr for h in expand_hostlist(e)]
    expr = str(expr).strip()
    if not expr or expr in ("(null)", "None"):
        return []
    out = []
    for part in _split_commas(expr):
        m = re.match(r"^(.*?)\[([^\]]+)\](.*)$", part)
        if not m:
            out.append(part)
            continue
        pre, body, post = m.groups()
        for rng in body.split(","):
            if "-" in rng:
                a, b = rng.split("-", 1)
                for i in range(int(a), int(b) + 1):
                    out.extend(expand_hostlist(f"{pre}{i:0{len(a)}d}{post}"))
            else:
                out.extend(expand_hostlist(pre + rng + post))
    return out


def compress_hostlist(names) -> str:
    names = list(dict.fromkeys(names))
    if not names:
        return ""
    groups, order, singles = {}, [], []
    for n in names:
        m = re.match(r"^(.*?)(\d+)$", n)
        if not m:
            singles.append(n)
            continue
        key = (m.group(1), len(m.group(2)))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(int(m.group(2)))
    parts = []
    for pre, w in order:
        nums = sorted(set(groups[(pre, w)]))
        if len(nums) == 1:
            parts.append(f"{pre}{nums[0]:0{w}d}")
            continue
        ranges, start, prev = [], nums[0], nums[0]
        for x in nums[1:] + [None]:
            if x is not None and x == prev + 1:
                prev = x
                continue
            ranges.append(f"{start:0{w}d}" if start == prev else f"{start:0{w}d}-{prev:0{w}d}")
            if x is not None:
                start = prev = x
        parts.append(f"{pre}[{','.join(ranges)}]")
    return ",".join(parts + singles)


def parse_time(spec) -> int | None:
    """Slurm time spec -> minutes (None = unlimited)."""
    if spec is None:
        return None
    if isinstance(spec, (int, float)):
        return int(spec)
    s = str(spec).strip()
    if s.upper() in ("UNLIMITED", "INFINITE", "", "NONE"):
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
        parts = [int(x) for x in s.split(":")]
        h, m, sec = (parts + [0, 0, 0])[:3]
        return days * 1440 + h * 60 + m + (1 if sec else 0)
    parts = [int(x) for x in s.split(":")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] + (1 if parts[1] else 0)
    return parts[0] * 60 + parts[1] + (1 if parts[2] else 0)


def fmt_limit(minutes) -> str:
    if minutes is None:
        return "infinite"
    d, rem = divmod(int(minutes), 1440)
    h, m = divmod(rem, 60)
    return f"{d}-{h:02d}:{m:02d}:00" if d else f"{h}:{m:02d}:00"


def fmt_limit_upper(minutes) -> str:
    if minutes is None:
        return "UNLIMITED"
    d, rem = divmod(int(minutes), 1440)
    h, m = divmod(rem, 60)
    return f"{d}-{h:02d}:{m:02d}:00" if d else f"{h:02d}:{m:02d}:00"


def fmt_elapsed(sec: int, long=False) -> str:
    sec = max(0, int(sec))
    d, rem = divmod(sec, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if long:
        return f"{d}-{h:02d}:{m:02d}:{s:02d}" if d else f"{h:02d}:{m:02d}:{s:02d}"
    if d:
        return f"{d}-{h:02d}:{m:02d}:{s:02d}"
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _ts(state, sec=None) -> str:
    if sec is None:
        sec = state.slurm.get("clock", 0)
    return (BASE_TIME + _dt.timedelta(seconds=int(sec))).strftime("%Y-%m-%dT%H:%M:%S")


def _banner_time(state) -> str:
    return (BASE_TIME + _dt.timedelta(seconds=int(state.slurm.get("clock", 0)))).strftime("%a %b %d %H:%M:%S %Y")


def parse_gres_spec(spec) -> tuple[str | None, int]:
    """'gpu:h100:8' -> ('h100', 8); 'gpu:8' -> (None, 8); '(null)' -> (None, 0)."""
    if not spec or str(spec) in ("(null)", "None"):
        return None, 0
    for tok in _split_commas(str(spec)):
        tok = re.sub(r"\(.*\)$", "", tok.strip())
        tok = tok.removeprefix("gres/").removeprefix("gres:")
        parts = tok.split(":")
        if parts[0] != "gpu":
            continue
        if len(parts) == 1:
            return None, 1
        if len(parts) == 2:
            return (None, int(parts[1])) if parts[1].isdigit() else (parts[1], 1)
        return parts[1], int(parts[2]) if parts[2].isdigit() else 1
    return None, 0


def parse_tres(spec) -> dict:
    """'gres/gpu=8,cpu=64' -> {'gres/gpu': 8, 'cpu': 64}"""
    out = {}
    for tok in str(spec or "").split(","):
        if "=" in tok:
            k, v = tok.split("=", 1)
            k = k.strip().lower()
            if k == "gpu":
                k = "gres/gpu"
            with contextlib.suppress(ValueError):
                out[k] = int(v)
    return out


def _kv(tokens) -> list[tuple[str, str]]:
    out = []
    for t in tokens:
        if "=" in t:
            k, v = t.split("=", 1)
            out.append((k, v))
        else:
            out.append((t, ""))
    return out


def _find_job(state, jid):
    jid = str(jid).split(".")[0].split("_")[0]
    for j in state.slurm.jobs:
        if str(j.get("id")) == jid:
            return j
    return None


_FMT_RX = re.compile(r"%(\.?)(\d*)([a-zA-Z])")


def _render(fmt, rows, header_map, noheader=False):
    specs = []
    pos = 0
    literals = []
    for m in _FMT_RX.finditer(fmt):
        literals.append(fmt[pos : m.start()])
        specs.append((m.group(1) == ".", int(m.group(2)) if m.group(2) else None, m.group(3)))
        pos = m.end()
    tail = fmt[pos:]

    def cell(val, right, width):
        val = str(val)
        if width is None:
            return val
        val = val[:width]
        return val.rjust(width) if right else val.ljust(width)

    out = []
    if not noheader:
        line = ""
        for lit, (right, width, letter) in zip(literals, specs):
            line += lit + cell(header_map.get(letter, letter.upper()), right, width)
        out.append((line + tail).rstrip())
    for r in rows:
        line = ""
        for lit, (right, width, letter) in zip(literals, specs):
            line += lit + cell(r.get(letter, ""), right, width)
        out.append((line + tail).rstrip())
    return "\n".join(out) + "\n"


def _ranges(nums):
    nums = sorted(nums)
    out, start, prev = [], nums[0], nums[0]
    for x in nums[1:] + [None]:
        if x is not None and x == prev + 1:
            prev = x
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        if x is not None:
            start = prev = x
    return out
