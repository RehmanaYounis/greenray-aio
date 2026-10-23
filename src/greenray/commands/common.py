"""Small helpers every command module needs: option lookup, errors and plain tables."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..core.errors import CommandError


def opt(args: Sequence[str], *names: str, default=None):
    """Value of ``-x VAL``, ``-xVAL``, ``--long VAL`` or ``--long=VAL``; ``default`` when absent."""
    for i, a in enumerate(args):
        for n in names:
            if a == n and i + 1 < len(args):
                return args[i + 1]
            if n.startswith("--") and a.startswith(n + "="):
                return a.split("=", 1)[1]
            if not n.startswith("--") and a.startswith(n) and len(a) > len(n) and a[len(n)] != "-":
                return a[len(n) :]
    return default


def has(args: Sequence[str], *flags: str) -> bool:
    return any(a in flags for a in args)


def err(msg: str, code: int = 1) -> CommandError:
    return CommandError(msg, code)


def columns(rows: Sequence[Sequence[object]], gap: int = 3) -> str:
    """Left-aligned columns separated by ``gap`` spaces, the layout of kubectl and docker listings."""
    if not rows:
        return ""
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    out = []
    for r in rows:
        cells = [str(c).ljust(widths[i] + gap) if i < len(r) - 1 else str(c) for i, c in enumerate(r)]
        out.append("".join(cells).rstrip())
    return "\n".join(out) + "\n"


def dashed_table(headers: Sequence[str], rows: Iterable[Sequence[object]], widths: list[int] | None = None) -> str:
    """A header row underlined with dashes, the layout of cmsh listings."""
    rows = list(rows)
    widths = widths or [
        max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h)) for i, h in enumerate(headers)
    ]
    widths = [max(w, len(h)) for w, h in zip(widths, headers)]
    line = " ".join(f"{h:<{w}}" for h, w in zip(headers, widths)).rstrip()
    sep = " ".join("-" * w for w in widths)
    body = [" ".join(f"{str(c):<{w}}" for c, w in zip(r, widths)).rstrip() for r in rows]
    return "\n".join([line, sep, *body])
