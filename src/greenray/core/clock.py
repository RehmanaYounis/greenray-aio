"""Simulated wall clock: starts at the real time of the session and moves a few seconds per command.

Commands that print dates read it, so a lab looks like it is happening now rather
than on the day the scenario was written.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

BOOT_AGE = timedelta(days=12, hours=3, minutes=11)  # how long the nodes have been "up"
TICK = 2.0  # seconds a typical command takes


def init(state) -> None:
    state.flags.setdefault("clock", time.time())


def now(state) -> datetime:
    init(state)
    return datetime.fromtimestamp(state.flags["clock"])


def advance(state, seconds: float = TICK) -> None:
    init(state)
    state.flags["clock"] += seconds


def booted(state) -> datetime:
    return now(state) - BOOT_AGE


def uptime_text(state) -> str:
    """``12 days,  3:11`` in the format ``uptime`` uses."""
    age = now(state) - booted(state)
    hours, rem = divmod(age.seconds, 3600)
    return f"{age.days} days, {hours:2d}:{rem // 60:02d}"


def stamp(state, fmt: str = "%a %b %d %H:%M:%S %Y") -> str:
    return now(state).strftime(fmt)
