"""Simulated wall clock: the day every scenario is set on, moving a few seconds per command.

Slurm, kubectl, cmsh and the scenario journals all date their output 19 October 2026, so
``date``, ``uptime`` and the login banner start there too instead of at the real time; a learner
comparing a journal line with ``date`` sees one day, not two.
"""

from __future__ import annotations

from datetime import datetime, timedelta

START = datetime(2026, 10, 19, 10, 15, 13)  # the same instant cmsh stamps its events with
BOOT_AGE = timedelta(days=12, hours=3, minutes=11)  # how long the nodes have been "up"
TICK = 2.0  # seconds a typical command takes


def init(state) -> None:
    state.flags.setdefault("clock", START.timestamp())


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
