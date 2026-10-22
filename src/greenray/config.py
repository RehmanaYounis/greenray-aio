"""Runtime settings from ``GREENRAY_<NAME>`` environment variables."""

from __future__ import annotations

import os
from pathlib import Path

PREFIX = "GREENRAY_"


def env(name: str, default: str | None = None) -> str | None:
    """Value of ``GREENRAY_<name>``, else ``default``. Empty values count as unset."""
    return os.environ.get(PREFIX + name) or default


def home() -> Path:
    """Directory for local progress and history (``~/.greenray``), created on first use."""
    explicit = env("HOME")
    base = Path(explicit) if explicit else Path.home() / ".greenray"
    base.mkdir(parents=True, exist_ok=True)
    return base


def debug() -> bool:
    return bool(env("DEBUG"))
