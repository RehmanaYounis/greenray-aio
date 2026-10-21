"""Runtime settings from the environment.

Every setting is read from ``GREENRAY_<NAME>`` first and falls back to the legacy
``AIOLAB_<NAME>`` (the project's former name), so existing setups keep working.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

PREFIX = "GREENRAY_"
LEGACY_PREFIX = "AIOLAB_"


def env(name: str, default: str | None = None) -> str | None:
    """Value of ``GREENRAY_<name>``, else ``AIOLAB_<name>``, else ``default``."""
    for prefix in (PREFIX, LEGACY_PREFIX):
        value = os.environ.get(prefix + name)
        if value:
            return value
    return default


def home() -> Path:
    """Directory for local progress (``~/.greenray``), created on first use.

    If only the legacy ``~/.aiolab`` exists, its contents are copied over once.
    """
    explicit = env("HOME")
    base = Path(explicit) if explicit else Path.home() / ".greenray"
    if not explicit and not base.exists():
        legacy = Path.home() / ".aiolab"
        if legacy.is_dir():
            shutil.copytree(legacy, base)
    base.mkdir(parents=True, exist_ok=True)
    return base


def debug() -> bool:
    return bool(env("DEBUG"))
