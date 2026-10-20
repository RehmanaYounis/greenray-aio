"""Short, scenario-agnostic command guides shown by the lab's ``explain <command>``.

Guides live in ``aiolab/data/command_guides.yaml``. They teach what a tool is for and
how to read it; they must never describe how to solve a particular scenario.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

GUIDE_FILE = Path(__file__).resolve().parent.parent / "data" / "command_guides.yaml"
SECTIONS = (
    ("purpose", "What it does"),
    ("when", "When an engineer reaches for it"),
    ("options", "Useful options"),
    ("read", "What to look for in the output"),
    ("errors", "Errors you may see"),
)


@lru_cache(maxsize=1)
def load() -> dict[str, dict]:
    data = yaml.safe_load(GUIDE_FILE.read_text(encoding="utf-8")) or {}
    aliases = {}
    for entry in data.values():
        for alias in entry.get("aliases", []):
            aliases[alias] = entry
    return {**aliases, **data}


def names() -> list[str]:
    return sorted(yaml.safe_load(GUIDE_FILE.read_text(encoding="utf-8")) or {})


def guide(name: str) -> str | None:
    entry = load().get(name)
    if entry is None:
        return None
    lines = [f"{name}: {entry['summary']}"]
    for key, heading in SECTIONS:
        value = entry.get(key)
        if not value:
            continue
        lines.append(f"\n{heading}:")
        if isinstance(value, list):
            lines += [f"  {item}" for item in value]
        else:
            lines.append(f"  {value.strip()}")
    return "\n".join(lines) + "\n"
