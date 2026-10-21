"""Realistic ``--help`` / ``help`` / ``man`` pages for simulated commands.

Layout (one folder per command, so pages can be added independently)::

    data/help/<command>/help.txt      what `<command> --help` prints
    data/help/<command>/<sub>.txt     what `<command> <sub> --help` prints (kubectl get, helm install ...)
    data/help/<command>/man.txt       optional longer page for `man <command>`
    data/help/<command>/meta.yaml     optional: {flags: [--help, -h], aliases: [k], wrapper: true, value_options: [-p, -N]}

Help flags are per tool because ``-h`` means something else in many tools
(``sinfo -h`` hides the header, ``df -h`` prints human-readable sizes).

Arguments after ``--`` never count. Commands that run another command
(``ssh``, ``chroot``, ``srun``, ``mpirun`` ...) are ``wrapper: true``: only flags
before the wrapped command count, so ``ssh node df -h`` runs ``df -h`` remotely.
Pages are original text in each tool's usual format and list the options this
simulator supports.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml

HELP_DIR = Path(__file__).resolve().parent.parent / "data" / "help"
DEFAULT_FLAGS = ("--help",)


@cache
def _index() -> dict[str, Path]:
    """Command name (and aliases) -> its help folder."""
    out: dict[str, Path] = {}
    if not HELP_DIR.is_dir():
        return out
    for folder in sorted(p for p in HELP_DIR.iterdir() if p.is_dir()):
        out[folder.name] = folder
        for alias in _meta(folder).get("aliases", []):
            out.setdefault(alias, folder)
    return out


@cache
def _meta(folder: Path) -> dict:
    meta = folder / "meta.yaml"
    return (yaml.safe_load(meta.read_text(encoding="utf-8")) or {}) if meta.is_file() else {}


def commands_with_help() -> list[str]:
    return sorted(_index())


def _read(folder: Path, name: str) -> str | None:
    path = folder / f"{name}.txt"
    return path.read_text(encoding="utf-8") if path.is_file() else None


def page(command: str, sub: str | None = None) -> str | None:
    folder = _index().get(command)
    if folder is None:
        return None
    return _read(folder, sub) if sub else _read(folder, "help")


def help_request(argv: list[str]) -> str | None:
    """The page to print if ``argv`` asks for help, else ``None`` (run the command normally)."""
    command, args = argv[0], argv[1:]
    folder = _index().get(command)
    if folder is None or not args:
        return None
    meta = _meta(folder)
    flags = tuple(meta.get("flags", DEFAULT_FLAGS))
    if "--" in args:  # everything after -- belongs to another program
        args = args[: args.index("--")]
    if not args:
        return None
    if meta.get("wrapper"):
        args = _before_wrapped_command(args, set(meta.get("value_options", [])))
        if not args:
            return None
    positional = [a for a in args if not a.startswith("-")]
    # `kubectl help get`, `helm help install`, `systemctl help`
    if args[0] == "help" and _read(folder, "help") is not None:
        sub = args[1] if len(args) > 1 else None
        return (sub and _read(folder, sub)) or _read(folder, "help")
    if not any(a in flags for a in args):
        return None
    if positional and (text := _read(folder, positional[0])) is not None:
        return text
    return _read(folder, "help")  # like most tools, --help anywhere prints the main page


def _before_wrapped_command(args: list[str], value_options: set[str]) -> list[str]:
    """Arguments up to the wrapped command, skipping the values of options that take one."""
    i = 0
    while i < len(args):
        arg = args[i]
        if not arg.startswith("-"):
            break
        i += 2 if arg in value_options else 1
    return args[: min(i, len(args))]


def manual(command: str) -> str | None:
    folder = _index().get(command)
    if folder is None:
        return None
    return _read(folder, "man") or _read(folder, "help")
