"""Command registry.

Each module in this package registers commands with ``@command("name")``.
A command is ``fn(ctx, args: list[str]) -> str`` where ``args`` excludes the
command name. Raise ``greenray.shell.CommandError`` for failures.
"""

from __future__ import annotations

import importlib
import pkgutil

REGISTRY: dict = {}
HELP: dict = {}
# fn(ctx, service_name, action) called after systemctl start/stop/restart
HOOKS_SERVICE: list = []
# fn(ctx, node) called when a node reboots
HOOKS_REBOOT: list = []


def command(*names: str, help: str = "", override: bool = False):
    """Register ``fn(ctx, args) -> str`` under ``names``.

    Registering a name twice is an error unless ``override=True``, which a module
    uses when it deliberately wraps another module's command (and imports it first).
    """

    def deco(fn):
        for n in names:
            if n in REGISTRY and not override:
                raise RuntimeError(f"command {n!r} already registered by {REGISTRY[n].__module__}")
            REGISTRY[n] = fn
            if help:
                HELP[n] = help
        return fn

    return deco


_loaded = False


def _load_all():
    global _loaded
    if _loaded:
        return
    _loaded = True
    for mod in pkgutil.iter_modules(__path__):
        importlib.import_module(f"{__name__}.{mod.name}")


def dispatch(ctx, argv: list[str]) -> str:
    from ..core.errors import CommandError

    _load_all()
    name = argv[0]
    fn = REGISTRY.get(name)
    if fn is None:
        raise CommandError(f"bash: {name}: command not found", 127)
    if not ctx.node.up and name not in ("exit", "logout"):
        raise CommandError(f"ssh: connect to host {ctx.host}: No route to host", 255)
    from ..content import helppages

    page = helppages.help_request(argv)
    if page is not None:
        return page
    return fn(ctx, argv[1:])


def names() -> list[str]:
    _load_all()
    return sorted(REGISTRY)
