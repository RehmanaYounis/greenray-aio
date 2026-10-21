"""Text editors: `nano`, `vi`, `vim`.

On a real terminal the learner's own editor opens on a temporary copy of the
simulated file, and the result is written back into the simulated node. Without
a terminal (piped input, tests) or without any editor installed, an inline
editor reads the new content line by line.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

from ..core.errors import CommandError
from . import command

INLINE_HELP = (
    "[ inline editor: type the complete new content; end with a line containing only '.' to save,\n"
    "  ':wq' to save, or ':q!' to cancel. A real terminal opens your normal editor instead. ]"
)


def _external_editor(requested: str) -> list[str] | None:
    """The editor to launch for ``requested`` (nano/vi/vim), or None if none is usable."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    choices = [requested, os.environ.get("VISUAL", ""), os.environ.get("EDITOR", ""), "nano", "vim", "vi"]
    if os.name == "nt":
        choices.append("notepad")
    for choice in filter(None, choices):
        exe = shutil.which(choice.split()[0])
        if exe:
            return [exe, *choice.split()[1:]]
    return None


def run_external(editor: list[str], content: str) -> str | None:
    """Open ``content`` in ``editor``; return the saved text, or None if the editor failed."""
    fd, path = tempfile.mkstemp(prefix="greenray-", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        if subprocess.call([*editor, path]) != 0:
            return None
        with open(path, encoding="utf-8") as f:
            return f.read()
    finally:
        os.unlink(path)


def run_inline(ctx, name: str, path: str, content: str) -> str | None:
    """Line-based fallback editor using the shell's reader. Returns new text, or None if cancelled."""
    reader = ctx.reader
    if reader is None:
        raise CommandError(f"{name}: no terminal available to edit {path}. Use: cat > {path} <<EOF ... EOF")
    shown = content if content else "(new file)\n"
    lines: list[str] = []
    prompt = f"  GNU {name} (simulated) -- {path}\n{shown.rstrip()}\n{INLINE_HELP}\n> "
    while True:
        line = reader(prompt)
        prompt = "> "
        if line is None or line.strip() == ":q!":
            return None
        if line.strip() in (".", ":wq", ":x"):
            return "\n".join(lines) + ("\n" if lines else "")
        lines.append(line)


def edit(ctx, args, name: str = "nano"):
    files = [a for a in args if not a.startswith(("-", "+"))]
    if not files:
        raise CommandError(f"{name}: a file name is required in this simulator (e.g. {name} /etc/slurm/gres.conf)")
    path = ctx.abspath(files[-1])
    if path.endswith("/"):
        raise CommandError(f'{name}: "{path}" is a directory')
    old = ctx.node.files.get(path, "")
    editor = _external_editor(name)
    new = run_external(editor, old) if editor else run_inline(ctx, name, path, old)
    if new is None:
        return f"[ Cancelled: {path} not changed ]\n"
    ctx.node.files[path] = new
    if new == old:
        return ""
    return f"[ Wrote {len(new.splitlines())} lines to {path} ]\n"


def _register(name: str) -> None:
    @command(name, help=f"{name} FILE: edit a file (opens your real editor when run in a terminal)")
    def _editor(ctx, args):
        return edit(ctx, args, name)


for _name in ("nano", "vi", "vim"):
    _register(_name)
