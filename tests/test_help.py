"""Realistic --help / help / man pages and the text editors."""

import re

import pytest

from greenray import commands
from greenray.commands import editor
from greenray.content import helppages
from greenray.content import scenarios as sc
from greenray.core.shell import Shell
from greenray.core.state import make_cluster

ON_HEAD = {"cmsh", "cmha", "cm-chroot-sw-img", "chroot"}


def shell_for(name):
    sh = Shell(make_cluster(), "bcm-head" if name in ON_HEAD else "dgx-01")
    return sh


def test_every_help_folder_is_a_registered_command():
    registered = set(commands.names())
    assert set(helppages.commands_with_help()) <= registered


@pytest.mark.parametrize("name", [n for n in commands.names() if "/" not in n])
def test_help_flag_never_errors(name):
    out = shell_for(name).run(f"{name} --help", heredoc_reader=lambda p: None)
    for bad in ("internal simulator error", "unrecognized option '--help'", "invalid option -- '-'"):
        assert bad not in out, out[:200]


@pytest.mark.parametrize("name", helppages.commands_with_help())
def test_help_pages_are_generic(name):
    folder = helppages._index()[name]
    ids = {s.id for s in sc.load_all()}
    for path in folder.glob("*.txt"):
        text = path.read_text(encoding="utf-8")
        assert text.strip(), path
        assert not re.search(r"\bdgx-0\d\b|bcm-head|dpu-0\d", text), path
        assert not any(i in text for i in ids), path


@pytest.mark.parametrize(
    "line,must_not_contain",
    [
        ("sinfo -h", "PARTITION"),  # -h = no header in Slurm
        ("squeue -h", "JOBID"),
    ],
)
def test_dash_h_keeps_its_real_meaning(line, must_not_contain):
    assert must_not_contain not in shell_for("x").run(line)


def test_dash_h_human_readable_tools_still_work():
    sh = shell_for("x")
    assert "Filesystem" in sh.run("df -h")
    assert "Mem:" in sh.run("free -h")


def test_subcommand_and_help_word_forms():
    if not helppages.page("kubectl", "get"):
        pytest.skip("kubectl pages not written yet")
    sh = shell_for("kubectl")
    get_page = helppages.page("kubectl", "get")
    assert sh.run("kubectl get pods --help") == get_page.rstrip("\n") or sh.run("kubectl get pods --help") == get_page
    assert sh.run("kubectl help get").strip() == get_page.strip()
    assert sh.run("kubectl --help").strip() == helppages.page("kubectl").strip()


def test_man():
    sh = shell_for("x")
    assert "No manual entry for frobnicate" in sh.run("man frobnicate")
    assert "What manual page" in sh.run("man")
    with_help = helppages.commands_with_help()
    if with_help:
        assert sh.run(f"man {with_help[0]}").strip()


# ---------------------------------------------------------------- editors


def lines(*items):
    it = iter(items)
    return lambda prompt: next(it, None)


def test_inline_editor_saves_and_cancels():
    sh = shell_for("x")
    out = sh.run("nano /root/notes.txt", heredoc_reader=lines("first", "second", "."))
    assert "[ Wrote 2 lines to /root/notes.txt ]" in out
    assert sh.run("cat /root/notes.txt") == "first\nsecond"
    out = sh.run("vi /root/notes.txt", heredoc_reader=lines("replaced", ":q!"))
    assert "Cancelled" in out and sh.run("cat /root/notes.txt") == "first\nsecond"


def test_external_editor_round_trip(monkeypatch):
    seen = {}

    def fake_external(cmd, content):
        seen["cmd"], seen["content"] = cmd, content
        return content + "added by editor\n"

    monkeypatch.setattr(editor, "_external_editor", lambda name: ["/usr/bin/fake-" + name])
    monkeypatch.setattr(editor, "run_external", fake_external)
    sh = shell_for("x")
    sh.run("echo original > /root/f.txt")
    assert "Wrote 2 lines" in sh.run("vim /root/f.txt")
    assert seen["cmd"] == ["/usr/bin/fake-vim"] and seen["content"] == "original\n"
    assert sh.run("cat /root/f.txt") == "original\nadded by editor"


def test_run_external_uses_a_real_process(tmp_path):
    script = tmp_path / "ed.sh"
    script.write_text('#!/bin/sh\necho "line from editor" >> "$1"\n')
    script.chmod(0o755)
    if editor.os.name == "nt":
        pytest.skip("POSIX shell script")
    assert editor.run_external([str(script)], "x\n") == "x\nline from editor\n"


def test_editor_requires_a_file():
    assert "a file name is required" in shell_for("x").run("nano")


def test_wrapped_commands_keep_their_own_flags():
    sh = shell_for("ssh")
    assert "Filesystem" in sh.run("ssh dgx-01 df -h")  # -h belongs to df, not ssh
    assert helppages.page("ssh").strip() in sh.run("ssh --help")
    assert "usage" in sh.run("ssh -h").lower()
    head = shell_for("chroot")
    assert helppages.page("chroot").strip() == head.run("chroot --help").strip()
    if helppages.page("kubectl"):
        assert helppages.page("kubectl").strip() != sh.run("kubectl exec my-pod -- ls --help").strip()


def test_wrapper_option_values_are_skipped():
    sh = shell_for("srun")
    srun_help = helppages.page("srun").strip()
    assert sh.run("srun -p defq --help").strip() == srun_help
    assert sh.run("mpiexec -np 1 --help").strip() == helppages.page("mpirun").strip()
    assert sh.run("srun -N1 nvidia-smi --help").strip() != srun_help  # help of the wrapped tool, not srun
