"""Terminal realism: working directory, relative paths, prompt, login banner, real-tool colours."""

from conftest import ScriptedUI, concrete
from greenray.app import lab
from greenray.core.shell import Shell
from greenray.core.state import make_cluster

ESC = "\033["


def sh(color=False):
    return Shell(make_cluster(), color=color)


def test_cd_pwd_and_relative_paths():
    s = sh()
    s.run("ssh dgx-01")
    assert s.run("cd /etc/slurm") == ""
    assert s.prompt == "root@dgx-01:/etc/slurm# "
    assert "Name=gpu" in s.run("cat gres.conf")
    s.run("cd ..")
    assert s.run("pwd") == "/etc"
    assert "No such file or directory" in s.run("cd nope")
    assert "Not a directory" in s.run("cd /etc/hostname")
    s.run("cd")
    assert s.run("pwd") == "/root"


def test_new_files_land_in_the_working_directory():
    s = sh()
    s.run("cd /tmp")
    s.run("echo hello > note.txt")
    s.run("touch empty.txt")
    assert s.ctx.node.files["/tmp/note.txt"] == "hello\n"
    assert "/tmp/empty.txt" in s.ctx.node.files
    assert s.run("cat ~/../tmp/note.txt") == "hello"


def test_ssh_starts_at_home_and_exit_restores_directory():
    s = sh()
    s.run("cd /etc")
    banner = s.run("ssh dgx-02")
    assert banner.startswith("Welcome to") and "GPUs:" in banner and "Last login:" in banner
    assert s.prompt == "root@dgx-02:~# "
    s.run("exit")
    assert s.prompt == "root@bcm-head:/etc# "


def test_colours_only_on_a_terminal_never_in_pipes():
    s = sh(color=True)
    s.run("ssh dgx-01")
    assert ESC in s.run("systemctl status nvidia-dcgm")  # straight to the terminal: coloured
    assert ESC not in s.run("systemctl status nvidia-dcgm | head -2")  # piped: plain, like real tools
    assert ESC not in s.run("dmesg > /tmp/k.log; cat /tmp/k.log")  # redirected: plain
    assert ESC in s.run("dmesg | grep -i nvrm")  # grep --color=auto highlights the match
    assert ESC not in sh(color=False).run("ssh dgx-01 systemctl status nvidia-dcgm")


def test_styled_prompt_uses_ubuntu_colours():
    s = sh(color=True)
    assert s.styled_prompt() == "\033[01;32mroot@bcm-head\033[00m:\033[01;34m~\033[00m# "


def test_plain_ui_has_no_escape_codes():
    ui = ScriptedUI(["quit"])
    lab.run_lab(ui, concrete("tsh-power-cap-persistence"), incident=True)
    assert ESC not in ui.text and all(ESC not in p for p in ui.prompts)
    assert "Incident report INC" in ui.text
