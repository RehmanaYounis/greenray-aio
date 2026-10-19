from aiolab.core.shell import Shell
from aiolab.core.state import make_cluster


def sh():
    return Shell(make_cluster())


def test_pipes_and_grep():
    s = sh()
    s.run("ssh dgx-01")
    out = s.run("nvidia-smi -L | grep -c GPU")
    assert out.strip() == "8"


def test_redirect_and_sed():
    s = sh()
    s.run("echo Name=gpu File=/dev/nvidia[0-3] > /tmp/g.conf")
    s.run("sed -i 's/0-3/0-7/' /tmp/g.conf")
    assert s.run("cat /tmp/g.conf") == "Name=gpu File=/dev/nvidia[0-7]"


def test_heredoc():
    s = sh()
    lines = iter(["line one", "line two", "EOF"])
    s.run("cat > /tmp/x <<EOF", heredoc_reader=lambda _p: next(lines))
    assert s.ctx.node.files["/tmp/x"] == "line one\nline two\n"


def test_and_chain_stops_on_failure():
    s = sh()
    out = s.run("cat /nope && echo should-not-print")
    assert "should-not-print" not in out


def test_ssh_and_exit():
    s = sh()
    s.run("ssh dgx-03")
    assert s.run("hostname") == "dgx-03"
    s.run("exit")
    assert s.run("hostname") == "bcm-head"


def test_unknown_command():
    assert "command not found" in sh().run("frobnicate")


def test_mig_select_single_gpu():
    s = sh()
    s.run("ssh dgx-01")
    s.run("nvidia-smi -i 0 -mig 1")
    gpus = s.ctx.state.nodes["dgx-01"].gpus
    assert gpus[0].mig_mode and not gpus[1].mig_mode


def test_reboot_clears_xid():
    s = sh()
    g = s.ctx.state.nodes["dgx-01"].gpus[2]
    g.fallen_off_bus = True
    g.xid = 79
    s.run("ssh dgx-01")
    assert "Unable to determine the device handle" in s.run("nvidia-smi")
    s.run("reboot")
    assert not g.fallen_off_bus and g.xid is None
