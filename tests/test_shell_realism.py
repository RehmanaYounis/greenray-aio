"""The shell behaves like bash where a candidate notices: variables, exit codes, globbing, redirections."""

from greenray.core.shell import Shell
from greenray.core.state import make_cluster


def sh():
    return Shell(make_cluster())


def test_exit_status_follows_the_last_pipeline():
    s = sh()
    assert s.run("false; echo $?") == "1"
    assert s.run("true; echo $?") == "0"
    assert s.run("cat /nosuch; echo $?").splitlines()[-1] == "1"
    assert s.run("grep -c xyz /etc/hostname; echo $?").splitlines()[-1] == "1"


def test_assignments_and_expansions():
    s = sh()
    assert s.run("X=dgx-02; echo $X") == "dgx-02"
    assert s.run("echo ${Y:-fallback}") == "fallback"
    assert s.run("echo ${Y:=set}; echo $Y") == "set\nset"
    assert s.run("echo $((2 + 3 * 4))") == "14"
    assert s.run("N=3; echo $((N * 2))") == "6"
    assert s.run("echo $HOME $USER $HOSTNAME") == "/root root bcm-head"
    assert s.run("cd /etc; echo $PWD $OLDPWD") == "/etc /root"
    assert s.run("env FOO=bar printenv FOO") == "bar"
    assert s.run("printenv NOPE; echo $?") == "1"


def test_cd_dash_returns_to_the_previous_directory():
    s = sh()
    s.run("cd /etc/slurm")
    s.run("cd /tmp")
    assert s.run("cd -") == "/etc/slurm"
    assert s.ctx.cwd == "/etc/slurm"


def test_bare_words_are_not_turned_into_paths():
    s = sh()
    s.run("mkdir /root/data")
    assert s.run("echo data") == "data"
    assert s.run("echo '{\"a\":1}' | jq .a") == "1"
    s.run("cd /etc")
    assert s.run("echo hosts") == "hosts"  # a word that happens to name a file in the cwd stays a word
    assert "NotFound" in s.run("kubectl get ns slurm")  # not rewritten to /etc/slurm


def test_globbing():
    s = sh()
    assert s.run("ls /etc/slurm/gres*") == "/etc/slurm/gres.conf"
    s.run("cd /etc/slurm")
    assert s.run("echo *.conf") == "cgroup.conf gres.conf slurm.conf"
    assert s.run("ls *.conf").split() == ["cgroup.conf", "gres.conf", "slurm.conf"]
    assert s.run("echo *.nomatch") == "*.nomatch"  # bash leaves an unmatched pattern as typed
    assert s.run("echo '*.conf'") == "*.conf"
    s.run("ssh dgx-01")
    assert s.run("ls /dev/nvidia? | wc -l") == "8"
    assert "/dev/nvidiactl" in s.run("ls /dev/nvidia*")


def test_stderr_stays_out_of_pipes_and_files():
    s = sh()
    assert s.run("cat /nosuch | wc -l").splitlines() == ["cat: /nosuch: No such file or directory", "0"]
    assert s.run("cat /nosuch 2>&1 | wc -l") == "1"
    assert s.run("cat /nosuch 2>/dev/null; echo $?") == "1"
    out = s.run("cat /nosuch > /root/out.txt")
    assert "No such file" in out
    assert s.ctx.node.files["/root/out.txt"] == ""
    s.run("cat /nosuch 2> /root/err.txt")
    assert "No such file" in s.ctx.node.files["/root/err.txt"]
    s.run("echo ok &> /root/both.txt; cat /nosuch &>> /root/both.txt")
    assert s.ctx.node.files["/root/both.txt"].splitlines()[0] == "ok"
    assert "No such file" in s.ctx.node.files["/root/both.txt"]


def test_stdin_redirect_and_quoted_heredoc_with_redirect():
    s = sh()
    s.run("echo -e 'b\\na' > /root/in.txt")
    assert s.run("sort < /root/in.txt") == "a\nb"
    lines = iter(["echo $HOME untouched", "EOF"])
    s.run("cat <<'EOF' > /root/script.sh", heredoc_reader=lambda _p: next(lines))
    assert s.ctx.node.files["/root/script.sh"] == "echo $HOME untouched\n"
    assert s.run("bash /root/script.sh") == "/root untouched"


def test_sudo_runs_as_another_user():
    s = sh()
    s.run("sacctmgr -i add user alice account=root")
    assert s.run("sudo -u alice whoami") == "alice"
    assert s.run("sudo whoami") == "root"
    assert "Usage" in s.run("sudo --help") or "usage" in s.run("sudo --help").lower()


def test_ssh_runs_a_quoted_command_line_remotely():
    s = sh()
    s.run("ssh dgx-01 'hostname > /root/where.txt'")
    assert s.ctx.state.nodes["dgx-01"].files["/root/where.txt"] == "dgx-01\n"
    assert s.ctx.host == "bcm-head"


def test_node_files_follow_the_driver():
    s = sh()
    assert "bcm-head" in s.run("cat /etc/hosts") and "dgx-04" in s.run("cat /etc/hosts")
    assert 'VERSION_ID="22.04"' in s.run("cat /etc/os-release")
    assert float(s.run("cat /proc/uptime").split()[0]) > 0
    s.run("ssh dgx-01")
    assert "550.90.07" in s.run("cat /proc/driver/nvidia/version")
    assert s.run("ls -l /dev/nvidia0").startswith("crw-rw-rw-")
    s.ctx.node.driver_loaded = False
    assert "No such file" in s.run("cat /proc/driver/nvidia/version")
    assert "/dev/nvidia0" not in s.run("ls /dev/nvidia*")  # no device nodes without the driver
