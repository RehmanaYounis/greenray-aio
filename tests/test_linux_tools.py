"""The everyday Linux tools exist on every node and behave like the real ones in the cases an admin uses."""

import pytest

from greenray.commands import names
from greenray.core.shell import Shell
from greenray.core.state import make_cluster

EXPECTED = [
    "less",
    "more",
    "cut",
    "uniq",
    "tr",
    "printf",
    "mv",
    "ln",
    "rmdir",
    "find",
    "which",
    "file",
    "stat",
    "du",
    "basename",
    "dirname",
    "realpath",
    "date",
    "sleep",
    "true",
    "false",
    "seq",
    "xargs",
    "test",
    "expr",
    "diff",
    "cmp",
    "history",
    "bash",
    "sh",
    "sudo",
    "tar",
    "gzip",
    "gunzip",
    "jq",
    "nproc",
    "htop",
    "pgrep",
    "killall",
    "w",
    "who",
    "lsof",
    "fuser",
    "vmstat",
    "mpstat",
    "hostnamectl",
    "timedatectl",
    "modinfo",
    "ss",
    "netstat",
    "ifconfig",
    "ethtool",
    "nslookup",
    "dig",
    "traceroute",
    "curl",
    "wget",
    "shutdown",
    "poweroff",
    "top",
]


def shell(host=None):
    s = Shell(make_cluster())
    if host:
        s.run(f"ssh {host}")
    return s


def test_every_everyday_tool_is_registered():
    assert sorted(set(EXPECTED) - set(names())) == []


def test_text_tools_compose_through_pipes():
    s = shell()
    s.run("echo -e 'b\\na\\na' > /root/x.txt")
    assert s.run("sort /root/x.txt | uniq -c").split() == ["2", "a", "1", "b"]
    assert s.run("cut -d= -f2 /etc/slurm/slurm.conf | head -2").splitlines()[1] == "superpod"
    assert s.run("echo hello | tr a-z A-Z").strip() == "HELLO"
    assert s.run("printf '%s=%d\\n' a 1 b 2") == "a=1\nb=2"
    assert s.run("echo a b | xargs -n 1 echo x") == "x a\nx b"
    assert s.run('echo \'{"a":{"b":2}}\' | jq .a.b').strip() == "2"


def test_file_tools_follow_the_simulated_tree():
    s = shell()
    assert "/etc/slurm/gres.conf" in s.run("find /etc -name '*.conf'")
    s.run("echo x > /root/a.txt && mv /root/a.txt /root/b.txt")
    assert "b.txt" in s.run("ls /root") and "a.txt" not in s.run("ls /root")
    assert s.run("file /root/b.txt").strip() == "/root/b.txt: ASCII text"
    assert s.run("basename /etc/slurm/slurm.conf .conf").strip() == "slurm"
    assert "regular file" in s.run("stat /root/b.txt")
    assert s.run("test -f /root/b.txt && echo yes").strip() == "yes"
    assert "No such file" in s.run("mv /root/nosuch /tmp")
    s.run("tar -czf /root/s.tgz /etc/slurm")
    assert "etc/slurm/slurm.conf" in s.run("tar -tf /root/s.tgz")
    s.run("cp /root/b.txt /root/c.txt && echo y >> /root/c.txt")
    assert "> y" in s.run("diff /root/b.txt /root/c.txt")


def test_scripts_run_on_the_node_they_are_started_on():
    s = shell("dgx-02")
    s.run("echo -e '#!/bin/bash\\nhostname\\nuname -r' > /root/s.sh")
    assert s.run("bash /root/s.sh").splitlines() == ["dgx-02", "5.15.0-1042-nvidia"]
    assert s.run("sudo hostname").strip() == "dgx-02"


def test_diagnostic_tools_read_the_state():
    s = shell()
    assert "slurmctld" in s.run("pgrep -l slurm") and "0.0.0.0:6817" in s.run("ss -tlnp")
    assert s.run("curl -sk https://10.141.255.254:6443/healthz") == "ok"
    assert "Connection refused" in s.run("curl http://dgx-01:9999/")
    s.run("systemctl stop slurmctld")
    assert "6817" not in s.run("ss -tln")
    assert "Address: 10.141.0.2" in s.run("nslookup dgx-02")
    s.run("ssh dgx-01")
    assert "Link detected: yes" in s.run("ethtool ibp24s0")
    assert "Static hostname: dgx-01" in s.run("hostnamectl")
    assert "nvidia_peermem.ko" in s.run("modinfo nvidia_peermem")
    assert s.run("htop").startswith("top - ")


def test_shutdown_powers_the_node_off_and_is_audited_with_jobs():
    s = shell()
    s.run("echo -e '#!/bin/bash\\nsleep 600' > /root/j.sh && sbatch -w dgx-01 /root/j.sh")
    s.run("ssh dgx-01")
    assert "closed by remote host" in s.run("shutdown -h now")
    assert s.ctx.host == "bcm-head"
    assert "Unreachable" in s.run("ping dgx-01")
    assert any(e["kind"] == "power_off_with_jobs" for e in s.ctx.state.audit)


@pytest.mark.parametrize("line", ["cut /etc/slurm/slurm.conf", "seq", "expr 1 /", "tar -xf /root/nosuch.tgz"])
def test_wrong_usage_is_rejected_like_the_real_tool(line):
    out = shell().run(line)
    assert out and (
        "usage" in out.lower()
        or "missing" in out.lower()
        or "error" in out.lower()
        or "cannot" in out.lower()
        or "must" in out.lower()
    )
