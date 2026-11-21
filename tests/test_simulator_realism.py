"""The simulated tools refuse what the real ones refuse: unknown options, unknown subcommands, failed remotes."""

from __future__ import annotations

import pytest

from conftest import concrete, requires_solutions
from greenray.content import helppages
from greenray.content import scenarios as sc
from greenray.core.shell import Shell
from greenray.core.state import make_cluster


def run(line: str, shell: Shell | None = None) -> tuple[str, int]:
    shell = shell or Shell(make_cluster())
    out = shell.run(line)
    return out, shell.ctx.last_code


# ---- ssh <host> <command> ends with the status of the remote command


@pytest.mark.parametrize(
    ("line", "code"),
    [
        ("ssh dgx-01 true", 0),
        ("ssh dgx-01 hostname", 0),
        ("ssh dgx-01 false", 1),
        ("ssh dgx-01 'ls /nosuch'", 2),
        ("ssh dgx-01 nvidia-smi -i 99", 6),
        ("ssh dgx-01 nosuchcommand", 127),
        ("ssh nosuchhost true", 255),
    ],
)
def test_ssh_returns_the_remote_exit_status(line, code):
    assert run(line)[1] == code


def test_ssh_status_drives_shell_logic_and_keeps_the_message():
    assert run("ssh dgx-01 true && echo reached")[0].strip() == "reached"
    assert "reached" not in run("ssh dgx-01 false && echo reached")[0]
    assert run("ssh dgx-01 false || echo fallback")[0].strip() == "fallback"
    assert "No devices were found matching the id 99" in run("ssh dgx-01 nvidia-smi -i 99")[0]


def test_a_failed_remote_command_is_not_recorded_as_completed_anywhere():
    shell = Shell(make_cluster())
    shell.run("ssh dgx-01 nvidia-smi -i 99")
    assert [completed for _pos, _host, _argv, _text, completed in shell.ctx.results] == [False, False]


# ---- unknown long options are refused in the tool's own words

UNKNOWN = [
    ("squeue --bogus", "squeue: unrecognized option '--bogus'\nTry \"squeue --help\" for more information"),
    ("sinfo --bogus", "sinfo: unrecognized option '--bogus'"),
    ("sacct --bogus", "sacct: unrecognized option '--bogus'"),
    ("sshare --bogus", "sshare: unrecognized option '--bogus'"),
    ("kubectl get pods --bogus", "error: unknown flag: --bogus"),
    ("helm list --bogus", "Error: unknown flag: --bogus"),
    ("nvidia-smi --bogus", "Invalid combination of input arguments"),
    ("dcgmi health --bogus", "dcgmi: unrecognized option '--bogus'"),
    ("df --bogus", "df: unrecognized option '--bogus'\nTry 'df --help' for more information."),
    ("lscpu --bogus", "lscpu: unrecognized option '--bogus'"),
    ("lsmod --bogus", "lsmod: unrecognized option '--bogus'"),
    ("dmesg --bogus", "dmesg: unrecognized option '--bogus'"),
    ("findmnt --bogus", "findmnt: unrecognized option '--bogus'"),
    ("ibstat --bogus", "ibstat: unrecognized option '--bogus'"),
    ("showmount --bogus", "showmount: unrecognized option '--bogus'"),
    ("ls --bogus", "ls: unrecognized option '--bogus'"),
]


@pytest.mark.parametrize(("line", "message"), UNKNOWN, ids=[u[0] for u in UNKNOWN])
def test_an_unknown_long_option_is_refused(line, message):
    out, code = run(line)
    assert message in out and code == 1
    assert len(out.strip().splitlines()) <= 2  # the refusal only, none of the normal output


@pytest.mark.parametrize(
    "line",
    [
        "squeue --me",
        "squeue --states=PENDING",
        "sinfo --Node --long",
        "sacct --format=JobID,State",
        "kubectl get pods --all-namespaces",
        "kubectl get nodes --show-labels",
        "helm search repo nvidia/network-operator --versions",
        "helm list --all-namespaces",
        "nvidia-smi --query-gpu=index,name --format=csv",
        "ssh dgx-01 nvidia-smi --query-gpu=index --format=csv",
        "df --human-readable",
        "grep -- --bogus /etc/hosts",
        "echo --bogus",
        "ssh dgx-01 echo --bogus",
        "mpirun --allow-run-as-root -np 2 -H dgx-01:1,dgx-02:1 hostname",
    ],
)
def test_supported_options_and_other_programs_arguments_are_left_alone(line):
    out, _code = run(line)
    assert "unrecognized option" not in out and "unknown flag" not in out and "Invalid combination" not in out


def test_only_long_options_are_judged():
    for argv in (["ls", "-la"], ["head", "-5", "/etc/hosts"], ["squeue", "-h"], ["echo", "--bogus"]):
        assert helppages.unknown_long_option(argv, "greenray.commands.coreutils") is None


@requires_solutions
def test_no_reference_solution_uses_an_option_the_gate_refuses():
    """A lab may show a tool refusing an option (srun before Pyxis is installed); the generic gate must not add any."""
    from greenray import commands

    for raw in sc.load_all():
        for s in raw.all_variants():
            shell = Shell(s.build_state())
            sc.replay_solution(s, shell)
            for _host, argv in shell.ctx.executed:
                fn = commands.REGISTRY.get(argv[0])
                assert fn is None or helppages.unknown_long_option(argv, fn.__module__) is None, (s.id, argv)


# ---- unknown subcommands and names


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("ipmitool bogus", "Invalid command: bogus"),
        ("ipmitool sel bogus", "Invalid SEL command: bogus"),
        ("ipmitool power bogus", "Invalid chassis power command: bogus"),
        ("systemctl bogus", "Unknown command verb bogus."),
        ("systemctl bogus docker", "Unknown command verb bogus."),
        ('cmsh -c "device; status missing"', "Error: device missing not found"),
    ],
)
def test_unknown_subcommands_and_names_are_refused(line, message):
    out, code = run(line)
    assert message in out and code == 1


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("ipmitool sel list", "SEL has no entries"),
        ("ipmitool sel elist", "SEL has no entries"),
        ("ipmitool sdr", "Inlet Temp"),
        ("ipmitool power status", "Chassis Power is on"),
        ("ipmitool chassis status", "System Power"),
        ("ipmitool -I lanplus -H 10.141.0.1 -U admin -P secret sel list", "SEL has no entries"),
        ("systemctl status slurmctld", "slurmctld"),
        ("systemctl is-active slurmctld", "active"),
        ("systemctl list-units", ".service"),
        ("systemctl daemon-reload", ""),
        ('cmsh -c "device; status"', "dgx-04"),
        ('cmsh -c "device; status dgx-02"', "dgx-02"),
        ('cmsh -c "device; status -n dgx-02"', "dgx-02"),
    ],
)
def test_valid_subcommands_still_work(line, expected):
    out, code = run(line)
    assert expected in out and code == 0


def test_sshare_shows_only_the_accounts_asked_for():
    shell = Shell(make_cluster())
    assert len(shell.run("sshare -A nosuchaccount").strip().splitlines()) == 2  # the header and its rule
    assert "root" in shell.run("sshare -A root")
    assert "root" in shell.run("sshare")


# ---- nvsm names the OS the node runs


def test_nvsm_versions_names_the_installed_os():
    s = concrete("ins-container-toolkit-docker")
    shell = Shell(s.build_state())
    stock = shell.run("ssh dgx-03 nvsm show versions")
    assert "DGX H100 (Ubuntu 22.04.4 LTS)" in stock and "DGX OS" not in stock
    assert "DGX H100 (DGX OS 6.2)" in shell.run("ssh dgx-01 nvsm show versions")
