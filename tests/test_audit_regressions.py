"""Regressions from the 1.5.1 audit: every finding that was reproduced is pinned here.

Grouped as the audit was: inputs that crashed the simulator, tools whose output misled, and
cmsh behaviour. Each case states the real tool's behaviour in its expected text.
"""

from __future__ import annotations

import re

import pytest

from conftest import concrete
from greenray.core.shell import Shell
from greenray.core.state import make_cluster


def sh(host: str | None = None) -> Shell:
    s = Shell(make_cluster())
    if host:
        s.run(f"ssh {host}")
    return s


# ---------------------------------------------------------------- A: the tool's error, never a traceback

CRASHES = [
    ("sbatch --time=abc --wrap=hostname", "Invalid --time specification"),
    ("sbatch -N abc --wrap=hostname", 'Invalid numeric value "abc" for -N'),
    ("sbatch -n x --wrap=hostname", 'Invalid numeric value "x" for -n'),
    ("sbatch -c two --wrap=hostname", 'Invalid numeric value "two" for -c'),
    ("sbatch -N 0 --wrap=hostname", 'Invalid numeric value "0" for -N'),
    ("srun -N x hostname", 'Invalid numeric value "x" for -N'),
    ("srun --nice=abc hostname", "Invalid --nice value"),
    ("salloc -t abc", "Invalid --time specification"),
    ("scontrol update NodeName=dgx-01 Weight=x", "Invalid argument"),
    ("scontrol update PartitionName=defq PriorityTier=x", "Invalid argument"),
    ("sacctmgr -i add qos q1 Priority=abc", "Bad priority value"),
    ("sacctmgr -i modify qos normal set MaxWall=abc", "Bad MaxWall time format"),
    ("sacctmgr -i modify qos normal set GrpTRES=gres/gpu=x", "Bad TRES value"),
    ('cmsh -c "device; use dgx-01; bmcsettings; set userid abc"', "userid must be a number"),
    ("kubectl create ns", "exactly one NAME is required"),
    ("kubectl get pods --field-selector=x", "must be a key=value pair"),
    ("kubectl get pods -o jsonpath='{.items[}'", "unclosed array"),
]


@pytest.mark.parametrize(("line", "wording"), CRASHES, ids=[c[0] for c in CRASHES])
def test_bad_input_gets_the_tools_error(line, wording):
    out = sh().run(line)
    assert "internal simulator error" not in out
    assert wording in out


def test_bad_values_are_refused_before_they_are_stored():
    s = sh()
    s.run("sacctmgr -i modify qos normal set MaxWall=abc")
    assert "internal simulator error" not in s.run("sacctmgr show qos")
    assert s.ctx.state.slurm.qos["normal"].get("max_wall") in (None, "")


def test_pexec_runs_on_every_node():
    out = sh().run('cmsh -c "device; pexec -n dgx-01,dgx-02 hostname"')
    assert "[dgx-01]\ndgx-01" in out and "[dgx-02]\ndgx-02" in out


def test_objects_in_use_cannot_be_removed():
    s = sh()
    assert "in use by dgx-01" in s.run('cmsh -c "category; remove dgx-h100; commit"')
    assert "dgx-h100" in s.ctx.state.bcm.categories
    assert "in use by" in s.run('cmsh -c "softwareimage; remove dgx-os-6.2-h100-image; commit"')
    assert "internal simulator error" not in s.run('cmsh -c "device; imageupdate -n dgx-01"')


def test_gpu_operator_daemonsets_dump_as_yaml():
    s = sh()
    s.run("helm repo add nvidia https://helm.ngc.nvidia.com/nvidia")
    s.run("helm install gpu-operator nvidia/gpu-operator -n gpu-operator --create-namespace --set driver.enabled=false")
    out = s.run("kubectl get ds nvidia-device-plugin-daemonset -n gpu-operator -o yaml")
    assert "tolerations:" in out and "internal simulator error" not in out
    assert "Name:" in s.run("kubectl describe ds nvidia-device-plugin-daemonset -n gpu-operator")


@pytest.mark.parametrize(
    ("line", "wording"),
    [
        ("head -n x /etc/hostname", "invalid number of lines"),
        ("tail -n x /etc/hostname", "invalid number of lines"),
        ("sed -e", "option requires an argument"),
        ("echo abc | sed 's/[/x/'", "unterminated character set"),
        ("dcgmi dmon -e 150 -c x", "Invalid value for -c/--count"),
        ("helm rollback nosuch x", "release: not found"),
    ],
)
def test_text_and_gpu_tools_reject_bad_arguments(line, wording):
    out = sh("dgx-01" if not line.startswith("helm") else None).run(line)
    assert "internal simulator error" not in out and wording in out


def test_sed_handles_posix_syntax():
    s = sh("dgx-01")
    assert s.run("echo 'a/b c' | sed 's/a\\/b/c/'") == "c c"
    assert s.run("echo 'MaxTime=INFINITE' | sed 's/^\\(MaxTime=\\)INFINITE/\\12-00:00:00/'") == "MaxTime=2-00:00:00"
    assert s.run("printf 'a\\nb\\nc\\n' | sed -n '2p'") == "b"
    assert s.run("printf 'a\\nb\\nc\\nd\\n' | sed -n '2,3p'") == "b\nc"
    assert s.run("printf 'a\\nb\\n' | sed '1d'") == "b"
    assert s.run("echo 'x(1)' | sed 's/(1)/[1]/'") == "x[1]"
    assert s.run("echo abc | sed -E 's/(b)/[\\1]/'") == "a[b]c"


# ---------------------------------------------------------------- D: output that used to mislead


def test_sacct_j_shows_other_users_jobs_and_root_sees_everyone():
    s = sh()
    s.run("sacctmgr -i add user alice account=root")
    s.run("su - alice -c 'sbatch --wrap=hostname'")
    assert "1001" in s.run("sacct -j 1001 -X")
    assert "alice" in s.run("sacct -X --format=JobID,User")


def test_scontrol_oneliner_and_options_anywhere():
    s = sh()
    s.run("sbatch --wrap='sleep 600'")
    assert "\n" not in s.run("scontrol -o show job 1001")
    assert "CPU_IDs" in s.run("scontrol show job 1001 -d")


def test_partition_oversubscribe_and_allowgroups_are_applied():
    s = sh()
    assert s.run("scontrol update PartitionName=defq OverSubscribe=YES AllowGroups=ml") == ""
    out = s.run("scontrol show partition defq")
    assert "OverSubscribe=YES" in out and "AllowGroups=ml" in out
    assert "Invalid partition parameter" in s.run("scontrol update PartitionName=defq Bogus=1")


def test_sacct_brief_long_fields_and_time_windows():
    s = sh()
    s.run("sbatch --wrap=hostname")
    assert s.run("sacct -b").splitlines()[0].split() == ["JobID", "State", "ExitCode"]
    assert "NCPUS" in s.run("sacct -X --format=JobID,NCPUS,ReqMem")
    assert "Invalid time specified" in s.run("sacct -S bogus")


def test_squeue_long_format_fields():
    s = sh()
    s.run("sbatch --wrap='sleep 600' --gres=gpu:2")
    out = s.run("squeue -O jobid:10,state,tres-per-node")
    assert "JOBID" in out and "gpu:2" in out
    assert "Invalid field specification" in s.run("squeue -O nosuch")


def test_job_steps_see_their_environment_and_cpu_limit():
    s = sh()
    assert s.run("srun -N1 -n1 -c 4 nproc").strip() == "4"
    assert s.run("srun -n2 bash -c 'echo rank $SLURM_PROCID'") == "rank 0\nrank 1"
    assert s.run("srun --gres=gpu:2 bash -c 'echo $CUDA_VISIBLE_DEVICES'").strip() == "0,1"


def test_output_pattern_node_is_the_node_the_job_ran_on():
    s = sh()
    s.run("sbatch -o /tmp/%u-%N-%j.out --wrap=hostname")
    assert "StdOut=/tmp/root-dgx-01-1001.out" in s.run("scontrol show job 1001")


def test_missing_command_in_a_batch_script_reads_like_bash():
    s = sh()
    s.run("sbatch --wrap=nosuchcmd")
    for _ in range(4):
        s.run("sinfo")
    assert "nosuchcmd: command not found" in s.run("cat /root/slurm-1001.out")
    assert "127:0" in s.run("sacct -j 1001 -X --format=JobID,ExitCode")


def test_scancel_signal_and_me():
    s = sh()
    s.run("sbatch --wrap='sleep 600'")
    assert "Invalid job signal" in s.run("scancel -s BOGUS 1001")
    s.run("scancel -s TERM 1001")
    job = s.ctx.state.slurm.jobs[0]
    assert job.state == "RUNNING" and job.get("signals") == ["TERM"]  # delivered, not killed
    s = sh()
    s.run("sbatch --wrap='sleep 600'")
    s.run("scancel --me")
    assert s.ctx.state.slurm.jobs[0].state == "CANCELLED"


def test_srun_mpi_list():
    assert "pmix" in sh().run("srun --mpi=list")


def test_sacctmgr_options_anywhere_and_parsable_variants():
    s = sh()
    out = s.run("sacctmgr add account research -i")
    assert "research" in out and "-i" not in s.ctx.state.slurm.accounts
    assert s.run("sacctmgr -P show qos").splitlines()[0].endswith("MaxSubmitPU")
    assert s.run("sacctmgr -p show qos").splitlines()[0].endswith("|")
    assert "gres" in s.run("sacctmgr show tres")


def test_sacctmgr_names_the_jobs_that_block_a_qos_delete():
    s = sh()
    s.run("sacctmgr -i add qos lowprio")
    s.run("sbatch --qos=lowprio --wrap='sleep 600'")
    assert "JobID = 1001" in s.run("sacctmgr -i delete qos lowprio")


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("printf '3\\n10\\n2\\n' | sort -n", "2\n3\n10"),
        ("printf 'b\\na\\nb\\n' | sort -u", "a\nb"),
        ("printf 'x 2\\ny 10\\n' | sort -k2 -n -r", "y 10\nx 2"),
        ("printf 'a b c\\n' | wc -w", "3"),
        ("printf 'ab\\n' | wc -c", "3"),
        ("printf '1 x\\n2 y\\n3 z\\n' | awk '$1>1{print $2}'", "y\nz"),
        ("printf '1 x\\n2 y\\n' | awk '{print $NF}'", "x\ny"),
        ("printf 'a:b\\n' | awk -F: '{print $2}'", "b"),
        ("printf '1\\n2\\n3\\n' | awk '{sum+=$1} END {print sum}'", "6"),
        ("echo hello | grep -o ell", "ell"),
        ("printf 'a\\nb\\n' | grep -n b", "2:b"),
        ("echo 'the cat' | grep -w ca", ""),
        ("printf 'x\\ny\\n' | xargs -n1 echo", "x\ny"),
        ("printf '1\\n2\\n' > /tmp/a; printf '1\\n3\\n' > /tmp/b; diff /tmp/a /tmp/b", "2c2\n< 2\n---\n> 3"),
        ("echo '{\"a\":[1,2]}' | jq '.a | length'", "2"),
        ("echo '{\"a\":[1,2]}' | jq -c '.a[]'", "1\n2"),
        ('echo \'{"b":1,"a":2}\' | jq -c keys', '["a", "b"]'),
        ('echo \'[{"n":"a"},{"n":"b"}]\' | jq -r \'.[] | select(.n == "b") | .n\'', "b"),
        ("printf 'a\\nb\\nc\\n' | tail -n +2", "b\nc"),
        ("printf 'one\\n' | cat -n", "     1\tone"),
    ],
)
def test_text_tools_honour_their_options(line, expected):
    assert sh("dgx-01").run(line) == expected


def test_grep_count_exits_one_without_a_match():
    s = sh("dgx-01")
    assert s.run("echo hi | grep -c nothere && echo matched") == "0"
    assert "matched" in s.run("echo hi | grep -c hi && echo matched")


def test_grep_e_patterns_make_positionals_files():
    s = sh()
    assert "NodeName" in s.run("grep -e NodeName -e nosuch /etc/slurm/slurm.conf")
    assert "slurm.conf:" in s.run("grep -H NodeName /etc/slurm/slurm.conf")


def test_journalctl_and_dmesg_filters():
    s = sh("dgx-01")
    assert "Started nvidia-fabricmanager.service" in s.run("journalctl -u nvidia-fabricmanager")
    assert s.run("journalctl -k").startswith("[")
    assert "-- No entries --" in s.run("journalctl -u slurmd -p err")
    assert "Linux version" not in s.run("dmesg -l err")
    assert "unrecognized option" in s.run("journalctl --bogus")


def test_modprobe_refuses_to_unload_a_module_in_use():
    s = sh("dgx-01")
    assert "is in use" in s.run("modprobe -r nvidia")
    assert "nvidia" in s.ctx.state.nodes["dgx-01"].modules


def test_ls_hides_dotfiles_and_lists_bare_names():
    s = sh()
    s.run("echo x > /root/.hidden; echo y > /root/shown.txt")
    assert ".hidden" not in s.run("ls /root") and "shown.txt" in s.run("ls /root")
    assert ".hidden" in s.run("ls -a /root")
    listing = s.run("cd /etc/slurm; ls")
    assert "gres.conf" in listing and "/" not in listing
    assert "-rw-r--r--" in s.run("ls -l /etc/slurm")


def test_nvidia_smi_clock_and_process_commands():
    s = sh("dgx-01")
    s.run("nvidia-smi -lgc 1000,1500 -i 0")
    s.run("nvidia-smi -rgc -i 0")
    assert s.ctx.state.nodes["dgx-01"].gpus[0].sm_clock_mhz == 345
    assert "# gpu" in s.run("nvidia-smi pmon -c 1")
    assert s.run("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv").startswith(
        "pid, process_name"
    )
    assert "Applications clocks set" in s.run("nvidia-smi -ac 2619,1980 -i 0")
    assert "Replay Errors" in s.run("nvidia-smi nvlink -e -i 0")
    assert "Invalid ECC config value" in s.run("nvidia-smi -e 5")


def test_mig_placements_match_an_h100():
    s = sh("dgx-01")
    s.run("nvidia-smi -i 0 -mig 1")
    out = s.run("nvidia-smi mig -i 0 -lgipp")
    assert "Profile ID  9 Placements: {0,4}:3" in out and "Profile ID 14 Placements: {0,2,4}:2" in out
    out = s.run("nvidia-smi mig -i 0 -cgi 9,9,9 -C")
    assert out.count("Successfully created GPU instance") == 2 and "Insufficient Resources" in out
    assert "4:3" in s.run("nvidia-smi mig -i 0 -lgi")


def test_dcgmi_stats_end_time_and_default_groups():
    s = sh("dgx-01")
    s.run("dcgmi stats -g 0 -e; dcgmi stats -s job1; dcgmi stats -x job1")
    assert "Still Running" not in s.run("dcgmi stats -j job1")
    assert "Cannot modify the default group" in s.run("dcgmi group -g 1 -a 2,3")
    assert "Instance Hierarchy" in s.run("dcgmi discovery -c")


def test_docker_ps_quiet_format_and_consistent_ids():
    s = sh("dgx-01")
    s.run("docker run -d --name web nvcr.io/nvidia/pytorch:24.05-py3 sleep 1000")
    cid = s.run("docker ps -q").strip()
    assert len(cid) == 12
    assert s.run("docker ps --format '{{.ID}} {{.Names}}'").strip() == f"{cid} web"
    assert s.run("docker inspect web --format '{{.Id}}'").startswith(cid)
    assert "seconds ago" in s.run("docker ps")


def test_ip_route_and_link_show():
    s = sh("dgx-01")
    assert s.run("ip route").startswith("default via 10.141.255.254")
    assert "ibp24s0" in s.run("ip link show ibp24s0")


def test_helm_json_listing_and_stable_deploy_time():
    s = sh()
    s.run("helm repo add nvidia https://helm.ngc.nvidia.com/nvidia")
    first = s.run("helm install gpu-operator nvidia/gpu-operator -n gpu-operator --create-namespace")
    stamp = next(line for line in first.splitlines() if line.startswith("LAST DEPLOYED"))
    s.run("sinfo")
    assert stamp in s.run("helm status gpu-operator -n gpu-operator")
    assert '"name": "gpu-operator"' in s.run("helm list -A -o json")
    assert "allowed formats are: custom-columns,json" in s.run("kubectl get nodes -o go-template={{.x}}")


def test_cmsh_validates_addresses_and_renames_hosts():
    s = sh()
    assert "not a valid IP address" in s.run('cmsh -c "device; use dgx-01; set ip notanip"')
    assert "not a valid MAC address" in s.run('cmsh -c "device; use dgx-01; set mac zz"')
    assert "already used by dgx-02" in s.run('cmsh -c "device; use dgx-01; set ip 10.141.0.2"')
    s.run('cmsh -c "device; use dgx-04; set hostname dgx-04b; commit"')
    assert "dgx-04b" in s.ctx.state.nodes and "dgx-04" not in s.ctx.state.nodes
    assert s.run("ssh dgx-04b hostname").strip() == "dgx-04b"


def test_cmsh_drain_reason_forms_and_timestamp():
    s = sh()
    for _ in range(3):
        s.run("sinfo")  # the Slurm clock moves on; the drain must be stamped with the time it happened
    s.run('cmsh -c "device; drain -n dgx-01 --reason=fw"')
    sn = s.ctx.state.slurm.nodes["dgx-01"]
    assert sn.reason == "fw" and sn.drain and sn.get("reason_time")
    assert "09:00:00" not in s.run("sinfo -R")


def test_cmsh_help_lists_every_mode_and_roles_cover_slurm():
    s = sh()
    assert "monitoring" in s.run("cmsh -h") and "wlm" in s.run("man cmsh")
    assert s.run('cmsh -c "device; use dgx-01; roles; assign slurmsubmit; commit"') == ""


def test_man_pages_for_slurm_configuration_exist():
    s = sh()
    assert "SCONTROL(1)" in s.run("man scontrol")
    assert "SLURM.CONF(5)" in s.run("man slurm.conf")
    assert "OUT_OF_MEMORY" in s.run("man sacct")
    assert "--pty is an srun option" in s.run("man salloc")


def test_dgx_package_list_includes_the_running_stack():
    out = sh("dgx-01").run("dpkg -l")
    assert "nvidia-fabricmanager-550" in out and "datacenter-gpu-manager" in out and "nvsm" in out


def test_kubectl_takes_the_namespace_before_the_verb():
    s = sh()
    s.run("kubectl create ns llm")
    assert "No resources found in llm namespace" in s.run("kubectl -n llm get pods")


def test_ngc_names_the_unsupported_registry_entity():
    assert "invalid choice: 'model'" in sh("dgx-01").run("ngc registry model list")


def test_date_and_journals_share_one_day():
    s = sh()
    assert "Oct 19" in s.run("date") and "2026" in s.run("date")
    assert "Oct 19" in s.run("ssh dgx-01")


# ---- tsh-first-look-at-logs: credit for the accounting lookup needs the query to return the incident job


def _tsh1(variant: int):
    s = concrete("tsh-first-look-at-logs", variant)
    shell = Shell(s.build_state())
    return s, shell


def _accounting_check(s, shell) -> bool:
    desc, ok, err = s.grade(shell)[0]
    assert "accounting" in desc and not err, (desc, err)
    return ok


@pytest.mark.parametrize("variant", [0, 1, 2])
def test_empty_or_wrong_accounting_queries_earn_no_credit(variant):
    s, shell = _tsh1(variant)
    assert not _accounting_check(s, shell)
    assert shell.run("sacct -j 999999").count("\n") == 1  # the header only
    shell.run("sacct -j 1")
    shell.run("scontrol show job 999999")
    assert "requires an argument" in shell.run("sacct -j")
    assert not _accounting_check(s, shell)


@pytest.mark.parametrize("variant", [0, 1, 2])
@pytest.mark.parametrize(
    "query",
    [
        "sacct -j {job}",
        "sacct -j {job} --format=JobID,State,ExitCode",
        "sacct -j {job} --format=State,ExitCode",
        "sacct -j {job} -p -n",
        "sacct -u {user}",
        "sacct",
        "sacct -a -X",
        "sacct -j {job}.batch",
        "scontrol show job {job}",
    ],
)
def test_valid_accounting_queries_earn_credit(variant, query):
    s, shell = _tsh1(variant)
    shell.run(query.format(**s.params))
    assert _accounting_check(s, shell)


def test_accounting_credit_follows_a_reset_and_a_crash():
    s, shell = _tsh1(2)
    shell.run("sacct -j 2954")
    assert _accounting_check(s, shell)
    shell = Shell(s.build_state())  # what `reset` does
    assert not _accounting_check(s, shell) and shell.ctx.outputs == []
    from greenray import commands

    @commands.command("crashsacct")
    def crashsacct(ctx, args):
        raise RuntimeError("boom")

    try:
        out = shell.run("crashsacct 2954")
    finally:
        commands.REGISTRY.pop("crashsacct", None)
    assert "internal simulator error" in out and shell.ctx.outputs == []


def test_tsh1_feedback_does_not_name_the_cause():
    s, shell = _tsh1(2)
    for desc, _ok, _ in s.grade(shell):
        assert not re.search(r"oom|memory", desc, re.I), desc
    shell.run("echo 'Job 2954 ended OUT_OF_MEMORY: oom_kill in step 2954.batch' > /root/ticket-4488.txt")
    assert all(ok for desc, ok, _ in s.grade(shell) if "ticket" in desc)


def test_journalctl_without_a_unit_prints_nothing_but_the_error():
    shell = Shell(concrete("tsh-first-look-at-logs", 2).build_state())
    shell.run("ssh dgx-01")
    for line in ("journalctl -u", "journalctl --unit", "journalctl -xeu"):
        out = shell.run(line)
        assert "requires an argument" in out and "oom" not in out, line
    assert "oom_kill" in shell.run("journalctl -u slurmd")
    assert "oom_kill" in shell.run("journalctl --unit=slurmd.service")
    assert "oom_kill" in shell.run("journalctl -xeu slurmd")
    assert shell.run("journalctl").count("\n") >= 3  # unfiltered on purpose still works
    assert shell.run("journalctl -k").startswith("[")


# ---- tsh-which-gpu-is-which: nvidia-smi evidence needs output that shows the target GPU


def _tsh2(variant: int):
    s = concrete("tsh-which-gpu-is-which", variant)
    shell = Shell(s.build_state())
    shell.run(f"ssh {s.params['node']}")
    return s, shell


def _evidence(s, shell) -> bool:
    desc, ok, err = s.grade(shell)[0]
    assert "nvidia-smi" in desc and not err, (desc, err)
    return ok


@pytest.mark.parametrize("variant", [0, 1, 2])
@pytest.mark.parametrize(
    "bad", ["nvidia-smi --query-gpu", "nvidia-smi -i 9 -q", "nvidia-smi --query-gpu=index,pci.bus_id --format=csv -i 9"]
)
def test_failed_nvidia_smi_queries_earn_no_evidence(variant, bad):
    s, shell = _tsh2(variant)
    assert not _evidence(s, shell)
    out = shell.run(bad)
    assert "Invalid combination" in out or "No devices were found" in out
    assert not _evidence(s, shell)


@pytest.mark.parametrize("variant", [0, 1, 2])
@pytest.mark.parametrize(
    "good",
    [
        "nvidia-smi",
        "nvidia-smi --query-gpu=index,pci.bus_id,uuid,serial --format=csv",
        "nvidia-smi -q -i {gpu}",
        "nvidia-smi -q -d PCI",
        "nvidia-smi --query-gpu=uuid --format=csv,noheader -i 00000000:{bus}:00.0",
        "nvidia-smi -L",
    ],
)
def test_nvidia_smi_output_that_shows_the_target_gpu_earns_evidence(variant, good):
    s, shell = _tsh2(variant)
    shell.run(good.format(**s.params))
    assert _evidence(s, shell)


def test_nvidia_smi_on_another_node_or_for_another_gpu_only_earns_nothing():
    s, shell = _tsh2(0)
    shell.run("nvidia-smi -q -i 0")  # a real GPU, not the one at 5D:00.0
    assert not _evidence(s, shell)
    shell.run("exit")
    shell.run("ssh dgx-01")
    shell.run("nvidia-smi")
    assert not _evidence(s, shell)


def test_tsh2_feedback_does_not_name_the_index():
    s = concrete("tsh-which-gpu-is-which", 0)
    shell = Shell(s.build_state())
    for desc, _ok, _ in s.grade(shell):
        assert not re.search(r"\(3\)|\b3\b", desc), desc
    shell.run("echo 'GPU index: 3' > /root/gpu-report-dgx-03.txt")
    assert next(ok for desc, ok, _ in s.grade(shell) if "index" in desc)


# ---- adm-bcm-cmsh-inventory: cmsh evidence needs output that carries an inventory fact


def _adm1():
    s = concrete("adm-bcm-cmsh-inventory")
    return s, Shell(s.build_state())


def _cmsh_evidence(s, shell) -> bool:
    desc, ok, err = s.grade(shell)[0]
    assert "cmsh" in desc and not err, (desc, err)
    return ok


@pytest.mark.parametrize(
    "bad",
    [
        "cmsh -c 'device; use dgx-99; show'",
        "cmsh -c 'category; use nosuch; show'",
        "cmsh -c",
        "cmsh -c ''",
        'cmsh -c "partition; list"',
    ],
)
def test_cmsh_without_inventory_output_earns_no_evidence(bad):
    s, shell = _adm1()
    assert not _cmsh_evidence(s, shell)
    shell.run(bad)
    assert not _cmsh_evidence(s, shell)


@pytest.mark.parametrize(
    "good",
    [
        'cmsh -c "device; list"',
        'cmsh -c "device; use dgx-03; show"',
        'cmsh -c "device; use dgx-03; get mac"',
        'cmsh -c "device; use dgx-03; get mac; get category"',
        'cmsh -c "category; use dgx-h100; show"',
        'cmsh -c "category; use dgx-h100; get softwareimage"',
        'cmsh -c "softwareimage; list"',
        'cmsh -c "softwareimage; use dgx-os-6.2-h100-image; get kernelversion"',
    ],
)
def test_cmsh_inventory_reads_earn_evidence(good):
    s, shell = _adm1()
    shell.run(good)
    assert _cmsh_evidence(s, shell)


def test_adm1_feedback_does_not_name_the_answers():
    s, shell = _adm1()
    for desc, _ok, _ in s.grade(shell):
        assert not re.search(r"dgx-h100|dgx-os-6\.2|7c:c2:55|5\.15\.0-1042", desc, re.I), desc


def test_cmsh_dash_c_needs_a_script():
    shell = Shell(make_cluster())
    for line in ("cmsh -c", "cmsh -c ''"):
        out = shell.run(line)
        assert "option requires an argument -- 'c'" in out and "Usage: cmsh" in out and shell.ctx.last_code == 1
    assert "dgx-03" in shell.run('cmsh -c "device; list"')


# ---- wkl-slurm-first-gpu-job: the job must match the briefing, and the queue must have shown it

WKL1_JOB, WKL1_WATCH = 2, 4  # positions of the job and the squeue/sacct criteria


def _wkl1(gres: str | None = "--gres=gpu:1", time: str | None = "--time=00:05:00", submit: str = ""):
    """A fresh wkl-1 with the script written and submitted; ``None`` leaves that #SBATCH line out."""
    s = concrete("wkl-slurm-first-gpu-job")
    shell = Shell(s.build_state())
    options = [f"#SBATCH {o}" for o in (gres, time) if o]
    feed = iter(["#!/bin/bash", "#SBATCH --job-name=hello-gpu", *options, "nvidia-smi -L", "EOF"])
    shell.run("cat > /root/hello-gpu.sbatch <<EOF", heredoc_reader=lambda _p: next(feed, None))
    assert "Submitted batch job 1001" in shell.run(f"sbatch {submit} /root/hello-gpu.sbatch".replace("  ", " "))
    return s, shell


def _wkl1_finish(shell) -> None:
    for _ in range(6):  # every command moves the simulated clock on
        if "COMPLETED" in shell.run("scontrol show job 1001"):
            return
    raise AssertionError("job 1001 did not complete")


def _wkl1_check(s, shell, index: int) -> bool:
    _desc, ok, err = s.grade(shell)[index]
    assert not err, err
    return ok


@pytest.mark.parametrize("gres", ["--gres=gpu:2", "--gpus=2", "-G 4", "--gpus-per-node=2", "--gres=gpu:1 -N 2"])
def test_wkl1_job_with_the_wrong_gpu_count_earns_no_job_credit(gres):
    s, shell = _wkl1(gres=gres)
    _wkl1_finish(shell)
    assert not _wkl1_check(s, shell, WKL1_JOB)


@pytest.mark.parametrize("time", ["--time=00:10:00", "--time=10", "-t 4", "-t 00:05:30", "-t 5:00:00", None])
def test_wkl1_job_with_the_wrong_time_limit_earns_no_job_credit(time):
    s, shell = _wkl1(time=time)
    _wkl1_finish(shell)
    assert not _wkl1_check(s, shell, WKL1_JOB)


@pytest.mark.parametrize(
    "gres", ["--gres=gpu:1", "--gres=gpu", "--gres=gpu:h100:1", "--gpus=1", "-G 1", "--gpus-per-node=1"]
)
@pytest.mark.parametrize("time", ["--time=00:05:00", "--time=5", "-t 5", "-t 5:00", "-t 0-00:05"])
def test_wkl1_accepts_every_spelling_of_one_gpu_and_five_minutes(gres, time):
    s, shell = _wkl1(gres=gres, time=time)
    assert not _wkl1_check(s, shell, WKL1_JOB)  # still running
    _wkl1_finish(shell)
    shell.run("sacct -j 1001")
    assert all(ok for _desc, ok, _err in s.grade(shell))


def test_wkl1_options_on_the_sbatch_command_line_count():
    s, shell = _wkl1(gres="--gres=gpu:2", time=None, submit="--gres=gpu:1 -t 5")
    _wkl1_finish(shell)
    assert _wkl1_check(s, shell, WKL1_JOB)


@pytest.mark.parametrize(
    "query",
    [
        "sacct -j 999999",
        "sacct",
        "sacct -u nobody",
        "squeue",
        "squeue -j 999999",
        "watch squeue",
        "sacct --name hello-gpu",
    ],
)
def test_wkl1_queries_before_the_job_exists_earn_no_credit(query):
    s = concrete("wkl-slurm-first-gpu-job")
    shell = Shell(s.build_state())
    shell.run(query)
    assert not _wkl1_check(s, shell, WKL1_WATCH)
    assert not any(ok for _desc, ok, _err in s.grade(shell))


def test_wkl1_queries_that_do_not_show_the_job_earn_no_credit():
    s, shell = _wkl1()
    _wkl1_finish(shell)
    for query in ("squeue", "sacct -j 999999", "sacct -u nobody", "squeue -u nobody", "scontrol show job 1001"):
        shell.run(query)
    assert not _wkl1_check(s, shell, WKL1_WATCH)


@pytest.mark.parametrize(
    ("query", "finished"),
    [
        ("squeue", False),
        ("squeue -u root", False),
        ("squeue -h -o %i", False),
        ("watch squeue", False),
        ("squeue | grep hello", False),
        ("squeue -j 1001", True),
        ("sacct", True),
        ("sacct -j 1001", True),
        ("sacct -j 1001 -p -n", True),
        ("sacct -X -n -o JobID,State", True),
        ("sacct --name hello-gpu", True),
    ],
)
def test_wkl1_queries_that_show_the_job_earn_credit(query, finished):
    s, shell = _wkl1()
    if finished:
        _wkl1_finish(shell)
    assert not _wkl1_check(s, shell, WKL1_WATCH)
    shell.run(query)
    assert _wkl1_check(s, shell, WKL1_WATCH)


def test_wkl1_criteria_are_not_cut_short_and_do_not_name_the_answers():
    s, shell = _wkl1()
    descs = [desc for desc, _ok, _err in s.grade(shell)]
    assert any("#SBATCH line" in d for d in descs)
    for desc in (descs[WKL1_JOB], descs[WKL1_WATCH]):
        assert not re.search(r"gpu:1|\bone\b|\bfive\b|\b5\b|00:05", desc), desc


def test_no_yaml_text_is_cut_at_a_hash():
    """A plain YAML scalar ends at ' #', so text that mentions #SBATCH must be quoted."""
    from pathlib import Path

    import greenray

    data = Path(greenray.__file__).parent / "data"
    cut = [
        f"{path.name}:{n}"
        for path in sorted(data.rglob("*.yaml"))
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if re.match(r"\s*(- )?(\w+: )?[^\s\"'|>#\[{-][^\"']* #", line) and re.match(r"\s*(- )?\w+: |\s*- ", line)
    ]
    assert cut == []


# ---- shared helpers for the evidence fixes below


def _lab(scenario_id: str):
    s = concrete(scenario_id)
    return s, Shell(s.build_state())


def _check(s, shell, wording: str) -> bool:
    """The result of the one criterion whose description contains ``wording``."""
    found = [(ok, err) for desc, ok, err in s.grade(shell) if wording in desc]
    assert len(found) == 1 and not found[0][1], found
    return found[0][0]


def _solve(s, shell, skip: str) -> None:
    """Replay the reference solution without the read-only steps matching ``skip``."""
    for step in s.solution:
        if not re.search(skip, step):
            lines = step.split("\n")
            feed = iter(lines[1:])
            shell.run(lines[0], heredoc_reader=lambda _p, feed=feed: next(feed, None))


# ---- adm-slurm-partition-time-limit: scontrol must have shown the defq partition

ADM2_FIX = ["sed -i 's/MaxTime=INFINITE/MaxTime=2-00:00:00/' /etc/slurm/slurm.conf", "scontrol reconfigure"]


@pytest.mark.parametrize(
    "bad",
    [
        "scontrol show partition missing",
        "scontrol show partition",  # with no partition left to show, see the test body
        "scontrol show part defq,missing",
        "scontrol show config",
        "scontrol show node dgx-01",
        "sinfo",
    ],
)
def test_adm2_failed_or_unrelated_lookups_earn_no_evidence(bad):
    s, shell = _lab("adm-slurm-partition-time-limit")
    for line in ADM2_FIX:
        shell.run(line)
    if bad == "scontrol show partition":
        shell.ctx.state.slurm.partitions.clear()
    out = shell.run(bad)
    assert "PartitionName=defq" not in out
    results = s.grade(shell)
    assert not _check(s, shell, "scontrol showed you")
    if shell.ctx.state.slurm.partitions:
        assert [ok for _desc, ok, _err in results] == [True, True, True, False]


@pytest.mark.parametrize(
    "good",
    [
        "scontrol show partition defq",
        "scontrol show partition",
        "scontrol show partitions",
        "scontrol show part defq",
        "scontrol show part defq -o",
        "scontrol -o show partition defq",
        "scontrol show partition defq | grep MaxTime",
    ],
)
@pytest.mark.parametrize("before_the_change", [True, False])
def test_adm2_a_successful_partition_lookup_earns_evidence(good, before_the_change):
    s, shell = _lab("adm-slurm-partition-time-limit")
    order = [good, *ADM2_FIX] if before_the_change else [*ADM2_FIX, good]
    for line in order:
        shell.run(line)
    assert all(ok for _desc, ok, _err in s.grade(shell))


def test_adm2_still_protects_the_running_job():
    s, shell = _lab("adm-slurm-partition-time-limit")
    for line in ["scontrol show partition defq", *ADM2_FIX, "scancel 2301"]:
        shell.run(line)
    assert not _check(s, shell, "job 2301 is still running")
    assert _check(s, shell, "scontrol showed you")


# ---- wkl-k8s-pod-pending-nodeselector: kubectl must have returned the events and the label

WKL2 = "wkl-k8s-pod-pending-nodeselector"
WKL2_EVENTS, WKL2_LABELS = "scheduling events", "GPU product label"


@pytest.mark.parametrize(
    "bad",
    [
        "kubectl describe pod missing",
        "kubectl describe pod cuda-test -n kube-system",
        "kubectl get events -n kube-system",
        "kubectl get events --field-selector involvedObject.name=nope",
        "kubectl get pods",
    ],
)
def test_wkl2_lookups_that_return_no_pod_events_earn_no_evidence(bad):
    s, shell = _lab(WKL2)
    assert "FailedScheduling" not in shell.run(bad)
    assert not _check(s, shell, WKL2_EVENTS)


@pytest.mark.parametrize(
    "bad",
    [
        "kubectl get node missing --show-labels",
        "kubectl describe node missing",
        "kubectl get nodes",
        "kubectl get nodes -L nosuch",
        "kubectl get node bcm-head --show-labels",
        "kubectl describe node bcm-head",
        "kubectl get nodes -l nvidia.com/gpu.product=NVIDIA-H100-80GB",
        "kubectl describe pod cuda-test",
    ],
)
def test_wkl2_lookups_that_return_no_gpu_label_earn_no_evidence(bad):
    s, shell = _lab(WKL2)
    shell.run(bad)
    assert not _check(s, shell, WKL2_LABELS)


def test_wkl2_the_reported_failed_lookups_earn_nothing():
    s, shell = _lab(WKL2)
    assert "NotFound" in shell.run("kubectl describe pod missing")
    assert "NotFound" in shell.run("kubectl get node missing --show-labels")
    assert not any(ok for _desc, ok, _err in s.grade(shell))


@pytest.mark.parametrize(
    "good",
    [
        "kubectl describe pod cuda-test",
        "kubectl describe pods",
        "kubectl describe po cuda-test",
        "kubectl -n default describe pod cuda-test",
        "kubectl get events",
        "kubectl get ev -A",
        "kubectl get events | grep cuda-test",
    ],
)
def test_wkl2_pod_events_earn_evidence(good):
    s, shell = _lab(WKL2)
    shell.run(good)
    assert _check(s, shell, WKL2_EVENTS) and not _check(s, shell, WKL2_LABELS)


@pytest.mark.parametrize(
    "good",
    [
        "kubectl get nodes -L nvidia.com/gpu.product",
        "kubectl get nodes --show-labels",
        "kubectl get node dgx-02 --show-labels",
        "kubectl get no --show-labels",
        "kubectl describe node dgx-01",
        "kubectl describe nodes",
        "kubectl describe node dgx-01 | grep -i product",
        "kubectl get node dgx-01 -o yaml",
        "kubectl get nodes -l nvidia.com/gpu.product=NVIDIA-H100-80GB-HBM3",
    ],
)
def test_wkl2_node_labels_earn_evidence(good):
    s, shell = _lab(WKL2)
    shell.run(good)
    assert _check(s, shell, WKL2_LABELS) and not _check(s, shell, WKL2_EVENTS)


def test_wkl2_recreating_the_pod_without_diagnosis_leaves_only_the_evidence_open():
    s, shell = _lab(WKL2)
    _solve(s, shell, r"^kubectl (get|describe|logs)|^cat ")
    assert [ok for _desc, ok, _err in s.grade(shell)] == [False, False, True, True, True]
    shell.run("kubectl describe pod cuda-test")  # the recreated pod has no FailedScheduling event to show
    assert not _check(s, shell, WKL2_EVENTS)


# ---- wkl-k8s-taint-toleration: evidence needs output, and the manifest is graded as parsed YAML

WKL3 = "wkl-k8s-taint-toleration"
WKL3_SEEN, WKL3_FILE = "kubectl showed you", "/root/triton.yaml"


@pytest.mark.parametrize(
    "bad",
    [
        "kubectl describe pod missing",
        "kubectl describe node missing",
        "kubectl describe node bcm-head",
        "kubectl describe deployment triton-inference",
        "kubectl get events -n kube-system",
        "kubectl get pods",
    ],
)
def test_wkl3_lookups_that_show_neither_events_nor_taints_earn_no_evidence(bad):
    s, shell = _lab(WKL3)
    shell.run(bad)
    assert not _check(s, shell, WKL3_SEEN)


@pytest.mark.parametrize(
    "good",
    [
        "kubectl describe pod -l app=triton-inference",
        "kubectl describe pods",
        "kubectl get events",
        "kubectl describe node dgx-01",
        "kubectl describe node dgx-01 | grep Taints",
        "kubectl describe nodes",
    ],
)
def test_wkl3_pod_events_or_node_taints_earn_evidence(good):
    s, shell = _lab(WKL3)
    shell.run(good)
    assert _check(s, shell, WKL3_SEEN)


def _wkl3_with_toleration(toleration: str):
    """wkl-3 after the manifest's toleration block was replaced by ``toleration`` and applied."""
    s, shell = _lab(WKL3)
    files = shell.ctx.state.nodes["bcm-head"].files
    old = '      - key: nvidia.com/gpu\n        operator: Equal\n        value: "true"\n        effect: NoSchedule\n'
    assert old in files["/root/triton.yaml"]
    files["/root/triton.yaml"] = files["/root/triton.yaml"].replace(old, toleration)
    shell.run("kubectl describe pod -l app=triton-inference")
    shell.run("kubectl apply -f /root/triton.yaml")
    shell.run("kubectl rollout status deployment/triton-inference")
    return s, shell


@pytest.mark.parametrize(
    "toleration",
    [
        '      - key: nvidia.com/gpu\n        operator: Equal\n        value: "present"\n        effect: NoSchedule\n',
        "      - key: nvidia.com/gpu\n        operator: Equal\n        value: 'present'\n        effect: NoSchedule\n",
        "      - key: nvidia.com/gpu\n        operator: Equal\n        value: present\n        effect: NoSchedule\n",
        "      - key: 'nvidia.com/gpu'\n        value:   present   # the taint's value\n        effect: \"NoSchedule\"\n",
        "      - key: nvidia.com/gpu\n        operator: Exists\n        effect: NoSchedule\n",
        "      - key: nvidia.com/gpu\n        operator: 'Exists'\n",
        "      - {key: nvidia.com/gpu, operator: Equal, value: present, effect: NoSchedule}\n",
        "      - operator: Exists\n",
    ],
    ids=["double", "single", "plain", "comment", "exists", "exists-any-effect", "flow", "exists-any-key"],
)
def test_wkl3_every_yaml_spelling_of_a_matching_toleration_passes(toleration):
    s, shell = _wkl3_with_toleration(toleration)
    assert all(ok for _desc, ok, _err in s.grade(shell)), s.grade(shell)


@pytest.mark.parametrize(
    "toleration",
    [
        '      - key: nvidia.com/gpu\n        operator: Equal\n        value: "true"\n        effect: NoSchedule\n',
        "      - key: nvidia.com/gpu\n        operator: Equal\n        value: present\n        effect: NoExecute\n",
        "      - key: nvidia.com/mig\n        operator: Exists\n",
        "      # value: present\n      # operator: Exists\n      - key: nvidia.com/gpu\n        value: absent\n",
    ],
    ids=["unchanged", "wrong-effect", "wrong-key", "fix-only-in-a-comment"],
)
def test_wkl3_a_toleration_that_does_not_match_the_taint_fails(toleration):
    s, shell = _wkl3_with_toleration(toleration)
    assert not _check(s, shell, WKL3_FILE)
    assert not _check(s, shell, "replicas Running")


def test_wkl3_a_manifest_that_is_not_valid_yaml_fails_without_an_error():
    s, shell = _lab(WKL3)
    shell.ctx.state.nodes["bcm-head"].files["/root/triton.yaml"] = "kind: Deployment\n  value: present\n: ["
    assert not _check(s, shell, WKL3_FILE)
    del shell.ctx.state.nodes["bcm-head"].files["/root/triton.yaml"]
    assert not _check(s, shell, WKL3_FILE)


def test_wkl3_fixing_the_live_object_only_does_not_satisfy_the_manifest_criterion():
    s, shell = _lab(WKL3)
    patch = '{"spec":{"template":{"spec":{"tolerations":[{"key":"nvidia.com/gpu","operator":"Exists"}]}}}}'
    assert "patched" in shell.run(f"kubectl patch deployment triton-inference -p '{patch}'")
    assert _check(s, shell, "Deployment's pod template")
    assert not _check(s, shell, WKL3_FILE)


# ---- wkl-k8s-pod-pending-gpu: the manifest is graded as parsed YAML, the diagnosis by its output

WKL4 = "wkl-k8s-pod-pending-gpu"
WKL4_FILE, WKL4_SEEN = "/root/bert-pretrain.yaml", "kubectl showed you"


def _wkl4(variant: int = 0):
    s = concrete(WKL4, variant)
    return s, Shell(s.build_state())


def _wkl4_recreate(s, shell, resources: str) -> None:
    """Replace the trainer's resources block in the manifest, uncordon the nodes and recreate the pod."""
    files = shell.ctx.state.nodes["bcm-head"].files
    old = f"    resources:\n      limits:\n        nvidia.com/gpu: {s.params['bad']}\n"
    assert old in files[WKL4_FILE]
    files[WKL4_FILE] = files[WKL4_FILE].replace(old, resources)
    shell.run("kubectl describe pod bert-pretrain")
    shell.run(f"kubectl uncordon {s.params['c1']} {s.params['c2']}")
    shell.run(f"kubectl replace --force -f {WKL4_FILE}")


@pytest.mark.parametrize("variant", [0, 1, 2])
@pytest.mark.parametrize(
    "resources",
    [
        "    resources:\n      limits:\n        nvidia.com/gpu: 8\n",
        '    resources:\n      limits:\n        "nvidia.com/gpu": 8\n',
        "    resources:\n      limits:\n        'nvidia.com/gpu': 8\n",
        '    resources:\n      limits:\n        nvidia.com/gpu: "8"\n',
        "    resources:\n      limits:\n        nvidia.com/gpu: 8   # one full DGX\n",
        "    resources:\n      limits: {nvidia.com/gpu: 8}\n",
        "    resources:\n      requests:\n        nvidia.com/gpu: 8\n      limits:\n        nvidia.com/gpu: 8\n",
    ],
    ids=["plain", "quoted-key", "single-quoted-key", "quoted-value", "comment", "flow", "requests-and-limits"],
)
def test_wkl4_every_yaml_spelling_of_eight_gpus_passes(variant, resources):
    s, shell = _wkl4(variant)
    _wkl4_recreate(s, shell, resources)
    assert "Running" in shell.run("kubectl get pod bert-pretrain")
    assert all(ok for _desc, ok, _err in s.grade(shell)), s.grade(shell)


@pytest.mark.parametrize(
    "resources",
    [
        "    resources:\n      limits:\n        nvidia.com/gpu: 4\n",
        "    resources:\n      limits:\n        nvidia.com/gpu: 80\n",
        "    resources:\n      limits:\n        nvidia.com/mig-1g.10gb: 8\n",
        "    # nvidia.com/gpu: 8\n    resources:\n      limits:\n        nvidia.com/gpu: 2\n",
    ],
    ids=["four", "eighty", "another-resource", "eight-only-in-a-comment"],
)
def test_wkl4_a_manifest_without_eight_gpus_fails(resources):
    s, shell = _wkl4()
    _wkl4_recreate(s, shell, resources)
    assert not _check(s, shell, WKL4_FILE)


def test_wkl4_eight_gpus_on_another_pod_or_container_do_not_count():
    s, shell = _wkl4()
    files = shell.ctx.state.nodes["bcm-head"].files
    wrong_pod = (
        files[WKL4_FILE].replace("name: bert-pretrain", "name: other").replace(f"gpu: {s.params['bad']}", "gpu: 8")
    )
    sidecar = "  - name: sidecar\n    image: busybox\n    resources:\n      limits:\n        nvidia.com/gpu: 8\n"
    for text in (wrong_pod, files[WKL4_FILE] + sidecar, "kind: Pod\n  nvidia.com/gpu: 8\n: [", ""):
        files[WKL4_FILE] = text
        assert not _check(s, shell, WKL4_FILE)


@pytest.mark.parametrize(
    "bad",
    [
        "kubectl describe pod missing",
        "kubectl describe pod -l app=inference",
        "kubectl describe pod bert-pretrain -n kube-system",
        "kubectl get events -n kube-system",
        "kubectl get pods",
    ],
)
def test_wkl4_lookups_without_the_scheduling_event_earn_no_evidence(bad):
    s, shell = _wkl4()
    shell.run(bad)
    assert not _check(s, shell, WKL4_SEEN)
    assert [ok for _desc, ok, _err in s.grade(shell)] == [False, False, False, False, True, True, False]


@pytest.mark.parametrize(
    "good",
    [
        "kubectl describe pod bert-pretrain",
        "kubectl describe pods",
        "kubectl describe po -l app=bert-pretrain",
        "kubectl get events",
        "kubectl get ev -A",
        "kubectl get events -A | grep bert",
    ],
)
def test_wkl4_the_pods_scheduling_event_earns_evidence(good):
    s, shell = _wkl4()
    shell.run(good)
    assert _check(s, shell, WKL4_SEEN)


def test_wkl4_safety_criteria_still_catch_a_deleted_inference_pod():
    s, shell = _wkl4()
    victim = next(p.name for p in shell.ctx.state.k8s.pods if p.name.startswith("inference-"))
    shell.run(f"kubectl delete pod {victim}")
    assert not _check(s, shell, "no running pod was deleted")


# ---- ins-bcm-add-node-category: cmsh must have returned inventory of dgx-04 or of dgx-h100

INS2 = "ins-bcm-add-node-category"


@pytest.mark.parametrize(
    "bad",
    [
        'cmsh -c "device; show missing"',
        'cmsh -c "device; use missing; show"',
        'cmsh -c "category; use nosuch; show"',
        'cmsh -c "device; use dgx-01; show"',
        'cmsh -c "device; status dgx-01"',
        'cmsh -c "category; use default; show"',
        'cmsh -c "partition; list"',
        'cmsh -c "device; use dgx-04; get nosuchproperty"',
    ],
)
def test_ins2_cmsh_without_relevant_inventory_earns_no_evidence(bad):
    s, shell = _lab(INS2)
    shell.run(bad)
    assert not _check(s, shell, "cmsh showed you")


def test_ins2_changing_the_node_without_looking_leaves_the_evidence_open():
    s, shell = _lab(INS2)
    shell.run('cmsh -c "device; use dgx-04; set category dgx-h100; commit"')
    shell.run('cmsh -c "device; reboot -n dgx-04"')
    assert [ok for _desc, ok, _err in s.grade(shell)] == [True, True, True, True, False]


@pytest.mark.parametrize(
    "good",
    [
        'cmsh -c "device; list"',
        'cmsh -c "device; use dgx-04; show"',
        'cmsh -c "device; use dgx-04; get category"',
        'cmsh -c "device; status"',
        'cmsh -c "device; status dgx-04"',
        'cmsh -c "category; list"',
        'cmsh -c "category; use dgx-h100; show"',
        'cmsh -c "category; use dgx-h100; get softwareimage"',
        'cmsh -c "softwareimage; list"',
    ],
)
def test_ins2_device_and_category_inspections_earn_evidence(good):
    s, shell = _lab(INS2)
    shell.run(good)
    assert _check(s, shell, "cmsh showed you")
