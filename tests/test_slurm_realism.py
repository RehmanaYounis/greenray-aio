"""Slurm the way the controller behaves: arrays, memory, time limits, reservations, permissions."""

import pytest

from greenray.commands.slurm.scheduler import tick
from greenray.core.shell import Shell
from greenray.core.state import make_cluster


def sh():
    return Shell(make_cluster())


def test_job_arrays_run_within_their_throttle():
    s = sh()
    assert s.run("sbatch --array=1-4%2 --wrap='sleep 600'") == "Submitted batch job 1001"
    q = s.run("squeue")
    assert "1001_[3-4%2]" in q and "1001_1" in q and "1001_2" in q
    assert sorted(s.run("squeue -r -h -o %i").split()) == ["1001_1", "1001_2", "1001_3", "1001_4"]
    assert "ArrayJobId=1001 ArrayTaskId=2" in s.run("scontrol show job 1001_2")
    s.run("scancel 1001_2")
    assert "1001_2" not in s.run("squeue -h -o %i -t R")
    s.run("scancel 1001")
    assert s.run("squeue -h") == ""
    assert "1001_1" in s.run("sacct -j 1001 -o jobid -n")


def test_array_task_env_and_output_pattern():
    s = sh()
    s.run("sbatch --array=0-1 --wrap='echo task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT'")
    for _ in range(5):
        tick(s.ctx)
    files = s.ctx.node.files
    assert files["/root/slurm-1001_0.out"].strip() == "task 0 of 2"
    assert files["/root/slurm-1001_1.out"].strip() == "task 1 of 2"


def test_test_only_creates_nothing():
    s = sh()
    out = s.run("sbatch --test-only -N 2 --wrap=hostname")
    assert out.startswith("sbatch: Job 1001 to start at") and "on nodes dgx-[01-02]" in out
    assert s.run("squeue -h") == ""
    assert "sbatch: Job 1001" in s.run("sbatch --test-only --wrap=hostname")  # the id was not consumed


def test_begin_time_holds_the_job():
    s = sh()
    s.run("sbatch --begin=now+1hour --wrap=hostname")
    assert "(BeginTime)" in s.run("squeue")
    assert "EligibleTime=" in s.run("scontrol show job 1001")
    s.ctx.state.slurm.clock += 3700
    tick(s.ctx)
    assert "BeginTime" not in s.run("squeue -h -o %r")
    assert "Invalid --begin" in s.run("sbatch --begin=whenever --wrap=hostname")


def test_memory_is_checked_and_accounted():
    s = sh()
    assert "Requested node configuration is not available" in s.run("sbatch --mem=3000G --wrap=hostname")
    s.run("sbatch -w dgx-01 --mem=1500G --wrap='sleep 600'")  # 1001: a rejected submission takes no id
    s.run("sbatch -w dgx-01 --mem=600G --wrap='sleep 600'")
    q = s.run("squeue -h -o '%i %t %r'")
    assert "1001 R" in q and "1002 PD Resources" in q
    show = s.run("scontrol show job 1001")
    assert "mem=1500G" in show and "MinMemoryNode=1500G" in show
    assert "1500G" in s.run("sacct -j 1001 -o reqmem -n")


def test_time_limit_ends_the_job_with_timeout():
    s = sh()
    s.run("sbatch -t 1 --wrap='sleep 3600'")
    for _ in range(4):
        tick(s.ctx)
    assert "TIMEOUT" in s.run("sacct -j 1001 -o state -n")
    assert any("Time limit exhausted for JobId=1001" in ln for ln in s.ctx.node.journal["slurmctld"])


def test_wrap_sleep_sets_the_duration_and_short_jobs_finish():
    s = sh()
    s.run("sbatch --wrap='sleep 10'")
    tick(s.ctx)
    assert "COMPLETED" in s.run("sacct -j 1001 -o state -n")


def test_salloc_shell_is_released_by_exit():
    s = sh()
    out = s.run("salloc -N 1 --gres=gpu:2")
    assert "Granted job allocation 1001" in out
    assert s.ctx.host == "bcm-head"
    assert s.run("echo $SLURM_JOB_ID") == "1001"
    assert "1001" in s.run("squeue -h -o %i -t R")
    assert s.run("srun hostname") == "dgx-01"
    out = s.run("exit")
    assert "salloc: Relinquishing job allocation 1001" in out
    assert s.run("squeue -h") == ""
    assert s.run("echo $SLURM_JOB_ID") == ""
    assert s.ctx.host == "bcm-head"


def test_salloc_with_a_command_runs_it_and_releases():
    s = sh()
    out = s.run("salloc -N 2 srun hostname")
    assert "dgx-01" in out and "dgx-02" in out and "Relinquishing" in out
    assert s.run("squeue -h") == ""


def test_dependency_validation():
    s = sh()
    assert "Job dependency problem" in s.run("sbatch --dependency=afterok:9999 --wrap=hostname")
    s.run("sbatch -J same --wrap='sleep 600'")
    s.run("sbatch -J same --dependency=singleton --wrap='sleep 600'")
    assert "Dependency" in s.run("squeue -j 1002 -h -o %r")


def test_exclude_leaving_too_few_nodes_is_rejected():
    s = sh()
    assert "Requested node configuration is not available" in s.run("sbatch -N 4 -x dgx-01 --wrap=hostname")


def test_sbatch_reads_the_script_from_stdin_and_waits():
    s = sh()
    lines = iter(["#!/bin/bash", "#SBATCH -J fromstdin", "hostname", "EOF"])
    out = s.run("sbatch <<EOF", heredoc_reader=lambda _p: next(lines))
    assert out == "Submitted batch job 1001"
    assert "fromstdin" in s.run("sacct -j 1001 -o jobname -n")
    assert s.run("sbatch --wait --wrap='sleep 60'") == "Submitted batch job 1002"
    assert "COMPLETED" in s.run("sacct -j 1002 -o state -n")


def test_mail_and_other_accepted_options_are_recorded():
    s = sh()
    s.run("sbatch --mail-type=END,FAIL --mail-user=ops@example.com --tasks-per-node=2 --wrap=hostname")
    show = s.run("scontrol show job 1001")
    assert "MailUser=ops@example.com MailType=END,FAIL" in show
    assert "NumTasks=2" in show


def test_maintenance_reservation_blocks_other_jobs():
    s = sh()
    assert (
        s.run(
            "scontrol create reservation ReservationName=maint StartTime=now Duration=120 Nodes=dgx-03 Users=root Flags=MAINT"
        )
        == "Reservation created: maint"
    )
    assert "Nodes=dgx-03" in s.run("scontrol show reservation") and "Flags=MAINT" in s.run("scontrol show reservation")
    assert "maint" in s.run("sinfo -h -o %T -n dgx-03")
    s.run("sbatch -w dgx-03 --wrap='sleep 600'")
    assert "ReqNodeNotAvail, Reserved for maintenance" in s.run("squeue -h -o %r")
    s.run("sbatch -w dgx-03 --reservation=maint --wrap='sleep 600'")
    assert "1002 R" in s.run("squeue -h -o '%i %t'")
    s.run("scontrol update reservation ReservationName=maint Duration=240")
    assert "Duration=04:00:00" in s.run("scontrol show reservation")
    assert s.run("scontrol delete reservation maint") == ""
    assert "No reservations" in s.run("scontrol show reservation")
    assert "Invalid reservation name" in s.run("scontrol delete ReservationName=maint")


def test_suspend_resume_and_top():
    s = sh()
    s.run("sbatch --wrap='sleep 600'")
    assert s.run("scontrol suspend 1001") == ""
    assert s.run("squeue -h -o %t -j 1001") == "S"
    assert s.run("scontrol resume 1001") == ""
    assert s.run("squeue -h -o %t -j 1001") == "R"
    s.run("sacctmgr -i add user alice account=root")
    s.run("su - alice -c 'sbatch -H --wrap=hostname'")
    s.run("su - alice -c 'sbatch -H --wrap=hostname'")
    s.run("scontrol top 1003")
    pri = {j.id: j.priority for j in s.ctx.state.slurm.jobs}
    assert pri[1003] > pri[1002]


def test_plain_users_cannot_administer():
    s = sh()
    s.run("sacctmgr -i add user alice account=root")
    s.run("sbatch --wrap='sleep 600'")
    alice = "su - alice -c "
    assert "Access/permission denied" in s.run(alice + "'scancel 1001'")
    assert "Access/permission denied" in s.run(alice + "'scontrol update NodeName=dgx-04 State=DRAIN Reason=x'")
    assert "Access/permission denied" in s.run(alice + "'scontrol reconfigure'")
    assert "Access/permission denied" in s.run(alice + "'sacctmgr -i add account team'")
    assert "Permission denied" in s.run(alice + "'cmsh -c \"device; list\"'")
    s.run(alice + "'sbatch -H --wrap=hostname'")
    assert s.run(alice + "'scontrol update JobId=1002 TimeLimit=10'") == ""  # your own pending job is yours to change
    assert "1001" in s.run("squeue -h -o %i")
    assert s.ctx.state.nodes["dgx-04"].get("slurm_state", "").upper() != "DRAIN"


def test_admin_profile_user_may_run_cmsh():
    s = sh()
    s.run('cmsh -c "user; add bob; set profile admin; commit"')
    assert "dgx-01" in s.run("su - bob -c 'cmsh -c \"device; list\"'")


def test_reconfigure_rejects_a_broken_slurm_conf():
    s = sh()
    s.run("echo 'Bogus=1' >> /etc/slurm/slurm.conf")
    out = s.run("scontrol reconfigure")
    assert "Parsing error at unrecognized key: Bogus" in out and "slurm_reconfigure error" in out
    s.run("sed -i '/Bogus/d' /etc/slurm/slurm.conf")
    s.run("echo 'PartitionName=bad Nodes=dgx-09 State=UP' >> /etc/slurm/slurm.conf")
    assert "Invalid node names in partition bad" in s.run("scontrol reconfigure")
    assert "bad" not in s.run("sinfo -h -o %P")


@pytest.mark.parametrize("spec,count", [("0,3,5", 3), ("1-10:2", 5), ("2-3", 2)])
def test_array_specs(spec, count):
    s = sh()
    s.run(f"sbatch --array={spec} --wrap='sleep 600'")
    assert len([i for i in s.run("squeue -r -h -o %i").split() if "_" in i]) == count


def test_cmsh_staging_is_discarded_at_the_end_of_an_invocation():
    s = sh()
    mac = s.ctx.state.bcm.devices["dgx-01"].mac
    out = s.run('cmsh -c "device; use dgx-01; set mac 00:11:22:33:44:55"')
    assert "uncommitted changes discarded" in out
    assert not s.ctx.state.bcm.uncommitted
    assert s.ctx.state.bcm.devices["dgx-01"].mac == mac
    s.run('cmsh -c "device; use dgx-01; set mac 00:11:22:33:44:55; commit"')
    assert s.ctx.state.bcm.devices["dgx-01"].mac == "00:11:22:33:44:55"
