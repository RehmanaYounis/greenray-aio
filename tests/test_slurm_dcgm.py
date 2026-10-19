"""Focused tests for the Slurm and DCGM emulators (aiolab/commands/slurm.py, dcgmi.py)."""

from aiolab.commands import slurm as sl
from aiolab.core.shell import Shell
from aiolab.core.state import make_cluster


def sh(host=None):
    s = Shell(make_cluster())
    if host:
        s.run(f"ssh {host}")
    return s


def write(s, path, text, host="bcm-head"):
    s.ctx.state.nodes[host].files[path] = text


# ---------------------------------------------------------------- parsers


def test_hostlist_roundtrip():
    assert sl.expand_hostlist("dgx-[01-03,05]") == ["dgx-01", "dgx-02", "dgx-03", "dgx-05"]
    assert sl.expand_hostlist("dgx-01,dgx-[03-04]") == ["dgx-01", "dgx-03", "dgx-04"]
    assert sl.expand_hostlist("/dev/nvidia[0-3]") == [f"/dev/nvidia{i}" for i in range(4)]
    assert sl.compress_hostlist(["dgx-01", "dgx-02", "dgx-04"]) == "dgx-[01-02,04]"
    assert sl.compress_hostlist(["dgx-03"]) == "dgx-03"


def test_time_and_gres_parsing():
    assert sl.parse_time("04:00:00") == 240
    assert sl.parse_time("2-00:00:00") == 2880
    assert sl.parse_time("30") == 30
    assert sl.parse_time("UNLIMITED") is None
    assert sl.parse_gres_spec("gpu:h100:8") == ("h100", 8)
    assert sl.parse_gres_spec("gpu:4") == (None, 4)
    assert sl.parse_gres_spec("(null)") == (None, 0)
    assert sl.parse_tres("gres/gpu=8,cpu=64") == {"gres/gpu": 8, "cpu": 64}


def test_slurm_conf_parser():
    conf = sl.parse_slurm_conf(
        "GresTypes=gpu\nNodeName=dgx-[01-02] CPUs=224 Gres=gpu:h100:8\n"
        "PartitionName=defq Nodes=dgx-[01-02] Default=YES MaxTime=04:00:00 QOS=x\n"
    )
    assert conf["params"]["grestypes"] == "gpu"
    assert conf["nodes"]["dgx-02"]["gres"] == "gpu:h100:8"
    assert conf["partitions"]["defq"]["qos"] == "x"


def test_job_option_parser():
    opts, rest = sl.parse_job_opts("srun", ["-N2", "--gres=gpu:8", "-p", "defq", "--pty", "bash"])
    assert opts == {"nodes": "2", "gres": "gpu:8", "partition": "defq", "pty": True}
    assert rest == ["bash"]


# ---------------------------------------------------------------- sinfo / squeue / scontrol


def test_sinfo_default_and_drain_reason():
    s = sh()
    out = s.run("sinfo")
    assert "defq*" in out and "idle dgx-[01-04]" in out
    assert "reason" in s.run("scontrol update nodename=dgx-02 state=drain")  # reason is mandatory
    s.run('scontrol update nodename=dgx-02 state=drain reason="fan failure"')
    assert "drain dgx-02" in s.run("sinfo")
    assert "fan failure" in s.run("sinfo -R")
    assert "Invalid node state" in s.run("scontrol update nodename=dgx-01 state=resume")
    s.run("scontrol update nodename=dgx-02 state=resume")
    assert s.ctx.state.slurm.nodes["dgx-02"].state == "IDLE"


def test_sbatch_script_directives_and_output_file():
    s = sh()
    write(
        s,
        "/root/job.sh",
        "#!/bin/bash\n#SBATCH -J t1\n#SBATCH -N 1\n#SBATCH --gres=gpu:2\n#SBATCH -o /root/t1.out\nnvidia-smi -L\n",
    )
    assert s.run("sbatch /root/job.sh") == "Submitted batch job 1001"
    j = s.ctx.state.slurm.jobs[0]
    assert (j.name, j.gpus_per_node, j.state) == ("t1", 2, "RUNNING")
    # cgroup ConstrainDevices: the job only sees its 2 GPUs
    assert s.run("cat /root/t1.out").count("GPU ") == 2
    for _ in range(4):
        s.run("squeue")
    assert j.state == "COMPLETED"


def test_sbatch_rejections():
    s = sh()
    assert "does not look like a batch script" in s.run("echo hostname > /root/x.sh; sbatch /root/x.sh")
    assert "Requested node configuration is not available" in s.run("sbatch --gres=gpu:16 --wrap=hostname")
    assert "invalid partition" in s.run("sbatch -p nope --wrap=hostname")


def test_srun_gpu_listing_and_multinode():
    s = sh()
    out = s.run("srun -N 2 --gres=gpu:4 nvidia-smi -L")
    assert out.count("GPU ") == 8
    assert sorted(set(s.run("srun -N 2 --ntasks-per-node=2 hostname").split())) == ["dgx-01", "dgx-02"]


def test_pending_reasons():
    s = sh()
    s.run("scontrol update partitionname=defq state=down")
    s.run("sbatch --gres=gpu:8 --wrap=hostname")
    assert "(PartitionDown)" in s.run("squeue")
    s.run("scontrol update partitionname=defq state=up")
    assert s.ctx.state.slurm.jobs[0].state == "RUNNING"
    # fill the cluster, then a further job waits for Resources and the next one for Priority
    s.run("sbatch -N 3 --gres=gpu:8 --wrap='sleep 600'")
    s.run("sbatch -N 2 --gres=gpu:8 --wrap='sleep 600'")
    s.run("sbatch -N 2 --gres=gpu:8 --wrap='sleep 600'")
    reasons = [j.reason for j in s.ctx.state.slurm.jobs if j.state == "PENDING"]
    assert reasons == ["Resources", "Priority"]


def test_requested_drained_node_reason():
    s = sh()
    s.run("scontrol update nodename=dgx-04 state=drain reason=maint")
    s.run("sbatch -w dgx-04 --wrap=hostname")
    assert "ReqNodeNotAvail, UnavailableNodes:dgx-04" in s.run("squeue -o '%r'")


# ---------------------------------------------------------------- slurmd registration / hooks


def test_gres_mismatch_drains_node_with_inval():
    s = sh("dgx-03")
    write(s, "/etc/slurm/gres.conf", "Name=gpu Type=h100 File=/dev/nvidia[0-3]\n", host="dgx-03")
    s.run("systemctl restart slurmd")
    sn = s.ctx.state.slurm.nodes["dgx-03"]
    assert sn.state == "INVAL"
    assert sn.reason == "gres/gpu count reported lower than configured (4 < 8)"
    assert "4 GPU" in s.run("journalctl -u slurmd") or "Count=4" in s.run("journalctl -u slurmd")
    s.run("exit")
    s.run("scontrol update nodename=dgx-03 state=resume")  # slurmd still reports 4
    assert sn.state == "INVAL"
    write(s, "/etc/slurm/gres.conf", "Name=gpu Type=h100 File=/dev/nvidia[0-7]\n", host="dgx-03")
    s.run("ssh dgx-03 systemctl restart slurmd")
    assert sn.state == "DRAIN"  # registration valid, still drained
    s.run("scontrol update nodename=dgx-03 state=resume")
    assert sn.state == "IDLE"


def test_bad_device_file_makes_slurmd_fail():
    s = sh("dgx-01")
    write(s, "/etc/slurm/gres.conf", "Name=gpu File=/dev/nvidia[0-15]\n", host="dgx-01")
    assert "failed" in s.run("systemctl restart slurmd")
    assert "can't stat gres.conf file /dev/nvidia8" in s.run("journalctl -u slurmd")


def test_reconfigure_rereads_slurm_conf():
    s = sh()
    s.run("sinfo")  # controller has loaded its running config
    conf = s.ctx.state.nodes["bcm-head"].files["/etc/slurm/slurm.conf"]
    write(s, "/etc/slurm/slurm.conf", conf + "PartitionName=debug Nodes=dgx-01 MaxTime=30 State=UP\n")
    assert "debug" not in s.run("sinfo")
    s.run("scontrol reconfigure")
    assert "debug" in s.run("sinfo")


def test_unexpected_reboot_marks_node_down():
    s = sh("dgx-01")
    s.run("reboot")
    sn = s.ctx.state.slurm.nodes["dgx-01"]
    assert sn.state == "DOWN" and "unexpectedly rebooted" in sn.reason


def test_scontrol_reboot_nextstate_resume():
    s = sh()
    s.run("scontrol update nodename=dgx-01 state=drain reason=x")
    s.run("scontrol reboot nextstate=RESUME dgx-01")
    assert s.ctx.state.slurm.nodes["dgx-01"].state == "IDLE"


def test_controller_down():
    s = sh()
    s.run("systemctl stop slurmctld")
    assert "Unable to contact slurm controller" in s.run("sinfo")
    s.run("systemctl start slurmctld")
    assert "idle" in s.run("sinfo")


# ---------------------------------------------------------------- pyxis


def test_pyxis_image_syntax_and_plugstack():
    s = sh()
    bad = s.run("srun --gres=gpu:1 --container-image=nvcr.io/nvidia/pytorch:24.05-py3 true")
    assert "401 Unauthorized" in bad and "registry-1.docker.io" in bad
    ok = s.run(
        "srun --gres=gpu:1 --container-image=nvcr.io#nvidia/pytorch:24.05-py3 "
        'python -c "import torch; print(torch.cuda.device_count())"'
    )
    assert ok.splitlines()[-1] == "1"
    write(s, "/etc/slurm/plugstack.conf", "# empty\n")
    assert "unrecognized option '--container-image" in s.run(
        "srun --container-image=nvcr.io#nvidia/pytorch:24.05-py3 hostname"
    )


# ---------------------------------------------------------------- accounting


def test_qos_limit_needs_enforcement():
    s = sh()
    s.run("sacctmgr -i add qos gpu8")
    s.run("sacctmgr -i modify qos gpu8 set MaxTRESPerUser=gres/gpu=8")
    assert "gres/gpu=8" in s.run("sacctmgr show qos gpu8 format=Name,MaxTRESPU")
    s.run("sacctmgr -i modify user root set qos+=gpu8")
    s.run("sbatch --qos=gpu8 -N 2 --gres=gpu:8 --wrap='sleep 600'")
    assert s.ctx.state.slurm.jobs[-1].state == "RUNNING"  # not enforced yet
    s.run("scancel 1001")
    s.run("echo AccountingStorageEnforce=associations,limits,qos >> /etc/slurm/slurm.conf")
    s.run("scontrol reconfigure")
    s.run("sbatch --qos=gpu8 -N 2 --gres=gpu:8 --wrap='sleep 600'")
    assert s.ctx.state.slurm.jobs[-1].reason == "QOSMaxGRESPerUser"


def test_sacct_shows_steps():
    s = sh()
    s.run("srun --gres=gpu:1 hostname")
    out = s.run("sacct")
    assert "1001.0" in out and "COMPLETED" in out
    assert "1001.0" not in s.run("sacct -X")


# ---------------------------------------------------------------- dcgmi


def test_dcgmi_requires_hostengine():
    s = sh()
    assert "Unable to connect to host engine" in s.run("dcgmi discovery -l")  # head has no nv-hostengine
    s.run("ssh dgx-01")
    assert "8 GPUs found." in s.run("dcgmi discovery -l")
    s.run("systemctl stop nvidia-dcgm")
    assert "Unable to connect to host engine" in s.run("dcgmi diag -r 1")


def test_dcgmi_group_health_and_diag():
    s = sh("dgx-01")
    st = s.ctx.state
    st.nodes["dgx-01"].gpus[4].ecc_volatile_dbe = 3
    assert "group ID of 2" in s.run("dcgmi group -c all --default")
    assert "Healthy" in s.run("dcgmi health -g 2 -c")  # nothing watched yet
    s.run("dcgmi health -g 2 -s a")
    rep = s.run("dcgmi health -g 2 -c")
    assert "Failure" in rep and "GPU ID: 4" in rep
    assert "Pass" in s.run("dcgmi diag -r 1")  # level 1 doesn't test memory
    out = s.run("dcgmi diag -r 2")
    assert "Fail - GPU: 4" in out
    assert st.dcgm.diag_results["dgx-01"]["result"] == "Fail"
    assert st.dcgm.diag_results["dgx-01"]["failed_gpus"] == [4]


def test_dcgmi_diag_fallen_gpu_then_reboot_clears():
    s = sh("dgx-02")
    st = s.ctx.state
    st.nodes["dgx-02"].gpus[3].fallen_off_bus = True
    s.run("dcgmi group -c g --default")
    assert "Fail" in s.run("dcgmi diag -r 2")
    s.run("reboot")
    assert "dgx-02" not in st.dcgm.diag_results  # results are stale after a reboot
    assert st.dcgm.groups.get("dgx-02", {}) == {}  # host-engine groups are not persistent
    s.run("ssh dgx-02")
    s.run("dcgmi diag -r 2")
    assert st.dcgm.diag_results["dgx-02"]["result"] == "Pass"


def test_dcgmi_dmon_fields():
    s = sh("dgx-01")
    out = s.run("dcgmi dmon -e 150,155,203 -c 1")
    assert out.splitlines()[0].split() == ["#Entity", "GPUTMP", "POWER", "GPUTL"]
    assert len(out.splitlines()) == 2 + 8
    assert "not a valid field" in s.run("dcgmi dmon -e 99999")
