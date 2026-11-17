"""nvsm: health and versions reflect the node, a broken NVSwitch firmware set takes NVLinks down, a reboot applies an update."""

from greenray.commands.nvsm import NVSWITCH_FW, break_nvswitch_firmware
from greenray.core.shell import Shell
from greenray.core.state import make_cluster


def node_shell(host="dgx-03"):
    s = Shell(make_cluster())
    s.run(f"ssh {host}")
    return s


def test_only_on_dgx_nodes_and_needs_the_daemon():
    s = Shell(make_cluster())
    assert "command not found" in s.run("nvsm show health")
    s.run("ssh dgx-01")
    s.ctx.state.nodes["dgx-01"].services["nvsm"] = "inactive"
    assert "nvsm daemon" in s.run("nvsm show health")


def test_healthy_node_reports_healthy_and_versions():
    s = node_shell()
    out = s.run("nvsm show health")
    assert "Health Summary: Healthy" in out and "[FAIL]" not in out
    versions = s.run("nvsm show versions")
    assert versions.count(NVSWITCH_FW) == 4 and "550.90.07" in versions and "28.39.1002" in versions
    assert "No active alerts" in s.run("nvsm show alerts")


def test_inconsistent_nvswitch_firmware_breaks_nvlink_and_shows_in_health():
    s = node_shell()
    break_nvswitch_firmware(s.ctx.state.nodes["dgx-03"])
    out = s.run("nvsm show health")
    assert "Unhealthy" in out and "versions differ" in out and "72 of 144 links active" in out
    assert "<inactive>" in s.run("nvidia-smi nvlink -s -i 0")
    assert "2 active alert(s)" in s.run("nvsm show alerts")
    slow = s.run("all_reduce_perf -b 8G -e 8G -g 8")
    assert s.ctx.state.flags["nccl_last"]["busbw"] < 300, slow


def test_update_is_refused_under_a_job_unless_forced_and_applied_by_reboot():
    s = node_shell()
    state = s.ctx.state
    break_nvswitch_firmware(state.nodes["dgx-03"])
    s.run("sbatch --wrap 'sleep 1000' -w dgx-03 --gpus=8")
    out = s.run("nvsm update firmware --type nvswitch && echo OK")
    assert "refusing" in out and "OK" not in out
    assert "--type must be" in s.run("nvsm update firmware --type bogus")
    s.run("nvsm update firmware --type nvswitch --force")
    assert state.audit and state.audit[-1]["kind"] == "firmware_update_with_jobs"
    assert "pending" in s.run("nvsm show versions")
    s.run("reboot")
    node = state.nodes["dgx-03"]
    assert node.firmware["nvswitch"] == [NVSWITCH_FW] * 4 and not node.firmware_pending
    assert not any(g.nvlink_down for g in node.gpus)
    s.run("ssh dgx-03")
    assert "Healthy" in s.run("nvsm show health")
    s.run("all_reduce_perf -b 8G -e 8G -g 8")
    assert state.flags["nccl_last"]["busbw"] > 450


def test_dump_health_writes_a_bundle():
    s = node_shell()
    out = s.run("nvsm dump health")
    assert "/tmp/nvsm-health-dgx-03" in out
    assert "nvsm-health-dgx-03" in s.run("ls /tmp")
