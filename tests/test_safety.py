"""Safe-practice grading: the audit log, command ordering and tagged results."""

from conftest import ScriptedUI, concrete
from greenray.app import lab
from greenray.core import audit
from greenray.core.shell import Shell
from greenray.core.state import make_cluster


def shell_for(sid):
    return Shell(concrete(sid).build_state())


def test_reboot_and_power_off_with_running_jobs_are_recorded():
    sh = shell_for("tsh-ecc-row-remap")  # alice's job 3101 runs on dgx-04
    sh.run("ssh dgx-04")
    sh.run("reboot")
    assert [e["kind"] for e in audit.events(sh.ctx.state)] == [audit.REBOOT_WITH_JOBS]
    sh = shell_for("tsh-ecc-row-remap")
    sh.run('cmsh -c "device; power off -n dgx-04"')
    assert audit.events(sh.ctx.state, audit.POWER_OFF_WITH_JOBS)
    sh = Shell(make_cluster())  # idle node: nothing to record
    sh.run("ssh dgx-01")
    sh.run("reboot")
    assert not audit.events(sh.ctx.state)


def test_node_down_and_cancel_are_recorded_but_drain_and_requeue_are_not():
    sh = shell_for("tsh-ecc-row-remap")
    sh.run("scontrol update nodename=dgx-04 state=drain reason=maintenance")
    sh.run("scontrol requeue 3101")
    assert not audit.events(sh.ctx.state)
    sh = shell_for("tsh-ecc-row-remap")
    sh.run("scancel 3101")
    assert audit.events(sh.ctx.state, audit.CANCEL_USER_JOB)[0]["detail"].startswith("3101")
    sh = shell_for("tsh-ecc-row-remap")
    sh.run("scontrol update nodename=dgx-04 state=down reason=broken")
    assert audit.events(sh.ctx.state, audit.NODE_DOWN_WITH_JOBS)


def test_deleting_a_running_pod_is_recorded():
    sh = shell_for("wkl-k8s-pod-pending-gpu")
    name = next(p.name for p in sh.ctx.state.k8s.pods if p.name.startswith("inference-") and p.status == "Running")
    sh.run(f"kubectl delete pod {name}")
    sh.run("kubectl delete pod bert-pretrain")  # Pending: not a loss of work
    assert [e["detail"] for e in audit.events(sh.ctx.state, audit.DELETE_RUNNING_POD)] == [f"default/{name}"]


def test_ran_before_needs_both_in_order():
    s = concrete("tsh-xid79-drained-node")
    sh = Shell(s.build_state())
    sh.run("ssh dgx-02")
    sh.run("reboot")
    sh.run("ssh dgx-02")
    sh.run("dmesg")
    order = {d: ok for d, ok, _ in s.grade(sh)}
    assert order["you read the kernel log on dgx-02 (the Xid tells you what failed)"]
    assert not order["you read the logs before rebooting or power-cycling the node"]


def test_results_are_tagged_by_kind():
    s = concrete("tsh-ecc-row-remap")
    assert set(s.kinds) == {"outcome", "evidence", "safety"}
    ui = ScriptedUI([])
    lab.show_results(ui, s.grade(Shell(s.build_state())), kinds=s.kinds)
    assert "[safe practice]" in ui.text and "[evidence]" in ui.text
