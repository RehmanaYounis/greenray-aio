"""NVIDIA Run:ai emulation: departments, projects, quotas, the Run:ai scheduler and the CLI.

Everything lives in ``state.runai`` (created by ``install``) so scenarios can grade it:

  runai.departments[name]   {name, gpu_quota, over_quota}
  runai.projects[name]      {name, department, gpu_quota (deserved), over_quota (bool),
                             max_gpus (None = no limit), namespace ("runai-<name>")}
  runai.access_rules        [{user, role, project}]
  runai.workloads           [{name, project, type, image, gpu (float), preemptible, status,
                              node, alloc [[gpu_index, fraction], ...], user, created, events}]
  runai.default_project     what `runai config project` / `runai project set` stored

Model (simplified, but the rules are the ones the exam asks about):
  * A workload is *in quota* if its project (and department) stay within their deserved GPUs.
  * Non-preemptible workloads (interactive/workspace by default, inference always) may only run
    in quota. Preemptible ones (training by default, interactive with --preemptible) may go
    over quota on idle GPUs, if the project allows over-quota and stays under its max limit.
  * An in-quota request that does not fit reclaims GPUs by preempting over-quota preemptible
    workloads of other projects (newest first). Preempted workloads go back to Pending.
  * Fractions (-g 0.5, --gpu-portion-request 0.5, --gpu-memory 40G) share one physical GPU;
    a runai-reservation pod holds the whole GPU for the Kubernetes scheduler.

Real tool: ``runai`` CLI. Both the v1 syntax (runai submit / list jobs / config project) and the
common v2 syntax (runai project list|set, runai training|workspace|inference submit, runai workload list)
are accepted.

Lab helper (clearly NOT a real command): ``runai-ui`` stands in for the Run:ai web UI / REST API,
where admins create departments and projects, set quotas and assign users (access rules).
"""

from __future__ import annotations

import re

from ..core.errors import CommandError
from ..core.state import HEAD, AttrDict
from . import command
from . import kubectl as kc

POOL_LABEL = "node-role.kubernetes.io/runai-gpu-worker"
CLI_VERSION = "2.18.47"
ROLES = (
    "L1 researcher",
    "L2 researcher",
    "Research manager",
    "ML engineer",
    "Viewer",
    "Department administrator",
    "Editor",
    "System administrator",
)
GPU_MEM_GIB = 80
EPS = 1e-6


# ====================================================================== state / install


def rstate(state):
    r = state.get("runai")
    if not r or not r.get("installed"):
        raise CommandError(
            "Error: unable to connect to the Run:ai control plane: Run:ai is not installed on this cluster", 1
        )
    return r


def install(state, nodes=None, cluster="superpod"):
    """Scenario helper: install the Run:ai cluster components (assumes the GPU Operator is installed)."""
    kc.k8s_init(state)
    state["runai"] = {
        "installed": True,
        "cluster": cluster,
        "departments": {"default": {"name": "default", "gpu_quota": 0, "over_quota": True, "created": kc.now(state)}},
        "projects": {},
        "access_rules": [],
        "users": ["admin@run.ai"],
        "workloads": [],
        "default_project": "",
        "current_user": "admin@run.ai",
    }
    kc.ensure_namespace(state, "runai")
    kc.ensure_namespace(state, "runai-reservation")
    for name in ("runai-scheduler-default", "runai-agent", "cluster-sync", "researcher-service", "runai-operator"):
        if not any(d.name == name and d.namespace == "runai" for d in state.k8s.deployments):
            kc.add_deployment(
                state, "runai", name, f"gcr.io/run-ai-prod/{name}:{CLI_VERSION}", component="system", created=None
            )
    for n in nodes or [n for n, sn in state.nodes.items() if sn.gpus]:
        state.k8s.nodes[n].labels[POOL_LABEL] = "true"
    kc.reconcile(state)
    return state.runai


def create_department(state, name, gpu_quota, over_quota=True):
    r = rstate(state)
    r.departments[name] = {"name": name, "gpu_quota": gpu_quota, "over_quota": over_quota, "created": kc.now(state)}
    return r.departments[name]


def create_project(state, name, department="default", gpu_quota=0, over_quota=True, max_gpus=None):
    r = rstate(state)
    _check_project_name(name)
    if name in r.projects:
        raise CommandError(f'Error: project "{name}" already exists', 1)
    if department not in r.departments:
        raise CommandError(f'Error: department "{department}" not found', 1)
    _check_dept_sum(r, department, gpu_quota, exclude=None)
    ns = f"runai-{name}"
    r.projects[name] = {
        "name": name,
        "department": department,
        "gpu_quota": gpu_quota,
        "over_quota": over_quota,
        "max_gpus": max_gpus,
        "namespace": ns,
        "created": kc.now(state),
    }
    kc.ensure_namespace(state, ns)
    return r.projects[name]


def _check_project_name(name):
    if not re.fullmatch(r"[a-z]([-a-z0-9]{0,34}[a-z0-9])?", name or ""):
        raise CommandError(
            f'Error: invalid project name "{name}": must be a lowercase RFC 1123 label (a-z, 0-9, "-"), '
            "start with a letter and be at most 36 characters (the namespace runai-<name> must be valid)",
            1,
        )


def _check_dept_sum(r, department, quota, exclude):
    if department == "default":
        return
    dept = r.departments[department]
    total = sum(p.gpu_quota for p in r.projects.values() if p.department == department and p.name != exclude)
    if total + quota > dept.gpu_quota + EPS:
        raise CommandError(
            f"Error: the sum of project GPU quotas in department {department} ({_g(total + quota)}) would exceed "
            f"the department quota ({_g(dept.gpu_quota)}). Increase the department quota first.",
            1,
        )


def submit(
    state,
    name,
    project,
    image,
    gpu=1.0,
    wtype="Training",
    preemptible=None,
    user=None,
    command="",
    created=None,
):
    """Create a workload (used by the CLI and by scenario setup)."""
    r = rstate(state)
    if project not in r.projects:
        raise CommandError(
            f'Error: project "{project}" does not exist. Run "runai list projects" to see the available projects', 1
        )
    if any(w.name == name and w.project == project for w in r.workloads):
        raise CommandError(f'Error: a workload with the name "{name}" already exists in project {project}', 1)
    if not re.fullmatch(r"[a-z]([-a-z0-9]*[a-z0-9])?", name):
        raise CommandError(f'Error: invalid workload name "{name}": must be a lowercase RFC 1123 label', 1)
    if gpu < 0 or (gpu > 1 and abs(gpu - round(gpu)) > EPS):
        raise CommandError("Error: GPU fractions above 1 are not supported. Use a whole number of GPUs, or <1.", 1)
    if preemptible is None:
        preemptible = wtype == "Training"
    if wtype == "Inference":
        preemptible = False
    r.workloads.append(
        {
            "name": name,
            "project": project,
            "type": wtype,
            "image": image,
            "gpu": float(gpu),
            "preemptible": bool(preemptible),
            "user": user or r.current_user,
            "command": command,
            "created": kc.now(state) if created is None else created,
            "status": "Pending",
            "node": None,
            "alloc": [],
            "reason": "",
            "events": [],
            "preemptions": 0,
        }
    )
    w = r.workloads[-1]
    _wevent(state, w, "Normal", "Submitted", f"Workload {name} submitted to project {project}")
    return w


# ====================================================================== scheduler


def pool_nodes(state):
    k = state.k8s
    labelled = [n for n, kn in k.nodes.items() if kn.labels.get(POOL_LABEL) == "true"]
    return sorted(
        n
        for n in labelled
        if state.nodes[n].gpus and k.nodes[n].get("ready", True) and k.nodes[n].get("schedulable", True)
    )


def _is_runai_pod(p):
    return (p.get("component") or "").startswith("runai")


def _capacity(state):
    """{node: {gpu_index: free fraction}} before Run:ai allocations."""
    k = state.k8s
    out = {}
    for n in pool_nodes(state):
        sn = state.nodes[n]
        if int(kc.qty(k.nodes[n].allocatable.get("nvidia.com/gpu", 0))) == 0:
            continue  # device plugin not advertising GPUs on this node
        healthy = [g.index for g in sn.gpus if g.healthy and not g.reset_required and not g.fallen_off_bus]
        used = {
            d[0]
            for p in k.pods
            if p.node == n and not _is_runai_pod(p) and p.phase not in ("Succeeded", "Failed")
            for d in (p.get("devices") or [])
        }
        out[n] = {i: 1.0 for i in healthy if i not in used}
    return out


def _place(free, req):
    """-> (node, [[idx, frac], ...]) or None. Fractions bin-pack onto partly used GPUs first."""
    if req <= EPS:
        n = next(iter(free), None)
        return (n, []) if n else None
    if req < 1 - EPS:
        best = None
        for n in sorted(free):
            for idx, left in sorted(free[n].items()):
                if left + EPS >= req:
                    key = (left > 1 - EPS, left, n, idx)  # prefer partly used GPUs (bin-packing)
                    if best is None or key < best[0]:
                        best = (key, n, idx)
        if best is None:
            return None
        return best[1], [[best[2], req]]
    need = int(round(req))
    for n in sorted(free, key=lambda n: (-sum(1 for v in free[n].values() if v > 1 - EPS), n)):
        whole = [i for i, v in sorted(free[n].items()) if v > 1 - EPS]
        if len(whole) >= need:
            return n, [[i, 1.0] for i in whole[:need]]
    return None


def _alloc(r, project=None, department=None):
    tot = 0.0
    for w in r.workloads:
        if w.status != "Running":
            continue
        p = r.projects.get(w.project)
        if project and w.project != project:
            continue
        if department and (p is None or p.department != department):
            continue
        tot += w.gpu
    return tot


def _wevent(state, w, etype, reason, msg):
    ev = w.setdefault("events", [])
    if ev and ev[-1][2] == reason and ev[-1][3] == msg:
        return
    ev.append([kc.now(state), etype, reason, msg])


def _release(w):
    w.status, w.node, w.alloc = "Pending", None, []


def runai_sync(state):
    r = state.get("runai")
    if not r or not r.get("installed"):
        return
    free = _capacity(state)
    running = [w for w in r.workloads if w.status == "Running"]
    for w in running:
        if w.node not in free or any(free[w.node].get(i, -1) + EPS < f for i, f in w.alloc):
            _release(w)  # node or GPU went away
            continue
        for i, f in w.alloc:
            free[w.node][i] -= f
    pending = sorted((w for w in r.workloads if w.status == "Pending"), key=lambda w: (w.created, w.name))
    for w in pending:
        _schedule_one(state, r, w, free)
    _mirror_pods(state, r)


def _schedule_one(state, r, w, free):
    proj = r.projects.get(w.project)
    if proj is None:
        w.reason = f"project {w.project} no longer exists"
        return
    dept = r.departments.get(proj.department) or AttrDict({"gpu_quota": 0, "name": proj.department})
    req = w.gpu
    p_alloc = _alloc(r, project=w.project)
    d_alloc = _alloc(r, department=proj.department)
    in_quota = p_alloc + req <= proj.gpu_quota + EPS and (
        proj.department == "default" or d_alloc + req <= dept.gpu_quota + EPS
    )
    if not in_quota:
        if not w.preemptible:
            w.reason = (
                f"Non-preemptible workload is over quota. Workload requested {_g(req)} GPU(s), project {w.project} "
                f"is using {_g(p_alloc)} of its {_g(proj.gpu_quota)} deserved GPU(s). Non-preemptible workloads "
                "can only use the project's deserved quota."
            )
            _wevent(state, w, "Warning", "Unschedulable", w.reason)
            return
        if not proj.over_quota:
            w.reason = (
                f"Workload requested {_g(req)} GPU(s); project {w.project} is using {_g(p_alloc)} of its "
                f"{_g(proj.gpu_quota)} deserved GPU(s) and over-quota is disabled for this project."
            )
            _wevent(state, w, "Warning", "Unschedulable", w.reason)
            return
    if proj.get("max_gpus") is not None and p_alloc + req > proj.max_gpus + EPS:
        w.reason = (
            f"Workload requested {_g(req)} GPU(s); project {w.project} would use {_g(p_alloc + req)} GPU(s), "
            f"above its max GPU limit of {_g(proj.max_gpus)}."
        )
        _wevent(state, w, "Warning", "Unschedulable", w.reason)
        return
    spot = _place(free, req)
    if spot is None and in_quota:
        # find the smallest set of over-quota victims that makes room, then preempt only those
        cands = _reclaim_candidates(r, w)
        chosen = []
        for n in sorted(free):
            trial = {n: dict(free[n])}
            pick = []
            for v in (v for v in cands if v.node == n):
                for i, f in v.alloc:
                    trial[n][i] += f
                pick.append(v)
                if _place(trial, req) is not None:
                    if not chosen or len(pick) < len(chosen):
                        chosen = list(pick)
                    break
        for v in chosen:
            for i, f in v.alloc:
                free[v.node][i] += f
            msg = (
                f"Workload preempted by the Run:ai scheduler to reclaim deserved quota for project {w.project} "
                f"(workload {w.name}). Project {v.project} was over its quota."
            )
            _release(v)
            v.preemptions = v.get("preemptions", 0) + 1
            _wevent(state, v, "Warning", "Preempted", msg)
        if chosen:
            spot = _place(free, req)
    if spot is None:
        nodes = len(free)
        w.reason = (
            f"Unable to schedule workload: 0/{nodes} nodes in node pool default have {_g(req)} free GPU(s)"
            + (" (fractional workloads need one GPU with enough free memory)" if req < 1 else "")
            + (
                ". The request is over quota, so it can only use idle GPUs"
                if not in_quota
                else ". No over-quota workloads left to preempt"
            )
            + "."
        )
        _wevent(state, w, "Warning", "Unschedulable", w.reason)
        return
    n, alloc = spot
    for i, f in alloc:
        free[n][i] -= f
    w.status, w.node, w.alloc, w.reason = "Running", n, alloc, ""
    w["started"] = kc.now(state)
    oq = "" if in_quota else " (over quota)"
    _wevent(
        state,
        w,
        "Normal",
        "Scheduled",
        f"Successfully assigned {proj.namespace}/{w.name}-0-0 to {n}, GPU(s) "
        + ",".join(f"{i}" + (f" ({_g(f)})" if f < 1 else "") for i, f in alloc)
        + oq,
    )


def _reclaim_candidates(r, w):
    """Over-quota preemptible workloads of other projects, newest first, only while that project stays over quota."""
    out = []
    over = {}
    for p in r.projects.values():
        over[p.name] = _alloc(r, project=p.name) - p.gpu_quota
    for v in sorted(
        (v for v in r.workloads if v.status == "Running" and v.preemptible and v.project != w.project),
        key=lambda v: (-v.created, v.name),
    ):
        if over.get(v.project, 0) > EPS:
            out.append(v)
            over[v.project] -= v.gpu
    return out


def _mirror_pods(state, r):
    """Keep Kubernetes pods in sync with workloads: <name>-0-0 in runai-<project>, plus reservation pods."""
    k = state.k8s
    want = {}
    shared = {}
    for w in r.workloads:
        proj = r.projects.get(w.project)
        if proj is None:
            continue
        want[(proj.namespace, f"{w.name}-0-0")] = w
        if w.status == "Running":
            for i, f in w.alloc:
                if f < 1 - EPS:
                    shared.setdefault((w.node, i), []).append(w)
    for p in list(k.pods):
        stale_workload = p.get("component") == "runai-workload" and (p.namespace, p.name) not in want
        stale_reservation = p.get("component") == "runai-reservation" and (p.node, p.get("gpu_index")) not in shared
        if stale_workload or stale_reservation:
            k.pods.remove(p)
    for (ns, pname), w in want.items():
        pod = next((p for p in k.pods if p.namespace == ns and p.name == pname), None)
        whole = int(round(w.gpu)) if w.gpu >= 1 - EPS else 0
        if pod is None:
            limits = {"nvidia.com/gpu": whole} if whole else {}
            spec = {
                "schedulerName": "runai-scheduler",
                "restartPolicy": "Never" if w.type == "Training" else "Always",
                "containers": [
                    {
                        "name": w.name,
                        "image": w.image,
                        **({"resources": {"limits": limits}} if limits else {}),
                    }
                ],
            }
            pod = kc.new_pod(
                state,
                ns,
                pname,
                spec,
                labels={"app": "runai", "workloadName": w.name, "project": w.project, "user": w.user.split("@")[0]},
                owner={"kind": "RunaiWorkload", "name": w.name},
                component="runai-workload",
                created=w.created,
            )
        pod["gpu_fraction"] = w.gpu if 0 < w.gpu < 1 - EPS else None
        if w.status == "Running":
            pod.node = w.node
            pod.devices = [[i, None] for i, _f in w.alloc]
            kc.set_status(state, pod, "Running", logs=_logs(w))
        else:
            pod.node, pod.devices, pod.started, pod.ip = None, [], None, None
            kc.set_status(state, pod, "Pending", logs=[])
            if w.reason:
                pod.sched_msg = w.reason
                kc.event(state, "Pod", pod, "Warning", "Unschedulable", w.reason, source="runai-scheduler")
    for (n, i), ws in shared.items():
        pname = f"runai-reservation-gpu-{n}-{kc.sfx(n, i)}"
        pod = next((p for p in k.pods if p.get("component") == "runai-reservation" and p.name == pname), None)
        if pod is None:
            pod = kc.new_pod(
                state,
                "runai-reservation",
                pname,
                {
                    "schedulerName": "runai-scheduler",
                    "containers": [
                        {
                            "name": "runai-reservation",
                            "image": f"gcr.io/run-ai-prod/runai-reservation:{CLI_VERSION}",
                            "resources": {"limits": {"nvidia.com/gpu": 1}},
                        }
                    ],
                },
                labels={"app": "runai-reservation"},
                owner={"kind": "RunaiReservation", "name": pname},
                component="runai-reservation",
                node=n,
            )
            pod["gpu_index"] = i
        pod.devices = [[i, None]]
        kc.set_status(state, pod, "Running", logs=[f"reserving GPU {i} for {', '.join(w.name for w in ws)}"])


def _logs(w):
    img = w.image
    if "jupyter" in img or w.type == "Interactive":
        return [
            "[I 2026-10-19 09:41:02.118 ServerApp] Jupyter Server 2.14.1 is running at:",
            f"[I 2026-10-19 09:41:02.118 ServerApp] http://{w.name}-0-0:8888/lab",
            "[I 2026-10-19 09:41:02.119 ServerApp] Use Control-C to stop this server.",
        ]
    if "triton" in img or w.type == "Inference":
        return [
            "I1019 09:41:01.000000 1 server.cc:676] +-------+---------+--------+",
            "I1019 09:41:01.000000 1 grpc_server.cc:2463] Started GRPCInferenceService at 0.0.0.0:8001",
            "I1019 09:41:01.000000 1 http_server.cc:4692] Started HTTPService at 0.0.0.0:8000",
        ]
    gpus = max(1, int(round(w.gpu)))
    return [
        f"[rank0]: world_size={gpus} backend=nccl",
        "epoch 0 step 100/20000 loss 6.731 throughput 2110 samples/s",
    ]


# ====================================================================== formatting helpers


def _g(v):
    v = float(v)
    return str(int(v)) if abs(v - round(v)) < EPS else f"{v:.2f}".rstrip("0").rstrip(".")


def _g2(v):
    return f"{float(v):.2f}"


def _age(state, t):
    return kc.human(kc.now(state) - (t or 0))


# ====================================================================== kubectl kinds (read-only CRs)


class _ProjectKind:
    names = ("project", "projects", "project.run.ai", "projects.run.ai")
    api_version = "run.ai/v2"
    kind = "Project"
    namespaced = False

    @staticmethod
    def list(state, opts):
        r = state.get("runai") or {}
        return [AttrDict(dict(p)) for _, p in sorted((r.get("projects") or {}).items())]

    @staticmethod
    def table(state, items):
        return ["NAME", "DEPARTMENT", "DESERVED GPUS", "NAMESPACE", "AGE"], [
            [p.name, p.department, _g(p.gpu_quota), p.namespace, _age(state, p.created)] for p in items
        ]

    @staticmethod
    def api(state, p):
        spec = {"deservedGpus": p.gpu_quota, "department": p.department, "namespace": p.namespace}
        if p.get("max_gpus") is not None:
            spec["maxAllowedGpus"] = p.max_gpus
        spec["gpuOverQuotaWeight"] = 1 if p.over_quota else 0
        return {
            "apiVersion": "run.ai/v2",
            "kind": "Project",
            "metadata": {"name": p.name, "labels": {"runai/department": p.department}},
            "spec": spec,
        }

    @staticmethod
    def apply(state, doc, src):
        name = (doc.get("metadata") or {}).get("name")
        raise CommandError(
            f'Error from server (Forbidden): error when applying "{src}": admission webhook '
            f'"project-validator.run.ai" denied the request: project "{name}" is managed by the Run:ai control '
            "plane. Create and edit projects in the Run:ai UI or REST API (in this lab: runai-ui project create).",
            1,
        )

    @staticmethod
    def delete(state, name):
        raise CommandError(
            f'Error from server (Forbidden): admission webhook "project-validator.run.ai" denied the request: '
            f'project "{name}" is managed by the Run:ai control plane (in this lab: runai-ui project delete).',
            1,
        )


class _DepartmentKind(_ProjectKind):
    names = ("department", "departments", "department.run.ai", "departments.run.ai")
    api_version = "run.ai/v1"
    kind = "Department"

    @staticmethod
    def list(state, opts):
        r = state.get("runai") or {}
        return [AttrDict(dict(d)) for _, d in sorted((r.get("departments") or {}).items())]

    @staticmethod
    def table(state, items):
        return ["NAME", "DESERVED GPUS", "AGE"], [[d.name, _g(d.gpu_quota), _age(state, d.created)] for d in items]

    @staticmethod
    def api(state, d):
        return {
            "apiVersion": "run.ai/v1",
            "kind": "Department",
            "metadata": {"name": d.name},
            "spec": {"deservedGpus": d.gpu_quota},
        }


kc.register_kind("projects", _ProjectKind)
kc.register_kind("departments", _DepartmentKind)
if runai_sync not in kc.RECONCILE_HOOKS:
    kc.RECONCILE_HOOKS.append(runai_sync)


# ====================================================================== CLI


def _opts(args, value_flags, aliases=None):
    aliases = aliases or {}
    pos, opts, cmd = [], {}, []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            cmd = args[i + 1 :]
            break
        if a.startswith("--") and "=" in a:
            k_, v = a[2:].split("=", 1)
            opts[aliases.get("--" + k_, k_)] = v
        elif a in value_flags:
            key = aliases.get(a, a.lstrip("-"))
            if i + 1 >= len(args):
                raise CommandError(f"Error: flag needs an argument: {a}", 1)
            opts[key] = args[i + 1]
            i += 1
        elif a.startswith("-") and len(a) > 1:
            opts[aliases.get(a, a.lstrip("-"))] = True
        else:
            pos.append(a)
        i += 1
    return pos, opts, cmd


def _project_of(ctx, r, opts):
    p = opts.get("project") or r.default_project
    if not p:
        raise CommandError(
            "Error: no project specified. Use -p <project> or set a default with `runai config project <name>`", 1
        )
    if p not in r.projects:
        raise CommandError(f'Error: project "{p}" does not exist', 1)
    return p


def _gpu_request(opts):
    if opts.get("gpu-memory") or opts.get("gpu-memory-request"):
        raw = str(opts.get("gpu-memory") or opts.get("gpu-memory-request"))
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([GM])i?B?", raw, re.I)
        if not m:
            raise CommandError(f'Error: invalid GPU memory "{raw}" (use e.g. 20G or 512M)', 1)
        gib = float(m.group(1)) / (1024 if m.group(2).upper() == "M" else 1)
        if gib > GPU_MEM_GIB:
            raise CommandError(f"Error: requested GPU memory {raw} is larger than a GPU ({GPU_MEM_GIB}G)", 1)
        return round(gib / GPU_MEM_GIB, 4)
    raw = opts.get("gpu") or opts.get("gpu-portion-request") or opts.get("gpu-devices-request")
    if raw in (None, True):
        return 0.0
    try:
        v = float(raw)
    except ValueError:
        raise CommandError(f'Error: invalid argument "{raw}" for "-g, --gpu" flag: must be a number', 1) from None
    if opts.get("gpu-portion-request") and not 0 < v <= 1:
        raise CommandError("Error: --gpu-portion-request must be between 0 and 1", 1)
    return v


SUBMIT_FLAGS = {
    "-i",
    "--image",
    "-g",
    "--gpu",
    "--gpu-memory",
    "--gpu-portion-request",
    "--gpu-devices-request",
    "--gpu-memory-request",
    "-p",
    "--project",
    "--name",
    "--cpu",
    "--memory",
    "-e",
    "--environment",
    "--port",
    "--service-type",
    "--node-type",
    "--node-pools",
    "--preemptibility",
    "--working-dir",
    "--volume",
    "-v",
}
ALIASES = {"-i": "image", "-g": "gpu", "-p": "project", "-e": "environment", "-A": "all-projects", "-v": "volume"}


def _submit(ctx, r, args, wtype, v2=False):
    pos, opts, cmd = _opts(args, SUBMIT_FLAGS | {"-n"}, ALIASES)
    name = opts.get("name") or (pos[0] if pos else None)
    if not name:
        raise CommandError("Error: a workload name is required (runai submit <name> -i <image> ...)", 1)
    image = opts.get("image")
    if not image or image is True:
        raise CommandError('Error: required flag(s) "image" not set', 1)
    if opts.get("interactive") or opts.get("jupyter"):
        wtype = "Interactive"
    project = _project_of(ctx, r, opts)
    gpu = _gpu_request(opts)
    pre = None
    if opts.get("preemptible") is True:
        pre = True
    if opts.get("non-preemptible") is True:
        pre = False
    if isinstance(opts.get("preemptibility"), str):
        pre = opts["preemptibility"].lower() == "preemptible"
    if wtype == "Interactive" and pre is None:
        pre = False
    if wtype == "Inference" and pre:
        raise CommandError("Error: inference workloads are always non-preemptible", 1)
    command_line = " ".join(cmd)  # v1: `--command -- CMD ARGS` overrides the entrypoint; plain `-- ARGS` passes args
    submit(ctx.state, name, project, image, gpu, wtype, pre, command=command_line)
    kc.advance(ctx.state, 20)
    kc.reconcile(ctx.state)
    noun = {"Training": "training", "Interactive": "workspace", "Inference": "inference"}[wtype]
    if v2:
        return f'Creating {noun} workload "{name}" in project "{project}"\nThe {noun} "{name}" has been submitted successfully\n'
    return (
        f"The job '{name}' has been submitted successfully\n"
        f"You can run `runai describe job {name} -p {project}` to check the job status\n"
    )


def _find(r, name, project):
    w = next((w for w in r.workloads if w.name == name and w.project == project), None)
    if w is None:
        others = [w.project for w in r.workloads if w.name == name]
        hint = f" (found in project {', '.join(others)}; use -p)" if others else ""
        raise CommandError(f'Error: workload "{name}" not found in project {project}{hint}', 1)
    return w


def _list_jobs(ctx, r, opts, v2=False):
    state = ctx.state
    if opts.get("all-projects"):
        ws = list(r.workloads)
        head = "Showing jobs for all projects"
    else:
        p = _project_of(ctx, r, opts)
        ws = [w for w in r.workloads if w.project == p]
        head = f"Showing jobs for project {p}"
    if v2:
        rows = [("Workload", "Type", "Status", "Project", "Preemptible", "Running/Requested Pods", "GPU Allocation")]
        for w in ws:
            rows.append(
                (
                    w.name,
                    {"Interactive": "Workspace"}.get(w.type, w.type),
                    w.status,
                    w.project,
                    "Yes" if w.preemptible else "No",
                    f"{1 if w.status == 'Running' else 0}/1",
                    f"{_g2(w.gpu if w.status == 'Running' else 0)}/{_g2(w.gpu)}",
                )
            )
        return kc.table(rows, gap=2)
    rows = [
        (
            "NAME",
            "STATUS",
            "AGE",
            "NODE",
            "IMAGE",
            "TYPE",
            "PROJECT",
            "USER",
            "GPUs Allocated (Requested)",
            "PODs Running (Pending)",
            "SERVICE URL(S)",
        )
    ]
    for w in ws:
        rows.append(
            (
                w.name,
                w.status,
                _age(state, w.created),
                w.node or "",
                w.image,
                w.type,
                w.project,
                w.user.split("@")[0],
                f"{_g2(w.gpu if w.status == 'Running' else 0)} ({_g2(w.gpu)})",
                f"{1 if w.status == 'Running' else 0} ({0 if w.status == 'Running' else 1})",
                "",
            )
        )
    return head + "\n\n" + kc.table(rows, gap=2)


def _describe(ctx, r, w):
    state = ctx.state
    proj = r.projects[w.project]
    run = w.status == "Running"
    lines = [
        f"Name: {w.name}",
        f"Namespace: {proj.namespace}",
        f"Type: {w.type}",
        f"Status: {w.status}",
        f"Duration: {_age(state, w.get('started') or kc.now(state)) if run else '0s'}",
        f"GPUs: {_g2(w.gpu)}",
        f"Total Requested GPUs: {_g2(w.gpu)}",
        f"Allocated GPUs: {_g2(w.gpu if run else 0)}",
        f"Allocated GPUs memory: {int(w.gpu * GPU_MEM_GIB * 1024) if run else 0}M",
        f"Preemptible: {'true' if w.preemptible else 'false'}",
        f"Running PODs: {1 if run else 0}",
        f"Pending PODs: {0 if run else 1}",
        "Parallelism: 1",
        "Completions: 1",
        "Succeeded PODs: 0",
        "Failed PODs: 0",
        "Is Distributed Workload: false",
        "Service URLs:",
        f"Command: {w.get('command') or ''}",
        "",
        "Pods:",
    ]
    lines.append(
        kc.table(
            [
                ("POD", "STATUS", "TYPE", "AGE", "NODE", "GPU(s)"),
                (
                    f"{w.name}-0-0",
                    w.status.upper(),
                    w.type.upper(),
                    _age(state, w.created),
                    w.node or "",
                    ",".join(str(i) for i, _ in w.alloc) if run else "",
                ),
            ],
            gap=2,
        ).rstrip("\n")
    )
    lines += ["", "Events:"]
    ev = [("  TIME", "TYPE", "REASON", "MESSAGE")]
    for t, et, reason, msg in w.events[-8:]:
        ev.append((f"  {_age(state, t)} ago", et, reason, msg))
    lines.append(kc.table(ev, gap=2).rstrip("\n"))
    return "\n".join(lines) + "\n"


def _list_projects(ctx, r):
    rows = [("PROJECT", "DEPARTMENT", "DESERVED GPUs", "ALLOCATED GPUs", "OVER QUOTA", "MAX GPUs", "NAMESPACE")]
    for name, p in sorted(r.projects.items()):
        rows.append(
            (
                name + (" (default)" if name == r.default_project else ""),
                p.department,
                _g(p.gpu_quota),
                _g(_alloc(r, project=name)),
                "enabled" if p.over_quota else "disabled",
                "-" if p.get("max_gpus") is None else _g(p.max_gpus),
                p.namespace,
            )
        )
    return kc.table(rows, gap=2)


def _list_nodes(ctx, r):
    state = ctx.state
    cap = _capacity(state)
    rows = [("NODE", "STATUS", "GPUs", "ALLOCATED GPUs", "FREE GPUs", "SHARED GPUs (fractions)")]
    for n in pool_nodes(state):
        total = len(cap.get(n, {}))
        held = {}
        for w in r.workloads:
            if w.status == "Running" and w.node == n:
                for i, f in w.alloc:
                    held[i] = held.get(i, 0) + f
        frac_gpus = {i for w in r.workloads if w.status == "Running" and w.node == n for i, f in w.alloc if f < 1 - EPS}
        shared = [f"gpu{i}={_g(held[i])}" for i in sorted(frac_gpus)]
        rows.append(
            (
                n,
                "Ready" if state.k8s.nodes[n].get("ready", True) else "NotReady",
                str(total),
                _g(sum(held.values())),
                str(total - len(held)),
                " ".join(shared) or "-",
            )
        )
    return kc.table(rows, gap=2)


def _delete(ctx, r, args):
    pos, opts, _ = _opts(args, {"-p", "--project"}, ALIASES)
    project = _project_of(ctx, r, opts)
    names = pos
    if opts.get("all"):
        names = [w.name for w in r.workloads if w.project == project]
    if not names:
        raise CommandError("Error: specify the workload name(s) to delete", 1)
    out = []
    for nm in names:
        w = _find(r, nm, project)
        r.workloads.remove(w)
        out.append(f"The job '{nm}' has been deleted from project {project}")
    kc.reconcile(ctx.state)
    return "\n".join(out) + "\n"


def _exec(ctx, r, args):
    pos, opts, cmd = _opts(args, {"-p", "--project"}, ALIASES)
    if not pos:
        raise CommandError("Error: specify the workload name", 1)
    w = _find(r, pos[0], _project_of(ctx, r, opts))
    if w.status != "Running":
        raise CommandError(f"Error: workload {w.name} is {w.status}; it has no running pod to exec into", 1)
    proj = r.projects[w.project]
    pod = next(p for p in ctx.state.k8s.pods if p.namespace == proj.namespace and p.name == f"{w.name}-0-0")
    return kc.cmd_exec(ctx, [pod.name], {"namespace": proj.namespace}, cmd or pos[1:] or ["bash"])


@command(
    "runai",
    help="Run:ai CLI: runai submit|list jobs|describe job|delete job|list projects|config project "
    "(v2: runai project list|set, runai training|workspace|inference submit, runai workload list|describe)",
)
def runai(ctx, args):
    state = ctx.state
    if not args or args[0] in ("-h", "--help", "help"):
        return (
            "Run:ai command-line interface (lab emulation; accepts v1 and v2 syntax)\n\n"
            "v1:  runai submit NAME -i IMAGE [-g 1|0.5] [--gpu-memory 20G] [-p PROJECT] [--interactive [--preemptible]]\n"
            "     runai list jobs [-p P | -A] | runai list projects | runai list nodes\n"
            "     runai describe job NAME [-p P] | runai logs NAME | runai delete job NAME [-p P]\n"
            "     runai config project NAME | runai exec NAME -- CMD\n"
            "v2:  runai project list | runai project set NAME\n"
            "     runai training|workspace|inference submit NAME -i IMAGE [--gpu-devices-request N | "
            "--gpu-portion-request 0.5] [-p P]\n"
            "     runai workload list [-A] | runai workload describe NAME | runai training|workspace delete NAME\n"
            "     runai node list | runai whoami | runai version\n"
        )
    if args[0] in ("version", "--version"):
        return f"Version: {CLI_VERSION}\nBuildDate: 2024-09-02T10:11:32Z\nGitCommit: 6b1fd7e\nGoVersion: go1.22.5\n"
    r = rstate(state)
    if ctx.host != HEAD:
        raise CommandError("Error: no kubeconfig / Run:ai context on this host. Use the CLI on bcm-head.", 1)
    kc.reconcile(state)
    sub, rest = args[0], args[1:]
    if sub == "whoami":
        return f"User: {r.current_user}\nCluster: {r.cluster}\n"
    if sub == "login":
        return f"Logged in successfully as {r.current_user}\n"
    if sub == "logout":
        return "Logged out successfully\n"
    # ---- v1
    if sub == "submit":
        return _submit(ctx, r, rest, "Training")
    if sub == "config" and rest[:1] == ["project"]:
        if len(rest) < 2:
            raise CommandError("Error: runai config project <name>", 1)
        if rest[1] not in r.projects:
            raise CommandError(f'Error: project "{rest[1]}" does not exist', 1)
        r.default_project = rest[1]
        return f"Project {rest[1]} has been set as default project\n"
    if sub == "list":
        what = rest[0] if rest else "jobs"
        _, opts, _ = _opts(rest[1:], {"-p", "--project"}, ALIASES)
        if what in ("jobs", "job", "workloads"):
            return _list_jobs(ctx, r, opts)
        if what in ("projects", "project"):
            return _list_projects(ctx, r)
        if what in ("nodes", "node"):
            return _list_nodes(ctx, r)
        raise CommandError(f'Error: unknown resource "{what}" (jobs, projects, nodes)', 1)
    if sub == "describe":
        if not rest or rest[0] not in ("job", "jobs", "workload") or len(rest) < 2:
            raise CommandError("Error: runai describe job NAME [-p PROJECT]", 1)
        pos, opts, _ = _opts(rest[1:], {"-p", "--project"}, ALIASES)
        return _describe(ctx, r, _find(r, pos[0], _project_of(ctx, r, opts)))
    if sub == "delete":
        if rest and rest[0] in ("job", "jobs"):
            rest = rest[1:]
        return _delete(ctx, r, rest)
    if sub == "logs":
        pos, opts, _ = _opts(rest, {"-p", "--project", "--tail"}, ALIASES)
        if not pos:
            raise CommandError("Error: runai logs NAME [-p PROJECT]", 1)
        w = _find(r, pos[0], _project_of(ctx, r, opts))
        if w.status != "Running":
            raise CommandError(f"Error: workload {w.name} has no running pod (status {w.status})", 1)
        return "\n".join(_logs(w)) + "\n"
    if sub in ("exec", "bash"):
        return _exec(ctx, r, rest)
    # ---- v2
    if sub == "project":
        action = rest[0] if rest else "list"
        if action == "list":
            return _list_projects(ctx, r)
        if action == "set":
            if len(rest) < 2 or rest[1] not in r.projects:
                raise CommandError(f'Error: project "{rest[1] if len(rest) > 1 else ""}" does not exist', 1)
            r.default_project = rest[1]
            return f"Project {rest[1]} set as default\n"
        raise CommandError(
            f'Error: unknown command "{action}" for "runai project". Projects are created by an administrator '
            "in the Run:ai UI/API (lab: runai-ui project create).",
            1,
        )
    if sub in ("training", "workspace", "inference"):
        action = rest[0] if rest else ""
        wtype = {"training": "Training", "workspace": "Interactive", "inference": "Inference"}[sub]
        if action == "submit":
            return _submit(ctx, r, rest[1:], wtype, v2=True)
        if action == "list":
            _, opts, _ = _opts(rest[1:], {"-p", "--project"}, ALIASES)
            out = _list_jobs(ctx, r, opts, v2=True).splitlines()
            label = {"Interactive": "Workspace"}.get(wtype, wtype)
            return "\n".join([out[0]] + [ln for ln in out[1:] if f"  {label}  " in ln]) + "\n"
        if action == "describe":
            pos, opts, _ = _opts(rest[1:], {"-p", "--project"}, ALIASES)
            return _describe(ctx, r, _find(r, pos[0], _project_of(ctx, r, opts)))
        if action == "delete":
            return _delete(ctx, r, rest[1:])
        if action in ("logs", "exec", "bash"):
            return runai(ctx, [action] + rest[1:])
        raise CommandError(f'Error: unknown command "{action}" for "runai {sub}"', 1)
    if sub == "workload":
        action = rest[0] if rest else "list"
        pos, opts, _ = _opts(rest[1:], {"-p", "--project"}, ALIASES)
        if action == "list":
            return _list_jobs(ctx, r, opts, v2=True)
        if action == "describe":
            if not pos:
                raise CommandError("Error: runai workload describe NAME [-p PROJECT]", 1)
            return _describe(ctx, r, _find(r, pos[0], _project_of(ctx, r, opts)))
        raise CommandError(f'Error: unknown command "{action}" for "runai workload"', 1)
    if sub == "node" and rest[:1] in (["list"], []):
        return _list_nodes(ctx, r)
    raise CommandError(f'Error: unknown command "{sub}" for "runai"\nRun \'runai --help\' for usage.', 1)


# ====================================================================== lab helper: runai-ui


def _on_off(v, default=True):
    if v is None:
        return default
    s = str(v).lower()
    if s in ("enabled", "enable", "true", "yes", "on", "1"):
        return True
    if s in ("disabled", "disable", "false", "no", "off", "0"):
        return False
    raise CommandError(f'Error: --over-quota must be "enabled" or "disabled", got "{v}"', 1)


def _num(v, flag):
    if v is None or v is True:
        raise CommandError(f"Error: {flag} needs a number", 1)
    try:
        f = float(v)
    except ValueError:
        raise CommandError(f'Error: invalid value "{v}" for {flag}', 1) from None
    if f < 0:
        raise CommandError(f"Error: {flag} cannot be negative", 1)
    return int(f) if f.is_integer() else f


UI_FLAGS = {
    "--department",
    "--gpu-quota",
    "--over-quota",
    "--max-gpus",
    "--user",
    "--users",
    "--project",
    "--role",
}


@command(
    "runai-ui",
    help="LAB HELPER (stands in for the Run:ai web UI / REST API): runai-ui department|project|access-rule "
    "create|update|list|delete ...",
)
def runai_ui(ctx, args):
    state = ctx.state
    r = rstate(state)
    usage = (
        "runai-ui  -  LAB HELPER. Stands in for the Run:ai web UI / REST API (there is no such real command).\n\n"
        "  runai-ui department list\n"
        "  runai-ui department create|update NAME --gpu-quota N [--over-quota enabled|disabled]\n"
        "  runai-ui project list\n"
        "  runai-ui project create NAME --department D --gpu-quota N [--over-quota enabled|disabled] [--max-gpus N]\n"
        "  runai-ui project update NAME [--gpu-quota N] [--over-quota enabled|disabled] [--max-gpus N|none]\n"
        "  runai-ui project delete NAME\n"
        "  runai-ui access-rule create --user U --project P [--role 'L1 researcher']\n"
        "  runai-ui access-rule list | delete --user U --project P\n"
        "  runai-ui user list\n"
    )
    if not args or args[0] in ("-h", "--help", "help"):
        return usage
    what, rest = args[0], args[1:]
    action = rest[0] if rest else "list"
    pos, opts, _ = _opts(rest[1:], UI_FLAGS)
    if what in ("department", "departments"):
        if action == "list":
            rows = [("DEPARTMENT", "GPU QUOTA", "OVER QUOTA", "PROJECTS", "ALLOCATED GPUs")]
            for name, d in sorted(r.departments.items()):
                projs = sorted(p.name for p in r.projects.values() if p.department == name)
                rows.append(
                    (
                        name,
                        _g(d.gpu_quota),
                        "enabled" if d.over_quota else "disabled",
                        ",".join(projs) or "-",
                        _g(_alloc(r, department=name)),
                    )
                )
            return kc.table(rows, gap=2)
        if action in ("create", "update"):
            if not pos:
                raise CommandError("Error: department name required", 1)
            name = pos[0]
            if action == "create" and name in r.departments:
                raise CommandError(f'Error: department "{name}" already exists', 1)
            if action == "update" and name not in r.departments:
                raise CommandError(f'Error: department "{name}" not found', 1)
            cur = r.departments.get(name) or {}
            quota = _num(opts.get("gpu-quota"), "--gpu-quota") if "gpu-quota" in opts else cur.get("gpu_quota", 0)
            create_department(state, name, quota, _on_off(opts.get("over-quota"), cur.get("over_quota", True)))
            return f'Department "{name}" {action}d (GPU quota {_g(quota)})\n'
        raise CommandError(f"Error: unknown action {action}\n\n" + usage, 1)
    if what in ("project", "projects"):
        if action == "list":
            return _list_projects(ctx, r)
        if not pos:
            raise CommandError("Error: project name required", 1)
        name = pos[0]
        if action == "create":
            if "gpu-quota" not in opts:
                raise CommandError("Error: --gpu-quota is required (the project's deserved GPUs; 0 is allowed)", 1)
            max_g = opts.get("max-gpus")
            p = create_project(
                state,
                name,
                department=opts.get("department") or "default",
                gpu_quota=_num(opts.get("gpu-quota"), "--gpu-quota"),
                over_quota=_on_off(opts.get("over-quota")),
                max_gpus=None if max_g in (None, "none", "unlimited") else _num(max_g, "--max-gpus"),
            )
            kc.reconcile(state)
            return (
                f'Project "{name}" created in department {p.department}: deserved GPUs {_g(p.gpu_quota)}, '
                f"over-quota {'enabled' if p.over_quota else 'disabled'}"
                + (f", max GPUs {_g(p.max_gpus)}" if p.max_gpus is not None else "")
                + f"\nNamespace {p.namespace} created\n"
            )
        if name not in r.projects:
            raise CommandError(f'Error: project "{name}" not found', 1)
        p = r.projects[name]
        if action == "update":
            if "department" in opts and opts["department"] != p.department:
                if opts["department"] not in r.departments:
                    raise CommandError(f'Error: department "{opts["department"]}" not found', 1)
                _check_dept_sum(r, opts["department"], p.gpu_quota, exclude=name)
                p.department = opts["department"]
            if "gpu-quota" in opts:
                q = _num(opts["gpu-quota"], "--gpu-quota")
                _check_dept_sum(r, p.department, q, exclude=name)
                p.gpu_quota = q
            if "over-quota" in opts:
                p.over_quota = _on_off(opts["over-quota"])
            if "max-gpus" in opts:
                mg = opts["max-gpus"]
                p.max_gpus = None if mg in ("none", "unlimited") else _num(mg, "--max-gpus")
            kc.reconcile(state)
            return f'Project "{name}" updated\n'
        if action == "delete":
            if any(w.project == name for w in r.workloads):
                raise CommandError(f'Error: project "{name}" still has workloads; delete them first', 1)
            del r.projects[name]
            r.access_rules = [a for a in r.access_rules if a.project != name]
            if r.default_project == name:
                r.default_project = ""
            return f'Project "{name}" deleted\n'
        raise CommandError(f"Error: unknown action {action}\n\n" + usage, 1)
    if what in ("access-rule", "access-rules"):
        if action == "list":
            rows = [("SUBJECT", "TYPE", "ROLE", "SCOPE")]
            for a in r.access_rules:
                rows.append((a.user, "user", a.role, f"project/{a.project}"))
            return kc.table(rows, gap=2)
        users = [u for u in str(opts.get("user") or opts.get("users") or "").split(",") if u]
        project = opts.get("project")
        if not users or not project or project is True:
            raise CommandError("Error: --user and --project are required", 1)
        if project not in r.projects:
            raise CommandError(f'Error: project "{project}" not found', 1)
        if action == "create":
            role = opts.get("role") or "L1 researcher"
            match = next((x for x in ROLES if x.lower() == str(role).lower()), None)
            if match is None:
                raise CommandError(f'Error: unknown role "{role}". Roles: {", ".join(ROLES)}', 1)
            out = []
            for u in users:
                if not any(a.user == u and a.project == project and a.role == match for a in r.access_rules):
                    r.access_rules.append({"user": u, "role": match, "project": project})
                if u not in r.users:
                    r.users.append(u)
                out.append(f"Access rule created: user {u} -> {match} on project/{project}")
            return "\n".join(out) + "\n"
        if action == "delete":
            before = len(r.access_rules)
            r.access_rules = [a for a in r.access_rules if not (a.user in users and a.project == project)]
            return f"{before - len(r.access_rules)} access rule(s) deleted\n"
        raise CommandError(f"Error: unknown action {action}\n\n" + usage, 1)
    if what in ("user", "users"):
        rows = [("USER", "PROJECTS")]
        for u in sorted(set(r.users)):
            rows.append((u, ",".join(sorted({a.project for a in r.access_rules if a.user == u})) or "-"))
        return kc.table(rows, gap=2)
    raise CommandError(usage, 1)


def members(state, project):
    """Users with an access rule on ``project`` (for scenario checks)."""
    r = state.get("runai") or {}
    return sorted({a.user for a in r.get("access_rules", []) if a.project == project})


def workload(state, name, project=None):
    r = state.get("runai") or {}
    return next(
        (w for w in r.get("workloads", []) if w.name == name and (project is None or w.project == project)), None
    )
