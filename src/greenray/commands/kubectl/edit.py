"""In-place edits: ``patch``, ``label``, ``taint``, ``cordon``/``drain``, ``rollout``, ``scale`` and ``wait``."""

from __future__ import annotations

import copy
import json
import re

import yaml

from ...core.errors import CommandError
from ...core.state import wrap
from .apply import json_norm, pod_update_allowed
from .controllers import ds_eligible
from .model import (
    DEFAULT_POD_TOLERATIONS,
    KIND_PREFIX,
    SINGULAR,
    advance,
    now,
    ns_of,
    pod_tolerations,
    pods_of,
    remove_pod,
    resolve_kind,
    selector_matches,
    split_targets,
    tolerates,
    ts,
    validate_pod_spec,
)
from .output import api_obj, list_objects
from .scheduler import reconcile
from .views import deploy_obj


def merge_patch(target, patch, strategic: bool = False):
    if not isinstance(patch, dict) or not isinstance(target, dict):
        return copy.deepcopy(patch)
    out = dict(target)
    for key, v in patch.items():
        if v is None:
            out.pop(key, None)
        elif isinstance(v, dict) and isinstance(out.get(key), dict):
            out[key] = merge_patch(out[key], v, strategic)
        elif (
            strategic
            and isinstance(v, list)
            and isinstance(out.get(key), list)
            and key in ("containers", "env", "volumes")
            and all(isinstance(x, dict) and "name" in x for x in v + out[key])
        ):
            merged = {x["name"]: x for x in out[key]}
            for x in v:
                merged[x["name"]] = merge_patch(merged.get(x["name"], {}), x, strategic)
            out[key] = list(merged.values())
        else:
            out[key] = copy.deepcopy(v)
    return out


def json_patch(doc, ops: list[dict]):
    doc = copy.deepcopy(doc)
    for op in ops:
        path = [p.replace("~1", "/").replace("~0", "~") for p in op["path"].split("/")[1:]]
        parent = doc
        for p in path[:-1]:
            parent = parent[int(p)] if isinstance(parent, list) else parent.setdefault(p, {})
        last = path[-1]
        kind = op["op"]
        if kind in ("add", "replace"):
            if isinstance(parent, list):
                if last == "-":
                    parent.append(op["value"])
                elif kind == "add":
                    parent.insert(int(last), op["value"])
                else:
                    parent[int(last)] = op["value"]
            else:
                if kind == "replace" and last not in parent:
                    raise CommandError(
                        "The request is invalid: the server rejected our request due to an error in our request", 1
                    )
                parent[last] = op["value"]
        elif kind == "remove":
            if isinstance(parent, list):
                parent.pop(int(last))
            elif last in parent:
                del parent[last]
            else:
                raise CommandError(
                    "The request is invalid: the server rejected our request due to an error in our request", 1
                )
    return doc


def cmd_patch(ctx, pos: list[str], opts: dict) -> str:
    state = ctx.state
    k = state.k8s
    targets = split_targets(pos)
    kind, names = targets[0]
    if not names:
        raise CommandError("error: resource(s) were provided, but no name was specified", 1)
    name = names[0]
    raw = opts.get("patch")
    if not raw:
        raise CommandError("error: must specify -p to patch", 1)
    try:
        patch = json.loads(raw)
    except ValueError:
        try:
            patch = yaml.safe_load(raw)
        except yaml.YAMLError:
            raise CommandError(f'error: unable to parse "{raw}": yaml: invalid', 1)
    ptype = opts.get("type", "strategic")
    if ptype not in ("json", "merge", "strategic"):
        raise CommandError(f'error: --type must be one of [json merge strategic], not "{ptype}"', 1)
    if ptype == "json" and not isinstance(patch, list):
        raise CommandError("error: json: cannot unmarshal object into Go value of type jsonpatch.Patch", 1)
    ns = ns_of(opts)
    items = list_objects(state, kind, opts, [name])
    obj = api_obj(state, kind, items[0])
    before = json_norm(obj)
    if ptype == "json":
        new = json_patch(obj, patch)
    else:
        new = merge_patch(obj, patch, strategic=(ptype == "strategic"))
    if json_norm(new) == before:
        return f"{KIND_PREFIX[kind]}/{name} patched (no change)\n"
    if kind == "nodes":
        kn = k.nodes[name]
        kn.labels = dict(new.get("metadata", {}).get("labels") or {})
        sp = new.get("spec") or {}
        kn.schedulable = not sp.get("unschedulable", False)
        kn.taints = [t for t in sp.get("taints") or [] if t.get("key") != "node.kubernetes.io/unschedulable"]
    elif kind == "pods":
        p = items[0]
        spec = new.get("spec") or {}
        spec.pop("nodeName", None)
        spec["tolerations"] = [t for t in spec.get("tolerations") or [] if t not in DEFAULT_POD_TOLERATIONS]
        for key_ in ("restartPolicy", "schedulerName", "serviceAccountName"):
            if key_ not in p.spec:
                spec.pop(key_, None)
        if not pod_update_allowed(p.spec, spec):
            raise CommandError(
                f'The Pod "{name}" is invalid: spec: Forbidden: pod updates may not change fields other than '
                "`spec.containers[*].image`,`spec.initContainers[*].image`,`spec.activeDeadlineSeconds`,"
                "`spec.tolerations` (only additions to existing tolerations),`spec.terminationGracePeriodSeconds` "
                "(allow it to be set to 1 if it was previously negative)",
                1,
            )
        p.spec = wrap(spec)
        p.labels = dict(new.get("metadata", {}).get("labels") or {})
    elif kind == "deployments":
        d = items[0]
        sp = new.get("spec") or {}
        validate_pod_spec("Deployment", name, (sp.get("template") or {}).get("spec") or {}, "spec.template.spec")
        d.replicas = int(sp.get("replicas", d.replicas))
        d.template = sp.get("template", d.template)
        d.generation = d.get("generation", 1) + 1
    elif kind == "clusterpolicies":
        k.clusterpolicy.spec = new.get("spec") or {}
    elif kind == "nicclusterpolicies":
        k.nicclusterpolicy.spec = new.get("spec") or {}
    elif kind == "serviceaccounts":
        k.serviceaccounts.setdefault(ns, {})[name] = {"imagePullSecrets": new.get("imagePullSecrets") or []}
    elif kind == "configmaps":
        items[0].data = {a: str(b) for a, b in (new.get("data") or {}).items()}
    elif kind == "daemonsets":
        d = items[0]
        tspec = ((new.get("spec") or {}).get("template") or {}).get("spec") or {}
        d.tolerations = tspec.get("tolerations") or []
        d.node_selector = tspec.get("nodeSelector") or {}
        if tspec.get("containers"):
            d.image = tspec["containers"][0].get("image", d.image)
    else:
        raise CommandError(f"error: patching {kind} is not supported in this lab", 1)
    return f"{KIND_PREFIX[kind]}/{name} patched\n"


def _target_nodes(state, pos, opts, verb):
    k = state.k8s
    if pos and "/" in pos[0]:
        names = [p.split("/", 1)[1] for p in pos if "/" in p]
        rest = [p for p in pos if "/" not in p]
    else:
        if not pos:
            raise CommandError("error: You must provide one or more resources by argument or filename.", 1)
        resolve_kind(pos[0])
        names = []
        rest = pos[1:]
        if opts.get("all"):
            names = sorted(k.nodes)
        elif opts.get("selector"):
            names = [n for n in sorted(k.nodes) if selector_matches(opts["selector"], k.nodes[n].labels)]
        else:
            while rest and "=" not in rest[0] and not rest[0].endswith("-") and ":" not in rest[0]:
                names.append(rest.pop(0))
    for n in names:
        if n not in k.nodes:
            raise CommandError(f'Error from server (NotFound): nodes "{n}" not found', 1)
    return names, rest


def cmd_label(state, pos: list[str], opts: dict) -> str:
    k = state.k8s
    kind = resolve_kind(pos[0].split("/")[0]) if pos else None
    if kind not in ("nodes", "pods", "namespaces"):
        raise CommandError("error: this lab supports labeling nodes, pods and namespaces", 1)
    if kind == "nodes":
        names, changes = _target_nodes(state, pos, opts, "label")
        targets = [(f"node/{n}", k.nodes[n].labels) for n in names]
    elif kind == "pods":
        if "/" in pos[0]:
            names, changes = [pos[0].split("/", 1)[1]], pos[1:]
        else:
            names, changes = [pos[1]], pos[2:]
        targets = []
        for nm in names:
            p = next((p for p in k.pods if p.namespace == ns_of(opts) and p.name == nm), None)
            if p is None:
                raise CommandError(f'Error from server (NotFound): pods "{nm}" not found', 1)
            targets.append((f"pod/{nm}", p.labels))
    else:
        raise CommandError("error: namespace labels are not modelled in this lab", 1)
    if not changes:
        raise CommandError("error: at least one label update is required", 1)
    out = []
    for ref, labels in targets:
        changed, removed = False, False
        for ch in changes:
            if ch.endswith("-") and "=" not in ch:
                key = ch[:-1]
                if key in labels:
                    del labels[key]
                    removed = True
                continue
            if "=" not in ch:
                raise CommandError(f"error: invalid label spec: {ch}", 1)
            key, val = ch.split("=", 1)
            if (
                not re.fullmatch(r"([a-z0-9.-]+/)?[A-Za-z0-9]([-A-Za-z0-9_.]*[A-Za-z0-9])?", key)
                or len(val) > 63
                or (val and not re.fullmatch(r"[A-Za-z0-9]([-A-Za-z0-9_.]*[A-Za-z0-9])?", val))
            ):
                raise CommandError(
                    f'error: invalid label value: "{ch}": a valid label must be an empty string or consist of '
                    "alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character",
                    1,
                )
            if key in labels and labels[key] != val and not opts.get("overwrite"):
                raise CommandError(f"error: '{key}' already has a value ({labels[key]}), and --overwrite is false", 1)
            if labels.get(key) != val:
                labels[key] = val
                changed = True
        out.append(f"{ref} {'labeled' if changed else 'unlabeled' if removed else 'not labeled'}")
    return "\n".join(out) + "\n"


def cmd_taint(state, pos: list[str], opts: dict) -> str:
    k = state.k8s
    names, specs = _target_nodes(state, pos, opts, "taint")
    if not specs:
        raise CommandError("error: at least one taint update is required", 1)
    out = []
    for n in names:
        kn = k.nodes[n]
        changed = False
        removed = False
        for s in specs:
            remove = s.endswith("-")
            s2 = s[:-1] if remove else s
            key, eff = s2, None
            if ":" in s2:
                key, eff = s2.rsplit(":", 1)
            val = ""
            if "=" in key:
                key, val = key.split("=", 1)
            if remove:
                before = len(kn.taints)
                kn.taints = [t for t in kn.taints if not (t.key == key and (eff is None or t.effect == eff))]
                if len(kn.taints) == before:
                    raise CommandError(f'error: taint "{s2}" not found', 1)
                removed = True
                continue
            if eff not in ("NoSchedule", "PreferNoSchedule", "NoExecute"):
                raise CommandError(f"error: invalid taint effect: {eff}, unsupported taint effect", 1)
            same = [t for t in kn.taints if t.key == key and t.effect == eff]
            if same and not opts.get("overwrite"):
                raise CommandError(
                    f"error: node {n} already has {key} taint(s) with same effect(s) and --overwrite is false", 1
                )
            kn.taints = [t for t in kn.taints if not (t.key == key and t.effect == eff)]
            kn.taints.append({"key": key, "value": val, "effect": eff})
            changed = True
            if eff == "NoExecute":
                for p in list(k.pods):
                    if p.node == n and not p.get("component") and not tolerates(pod_tolerations(p), kn.taints[-1]):
                        remove_pod(state, p)
        out.append(f"node/{n} {'tainted' if changed else 'untainted' if removed else 'modified'}")
    return "\n".join(out) + "\n"


def cmd_cordon(state, pos: list[str], opts: dict, on: bool = True) -> str:
    k = state.k8s
    names = [p.split("/", 1)[-1] for p in pos] or (sorted(k.nodes) if opts.get("all") else [])
    if opts.get("selector"):
        names = [n for n in sorted(k.nodes) if selector_matches(opts["selector"], k.nodes[n].labels)]
    out = []
    for n in names:
        if n not in k.nodes:
            raise CommandError(f'Error from server (NotFound): nodes "{n}" not found', 1)
        kn = k.nodes[n]
        if kn.get("schedulable", True) == (not on):
            out.append(f"node/{n} already {'cordoned' if on else 'uncordoned'}")
            continue
        kn.schedulable = not on
        out.append(f"node/{n} {'cordoned' if on else 'uncordoned'}")
    return "\n".join(out) + "\n"


def cmd_drain(state, pos: list[str], opts: dict) -> str:
    k = state.k8s
    names = [p.split("/", 1)[-1] for p in pos]
    out = []
    for n in names:
        if n not in k.nodes:
            raise CommandError(f'Error from server (NotFound): nodes "{n}" not found', 1)
        pods = [
            p
            for p in k.pods
            if p.node == n and p.phase not in ("Succeeded", "Failed") and p.get("component") != "static"
        ]
        ds_pods = [p for p in pods if p.get("owner") and p.owner.get("kind") == "DaemonSet"]
        bare = [p for p in pods if not p.get("owner")]
        errs = []
        if ds_pods and not opts.get("ignore-daemonsets"):
            errs.append(
                "cannot delete DaemonSet-managed Pods (use --ignore-daemonsets to ignore): "
                + ", ".join(f"{p.namespace}/{p.name}" for p in ds_pods)
            )
        if bare and not opts.get("force"):
            errs.append(
                "cannot delete Pods declare no controller (use --force to override): "
                + ", ".join(f"{p.namespace}/{p.name}" for p in bare)
            )
        lines = []
        if k.nodes[n].get("schedulable", True):
            k.nodes[n].schedulable = False
            lines.append(f"node/{n} cordoned")
        else:
            lines.append(f"node/{n} already cordoned")
        if errs:
            raise CommandError(
                "\n".join(
                    lines
                    + [
                        f'error: unable to drain node "{n}" due to error:'
                        + ", ".join(errs)
                        + ", continuing command...",
                        "There are pending nodes to be drained:",
                        f" {n}",
                    ]
                    + [f"error: {e}" for e in errs]
                ),
                1,
            )
        if ds_pods:
            lines.append(
                "Warning: ignoring DaemonSet-managed Pods: " + ", ".join(f"{p.namespace}/{p.name}" for p in ds_pods)
            )
        if bare:
            lines.append(
                "Warning: deleting Pods that declare no controller: "
                + ", ".join(f"{p.namespace}/{p.name}" for p in bare)
            )
        victims = [p for p in pods if p not in ds_pods]
        for p in victims:
            lines.append(f"evicting pod {p.namespace}/{p.name}")
        for p in victims:
            remove_pod(state, p)
            lines.append(f"pod/{p.name} evicted")
        lines.append(f"node/{n} drained")
        out.extend(lines)
    return "\n".join(out) + "\n"


def _ref(pos: list[str], start: int = 0) -> tuple[str, str]:
    """TYPE/NAME or TYPE NAME -> (kind, name)."""
    if len(pos) <= start:
        raise CommandError("error: required resource not specified", 1)
    ref = pos[start] if "/" in pos[start] else f"{pos[start]}/{pos[start + 1] if len(pos) > start + 1 else ''}"
    kw, name = ref.split("/", 1)
    if not name:
        raise CommandError("error: required resource not specified", 1)
    return resolve_kind(kw), name


def _deployment(state, ns: str, name: str):
    d = next((d for d in state.k8s.deployments if d.namespace == ns and d.name == name), None)
    if d is None:
        raise CommandError(f'Error from server (NotFound): deployments.apps "{name}" not found', 1)
    return d


def revisions(d) -> list:
    """The deployment's rollout history; revision 1 is recorded the first time anything asks."""
    from .controllers import template_hash

    if not d.get("revisions"):
        d["revisions"] = [
            {"revision": 1, "hash": template_hash(d.template), "template": copy.deepcopy(d.template), "cause": "<none>"}
        ]
    return d.revisions


def new_revision(state, d, template: dict, cause: str) -> None:
    """Record a template change as a new revision, the way the deployment controller keeps old ReplicaSets."""
    from .controllers import template_hash

    hist = revisions(d)
    d.template = wrap(copy.deepcopy(template))
    d.generation = int(d.get("generation", 1)) + 1
    h = template_hash(d.template)
    hist[:] = [r for r in hist if r["hash"] != h]  # rolling back re-uses the old ReplicaSet under a new number
    hist.append(
        {
            "revision": max([r["revision"] for r in hist], default=0) + 1,
            "hash": h,
            "template": copy.deepcopy(template),
            "cause": cause,
        }
    )
    del hist[:-10]  # revisionHistoryLimit


def cmd_set(state, pos: list[str], opts: dict) -> str:
    if not pos or pos[0] != "image":
        raise CommandError("error: kubectl set supports `set image` in this lab", 1)
    kind, name = _ref(pos, 1)
    start = 2 if "/" in pos[1] else 3
    pairs = pos[start:]
    if not pairs:
        raise CommandError("error: at least one image update is required", 1)
    ns = ns_of(opts)
    if kind == "deployments":
        d = _deployment(state, ns, name)
        template = json.loads(json.dumps(d.template, default=str))
        label = f"deployment.apps/{name}"
    elif kind == "daemonsets":
        d = next((x for x in state.k8s.daemonsets if x.namespace == ns and x.name == name), None)
        if d is None:
            raise CommandError(f'Error from server (NotFound): daemonsets.apps "{name}" not found', 1)
        template = None
        label = f"daemonset.apps/{name}"
    else:
        raise CommandError(f"error: set image is not supported for {kind} in this lab", 1)
    containers = template["spec"]["containers"] if template else [{"name": d.container, "image": d.image}]
    for pair in pairs:
        cname, _, image = pair.partition("=")
        if not image:
            raise CommandError(f'error: invalid image specification "{pair}", expected CONTAINER=IMAGE', 1)
        hit = [c for c in containers if cname == "*" or c.get("name") == cname]
        if not hit:
            raise CommandError(f'error: unable to find container named "{cname}"', 1)
        for c in hit:
            c["image"] = image
    if template:
        if json.dumps(template, sort_keys=True) == json.dumps(
            json.loads(json.dumps(d.template, default=str)), sort_keys=True
        ):
            return f"{label} image updated (no change)\n"
        new_revision(state, d, template, f"kubectl set image {' '.join(pos[1:])}")
    else:
        d.image = containers[0]["image"]
        for p in pods_of(state, "DaemonSet", name, ns):
            remove_pod(state, p)
    return f"{label} image updated\n"


def cmd_expose(state, pos: list[str], opts: dict) -> str:
    kind, name = _ref(pos)
    ns = ns_of(opts)
    if kind not in ("deployments", "pods"):
        raise CommandError(f"error: cannot expose a {SINGULAR[kind]}: it has no pod template", 1)
    if kind == "deployments":
        d = _deployment(state, ns, name)
        selector = dict(d.selector.get("matchLabels", {}))
    else:
        p = next((p for p in state.k8s.pods if p.namespace == ns and p.name == name), None)
        if p is None:
            raise CommandError(f'Error from server (NotFound): pods "{name}" not found', 1)
        selector = dict(p.labels)
    if "port" not in opts:
        raise CommandError("error: couldn't find port via --port flag or introspection", 1)
    port = str(opts["port"])
    if not port.isdigit():
        raise CommandError(
            f'error: invalid argument "{port}" for "--port" flag: strconv.ParseInt: parsing "{port}": invalid syntax', 1
        )
    stype = opts.get("type", "ClusterIP")
    if stype not in ("ClusterIP", "NodePort", "LoadBalancer"):
        raise CommandError(f"error: '{stype}' is not a valid service type: ClusterIP, NodePort or LoadBalancer", 1)
    svc_name = opts.get("name", name)
    k = state.k8s
    if any(s.namespace == ns and s.name == svc_name for s in k.services) or (
        ns == "default" and svc_name == "kubernetes"
    ):
        raise CommandError(f'Error from server (AlreadyExists): services "{svc_name}" already exists', 1)
    proto = opts.get("protocol", "TCP")
    ports = [f"{port}/{proto}" if stype == "ClusterIP" else f"{port}:{30000 + len(k.services)}/{proto}"]
    k.services.append(
        {
            "namespace": ns,
            "name": svc_name,
            "type": stype,
            "ports": ports,
            "target_port": str(opts.get("target-port", port)),
            "selector": selector,
            "cluster_ip": f"10.96.{len(k.services) // 250}.{10 + len(k.services) % 250}",
            "created": now(state),
        }
    )
    return f"service/{svc_name} exposed\n"


def cmd_rollout(state, pos: list[str], opts: dict) -> str:
    if len(pos) < 2:
        raise CommandError("error: required resource not specified", 1)
    action, ref = pos[0], pos[1] if "/" in pos[1] else f"{pos[1]}/{pos[2]}"
    kw, name = ref.split("/", 1)
    kind = resolve_kind(kw)
    ns = ns_of(opts)
    k = state.k8s
    if kind == "daemonsets":
        d = next((d for d in k.daemonsets if d.namespace == ns and d.name == name), None)
        if d is None:
            raise CommandError(f'Error from server (NotFound): daemonsets.apps "{name}" not found', 1)
        pods = pods_of(state, "DaemonSet", name, ns)
        if action == "restart":
            for p in list(pods):
                remove_pod(state, p)
            return f"daemonset.apps/{name} restarted\n"
        if action in ("history", "undo"):
            raise CommandError(f"error: rollout {action} is only supported for deployments in this lab", 1)
        if action == "status":
            reconcile(state)
            pods = pods_of(state, "DaemonSet", name, ns)
            ready = sum(1 for p in pods if p.status == "Running")
            want = len(ds_eligible(state, d))
            if ready == want:
                return f'daemon set "{name}" successfully rolled out\n'
            raise CommandError(
                f'Waiting for daemon set "{name}" rollout to finish: {ready} of {want} updated pods are available...\n'
                "error: timed out waiting for the condition",
                1,
            )
    if kind == "deployments":
        d = next((d for d in k.deployments if d.namespace == ns and d.name == name), None)
        if d is None:
            raise CommandError(f'Error from server (NotFound): deployments.apps "{name}" not found', 1)
        if action == "restart":
            template = json.loads(json.dumps(d.template, default=str))
            template.setdefault("metadata", {}).setdefault("annotations", {})["kubectl.kubernetes.io/restartedAt"] = ts(
                now(state)
            )
            new_revision(state, d, template, "<none>")
            return f"deployment.apps/{name} restarted\n"
        if action == "history":
            hist = revisions(d)
            if "revision" in opts:
                rev = next((r for r in hist if str(r["revision"]) == str(opts["revision"])), None)
                if rev is None:
                    raise CommandError("error: unable to find the specified revision", 1)
                tpl = rev["template"]
                lines = [f"deployment.apps/{name} with revision #{rev['revision']}", "Pod Template:"]
                lines.append(
                    "  Labels:\t"
                    + ",".join(f"{a}={b}" for a, b in sorted(tpl.get("metadata", {}).get("labels", {}).items()))
                    + f",pod-template-hash={rev['hash']}"
                )
                lines.append("  Containers:")
                for c in tpl.get("spec", {}).get("containers", []):
                    lines += [
                        f"   {c.get('name')}:",
                        f"    Image:\t{c.get('image')}",
                        "    Environment:\t<none>",
                        "    Mounts:\t<none>",
                    ]
                lines += ["  Volumes:\t<none>", "  Node-Selectors:\t<none>", "  Tolerations:\t<none>"]
                return "\n".join(lines) + "\n"
            lines = [f"deployment.apps/{name} ", "REVISION  CHANGE-CAUSE"]
            lines += [f"{r['revision']:<9} {r['cause']}" for r in hist]
            return "\n".join(lines) + "\n"
        if action == "undo":
            hist = revisions(d)
            if len(hist) < 2 and "to-revision" not in opts:
                raise CommandError("error: no rollout history found", 1)
            if "to-revision" in opts:
                want_rev = str(opts["to-revision"])
                target = next((r for r in hist if str(r["revision"]) == want_rev), None)
                if target is None:
                    raise CommandError(f"error: unable to find specified revision {want_rev} in history", 1)
                if target is hist[-1]:
                    return f"deployment.apps/{name} skipped rollback (current template already matches revision {want_rev})\n"
            else:
                target = hist[-2]
            new_revision(state, d, target["template"], target["cause"])
            return f"deployment.apps/{name} rolled back\n"
        if action == "status":
            reconcile(state)
            st = deploy_obj(state, d)["status"]
            if st["readyReplicas"] == int(d.replicas):
                return f'deployment "{name}" successfully rolled out\n'
            raise CommandError(
                f'Waiting for deployment "{name}" rollout to finish: {st["readyReplicas"]} of {d.replicas} '
                "updated replicas are available...\nerror: timed out waiting for the condition",
                1,
            )
    raise CommandError(f"error: rollout {action} is not supported for {kind} in this lab", 1)


def cmd_scale(state, pos: list[str], opts: dict) -> str:
    if not pos or ("/" not in pos[0] and len(pos) < 2):
        raise CommandError("error: required resource not specified", 1)
    ref = pos[0] if "/" in pos[0] else f"{pos[0]}/{pos[1]}"
    kw, name = ref.split("/", 1)
    kind = resolve_kind(kw)
    if kind != "deployments" or "replicas" not in opts:
        raise CommandError("error: scale supports deployments with --replicas in this lab", 1)
    d = next((d for d in state.k8s.deployments if d.namespace == ns_of(opts) and d.name == name), None)
    if d is None:
        raise CommandError(f'Error from server (NotFound): deployments.apps "{name}" not found', 1)
    if not str(opts["replicas"]).isdigit():
        raise CommandError(
            f'error: invalid argument "{opts["replicas"]}" for "--replicas" flag: strconv.ParseInt: parsing "{opts["replicas"]}": invalid syntax',
            1,
        )
    d.replicas = int(opts["replicas"])
    return f"deployment.apps/{name} scaled\n"


def cmd_wait(state, pos: list[str], opts: dict) -> str:
    cond = opts.get("for", "")
    advance(state, 60)
    reconcile(state)
    out = []
    for kind, names in split_targets(pos):
        if names and opts.get("selector"):
            raise CommandError("error: name cannot be provided when a selector is specified", 1)
        if names and opts.get("all"):
            raise CommandError("error: cannot use --all together with a resource name", 1)
        items = list_objects(state, kind, opts, names) if names else list_objects(state, kind, opts)
        for i in items:
            ok = False
            if kind == "pods":
                if cond.lower() in ("condition=ready", "condition=ready=true"):
                    ok = i.status == "Running"
                elif cond.lower().startswith("delete"):
                    ok = False
                else:
                    ok = i.status in ("Running", "Completed")
            elif kind == "nodes":
                ok = state.k8s.nodes[i.name].get("ready", True)
            else:
                ok = True
            if not ok:
                raise CommandError(f"error: timed out waiting for the condition on {kind}/{i.get('name')}", 1)
            out.append(f"{SINGULAR[kind]}/{i.get('name')} condition met")
    return "\n".join(out) + "\n"
