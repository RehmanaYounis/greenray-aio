"""Manifest handling: ``apply``/``create``/``delete`` of API objects and the scenario helper ``apply_yaml``."""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import yaml

from ...core import audit
from ...core.errors import CommandError
from ...core.state import wrap
from .. import container as ct
from .gpu_operator import strip_operator_labels
from .model import (
    EXTRA_KINDS,
    SINGULAR,
    add_daemonset,
    add_deployment,
    ensure_namespace,
    new_pod,
    now,
    ns_of,
    pod_requests,
    remove_pod,
    validate_pod_spec,
)
from .output import crd_installed
from .scheduler import k8s_init, reconcile


def _read_manifest(ctx, path):
    if path == "-":
        return ctx.stdin
    if path.startswith(("http://", "https://")):
        if "k8s-device-plugin" in path and "nvidia-device-plugin.yml" in path:
            return STATIC_DEVICE_PLUGIN
        raise CommandError(f'error: unable to read URL "{path}", server reported 404 Not Found, status code=404', 1)
    full = path if path.startswith("/") else f"{ctx.cwd.rstrip('/')}/{path}"
    files = ctx.node.files
    if full not in files and path not in files:
        raise CommandError(f'error: the path "{path}" does not exist', 1)
    return files.get(full, files.get(path))


STATIC_DEVICE_PLUGIN = """\
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: nvidia-device-plugin-daemonset
  namespace: kube-system
spec:
  selector:
    matchLabels:
      name: nvidia-device-plugin-ds
  template:
    metadata:
      labels:
        name: nvidia-device-plugin-ds
    spec:
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      priorityClassName: "system-node-critical"
      containers:
      - image: nvcr.io/nvidia/k8s-device-plugin:v0.15.0
        name: nvidia-device-plugin-ctr
        env:
          - name: FAIL_ON_INIT_ERROR
            value: "false"
"""


def load_docs(ctx, opts: dict) -> list[tuple[str, dict]]:
    files = opts.get("filename") or []
    if not files:
        raise CommandError("error: must specify one of -f and -k", 1)
    docs = []
    for f in files:
        text = _read_manifest(ctx, f)
        try:
            for d in yaml.safe_load_all(text):
                if d is None:
                    continue
                if not isinstance(d, dict):
                    raise CommandError(
                        f"error: error parsing {f}: error unmarshaling JSON: json: cannot unmarshal string into Go value of type map[string]interface {{}}",
                        1,
                    )
                if d.get("kind") == "List":
                    docs.extend((f, x) for x in d.get("items", []))
                else:
                    docs.append((f, d))
        except yaml.YAMLError as e:
            mark = getattr(e, "problem_mark", None)
            line = mark.line + 1 if mark else 1
            raise CommandError(
                f"error: error parsing {f}: error converting YAML to JSON: yaml: line {line}: "
                f"{getattr(e, 'problem', 'could not parse')}",
                1,
            )
    return docs


ALLOWED_POD_UPDATE = ("image", "activeDeadlineSeconds", "terminationGracePeriodSeconds")


def json_norm(o):
    return json.loads(json.dumps(o, sort_keys=True, default=str))


def pod_update_allowed(old_spec, new_spec) -> bool:
    o, n = json_norm(old_spec), json_norm(new_spec)
    for spec in (o, n):
        spec.pop("activeDeadlineSeconds", None)
        spec.pop("terminationGracePeriodSeconds", None)
        for c in spec.get("containers", []):
            c.pop("image", None)
    ot, nt = o.pop("tolerations", None) or [], n.pop("tolerations", None) or []
    if any(t not in nt for t in ot):
        return False
    return o == n


def apply_doc(ctx, src: str, doc: dict, mode: str = "apply") -> str:
    state = ctx.state
    k = state.k8s
    kind = doc.get("kind")
    api = doc.get("apiVersion", "")
    meta = doc.get("metadata") or {}
    name = meta.get("name")
    if not kind or not api:
        raise CommandError(
            f'error: error validating "{src}": error validating data: [apiVersion not set, kind not set]; '
            "if you choose to ignore these errors, turn validation off with --validate=false",
            1,
        )
    if not name and kind != "List":
        raise CommandError(
            f'error: error when retrieving current configuration of:\nResource: "{kind.lower()}s", '
            f'GroupVersionKind: "{api}, Kind={kind}"\nName: "", Namespace: "default"\nfrom server for: '
            f'"{src}": resource name may not be empty',
            1,
        )
    ns = meta.get("namespace") or ctx_ns(ctx) or "default"
    labels = dict(meta.get("labels") or {})
    verb_new = "created"
    exists_err = f'Error from server (AlreadyExists): error when creating "{src}": {{kind}} "{name}" already exists'

    def need_ns():
        if ns not in k.namespaces:
            raise CommandError(
                f'Error from server (NotFound): error when creating "{src}": namespaces "{ns}" not found', 1
            )

    if kind == "Namespace":
        if name in k.namespaces:
            if mode == "create":
                raise CommandError(exists_err.format(kind="namespaces"), 1)
            return f"namespace/{name} unchanged"
        ensure_namespace(state, name)
        return f"namespace/{name} {verb_new}"
    if kind == "Pod":
        need_ns()
        spec = doc.get("spec") or {}
        validate_pod_spec("Pod", name, spec)
        if spec.get("runtimeClassName") and not any(r.name == spec["runtimeClassName"] for r in k.runtimeclasses):
            raise CommandError(
                f'Error from server (Forbidden): error when creating "{src}": pods "{name}" is forbidden: '
                f'pod rejected: RuntimeClass "{spec["runtimeClassName"]}" not found',
                1,
            )
        cur = next((p for p in k.pods if p.namespace == ns and p.name == name), None)
        if cur:
            if mode == "create":
                raise CommandError(exists_err.format(kind="pods"), 1)
            if (
                json_norm(cur.spec) == json_norm(with_sa_secrets(state, ns, spec, cur.spec))
                and dict(cur.labels) == labels
            ):
                return f"pod/{name} unchanged"
            if not pod_update_allowed(cur.spec, with_sa_secrets(state, ns, spec, cur.spec)):
                diff = _spec_diff(cur.spec, spec)
                raise CommandError(
                    f'The Pod "{name}" is invalid: spec: Forbidden: pod updates may not change fields other than '
                    "`spec.containers[*].image`,`spec.initContainers[*].image`,`spec.activeDeadlineSeconds`,"
                    "`spec.tolerations` (only additions to existing tolerations),`spec.terminationGracePeriodSeconds` "
                    "(allow it to be set to 1 if it was previously negative)\n" + diff,
                    1,
                )
            cur.spec = wrap(with_sa_secrets(state, ns, spec, cur.spec))
            cur.labels = labels
            cur.requests = pod_requests(cur.spec)
            return f"pod/{name} configured"
        spec = with_sa_secrets(state, ns, spec)
        new_pod(state, ns, name, spec, labels=labels)
        return f"pod/{name} {verb_new}"
    if kind in ("Deployment", "Job", "DaemonSet") and api.startswith(("apps/", "batch/")):
        need_ns()
        spec = doc.get("spec") or {}
        tmpl = spec.get("template") or {}
        validate_pod_spec(kind, name, tmpl.get("spec") or {}, "spec.template.spec")
        tl = (tmpl.get("metadata") or {}).get("labels") or {}
        ml = (spec.get("selector") or {}).get("matchLabels")
        if kind != "Job" and (ml is None or not all(tl.get(a) == b for a, b in ml.items())):
            raise CommandError(
                f'The {kind} "{name}" is invalid: spec.template.metadata.labels: Invalid value: '
                f"map[string]string{{{', '.join(f'{a!r}:{b!r}' for a, b in tl.items())}}}: "
                "`selector` does not match template `labels`",
                1,
            )
        tmpl = json_norm(tmpl)
        tmpl["spec"] = with_sa_secrets(state, ns, tmpl.get("spec") or {})
        if kind == "Deployment":
            cur = next((d for d in k.deployments if d.namespace == ns and d.name == name), None)
            if cur:
                if mode == "create":
                    raise CommandError(exists_err.format(kind="deployments.apps"), 1)
                if json_norm(cur.template) == tmpl and int(cur.replicas) == int(spec.get("replicas", 1)):
                    return f"deployment.apps/{name} unchanged"
                cur.template = tmpl
                cur.replicas = int(spec.get("replicas", 1))
                cur.generation = cur.get("generation", 1) + 1
                cur.selector = {"matchLabels": dict(ml)}
                return f"deployment.apps/{name} configured"
            d = add_deployment(
                state, ns, name, "", replicas=int(spec.get("replicas", 1)), template=tmpl, labels=labels or None
            )
            d.selector = {"matchLabels": dict(ml)}
            return f"deployment.apps/{name} {verb_new}"
        if kind == "Job":
            cur = next((j for j in k.jobs if j.namespace == ns and j.name == name), None)
            if cur:
                if json_norm(cur.template) == tmpl:
                    return f"job.batch/{name} unchanged"
                raise CommandError(
                    f'The Job "{name}" is invalid: spec.template: Invalid value: core.PodTemplateSpec{{...}}: field is immutable',
                    1,
                )
            k.jobs.append(
                {
                    "namespace": ns,
                    "name": name,
                    "template": tmpl,
                    "parallelism": spec.get("parallelism", 1),
                    "completions": spec.get("completions", 1),
                    "created": now(state),
                }
            )
            return f"job.batch/{name} {verb_new}"
        # DaemonSet
        cspec = tmpl["spec"]["containers"][0]
        image = cspec.get("image", "")
        comp = "standalone-device-plugin" if "k8s-device-plugin" in image else "generic"
        env = {e.get("name"): e.get("value") for e in cspec.get("env") or []}
        existed = any(d.namespace == ns and d.name == name for d in k.daemonsets)
        add_daemonset(
            state,
            ns,
            name,
            image,
            comp,
            node_selector=tmpl["spec"].get("nodeSelector") or {},
            tolerations=tmpl["spec"].get("tolerations") or [],
            owner="user",
            container=cspec.get("name"),
            runtime_class=tmpl["spec"].get("runtimeClassName"),
            extra={"mig_strategy": env.get("MIG_STRATEGY", "none")},
        )
        return f"daemonset.apps/{name} {'configured' if existed else verb_new}"
    if kind == "Secret":
        need_ns()
        data = {a: base64.b64decode(str(b)).decode(errors="replace") for a, b in (doc.get("data") or {}).items()}
        data.update({a: str(b) for a, b in (doc.get("stringData") or {}).items()})
        return _upsert(
            k.secrets, ns, name, {"type": doc.get("type", "Opaque"), "data": data}, state, "secret", mode, src
        )
    if kind == "ConfigMap":
        need_ns()
        return _upsert(k.configmaps, ns, name, {"data": dict(doc.get("data") or {})}, state, "configmap", mode, src)
    if kind == "ServiceAccount":
        need_ns()
        k.serviceaccounts.setdefault(ns, {})[name] = {"imagePullSecrets": list(doc.get("imagePullSecrets") or [])}
        return f"serviceaccount/{name} configured"
    if kind == "Service":
        need_ns()
        return f"service/{name} {verb_new}"
    if kind == "RuntimeClass":
        cur = next((r for r in k.runtimeclasses if r.name == name), None)
        if cur:
            cur.handler = doc.get("handler", name)
            return f"runtimeclass.node.k8s.io/{name} configured"
        k.runtimeclasses.append({"name": name, "handler": doc.get("handler", name), "created": now(state)})
        return f"runtimeclass.node.k8s.io/{name} {verb_new}"
    if kind == "ClusterPolicy":
        if not crd_installed(state, "clusterpolicies"):
            raise CommandError(
                f'error: resource mapping not found for name: "{name}" namespace: "" from "{src}": no matches for kind '
                f'"ClusterPolicy" in version "{api}"\nensure CRDs are installed first',
                1,
            )
        if k.get("clusterpolicy"):
            if json_norm(k.clusterpolicy.spec) == json_norm(doc.get("spec") or {}):
                return f"clusterpolicy.nvidia.com/{name} unchanged"
            k.clusterpolicy.spec = doc.get("spec") or {}
            return f"clusterpolicy.nvidia.com/{name} configured"
        rel = next((r for r in k.helm_releases if r.chart == "gpu-operator"), None)
        k.clusterpolicy = {
            "name": name,
            "namespace": rel.namespace if rel else "gpu-operator",
            "spec": doc.get("spec") or {},
            "created": now(state),
            "version": rel.version if rel else "v24.6.2",
            "release": None,
        }
        return f"clusterpolicy.nvidia.com/{name} {verb_new}"
    ext = next((h for h in EXTRA_KINDS.values() if h.kind == kind), None)
    if ext is not None:
        return ext.apply(state, doc, src)
    if kind == "NicClusterPolicy":
        if not crd_installed(state, "nicclusterpolicies"):
            raise CommandError(
                f'error: resource mapping not found for name: "{name}" namespace: "" from "{src}": no matches for kind '
                f'"NicClusterPolicy" in version "{api}"\nensure CRDs are installed first',
                1,
            )
        rel = next((r for r in k.helm_releases if r.chart == "network-operator"), None)
        if k.get("nicclusterpolicy"):
            k.nicclusterpolicy.spec = doc.get("spec") or {}
            return f"nicclusterpolicy.mellanox.com/{name} configured"
        k.nicclusterpolicy = {
            "name": name,
            "namespace": rel.namespace if rel else "nvidia-network-operator",
            "spec": doc.get("spec") or {},
            "created": now(state),
            "release": None,
        }
        return f"nicclusterpolicy.mellanox.com/{name} {verb_new}"
    raise CommandError(
        f'error: resource mapping not found for name: "{name}" namespace: "{ns}" from "{src}": no matches for kind '
        f'"{kind}" in version "{api}"\nensure CRDs are installed first',
        1,
    )


def _upsert(lst, ns, name, fields, state, label, mode, src):
    cur = next((s for s in lst if s.namespace == ns and s.name == name), None)
    if cur:
        if mode == "create":
            raise CommandError(
                f'Error from server (AlreadyExists): error when creating "{src}": {label}s "{name}" already exists', 1
            )
        cur.update(fields)
        return f"{label}/{name} configured"
    rec = {"namespace": ns, "name": name, "created": now(state)}
    rec.update(fields)
    lst.append(rec)
    return f"{label}/{name} created"


def with_sa_secrets(state, ns: str, spec: dict, previous=None) -> dict:
    spec = json_norm(spec)
    if previous is not None and previous.get("imagePullSecrets") and not spec.get("imagePullSecrets"):
        spec["imagePullSecrets"] = json_norm(previous.imagePullSecrets)
    if not spec.get("imagePullSecrets"):
        sa = (state.k8s.serviceaccounts.get(ns, {}) or {}).get(spec.get("serviceAccountName", "default")) or {}
        if sa.get("imagePullSecrets"):
            spec["imagePullSecrets"] = json_norm(sa["imagePullSecrets"])
    return spec


def _spec_diff(old, new):
    o = yaml.safe_dump(json_norm(old), default_flow_style=False).splitlines()
    n = yaml.safe_dump(json_norm(new), default_flow_style=False).splitlines()
    out = ["  core.PodSpec{"]
    for line in o:
        if line not in n:
            out.append("- " + line)
    for line in n:
        if line not in o:
            out.append("+ " + line)
    out.append("  }")
    return "\n".join(out)


def ctx_ns(ctx) -> str | None:
    return getattr(ctx, "_kns", None)


def delete_obj(state, kind: str, ns: str, name: str, opts: dict) -> str:
    k = state.k8s
    if kind == "pods":
        p = next((p for p in k.pods if p.namespace == ns and p.name == name), None)
        if p is None:
            raise CommandError(f'Error from server (NotFound): pods "{name}" not found', 1)
        if p.status == "Running":
            audit.record(state, audit.DELETE_RUNNING_POD, "bcm-head", "kubectl delete pod", f"{ns}/{name}")
        remove_pod(state, p)
        return f'pod "{name}" deleted'
    if kind == "deployments":
        d = next((d for d in k.deployments if d.namespace == ns and d.name == name), None)
        if d is None:
            raise CommandError(f'Error from server (NotFound): deployments.apps "{name}" not found', 1)
        k.deployments.remove(d)
        k.pods = [p for p in k.pods if not (p.get("owner") and p.owner.get("deployment") == name and p.namespace == ns)]
        return f'deployment.apps "{name}" deleted'
    if kind == "jobs":
        j = next((j for j in k.jobs if j.namespace == ns and j.name == name), None)
        if j is None:
            raise CommandError(f'Error from server (NotFound): jobs.batch "{name}" not found', 1)
        k.jobs.remove(j)
        k.pods = [p for p in k.pods if not (p.get("owner") and p.owner.get("kind") == "Job" and p.owner.name == name)]
        return f'job.batch "{name}" deleted'
    if kind == "daemonsets":
        d = next((d for d in k.daemonsets if d.namespace == ns and d.name == name), None)
        if d is None:
            raise CommandError(f'Error from server (NotFound): daemonsets.apps "{name}" not found', 1)
        if d.get("owner") == "clusterpolicy":
            return f'daemonset.apps "{name}" deleted'  # the operator recreates it immediately
        k.daemonsets.remove(d)
        return f'daemonset.apps "{name}" deleted'
    if kind == "namespaces":
        if name not in k.namespaces:
            raise CommandError(f'Error from server (NotFound): namespaces "{name}" not found', 1)
        if name in ("default", "kube-system"):
            raise CommandError(
                f'Error from server (Forbidden): namespaces "{name}" is forbidden: this namespace may not be deleted', 1
            )
        k.namespaces.remove(name)
        for lst in ("pods", "deployments", "jobs", "secrets", "configmaps", "daemonsets"):
            k[lst] = [o for o in k[lst] if o.get("namespace") != name]
        return f'namespace "{name}" deleted'
    if kind in ("secrets", "configmaps"):
        lst = k[kind]
        o = next((s for s in lst if s.namespace == ns and s.name == name), None)
        if o is None:
            raise CommandError(f'Error from server (NotFound): {kind} "{name}" not found', 1)
        lst.remove(o)
        return f'{SINGULAR[kind]} "{name}" deleted'
    if kind == "clusterpolicies":
        if not k.get("clusterpolicy") or k.clusterpolicy.name != name:
            raise CommandError(f'Error from server (NotFound): clusterpolicies.nvidia.com "{name}" not found', 1)
        k.clusterpolicy = None
        strip_operator_labels(state)
        return f'clusterpolicy.nvidia.com "{name}" deleted'
    if kind == "nicclusterpolicies":
        if not k.get("nicclusterpolicy"):
            raise CommandError(f'Error from server (NotFound): nicclusterpolicies.mellanox.com "{name}" not found', 1)
        k.nicclusterpolicy = None
        return f'nicclusterpolicy.mellanox.com "{name}" deleted'
    if kind == "runtimeclasses":
        k.runtimeclasses = [r for r in k.runtimeclasses if r.name != name]
        return f'runtimeclass.node.k8s.io "{name}" deleted'
    if kind in EXTRA_KINDS:
        return EXTRA_KINDS[kind].delete(state, name)
    if kind == "nodes":
        raise CommandError("error: deleting nodes is not supported in this lab (use cordon/drain)", 1)
    raise CommandError(f"error: deleting {kind} is not supported in this lab", 1)


def cmd_create(ctx, pos: list[str], opts: dict) -> str:
    state = ctx.state
    k = state.k8s
    if opts.get("filename"):
        out = [apply_doc(ctx, src, d, mode="create") for src, d in load_docs(ctx, opts)]
        return "\n".join(out) + "\n"
    if not pos:
        raise CommandError("error: must specify one of -f and -k", 1)
    what = pos[0]
    ns = ns_of(opts)
    if what in ("namespace", "ns"):
        name = pos[1]
        if name in k.namespaces:
            raise CommandError(f'Error from server (AlreadyExists): namespaces "{name}" already exists', 1)
        ensure_namespace(state, name)
        return f"namespace/{name} created\n"
    if ns not in k.namespaces:
        raise CommandError(f'error: failed to create {what}: namespaces "{ns}" not found', 1)
    if what == "secret":
        stype, name = pos[1], pos[2] if len(pos) > 2 else None
        if not name:
            raise CommandError("error: exactly one NAME is required, got 0", 1)
        if any(s.namespace == ns and s.name == name for s in k.secrets):
            raise CommandError(f'error: failed to create secret secrets "{name}" already exists', 1)
        if stype == "docker-registry":
            server = opts.get("docker-server", "https://index.docker.io/v1/")
            user = opts.get("docker-username")
            pw = opts.get("docker-password")
            how = ct.unquoted_oauthtoken(ctx)
            if user == ct.OAUTH_USER and how in ("unquoted", "double"):
                user = ""
            if not user or not pw:
                raise CommandError(
                    "error: either --from-file or the combination of --docker-username, --docker-password "
                    "and --docker-server is required",
                    1,
                )
            auth = base64.b64encode(f"{user}:{pw}".encode()).decode()
            ent = {"username": user, "password": pw, "auth": auth}
            if opts.get("docker-email"):
                ent["email"] = opts["docker-email"]
            data = {".dockerconfigjson": json.dumps({"auths": {server: ent}})}
            k.secrets.append(
                {
                    "namespace": ns,
                    "name": name,
                    "type": "kubernetes.io/dockerconfigjson",
                    "data": data,
                    "created": now(state),
                }
            )
            return f"secret/{name} created\n"
        if stype in ("generic", "tls"):
            data = {}
            for lit in opts.get("from-literal", []):
                a, _, b = lit.partition("=")
                data[a] = b
            for ff in opts.get("from-file", []):
                key, _, path = ff.rpartition("=")
                content = _read_manifest(ctx, path)
                data[key or path.rsplit("/", 1)[-1]] = content
            k.secrets.append(
                {"namespace": ns, "name": name, "type": opts.get("type", "Opaque"), "data": data, "created": now(state)}
            )
            return f"secret/{name} created\n"
        raise CommandError(f"error: unknown secret type {stype}", 1)
    if what in ("configmap", "cm"):
        name = pos[1]
        if any(c.namespace == ns and c.name == name for c in k.configmaps):
            raise CommandError(f'error: failed to create configmap: configmaps "{name}" already exists', 1)
        data = {}
        for lit in opts.get("from-literal", []):
            a, _, b = lit.partition("=")
            data[a] = b
        for ff in opts.get("from-file", []):
            key, _, path = ff.rpartition("=")
            data[key or path.rsplit("/", 1)[-1]] = _read_manifest(ctx, path)
        k.configmaps.append({"namespace": ns, "name": name, "data": data, "created": now(state)})
        return f"configmap/{name} created\n"
    if what in ("deployment", "deploy"):
        name = pos[1]
        image = opts.get("image")
        if not image:
            raise CommandError('error: required flag(s) "image" not set', 1)
        tmpl = {
            "metadata": {"labels": {"app": name}},
            "spec": {"containers": [{"name": image.rsplit("/", 1)[-1].split(":")[0], "image": image}]},
        }
        add_deployment(state, ns, name, image, replicas=int(opts.get("replicas", 1)), template=tmpl)
        return f"deployment.apps/{name} created\n"
    raise CommandError(f'error: unknown command "{what}" for "kubectl create"', 1)


def apply_yaml(state, text: str, namespace: str | None = None) -> list[str]:
    """Apply manifest text as if with `kubectl apply -f` (for scenario setup)."""
    k8s_init(state)
    ctx = SimpleNamespace(state=state, _kns=namespace)
    out = [apply_doc(ctx, "manifest.yaml", d) for d in yaml.safe_load_all(text) if d]
    reconcile(state)
    return out
