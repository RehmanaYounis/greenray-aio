"""Image references, the nvcr.io registry model and credentials."""

from __future__ import annotations

import base64
import json
import re

from .runtime import DOCKER_AUTH, OAUTH_USER, container_init


def parse_image(ref: str):
    """-> (registry, repo, tag). docker.io short names get library/."""
    ref = ref.split("@", 1)[0]
    parts = ref.split("/")
    if len(parts) > 1 and ("." in parts[0] or ":" in parts[0] or parts[0] == "localhost"):
        registry, rest = parts[0], "/".join(parts[1:])
    else:
        registry, rest = "docker.io", ref
    tag = "latest"
    last = rest.rsplit("/", 1)[-1]
    if ":" in last:
        rest, tag = rest.rsplit(":", 1)
    if registry == "docker.io" and "/" not in rest:
        rest = "library/" + rest
    return registry, rest, tag


def norm_image(ref: str) -> str:
    reg, repo, tag = parse_image(ref)
    if reg == "docker.io":
        repo = repo.removeprefix("library/")
        return f"{repo}:{tag}"
    return f"{reg}/{repo}:{tag}"


def key_valid(state, key: str) -> bool:
    if not key or key.startswith("$"):
        return False
    keys = state.ngc.get("valid_api_keys") or []
    if keys:
        return key in keys
    return bool(re.fullmatch(r"nvapi-[A-Za-z0-9_-]{20,}", key) or re.fullmatch(r"[A-Za-z0-9+/=]{60,}", key))


def creds_valid(state, user: str, password: str) -> bool:
    return user == OAUTH_USER and key_valid(state, password)


# Public Docker Hub repositories the lab knows; anything else is "repository does not exist", as for real.
DOCKER_HUB_REPOS = frozenset(
    {
        "library/ubuntu",
        "library/busybox",
        "library/alpine",
        "library/python",
        "library/nginx",
        "library/hello-world",
        "library/debian",
        "nvidia/cuda",
        "pytorch/pytorch",
        "tensorflow/tensorflow",
        "rocm/pytorch",
        "jupyter/base-notebook",
        "bitnami/kubectl",
    }
)


def registry_check(state, ref: str, creds=None):
    """Return None if the image can be pulled with ``creds`` (user, password) else an error kind."""
    container_init(state)
    reg, repo, tag = parse_image(ref)
    if reg == "docker.io":
        return None if repo in DOCKER_HUB_REPOS else "not_found"
    if reg != "nvcr.io":
        return None
    authed = bool(creds) and creds_valid(state, *creds)
    if repo in state.ngc.catalog:
        return None if tag in state.ngc.catalog[repo] else "manifest_unknown"
    if repo in state.ngc.private_repos:
        if not authed:
            return "unauthorized"
        return None if tag in state.ngc.private_repos[repo] else "manifest_unknown"
    return "not_found" if authed else "unauthorized"


def docker_creds(node, registry="nvcr.io"):
    try:
        auths = json.loads(node.files.get(DOCKER_AUTH, "") or "{}").get("auths", {})
    except ValueError:
        return None
    for key, ent in auths.items():
        if key.replace("https://", "").rstrip("/") == registry and ent.get("auth"):
            user, _, pw = base64.b64decode(ent["auth"]).decode().partition(":")
            return user, pw
    return None


def unquoted_oauthtoken(ctx) -> str | None:
    """Mimic bash: was ``$oauthtoken`` on the raw line left unprotected by single quotes?

    The lab shell keeps unknown ``$vars`` literally, but real bash expands an unset
    ``$oauthtoken`` to an empty string unless it is in single quotes or escaped.
    Returns 'unquoted', 'double' or None.
    """
    line = ctx.history[-1] if ctx.history else ""
    i, quote = 0, None
    while i < len(line):
        ch = line[i]
        if ch == "\\" and quote != "'":
            i += 2
            continue
        if quote:
            if ch == quote:
                quote = None
            elif ch == "$" and quote == '"' and line.startswith("$oauthtoken", i):
                return "double"
        elif ch in "'\"":
            quote = ch
        elif ch == "$" and line.startswith("$oauthtoken", i):
            return "unquoted"
        i += 1
    return None


def pull_error_text(kind, ref, tool="docker"):
    reg, repo, tag = parse_image(ref)
    if tool == "docker":
        return {
            "unauthorized": f'Error response from daemon: Head "https://{reg}/v2/{repo}/manifests/{tag}": unauthorized: authentication required',
            "not_found": f"Error response from daemon: pull access denied for {reg}/{repo}, repository does not exist or may require 'docker login': denied: requested access to the resource is denied",
            "manifest_unknown": f"Error response from daemon: manifest for {norm_image(ref)} not found: manifest unknown: manifest unknown",
        }[kind]
    # containerd / kubelet wording
    full = norm_image(ref) if reg != "docker.io" else f"docker.io/{repo}:{tag}"
    base = f'failed to pull and unpack image "{full}": failed to resolve reference "{full}"'
    return {
        "unauthorized": f"{base}: pulling from host {reg} failed with status code [manifests {tag}]: 401 Unauthorized",
        "not_found": f"{base}: {full}: not found",
        "manifest_unknown": f"{base}: {full}: not found",
    }[kind]
