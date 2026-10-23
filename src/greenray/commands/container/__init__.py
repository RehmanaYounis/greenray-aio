"""Container tooling on DGX / BCM nodes: docker, ngc, enroot, nvidia-ctk, crictl.

Registry model (nvcr.io):
  nvcr.io/nvidia/...         public NGC catalog (state.ngc.catalog). Anonymous pulls work.
  nvcr.io/<org>/[<team>/]..  private org/team registry (state.ngc.private_repos, set by
                             scenarios). Needs username ``$oauthtoken`` and a valid NGC API key.
An API key is valid if it is listed in ``state.ngc.valid_api_keys``. If a scenario does not
set that list, any key that looks like an NGC key (``nvapi-...``) is accepted.

Container runtime model:
  The *file* /etc/containerd/config.toml (or /etc/docker/daemon.json) is what you edit.
  ``node.runtime_loaded`` is what the running daemon actually uses. It is refreshed only
  when the daemon is (re)started, so editing the config without a restart does nothing."""

from __future__ import annotations

# Importing the modules runs their @command registrations and installs the hooks.
from . import crictl, ctk, docker, enroot, images, ngc, packages, runtime, sandbox  # noqa: F401
from .ctk import nvidia_ctk
from .docker import (
    COMPOSE_NAMES,
    DOCKER_VALUE_FLAGS,
    container_devices,
    container_logs,
    docker_loaded,
    start_container,
)
from .images import (
    DOCKER_HUB_REPOS,
    creds_valid,
    docker_creds,
    key_valid,
    norm_image,
    parse_image,
    pull_error_text,
    registry_check,
    unquoted_oauthtoken,
)
from .ngc import ngc_cli
from .packages import (
    APT_HEAD,
    BASE_PKGS,
    CTK_PKGS,
    CTK_REPO_LIST,
    CTK_REPO_URL,
    NVIDIA_CTR_CONFIG,
    apt,
    apt_cache,
    ctk_available,
    dpkg,
    installed_packages,
)
from .runtime import (
    CDI_SPEC,
    CONTAINERD_CONF,
    CTK_VERSION,
    DOCKER_AUTH,
    DOCKER_CONF,
    ENROOT_CREDS,
    EXTRA_CATALOG,
    IMAGE_SIZES,
    NGC_CONFIG,
    OAUTH_USER,
    container_init,
    containerd_config,
    containerd_default,
    containerd_has,
    go_json_error,
    parse_containerd,
    parse_docker,
    reload_runtime,
    runtime_init,
    toolkit_installed,
)
from .sandbox import exec_in_container, gpu_proxy_node

__all__ = [
    "APT_HEAD",
    "BASE_PKGS",
    "CDI_SPEC",
    "COMPOSE_NAMES",
    "CONTAINERD_CONF",
    "CTK_PKGS",
    "CTK_REPO_LIST",
    "CTK_REPO_URL",
    "CTK_VERSION",
    "DOCKER_AUTH",
    "DOCKER_CONF",
    "DOCKER_HUB_REPOS",
    "DOCKER_VALUE_FLAGS",
    "ENROOT_CREDS",
    "EXTRA_CATALOG",
    "IMAGE_SIZES",
    "NGC_CONFIG",
    "NVIDIA_CTR_CONFIG",
    "OAUTH_USER",
    "apt",
    "apt_cache",
    "container_devices",
    "container_init",
    "container_logs",
    "containerd_config",
    "containerd_default",
    "containerd_has",
    "creds_valid",
    "ctk_available",
    "docker_creds",
    "docker_loaded",
    "dpkg",
    "exec_in_container",
    "go_json_error",
    "gpu_proxy_node",
    "installed_packages",
    "key_valid",
    "ngc_cli",
    "norm_image",
    "nvidia_ctk",
    "parse_containerd",
    "parse_docker",
    "parse_image",
    "pull_error_text",
    "registry_check",
    "reload_runtime",
    "runtime_init",
    "start_container",
    "toolkit_installed",
    "unquoted_oauthtoken",
]
