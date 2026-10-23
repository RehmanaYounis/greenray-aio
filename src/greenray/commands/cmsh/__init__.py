"""NVIDIA Base Command Manager (BCM) cluster management shell: ``cmsh``.

Only the one-shot forms are supported because the lab shell is line based:

    cmsh -c "device; use dgx-01; set category dgx-h100; commit"
    cmsh -f /root/commands.cmsh

Modes: device, category, softwareimage, partition, user, group, profile, main.
Submodes: roles, bmcsettings, fsmounts (category/device).
Changes are *staged* in ``state.bcm.uncommitted`` until ``commit``, exactly as
in real cmsh (forgetting ``commit`` is the classic mistake). Staged changes are
kept for the whole lab session so that a later ``cmsh -c "device; commit"``
still finds them.

Software images live on the head node as files under ``/cm/images/<name>/``
(``bcm-head`` file entries with that prefix). ``imageupdate`` and
(re)provisioning copy those files and the image's package list to the node.

Also here: ``chroot`` / ``cm-chroot-sw-img`` into a software image and ``cmha``."""

from __future__ import annotations

# Importing the modules runs their @command registrations and installs the hooks.
from . import head, image, model, shell  # noqa: F401
from .head import cmha, getent_cmd, id_cmd
from .image import chroot
from .model import (
    BASE_PKGS,
    DEFAULT_EXCLUDE,
    DEFAULT_PROFILES,
    DGX_PKGS,
    EXCLUDE_LISTS,
    FSMOUNT_PROPS,
    INSTALL_MODES,
    KIND_COLL,
    KIND_TITLE,
    MODES,
    NOW,
    PROPS,
    READONLY,
    REPO,
    bcm_init,
    effective_bmc,
    effective_fsmounts,
    effective_image,
    effective_installmode,
    effective_roles,
    exclude_patterns,
    expand_nodes,
    image_files,
    is_excluded,
    node_packages,
    provision,
    sync_image,
)
from .shell import Cmsh, CmshError, cmsh

__all__ = [
    "BASE_PKGS",
    "Cmsh",
    "CmshError",
    "DEFAULT_EXCLUDE",
    "DEFAULT_PROFILES",
    "DGX_PKGS",
    "EXCLUDE_LISTS",
    "FSMOUNT_PROPS",
    "INSTALL_MODES",
    "KIND_COLL",
    "KIND_TITLE",
    "MODES",
    "NOW",
    "PROPS",
    "READONLY",
    "REPO",
    "bcm_init",
    "chroot",
    "cmha",
    "cmsh",
    "effective_bmc",
    "effective_fsmounts",
    "effective_image",
    "effective_installmode",
    "effective_roles",
    "exclude_patterns",
    "expand_nodes",
    "getent_cmd",
    "id_cmd",
    "image_files",
    "is_excluded",
    "node_packages",
    "provision",
    "sync_image",
]
