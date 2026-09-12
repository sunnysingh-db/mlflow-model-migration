"""Bootstrap loader for workspace_registry_migrator.

Solves the chicken-and-egg problem on GCP workspaces where a
org-policy (constraints/storage.restrictAuthTypes blocking
SERVICE_ACCOUNT_HMAC_SIGNED_REQUESTS) prevents WSFS FUSE from
reading .py file content.  Metadata ops (os.stat, os.listdir)
still work; only open() on file content fails.

Strategy:
  1. Test whether FUSE can read __init__.py (one byte).
  2. If yes  → add the workspace root to sys.path; done.
  3. If no   → download every .py file in the package via the
     Databricks REST API (WorkspaceClient.workspace.download)
     to /tmp/_ws_mig_pkg/ and import from there.

This file is loaded by the notebook’s Cell 6 loader, which itself
handles the case where even *this* file can’t be read via FUSE
(single-file REST fallback before calling ensure_package_importable).
"""

from __future__ import annotations

import os
import shutil
import sys


def ensure_package_importable(ws_root: str) -> None:
    """Make ``workspace_registry_migrator`` importable from *ws_root*.

    Parameters
    ----------
    ws_root : str
        Absolute ``/Workspace/…`` path to the project root directory
        (the parent of the ``workspace_registry_migrator/`` package).
    """
    pkg_dir = os.path.join(ws_root, "workspace_registry_migrator")
    init_file = os.path.join(pkg_dir, "__init__.py")

    # Purge cached modules so we always get a fresh import
    for mod_name in list(sys.modules):
        if mod_name.startswith("workspace_registry_migrator"):
            sys.modules.pop(mod_name)

    # ---- FUSE probe ----
    try:
        with open(init_file, "r") as fh:
            fh.read(1)
        # FUSE can serve file content — standard path
        if ws_root not in sys.path:
            sys.path.insert(0, ws_root)
        return
    except (OSError, IOError):
        pass  # FUSE broken — fall through to REST download

    # ---- REST API fallback ----
    from databricks.sdk import WorkspaceClient

    local_pkg = "/tmp/_ws_mig_pkg/workspace_registry_migrator"
    if os.path.isdir(local_pkg):
        shutil.rmtree(local_pkg)
    os.makedirs(local_pkg, exist_ok=True)

    w = WorkspaceClient()
    ws_pkg_path = pkg_dir.removeprefix("/Workspace")
    for obj in w.workspace.list(ws_pkg_path):
        if obj.path and obj.path.endswith(".py"):
            content = w.workspace.download(obj.path)
            local_file = os.path.join(local_pkg, os.path.basename(obj.path))
            with open(local_file, "wb") as fh:
                fh.write(content.read())

    if "/tmp/_ws_mig_pkg" not in sys.path:
        sys.path.insert(0, "/tmp/_ws_mig_pkg")
