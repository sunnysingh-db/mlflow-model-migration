"""Notebook helpers for MLflow Workspace Registry Migration.

Single entry-point: ``execute_migration()`` handles discovery, migration,
verification, and clean consulting-grade output.
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Any


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def _setup_environment() -> None:
    warnings.filterwarnings("ignore")
    os.environ.update({
        "TQDM_DISABLE": "1",
        "MLFLOW_ENABLE_TQDM": "false",
        "MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR": "false",
    })


# ---------------------------------------------------------------------------
# Silent logger (captures framework noise)
# ---------------------------------------------------------------------------

class _CapturingLogger:
    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []
        self.show_warnings: bool = True

    def info(self, msg: str) -> None:
        self.entries.append(("INFO", msg))

    def warning(self, msg: str) -> None:
        self.entries.append(("WARN", msg))

    def error(self, msg: str) -> None:
        self.entries.append(("ERROR", msg))


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

class _R:
    W = 58

    @staticmethod
    def header(source: str, target: str, direction: str) -> None:
        print(f"\n{'\u2550' * _R.W}")
        print(f"  WORKSPACE REGISTRY MIGRATION")
        print(f"  {direction}")
        print(f"  Source: {source}")
        print(f"  Target: {target}")
        print(f"{'\u2550' * _R.W}")

    @staticmethod
    def phase(name: str) -> None:
        print(f"\n  {name}")
        print(f"  {'\u2500' * (_R.W - 4)}")

    @staticmethod
    def ok(name: str, detail: str = "") -> None:
        d = f"  {detail}" if detail else ""
        print(f"    \u2705 {name:<38}{d}")

    @staticmethod
    def fail(name: str, detail: str = "") -> None:
        d = f"  {detail}" if detail else ""
        print(f"    \u274c {name:<38}{d}")

    @staticmethod
    def warn(name: str, detail: str = "") -> None:
        d = f"  {detail}" if detail else ""
        print(f"    \u26a0\ufe0f  {name:<37}{d}")

    @staticmethod
    def note(text: str) -> None:
        print(f"    {text}")

    @staticmethod
    def footer(status: str, stats: dict[str, Any], duration: float) -> None:
        icon = ("\u2705" if status in ("COMPLETED", "EXPORT COMPLETE", "IMPORT COMPLETE")
                else "\u26a0\ufe0f " if status == "PARTIAL" else "\u274c")
        print(f"\n{'\u2550' * _R.W}")
        print(f"  SUMMARY")
        print(f"{'\u2550' * _R.W}")
        print(f"    Status:                {icon} {status}")
        for k, v in stats.items():
            print(f"    {k + ':':<23}{v}")
        print(f"    {'Duration:':<23}{duration:.1f}s")
        print(f"{'\u2550' * _R.W}\n")


_DIRECTIONS = {
    ("workspace", "workspace"): "Workspace \u2192 Workspace",
    ("workspace", "uc"): "Workspace \u2192 Unity Catalog",
    ("uc", "uc"): "Unity Catalog \u2192 Unity Catalog",
    ("uc", "workspace"): "Unity Catalog \u2192 Workspace",
}


# ---------------------------------------------------------------------------
# UC helpers
# ---------------------------------------------------------------------------

def _ensure_uc_target(catalog: str, schema: str) -> tuple[bool, str | None]:
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    try:
        w.catalogs.get(catalog)
    except Exception:
        try:
            w.catalogs.create(name=catalog)
        except Exception as e:
            return False, f"Cannot create catalog: {e}"
    try:
        w.schemas.get(f"{catalog}.{schema}")
    except Exception:
        try:
            w.schemas.create(name=schema, catalog_name=catalog)
        except Exception as e:
            return False, f"Cannot create schema: {e}"
    return True, None


def _get_target_credentials() -> tuple[str, str]:
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    host = w.config.host.rstrip("/")
    headers = w.config.authenticate()
    token = headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not token:
        raise RuntimeError("Cannot resolve target workspace token.")
    return host, token


# ---------------------------------------------------------------------------
# Export / Import mode handlers
# ---------------------------------------------------------------------------

def _execute_export(migrator, log, source_host, target_host, direction, t0, artifact_temp_dir):
    """Export mode: discover + download artifacts + write JSON manifests."""
    import time as _time
    _R.header(source_host, target_host, f"{direction} (EXPORT)")
    _R.phase("Export — Discovery + Artifact Download")
    _R.note(f"Bundle directory: {artifact_temp_dir}")

    bundle, manifest_path = migrator.export_bundle()

    for m in bundle.registered_models:
        n_ver = len(bundle.model_versions_by_name.get(m["name"], []))
        _R.ok(m["name"], f"{n_ver} versions exported")

    total_v = sum(len(v) for v in bundle.model_versions_by_name.values())
    total_r = sum(len(r) for r in bundle.runs_by_experiment_id.values())

    stats = {
        "Models exported": len(bundle.registered_models),
        "Versions exported": total_v,
        "Experiments": len(bundle.experiments),
        "Runs": total_r,
        "Bundle path": artifact_temp_dir,
    }
    _R.footer("EXPORT COMPLETE", stats, _time.time() - t0)

    # Show warnings
    warns = [msg for lvl, msg in log.entries if lvl in ("WARN", "ERROR")]
    if warns:
        print(f"\n  ⚠️  {len(warns)} warning(s) during export:")
        for w in warns[:10]:
            print(f"     • {w}")
    return None


def _execute_import(
    migrator, log, source_host, target_host, direction, t0,
    artifact_temp_dir, tracking_table, source_registry, target_registry,
    uc_target_catalog, uc_target_schema, model_name_prefix,
):
    """Import mode: read JSON manifests + artifacts, create on target."""
    import time as _time
    import pandas as pd
    import urllib.parse

    _R.header(source_host, target_host, f"{direction} (IMPORT)")
    _R.phase("Import — Loading Bundle")
    _R.note(f"Bundle directory: {artifact_temp_dir}")

    summary = migrator.import_bundle()

    _R.phase("Import Results")
    _R.ok("Models", str(summary.migrated_models))
    _R.ok("Versions", str(summary.migrated_model_versions))
    _R.ok("Experiments", str(summary.migrated_experiments))
    _R.ok("Runs", str(summary.migrated_runs))

    if summary.skipped_versions:
        for skip in summary.skipped_versions:
            _R.warn(
                f"{skip['model']} v{skip['version']}",
                skip.get('reason', 'unknown')[:60],
            )

    # Verification
    from workspace_registry_migrator.rest_client import DatabricksRestClient
    _tgt_host, _tgt_token = _get_target_credentials()
    target_rest = DatabricksRestClient(host=_tgt_host, token=_tgt_token)

    # Reload the bundle to get model names
    import json as _json
    bundle_dir = artifact_temp_dir
    models_dir = os.path.join(bundle_dir, "models")
    model_names_list: list[str] = []
    if os.path.isdir(models_dir):
        for model_folder in sorted(os.listdir(models_dir)):
            meta_path = os.path.join(models_dir, model_folder, "model_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as _f:
                    model_names_list.append(_json.load(_f)["name"])

    prefix = model_name_prefix

    def _target_name_for(source_name: str) -> str:
        if target_registry == "uc":
            parts = source_name.split(".")
            if len(parts) >= 3:
                cat = uc_target_catalog or parts[0]
                sch = uc_target_schema or parts[1]
                short = parts[2]
                return f"{cat}.{sch}.{prefix}{short}" if prefix else f"{cat}.{sch}.{short}"
            return source_name
        return f"{prefix}{source_name}" if prefix else source_name

    table_rows: list[dict] = []
    if model_names_list:
        _R.phase("Verification")
        for mname in model_names_list:
            tname = _target_name_for(mname)
            try:
                tgt_v = (target_rest.uc_search_all_model_versions(tname)
                         if target_registry == "uc"
                         else target_rest.search_all_model_versions(tname))
            except Exception:
                tgt_v = []
            t_ct = len([v for v in tgt_v if v.get("status") == "READY"])

            if target_registry == "uc":
                parts = tname.split(".")
                model_url = (f"{target_host.rstrip('/')}/explore/data/models/{parts[0]}/{parts[1]}/{parts[2]}"
                             if len(parts) >= 3 else "")
            else:
                model_url = f"{target_host.rstrip('/')}/#mlflow/models/{urllib.parse.quote(tname, safe='')}"

            _R.ok(f"{mname}", f"{t_ct} READY versions on target")
            table_rows.append({
                "Model": mname,
                "Target": tname,
                "Model URL": model_url,
                "Target Versions": t_ct,
                "Status": "✅ OK" if t_ct > 0 else "⚠️ CHECK",
            })

    stats = {
        "Models imported": summary.migrated_models,
        "Versions imported": summary.migrated_model_versions,
        "Experiments": summary.migrated_experiments,
        "Runs": summary.migrated_runs,
    }
    if summary.skipped_versions:
        stats["Skipped versions"] = len(summary.skipped_versions)
    _R.footer("IMPORT COMPLETE", stats, _time.time() - t0)

    warns = [msg for lvl, msg in log.entries if lvl in ("WARN", "ERROR")]
    if warns:
        print(f"\n  ⚠️  {len(warns)} warning(s) during import:")
        for w in warns[:10]:
            print(f"     • {w}")

    return pd.DataFrame(table_rows) if table_rows else None


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def execute_migration(
    source_host: str,
    source_token: str | None = None,
    auth_mode: str = "pat",
    sp_client_id: str | None = None,
    sp_client_secret: str | None = None,
    source_registry: str = "workspace",
    target_registry: str = "workspace",
    model_names: list[str] | None = None,
    include_catalogs: list[str] | None = None,
    exclude_catalogs: list[str] | None = None,
    exclude_schemas: list[str] | None = None,
    uc_target_catalog: str = "",
    uc_target_schema: str = "",
    model_name_prefix: str = "",
    tracking_table: str | None = None,
    include_artifacts: bool = True,
    include_deleted_runs: bool = False,
    create_dummy_versions: bool = True,
    batch_size: int = 10,
    artifact_temp_dir: str = "/tmp/workspace_registry_migration",
    migration_mode: str = "direct",
):
    _setup_environment()
    from workspace_registry_migrator import build_migrator
    from databricks.sdk import WorkspaceClient

    t0 = time.time()
    w = WorkspaceClient()
    target_host = w.config.host
    direction = _DIRECTIONS.get((source_registry, target_registry), f"{source_registry} > {target_registry}")

    if target_registry == "workspace" and not model_name_prefix:
        model_name_prefix = f"ws2ws_{int(time.time())}_"

    token = source_token if auth_mode == "pat" else None
    client_id = sp_client_id if auth_mode == "service_principal" else None
    client_secret = sp_client_secret if auth_mode == "service_principal" else None

    _R.header(source_host, target_host, direction)

    # Ensure staging directory exists
    os.makedirs(artifact_temp_dir, exist_ok=True)

    migrator = build_migrator(
        source_host=source_host,
        source_token=token,
        source_client_id=client_id,
        source_client_secret=client_secret,
        tracking_table=tracking_table,
        model_name_prefix=model_name_prefix,
        extra_model_names=model_names or [],
        extra_experiment_ids=[],
        source_registry=source_registry,
        target_registry=target_registry,
        uc_target_catalog=uc_target_catalog,
        uc_target_schema=uc_target_schema,
        create_dummy_versions=create_dummy_versions,
        include_uc_models=source_registry == "uc",
        include_catalogs=include_catalogs or [],
        exclude_catalogs=exclude_catalogs or [],
        exclude_schemas=exclude_schemas or [],
        shared_experiment_root="/Shared/mlflow-workspace-migration",
        artifact_temp_dir=artifact_temp_dir,
        batch_size=batch_size,
        max_workers=batch_size,
        download_workers=batch_size,
        register_workers=min(5, batch_size),
        download_artifacts=include_artifacts,
        migrate_experiments=True,
        migrate_registered_models=True,
        include_run_artifacts=include_artifacts,
        include_deleted_runs=include_deleted_runs,
        create_missing_experiments=True,
        skip_existing_model_versions=bool(tracking_table),
        stage_to_alias_mapping={"Production": "Champion", "Staging": "Challenger"},
    )

    log = _CapturingLogger()
    migrator.logger = log

    # ---- Mode routing ----
    if migration_mode not in ("direct", "export", "import"):
        raise ValueError(f"MIGRATION_MODE must be 'direct', 'export', or 'import' — got '{migration_mode}'")

    if migration_mode == "export":
        return _execute_export(migrator, log, source_host, target_host, direction, t0, artifact_temp_dir)

    if migration_mode == "import":
        return _execute_import(
            migrator, log, source_host, target_host, direction, t0,
            artifact_temp_dir, tracking_table, source_registry, target_registry,
            uc_target_catalog, uc_target_schema, model_name_prefix,
        )

    # ---- direct mode (original flow) ----

    # Phase 1: Discovery
    _R.phase("Phase 1 \u2014 Discovery")
    if source_registry == "uc" and not model_names:
        _R.note("UC bulk scan \u2014 discovering all visible models (this may take a moment)")
    bundle = migrator.discover()

    for m in bundle.registered_models:
        _R.ok(m["name"], f"{len(bundle.model_versions_by_name.get(m['name'], []))} versions")

    total_v = sum(len(v) for v in bundle.model_versions_by_name.values())
    total_r = sum(len(r) for r in bundle.runs_by_experiment_id.values())
    _R.note(f"Found: {len(bundle.registered_models)} models, {total_v} versions, "
            f"{len(bundle.experiments)} experiments, {total_r} runs")

    if not bundle.registered_models:
        _R.footer("EMPTY", {"Models found": 0}, time.time() - t0)
        return None

    if tracking_table:
        _R.phase("Tracking Table")
        persisted = migrator.persist_discovery_tracking(bundle)
        pending_model_names = migrator.get_pending_model_names()
        _R.ok("Discovery persisted", f"{persisted} models MERGED")
        _R.note(f"Tracking table: {tracking_table}")
        _R.note(f"Resumable models: {len(pending_model_names)}")
        if not pending_model_names:
            _R.footer(
                "COMPLETED",
                {
                    "Models found": len(bundle.registered_models),
                    "Pending models": 0,
                    "Tracking table": tracking_table,
                },
                time.time() - t0,
            )
            return None

        discovered_names = {m["name"] for m in bundle.registered_models}
        if set(pending_model_names) != discovered_names:
            bundle = migrator._discover_models(pending_model_names)
            _R.note(
                f"Continuing with {len(bundle.registered_models)} pending/failed models "
                f"from tracking table"
            )

    # UC target setup
    if target_registry == "uc":
        _R.phase("Target Setup")
        uc_targets: set[tuple[str, str]] = set()
        if uc_target_catalog and uc_target_schema:
            uc_targets.add((uc_target_catalog, uc_target_schema))
        elif source_registry == "uc":
            for m in bundle.registered_models:
                parts = m["name"].split(".")
                if len(parts) >= 3:
                    uc_targets.add((parts[0], parts[1]))
        for cat, sch in sorted(uc_targets):
            ok, err = _ensure_uc_target(cat, sch)
            _R.ok(f"{cat}.{sch}", "ready") if ok else _R.fail(f"{cat}.{sch}", err or "denied")

    # Phase 2: Migration (live output)
    _R.phase("Phase 2 \u2014 Migration")

    # 2a: Experiments (one-shot)
    _log_before_exp = len(log.entries)
    experiment_name_map = migrator._migrate_experiments(bundle)
    _exp_log = log.entries[_log_before_exp:]
    _exp_run_errors = sum(1 for lvl, _ in _exp_log if lvl in ("WARN", "ERROR"))
    _runs_migrated = total_r - _exp_run_errors

    for exp in bundle.experiments:
        runs = bundle.runs_by_experiment_id.get(exp["experiment_id"], [])
        name = exp.get("name", exp["experiment_id"]).split("/")[-1]
        _R.ok(f"Experiment: {name}", f"{len(runs)} runs migrated")
    if _exp_run_errors:
        _R.warn("Run warnings", f"{_exp_run_errors} run(s) had issues during cloning")

    print(f"  {'\u2500' * (_R.W - 4)}")

    # 2b: Models (one at a time, live)
    from workspace_registry_migrator import DiscoveryBundle
    import urllib.parse
    prefix = model_name_prefix

    def _target_name_for(source_name: str) -> str:
        """Compute the target model name accounting for UC catalog/schema remapping."""
        if target_registry == "uc":
            parts = source_name.split(".")
            if len(parts) >= 3:
                cat = uc_target_catalog or parts[0]
                sch = uc_target_schema or parts[1]
                short = parts[2]
                return f"{cat}.{sch}.{prefix}{short}" if prefix else f"{cat}.{sch}.{short}"
            return source_name
        return f"{prefix}{source_name}" if prefix else source_name

    # Reverse lookup: run_id -> source experiment id
    _run_to_exp_id: dict[str, str] = {}
    for _eid, _runs in bundle.runs_by_experiment_id.items():
        for _r in _runs:
            _rid = _r.get("info", {}).get("run_id") or _r.get("run_id", "")
            if _rid:
                _run_to_exp_id[_rid] = _eid
    _exp_id_to_name: dict[str, str] = {
        e["experiment_id"]: e.get("name", e["experiment_id"]).split("/")[-1]
        for e in bundle.experiments
    }

    totals = {"models": 0, "versions": 0, "runs": 0}
    all_skipped: list[Any] = []
    migration_rows: list[dict[str, Any]] = []

    for m in bundle.registered_models:
        mname = m["name"]
        versions = bundle.model_versions_by_name.get(mname, [])
        target_name = _target_name_for(mname)

        # Single-model bundle (experiments already migrated above)
        mini = DiscoveryBundle(
            registered_models=[m],
            model_versions_by_name={mname: versions},
            experiments=bundle.experiments,
            runs_by_experiment_id=bundle.runs_by_experiment_id,
            compat_report=bundle.compat_report,
        )

        row: dict[str, Any] = {
            "_target_name": target_name,
            "Model": mname,
            "Versions": 0,
            "Status": "",
            "Verified": "",
            "Comments": "",
            "_exp_ids": set(),
        }

        skipped_before = len(getattr(migrator, "_skipped_versions", []))
        log_before = len(log.entries)
        try:
            counts = migrator._migrate_models(mini, experiment_name_map)
        except Exception as exc:
            _R.fail(f"Model: {mname}", str(exc)[:60])
            row["Status"] = "\u274c FAILED"
            row["Comments"] = str(exc)[:200]
            migration_rows.append(row)
            continue
        new_skipped = getattr(migrator, "_skipped_versions", [])[skipped_before:]

        # Capture framework warnings for this model
        new_logs = log.entries[log_before:]
        log_warnings = [msg for lvl, msg in new_logs if lvl in ("WARN", "ERROR")]

        n_ok = len(versions) - len(new_skipped)
        row["Versions"] = n_ok
        if new_skipped:
            _R.warn(f"Model: {mname}", f"{n_ok}/{len(versions)} versions migrated")
            all_skipped.extend(new_skipped)
            row["Status"] = "\u26a0\ufe0f PARTIAL"
            row["Comments"] = f"{len(new_skipped)} version(s) skipped"
        else:
            _R.ok(f"Model: {mname}", f"{len(versions)} versions migrated")
            row["Status"] = "\u2705 OK"

        if log_warnings:
            prev = row["Comments"]
            row["Comments"] = "; ".join(filter(None, [prev] + log_warnings[:3]))

        # Resolve which experiments this model belongs to
        for v in versions:
            rid = v.get("run_id", "")
            eid = _run_to_exp_id.get(rid, "")
            if eid:
                row["_exp_ids"].add(eid)

        totals["models"] += counts.get("models", 0)
        totals["versions"] += counts.get("versions", 0)
        totals["runs"] += counts.get("runs", 0)
        migration_rows.append(row)

    # Phase 3: Verification
    from workspace_registry_migrator.rest_client import DatabricksRestClient
    _tgt_host, _tgt_token = _get_target_credentials()
    target_rest = DatabricksRestClient(host=_tgt_host, token=_tgt_token)
    effective_names = model_names or [m["name"] for m in bundle.registered_models]

    if effective_names:
        _R.phase("Phase 3 \u2014 Verification")
        for mname in effective_names:
            tname = _target_name_for(mname)
            try:
                src_v = (migrator.source_rest.uc_search_all_model_versions(mname)
                         if source_registry == "uc"
                         else migrator.source_rest.search_all_model_versions(mname))
            except Exception:
                src_v = []
            try:
                tgt_v = (target_rest.uc_search_all_model_versions(tname)
                         if target_registry == "uc"
                         else target_rest.search_all_model_versions(tname))
            except Exception:
                tgt_v = []
            s_ct = len([v for v in src_v if v.get("status") == "READY"])
            t_ct = len([v for v in tgt_v if v.get("status") == "READY"])

            # Update the row
            for row in migration_rows:
                if row["Model"] == mname:
                    if s_ct == t_ct:
                        row["Verified"] = f"\u2705 [{s_ct}] \u2192 [{t_ct}]"
                    else:
                        row["Verified"] = f"\u274c [{s_ct}] \u2192 [{t_ct}]"
                        if not row["Comments"]:
                            row["Comments"] = f"source {s_ct} READY vs target {t_ct}"
                    break

            if s_ct == t_ct:
                _R.ok(f"Model: {mname}", f"[{s_ct}] \u2192 [{t_ct}] verified")
            else:
                _R.fail(f"Model: {mname}", f"[{s_ct}] \u2192 [{t_ct}] mismatch")

    # Summary
    n_migrated = totals["models"]
    _ver_errors = sum(1 for r in migration_rows if "FAILED" in r.get("Status", ""))
    _total_errors = _exp_run_errors + len(all_skipped) + _ver_errors
    status = ("COMPLETED" if n_migrated > 0 and _total_errors == 0
              else "PARTIAL" if n_migrated > 0 else "FAILED")
    stats: dict[str, Any] = {
        "Models migrated": f"{n_migrated} / {len(bundle.registered_models)}",
        "Versions migrated": f"{totals['versions']} / {total_v}",
        "Experiments migrated": f"{len(experiment_name_map)} / {len(bundle.experiments)}",
        "Runs migrated": f"{_runs_migrated} / {total_r}" if _exp_run_errors else str(total_r),
    }
    if _total_errors:
        stats["Errors"] = f"{_total_errors} (see table)"
    _R.footer(status, stats, time.time() - t0)

    # ── Build migration summary table ──────────────────────────────
    import pandas as pd
    import mlflow as _mlflow

    host = target_host.rstrip("/")

    # Cache target experiment id lookups
    _tgt_exp_cache: dict[str, str] = {}  # target_name -> target_exp_id
    for src_eid, tgt_ename in experiment_name_map.items():
        try:
            _e = _mlflow.get_experiment_by_name(tgt_ename)
            if _e:
                _tgt_exp_cache[src_eid] = _e.experiment_id
        except Exception:
            pass

    table_rows: list[dict[str, str]] = []
    for row in migration_rows:
        mname = row["Model"]
        tname = row["_target_name"]

        # Model URL
        if target_registry == "uc":
            parts = tname.split(".")
            model_url = (f"{host}/explore/data/models/{parts[0]}/{parts[1]}/{parts[2]}"
                         if len(parts) >= 3 else "")
        else:
            model_url = f"{host}/#mlflow/models/{urllib.parse.quote(tname, safe='')}"

        # Experiment names + URLs
        exp_names: list[str] = []
        exp_urls: list[str] = []
        for eid in sorted(row.get("_exp_ids", set())):
            exp_names.append(_exp_id_to_name.get(eid, eid))
            tgt_eid = _tgt_exp_cache.get(eid, "")
            exp_urls.append(f"{host}/#mlflow/experiments/{tgt_eid}" if tgt_eid else "")

        table_rows.append({
            "Model": mname,
            "Model URL": model_url,
            "Experiment(s)": ", ".join(exp_names) or "\u2014",
            "Experiment URL(s)": ", ".join(u for u in exp_urls if u) or "\u2014",
            "Versions Migrated": row["Versions"],
            "Status": row["Status"],
            "Verified": row["Verified"],
            "Comments": row["Comments"] or "\u2014",
        })

    return pd.DataFrame(table_rows) if table_rows else None


def _is_excluded(name: str, excl_cats: set[str], excl_schs: set[str]) -> bool:
    parts = name.split(".")
    if len(parts) >= 3:
        if parts[0].lower() in excl_cats:
            return True
        if f"{parts[0]}.{parts[1]}".lower() in excl_schs:
            return True
    return False


def _verify(migrator: Any, model_names: list[str], prefix: str,
            source_registry: str, target_registry: str) -> None:
    from workspace_registry_migrator.rest_client import DatabricksRestClient
    target_host, target_token = _get_target_credentials()
    target_rest = DatabricksRestClient(host=target_host, token=target_token)

    for name in model_names:
        if target_registry == "uc":
            parts = name.split(".")
            if len(parts) >= 3:
                cat = parts[0]
                sch = parts[1]
                short = parts[2]
                target_name = f"{cat}.{sch}.{prefix}{short}" if prefix else name
            else:
                target_name = name
        else:
            target_name = f"{prefix}{name}" if prefix else name
        try:
            src = (migrator.source_rest.uc_search_all_model_versions(name)
                   if source_registry == "uc"
                   else migrator.source_rest.search_all_model_versions(name))
        except Exception:
            src = []
        try:
            tgt = (target_rest.uc_search_all_model_versions(target_name)
                   if target_registry == "uc"
                   else target_rest.search_all_model_versions(target_name))
        except Exception:
            tgt = []
        s = len([v for v in src if v.get("status") == "READY"])
        t = len([v for v in tgt if v.get("status") == "READY"])
        if s == t:
            _R.ok(name, f"[{s}] \u2192 [{t}] verified")
        else:
            _R.fail(name, f"[{s}] \u2192 [{t}] mismatch")


# Keep backward compat for existing imports
def create_migrator(
    source_host: str,
    source_token: str | None = None,
    auth_mode: str = "pat",
    sp_client_id: str | None = None,
    sp_client_secret: str | None = None,
    model_names: list[str] | None = None,
    experiment_ids: list[str] | None = None,
    target_registry: str = "workspace",
    uc_target_catalog: str = "",
    uc_target_schema: str = "",
    model_name_prefix: str = "",
    tracking_table: str | None = None,
    include_artifacts: bool = True,
    include_deleted_runs: bool = False,
    batch_size: int = 10,
    max_workers: int = 10,
):
    """Create a fully configured migrator from user-friendly inputs.

    Returns a ``WorkspaceRegistryMigrator`` ready for discovery / migration.
    """
    from workspace_registry_migrator import build_migrator
    from databricks.sdk import WorkspaceClient

    # Auto-prefix to avoid name collisions
    if not model_name_prefix:
        model_name_prefix = f"ws2ws_{int(time.time())}_"

    # Resolve auth mode
    token = source_token if auth_mode == "pat" else None
    client_id = sp_client_id if auth_mode == "service_principal" else None
    client_secret = sp_client_secret if auth_mode == "service_principal" else None

    migrator = build_migrator(
        source_host=source_host,
        source_token=token,
        source_client_id=client_id,
        source_client_secret=client_secret,
        tracking_table=tracking_table,
        model_name_prefix=model_name_prefix,
        extra_model_names=model_names or [],
        extra_experiment_ids=experiment_ids or [],
        source_registry="workspace",
        target_registry=target_registry,
        uc_target_catalog=uc_target_catalog,
        uc_target_schema=uc_target_schema,
        include_uc_models=False,
        shared_experiment_root="/Shared/mlflow-workspace-migration",
        batch_size=batch_size,
        max_workers=max_workers,
        download_workers=max_workers,
        register_workers=min(5, max_workers),
        download_artifacts=include_artifacts,
        migrate_experiments=True,
        migrate_registered_models=True,
        include_run_artifacts=include_artifacts,
        include_deleted_runs=include_deleted_runs,
        create_missing_experiments=True,
        skip_existing_model_versions=False,
        stage_to_alias_mapping={"Production": "Champion", "Staging": "Challenger"},
    )
    migrator.logger.show_warnings = True

    # Summary
    w = WorkspaceClient()
    target_host = w.config.host
    tgt_label = (
        f"{uc_target_catalog}.{uc_target_schema}"
        if target_registry == "uc"
        else "workspace registry"
    )

    print("\u2705 Migrator ready")
    print(f"   Source:     {source_host}")
    print(f"   Target:     {target_host} \u2192 {tgt_label}")
    print(f"   Auth:       {auth_mode}")
    print(f"   Prefix:     {model_name_prefix}")
    print(f"   Models:     {len(model_names or [])} requested")
    if include_artifacts:
        print("   Artifacts:  included")
    if tracking_table:
        print(f"   Tracking:   {tracking_table}")

    if source_host.rstrip("/") == target_host.rstrip("/"):
        print(
            "\n\u26a0\ufe0f  Source and target are the SAME workspace. "
            "Replace SOURCE_HOST / SOURCE_TOKEN for a real migration."
        )

    return migrator


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def preview_migration(migrator):
    """Discover source assets and print a human-friendly summary.

    Returns the ``DiscoveryBundle``.
    """
    print("\U0001f50d Discovering source assets...\n")
    bundle = migrator.discover()

    total_versions = sum(len(v) for v in bundle.model_versions_by_name.values())
    total_runs = sum(len(r) for r in bundle.runs_by_experiment_id.values())

    print(f"{'\u2500' * 50}")
    print(f"  Registered models:  {len(bundle.registered_models)}")
    print(f"  Model versions:     {total_versions}")
    print(f"  Experiments:        {len(bundle.experiments)}")
    print(f"  Runs:               {total_runs}")
    print(f"{'\u2500' * 50}")

    if bundle.compat_report and getattr(bundle.compat_report, "warnings", None):
        print(f"\n  \u26a0\ufe0f  {bundle.compat_report.summary()}")

    if bundle.registered_models:
        print("\n  Models found:")
        for m in bundle.registered_models:
            versions = bundle.model_versions_by_name.get(m["name"], [])
            stages = [v.get("current_stage", "None") for v in versions]
            print(f"    \u2022 {m['name']}: {len(versions)} versions {stages}")

    return bundle


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def run_migration(migrator):
    """Execute the bulk migration and print results.

    Returns the ``MigrationSummary``.
    """
    print("\U0001f680 Starting migration...\n")
    t0 = time.time()

    result = migrator.migrate_all()

    elapsed = time.time() - t0
    print(f"\n{'\u2500' * 50}")
    print(f"  \u2705 Migration Complete ({elapsed:.1f}s)")
    print(f"{'\u2500' * 50}")
    print(f"  Models migrated:    {result.migrated_models}")
    print(f"  Versions migrated:  {result.migrated_model_versions}")
    print(f"  Experiments cloned: {result.migrated_experiments}")
    print(f"  Runs cloned:        {result.migrated_runs}")

    if result.skipped_versions:
        print(f"\n  \u26a0\ufe0f  Skipped: {len(result.skipped_versions)}")
        for s in result.skipped_versions[:10]:
            print(f"    \u2022 {s['model']} v{s['version']}: {s['reason']}")
        if len(result.skipped_versions) > 10:
            print(f"    ... and {len(result.skipped_versions) - 10} more")

    print(f"{'\u2500' * 50}")
    return result


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _get_target_credentials() -> tuple[str, str]:
    """Resolve current workspace host + bearer token from notebook context."""
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    host = w.config.host.rstrip("/")
    headers = w.config.authenticate()
    auth_header = headers.get("Authorization", "")
    token = (
        auth_header.removeprefix("Bearer ").strip() if auth_header else ""
    )
    if not token:
        raise RuntimeError(
            "Could not resolve target workspace token. "
            "Ensure this runs inside a Databricks notebook."
        )
    return host, token


def verify_migration(
    migrator,
    model_names: list[str],
    model_name_prefix: str | None = None,
):
    """Compare source vs target versions to confirm the migration succeeded.

    Checks version counts, ordering (via ``source_model_version`` tags),
    and tag completeness for each model.

    Returns ``(all_passed, details_list)``.
    """
    from workspace_registry_migrator.rest_client import DatabricksRestClient

    prefix = model_name_prefix or migrator.options.model_name_prefix
    target_host, target_token = _get_target_credentials()
    target_rest = DatabricksRestClient(host=target_host, token=target_token)
    source_rest = migrator.source_rest

    print(f"\U0001f50e Verifying {len(model_names)} models...\n")
    all_pass = True
    details: list[dict[str, Any]] = []

    for model_name in model_names:
        target_name = f"{prefix}{model_name}"

        # Source versions
        try:
            src_raw = source_rest.search_all_model_versions(model_name)
        except Exception as exc:
            print(f"  \u26a0\ufe0f  Could not read source {model_name}: {exc}")
            src_raw = []
        src_ready = sorted(
            [v for v in src_raw if v.get("status") == "READY"],
            key=lambda v: int(v.get("version", 0)),
        )

        # Target versions
        try:
            tgt_raw = target_rest.search_all_model_versions(target_name)
        except Exception:
            tgt_raw = []
        tgt_ready = sorted(
            [v for v in tgt_raw if v.get("status") == "READY"],
            key=lambda v: int(v.get("version", 0)),
        )

        # Checks
        count_ok = len(src_ready) == len(tgt_ready)

        tgt_src_map: dict[str, str] = {}
        for tv in tgt_ready:
            tags = {t["key"]: t["value"] for t in tv.get("tags", [])}
            tgt_src_map[tv["version"]] = tags.get("source_model_version", "?")

        order_ok = (
            all(
                str(tgt_src_map.get(tgt_ready[i]["version"], "?"))
                == str(src_ready[i].get("version"))
                for i in range(min(len(src_ready), len(tgt_ready)))
            )
            if tgt_ready
            else False
        )
        tags_ok = "?" not in tgt_src_map.values() if tgt_src_map else False
        passed = count_ok and order_ok and tags_ok

        if not passed:
            all_pass = False

        src_stages = [v.get("current_stage", "None") for v in src_ready]
        tgt_stages = [v.get("current_stage", "None") for v in tgt_ready]

        emoji = "\u2705" if passed else "\u274c"
        print(f"  {emoji} {model_name} \u2192 {target_name}")
        print(f"     Source: {[v['version'] for v in src_ready]}  stages={src_stages}")
        print(f"     Target: {[v['version'] for v in tgt_ready]}  stages={tgt_stages}")
        if not passed:
            print(f"     Count:{count_ok}  Order:{order_ok}  Tags:{tags_ok}")

        details.append(
            {
                "model": model_name,
                "target": target_name,
                "status": "PASS" if passed else "FAIL",
                "source_versions": len(src_ready),
                "target_versions": len(tgt_ready),
            }
        )

    print(f"\n{'\u2500' * 50}")
    if all_pass:
        print(f"  \U0001f389 ALL {len(model_names)} MODELS VERIFIED")
    else:
        n_pass = sum(1 for d in details if d["status"] == "PASS")
        print(f"  {n_pass}/{len(model_names)} passed, {len(model_names) - n_pass} failed")
    print(f"{'\u2500' * 50}")

    return all_pass, details
