"""Reporting utilities for MLflow workspace registry migration."""

from __future__ import annotations

import tempfile
from datetime import datetime
from typing import Any

import mlflow
import pandas as pd
import yaml
from mlflow import MlflowClient

from workspace_registry_migrator.framework import DiscoveryBundle, MigrationOptions


def _get_client() -> MlflowClient:
    return MlflowClient(tracking_uri="databricks", registry_uri="databricks")


def _parse_model_metadata(run_id: str, tracking_uri: str = "databricks") -> dict[str, str | None]:
    """Extract flavor and requirements from a run's model artifacts."""
    result: dict[str, str | None] = {"flavors": None, "requirements": None, "python_version": None}
    if not run_id:
        return result
    with tempfile.TemporaryDirectory() as tmp:
        try:
            path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="model/MLmodel",
                dst_path=tmp, tracking_uri=tracking_uri,
            )
            with open(path) as f:
                mlmodel = yaml.safe_load(f)
            result["flavors"] = ", ".join(mlmodel.get("flavors", {}).keys())
            pf = mlmodel.get("flavors", {}).get("python_function", {})
            result["python_version"] = pf.get("python_version")
        except Exception:
            pass
        try:
            path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="model/requirements.txt",
                dst_path=tmp, tracking_uri=tracking_uri,
            )
            with open(path) as f:
                deps = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            result["requirements"] = "; ".join(deps)
        except Exception:
            pass
    return result


def generate_inventory_report(
    inventory: DiscoveryBundle,
    source_host: str = "",
    target_host: str = "",
    source_context=None,
    include_metadata: bool = True,
    max_workers: int = 20,
) -> pd.DataFrame:
    """Generate a model-level pre-migration inventory DataFrame.

    One row per registered model with aggregated metadata.
    Includes source_host and target_host for multi-workspace tracking.

    Args:
        source_context: SourceWorkspaceContext pointing to the SOURCE workspace.
                        If None, falls back to local _get_client() (same-workspace mode).
        include_metadata: If True, downloads MLmodel + requirements.txt per version
                          to extract flavors/requirements. Set False for fast discovery
                          (~10x faster for large registries, skips artifact downloads).
        max_workers: Max parallel threads for processing models concurrently.
    """
    # Apply source environment so all MLflow calls target the correct workspace
    _prev_env = None
    if source_context:
        _prev_env = source_context._apply_env()
        client = source_context._client()
        source_tracking_uri = source_context.credentials.resolved_tracking_uri()
    else:
        client = _get_client()
        source_tracking_uri = "databricks"
    try:
        return _generate_inventory_rows(
            inventory, source_host, target_host, client,
            source_tracking_uri, include_metadata, max_workers,
        )
    finally:
        if source_context and _prev_env is not None:
            source_context._restore_env(_prev_env)


def _process_single_model(
    model,
    inventory: DiscoveryBundle,
    source_host: str,
    target_host: str,
    client: MlflowClient,
    source_tracking_uri: str,
    include_metadata: bool,
    exp_lookup: dict,
    runs_count_by_exp: dict,
) -> dict[str, Any]:
    """Process a single model and return its inventory row dict."""
    all_versions = list(client.search_model_versions(
        filter_string=f"name='{model.name.replace(chr(39), chr(39) + chr(39))}'"
    ))

    users: set[str] = set()
    experiments: set[str] = set()
    experiment_names: set[str] = set()
    stages: set[str] = set()
    versions_ready = 0
    versions_blocked = 0
    flavors_seen: set[str] = set()
    python_versions_seen: set[str] = set()
    latest_created: datetime | None = None
    representative_meta: dict[str, str | None] = {"flavors": None, "requirements": None, "python_version": None}
    discovery_comments_parts: list[str] = []
    total_params = 0
    total_metrics = 0
    total_artifacts = 0

    for v in all_versions:
        if v.creation_timestamp:
            ts = datetime.fromtimestamp(v.creation_timestamp / 1000)
            if latest_created is None or ts > latest_created:
                latest_created = ts
        if v.current_stage and v.current_stage != "None":
            stages.add(v.current_stage)
        if v.user_id:
            users.add(v.user_id)

        try:
            run = client.get_run(v.run_id)
            user_email = run.data.tags.get("mlflow.user", v.user_id or "")
            if user_email:
                users.add(user_email)
            exp_id = run.info.experiment_id
            experiments.add(exp_id)
            exp = exp_lookup.get(exp_id)
            if exp:
                experiment_names.add(exp.name)
            else:
                try:
                    experiment_names.add(client.get_experiment(exp_id).name)
                except Exception:
                    experiment_names.add(f"(id: {exp_id})")
            versions_ready += 1
            total_params += len(run.data.params)
            total_metrics += len(run.data.metrics)
            try:
                total_artifacts += len(client.list_artifacts(run_id=v.run_id, path="model"))
            except Exception:
                pass

            if include_metadata:
                meta = _parse_model_metadata(v.run_id, tracking_uri=source_tracking_uri)
                if meta["flavors"]:
                    flavors_seen.add(meta["flavors"])
                    representative_meta = meta
                if meta["python_version"]:
                    python_versions_seen.add(meta["python_version"])
        except Exception as exc:
            versions_blocked += 1
            discovery_comments_parts.append(f"v{v.version}: {str(exc)[:80]}")

    total_runs = sum(runs_count_by_exp.get(eid, 0) for eid in experiments)

    if versions_ready == 0:
        readiness = "\u274c BLOCKED \u2014 all runs deleted"
    elif versions_blocked > 0:
        readiness = f"\u26a0\ufe0f PARTIAL \u2014 {versions_ready}/{len(all_versions)} versions migratable"
    else:
        readiness = "\u2705 READY"

    return {
        "source_host": source_host,
        "target_host": target_host,
        "model_name": model.name,
        "readiness": readiness,
        "source_versions": len(all_versions),
        "source_versions_migratable": versions_ready,
        "source_versions_blocked": versions_blocked,
        "source_experiments": len(experiments),
        "source_runs": total_runs,
        "source_params": total_params,
        "source_metrics": total_metrics,
        "source_artifacts": total_artifacts,
        "stages": ", ".join(sorted(stages)) or "None",
        "owner_emails": ", ".join(sorted(users)),
        "flavors": " | ".join(sorted(flavors_seen)) if flavors_seen else None,
        "requirements": representative_meta["requirements"],
        "python_version": ", ".join(sorted(python_versions_seen)) if python_versions_seen else None,
        "experiments": " | ".join(sorted(experiment_names)),
        "latest_version_created": latest_created.strftime("%Y-%m-%d %H:%M") if latest_created else "",
        "created": datetime.fromtimestamp(model.creation_timestamp / 1000).strftime("%Y-%m-%d") if model.creation_timestamp else "",
        # Target columns — populated during migration
        "target_versions": 0,
        "target_runs": 0,
        "target_params": 0,
        "target_metrics": 0,
        "target_artifacts": 0,
        "migration_status": "PENDING",
        "discovery_comments": "; ".join(discovery_comments_parts) if discovery_comments_parts else None,
        "migration_comments": None,
        "target_model_url": None,
        "target_experiment_urls": None,
        "last_updated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _generate_inventory_rows(
    inventory: DiscoveryBundle,
    source_host: str,
    target_host: str,
    client: MlflowClient,
    source_tracking_uri: str,
    include_metadata: bool = True,
    max_workers: int = 20,
) -> pd.DataFrame:
    """Build inventory DataFrame with parallel per-model processing."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    exp_lookup = {e.experiment_id: e for e in inventory.experiments}
    runs_count_by_exp = {eid: len(runs) for eid, runs in inventory.runs_by_experiment_id.items()}
    models = inventory.registered_models

    if not models:
        return pd.DataFrame([])

    rows: list[dict[str, Any]] = []
    num_workers = min(max_workers, len(models))

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        future_map = {
            pool.submit(
                _process_single_model,
                model, inventory, source_host, target_host, client,
                source_tracking_uri, include_metadata, exp_lookup, runs_count_by_exp,
            ): model.name
            for model in models
        }
        completed = 0
        for future in as_completed(future_map):
            model_name = future_map[future]
            try:
                row = future.result()
                rows.append(row)
            except Exception as exc:
                rows.append({
                    "source_host": source_host,
                    "target_host": target_host,
                    "model_name": model_name,
                    "readiness": f"\u274c ERROR \u2014 {str(exc)[:60]}",
                    "source_versions": 0,
                    "source_versions_migratable": 0,
                    "source_versions_blocked": 0,
                    "source_experiments": 0,
                    "source_runs": 0,
                    "source_params": 0,
                    "source_metrics": 0,
                    "source_artifacts": 0,
                    "stages": "None",
                    "owner_emails": "",
                    "flavors": None,
                    "requirements": None,
                    "python_version": None,
                    "experiments": "",
                    "latest_version_created": "",
                    "created": "",
                    "target_versions": 0,
                    "target_runs": 0,
                    "target_params": 0,
                    "target_metrics": 0,
                    "target_artifacts": 0,
                    "migration_status": "PENDING",
                    "discovery_comments": f"Inventory error: {str(exc)[:120]}",
                    "migration_comments": None,
                    "target_model_url": None,
                    "target_experiment_urls": None,
                    "last_updated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                })
            completed += 1
            if completed % 50 == 0 or completed == len(models):
                print(f"  [inventory] {completed}/{len(models)} models processed")

    return pd.DataFrame(rows)


def write_inventory_to_delta(
    inventory_df: pd.DataFrame,
    tracking_table: str,
) -> None:
    """MERGE inventory DataFrame into the Delta tracking table.

    Creates the table if it doesn't exist. Uses (source_host, model_name) as key.
    Only updates source-side columns; preserves target-side and migration_comments from prior migrations.
    """
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import current_timestamp

    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("No active SparkSession")

    sdf = spark.createDataFrame(inventory_df)
    sdf = sdf.withColumn("last_updated_at", current_timestamp())
    sdf.createOrReplaceTempView("_inventory_staging")

    # Create table if not exists
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {tracking_table} (
            source_host STRING,
            target_host STRING,
            model_name STRING,
            readiness STRING,
            source_versions INT,
            source_versions_migratable INT,
            source_versions_blocked INT,
            source_experiments INT,
            source_runs INT,
            source_params INT,
            source_metrics INT,
            source_artifacts INT,
            stages STRING,
            owner_emails STRING,
            flavors STRING,
            requirements STRING,
            python_version STRING,
            experiments STRING,
            latest_version_created STRING,
            created STRING,
            target_versions INT,
            target_runs INT,
            target_params INT,
            target_metrics INT,
            target_artifacts INT,
            migration_status STRING,
            discovery_comments STRING,
            migration_comments STRING,
            target_model_url STRING,
            target_experiment_urls STRING,
            last_updated_at TIMESTAMP
        )
        USING DELTA
    """)

    spark.sql(f"""
        MERGE INTO {tracking_table} AS target
        USING _inventory_staging AS source
        ON target.source_host = source.source_host AND target.model_name = source.model_name
        WHEN MATCHED THEN UPDATE SET
            target.target_host = source.target_host,
            target.readiness = source.readiness,
            target.source_versions = source.source_versions,
            target.source_versions_migratable = source.source_versions_migratable,
            target.source_versions_blocked = source.source_versions_blocked,
            target.source_experiments = source.source_experiments,
            target.source_runs = source.source_runs,
            target.source_params = source.source_params,
            target.source_metrics = source.source_metrics,
            target.source_artifacts = source.source_artifacts,
            target.stages = source.stages,
            target.owner_emails = source.owner_emails,
            target.flavors = source.flavors,
            target.requirements = source.requirements,
            target.python_version = source.python_version,
            target.experiments = source.experiments,
            target.latest_version_created = source.latest_version_created,
            target.created = source.created,
            target.discovery_comments = source.discovery_comments,
            target.last_updated_at = current_timestamp()
        WHEN NOT MATCHED THEN INSERT *
    """)
    spark.sql("DROP VIEW IF EXISTS _inventory_staging")
    print(f"\u2705 Inventory MERGED into {tracking_table} ({len(inventory_df)} models)")


def update_tracking_after_model(
    tracking_table: str,
    source_host: str,
    model_name: str,
    target_versions: int,
    target_runs: int,
    target_params: int,
    target_metrics: int,
    target_artifacts: int,
    migration_status: str,
    migration_comments: str | None = None,
    target_model_url: str | None = None,
    target_experiment_urls: str | None = None,
) -> None:
    """Update target-side columns in the tracking table after a model is migrated.

    Called by the framework after each model completes migration.
    Only writes to migration_comments — never touches discovery_comments.
    """
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession()
    if spark is None:
        return  # silently skip if no Spark (e.g., unit tests)

    escaped_host = source_host.replace("'", "''")
    escaped_name = model_name.replace("'", "''")
    comments_sql = f"'{migration_comments.replace(chr(39), chr(39)+chr(39))}'" if migration_comments else "NULL"
    model_url_sql = f"'{target_model_url.replace(chr(39), chr(39)+chr(39))}'" if target_model_url else "NULL"
    exp_urls_sql = f"'{target_experiment_urls.replace(chr(39), chr(39)+chr(39))}'" if target_experiment_urls else "NULL"

    spark.sql(f"""
        UPDATE {tracking_table}
        SET
            target_versions = {target_versions},
            target_runs = {target_runs},
            target_params = {target_params},
            target_metrics = {target_metrics},
            target_artifacts = {target_artifacts},
            migration_status = '{migration_status}',
            migration_comments = {comments_sql},
            target_model_url = {model_url_sql},
            target_experiment_urls = {exp_urls_sql},
            last_updated_at = current_timestamp()
        WHERE source_host = '{escaped_host}' AND model_name = '{escaped_name}'
    """)


def generate_migration_report(
    inventory: DiscoveryBundle,
    options: MigrationOptions,
    skipped_versions: list[dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Generate a post-migration comparison report.

    Compares source vs target for each model version: params, metrics, artifacts.
    """
    client = _get_client()
    prefix = options.model_name_prefix
    report_rows: list[dict[str, Any]] = []

    for source_model_name in [m.name for m in inventory.registered_models]:
        target_model_name = f"{prefix}{source_model_name}"
        escaped_src = source_model_name.replace("'", "''")
        escaped_tgt = target_model_name.replace("'", "''")

        source_versions = list(client.search_model_versions(filter_string=f"name='{escaped_src}'"))
        target_versions = list(client.search_model_versions(filter_string=f"name='{escaped_tgt}'"))

        target_by_source_v = {
            (tv.tags or {}).get("source_model_version"): tv
            for tv in target_versions
        }

        for sv in source_versions:
            tv = target_by_source_v.get(str(sv.version))

            src_run_ok = False
            src_params = src_metrics = src_artifacts = 0
            try:
                src_run = client.get_run(sv.run_id)
                src_run_ok = True
                src_params = len(src_run.data.params)
                src_metrics = len(src_run.data.metrics)
                try:
                    src_artifacts = len(client.list_artifacts(run_id=sv.run_id, path="model"))
                except Exception:
                    pass
            except Exception:
                pass

            tgt_params = tgt_metrics = tgt_artifacts = 0
            params_match = metrics_match = artifacts_match = False
            if tv and tv.run_id:
                try:
                    tgt_run = client.get_run(tv.run_id)
                    tgt_params = len(tgt_run.data.params)
                    tgt_metrics = len(tgt_run.data.metrics)
                    params_match = tgt_params == src_params
                    metrics_match = tgt_metrics == src_metrics
                    try:
                        tgt_artifacts = len(client.list_artifacts(run_id=tv.run_id, path="model"))
                        artifacts_match = tgt_artifacts == src_artifacts
                    except Exception:
                        pass
                except Exception:
                    pass

            if tv:
                status = "\u2705 MIGRATED"
            elif not src_run_ok:
                status = "\u274c RUN_DELETED"
            elif src_artifacts == 0:
                status = "\u274c NO_ARTIFACTS"
            else:
                status = "\u26a0\ufe0f FAILED"

            report_rows.append({
                "source_model": source_model_name,
                "version": sv.version,
                "target_model": target_model_name,
                "status": status,
                "source_run_accessible": src_run_ok,
                "source_params": src_params,
                "target_params": tgt_params if tv else None,
                "params_match": params_match if tv else None,
                "source_metrics": src_metrics,
                "target_metrics": tgt_metrics if tv else None,
                "metrics_match": metrics_match if tv else None,
                "source_model_files": src_artifacts,
                "target_model_files": tgt_artifacts if tv else None,
                "artifacts_match": artifacts_match if tv else None,
            })

    return pd.DataFrame(report_rows)


def print_inventory_summary(df: pd.DataFrame) -> None:
    """Print a concise summary header for the inventory report."""
    if df.empty:
        print("\u26a0\ufe0f  No models discovered \u2014 check credentials and source connectivity.")
        return
    ready = len(df[df["readiness"].str.startswith("\u2705")])
    blocked = len(df[df["readiness"].str.startswith("\u274c")])
    partial = len(df[df["readiness"].str.startswith("\u26a0")])
    total_versions = int(df["source_versions"].sum())
    migratable = int(df["source_versions_migratable"].sum())
    owners = len(set().union(*[set(r.split(", ")) for r in df["owner_emails"] if r]))

    print(f"PRE-MIGRATION INVENTORY: {len(df)} models, {total_versions} versions ({migratable} migratable)")
    print(f"  \u2705 Ready: {ready}  |  \u26a0\ufe0f Partial: {partial}  |  \u274c Blocked: {blocked}")
    print(f"  Unique owners: {owners}")
    print()


def print_migration_summary(df: pd.DataFrame, skipped: list[dict[str, str]] | None = None) -> None:
    """Print a concise summary header for the migration report."""
    total = len(df)
    migrated = len(df[df["status"] == "\u2705 MIGRATED"])
    failed = len(df[df["status"] == "\u26a0\ufe0f FAILED"])
    deleted = len(df[df["status"] == "\u274c RUN_DELETED"])
    no_arts = len(df[df["status"] == "\u274c NO_ARTIFACTS"])

    print(f"MIGRATION REPORT: {migrated}/{total} versions migrated successfully")
    print(f"  \u2705 Migrated: {migrated}  |  \u26a0\ufe0f Failed: {failed}  |  \u274c Run deleted: {deleted}  |  \u274c No artifacts: {no_arts}")
    print()

    if skipped:
        print(f"SKIPPED VERSIONS ({len(skipped)} failures):")
        from IPython.display import display as _display
        _display(pd.DataFrame(skipped))
        print()
