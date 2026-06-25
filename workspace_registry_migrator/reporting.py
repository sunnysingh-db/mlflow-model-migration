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


def _parse_model_metadata(run_id: str) -> dict[str, str | None]:
    """Extract flavor and requirements from a run's model artifacts."""
    result: dict[str, str | None] = {"flavors": None, "requirements": None, "python_version": None}
    if not run_id:
        return result
    with tempfile.TemporaryDirectory() as tmp:
        try:
            path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="model/MLmodel",
                dst_path=tmp, tracking_uri="databricks",
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
                dst_path=tmp, tracking_uri="databricks",
            )
            with open(path) as f:
                deps = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            result["requirements"] = "; ".join(deps)
        except Exception:
            pass
    return result


def generate_inventory_report(inventory: DiscoveryBundle) -> pd.DataFrame:
    """Generate a model-level pre-migration inventory DataFrame.

    One row per registered model with aggregated metadata:
    readiness, versions, stages, owners, flavors, requirements, experiments.
    """
    client = _get_client()
    exp_lookup = {e.experiment_id: e for e in inventory.experiments}
    runs_count_by_exp = {eid: len(runs) for eid, runs in inventory.runs_by_experiment_id.items()}

    rows: list[dict[str, Any]] = []
    for model in inventory.registered_models:
        accessible_versions = inventory.model_versions_by_name.get(model.name, [])
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

                meta = _parse_model_metadata(v.run_id)
                if meta["flavors"]:
                    flavors_seen.add(meta["flavors"])
                    representative_meta = meta
                if meta["python_version"]:
                    python_versions_seen.add(meta["python_version"])
            except Exception:
                versions_blocked += 1

        total_runs = sum(runs_count_by_exp.get(eid, 0) for eid in experiments)

        if versions_ready == 0:
            readiness = "\u274c BLOCKED \u2014 all runs deleted"
        elif versions_blocked > 0:
            readiness = f"\u26a0\ufe0f PARTIAL \u2014 {versions_ready}/{len(all_versions)} versions migratable"
        else:
            readiness = "\u2705 READY"

        rows.append({
            "model_name": model.name,
            "readiness": readiness,
            "total_versions": len(all_versions),
            "versions_migratable": versions_ready,
            "versions_blocked": versions_blocked,
            "stages": ", ".join(sorted(stages)) or "None",
            "owner_emails": ", ".join(sorted(users)),
            "flavors": " | ".join(sorted(flavors_seen)) if flavors_seen else None,
            "requirements": representative_meta["requirements"],
            "python_version": ", ".join(sorted(python_versions_seen)) if python_versions_seen else None,
            "experiments": " | ".join(sorted(experiment_names)),
            "num_experiments": len(experiments),
            "total_runs": total_runs,
            "latest_version_created": latest_created.strftime("%Y-%m-%d %H:%M") if latest_created else "",
            "created": datetime.fromtimestamp(model.creation_timestamp / 1000).strftime("%Y-%m-%d") if model.creation_timestamp else "",
        })

    return pd.DataFrame(rows)


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
    ready = len(df[df["readiness"].str.startswith("\u2705")])
    blocked = len(df[df["readiness"].str.startswith("\u274c")])
    partial = len(df[df["readiness"].str.startswith("\u26a0")])
    total_versions = int(df["total_versions"].sum())
    migratable = int(df["versions_migratable"].sum())
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
