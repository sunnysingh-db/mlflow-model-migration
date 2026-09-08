"""MLflow Model Migration Framework v2.

Key improvements over v1:
 - REST-based metadata operations (no env var mutation, fully thread-safe)
 - Two-phase pipeline: download artifacts first, then register
 - Sequential per-model version registration (preserves version ordering)
 - MLflow version compatibility checking
 - Support for both workspace and UC registry targets
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlflow import MlflowClient
from mlflow.entities import Metric, Param, RunTag

from workspace_registry_migrator.rest_client import (
    DatabricksRestClient,
    SourceArtifactDownloader,
    TargetArtifactUploader,
    CompatReport,
    check_compatibility,
)
from workspace_registry_migrator.utils import NotebookLogger, chunked, sanitize_name

logger = logging.getLogger("workspace_registry_migrator.framework_v2")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceWorkspaceCredentials:
    """Connection details for the source Databricks workspace."""
    host: str
    token: str | None = None
    tracking_uri: str | None = None
    registry_uri: str = "databricks"
    workspace_label: str = "source"
    client_id: str | None = None
    client_secret: str | None = None

    def normalized_host(self) -> str:
        return self.host.rstrip("/")

    def resolved_tracking_uri(self) -> str:
        return self.tracking_uri or "databricks"

    def auth_type(self) -> str:
        if self.token:
            return "pat"
        if self.client_id and self.client_secret:
            return "oauth-m2m"
        raise ValueError("Provide either a PAT token or a service principal client_id/client_secret")


@dataclass(frozen=True)
class MigrationOptions:
    """Runtime controls for workspace registry migration."""
    shared_experiment_root: str = "/Shared/mlflow-workspace-migration"
    model_name_prefix: str = ""
    experiment_name_prefix: str = ""
    batch_size: int = 10
    max_workers: int = 10
    download_workers: int = 20
    register_workers: int = 5
    download_artifacts: bool = True
    migrate_experiments: bool = True
    migrate_registered_models: bool = True
    include_run_artifacts: bool = True
    include_deleted_runs: bool = False
    max_runs_per_experiment: int | None = None
    max_model_versions_per_model: int | None = None
    create_missing_experiments: bool = True
    skip_existing_model_versions: bool = False
    artifact_temp_dir: str = "/tmp/workspace_registry_migration"
    extra_model_names: list[str] = field(default_factory=list)
    extra_experiment_ids: list[str] = field(default_factory=list)
    # UC migration options
    source_registry: str = "workspace"  # "workspace" or "uc" — where to read models FROM
    target_registry: str = "workspace"  # "workspace" or "uc" — where to write models TO
    uc_target_catalog: str = ""
    uc_target_schema: str = ""
    include_uc_models: bool = False
    include_catalogs: list[str] = field(default_factory=list)   # only scan these catalogs (empty = all)
    exclude_catalogs: list[str] = field(default_factory=list)   # skip these catalogs during bulk scan
    exclude_schemas: list[str] = field(default_factory=list)    # skip these schemas ("catalog.schema" format)
    stage_to_alias_mapping: dict[str, str] = field(default_factory=lambda: {
        "Production": "Champion", "Staging": "Challenger",
    })
    create_dummy_versions: bool = False  # create placeholder versions for deleted source runs

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if self.max_workers > 64:
            raise ValueError("max_workers must be 64 or less")
        if self.target_registry == "uc" and self.source_registry != "uc" and (not self.uc_target_catalog or not self.uc_target_schema):
            raise ValueError(
                "uc_target_catalog and uc_target_schema required when "
                "target_registry='uc' and source is not UC "
                "(UC→UC can mirror source catalog.schema automatically)"
            )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiscoveryBundle:
    """Collected source workspace registry assets."""
    registered_models: list[dict]
    model_versions_by_name: dict[str, list[dict]]
    experiments: list[dict]
    runs_by_experiment_id: dict[str, list[dict]]
    compat_report: CompatReport | None = None


@dataclass(frozen=True)
class MigrationSummary:
    """High-level migration results."""
    migrated_models: int
    migrated_model_versions: int
    migrated_experiments: int
    migrated_runs: int
    skipped_versions: list[dict] = field(default_factory=list)


@dataclass
class StagedVersion:
    """A model version whose artifacts have been downloaded to local staging."""
    model_name: str
    version: dict
    source_run: dict
    local_artifact_dir: str
    artifact_subpath: str
    experiment_id: str
    is_placeholder: bool = False


# ---------------------------------------------------------------------------
# Migrator
# ---------------------------------------------------------------------------

class WorkspaceRegistryMigrator:
    """Bulk migrator for Databricks workspace MLflow registry assets.

    v2 architecture:
    - REST-based metadata (no env var mutation)
    - Two-phase pipeline (download -> register)
    - Sequential per-model version registration
    - MLflow version compatibility checking
    """

    def __init__(
        self,
        source_credentials: SourceWorkspaceCredentials,
        options: MigrationOptions,
        logger: NotebookLogger | None = None,
        tracking_table: str | None = None,
    ) -> None:
        options.validate()
        self._configure_runtime_noise()
        self.source_credentials = source_credentials
        self.options = options
        self.logger = logger or NotebookLogger()
        self.tracking_table = tracking_table
        self._skipped_versions: list[dict[str, str]] = []

        # Source REST client (remote workspace — explicit auth required)
        self.source_rest = DatabricksRestClient(
            host=source_credentials.normalized_host(),
            token=source_credentials.token,
            client_id=source_credentials.client_id,
            client_secret=source_credentials.client_secret,
            pool_connections=options.download_workers,
            pool_maxsize=options.download_workers,
        )

        # Target SDK clients (current workspace — auto-authenticates via notebook context)
        _registry_uri = "databricks-uc" if options.target_registry == "uc" else "databricks"
        self._target_client = MlflowClient()  # tracking (experiments, runs)
        self._target_model_client = MlflowClient(registry_uri=_registry_uri)  # model registry

        # Artifact handlers
        self.source_downloader = SourceArtifactDownloader(
            host=source_credentials.normalized_host(),
            token=source_credentials.token,
            client_id=source_credentials.client_id,
            client_secret=source_credentials.client_secret,
        )
        self.target_uploader = TargetArtifactUploader()

        # Target workspace info
        self._target_host = self._get_target_host()

    @staticmethod
    def _get_target_host() -> str:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        return w.config.host

    @staticmethod
    def _configure_runtime_noise() -> None:
        import warnings
        logging.getLogger("mlflow").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")

    @staticmethod
    def _tags_to_dict(tags: list[dict]) -> dict[str, str]:
        """Convert REST-style [{key, value}] tags to SDK-style {key: value} dict."""
        return {t["key"]: t["value"] for t in tags if isinstance(t, dict) and "key" in t}

    # ---- Discovery ----

    def discover(self) -> DiscoveryBundle:
        """Discover source workspace assets without migrating."""
        compat = check_compatibility(self.source_rest, self.source_rest)  # target uses SDK; source-only compat check
        if compat.warnings:
            for w in compat.warnings:
                self.logger.info(f"[COMPAT] {w}")

        registered_models = self._list_registered_models()
        model_versions_by_name: dict[str, list[dict]] = {}

        if registered_models:
            # UC APIs have stricter rate limits — cap concurrency to 2 for UC sources
            _vlist_workers = min(
                2 if self._source_is_uc else self.options.max_workers,
                len(registered_models),
            )
            with ThreadPoolExecutor(max_workers=_vlist_workers) as pool:
                future_map: dict = {}
                for i, m in enumerate(registered_models):
                    if i > 0 and self._source_is_uc:
                        time.sleep(0.5)  # stagger submissions to avoid burst 429s
                    future_map[pool.submit(self._list_model_versions, m["name"])] = m["name"]
                for future in as_completed(future_map):
                    name = future_map[future]
                    try:
                        model_versions_by_name[name] = future.result()
                    except Exception as exc:
                        self.logger.warning(f"Failed to list versions for {name}: {exc}")
                        model_versions_by_name[name] = []

        experiments = self._list_experiments(registered_models, model_versions_by_name)
        runs_by_experiment_id: dict[str, list[dict]] = {}

        if experiments:
            with ThreadPoolExecutor(max_workers=min(self.options.max_workers, len(experiments))) as pool:
                future_map = {
                    pool.submit(self._list_runs, e["experiment_id"]): e["experiment_id"]
                    for e in experiments
                }
                for future in as_completed(future_map):
                    eid = future_map[future]
                    try:
                        runs_by_experiment_id[eid] = future.result()
                    except Exception as exc:
                        self.logger.warning(f"Failed to list runs for experiment {eid}: {exc}")
                        runs_by_experiment_id[eid] = []

        return DiscoveryBundle(
            registered_models=registered_models,
            model_versions_by_name=model_versions_by_name,
            experiments=experiments,
            runs_by_experiment_id=runs_by_experiment_id,
            compat_report=compat,
        )

    @property
    def _source_is_uc(self) -> bool:
        return self.options.source_registry == "uc"

    def _list_registered_models(self) -> list[dict]:
        requested = {name.strip() for name in self.options.extra_model_names}
        if requested:
            # Targeted search: fetch only requested models (avoids scanning all models)
            models: list[dict] = []
            for name in sorted(requested):
                try:
                    escaped = name.replace("'", "''")
                    if self._source_is_uc:
                        # UC has no search-by-name for models; use get instead
                        try:
                            model_data = self.source_rest.uc_get_registered_model(name)
                            rm = model_data.get("registered_model", model_data)
                            if "name" in rm:
                                models.append(rm)
                        except Exception:
                            self.logger.warning(f"UC model not found: {name}")
                    else:
                        data = self.source_rest.search_registered_models(
                            max_results=5, filter_string=f"name='{escaped}'",
                        )
                        models.extend(data.get("registered_models", []))
                except Exception as exc:
                    self.logger.warning(f"Could not find model '{name}': {exc}")
        else:
            if self._source_is_uc:
                models = self._list_all_uc_models()
            else:
                models = self.source_rest.search_all_registered_models()
        if not self._source_is_uc and not self.options.include_uc_models:
            models = [m for m in models if "." not in m["name"]]
        if requested:
            models = [m for m in models if m["name"].strip() in requested]
        for m in models:
            self.logger.info(f"Discovered model {m['name']}")
        return models

    def _list_all_uc_models(self) -> list[dict]:
        """Enumerate UC models by scanning catalogs → schemas → models in parallel.

        Creates a source-workspace SDK client from source credentials and
        iterates every visible catalog/schema concurrently, collecting
        models from each.  Respects ``include_catalogs`` (whitelist) and
        ``exclude_catalogs`` / ``exclude_schemas`` (blacklist) on
        :class:`MigrationOptions` so users can scope the scan.
        """
        from databricks.sdk import WorkspaceClient as _WC

        creds = self.source_credentials
        sdk_kwargs: dict[str, Any] = {"host": creds.normalized_host()}
        if creds.token:
            sdk_kwargs["token"] = creds.token
        elif creds.client_id and creds.client_secret:
            sdk_kwargs["client_id"] = creds.client_id
            sdk_kwargs["client_secret"] = creds.client_secret

        try:
            source_sdk = _WC(**sdk_kwargs)
        except Exception as exc:
            self.logger.warning(f"SDK client init failed, falling back to REST: {exc}")
            return self.source_rest.search_all_registered_models()

        _SKIP_SCHEMAS = {"information_schema"}
        _SKIP_CATALOGS = {"system", "__databricks_internal"}

        incl = {c.lower() for c in self.options.include_catalogs}
        excl_cats = {c.lower() for c in self.options.exclude_catalogs} | _SKIP_CATALOGS
        excl_schs = {s.lower() for s in self.options.exclude_schemas}  # "catalog.schema"

        # Step 1: list catalogs
        try:
            catalogs = [c for c in source_sdk.catalogs.list()
                        if c.name.lower() not in excl_cats
                        and (not incl or c.name.lower() in incl)]
        except Exception as exc:
            self.logger.warning(f"Failed to list catalogs: {exc}")
            return []
        self.logger.info(f"UC scan: {len(catalogs)} catalogs")

        # Step 2: list schemas per catalog (parallel)
        catalog_schemas: list[tuple[str, str]] = []

        def _list_schemas(cat_name: str) -> list[tuple[str, str]]:
            try:
                return [
                    (cat_name, s.name)
                    for s in source_sdk.schemas.list(catalog_name=cat_name)
                    if s.name not in _SKIP_SCHEMAS
                    and f"{cat_name}.{s.name}".lower() not in excl_schs
                ]
            except Exception:
                return []

        _schema_workers = min(self.options.max_workers, max(len(catalogs), 1))
        with ThreadPoolExecutor(max_workers=_schema_workers) as pool:
            for result in pool.map(_list_schemas, [c.name for c in catalogs]):
                catalog_schemas.extend(result)
        self.logger.info(f"UC scan: {len(catalog_schemas)} schemas")

        # Step 3: list models per schema (parallel)
        models: list[dict] = []
        _lock = __import__("threading").Lock()

        def _list_models_in_schema(cat_sch: tuple[str, str]) -> None:
            cat, sch = cat_sch
            try:
                for m in source_sdk.registered_models.list(
                    catalog_name=cat, schema_name=sch,
                ):
                    with _lock:
                        models.append({"name": m.full_name})
            except Exception:
                pass

        _model_workers = min(self.options.max_workers, max(len(catalog_schemas), 1))
        with ThreadPoolExecutor(max_workers=_model_workers) as pool:
            list(pool.map(_list_models_in_schema, catalog_schemas))

        self.logger.info(f"UC full scan discovered {len(models)} models")
        return models

    def _list_model_versions(self, model_name: str) -> list[dict]:
        if self._source_is_uc:
            versions = self.source_rest.uc_search_all_model_versions(model_name)
        else:
            versions = self.source_rest.search_all_model_versions(model_name)
        if self.options.max_model_versions_per_model is not None:
            versions = versions[:self.options.max_model_versions_per_model]
        return versions

    def _list_experiments(
        self,
        registered_models: list[dict],
        model_versions_by_name: dict[str, list[dict]],
    ) -> list[dict]:
        if not self.options.migrate_experiments:
            return []
        experiment_ids: set[str] = set(self.options.extra_experiment_ids)
        for model in registered_models:
            for version in model_versions_by_name.get(model["name"], []):
                run_id = version.get("run_id")
                if not run_id:
                    continue
                try:
                    run_data = self.source_rest.get_run(run_id)
                    eid = run_data.get("run", {}).get("info", {}).get("experiment_id")
                    if eid:
                        experiment_ids.add(eid)
                except Exception as exc:
                    self.logger.warning(f"Skipping version {model['name']} v{version.get('version')} — run unavailable: {exc}")
        experiments: list[dict] = []
        for eid in sorted(experiment_ids):
            try:
                data = self.source_rest.get_experiment(eid)
                exp = data.get("experiment", data)
                experiments.append(exp)
            except Exception as exc:
                self.logger.warning(f"Skipping experiment {eid}: {exc}")
        self.logger.info(f"Discovered {len(experiments)} source experiments")
        return experiments

    def _list_runs(self, experiment_id: str) -> list[dict]:
        return self.source_rest.search_all_runs(
            experiment_ids=[experiment_id],
            include_deleted=self.options.include_deleted_runs,
            max_runs=self.options.max_runs_per_experiment,
        )

    # ---- Migration: Two-Phase Pipeline ----

    def migrate_all(self) -> MigrationSummary:
        """Full migration: discover + migrate."""
        bundle = self.discover()
        return self.migrate_from_bundle(bundle)

    def migrate_from_bundle(self, bundle: DiscoveryBundle) -> MigrationSummary:
        """Migrate from a pre-discovered bundle (avoids re-scanning)."""
        experiment_name_map = self._migrate_experiments(bundle)
        model_counts = self._migrate_models(bundle, experiment_name_map)
        return MigrationSummary(
            migrated_models=model_counts["models"],
            migrated_model_versions=model_counts["versions"],
            migrated_experiments=len(experiment_name_map),
            migrated_runs=model_counts["runs"],
            skipped_versions=self._skipped_versions,
        )

    def migrate_pending(self) -> MigrationSummary:
        """Resume migration for models that are not yet completed.

        This method treats ``PENDING``, ``IN_PROGRESS``, ``PARTIAL``, and
        ``FAILED`` rows as resumable. Combined with
        ``skip_existing_model_versions=True``, this allows a rerun to continue
        from the last durable tracking-table checkpoint without duplicating
        already-registered target versions.
        """
        if not self.tracking_table:
            raise ValueError("migrate_pending() requires a tracking_table.")

        pending_names = self.get_pending_model_names()
        if not pending_names:
            self.logger.info("No resumable models in the tracking table.")
            return MigrationSummary(0, 0, 0, 0)

        self.logger.info(f"Found {len(pending_names)} resumable model(s)")
        bundle = self._discover_models(pending_names)
        experiment_name_map = self._migrate_experiments(bundle)
        model_counts = self._migrate_models(bundle, experiment_name_map)
        return MigrationSummary(
            migrated_models=model_counts["models"],
            migrated_model_versions=model_counts["versions"],
            migrated_experiments=len(experiment_name_map),
            migrated_runs=model_counts["runs"],
            skipped_versions=self._skipped_versions,
        )

    # ---- Export / Import (offline migration) ----

    def export_bundle(self) -> tuple[DiscoveryBundle, str]:
        """Export mode: discover + download artifacts + write JSON manifests.

        Returns (bundle, manifest_path).  Everything lands in
        ``self.options.artifact_temp_dir``.
        """
        bundle = self.discover()
        bundle_dir = self.options.artifact_temp_dir
        os.makedirs(bundle_dir, exist_ok=True)

        # Write manifest.json
        manifest = {
            "source_host": self.source_credentials.normalized_host(),
            "source_registry": self.options.source_registry,
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_count": len(bundle.registered_models),
        }
        manifest_path = os.path.join(bundle_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Write models metadata
        models_dir = os.path.join(bundle_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        for model in bundle.registered_models:
            mname = model["name"]
            safe_name = sanitize_name(mname)
            model_dir = os.path.join(models_dir, safe_name)
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, "model_meta.json"), "w") as f:
                json.dump(model, f, indent=2)
            # Write per-version metadata
            versions = bundle.model_versions_by_name.get(mname, [])
            versions_dir = os.path.join(model_dir, "versions")
            os.makedirs(versions_dir, exist_ok=True)
            for v in versions:
                ver_num = str(v.get("version", "0"))
                ver_dir = os.path.join(versions_dir, ver_num)
                os.makedirs(ver_dir, exist_ok=True)
                with open(os.path.join(ver_dir, "version_meta.json"), "w") as f:
                    json.dump(v, f, indent=2)

        # Write experiments + runs metadata
        experiments_dir = os.path.join(bundle_dir, "experiments")
        os.makedirs(experiments_dir, exist_ok=True)
        for exp in bundle.experiments:
            eid = exp["experiment_id"]
            exp_dir = os.path.join(experiments_dir, eid)
            os.makedirs(exp_dir, exist_ok=True)
            with open(os.path.join(exp_dir, "experiment_meta.json"), "w") as f:
                json.dump(exp, f, indent=2)
            runs = bundle.runs_by_experiment_id.get(eid, [])
            runs_dir = os.path.join(exp_dir, "runs")
            os.makedirs(runs_dir, exist_ok=True)
            for run in runs:
                run_id = run.get("info", run).get("run_id", "unknown")
                run_dir = os.path.join(runs_dir, run_id)
                os.makedirs(run_dir, exist_ok=True)
                # Collect full metric history before writing
                source_data = run.get("data", {})
                source_run_id = run.get("info", {}).get("run_id", "")
                if source_run_id and source_data.get("metrics"):
                    try:
                        run["_full_metric_history"] = self._collect_metric_history(
                            source_run_id, source_data,
                        )
                    except Exception:
                        pass
                with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
                    json.dump(run, f, indent=2, default=str)

        # Download artifacts (reuses existing Phase 1 logic)
        self.logger.info("Downloading artifacts for export...")
        os.makedirs(bundle_dir, exist_ok=True)
        for model in bundle.registered_models:
            mname = model["name"]
            safe_name = sanitize_name(mname)
            versions = bundle.model_versions_by_name.get(mname, [])
            for v in versions:
                run_id = v.get("run_id")
                if not run_id:
                    continue
                ver_num = str(v.get("version", "0"))
                artifact_dir = os.path.join(
                    models_dir, safe_name, "versions", ver_num, "artifacts",
                )
                os.makedirs(artifact_dir, exist_ok=True)
                try:
                    source_run = self.source_rest.get_run(run_id)
                    actual_run = source_run.get("run", source_run)
                    artifact_uri = actual_run.get("info", {}).get("artifact_uri", "")
                    artifact_subpath = self._source_model_artifact_path(v)
                    # Download model artifacts
                    self.source_downloader.download_artifacts(
                        run_id=run_id,
                        artifact_path=artifact_subpath,
                        dst_path=artifact_dir,
                    )
                    # Optionally download all run artifacts
                    if self.options.include_run_artifacts and self.options.download_artifacts:
                        try:
                            self.source_downloader.download_artifacts(
                                run_id=run_id, artifact_path="", dst_path=artifact_dir,
                            )
                        except Exception:
                            pass
                except Exception as exc:
                    self.logger.warning(
                        f"Export: Failed to download artifacts for {mname} v{ver_num}: {exc}"
                    )

        self.logger.info(
            f"Export complete: {len(bundle.registered_models)} models, "
            f"{sum(len(v) for v in bundle.model_versions_by_name.values())} versions "
            f"→ {bundle_dir}"
        )
        return bundle, manifest_path

    def import_bundle(self) -> MigrationSummary:
        """Import mode: read JSON manifests + artifacts from temp dir, create on target.

        No source API calls are made — everything is read from the exported
        bundle in ``self.options.artifact_temp_dir``.
        """
        bundle_dir = self.options.artifact_temp_dir
        manifest_path = os.path.join(bundle_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"No export bundle found at {bundle_dir}. "
                "Run with MIGRATION_MODE='export' first."
            )
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.logger.info(
            f"Importing bundle from {manifest.get('source_host', '?')} "
            f"exported at {manifest.get('exported_at', '?')}"
        )

        # Reconstruct DiscoveryBundle from JSON files
        models_dir = os.path.join(bundle_dir, "models")
        experiments_dir = os.path.join(bundle_dir, "experiments")

        registered_models: list[dict] = []
        model_versions_by_name: dict[str, list[dict]] = {}

        if os.path.isdir(models_dir):
            for model_folder in sorted(os.listdir(models_dir)):
                model_meta_path = os.path.join(models_dir, model_folder, "model_meta.json")
                if not os.path.exists(model_meta_path):
                    continue
                with open(model_meta_path) as f:
                    model = json.load(f)
                registered_models.append(model)
                mname = model["name"]
                versions_dir = os.path.join(models_dir, model_folder, "versions")
                versions: list[dict] = []
                if os.path.isdir(versions_dir):
                    for ver_num in sorted(os.listdir(versions_dir), key=lambda x: int(x) if x.isdigit() else 0):
                        vmeta_path = os.path.join(versions_dir, ver_num, "version_meta.json")
                        if os.path.exists(vmeta_path):
                            with open(vmeta_path) as f:
                                versions.append(json.load(f))
                model_versions_by_name[mname] = versions

        experiments: list[dict] = []
        runs_by_experiment_id: dict[str, list[dict]] = {}

        if os.path.isdir(experiments_dir):
            for eid_folder in sorted(os.listdir(experiments_dir)):
                exp_meta_path = os.path.join(experiments_dir, eid_folder, "experiment_meta.json")
                if not os.path.exists(exp_meta_path):
                    continue
                with open(exp_meta_path) as f:
                    exp = json.load(f)
                experiments.append(exp)
                eid = exp["experiment_id"]
                runs: list[dict] = []
                runs_dir = os.path.join(experiments_dir, eid_folder, "runs")
                if os.path.isdir(runs_dir):
                    for run_folder in sorted(os.listdir(runs_dir)):
                        rmeta_path = os.path.join(runs_dir, run_folder, "run_meta.json")
                        if os.path.exists(rmeta_path):
                            with open(rmeta_path) as f:
                                runs.append(json.load(f))
                runs_by_experiment_id[eid] = runs

        bundle = DiscoveryBundle(
            registered_models=registered_models,
            model_versions_by_name=model_versions_by_name,
            experiments=experiments,
            runs_by_experiment_id=runs_by_experiment_id,
        )
        self.logger.info(
            f"Loaded bundle: {len(registered_models)} models, "
            f"{sum(len(v) for v in model_versions_by_name.values())} versions, "
            f"{len(experiments)} experiments"
        )

        # Phase 2a: Migrate experiments (clone runs from JSON)
        experiment_name_map = self._import_experiments(bundle)

        # Phase 2b: Register models (use pre-downloaded artifacts)
        model_counts = self._import_models(bundle, experiment_name_map)

        return MigrationSummary(
            migrated_models=model_counts["models"],
            migrated_model_versions=model_counts["versions"],
            migrated_experiments=len(experiment_name_map),
            migrated_runs=model_counts["runs"],
            skipped_versions=self._skipped_versions,
        )

    def _import_experiments(self, bundle: DiscoveryBundle) -> dict[str, str]:
        """Import experiments + runs from bundle JSONs (no source API calls)."""
        experiment_name_map: dict[str, str] = {}
        for exp in bundle.experiments:
            target_name = self._shared_experiment_name(exp)
            self._ensure_target_experiment(target_name, exp)

            target_exp = self._target_client.get_experiment_by_name(target_name)
            if target_exp is None:
                self.logger.warning(f"Could not create target experiment: {target_name}")
                continue
            target_experiment_id = target_exp.experiment_id
            experiment_name_map[exp["experiment_id"]] = target_name

            runs = bundle.runs_by_experiment_id.get(exp["experiment_id"], [])
            for run in runs:
                try:
                    self._clone_run_from_bundle(run, target_experiment_id)
                except Exception as exc:
                    self.logger.warning(
                        f"Failed to clone run {run.get('info', {}).get('run_id')}: {exc}"
                    )
        return experiment_name_map

    def _clone_run_from_bundle(self, source_run: dict, target_experiment_id: str) -> str:
        """Clone a run using pre-exported JSON data + metric history.

        Uses ``_full_metric_history`` if present in the export (avoids source API call),
        otherwise falls back to the summary metrics.
        """
        if "info" in source_run:
            source_info = source_run["info"]
            source_data = source_run.get("data", {})
        else:
            source_info = source_run
            source_data = {}

        source_run_id = source_info.get("run_id", "")

        # Check for existing clone
        existing = self._find_existing_target_run(source_run_id, target_experiment_id)
        if existing:
            return existing

        # Build tags
        tags = {"source_run_id": source_run_id}
        for tag in source_data.get("tags", []):
            if isinstance(tag, dict):
                key = tag.get("key", "")
                if key and not key.startswith("mlflow."):
                    tags[key] = tag.get("value", "")
        tag_list = [{"key": k, "value": v} for k, v in tags.items()]
        run_name = source_info.get("run_name", "")

        run = self._target_client.create_run(
            experiment_id=target_experiment_id,
            start_time=int(source_info["start_time"]) if source_info.get("start_time") else None,
            tags=self._tags_to_dict(tag_list),
            run_name=run_name,
        )
        target_run_id = run.info.run_id

        # Log params and metrics
        params = [{"key": p["key"], "value": p["value"]} for p in source_data.get("params", [])]

        # Use pre-fetched full history from export if available
        metrics = source_run.get("_full_metric_history", [])
        if not metrics:
            # Fallback to summary
            metrics = [
                {"key": m["key"], "value": float(m["value"]),
                 "timestamp": m.get("timestamp", 0), "step": m.get("step", 0)}
                for m in source_data.get("metrics", [])
            ]

        run_tags = [{"key": k, "value": v} for k, v in tags.items()]

        if params or metrics or run_tags:
            for i in range(0, max(len(params), len(metrics), 1), 1000):
                batch_params = params[i:i+1000] if i < len(params) else []
                batch_metrics = metrics[i:i+1000] if i < len(metrics) else []
                batch_tags = run_tags if i == 0 else []
                if batch_params or batch_metrics or batch_tags:
                    self._target_client.log_batch(
                        run_id=target_run_id,
                        params=[Param(p["key"], p["value"]) for p in batch_params],
                        metrics=[
                            Metric(m["key"], float(m["value"]),
                                   int(m.get("timestamp", 0)), int(m.get("step", 0)))
                            for m in batch_metrics
                        ],
                        tags=[RunTag(t["key"], t["value"]) for t in batch_tags],
                    )

        status = source_info.get("status", "FINISHED")
        end_time = source_info.get("end_time")
        self._target_client.set_terminated(
            run_id=target_run_id, status=status,
            end_time=int(end_time) if end_time else None,
        )
        return target_run_id

    def _import_models(
        self,
        bundle: DiscoveryBundle,
        experiment_name_map: dict[str, str],
    ) -> dict[str, int]:
        """Register models from pre-downloaded artifacts in the bundle directory."""
        if not self.options.migrate_registered_models:
            return {"models": 0, "versions": 0, "runs": 0}

        models_dir = os.path.join(self.options.artifact_temp_dir, "models")
        total_models = 0
        total_versions = 0
        total_runs = 0

        for model in bundle.registered_models:
            model_name = model["name"]
            safe_name = sanitize_name(model_name)
            target_model_name = self._target_model_name(model_name)
            self._ensure_registered_model(target_model_name, model)

            versions = bundle.model_versions_by_name.get(model_name, [])
            versions.sort(key=lambda v: int(v.get("version", 0)))

            model_created = False
            for v in versions:
                ver_num = str(v.get("version", "0"))
                run_id = v.get("run_id")

                # Check for existing
                if self.options.skip_existing_model_versions:
                    if self._target_model_version_exists(target_model_name, ver_num):
                        continue

                # Locate pre-downloaded artifacts
                artifact_dir = os.path.join(
                    models_dir, safe_name, "versions", ver_num, "artifacts",
                )
                artifact_subpath = self._source_model_artifact_path(v)
                local_model_path = Path(artifact_dir) / artifact_subpath

                if not run_id or not local_model_path.exists():
                    if self.options.create_dummy_versions:
                        staged = StagedVersion(
                            model_name=model_name, version=v, source_run={},
                            local_artifact_dir="", artifact_subpath="model",
                            experiment_id="", is_placeholder=True,
                        )
                        self._register_placeholder_version(staged, target_model_name)
                        total_versions += 1
                        model_created = True
                    else:
                        self._skipped_versions.append({
                            "model": model_name,
                            "version": ver_num,
                            "reason": "No artifacts in export bundle",
                        })
                    continue

                # Find or create the target run for this version
                source_run_data = self._load_run_from_bundle(run_id)
                if source_run_data:
                    source_info = source_run_data.get("info", {})
                    source_eid = source_info.get("experiment_id", "")
                    target_exp_name = experiment_name_map.get(source_eid)
                    if target_exp_name:
                        target_exp = self._target_client.get_experiment_by_name(target_exp_name)
                        if target_exp:
                            target_run_id = self._find_existing_target_run(
                                run_id, target_exp.experiment_id,
                            )
                            if not target_run_id:
                                target_run_id = self._clone_run_from_bundle(
                                    source_run_data, target_exp.experiment_id,
                                )
                        else:
                            target_run_id = self._create_import_run(run_id, model_name, ver_num)
                    else:
                        target_run_id = self._create_import_run(run_id, model_name, ver_num)
                else:
                    target_run_id = self._create_import_run(run_id, model_name, ver_num)

                # Upload artifacts from export directory
                if local_model_path.is_dir():
                    self.target_uploader.log_artifacts(
                        run_id=target_run_id,
                        local_dir=str(local_model_path),
                        artifact_path=artifact_subpath,
                    )
                elif local_model_path.is_file():
                    parent = str(Path(artifact_subpath).parent)
                    self.target_uploader.log_artifact(
                        run_id=target_run_id,
                        local_path=str(local_model_path),
                        artifact_path=None if parent == "." else parent,
                    )

                # Upload non-model artifacts if present
                if self.options.include_run_artifacts:
                    for item in Path(artifact_dir).iterdir():
                        if item.name == artifact_subpath.split("/")[0]:
                            continue
                        if item.is_dir():
                            self.target_uploader.log_artifacts(
                                run_id=target_run_id, local_dir=str(item),
                                artifact_path=item.name,
                            )
                        elif item.is_file():
                            self.target_uploader.log_artifact(
                                run_id=target_run_id, local_path=str(item),
                            )

                # Create model version
                target_run = self.target_uploader.get_run(target_run_id)
                model_source = f"{target_run.info.artifact_uri}/{artifact_subpath}"

                version_tags = [
                    {"key": "source_workspace_host", "value": self.source_credentials.normalized_host()},
                    {"key": "source_model_name", "value": model_name},
                    {"key": "source_model_version", "value": ver_num},
                ]
                mv = self._target_model_client.create_model_version(
                    name=target_model_name, source=model_source,
                    run_id=target_run_id, description=v.get("description"),
                    tags=self._tags_to_dict(version_tags),
                )
                # Ensure tags are set
                for tag in version_tags + v.get("tags", []):
                    if isinstance(tag, dict):
                        try:
                            self._target_model_client.set_model_version_tag(
                                name=target_model_name, version=str(mv.version),
                                key=tag["key"], value=tag["value"],
                            )
                        except Exception:
                            pass

                # Handle stages / aliases
                stage = v.get("current_stage")
                if stage and stage != "None":
                    if self.options.target_registry == "uc":
                        alias = self.options.stage_to_alias_mapping.get(stage)
                        if alias:
                            try:
                                self._target_model_client.set_registered_model_alias(
                                    name=target_model_name, alias=alias,
                                    version=str(mv.version),
                                )
                            except Exception:
                                pass
                    else:
                        try:
                            self._target_model_client.transition_model_version_stage(
                                name=target_model_name, version=str(mv.version),
                                stage=stage, archive_existing_versions=False,
                            )
                        except Exception:
                            pass

                total_versions += 1
                total_runs += 1
                model_created = True

            if model_created:
                total_models += 1
            if self.tracking_table:
                self._update_tracking_table(
                    model_name=model_name,
                    migrated_versions=total_versions,
                    migrated_runs=total_runs,
                    failed_versions=[
                        s for s in self._skipped_versions if s["model"] == model_name
                    ],
                )

        return {"models": total_models, "versions": total_versions, "runs": total_runs}

    def _load_run_from_bundle(self, run_id: str) -> dict | None:
        """Load a run's metadata from the export bundle by searching experiment dirs."""
        experiments_dir = os.path.join(self.options.artifact_temp_dir, "experiments")
        if not os.path.isdir(experiments_dir):
            return None
        for eid_folder in os.listdir(experiments_dir):
            run_meta_path = os.path.join(
                experiments_dir, eid_folder, "runs", run_id, "run_meta.json",
            )
            if os.path.exists(run_meta_path):
                with open(run_meta_path) as f:
                    return json.load(f)
        return None

    def _create_import_run(
        self, source_run_id: str, model_name: str, ver_num: str,
    ) -> str:
        """Create a minimal target run for a version when no experiment mapping exists."""
        placeholder_exp = f"{self.options.shared_experiment_root}/_imported"
        try:
            exp = self._target_client.get_experiment_by_name(placeholder_exp)
            if exp is None:
                raise ValueError("not found")
        except Exception:
            self._target_client.create_experiment(placeholder_exp)
            exp = self._target_client.get_experiment_by_name(placeholder_exp)

        run = self._target_client.create_run(
            experiment_id=exp.experiment_id,
            tags={
                "source_run_id": source_run_id,
                "source_model_name": model_name,
                "source_model_version": ver_num,
                "imported_from_bundle": "true",
            },
            run_name=f"import_{model_name}_v{ver_num}",
        )
        self._target_client.set_terminated(run.info.run_id)
        return run.info.run_id

    def _discover_models(self, model_names: list[str]) -> DiscoveryBundle:
        """Targeted discovery for a list of model names."""
        requested = {name.strip() for name in model_names}
        registered_models: list[dict] = []
        for name in sorted(requested):
            try:
                if self._source_is_uc:
                    model_data = self.source_rest.uc_get_registered_model(name)
                    rm = model_data.get("registered_model", model_data)
                    if "name" in rm:
                        registered_models.append(rm)
                else:
                    escaped = name.replace("'", "''")
                    data = self.source_rest.search_registered_models(
                        max_results=5, filter_string=f"name='{escaped}'",
                    )
                    registered_models.extend(data.get("registered_models", []))
            except Exception as exc:
                self.logger.warning(f"Could not find model '{name}': {exc}")

        model_versions_by_name: dict[str, list[dict]] = {}
        if registered_models:
            _vlist_workers = min(
                2 if self._source_is_uc else self.options.max_workers,
                len(registered_models),
            )
            with ThreadPoolExecutor(max_workers=_vlist_workers) as pool:
                future_map: dict = {}
                for i, m in enumerate(registered_models):
                    if i > 0 and self._source_is_uc:
                        time.sleep(0.5)
                    future_map[pool.submit(self._list_model_versions, m["name"])] = m["name"]
                for future in as_completed(future_map):
                    name = future_map[future]
                    try:
                        model_versions_by_name[name] = future.result()
                    except Exception as exc:
                        self.logger.warning(f"Failed versions for {name}: {exc}")
                        model_versions_by_name[name] = []

        experiments = self._list_experiments(registered_models, model_versions_by_name)
        runs_by_experiment_id: dict[str, list[dict]] = {}
        if experiments:
            with ThreadPoolExecutor(max_workers=min(self.options.max_workers, len(experiments))) as pool:
                future_map = {
                    pool.submit(self._list_runs, e["experiment_id"]): e["experiment_id"]
                    for e in experiments
                }
                for future in as_completed(future_map):
                    eid = future_map[future]
                    try:
                        runs_by_experiment_id[eid] = future.result()
                    except Exception as exc:
                        runs_by_experiment_id[eid] = []

        return DiscoveryBundle(
            registered_models=registered_models,
            model_versions_by_name=model_versions_by_name,
            experiments=experiments,
            runs_by_experiment_id=runs_by_experiment_id,
        )

    # ---- Experiments Migration ----

    def _migrate_experiments(self, bundle: DiscoveryBundle) -> dict[str, str]:
        """Migrate experiments to target workspace. Returns experiment_id -> target_name map."""
        experiment_name_map: dict[str, str] = {}
        for batch in chunked(bundle.experiments, self.options.batch_size):
            self.logger.info(f"Migrating experiment batch of {len(batch)}")
            with ThreadPoolExecutor(max_workers=min(self.options.max_workers, len(batch) or 1)) as pool:
                future_map = {
                    pool.submit(
                        self._migrate_single_experiment, exp, bundle.runs_by_experiment_id,
                    ): exp
                    for exp in batch
                }
                for future in as_completed(future_map):
                    exp = future_map[future]
                    try:
                        target_name = future.result()
                        experiment_name_map[exp["experiment_id"]] = target_name
                    except Exception as exc:
                        self.logger.warning(f"Failed experiment {exp.get('name')}: {exc}")
        return experiment_name_map

    def _migrate_single_experiment(
        self, experiment: dict, runs_by_experiment_id: dict[str, list[dict]],
    ) -> str:
        target_name = self._shared_experiment_name(experiment)
        if self.options.create_missing_experiments:
            self._ensure_target_experiment(target_name, experiment)

        target_exp = self._target_client.get_experiment_by_name(target_name)
        if target_exp is None:
            raise ValueError(f"Target experiment not found: {target_name}")
        target_experiment_id = target_exp.experiment_id

        runs = runs_by_experiment_id.get(experiment["experiment_id"], [])
        for run in runs:
            try:
                self._clone_run(run, target_experiment_id)
            except Exception as exc:
                self.logger.warning(f"Failed to clone run {run.get('info', {}).get('run_id')}: {exc}")

        self.logger.info(f"Migrated experiment {experiment.get('name')} -> {target_name}")
        return target_name

    # ---- Models Migration: TWO-PHASE PIPELINE ----

    def _migrate_models(
        self,
        bundle: DiscoveryBundle,
        experiment_name_map: dict[str, str],
    ) -> dict[str, int]:
        """Migrate registered models using two-phase pipeline.

        Phase 1 (Download): Download all artifacts from source -> local staging.
                            Uses dedicated download workers, high concurrency.
        Phase 2 (Register): Upload artifacts + register versions on target.
                            Versions within each model are registered SEQUENTIALLY
                            (preserves version ordering). Models in parallel.
        """
        if not self.options.migrate_registered_models:
            return {"models": 0, "versions": 0, "runs": 0}

        import os as _os
        _os.makedirs(self.options.artifact_temp_dir, exist_ok=True)

        total_models = 0
        total_versions = 0
        total_runs = 0

        for batch in chunked(bundle.registered_models, self.options.batch_size):
            self.logger.info(f"Processing model batch of {len(batch)}")

            # Phase 1: Download all artifacts for this batch
            staged_by_model: dict[str, list[StagedVersion]] = {}
            download_tasks: list[tuple[dict, dict, str]] = []  # (model, version, staging_dir)

            for model in batch:
                model_name = model["name"]
                versions = bundle.model_versions_by_name.get(model_name, [])
                target_model_name = self._target_model_name(model_name)

                # Filter out already-migrated versions
                candidate_versions = [
                    v for v in versions
                    if not (
                        self.options.skip_existing_model_versions
                        and self._target_model_version_exists(target_model_name, v.get("version"))
                    )
                ]

                # Sort by version number ascending (critical for ordering)
                candidate_versions.sort(key=lambda v: int(v.get("version", 0)))

                for version in candidate_versions:
                    run_id = version.get("run_id")
                    if not run_id:
                        if self.options.create_dummy_versions:
                            staged_by_model.setdefault(model_name, []).append(StagedVersion(
                                model_name=model_name, version=version, source_run={},
                                local_artifact_dir="", artifact_subpath="model",
                                experiment_id="", is_placeholder=True,
                            ))
                        else:
                            self._skipped_versions.append({
                                "model": model_name,
                                "version": str(version.get("version")),
                                "reason": "No run_id",
                            })
                        continue
                    staging_dir = tempfile.mkdtemp(
                        prefix=f"stage_{sanitize_name(model_name)}_v{version.get('version')}_",
                        dir=self.options.artifact_temp_dir,
                    )
                    download_tasks.append((model, version, staging_dir))

            # Phase 1: Parallel artifact downloads
            self.logger.info(f"Phase 1: Downloading artifacts ({len(download_tasks)} versions)")
            if download_tasks:
                with ThreadPoolExecutor(max_workers=self.options.download_workers) as pool:
                    future_map = {
                        pool.submit(
                            self._download_version_artifacts,
                            model, version, staging_dir, experiment_name_map,
                        ): (model, version, staging_dir)
                        for model, version, staging_dir in download_tasks
                    }
                    for future in as_completed(future_map):
                        model, version, staging_dir = future_map[future]
                        model_name = model["name"]
                        try:
                            staged = future.result()
                            if staged:
                                staged_by_model.setdefault(model_name, []).append(staged)
                        except Exception as exc:
                            self.logger.warning(
                                f"Download failed for {model_name} v{version.get('version')}: {exc}"
                            )
                            if self.options.create_dummy_versions:
                                staged_by_model.setdefault(model_name, []).append(StagedVersion(
                                    model_name=model_name, version=version, source_run={},
                                    local_artifact_dir="", artifact_subpath="model",
                                    experiment_id="", is_placeholder=True,
                                ))
                            else:
                                self._skipped_versions.append({
                                    "model": model_name,
                                    "version": str(version.get("version")),
                                    "reason": f"Download failed: {str(exc)[:100]}",
                                })
                            shutil.rmtree(staging_dir, ignore_errors=True)

            # Phase 2: Register — sequential per model, models in parallel
            self.logger.info(f"Phase 2: Registering versions ({len(staged_by_model)} models)")
            if staged_by_model:
                with ThreadPoolExecutor(max_workers=self.options.register_workers) as pool:
                    future_map: dict[Any, str] = {}
                    for model_name, staged_versions in staged_by_model.items():
                        if self.tracking_table:
                            self._mark_tracking_status(
                                model_name=model_name,
                                migration_status="IN_PROGRESS",
                                migration_comments="Registration started",
                            )
                        future = pool.submit(
                            self._register_model_sequentially,
                            model_name,
                            staged_versions,
                            next(m for m in batch if m["name"] == model_name),
                            experiment_name_map,
                        )
                        future_map[future] = model_name

                    for future in as_completed(future_map):
                        model_name = future_map[future]
                        try:
                            counts = future.result()
                            total_models += counts["models"]
                            total_versions += counts["versions"]
                            total_runs += counts["runs"]
                            if self.tracking_table:
                                self._update_tracking_table(
                                    model_name=model_name,
                                    migrated_versions=counts["versions"],
                                    migrated_runs=counts["runs"],
                                    failed_versions=[s for s in self._skipped_versions if s["model"] == model_name],
                                )
                        except Exception as exc:
                            self.logger.warning(f"Registration failed for {model_name}: {exc}")
                            if self.tracking_table:
                                self._mark_tracking_status(
                                    model_name=model_name,
                                    migration_status="FAILED",
                                    migration_comments=str(exc)[:200],
                                )

        return {"models": total_models, "versions": total_versions, "runs": total_runs}

    def _download_version_artifacts(
        self,
        model: dict,
        version: dict,
        staging_dir: str,
        experiment_name_map: dict[str, str],
    ) -> StagedVersion | None:
        """Phase 1: Download artifacts for a single version to local staging."""
        run_id = version.get("run_id")
        if not run_id:
            return None

        # Fetch source run metadata via REST
        run_data = self.source_rest.get_run(run_id)
        source_run = run_data.get("run", run_data)
        experiment_id = source_run.get("info", {}).get("experiment_id", "")

        # Determine artifact subpath
        artifact_subpath = self._source_model_artifact_path(version)

        # Download model artifacts to staging
        try:
            self.source_downloader.download_artifacts(
                run_id=run_id,
                artifact_path=artifact_subpath,
                dst_path=staging_dir,
            )
        except Exception as exc:
            self.logger.warning(f"Model artifact download failed for {run_id}/{artifact_subpath}: {exc}")
            raise

        # Optionally download all run artifacts
        if self.options.include_run_artifacts and self.options.download_artifacts:
            # List root artifacts via REST and download each valid one individually.
            # Avoids passing artifact_path=None to mlflow, which hits phantom empty-path
            # entries in the artifact list and raises INVALID_PARAMETER_VALUE.
            try:
                root_artifacts = self.source_rest.list_artifacts(run_id)
                for art in root_artifacts:
                    if not art.path:
                        continue  # skip phantom empty-path entries
                    if art.path == artifact_subpath:
                        continue  # already downloaded as model artifact above
                    try:
                        self.source_downloader.download_artifacts(
                            run_id=run_id,
                            artifact_path=art.path,
                            dst_path=staging_dir,
                        )
                    except Exception as exc:
                        self.logger.warning(f"Run artifact download partial for {run_id}/{art.path}: {exc}")
            except Exception as exc:
                self.logger.warning(f"Run artifact listing failed for {run_id}: {exc}")

        return StagedVersion(
            model_name=model["name"],
            version=version,
            source_run=source_run,
            local_artifact_dir=staging_dir,
            artifact_subpath=artifact_subpath,
            experiment_id=experiment_id,
        )

    def _register_model_sequentially(
        self,
        model_name: str,
        staged_versions: list[StagedVersion],
        model: dict,
        experiment_name_map: dict[str, str],
    ) -> dict[str, int]:
        """Phase 2: Register all versions for ONE model in order.

        Versions are registered SEQUENTIALLY (sorted by source version number)
        so that target auto-increment matches source ordering.
        """
        target_model_name = self._target_model_name(model_name)
        self._ensure_registered_model(target_model_name, model)

        # Sort by source version number ascending
        staged_versions.sort(key=lambda sv: int(sv.version.get("version", 0)))

        migrated_versions = 0
        migrated_runs = 0

        for sv in staged_versions:
            try:
                result = self._register_staged_version(
                    sv, target_model_name, experiment_name_map,
                )
                migrated_versions += 1
                migrated_runs += result.get("runs", 0)

                # Version ordering check
                created_version = result.get("target_version")
                source_version = sv.version.get("version")
                if created_version and str(created_version) != str(source_version):
                    self.logger.warning(
                        f"Version mismatch: {model_name} source v{source_version} -> target v{created_version}"
                    )
            except Exception as exc:
                self.logger.warning(
                    f"Registration failed for {model_name} v{sv.version.get('version')}: {exc}"
                )
                self._skipped_versions.append({
                    "model": model_name,
                    "version": str(sv.version.get("version")),
                    "reason": f"Register failed: {str(exc)[:100]}",
                })
            finally:
                # Clean up staging dir for this version
                shutil.rmtree(sv.local_artifact_dir, ignore_errors=True)

        self.logger.info(
            f"Registered {model_name} -> {target_model_name}: {migrated_versions} versions"
        )
        return {"models": 1, "versions": migrated_versions, "runs": migrated_runs, "_model_name": model_name}

    def _register_staged_version(
        self,
        staged: StagedVersion,
        target_model_name: str,
        experiment_name_map: dict[str, str],
    ) -> dict[str, Any]:
        """Upload staged artifacts to target and register the version."""
        if staged.is_placeholder:
            return self._register_placeholder_version(staged, target_model_name)

        version = staged.version
        source_run = staged.source_run

        # Ensure target run exists
        target_run_id = self._ensure_target_run(source_run, experiment_name_map)

        # Upload model artifacts from staging to target
        local_model_path = Path(staged.local_artifact_dir) / staged.artifact_subpath
        if local_model_path.is_dir():
            self.target_uploader.log_artifacts(
                run_id=target_run_id,
                local_dir=str(local_model_path),
                artifact_path=staged.artifact_subpath,
            )
        elif local_model_path.is_file():
            parent = str(Path(staged.artifact_subpath).parent)
            self.target_uploader.log_artifact(
                run_id=target_run_id,
                local_path=str(local_model_path),
                artifact_path=None if parent == "." else parent,
            )

        # Upload non-model artifacts if present
        if self.options.include_run_artifacts and self.options.download_artifacts:
            staging_root = Path(staged.local_artifact_dir)
            for item in staging_root.iterdir():
                item_name = item.name
                if item_name == staged.artifact_subpath.split("/")[0]:
                    continue  # already uploaded
                if item.is_dir():
                    self.target_uploader.log_artifacts(
                        run_id=target_run_id, local_dir=str(item), artifact_path=item_name,
                    )
                elif item.is_file():
                    self.target_uploader.log_artifact(
                        run_id=target_run_id, local_path=str(item),
                    )

        # Get target run's artifact URI for model source
        target_run = self.target_uploader.get_run(target_run_id)
        model_source = f"{target_run.info.artifact_uri}/{staged.artifact_subpath}"

        # Create model version via SDK (auto-routes to UC or workspace based on registry_uri)
        version_tags = [
            {"key": "source_workspace_host", "value": self.source_credentials.normalized_host()},
            {"key": "source_model_name", "value": staged.model_name},
            {"key": "source_model_version", "value": str(version.get("version"))},
        ]
        mv = self._target_model_client.create_model_version(
            name=target_model_name, source=model_source,
            run_id=target_run_id, description=version.get("description"),
            tags=self._tags_to_dict(version_tags),
        )
        created_version = mv.version

        # Set migration metadata tags explicitly
        # (UC create API does not persist inline tags; workspace API may also drop them)
        if created_version:
            all_tags_to_set = list(version_tags)  # migration metadata tags
            # Also include source version tags
            for tag in version.get("tags", []):
                if isinstance(tag, dict):
                    all_tags_to_set.append(tag)
            for tag in all_tags_to_set:
                try:
                    self._target_model_client.set_model_version_tag(
                        name=target_model_name, version=str(created_version),
                        key=tag["key"], value=tag["value"],
                    )
                except Exception:
                    pass

        # Handle stages / aliases
        stage = version.get("current_stage")
        if stage and stage != "None":
            if self.options.target_registry == "uc":
                alias = self.options.stage_to_alias_mapping.get(stage)
                if alias:
                    try:
                        self._target_model_client.set_registered_model_alias(
                            name=target_model_name, alias=alias, version=str(created_version),
                        )
                    except Exception as exc:
                        self.logger.warning(f"Could not set alias {alias}: {exc}")
            else:
                try:
                    self._target_model_client.transition_model_version_stage(
                        name=target_model_name, version=str(created_version),
                        stage=stage, archive_existing_versions=False,
                    )
                except Exception as exc:
                    self.logger.warning(f"Could not set stage {stage}: {exc}")

        return {"runs": 1, "target_version": created_version}

    # ---- Placeholder versions ----

    def _register_placeholder_version(
        self,
        staged: StagedVersion,
        target_model_name: str,
    ) -> dict[str, Any]:
        """Create a dummy run + model version to preserve version numbering."""
        import os as _os

        placeholder_exp = f"{self.options.shared_experiment_root}/_placeholders"
        try:
            exp = self._target_client.get_experiment_by_name(placeholder_exp)
            if exp is None:
                raise ValueError("not found")
        except Exception:
            self._target_client.create_experiment(placeholder_exp)
            exp = self._target_client.get_experiment_by_name(placeholder_exp)

        src_ver = str(staged.version.get("version", "?"))
        run = self._target_client.create_run(
            experiment_id=exp.experiment_id,
            tags={
                "placeholder": "true",
                "source_model_name": staged.model_name,
                "source_model_version": src_ver,
                "source_workspace_host": self.source_credentials.normalized_host(),
                "reason": "Source run deleted or unavailable",
            },
            run_name=f"placeholder_{staged.model_name}_v{src_ver}",
        )
        target_run_id = run.info.run_id
        self._target_client.set_terminated(target_run_id)

        # Minimal but valid MLmodel file so UC model version can be created.
        # UC requires a proper MLmodel YAML with a model signature.
        with tempfile.TemporaryDirectory() as td:
            model_dir = _os.path.join(td, "model")
            _os.makedirs(model_dir)
            mlmodel_content = (
                "artifact_path: model\n"
                "flavors:\n"
                "  python_function:\n"
                "    loader_module: mlflow.pyfunc\n"
                "signature:\n"
                "  inputs: '[{\"type\": \"string\", \"name\": \"placeholder_input\"}]'\n"
                "  outputs: '[{\"type\": \"string\", \"name\": \"placeholder_output\"}]'\n"
                f"# Placeholder — source v{src_ver} run was deleted\n"
            )
            with open(_os.path.join(model_dir, "MLmodel"), "w") as f:
                f.write(mlmodel_content)
            self.target_uploader.log_artifacts(
                run_id=target_run_id, local_dir=model_dir, artifact_path="model",
            )

        model_source = f"{run.info.artifact_uri}/model"
        tag_dict = {
            "placeholder": "true",
            "source_model_version": src_ver,
            "source_workspace_host": self.source_credentials.normalized_host(),
            "source_model_name": staged.model_name,
        }
        mv = self._target_model_client.create_model_version(
            name=target_model_name, source=model_source,
            run_id=target_run_id,
            description=f"[Placeholder] Source v{src_ver} — run deleted",
            tags=tag_dict,
        )
        for k, v in tag_dict.items():
            try:
                self._target_model_client.set_model_version_tag(
                    name=target_model_name, version=str(mv.version), key=k, value=v,
                )
            except Exception:
                pass

        return {"runs": 0, "target_version": mv.version}

    # ---- Run cloning ----

    def _ensure_target_run(
        self, source_run: dict, experiment_name_map: dict[str, str],
    ) -> str:
        """Ensure a target run exists for the source run. Returns target run_id."""
        source_info = source_run.get("info", {})
        source_data = source_run.get("data", {})
        source_run_id = source_info.get("run_id", "")
        source_experiment_id = source_info.get("experiment_id", "")

        # Find or create target experiment
        target_experiment_name = experiment_name_map.get(source_experiment_id)
        if target_experiment_name is None:
            source_exp = self.source_rest.get_experiment(source_experiment_id)
            exp = source_exp.get("experiment", source_exp)
            target_experiment_name = self._shared_experiment_name(exp)
            self._ensure_target_experiment(target_experiment_name, exp)

        target_exp = self._target_client.get_experiment_by_name(target_experiment_name)
        if target_exp is None:
            raise ValueError(f"Missing target experiment {target_experiment_name}")
        target_experiment_id = target_exp.experiment_id

        return self._clone_run(source_run, target_experiment_id)

    def _clone_run(self, source_run: dict, target_experiment_id: str) -> str:
        """Clone a source run into the target experiment. Returns target run_id."""
        # Handle both flat and nested run structures
        if "info" in source_run:
            source_info = source_run["info"]
            source_data = source_run.get("data", {})
        else:
            source_info = source_run
            source_data = {}

        source_run_id = source_info.get("run_id", "")

        # Check if already cloned
        existing = self._find_existing_target_run(source_run_id, target_experiment_id)
        if existing:
            return existing

        # Build tags
        source_tags = {t["key"]: t["value"] for t in source_data.get("tags", [])}
        tags = {k: v for k, v in source_tags.items() if not k.startswith("mlflow.")}
        tags["source_workspace_host"] = self.source_credentials.normalized_host()
        tags["source_run_id"] = source_run_id

        tag_list = [{"key": k, "value": v} for k, v in tags.items()]
        run_name = source_tags.get("mlflow.runName")

        # Create target run via SDK
        run = self._target_client.create_run(
            experiment_id=target_experiment_id,
            start_time=int(source_info["start_time"]) if source_info.get("start_time") else None,
            tags=self._tags_to_dict(tag_list),
            run_name=run_name,
        )
        target_run_id = run.info.run_id
        if not target_run_id:
            raise ValueError(f"Failed to create target run for source {source_run_id}")

        # Log params and metrics
        params = [{"key": p["key"], "value": p["value"]} for p in source_data.get("params", [])]

        # Collect full metric history
        metrics = self._collect_metric_history(source_run_id, source_data)

        run_tags = [{"key": k, "value": v} for k, v in tags.items()]

        if params or metrics or run_tags:
            # log_batch has a limit of 1000 params/metrics per call
            for i in range(0, max(len(params), len(metrics), 1), 1000):
                batch_params = params[i:i+1000] if i < len(params) else []
                batch_metrics = metrics[i:i+1000] if i < len(metrics) else []
                batch_tags = run_tags if i == 0 else []
                if batch_params or batch_metrics or batch_tags:
                    self._target_client.log_batch(
                        run_id=target_run_id,
                        params=[Param(p["key"], p["value"]) for p in batch_params],
                        metrics=[Metric(m["key"], float(m["value"]), int(m.get("timestamp", 0)), int(m.get("step", 0))) for m in batch_metrics],
                        tags=[RunTag(t["key"], t["value"]) for t in batch_tags],
                    )

        # Terminate run
        status = source_info.get("status", "FINISHED")
        end_time = source_info.get("end_time")
        self._target_client.set_terminated(run_id=target_run_id, status=status, end_time=int(end_time) if end_time else None)

        return target_run_id

    def _collect_metric_history(self, source_run_id: str, source_data: dict) -> list[dict]:
        """Collect full metric history from source via REST."""
        metrics_summary = source_data.get("metrics", [])
        all_metrics: list[dict] = []
        for m in metrics_summary:
            key = m["key"]
            try:
                history = self.source_rest.get_metric_history(source_run_id, key)
                all_metrics.extend(history)
            except Exception:
                # Fallback to summary value
                all_metrics.append({
                    "key": key,
                    "value": float(m["value"]),
                    "timestamp": m.get("timestamp", 0),
                    "step": m.get("step", 0),
                })
        return all_metrics

    # ---- Target helpers (all via MLflow SDK — auto-authenticates to current workspace) ----

    def _ensure_target_experiment(self, name: str, source_experiment: dict) -> None:
        existing = self._target_client.get_experiment_by_name(name)
        if existing is None:
            tags = {
                "source_workspace_host": self.source_credentials.normalized_host(),
                "source_experiment_id": source_experiment.get("experiment_id", ""),
            }
            self._target_client.create_experiment(name, tags=tags)

        # Copy source tags
        exp_tags = source_experiment.get("tags", [])
        if isinstance(exp_tags, dict):
            exp_tags = [{"key": k, "value": v} for k, v in exp_tags.items()]
        for tag in exp_tags:
            key = tag.get("key", "") if isinstance(tag, dict) else ""
            if key and not key.startswith("mlflow."):
                try:
                    target_exp = self._target_client.get_experiment_by_name(name)
                    if target_exp:
                        self._target_client.set_experiment_tag(
                            target_exp.experiment_id, key, tag.get("value", ""),
                        )
                except Exception:
                    pass

    def _target_model_name(self, source_model_name: str) -> str:
        """Compute the target model name, accounting for UC or workspace registry."""
        # When source is UC (catalog.schema.model), strip the source catalog.schema
        source_catalog = source_schema = ""
        if self._source_is_uc and "." in source_model_name:
            parts = source_model_name.split(".")
            if len(parts) >= 3:
                source_catalog, source_schema, short_name = parts[0], parts[1], parts[2]
            else:
                short_name = source_model_name
        else:
            short_name = source_model_name

        base = f"{self.options.model_name_prefix}{short_name}"

        if self.options.target_registry == "uc":
            # Mirror source catalog.schema unless user explicitly overrides
            target_cat = self.options.uc_target_catalog or source_catalog
            target_sch = self.options.uc_target_schema or source_schema
            if not target_cat or not target_sch:
                raise ValueError(
                    f"Cannot determine UC target for '{source_model_name}': "
                    "provide UC_TARGET_CATALOG / UC_TARGET_SCHEMA."
                )
            return f"{target_cat}.{target_sch}.{base}"
        return base

    def _ensure_registered_model(self, target_model_name: str, source_model: dict) -> None:
        tags = {
            "source_workspace_host": self.source_credentials.normalized_host(),
            "source_model_name": source_model.get("name", ""),
        }
        try:
            self._target_model_client.get_registered_model(target_model_name)
        except Exception:
            self._target_model_client.create_registered_model(
                name=target_model_name,
                description=source_model.get("description"),
                tags=tags,
            )
        # Copy source tags (workspace registry supports set-tag)
        if self.options.target_registry != "uc":
            for tag in source_model.get("tags", []):
                if isinstance(tag, dict):
                    self._target_model_client.set_registered_model_tag(
                        target_model_name, tag["key"], tag["value"],
                    )

    def _target_model_version_exists(self, target_model_name: str, source_version: str | int) -> bool:
        try:
            escaped = target_model_name.replace("'", "''")
            versions = list(self._target_model_client.search_model_versions(
                filter_string=f"name='{escaped}'",
            ))
        except Exception:
            return False
        for v in versions:
            # UC ModelVersionSearch.tags is a method that throws — must
            # call get_model_version per version to obtain actual tags.
            try:
                full_v = self._target_model_client.get_model_version(
                    target_model_name, v.version,
                )
                tags = full_v.tags if isinstance(full_v.tags, dict) else {}
            except Exception:
                tags = {}
            if tags.get("source_model_version") == str(source_version):
                return True
        return False

    def _find_existing_target_run(self, source_run_id: str, target_experiment_id: str) -> str | None:
        runs = self._target_client.search_runs(
            experiment_ids=[target_experiment_id],
            filter_string=f"tags.source_run_id = '{source_run_id}'",
            max_results=1,
        )
        if runs:
            return runs[0].info.run_id
        return None

    def _source_model_artifact_path(self, version: dict) -> str:
        source = version.get("source", "")
        if "/artifacts/" in source:
            return source.split("/artifacts/", 1)[1].strip("/") or "model"
        return "model"

    def _shared_experiment_name(self, experiment: dict) -> str:
        name = experiment.get("name", "")
        base_name = name.rstrip("/").split("/")[-1]
        prefix = self.options.experiment_name_prefix or ""
        return f"{self.options.shared_experiment_root}/{prefix}{base_name}"

    # ---- Tracking table ----

    def persist_discovery_tracking(self, bundle: DiscoveryBundle) -> int:
        """MERGE discovery results into the tracking table before migration starts."""
        if not self.tracking_table:
            return 0

        from datetime import datetime

        import pandas as pd
        from workspace_registry_migrator.reporting import write_inventory_to_delta

        run_to_experiment_id = {
            run.get("info", {}).get("run_id", ""): experiment_id
            for experiment_id, runs in bundle.runs_by_experiment_id.items()
            for run in runs
            if run.get("info", {}).get("run_id")
        }
        experiment_name_by_id = {
            experiment.get("experiment_id", ""): experiment.get("name", "")
            for experiment in bundle.experiments
        }

        rows: list[dict[str, Any]] = []
        for model in bundle.registered_models:
            model_name = model["name"]
            versions = bundle.model_versions_by_name.get(model_name, [])
            total_versions = len(versions)
            migratable_versions = sum(
                1
                for version in versions
                if version.get("run_id") or self.options.create_dummy_versions
            )
            blocked_versions = total_versions - migratable_versions
            source_experiment_ids = {
                run_to_experiment_id[version.get("run_id", "")]
                for version in versions
                if version.get("run_id", "") in run_to_experiment_id
            }
            stage_values = sorted(
                {
                    version.get("current_stage", "")
                    for version in versions
                    if version.get("current_stage")
                    and version.get("current_stage") != "None"
                }
            )

            latest_version_created = max(
                (int(version.get("creation_timestamp", 0) or 0) for version in versions),
                default=0,
            )
            latest_version_created_str = ""
            if latest_version_created:
                latest_version_created_str = datetime.utcfromtimestamp(
                    latest_version_created / 1000
                ).strftime("%Y-%m-%d %H:%M")

            model_created = int(model.get("creation_timestamp", 0) or 0)
            created_str = ""
            if model_created:
                created_str = datetime.utcfromtimestamp(model_created / 1000).strftime(
                    "%Y-%m-%d"
                )

            if migratable_versions == 0:
                readiness = "BLOCKED"
            elif blocked_versions > 0:
                readiness = "PARTIAL"
            else:
                readiness = "READY"

            discovery_comments = None
            if blocked_versions:
                discovery_comments = (
                    f"{blocked_versions} version(s) need placeholders or were unavailable "
                    "at discovery time."
                )

            rows.append(
                {
                    "source_host": self.source_credentials.normalized_host(),
                    "target_host": self._target_host,
                    "model_name": model_name,
                    "readiness": readiness,
                    "source_versions": total_versions,
                    "source_versions_migratable": migratable_versions,
                    "source_versions_blocked": blocked_versions,
                    "source_experiments": len(source_experiment_ids),
                    "source_runs": sum(
                        len(bundle.runs_by_experiment_id.get(experiment_id, []))
                        for experiment_id in source_experiment_ids
                    ),
                    "source_params": 0,
                    "source_metrics": 0,
                    "source_artifacts": 0,
                    "stages": ", ".join(stage_values) if stage_values else "None",
                    "owner_emails": model.get("user_id") or "",
                    "flavors": None,
                    "requirements": None,
                    "python_version": None,
                    "experiments": " | ".join(
                        sorted(
                            experiment_name_by_id.get(experiment_id, experiment_id)
                            for experiment_id in source_experiment_ids
                        )
                    ),
                    "latest_version_created": latest_version_created_str,
                    "created": created_str,
                    "target_versions": 0,
                    "target_runs": 0,
                    "target_params": 0,
                    "target_metrics": 0,
                    "target_artifacts": 0,
                    "migration_status": "PENDING",
                    "discovery_comments": discovery_comments,
                    "migration_comments": None,
                    "target_model_url": None,
                    "target_experiment_urls": None,
                }
            )

        if not rows:
            return 0

        write_inventory_to_delta(pd.DataFrame(rows), self.tracking_table)
        return len(rows)

    def get_pending_model_names(self) -> list[str]:
        """Return models that should be resumed from the tracking table."""
        if not self.tracking_table:
            return []

        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if not spark:
            raise RuntimeError("No active SparkSession")

        escaped_host = self.source_credentials.normalized_host().replace("'", "''")
        pending_rows = spark.sql(f"""
            SELECT model_name
            FROM {self.tracking_table}
            WHERE source_host = '{escaped_host}'
              AND migration_status IN ('PENDING', 'IN_PROGRESS', 'PARTIAL', 'FAILED')
            ORDER BY model_name
        """).collect()
        return [row["model_name"] for row in pending_rows]

    def _mark_tracking_status(
        self,
        model_name: str,
        migration_status: str,
        migration_comments: str | None = None,
    ) -> None:
        """Update only the status/comment fields for a tracked model."""
        if not self.tracking_table:
            return

        try:
            from workspace_registry_migrator.reporting import update_tracking_after_model

            update_tracking_after_model(
                tracking_table=self.tracking_table,
                source_host=self.source_credentials.normalized_host(),
                model_name=model_name,
                target_versions=None,
                target_runs=None,
                target_params=None,
                target_metrics=None,
                target_artifacts=None,
                migration_status=migration_status,
                migration_comments=migration_comments,
                target_model_url=None,
                target_experiment_urls=None,
            )
        except Exception as exc:
            self.logger.warning(
                f"Failed to update tracking status for {model_name}: {exc}"
            )

    def _update_tracking_table(
        self,
        model_name: str,
        migrated_versions: int,
        migrated_runs: int,
        failed_versions: list[dict[str, str]],
    ) -> None:
        del migrated_versions, migrated_runs
        try:
            import re
            from workspace_registry_migrator.reporting import update_tracking_after_model

            target_model_name = self._target_model_name(model_name)
            escaped_target = target_model_name.replace("'", "''")
            target_versions_raw = list(
                self._target_model_client.search_model_versions(
                    filter_string=f"name='{escaped_target}'",
                )
            )
            target_versions = []
            for v in target_versions_raw:
                # UC ModelVersionSearch.tags is a method that throws —
                # must call get_model_version to obtain actual tags.
                try:
                    full_v = self._target_model_client.get_model_version(
                        target_model_name, v.version,
                    )
                    tags_dict = full_v.tags if isinstance(full_v.tags, dict) else {}
                except Exception:
                    tags_dict = {}
                target_versions.append({
                    "version": v.version,
                    "run_id": v.run_id,
                    "tags": [
                        {"key": key, "value": value}
                        for key, value in tags_dict.items()
                    ],
                })
            migrated_target_versions = [
                version
                for version in target_versions
                if any(
                    tag.get("key") == "source_model_version"
                    for tag in version.get("tags", [])
                )
            ]
            target_version_count = len(migrated_target_versions)

            total_params = 0
            total_metrics = 0
            total_artifacts = 0
            target_experiment_ids: set[str] = set()
            target_run_ids: set[str] = set()
            for target_version in migrated_target_versions:
                run_id = target_version.get("run_id")
                if not run_id:
                    continue
                target_run_ids.add(run_id)
                try:
                    run = self.target_uploader.get_run(run_id)
                    total_params += len(run.data.params)
                    total_metrics += len(run.data.metrics)
                    total_artifacts += len(
                        self.target_uploader.list_artifacts(run_id=run_id, path="model")
                    )
                    target_experiment_ids.add(run.info.experiment_id)
                except Exception:
                    pass

            source_migratable = self._get_source_migratable(model_name)
            if source_migratable > 0 and target_version_count >= source_migratable:
                status = "COMPLETED"
            elif target_version_count > 0:
                status = "PARTIAL"
            elif failed_versions:
                status = "FAILED"
            else:
                status = "PENDING"

            comments = (
                "; ".join(failed_version["reason"][:60] for failed_version in failed_versions)
                if failed_versions
                else None
            )
            host = self._target_host.rstrip("/")
            ws_id_match = re.search(r"adb-(\d+)", host)
            ws_id = ws_id_match.group(1) if ws_id_match else ""
            o_param = f"?o={ws_id}" if ws_id else ""
            target_model_url = f"{host}/ml/models/{target_model_name}{o_param}"
            target_exp_urls = (
                " | ".join(
                    f"{host}/ml/experiments/{experiment_id}"
                    for experiment_id in sorted(target_experiment_ids)
                )
                if target_experiment_ids
                else None
            )

            update_tracking_after_model(
                tracking_table=self.tracking_table,
                source_host=self.source_credentials.normalized_host(),
                model_name=model_name,
                target_versions=target_version_count,
                target_runs=len(target_run_ids),
                target_params=total_params,
                target_metrics=total_metrics,
                target_artifacts=total_artifacts,
                migration_status=status,
                migration_comments=comments,
                target_model_url=target_model_url,
                target_experiment_urls=target_exp_urls,
            )
        except Exception as exc:
            self.logger.warning(f"Failed to update tracking table for {model_name}: {exc}")

    def _get_source_migratable(self, model_name: str) -> int:
        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.getActiveSession()
            if not spark:
                return 0
            escaped_host = self.source_credentials.normalized_host().replace("'", "''")
            escaped_name = model_name.replace("'", "''")
            row = spark.sql(f"""
                SELECT source_versions_migratable
                FROM {self.tracking_table}
                WHERE source_host = '{escaped_host}' AND model_name = '{escaped_name}'
            """).first()
            return row["source_versions_migratable"] if row else 0
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_migrator(
    source_host: str,
    source_token: str | None = None,
    tracking_uri: str | None = None,
    registry_uri: str = "databricks",
    source_client_id: str | None = None,
    source_client_secret: str | None = None,
    tracking_table: str | None = None,
    **option_overrides: Any,
) -> WorkspaceRegistryMigrator:
    """Convenience builder for notebook use."""
    credentials = SourceWorkspaceCredentials(
        host=source_host,
        token=source_token,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
        client_id=source_client_id,
        client_secret=source_client_secret,
    )
    credentials.auth_type()
    options = MigrationOptions(**option_overrides)
    return WorkspaceRegistryMigrator(
        source_credentials=credentials,
        options=options,
        tracking_table=tracking_table,
    )
