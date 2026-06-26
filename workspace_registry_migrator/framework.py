from __future__ import annotations

import functools
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any
import warnings

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow import MlflowClient
from mlflow.entities import Experiment, Run
from mlflow.entities.metric import Metric
from mlflow.entities.model_registry import ModelVersion, RegisteredModel
from mlflow.entities.param import Param
from mlflow.entities.run_tag import RunTag

from workspace_registry_migrator.utils import NotebookLogger, chunked, sanitize_name, temporary_directory


def _retry_on_rate_limit(
    max_retries: int = 5,
    initial_backoff: float = 2.0,
    backoff_factor: float = 2.0,
    retryable_status_codes: tuple[int, ...] = (429, 500, 503),
):
    """Decorator that retries a function on HTTP rate-limit (429) or transient server errors."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            backoff = initial_backoff
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    status_code = getattr(exc, "error_code", None) or getattr(exc, "status_code", None)
                    # Also check common mlflow/requests exception patterns
                    exc_str = str(exc)
                    is_retryable = (
                        (isinstance(status_code, int) and status_code in retryable_status_codes)
                        or "429" in exc_str
                        or "RESOURCE_EXHAUSTED" in exc_str
                        or "Too Many Requests" in exc_str
                        or "503" in exc_str
                        or "TEMPORARILY_UNAVAILABLE" in exc_str
                    )
                    if not is_retryable or attempt == max_retries:
                        raise
                    sleep_time = backoff + (attempt * 0.5)  # jitter
                    logging.getLogger("workspace_registry_migrator").warning(
                        f"Rate limited on {func.__name__}, retrying in {sleep_time:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(sleep_time)
                    backoff *= backoff_factor
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Module-level lock to prevent concurrent env var pollution between
# source reads and target writes in multi-threaded execution.
_mlflow_env_lock = threading.Lock()


class ThreadSafeMlflowTarget:
    """Proxy around MlflowClient that ensures target env before every call."""

    def __init__(self, client: MlflowClient, target_host: str, target_token: str | None) -> None:
        self._client = client
        self._target_host = target_host
        self._target_token = target_token

    def _ensure_target_env(self) -> None:
        import os
        os.environ.pop("DATABRICKS_HOST", None)
        os.environ.pop("DATABRICKS_TOKEN", None)
        os.environ.pop("DATABRICKS_CLIENT_ID", None)
        os.environ.pop("DATABRICKS_CLIENT_SECRET", None)
        if self._target_host:
            os.environ["DATABRICKS_HOST"] = self._target_host
        if self._target_token:
            os.environ["DATABRICKS_TOKEN"] = self._target_token
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks")

    def __getattr__(self, name: str):
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr

        @functools.wraps(attr)
        def locked_call(*args, **kwargs):
            with _mlflow_env_lock:
                self._ensure_target_env()
                return attr(*args, **kwargs)

        return locked_call


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
        """Return host without trailing slash."""
        return self.host.rstrip("/")

    def resolved_tracking_uri(self) -> str:
        """Return MLflow tracking URI for the source workspace."""
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
    download_artifacts: bool = True
    migrate_experiments: bool = True
    migrate_registered_models: bool = True
    include_run_artifacts: bool = True
    include_deleted_runs: bool = False
    max_runs_per_experiment: int | None = None
    max_model_versions_per_model: int | None = None
    create_missing_experiments: bool = True
    skip_existing_model_versions: bool = False
    artifact_temp_dir: str = "/local_disk0/tmp/workspace_registry_migration"
    extra_model_names: list[str] = field(default_factory=list)
    extra_experiment_ids: list[str] = field(default_factory=list)

    def validate(self) -> None:
        """Validate runtime options."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if self.max_workers > 64:
            raise ValueError("max_workers must be 64 or less")


@dataclass(frozen=True)
class DiscoveryBundle:
    """Collected source workspace registry assets."""

    registered_models: list[RegisteredModel]
    model_versions_by_name: dict[str, list[ModelVersion]]
    experiments: list[Experiment]
    runs_by_experiment_id: dict[str, list[Run]]


@dataclass(frozen=True)
class MigrationSummary:
    """High-level migration results."""

    migrated_models: int
    migrated_model_versions: int
    migrated_experiments: int
    migrated_runs: int


class SourceWorkspaceContext:
    """Source workspace MLflow operations under explicit credentials."""

    def __init__(self, credentials: SourceWorkspaceCredentials) -> None:
        self.credentials = credentials
        if credentials.token:
            self.workspace = WorkspaceClient(
                host=credentials.normalized_host(),
                token=credentials.token,
            )
        else:
            self.workspace = WorkspaceClient(
                host=credentials.normalized_host(),
                client_id=credentials.client_id,
                client_secret=credentials.client_secret,
            )

    def _client(self) -> MlflowClient:
        return MlflowClient(
            tracking_uri=self.credentials.resolved_tracking_uri(),
            registry_uri=self.credentials.registry_uri,
        )

    def _env(self) -> dict[str, str | None]:
        return {
            "DATABRICKS_HOST": self.credentials.normalized_host(),
            "DATABRICKS_TOKEN": self.credentials.token,
            "DATABRICKS_CLIENT_ID": self.credentials.client_id,
            "DATABRICKS_CLIENT_SECRET": self.credentials.client_secret,
            "MLFLOW_TRACKING_URI": self.credentials.resolved_tracking_uri(),
            "MLFLOW_REGISTRY_URI": self.credentials.registry_uri,
            "TQDM_DISABLE": "1",
            "MLFLOW_ENABLE_TQDM": "false",
            "MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR": "false",
        }

    def _apply_env(self) -> dict[str, str | None]:
        import os

        previous = {key: os.getenv(key) for key in self._env()}
        for key, value in self._env().items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        mlflow.set_tracking_uri(self.credentials.resolved_tracking_uri())
        mlflow.set_registry_uri(self.credentials.registry_uri)
        return previous

    def _restore_env(self, previous: dict[str, str | None]) -> None:
        import os

        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        mlflow.set_tracking_uri(previous.get("MLFLOW_TRACKING_URI") or "databricks")
        mlflow.set_registry_uri(previous.get("MLFLOW_REGISTRY_URI") or "databricks")

    def with_client(self, callback):
        with _mlflow_env_lock:
            previous = self._apply_env()
            try:
                return callback(self._client())
            finally:
                self._restore_env(previous)

    def search_registered_models(self) -> list[RegisteredModel]:
        def _load(client: MlflowClient) -> list[RegisteredModel]:
            models: list[RegisteredModel] = []
            page_token: str | None = None
            while True:
                page = client.search_registered_models(max_results=100, page_token=page_token)
                models.extend(list(page))
                page_token = getattr(page, "token", None)
                if not page_token:
                    return models

        return self.with_client(_load)

    def search_model_versions(self, model_name: str) -> list[ModelVersion]:
        escaped_name = model_name.replace("'", "''")
        return self.with_client(
            lambda client: list(
                client.search_model_versions(
                    filter_string=f"name='{escaped_name}'",
                    max_results=10000,
                )
            )
        )

    def get_run(self, run_id: str) -> Run:
        return self.with_client(lambda client: client.get_run(run_id))

    def get_experiment(self, experiment_id: str) -> Experiment:
        return self.with_client(lambda client: client.get_experiment(experiment_id))

    def search_runs(
        self,
        experiment_id: str,
        include_deleted_runs: bool,
        max_runs: int | None,
    ) -> list[Run]:
        return self.with_client(
            lambda client: list(
                client.search_runs(
                    experiment_ids=[experiment_id],
                    run_view_type=3 if include_deleted_runs else 1,
                    max_results=max_runs or 1000,
                    order_by=["attributes.start_time DESC"],
                )
            )
        )

    def get_metric_history(self, run_id: str, key: str) -> list[Metric]:
        return self.with_client(lambda client: client.get_metric_history(run_id, key))

    def download_artifacts(self, run_id: str, dst_path: str, artifact_path: str | None = None) -> str:
        def _download(_: MlflowClient) -> str:
            return mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=artifact_path,
                dst_path=dst_path,
                tracking_uri=self.credentials.resolved_tracking_uri(),
                registry_uri=self.credentials.registry_uri,
            )

        return self.with_client(_download)


class WorkspaceRegistryMigrator:
    """Bulk migrator for Databricks workspace MLflow registry assets."""

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
        self.source = SourceWorkspaceContext(source_credentials)
        self.target_workspace = WorkspaceClient()
        self._target_host = self.target_workspace.config.host
        self._target_token = self.target_workspace.config.token
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks")
        _raw_target = MlflowClient(tracking_uri="databricks", registry_uri="databricks")
        self.target = ThreadSafeMlflowTarget(_raw_target, self._target_host, self._target_token)

    def _configure_runtime_noise(self) -> None:
        logging.getLogger("mlflow").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")

    def discover(self) -> DiscoveryBundle:
        """Return source workspace inventory without migrating."""
        registered_models = self._list_registered_models()
        model_versions_by_name: dict[str, list[ModelVersion]] = {}
        if registered_models:
            with ThreadPoolExecutor(max_workers=min(self.options.max_workers, len(registered_models))) as pool:
                future_map = {
                    pool.submit(self._list_model_versions, model.name): model.name
                    for model in registered_models
                }
                for future in as_completed(future_map):
                    model_versions_by_name[future_map[future]] = future.result()
        experiments = self._list_experiments(registered_models, model_versions_by_name)
        runs_by_experiment_id: dict[str, list[Run]] = {}
        if experiments:
            with ThreadPoolExecutor(max_workers=min(self.options.max_workers, len(experiments))) as pool:
                future_map = {
                    pool.submit(self._list_runs, experiment.experiment_id): experiment.experiment_id
                    for experiment in experiments
                }
                for future in as_completed(future_map):
                    runs_by_experiment_id[future_map[future]] = future.result()
        return DiscoveryBundle(
            registered_models=registered_models,
            model_versions_by_name=model_versions_by_name,
            experiments=experiments,
            runs_by_experiment_id=runs_by_experiment_id,
        )

    def migrate_all(self) -> MigrationSummary:
        """Migrate experiments and workspace registry models into current workspace."""
        bundle = self.discover()
        experiment_name_map = self._migrate_experiments(bundle)
        model_counts = self._migrate_models(bundle, experiment_name_map)
        return MigrationSummary(
            migrated_models=model_counts["models"],
            migrated_model_versions=model_counts["versions"],
            migrated_experiments=len(experiment_name_map),
            migrated_runs=model_counts["runs"],
        )

    def _list_registered_models(self) -> list[RegisteredModel]:
        models = self.source.search_registered_models()
        filtered = [model for model in models if "." not in model.name]
        requested = {name.strip() for name in self.options.extra_model_names}
        if requested:
            filtered = [model for model in filtered if model.name.strip() in requested]
        for model in filtered:
            self.logger.info(f"Discovered model {model.name}")
        return filtered

    def _list_model_versions(self, model_name: str) -> list[ModelVersion]:
        versions = self.source.search_model_versions(model_name)
        if self.options.max_model_versions_per_model is not None:
            versions = versions[: self.options.max_model_versions_per_model]
        return versions

    def _list_experiments(
        self,
        registered_models: list[RegisteredModel],
        model_versions_by_name: dict[str, list[ModelVersion]],
    ) -> list[Experiment]:
        if not self.options.migrate_experiments:
            return []
        experiment_ids = set(self.options.extra_experiment_ids)
        for model in registered_models:
            for version in model_versions_by_name.get(model.name, []):
                if not version.run_id:
                    continue
                try:
                    run = self.source.get_run(version.run_id)
                except Exception as exc:
                    self.logger.warning(
                        f"Skipping model version {model.name} v{version.version} during discovery because run {version.run_id} is unavailable: {exc}"
                    )
                    continue
                experiment_ids.add(run.info.experiment_id)
        experiments: list[Experiment] = []
        for experiment_id in sorted(experiment_ids):
            try:
                experiments.append(self.source.get_experiment(experiment_id))
            except Exception as exc:
                self.logger.warning(
                    f"Skipping experiment {experiment_id} during discovery because it is unavailable: {exc}"
                )
        self.logger.info(f"Discovered {len(experiments)} source experiments")
        return experiments

    def _list_runs(self, experiment_id: str) -> list[Run]:
        return self.source.search_runs(
            experiment_id=experiment_id,
            include_deleted_runs=self.options.include_deleted_runs,
            max_runs=self.options.max_runs_per_experiment,
        )

    def _migrate_experiments(self, bundle: DiscoveryBundle) -> dict[str, str]:
        experiment_name_map: dict[str, str] = {}
        for batch in chunked(bundle.experiments, self.options.batch_size):
            self.logger.info(f"Migrating experiment batch of size {len(batch)}")
            with ThreadPoolExecutor(max_workers=min(self.options.max_workers, len(batch) or 1)) as pool:
                future_map = {
                    pool.submit(self._migrate_single_experiment, experiment, bundle.runs_by_experiment_id): experiment
                    for experiment in batch
                }
                for future in as_completed(future_map):
                    experiment = future_map[future]
                    target_name = future.result()
                    experiment_name_map[experiment.experiment_id] = target_name
        return experiment_name_map

    def _migrate_single_experiment(
        self,
        experiment: Experiment,
        runs_by_experiment_id: dict[str, list[Run]],
    ) -> str:
        target_experiment_name = self._shared_experiment_name(experiment)
        if self.options.create_missing_experiments:
            self._ensure_target_experiment(target_experiment_name, experiment)
        target_experiment = self.target.get_experiment_by_name(target_experiment_name)
        if target_experiment is None:
            raise ValueError(f"Target experiment not found: {target_experiment_name}")
        runs = runs_by_experiment_id.get(experiment.experiment_id, [])
        if runs:
            with ThreadPoolExecutor(max_workers=min(self.options.max_workers, len(runs))) as pool:
                futures = [
                    pool.submit(self._clone_run, run, target_experiment.experiment_id)
                    for run in runs
                ]
                for future in as_completed(futures):
                    future.result()
        self.logger.info(f"Migrated experiment {experiment.name} to {target_experiment_name}")
        return target_experiment_name

    def _migrate_models(
        self,
        bundle: DiscoveryBundle,
        experiment_name_map: dict[str, str],
    ) -> dict[str, int]:
        if not self.options.migrate_registered_models:
            return {"models": 0, "versions": 0, "runs": 0}
        migrated_models = 0
        migrated_versions = 0
        migrated_runs = 0
        for batch in chunked(bundle.registered_models, self.options.batch_size):
            self.logger.info(f"Migrating model batch of size {len(batch)}")
            with ThreadPoolExecutor(max_workers=min(self.options.max_workers, len(batch) or 1)) as pool:
                future_map = {
                    pool.submit(
                        self._migrate_single_registered_model,
                        model,
                        bundle.model_versions_by_name.get(model.name, []),
                        experiment_name_map,
                    ): model
                    for model in batch
                }
                for future in as_completed(future_map):
                    counts = future.result()
                    migrated_models += counts["models"]
                    migrated_versions += counts["versions"]
                    migrated_runs += counts["runs"]
                    # Real-time Delta tracking update (main thread — Spark SQL safe)
                    if self.tracking_table:
                        model_name = counts.get("_model_name", "")
                        self._update_tracking_table(
                            model_name=model_name,
                            migrated_versions=counts["versions"],
                            migrated_runs=counts["runs"],
                            failed_versions=[s for s in self._skipped_versions if s["model"] == model_name],
                        )
        return {"models": migrated_models, "versions": migrated_versions, "runs": migrated_runs}

    def _migrate_single_registered_model(
        self,
        model: RegisteredModel,
        versions: list[ModelVersion],
        experiment_name_map: dict[str, str],
    ) -> dict[str, int]:
        target_model_name = f"{self.options.model_name_prefix}{model.name}"
        self._ensure_registered_model(target_model_name, model)
        candidate_versions = [
            version
            for version in versions
            if not (
                self.options.skip_existing_model_versions
                and self._target_model_version_exists(
                    target_model_name=target_model_name,
                    source_version=version.version,
                )
            )
        ]
        migrated_versions = 0
        migrated_runs = 0
        if candidate_versions:
            with ThreadPoolExecutor(max_workers=min(self.options.max_workers, len(candidate_versions))) as pool:
                future_map = {
                    pool.submit(
                        self._clone_model_version,
                        source_model_name=model.name,
                        target_model_name=target_model_name,
                        version=version,
                        experiment_name_map=experiment_name_map,
                    ): version
                    for version in candidate_versions
                }
                for future in as_completed(future_map):
                    version = future_map[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        self.logger.warning(
                            f"Skipping model version {model.name} v{version.version} because migration failed: {exc}"
                        )
                        self._skipped_versions.append({
                            "model": model.name,
                            "version": str(version.version),
                            "reason": str(exc)[:120],
                        })
                        continue
                    migrated_versions += 1
                    migrated_runs += result["runs"]
        self.logger.info(
            f"Migrated registered model {model.name} to {target_model_name} with {migrated_versions} versions"
        )
        return {"models": 1, "versions": migrated_versions, "runs": migrated_runs, "_model_name": model.name}

    def _update_tracking_table(
        self,
        model_name: str,
        migrated_versions: int,
        migrated_runs: int,
        failed_versions: list[dict[str, str]],
    ) -> None:
        """Update the Delta tracking table after a model completes migration."""
        try:
            from workspace_registry_migrator.reporting import update_tracking_after_model
            import re

            target_model_name = f"{self.options.model_name_prefix}{model_name}"
            escaped = target_model_name.replace("'", "''")
            all_target_versions = self.target.search_model_versions(filter_string=f"name='{escaped}'")

            # BUG 1+2 FIX: Only count versions created by migration (tagged with source_model_version)
            migrated_target_versions = [
                tv for tv in all_target_versions
                if (tv.tags or {}).get("source_model_version")
            ]
            total_params = total_metrics = total_artifacts = 0
            target_experiment_ids: set[str] = set()
            for tv in migrated_target_versions:
                if tv.run_id:
                    try:
                        run = self.target.get_run(tv.run_id)
                        total_params += len(run.data.params)
                        total_metrics += len(run.data.metrics)
                        total_artifacts += len(self.target.list_artifacts(run_id=tv.run_id, path="model"))
                        target_experiment_ids.add(run.info.experiment_id)
                    except Exception:
                        pass

            # BUG 3 FIX: Determine status by comparing migrated count to source_versions_migratable
            target_version_count = len(migrated_target_versions)
            source_migratable = self._get_source_migratable(model_name)
            if target_version_count >= source_migratable and source_migratable > 0:
                status = "COMPLETED"
            elif target_version_count > 0:
                status = "PARTIAL"
            elif failed_versions:
                status = "FAILED"
            else:
                status = "SKIPPED"

            comments = "; ".join(f["reason"][:60] for f in failed_versions) if failed_versions else None

            # Build target URLs
            host = self._target_host.rstrip("/")
            # Extract workspace ID from Azure host: adb-<workspace_id>.<num>.azuredatabricks.net
            ws_id_match = re.search(r"adb-(\d+)", host)
            ws_id = ws_id_match.group(1) if ws_id_match else ""
            o_param = f"?o={ws_id}" if ws_id else ""

            target_model_url = f"{host}/ml/models/{target_model_name}{o_param}"
            target_exp_urls = " | ".join(
                f"{host}/ml/experiments/{eid}" for eid in sorted(target_experiment_ids)
            ) if target_experiment_ids else None

            update_tracking_after_model(
                tracking_table=self.tracking_table,
                source_host=self.source_credentials.normalized_host(),
                model_name=model_name,
                target_versions=target_version_count,
                target_runs=migrated_runs,
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
        """Read source_versions_migratable from the tracking table for status comparison."""
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
            if not spark:
                return 0
            escaped_host = self.source_credentials.normalized_host().replace("'", "''")
            escaped_name = model_name.replace("'", "''")
            row = spark.sql(f"""
                SELECT source_versions_migratable FROM {self.tracking_table}
                WHERE source_host = '{escaped_host}' AND model_name = '{escaped_name}'
            """).first()
            return row["source_versions_migratable"] if row else 0
        except Exception:
            return 0

    @_retry_on_rate_limit(max_retries=5, initial_backoff=2.0)
    def _clone_model_version(
        self,
        source_model_name: str,
        target_model_name: str,
        version: ModelVersion,
        experiment_name_map: dict[str, str],
    ) -> dict[str, int]:
        if not version.run_id:
            raise ValueError(f"Model version {source_model_name} v{version.version} has no run_id")
        source_run = self.source.get_run(version.run_id)
        target_run_id = self._ensure_target_run_for_source_run(source_run, experiment_name_map)
        artifact_path = self._source_model_artifact_path(version)
        self._ensure_target_model_artifacts(
            source_run_id=source_run.info.run_id,
            target_run_id=target_run_id,
            artifact_path=artifact_path,
        )
        # Workspace registry requires full dbfs:/ path for model source
        target_run_info = self.target.get_run(target_run_id)
        model_source = f"{target_run_info.info.artifact_uri}/{artifact_path}"
        created_version = self.target.create_model_version(
            name=target_model_name,
            source=model_source,
            run_id=target_run_id,
            description=version.description,
            tags={
                "source_workspace_host": self.source_credentials.normalized_host(),
                "source_model_name": source_model_name,
                "source_model_version": str(version.version),
            },
        )
        for tag_key, tag_value in (version.tags or {}).items():
            self.target.set_model_version_tag(
                name=target_model_name,
                version=created_version.version,
                key=tag_key,
                value=tag_value,
            )
        if getattr(version, "current_stage", None):
            try:
                self.target.transition_model_version_stage(
                    name=target_model_name,
                    version=created_version.version,
                    stage=version.current_stage,
                    archive_existing_versions=False,
                )
            except Exception as exc:
                self.logger.warning(
                    f"Could not apply stage {version.current_stage} to {target_model_name} v{created_version.version}: {exc}"
                )
        return {"runs": 1}

    def _ensure_target_run_for_source_run(
        self,
        source_run: Run,
        experiment_name_map: dict[str, str],
    ) -> str:
        target_experiment_name = experiment_name_map.get(source_run.info.experiment_id)
        if target_experiment_name is None:
            source_experiment = self.source.get_experiment(source_run.info.experiment_id)
            target_experiment_name = self._shared_experiment_name(source_experiment)
            self._ensure_target_experiment(target_experiment_name, source_experiment)
        target_experiment = self.target.get_experiment_by_name(target_experiment_name)
        if target_experiment is None:
            raise ValueError(f"Missing target experiment {target_experiment_name}")
        return self._clone_run(source_run, target_experiment.experiment_id)

    @_retry_on_rate_limit(max_retries=5, initial_backoff=2.0)
    def _clone_run(self, source_run: Run, target_experiment_id: str) -> str:
        source_run_id = source_run.info.run_id
        existing = self._find_existing_target_run(source_run_id, target_experiment_id)
        if existing:
            return existing
        tags = {k: v for k, v in source_run.data.tags.items() if not k.startswith("mlflow.")}
        tags["source_workspace_host"] = self.source_credentials.normalized_host()
        tags["source_run_id"] = source_run_id
        created = self.target.create_run(
            experiment_id=target_experiment_id,
            start_time=source_run.info.start_time,
            tags=tags,
            run_name=source_run.data.tags.get("mlflow.runName"),
        )
        target_run_id = created.info.run_id

        params = [Param(key=key, value=value) for key, value in source_run.data.params.items()]
        metric_history = self._collect_metric_history(source_run)
        metrics = [
            Metric(
                key=metric.key,
                value=metric.value,
                timestamp=metric.timestamp,
                step=metric.step,
            )
            for metric in metric_history
        ]
        run_tags = [RunTag(key=key, value=value) for key, value in tags.items()]
        self.target.log_batch(run_id=target_run_id, params=params, metrics=metrics, tags=run_tags)

        if self.options.include_run_artifacts:
            self._copy_run_artifacts(source_run_id=source_run_id, target_run_id=target_run_id)

        self.target.set_terminated(
            run_id=target_run_id,
            status=source_run.info.status,
            end_time=source_run.info.end_time,
        )
        return target_run_id

    def _collect_metric_history(self, source_run: Run) -> list[Metric]:
        history: list[Metric] = []
        for metric_key in source_run.data.metrics:
            metric_points = self.source.get_metric_history(source_run.info.run_id, metric_key)
            if metric_points:
                history.extend(metric_points)
            else:
                history.append(
                    Metric(
                        key=metric_key,
                        value=float(source_run.data.metrics[metric_key]),
                        timestamp=source_run.info.end_time or source_run.info.start_time or 0,
                        step=0,
                    )
                )
        return history

    def _copy_run_artifacts(self, source_run_id: str, target_run_id: str) -> None:
        if not self.options.download_artifacts:
            return
        # List top-level artifacts first; skip runs with empty/corrupt artifact paths
        try:
            top_level = self.source.with_client(
                lambda client: client.list_artifacts(run_id=source_run_id)
            )
        except Exception as exc:
            self.logger.warning(f"Cannot list artifacts for run {source_run_id}: {exc}")
            return
        if not top_level:
            return
        # Download each top-level artifact individually to isolate failures
        for artifact in top_level:
            artifact_path = artifact.path
            if not artifact_path or not artifact_path.strip():
                self.logger.warning(f"Skipping artifact with empty path in run {source_run_id}")
                continue
            try:
                with temporary_directory(
                    prefix=f"artifacts_{sanitize_name(source_run_id)}_",
                    parent_dir=self.options.artifact_temp_dir,
                ) as temp_dir:
                    downloaded_path = Path(
                        self.source.download_artifacts(
                            run_id=source_run_id,
                            dst_path=temp_dir,
                            artifact_path=artifact_path,
                        )
                    )
                    if downloaded_path.is_file():
                        parent = str(Path(artifact_path).parent)
                        self.target.log_artifact(
                            run_id=target_run_id,
                            local_path=str(downloaded_path),
                            artifact_path=None if parent == "." else parent,
                        )
                    elif downloaded_path.is_dir():
                        self.target.log_artifacts(
                            run_id=target_run_id,
                            local_dir=str(downloaded_path),
                            artifact_path=artifact_path,
                        )
            except Exception as exc:
                self.logger.warning(
                    f"Skipping artifact '{artifact_path}' in run {source_run_id}: {exc}"
                )

    def _source_model_artifact_path(self, version: ModelVersion) -> str:
        source = getattr(version, "source", None) or ""
        if "/artifacts/" in source:
            return source.split("/artifacts/", 1)[1].strip("/") or "model"
        return "model"

    def _target_artifact_exists(self, target_run_id: str, artifact_path: str) -> bool:
        try:
            return len(self.target.list_artifacts(run_id=target_run_id, path=artifact_path)) > 0
        except Exception:
            return False

    def _ensure_target_model_artifacts(
        self,
        source_run_id: str,
        target_run_id: str,
        artifact_path: str,
    ) -> None:
        if self._target_artifact_exists(target_run_id, artifact_path):
            return
        with temporary_directory(
            prefix=f"model_{sanitize_name(source_run_id)}_",
            parent_dir=self.options.artifact_temp_dir,
        ) as temp_dir:
            downloaded_path = Path(
                self.source.download_artifacts(
                    run_id=source_run_id,
                    artifact_path=artifact_path,
                    dst_path=temp_dir,
                )
            )
            if downloaded_path.is_file():
                target_path = str(Path(artifact_path).parent)
                self.target.log_artifact(
                    run_id=target_run_id,
                    local_path=str(downloaded_path),
                    artifact_path=None if target_path == "." else target_path,
                )
                return
            root = downloaded_path if downloaded_path.is_dir() else Path(temp_dir)
            self.target.log_artifacts(
                run_id=target_run_id,
                local_dir=str(root),
                artifact_path=artifact_path,
            )

    def _ensure_target_experiment(self, experiment_name: str, source_experiment: Experiment) -> None:
        target = self.target.get_experiment_by_name(experiment_name)
        if target is None:
            experiment_id = self.target.create_experiment(
                experiment_name,
                tags={
                    "source_workspace_host": self.source_credentials.normalized_host(),
                    "source_experiment_id": source_experiment.experiment_id,
                },
            )
            target = self.target.get_experiment(experiment_id)
        for tag_key, tag_value in (source_experiment.tags or {}).items():
            if not tag_key.startswith("mlflow."):
                self.target.set_experiment_tag(target.experiment_id, tag_key, tag_value)

    def _ensure_registered_model(self, target_model_name: str, source_model: RegisteredModel) -> None:
        try:
            existing = self.target.get_registered_model(target_model_name)
        except Exception:
            existing = None
        if existing is None:
            self.target.create_registered_model(
                name=target_model_name,
                description=source_model.description,
                tags={
                    "source_workspace_host": self.source_credentials.normalized_host(),
                    "source_model_name": source_model.name,
                },
            )
        for tag_key, tag_value in (source_model.tags or {}).items():
            self.target.set_registered_model_tag(target_model_name, tag_key, tag_value)

    def _target_model_version_exists(self, target_model_name: str, source_version: str | int) -> bool:
        escaped_name = target_model_name.replace("'", "''")
        versions = self.target.search_model_versions(filter_string=f"name='{escaped_name}'")
        return any(
            (version.tags or {}).get("source_model_version") == str(source_version)
            for version in versions
        )

    def _find_existing_target_run(self, source_run_id: str, target_experiment_id: str) -> str | None:
        runs = self.target.search_runs(
            experiment_ids=[target_experiment_id],
            filter_string=f"tags.source_run_id = '{source_run_id}'",
            max_results=1,
        )
        if not runs:
            return None
        return runs[0].info.run_id

    def _shared_experiment_name(self, experiment: Experiment) -> str:
        base_name = experiment.name.rstrip("/").split("/")[-1]
        prefix = self.options.experiment_name_prefix or ""
        return f"{self.options.shared_experiment_root}/{prefix}{base_name}"


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
