from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import Experiment, Run
from mlflow.entities.model_registry import ModelVersion, RegisteredModel
from mlflow.tracking import MlflowClient

from workspace_registry_migrator.clients import (
    WorkspaceClients,
    create_source_clients,
    create_target_clients,
    mlflow_environment,
)
from workspace_registry_migrator.config import MigrationOptions, SourceWorkspaceCredentials
from workspace_registry_migrator.discovery import DiscoveryBundle, WorkspaceRegistryDiscovery
from workspace_registry_migrator.utils import NotebookLogger, chunked, ensure_directory, sanitize_name, temporary_directory


@dataclass(frozen=True)
class MigrationSummary:
    """High-level migration results."""

    migrated_models: int
    migrated_model_versions: int
    migrated_experiments: int
    migrated_runs: int


class WorkspaceRegistryMigrator:
    """Bulk migrator for Databricks workspace MLflow registry assets."""

    def __init__(
        self,
        source_credentials: SourceWorkspaceCredentials,
        options: MigrationOptions,
        logger: NotebookLogger | None = None,
    ) -> None:
        options.validate()
        self.source_credentials = source_credentials
        self.options = options
        self.logger = logger or NotebookLogger()
        self.source_clients = create_source_clients(source_credentials)
        self.target_clients = create_target_clients()
        self.discovery = WorkspaceRegistryDiscovery(
            source_clients=self.source_clients,
            options=self.options,
            logger=self.logger,
        )

    def discover(self) -> DiscoveryBundle:
        """Return source workspace inventory without migrating."""
        return self.discovery.collect()

    def migrate_all(self) -> MigrationSummary:
        """Migrate experiments and workspace registry models into current workspace."""
        bundle = self.discovery.collect()
        experiment_name_map = self._migrate_experiments(bundle)
        model_counts = self._migrate_models(bundle, experiment_name_map)
        return MigrationSummary(
            migrated_models=model_counts["models"],
            migrated_model_versions=model_counts["versions"],
            migrated_experiments=len(experiment_name_map),
            migrated_runs=model_counts["runs"],
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
            self._ensure_target_experiment(target_experiment_name)
        target_experiment = self.target_clients.mlflow_client.get_experiment_by_name(target_experiment_name)
        if target_experiment is None:
            raise ValueError(f"Target experiment not found: {target_experiment_name}")
        migrated_runs = 0
        for run in runs_by_experiment_id.get(experiment.experiment_id, []):
            self._clone_run(run, target_experiment.experiment_id)
            migrated_runs += 1
        self.logger.info(
            f"Migrated experiment {experiment.name} to {target_experiment_name} with {migrated_runs} runs"
        )
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
        return {"models": migrated_models, "versions": migrated_versions, "runs": migrated_runs}

    def _migrate_single_registered_model(
        self,
        model: RegisteredModel,
        versions: list[ModelVersion],
        experiment_name_map: dict[str, str],
    ) -> dict[str, int]:
        target_model_name = f"{self.options.model_name_prefix}{model.name}"
        self._ensure_registered_model(target_model_name, model)
        migrated_versions = 0
        migrated_runs = 0
        for version in versions:
            if self.options.skip_existing_model_versions and self._target_model_version_exists(
                target_model_name=target_model_name,
                source_version=version.version,
            ):
                continue
            result = self._clone_model_version(
                source_model_name=model.name,
                target_model_name=target_model_name,
                version=version,
                experiment_name_map=experiment_name_map,
            )
            migrated_versions += 1
            migrated_runs += result["runs"]
        self.logger.info(
            f"Migrated registered model {model.name} to {target_model_name} with {migrated_versions} versions"
        )
        return {"models": 1, "versions": migrated_versions, "runs": migrated_runs}

    def _clone_model_version(
        self,
        source_model_name: str,
        target_model_name: str,
        version: ModelVersion,
        experiment_name_map: dict[str, str],
    ) -> dict[str, int]:
        if not version.run_id:
            raise ValueError(f"Model version {source_model_name} v{version.version} has no run_id")
        source_run = self.source_clients.mlflow_client.get_run(version.run_id)
        target_run_id = self._ensure_target_run_for_source_run(source_run, experiment_name_map)
        runs_model_uri = f"runs:/{target_run_id}/model"
        created_version = self.target_clients.mlflow_client.create_model_version(
            name=target_model_name,
            source=runs_model_uri,
            run_id=target_run_id,
            description=version.description,
            tags={
                "source_workspace_host": self.source_clients.host,
                "source_model_name": source_model_name,
                "source_model_version": str(version.version),
            },
        )
        for tag in version.tags or {}:
            self.target_clients.mlflow_client.set_model_version_tag(
                name=target_model_name,
                version=created_version.version,
                key=tag,
                value=(version.tags or {})[tag],
            )
        return {"runs": 1}

    def _ensure_target_run_for_source_run(
        self,
        source_run: Run,
        experiment_name_map: dict[str, str],
    ) -> str:
        target_experiment_name = experiment_name_map.get(source_run.info.experiment_id)
        if target_experiment_name is None:
            source_experiment = self.source_clients.mlflow_client.get_experiment(source_run.info.experiment_id)
            target_experiment_name = self._shared_experiment_name(source_experiment)
            self._ensure_target_experiment(target_experiment_name)
        target_experiment = self.target_clients.mlflow_client.get_experiment_by_name(target_experiment_name)
        if target_experiment is None:
            raise ValueError(f"Missing target experiment {target_experiment_name}")
        return self._clone_run(source_run, target_experiment.experiment_id)

    def _clone_run(self, source_run: Run, target_experiment_id: str) -> str:
        source_run_id = source_run.info.run_id
        existing = self._find_existing_target_run(source_run_id, target_experiment_id)
        if existing:
            return existing
        tags = dict(source_run.data.tags)
        tags["source_workspace_host"] = self.source_clients.host
        tags["source_run_id"] = source_run_id
        created = self.target_clients.mlflow_client.create_run(
            experiment_id=target_experiment_id,
            start_time=source_run.info.start_time,
            tags=tags,
            run_name=source_run.data.tags.get("mlflow.runName"),
        )
        target_run_id = created.info.run_id
        params = [{"key": key, "value": value} for key, value in source_run.data.params.items()]
        metrics = []
        for key, value in source_run.data.metrics.items():
            metrics.append(
                {
                    "key": key,
                    "value": float(value),
                    "timestamp": source_run.info.end_time or source_run.info.start_time or 0,
                    "step": 0,
                }
            )
        if params:
            self.target_clients.mlflow_client.log_batch(run_id=target_run_id, params=params, metrics=metrics, tags=[])
        elif metrics:
            self.target_clients.mlflow_client.log_batch(run_id=target_run_id, params=[], metrics=metrics, tags=[])
        if self.options.include_run_artifacts:
            self._copy_run_artifacts(source_run_id=source_run_id, target_run_id=target_run_id)
        self.target_clients.mlflow_client.set_terminated(
            run_id=target_run_id,
            status=source_run.info.status,
            end_time=source_run.info.end_time,
        )
        return target_run_id

    def _copy_run_artifacts(self, source_run_id: str, target_run_id: str) -> None:
        if not self.options.download_artifacts:
            return
        with temporary_directory(
            prefix=f"artifacts_{sanitize_name(source_run_id)}_",
            parent_dir=self.options.artifact_temp_dir,
        ) as temp_dir:
            with mlflow_environment(
                host=self.source_credentials.normalized_host(),
                token=self.source_credentials.token,
                tracking_uri=self.source_credentials.resolved_tracking_uri(),
                registry_uri=self.source_credentials.registry_uri,
            ):
                mlflow.artifacts.download_artifacts(run_id=source_run_id, dst_path=temp_dir)
            artifact_root = Path(temp_dir)
            for local_path in artifact_root.rglob("*"):
                if local_path.is_file():
                    artifact_path = str(local_path.relative_to(artifact_root.parent))
                    self.target_clients.mlflow_client.log_artifact(
                        run_id=target_run_id,
                        local_path=str(local_path),
                        artifact_path=str(Path(artifact_path).parent).replace(".", ""),
                    )

    def _ensure_target_experiment(self, experiment_name: str) -> None:
        target = self.target_clients.mlflow_client.get_experiment_by_name(experiment_name)
        if target is None:
            self.target_clients.mlflow_client.create_experiment(experiment_name)

    def _ensure_registered_model(self, target_model_name: str, source_model: RegisteredModel) -> None:
        try:
            existing = self.target_clients.mlflow_client.get_registered_model(target_model_name)
        except Exception:
            existing = None
        if existing is None:
            self.target_clients.mlflow_client.create_registered_model(
                name=target_model_name,
                description=source_model.description,
                tags={
                    "source_workspace_host": self.source_clients.host,
                    "source_model_name": source_model.name,
                },
            )

    def _target_model_version_exists(self, target_model_name: str, source_version: str | int) -> bool:
        versions = self.target_clients.mlflow_client.search_model_versions(
            filter_string=f"name='{target_model_name}'"
        )
        return any(
            (version.tags or {}).get("source_model_version") == str(source_version)
            for version in versions
        )

    def _find_existing_target_run(self, source_run_id: str, target_experiment_id: str) -> str | None:
        runs = self.target_clients.mlflow_client.search_runs(
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
    source_token: str,
    tracking_uri: str | None = None,
    registry_uri: str = "databricks",
    **option_overrides: Any,
) -> WorkspaceRegistryMigrator:
    """Convenience builder for notebook use."""
    credentials = SourceWorkspaceCredentials(
        host=source_host,
        token=source_token,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )
    options = MigrationOptions(**option_overrides)
    return WorkspaceRegistryMigrator(source_credentials=credentials, options=options)
