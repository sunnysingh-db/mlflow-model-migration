from __future__ import annotations

from dataclasses import dataclass

from mlflow.entities import Experiment, Run
from mlflow.entities.model_registry import ModelVersion, RegisteredModel

from workspace_registry_migrator.clients import WorkspaceClients
from workspace_registry_migrator.config import MigrationOptions
from workspace_registry_migrator.utils import NotebookLogger


@dataclass(frozen=True)
class DiscoveryBundle:
    """Collected source workspace registry assets."""

    registered_models: list[RegisteredModel]
    model_versions_by_name: dict[str, list[ModelVersion]]
    experiments: list[Experiment]
    runs_by_experiment_id: dict[str, list[Run]]


class WorkspaceRegistryDiscovery:
    """Discover workspace registry models and related experiments."""

    def __init__(
        self,
        source_clients: WorkspaceClients,
        options: MigrationOptions,
        logger: NotebookLogger | None = None,
    ) -> None:
        self.source_clients = source_clients
        self.options = options
        self.logger = logger or NotebookLogger()

    def collect(self) -> DiscoveryBundle:
        """Collect all source assets required for migration."""
        registered_models = self._list_registered_models()
        model_versions_by_name = {
            model.name: self._list_model_versions(model.name) for model in registered_models
        }
        experiments = self._list_experiments(registered_models)
        runs_by_experiment_id = {
            experiment.experiment_id: self._list_runs(experiment.experiment_id)
            for experiment in experiments
        }
        return DiscoveryBundle(
            registered_models=registered_models,
            model_versions_by_name=model_versions_by_name,
            experiments=experiments,
            runs_by_experiment_id=runs_by_experiment_id,
        )

    def _list_registered_models(self) -> list[RegisteredModel]:
        self.logger.info("Discovering workspace registry models from source workspace")
        models = list(self.source_clients.mlflow_client.search_registered_models(max_results=1000))
        filtered = [model for model in models if "." not in model.name]
        requested = set(self.options.extra_model_names)
        if requested:
            filtered = [model for model in filtered if model.name in requested]
        self.logger.info(f"Discovered {len(filtered)} workspace registry models")
        return filtered

    def _list_model_versions(self, model_name: str) -> list[ModelVersion]:
        versions = list(
            self.source_clients.mlflow_client.search_model_versions(
                filter_string=f"name='{model_name}'"
            )
        )
        if self.options.max_model_versions_per_model is not None:
            versions = versions[: self.options.max_model_versions_per_model]
        return versions

    def _list_experiments(self, registered_models: list[RegisteredModel]) -> list[Experiment]:
        if not self.options.migrate_experiments:
            return []
        experiment_ids = set(self.options.extra_experiment_ids)
        for model in registered_models:
            for version in self._list_model_versions(model.name):
                if version.run_id:
                    run = self.source_clients.mlflow_client.get_run(version.run_id)
                    experiment_ids.add(run.info.experiment_id)
        experiments = [
            self.source_clients.mlflow_client.get_experiment(experiment_id)
            for experiment_id in sorted(experiment_ids)
        ]
        self.logger.info(f"Discovered {len(experiments)} source experiments")
        return experiments

    def _list_runs(self, experiment_id: str) -> list[Run]:
        runs = list(
            self.source_clients.mlflow_client.search_runs(
                experiment_ids=[experiment_id],
                max_results=self.options.max_runs_per_experiment,
                run_view_type=3 if self.options.include_deleted_runs else 1,
                order_by=["attributes.start_time DESC"],
            )
        )
        return runs
