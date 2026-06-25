from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SourceWorkspaceCredentials:
    """Connection details for the source Databricks workspace."""

    host: str
    token: str
    tracking_uri: Optional[str] = None
    registry_uri: str = "databricks"
    workspace_label: str = "source"

    def normalized_host(self) -> str:
        """Return host without trailing slash."""
        return self.host.rstrip("/")

    def resolved_tracking_uri(self) -> str:
        """Return MLflow tracking URI for the source workspace."""
        return self.tracking_uri or "databricks"


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
    max_runs_per_experiment: Optional[int] = None
    max_model_versions_per_model: Optional[int] = None
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
