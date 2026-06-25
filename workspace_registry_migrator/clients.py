from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow import MlflowClient

from workspace_registry_migrator.config import SourceWorkspaceCredentials


@dataclass(frozen=True)
class WorkspaceClients:
    """Databricks and MLflow clients bound to one workspace."""

    workspace: WorkspaceClient
    mlflow_client: MlflowClient
    host: str
    tracking_uri: str
    registry_uri: str


def create_source_clients(credentials: SourceWorkspaceCredentials) -> WorkspaceClients:
    """Build source workspace clients using explicit credentials."""
    workspace = WorkspaceClient(host=credentials.normalized_host(), token=credentials.token)
    with mlflow_environment(
        host=credentials.normalized_host(),
        token=credentials.token,
        tracking_uri=credentials.resolved_tracking_uri(),
        registry_uri=credentials.registry_uri,
    ):
        mlflow_client = MlflowClient(
            tracking_uri=credentials.resolved_tracking_uri(),
            registry_uri=credentials.registry_uri,
        )
    return WorkspaceClients(
        workspace=workspace,
        mlflow_client=mlflow_client,
        host=credentials.normalized_host(),
        tracking_uri=credentials.resolved_tracking_uri(),
        registry_uri=credentials.registry_uri,
    )


def create_target_clients() -> WorkspaceClients:
    """Build clients for the current workspace runtime context."""
    workspace = WorkspaceClient()
    mlflow.set_registry_uri("databricks")
    mlflow_client = MlflowClient(tracking_uri="databricks", registry_uri="databricks")
    return WorkspaceClients(
        workspace=workspace,
        mlflow_client=mlflow_client,
        host=workspace.config.host,
        tracking_uri="databricks",
        registry_uri="databricks",
    )


@contextmanager
def mlflow_environment(
    host: str,
    token: str,
    tracking_uri: str,
    registry_uri: str,
) -> Iterator[None]:
    """Temporarily set MLflow environment variables for a workspace."""
    previous = {
        "DATABRICKS_HOST": os.getenv("DATABRICKS_HOST"),
        "DATABRICKS_TOKEN": os.getenv("DATABRICKS_TOKEN"),
        "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI"),
        "MLFLOW_REGISTRY_URI": os.getenv("MLFLOW_REGISTRY_URI"),
        "TQDM_DISABLE": os.getenv("TQDM_DISABLE"),
    }
    os.environ["DATABRICKS_HOST"] = host
    os.environ["DATABRICKS_TOKEN"] = token
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_REGISTRY_URI"] = registry_uri
    os.environ["TQDM_DISABLE"] = "1"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        mlflow.set_tracking_uri(previous["MLFLOW_TRACKING_URI"] or "databricks")
        mlflow.set_registry_uri(previous["MLFLOW_REGISTRY_URI"] or "databricks")
