"""Workspace registry MLflow migration framework — v2.

v2 architecture:
- REST-based metadata operations (no env var mutation, fully thread-safe)
- Two-phase pipeline: download artifacts first, then register
- Sequential per-model version registration (preserves version ordering)
- MLflow version compatibility checking
- Unity Catalog model registry support (workspace -> UC migration)
"""

from workspace_registry_migrator.framework_v2 import (
    MigrationOptions,
    SourceWorkspaceCredentials,
    WorkspaceRegistryMigrator,
    build_migrator,
    DiscoveryBundle,
    MigrationSummary,
    StagedVersion,
)

__all__ = [
    "MigrationOptions",
    "SourceWorkspaceCredentials",
    "WorkspaceRegistryMigrator",
    "build_migrator",
    "DiscoveryBundle",
    "MigrationSummary",
    "StagedVersion",
]

