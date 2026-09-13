# MLflow Model Migration Framework — Runbook

> **Purpose:** Step-by-step operational guide for migrating MLflow models between Databricks workspaces.  
> **Audience:** Platform engineers, ML engineers, and admins who need to move MLflow models across workspaces, registries, or air-gapped environments.  
> **Last updated:** September 2026

---

## Table of Contents

1. [What This Framework Does](#1-what-this-framework-does)
2. [What Gets Migrated](#2-what-gets-migrated)
3. [Architecture Overview](#3-architecture-overview)
4. [Project Structure](#4-project-structure)
5. [Prerequisites](#5-prerequisites)
6. [Supported Migration Scenarios](#6-supported-migration-scenarios)
7. [Configuration Parameter Reference](#7-configuration-parameter-reference)
8. [How to Run — Step by Step](#8-how-to-run--step-by-step)
9. [Scenario A: Workspace to Workspace](#9-scenario-a-workspace-to-workspace)
10. [Scenario B: Workspace to Unity Catalog](#10-scenario-b-workspace-to-unity-catalog)
11. [Scenario C: Unity Catalog to Unity Catalog](#11-scenario-c-unity-catalog-to-unity-catalog)
12. [Scenario D: Unity Catalog to Workspace](#12-scenario-d-unity-catalog-to-workspace)
13. [Export / Import Mode (Air-Gapped Migrations)](#13-export--import-mode-air-gapped-migrations)
14. [Tracking Table](#14-tracking-table)
15. [Resuming a Failed or Partial Migration](#15-resuming-a-failed-or-partial-migration)
16. [Reading the Output](#16-reading-the-output)
17. [Concurrency Model and Performance Tuning](#17-concurrency-model-and-performance-tuning)
18. [Troubleshooting](#18-troubleshooting)
19. [Security Best Practices](#19-security-best-practices)
20. [FAQ](#20-faq)

---

## 1. What This Framework Does

This framework bulk-migrates MLflow registered models — along with their experiments, runs, metrics, parameters, tags, and model artifacts — from one Databricks workspace to another. It supports four migration paths across two registry types (legacy Workspace Registry and Unity Catalog), plus an export/import mode for air-gapped or cross-cloud transfers.

**Key capabilities:**

- Migrate specific models by name or bulk-scan all models in a workspace or catalog
- Supports both PAT (Personal Access Token) and Service Principal (OAuth M2M) authentication
- Tracks migration progress in a Delta table — safe to resume if interrupted
- Idempotent by design — re-running skips already-completed models
- Handles deleted source runs gracefully by creating placeholder versions
- Thread-safe REST-based operations (no environment variable mutation)
- Two-phase pipeline: download artifacts first, then register (preserves version ordering)
- Built-in retry logic with exponential backoff for rate-limited APIs

---

## 2. What Gets Migrated

| Asset | What Gets Copied |
| --- | --- |
| **Registered Models** | Name, description, tags |
| **Model Versions** | Source path, stage-to-alias mapping (Production → Champion, Staging → Challenger), tags, description |
| **Experiments** | Name, tags. Created under `/Shared/mlflow-workspace-migration/` on the target |
| **Runs** | Parameters, full metric history (all steps), tags, run status, start/end time |
| **Artifacts** | Model files (MLmodel, pkl/pth/onnx, conda.yaml, requirements.txt, etc.) and optionally all run artifacts |

**What is NOT migrated:**

- Permissions / ACLs (must be re-applied manually on the target)
- Experiment notebooks (the notebook source code that created the run)
- MLflow tracking server configuration
- Webhooks or event triggers

---

## 3. Architecture Overview

### High-Level Flow

```
┌─────────────────┐                          ┌─────────────────┐
│  SOURCE          │     REST API / SDK       │  TARGET          │
│  Workspace       │ ◄──────────────────────► │  Workspace       │
│                  │                          │  (current)       │
│  - Models        │   Phase 1: Discovery     │                  │
│  - Versions      │   ────────────────►      │  - Models        │
│  - Experiments   │                          │  - Versions      │
│  - Runs          │   Phase 2: Migration     │  - Experiments   │
│  - Artifacts     │   ────────────────►      │  - Runs          │
│                  │                          │  - Artifacts     │
│                  │   Phase 3: Verification  │                  │
│                  │   ◄───────────────►      │  - Tracking Table│
└─────────────────┘                          └─────────────────┘
```

### Phase Details

**Phase 1 — Discovery:**
- Scans the source workspace for registered models (by name or bulk)
- Fetches all model versions for each model
- Identifies linked experiments and runs
- MERGEs the inventory into the Delta tracking table (all start as PENDING)

**Phase 2 — Migration:**
- Downloads artifacts from the source to a local staging directory
- Creates experiments under `/Shared/mlflow-workspace-migration/` on the target
- Clones runs (params, metrics, tags, artifacts) into target experiments
- Registers model versions on the target, linking to the cloned runs
- Updates the tracking table per model (COMPLETED / PARTIAL / FAILED)

**Phase 3 — Verification:**
- Compares source vs. target version counts for each model
- Reports mismatches in the output table

### Authentication Model

- **Source workspace:** Accessed via explicit credentials you provide (PAT token or Service Principal client_id/secret). All source operations go through a thread-safe REST client with connection pooling.
- **Target workspace:** The current workspace where you run the notebook. Uses the notebook's built-in authentication context (no credentials needed).

---

## 4. Project Structure

```
mlflow-model-migration/
├── MLflow Model Migration Framework     ← Main notebook (run this!)
├── README.md                             ← Project overview
├── RUNBOOK.md                            ← This file
└── workspace_registry_migrator/          ← Framework Python package
    ├── __init__.py                       ← Package exports
    ├── framework_v2.py                   ← Core migration engine (v2)
    │                                       - SourceWorkspaceCredentials
    │                                       - MigrationOptions
    │                                       - WorkspaceRegistryMigrator
    │                                       - DiscoveryBundle, MigrationSummary
    │                                       - Discovery, experiment/model migration
    │                                       - Export/import bundle logic
    ├── notebook_helpers.py               ← Notebook-friendly entry point
    │                                       - execute_migration() function
    │                                       - Console-formatted progress output
    │                                       - Mode routing (direct/export/import)
    ├── rest_client.py                    ← Thread-safe REST API client
    │                                       - DatabricksRestClient class
    │                                       - PAT and OAuth M2M authentication
    │                                       - Presigned-URL artifact downloads
    │                                       - Rate-limit retry with backoff
    │                                       - Workspace and UC model registry APIs
    ├── reporting.py                      ← Delta tracking table and reporting
    │                                       - Inventory report generation
    │                                       - write_inventory_to_delta()
    │                                       - update_tracking_after_model()
    │                                       - Post-migration comparison report
    ├── utils.py                          ← Shared utilities
    │                                       - sanitize_name(), chunked()
    │                                       - NotebookLogger, temporary_directory()
    ├── bootstrap.py                      ← Package loader for GCP FUSE workaround
    ├── config.py                         ← (deprecated — logic moved to framework_v2)
    ├── clients.py                        ← (deprecated — logic moved to rest_client)
    ├── discovery.py                      ← (deprecated — logic moved to framework_v2)
    ├── migrate.py                        ← (deprecated — logic moved to framework_v2)
    └── README.md                         ← Package-level documentation
```

---

## 5. Prerequisites

### 5.1 Network Connectivity

- The **target workspace** (where you run the notebook) must be able to reach the **source workspace** REST API over HTTPS (port 443).
- If workspaces use Private Link or VNet, ensure the NSG/firewall allows outbound HTTPS to the source workspace.
- For cross-region migrations, verify VNet peering or public egress is enabled.
- For air-gapped environments where direct connectivity is not possible, use [Export / Import mode](#13-export--import-mode-air-gapped-migrations).

### 5.2 Authentication

You need credentials for the **source** workspace only. The target uses the notebook's runtime context.

| Method | When to Use | What You Need |
| --- | --- | --- |
| **PAT** (Personal Access Token) | Simplest option; good for one-time or dev migrations | Source workspace URL + PAT token |
| **Service Principal** (OAuth M2M) | Production migrations, automated pipelines, no human token needed | Source workspace URL + SP `client_id` + `client_secret` |

**How to create a PAT:** Source workspace → Settings → Developer → Access Tokens → Generate New Token.

**How to create a Service Principal:** Admin console → Service Principals → Add → Note the `client_id` and generate a `client_secret`.

### 5.3 Permissions

| Where | Required Permissions |
| --- | --- |
| **Source workspace** | Read access to MLflow Tracking (experiments, runs), Model Registry (models, versions), and artifact download |
| **Target workspace** | Write access to `/Shared` experiments, create registered models and versions |
| **UC targets** | `USE CATALOG`, `USE SCHEMA`, `CREATE MODEL` on the target catalog/schema |
| **Tracking table** | `CREATE TABLE` permission on the schema where the tracking table lives |

### 5.4 Compute

- **Serverless CPU** — recommended, works out of the box, no setup needed
- Any cluster with `mlflow` and `databricks-sdk` pre-installed
- No GPU required
- Minimum recommended: 4 cores / 16 GB RAM for large migrations (1000+ models)

---

## 6. Supported Migration Scenarios

| Scenario | Source | Target | Primary Use Case |
| --- | --- | --- | --- |
| **A** | Workspace Registry | Workspace Registry | Clone models to another workspace, or duplicate within the same workspace with a prefix |
| **B** | Workspace Registry | Unity Catalog | Upgrade legacy workspace models to UC (modernization) |
| **C** | Unity Catalog | Unity Catalog | Move UC models across catalogs, schemas, or workspaces |
| **D** | Unity Catalog | Workspace Registry | Downgrade UC models to legacy registry (rare, for backward compatibility) |

Each scenario can run in three modes:
- **`direct`** — End-to-end migration (source and target must be reachable)
- **`export`** — Discovery + artifact download + manifest generation (source only)
- **`import`** — Read manifests + create on target (target only)

---

## 7. Configuration Parameter Reference

### 7.1 Authentication Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `AUTH_MODE` | `str` | Yes | Authentication method. `"pat"` for Personal Access Token, `"service_principal"` for OAuth M2M. |
| `SOURCE_HOST` | `str` | Yes | Full URL of the source workspace. Example: `"https://adb-1234567890.1.azuredatabricks.net"` |
| `SOURCE_TOKEN` | `str` | If PAT | Personal Access Token for the source workspace. **Never hardcode** — use `dbutils.secrets.get()`. |
| `SP_CLIENT_ID` | `str` | If SP | Service Principal application (client) ID. Only used when `AUTH_MODE = "service_principal"`. |
| `SP_CLIENT_SECRET` | `str` | If SP | Service Principal client secret. Only used when `AUTH_MODE = "service_principal"`. |

### 7.2 Migration Direction

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `SOURCE_REGISTRY` | `str` | Yes | Where to read models FROM. `"workspace"` for legacy Workspace Registry, `"uc"` for Unity Catalog. |
| `TARGET_REGISTRY` | `str` | Yes | Where to write models TO. `"workspace"` for legacy Workspace Registry, `"uc"` for Unity Catalog. |

### 7.3 Model Selection

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `MODEL_NAMES` | `list[str]` | No | Specific models to migrate by name. Use full names: `"my_model"` for workspace, `"catalog.schema.model"` for UC. Set to `[]` to scan **all** visible models (bulk mode). |
| `INCLUDE_CATALOGS` | `list[str]` | No | UC bulk scan only. Whitelist — only scan these catalogs. Empty = scan all visible catalogs. |
| `EXCLUDE_CATALOGS` | `list[str]` | No | UC bulk scan only. Blacklist — skip these catalogs during scan. |
| `EXCLUDE_SCHEMAS` | `list[str]` | No | UC bulk scan only. Skip these schemas. Use `"catalog.schema"` format. |

### 7.4 Target Settings

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `UC_TARGET_CATALOG` | `str` | If UC target | Override the target catalog name. Required for WS→UC. For UC→UC, if empty, mirrors the source catalog. |
| `UC_TARGET_SCHEMA` | `str` | If UC target | Override the target schema name. Required for WS→UC. For UC→UC, if empty, mirrors the source schema. |
| `MODEL_NAME_PREFIX` | `str` | No | Prefix added to model names on the target. Useful to avoid name collisions on same-workspace migrations. Example: `"migrated_"` turns `my_model` into `migrated_my_model`. Leave blank (`""`) or `None` for no prefix — models keep their original names on the target. |

### 7.5 Migration Mode

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `MIGRATION_MODE` | `str` | No | `"direct"` (default) — full end-to-end migration. `"export"` — discover + download + write manifests (no target operations). `"import"` — read manifests + create on target (no source API calls). |

### 7.6 Artifact Staging

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `ARTIFACT_TEMP_DIR` | `str` | No | Local directory for staging downloaded artifacts. Also used as the bundle directory for export/import. Default: `"/tmp/ws_export_bundle"`. Supports: `/tmp/...`, `/Volumes/...`, `/dbfs/tmp/...`, cloud storage paths. |

### 7.7 Options

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `TRACKING_TABLE` | `str` or `None` | `None` | Fully qualified Delta table name for migration progress tracking. Example: `"catalog.schema.migration_tracking"`. Set to `None` to disable tracking (migration still works but cannot resume). |
| `INCLUDE_ARTIFACTS` | `bool` | `True` | When `True`, copies all model artifacts (MLmodel, weights, conda.yaml, etc.). Set `False` for a metadata-only migration (much faster but models won't be loadable on target). |
| `INCLUDE_DELETED` | `bool` | `False` | When `True`, includes soft-deleted (trashed) runs from the source. Rarely needed. |
| `CREATE_DUMMY_VERSIONS` | `bool` | `True` | When `True`, creates placeholder model versions on the target for source versions whose runs have been permanently deleted. Keeps version numbering consistent. |
| `BATCH_SIZE` | `int` | `10` | Number of models processed in parallel per batch. Also controls the thread pool size. Reduce to `5` if you hit rate limits (HTTP 429). |

---

## 8. How to Run — Step by Step

### Step 1: Open the Notebook

Open the **"MLflow Model Migration Framework"** notebook in your **target** workspace (the workspace you want to migrate models INTO).

### Step 2: Restart Python (Cell 2)

Run the `%restart_python` cell to ensure a clean environment with no stale module caches.

### Step 3: Choose and Configure a Scenario (Cell 3, 4, or 5)

The notebook contains three pre-built configuration cells:

| Cell | Title | Scenario |
| --- | --- | --- |
| Cell 3 | Configuration Scenario A | Workspace Registry → Workspace Registry |
| Cell 4 | Configuration Scenario B | UC → UC |
| Cell 5 | Configuration Scenario C | Export / Import mode |

**Run ONLY ONE configuration cell** — the one matching your migration scenario. Edit the parameter values in that cell before running it.

> **Important:** Running a configuration cell only sets Python variables. It does not start the migration.

### Step 4: Run Migration (Cell 6)

Run the **"Run Migration"** cell. This cell:
1. Bootstraps the framework package (handles FUSE edge cases on GCP)
2. Imports and calls `execute_migration()` with the variables from the configuration cell
3. Runs all three phases: Discovery → Migration → Verification
4. Displays a results DataFrame at the end

### Step 5: Review Results

The output table shows per-model migration status. See [Reading the Output](#16-reading-the-output) for details.

### Step 6: Resume if Needed

If the migration was interrupted or some models failed, simply **re-run the same cells** (configuration + Run Migration). Completed models are skipped automatically.

---

## 9. Scenario A: Workspace to Workspace

**Use case:** Clone workspace registry models to another workspace, or duplicate within the same workspace (with a name prefix).

**Configuration (Cell 3):**

```
AUTH_MODE        = "pat"
SOURCE_HOST      = "https://<source-workspace>.azuredatabricks.net"
SOURCE_TOKEN     = "<source-pat-token>"          # Use dbutils.secrets.get() in production

SOURCE_REGISTRY  = "workspace"
TARGET_REGISTRY  = "workspace"

MODEL_NAMES      = ["my_model_1", "my_model_2"]  # [] for all models
MODEL_NAME_PREFIX = "migrated_"                   # Avoids name collision
TRACKING_TABLE   = None                           # Or "catalog.schema.tracking"
INCLUDE_ARTIFACTS = True
BATCH_SIZE       = 10
```

**Key behaviors:**
- If `MODEL_NAME_PREFIX` is left blank (`""`) or `None`, **no prefix is added** — models are created on the target with their original names. Set an explicit prefix (e.g., `"migrated_"`) if you need to avoid name collisions on same-workspace migrations.
- Experiments are recreated under `/Shared/mlflow-workspace-migration/` on the target.
- Each target model version is tagged with `source_workspace_host`, `source_model_name`, and `source_model_version` for traceability.

**Same-workspace migration:** Set `SOURCE_HOST` to your current workspace URL and use a `MODEL_NAME_PREFIX` to differentiate.

---

## 10. Scenario B: Workspace to Unity Catalog

**Use case:** Upgrade legacy workspace registry models to Unity Catalog.

**Configuration:**

```
AUTH_MODE        = "pat"
SOURCE_HOST      = "https://<source-workspace>.azuredatabricks.net"
SOURCE_TOKEN     = "<source-pat-token>"

SOURCE_REGISTRY  = "workspace"
TARGET_REGISTRY  = "uc"

MODEL_NAMES      = ["my_model_1"]
UC_TARGET_CATALOG = "ml_catalog"
UC_TARGET_SCHEMA  = "models"
MODEL_NAME_PREFIX = ""                            # Optional prefix
TRACKING_TABLE   = "ml_catalog.models.migration_tracking"
```

**Key behaviors:**
- `UC_TARGET_CATALOG` and `UC_TARGET_SCHEMA` are **required** for this scenario.
- The framework auto-creates the catalog and schema on the target if they don't exist (requires `CREATE CATALOG`/`CREATE SCHEMA` permissions).
- Workspace model stages (Production, Staging, Archived, None) are mapped to UC aliases: Production → Champion, Staging → Challenger.

---

## 11. Scenario C: Unity Catalog to Unity Catalog

**Use case:** Move UC models across catalogs, schemas, or workspaces.

**Configuration (Cell 4):**

```
AUTH_MODE        = "pat"
SOURCE_HOST      = "https://<source-workspace>.azuredatabricks.net"
SOURCE_TOKEN     = "<source-pat-token>"

SOURCE_REGISTRY  = "uc"
TARGET_REGISTRY  = "uc"

MODEL_NAMES      = ["source_catalog.source_schema.my_model"]
UC_TARGET_CATALOG = "target_catalog"
UC_TARGET_SCHEMA  = "target_schema"
TRACKING_TABLE   = "target_catalog.target_schema.migration_tracking"
```

**Key behaviors:**
- Model names must be fully qualified: `catalog.schema.model`.
- If `UC_TARGET_CATALOG` and `UC_TARGET_SCHEMA` are left empty, the framework mirrors the source catalog/schema names on the target.
- For bulk UC scans (`MODEL_NAMES = []`), use `INCLUDE_CATALOGS` / `EXCLUDE_CATALOGS` / `EXCLUDE_SCHEMAS` to scope the scan.
- UC APIs have stricter rate limits — the framework automatically caps concurrency to 2 and adds 0.5s delays between requests.

**Same-workspace UC→UC:** Use `UC_TARGET_CATALOG` or `UC_TARGET_SCHEMA` to write to a different catalog/schema.

---

## 12. Scenario D: Unity Catalog to Workspace

**Use case:** Downgrade UC models back to the legacy workspace registry.

**Configuration:**

```
AUTH_MODE        = "pat"
SOURCE_HOST      = "https://<source-workspace>.azuredatabricks.net"
SOURCE_TOKEN     = "<source-pat-token>"

SOURCE_REGISTRY  = "uc"
TARGET_REGISTRY  = "workspace"

MODEL_NAMES      = ["catalog.schema.my_uc_model"]
MODEL_NAME_PREFIX = "from_uc_"                    # Prefix for workspace target names
TRACKING_TABLE   = "catalog.schema.tracking"
```

**Key behaviors:**
- The three-part UC name is flattened. `catalog.schema.my_model` becomes `from_uc_my_model` on the workspace registry.
- `MODEL_NAME_PREFIX` is auto-generated if left blank.

---

## 13. Export / Import Mode (Air-Gapped Migrations)

When source and target workspaces **cannot reach each other** (air-gapped, different clouds, strict firewall), use the three-step export → transfer → import workflow.

### How It Works

| Mode | Runs On | What It Does |
| --- | --- | --- |
| `"export"` | **Source** workspace | Discovers models → downloads artifacts → writes JSON manifests to `ARTIFACT_TEMP_DIR`. No target operations. |
| (manual transfer) | — | You copy the bundle directory to the target environment |
| `"import"` | **Target** workspace | Reads manifests + artifacts from `ARTIFACT_TEMP_DIR` → creates experiments, runs, models, versions on target. No source API calls. |

### Step-by-Step

**Step 1 — Export from Source:**

On the source workspace, set `MIGRATION_MODE = "export"`, configure source credentials and `MODEL_NAMES` as usual. Target settings are ignored. Run the notebook.

**Step 2 — Transfer the Bundle:**

Copy the `ARTIFACT_TEMP_DIR` contents to the target workspace using any method:
- Azure: `azcopy copy`
- AWS: `aws s3 sync`
- GCP: `gsutil rsync`
- Or: `databricks fs cp -r`, zip + scp, etc.

**Step 3 — Import on Target:**

On the target workspace, set `MIGRATION_MODE = "import"`, configure target settings (`UC_TARGET_CATALOG`, `UC_TARGET_SCHEMA`, or `MODEL_NAME_PREFIX`). Source credentials are ignored. Run the notebook.

### Export Bundle Structure

```
ARTIFACT_TEMP_DIR/
├── models/
│   └── <model_name>/
│       ├── model_meta.json        ← Model metadata (name, description, tags)
│       └── versions/
│           └── v<N>/
│               ├── version_meta.json  ← Version details
│               ├── run_meta.json      ← Run params, metrics, tags
│               └── artifacts/
│                   └── model/         ← MLmodel, pkl, conda.yaml, etc.
└── manifests/
    └── (optional summary manifests)
```

### Tips for Export/Import

- **Test with 1 model first** — verify the full cycle before bulk runs
- **Tracking table works in both modes** — export writes `EXPORTED` status; import writes `COMPLETED`
- **Resumable** — re-running import skips already-imported models
- **Bundle is portable** — any path both workspaces can access works
- **Don't mix modes** — complete export fully before starting import

---

## 14. Tracking Table

The Delta tracking table is your migration's single source of truth. It enables resumability and provides an audit trail.

### Schema

| Column | Set By | Description |
| --- | --- | --- |
| `source_host` | Discovery | Source workspace URL |
| `model_name` | Discovery | Source model name |
| `readiness` | Discovery | `READY` / `PARTIAL` / `BLOCKED` |
| `source_versions` | Discovery | Total version count on source |
| `source_versions_migratable` | Discovery | Versions with accessible runs |
| `source_versions_blocked` | Discovery | Versions with deleted/inaccessible runs |
| `source_experiments` | Discovery | Number of linked experiments |
| `source_runs` | Discovery | Number of linked runs |
| `stages` | Discovery | Comma-separated stages (Production, Staging, etc.) |
| `owner_emails` | Discovery | Comma-separated user emails |
| `flavors` | Discovery | MLflow model flavors (sklearn, pytorch, etc.) |
| `requirements` | Discovery | Python requirements from model artifacts |
| `migration_status` | Both | `PENDING` → `COMPLETED` / `PARTIAL` / `FAILED` |
| `target_versions` | Migration | Count of successfully migrated versions |
| `target_runs` | Migration | Count of cloned runs |
| `target_model_url` | Migration | Clickable link to the target model |
| `target_experiment_urls` | Migration | Links to target experiments |
| `migration_comments` | Migration | Error details (if any) |
| `last_updated_at` | Both | Timestamp of last change |

### Primary Key

`(source_host, model_name)` — safe for multi-workspace migrations into the same target.

### Status Flow

```
PENDING → IN_PROGRESS → COMPLETED   (all versions migrated successfully)
                      → PARTIAL     (some versions migrated, others failed)
                      → FAILED      (zero versions, errors occurred)
```

---

## 15. Resuming a Failed or Partial Migration

The framework is designed to be **idempotent**. To resume:

1. **Simply re-run** the same configuration cell + Run Migration cell.
2. The framework reads the tracking table and:
   - Skips models with status `COMPLETED`
   - Retries models with status `PENDING`, `PARTIAL`, or `FAILED`
   - Skips already-created versions (detected via `source_model_version` tag on target)

### Force-Retry Failed Models

If you want to retry specific failed models after fixing the root cause:

```sql
UPDATE catalog.schema.migration_tracking
SET migration_status = 'PENDING'
WHERE migration_status = 'FAILED'
```

Then re-run the notebook.

### Without a Tracking Table

If `TRACKING_TABLE = None`, the framework still detects duplicates via tags on the target (slower but functional). However, you lose the ability to see progress and the framework must re-scan the source each time.

---

## 16. Reading the Output

The final output is a pandas DataFrame displayed in the notebook. Each row represents one model:

| Column | Meaning |
| --- | --- |
| **Model** | Source model name |
| **Target** | Target model name (with prefix/catalog.schema applied) |
| **Model URL** | Clickable hyperlink to the target model in the Databricks UI |
| **Versions Migrated** | Count of successfully created versions on the target |
| **Status** | `OK` = all versions migrated, `FAILED` = errors occurred, `PARTIAL` = some versions succeeded |
| **Verified** | Source count → Target count comparison. Match = good. Mismatch = investigate. |
| **Comments** | Error details or notes about skipped versions |

### Interpreting Verification Results

- **`[4] → [4]` match** — all versions migrated and verified
- **`[4] → [6]` target > source** — usually harmless; can happen with `CREATE_DUMMY_VERSIONS=True` (placeholders are READY) or re-runs
- **`[4] → [2]` target < source** — some versions failed; check Comments column
- **`[0] → [0]`** — model had no migratable versions (all runs deleted)

---

## 17. Concurrency Model and Performance Tuning

### Thread Architecture

```
execute_migration()
├── Phase 1 — Discovery
│   ├── ThreadPool: fetch model versions    (parallel per model)
│   ├── ThreadPool: fetch experiment runs   (parallel per experiment)
│   └── MERGE into tracking table           (main thread — Spark SQL is not thread-safe)
├── Phase 2a — Experiments
│   └── batched ThreadPool                  (parallel per experiment)
├── Phase 2b — Models
│   └── for batch in chunked(models, BATCH_SIZE):
│       ├── ThreadPool: download artifacts  (high concurrency)
│       └── ThreadPool: register versions   (sequential per model, capped at 5)
│           └── UPDATE tracking table       (main thread after each model)
└── Phase 3 — Verification
    └── Compare source vs target counts
```

### Tuning Recommendations

| Scenario | Recommended `BATCH_SIZE` | Notes |
| --- | --- | --- |
| Default (serverless) | `10` | Good balance of speed and rate-limit avoidance |
| Hitting HTTP 429 errors | `5` | Reduces parallel pressure on source APIs |
| Large workspace (1000+ models) | `10` | Framework handles batching internally |
| UC source (stricter rate limits) | `10` | Framework auto-caps UC concurrency to 2 |
| Very large artifacts (>1 GB per model) | `5` | Prevents disk space exhaustion in staging dir |

### Performance Expectations

| Migration Size | Estimated Time |
| --- | --- |
| 1-5 models, ~10 versions | 1-5 minutes |
| 50 models, ~200 versions | 15-30 minutes |
| 500+ models, 1000+ versions | 1-3 hours |
| 1000+ models (bulk scan) | 3-8 hours (UC sources may be slower due to rate limits) |

---

## 18. Troubleshooting

### Authentication Errors

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ValueError: Provide either a PAT token or a service principal` | Empty `SOURCE_HOST` or `SOURCE_TOKEN` | Fill in credentials in the configuration cell |
| `401 Unauthorized` | Token expired or invalid | Generate a new PAT token on the source workspace |
| `403 Forbidden` | Token lacks required permissions | Grant MLflow read access on source, write access on target |
| OAuth token refresh fails | Incorrect `SP_CLIENT_ID` or `SP_CLIENT_SECRET` | Verify the service principal credentials |

### Discovery Issues

| Symptom | Cause | Fix |
| --- | --- | --- |
| `0 models discovered` | `MODEL_NAMES` doesn't match source names | Check for typos/whitespace; use `[]` for bulk scan |
| UC bulk scan finds nothing | `INCLUDE_CATALOGS` is set but doesn't match any catalog | Check catalog names; leave empty for all catalogs |
| `source_versions = 1000` for all models | Pagination cap hit | Fixed in v2 — framework now paginates fully |

### Migration Errors

| Symptom | Cause | Fix |
| --- | --- | --- |
| `RESOURCE_DOES_NOT_EXIST` | Source run was permanently deleted | Expected — framework creates placeholder if `CREATE_DUMMY_VERSIONS=True` |
| `INVALID_PARAMETER_VALUE: Got an invalid source` | Stale framework cached in Python memory | Run `%restart_python` (Cell 2) first |
| `404 Not Found` for `runs/get` | Source run deleted after version was created | Placeholder version created automatically |
| `Failed to download artifacts` | Model artifacts purged from source | Version created as placeholder; noted in Comments |
| Migration hangs or is very slow | Rate limiting (HTTP 429) | Reduce `BATCH_SIZE` to `5`; framework auto-retries with backoff |

### Tracking Table Issues

| Symptom | Cause | Fix |
| --- | --- | --- |
| Tracking table not created | Missing `CREATE TABLE` permission | Grant permission on the target schema |
| Status stuck at `PENDING` | Migration failed before status update | Check Comments column; fix error and re-run |
| All models re-migrating on re-run | `TRACKING_TABLE` not set | Set a tracking table to enable resume |

### Permission Errors

| Symptom | Cause | Fix |
| --- | --- | --- |
| `PERMISSION_DENIED` on source | PAT/SP lacks MLflow read access | Grant `Can Read` on source experiments + models |
| `PERMISSION_DENIED` on UC target | Missing UC privileges | Grant `USE CATALOG`, `USE SCHEMA`, `CREATE MODEL` |

---

## 19. Security Best Practices

1. **Never hardcode tokens.** Use Databricks secret scopes:
   ```python
   SOURCE_TOKEN = dbutils.secrets.get(scope="my-scope", key="source-pat")
   ```

2. **Use Service Principals** for production migrations — they don't expire like PAT tokens and can be scoped narrowly.

3. **Rotate credentials** after migration is complete. If you created a temporary PAT, revoke it.

4. **Use the tracking table** — it provides a complete audit trail of what was migrated, when, and from where.

5. **Restrict notebook access** — the configuration cell contains (or references) credentials. Limit who can view/edit the notebook.

6. **Clean up staging artifacts** — after migration, remove `ARTIFACT_TEMP_DIR` contents to avoid leaving model files on local disk.

---

## 20. FAQ

**Q: Can I migrate from the same workspace to itself?**  
A: Yes. Set `SOURCE_HOST` to your current workspace URL. Use `MODEL_NAME_PREFIX` (WS targets) or a different `UC_TARGET_SCHEMA` (UC targets) to avoid overwriting source models.

**Q: What happens if I run the migration twice?**  
A: Nothing bad — the framework is idempotent. Completed models are skipped (via tracking table or tag-based dedup). Already-created versions are detected and skipped.

**Q: Can I migrate just one version of a model, not all versions?**  
A: Not directly via configuration. The framework migrates all versions for each named model. You could modify the configuration post-discovery, but it's not a standard workflow.

**Q: Does the framework handle model serving endpoints?**  
A: No. Only registered models, versions, experiments, runs, and artifacts are migrated. Serving endpoints must be recreated manually on the target.

**Q: What if the source run is deleted but the model version still exists?**  
A: If `CREATE_DUMMY_VERSIONS = True` (default), the framework creates a placeholder version on the target without artifacts. The version is tagged as a placeholder. If `False`, the version is skipped.

**Q: How do I know which models have been migrated?**  
A: Query the tracking table:
```sql
SELECT model_name, migration_status, target_versions, target_model_url
FROM catalog.schema.migration_tracking
ORDER BY last_updated_at DESC
```

**Q: Can I migrate models between clouds (e.g., Azure → AWS)?**  
A: Yes, using Export/Import mode. Export on the source cloud, transfer the bundle to the target cloud, then import.

**Q: What if the migration is interrupted mid-way?**  
A: Re-run the notebook. The tracking table remembers which models are `COMPLETED`. Only `PENDING`/`FAILED` models are retried.

**Q: Does the framework modify anything on the source workspace?**  
A: No. All source operations are read-only (search models, get versions, download artifacts). Nothing is written, updated, or deleted on the source.

**Q: How are workspace stages mapped to UC aliases?**  
A: `Production` → `Champion`, `Staging` → `Challenger`. Other stages (Archived, None) are not mapped to aliases.

**Q: What's the maximum number of models I can migrate at once?**  
A: There is no hard limit. The framework processes models in batches of `BATCH_SIZE` with rate-limit retry. Migrations of 1000+ models have been tested successfully.

---

*End of Runbook*