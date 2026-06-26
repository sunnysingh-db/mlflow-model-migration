# MLflow Workspace Registry Migration Framework

Bulk-migrates MLflow **workspace registry** models, experiments, runs, and artifacts from a source Databricks workspace into the current (target) workspace. Tracks progress in a Delta table for **resumable migrations**.

---

## What Gets Migrated

| Asset | Details |
| --- | --- |
| Registered Models | Name, description, tags |
| Model Versions | Source path, stage, tags, description |
| Experiments | Name, tags (placed under `/Shared`) |
| Runs | Parameters, metrics (full history), tags, status |
| Artifacts | Model files (MLmodel, pkl, conda.yaml, etc.) and optionally all run artifacts |

---

## Project Structure

```
mlflow-model-migration/
├── README.md                          ← You are here
├── Workspace Registry Migration       ← Main notebook (run this)
└── workspace_registry_migrator/       ← Framework package
    ├── __init__.py
    ├── framework.py                   ← Core migration logic + migrate_pending()
    ├── reporting.py                   ← Inventory reports, Delta tracking, URLs
    ├── config.py                      ← Configuration helpers
    ├── clients.py                     ← Client wrappers
    └── utils.py                       ← Utilities (logging, chunking, temp dirs)
```

---

## Prerequisites

### 1. Network Connectivity

- Target workspace must reach the source workspace REST API (port 443).
- If using Private Link / VNet: ensure NSG/firewall rules allow HTTPS egress.
- Cross-region: verify VNet peering or public egress is available.

### 2. Authentication (choose one)

| Method | Required Values |
| --- | --- |
| **PAT** | Source workspace URL + Personal Access Token |
| **Service Principal** | Source workspace URL + `client_id` + `client_secret` |

> **Recommendation**: Store credentials in a Databricks secret scope.

### 3. Permissions

**On source workspace:**
- Read MLflow Tracking (experiments, runs)
- Read Model Registry (models, versions)
- Download artifacts

**On target workspace (current):**
- Write to `/Shared` experiments
- Create registered models and versions

### 4. Compute

- Serverless (CPU) or any cluster with `mlflow` and `databricks-sdk` available.
- No GPU required.

---

## Execution Flow

The framework operates in two phases: **Discover** (builds inventory) and **Migrate** (processes PENDING models from tracking table).

```
┌─────────────────────────────────────────────────────────────────┐
│  Cell 1-2: Documentation (no execution needed)                  │
├─────────────────────────────────────────────────────────────────┤
│  Cell 3: Setup Imports                                          │
│  • Adds project to sys.path, force-reloads framework            │
├─────────────────────────────────────────────────────────────────┤
│  Cell 4: Source Workspace Credentials                           │
│  • AUTH_MODE = "pat" or "service_principal"                      │
│  • SOURCE_HOST, TOKEN or SP_CLIENT_ID/SECRET                    │
├─────────────────────────────────────────────────────────────────┤
│  Cell 5: Migration Options + Tracking Table Config              │
│  • TRACKING_TABLE = catalog.schema.table                        │
│  • Prefixes, batch_size, max_workers, allowlists                │
├─────────────────────────────────────────────────────────────────┤
│  Cell 6: Discover Source Assets (Phase 1)                       │
│  • Discovers ALL models from source (parallelized)              │
│  • Generates inventory report (parallel, 20 workers)            │
│  • MERGEs to Delta tracking table (all start as PENDING)        │
│  • include_metadata=True/False controls artifact downloads      │
├─────────────────────────────────────────────────────────────────┤
│  Cell 7: Run Bulk Migration (Phase 2)                           │
│  • migrate_pending(): reads PENDING from tracking table         │
│  • Targeted discovery — only for pending models                 │
│  • Updates tracking table per-model (COMPLETED/PARTIAL/FAILED)  │
│  • Populates target_model_url + target_experiment_urls           │
├─────────────────────────────────────────────────────────────────┤
│  Cell 8: Migration Report                                       │
│  • Displays tracking table grouped by status                    │
│  • Shows: COMPLETED, PARTIAL, PENDING, FAILED counts            │
└─────────────────────────────────────────────────────────────────┘
```

### Resume Behavior

Re-running **cell 7** after a partial migration only processes models still in `PENDING` state.
Already-completed models are skipped entirely — no re-discovery, no redundant API calls.
For 900 models where 700 are done, it only touches the remaining 200.

---

## Step-by-Step Execution

### Step 1: Configure Credentials (Cell 4)

Replace the placeholder values:

```python
AUTH_MODE = "pat"  # or "service_principal"
SOURCE_HOST = "https://<source-workspace>.azuredatabricks.net"
SOURCE_TOKEN = dbutils.secrets.get("migration-scope", "source-token")
```

For service principal:
```python
AUTH_MODE = "service_principal"
SOURCE_HOST = "https://<source-workspace>.azuredatabricks.net"
SP_CLIENT_ID = dbutils.secrets.get("migration-scope", "sp-client-id")
SP_CLIENT_SECRET = dbutils.secrets.get("migration-scope", "sp-client-secret")
```

### Step 2: Set Migration Options (Cell 5)

| Option | Purpose | Default |
| --- | --- | --- |
| `model_name_prefix` | Prefix for target model names (avoids collisions) | `""` |
| `experiment_name_prefix` | Prefix for target experiment names | `""` |
| `extra_model_names` | Allowlist of model names to migrate (`[]` = all) | `[]` |
| `max_model_versions_per_model` | Max versions per model (`None` = all) | `None` |
| `max_runs_per_experiment` | Max runs per experiment (`None` = all, fully paginated) | `None` |
| `skip_existing_model_versions` | Skip versions already in target (by tag match) | `True` |
| `download_artifacts` | Controls non-model artifact transfer (see below) | `True` |
| `include_run_artifacts` | Whether to invoke artifact copy at all | `True` |
| `include_deleted_runs` | Include soft-deleted runs | `True` |
| `batch_size` | Models/experiments per parallel batch | `20` |
| `max_workers` | Thread pool concurrency | `20` |

### Step 3: Run Discovery (Cell 6)

Run cell 6 to preview what will be migrated **without making changes**.

Expected output:
```python
{'registered_models': 15, 'experiments': 8, 'runs': 120, 'model_versions': 25}
```

### Step 4: Execute Migration (Cell 7)

Run cell 7. Expected output:
```python
{'migrated_models': 15,
 'migrated_model_versions': 22,
 'migrated_experiments': 8,
 'migrated_runs': 22,
 'skipped_versions': [
   {'model': 'some_model', 'version': '3', 'reason': 'No model artifacts...'}
 ]}
```

### Step 5: Review Report (Cell 8)

Run cell 8 for a detailed DataFrame report:

```
MIGRATION REPORT: 22/25 versions migrated successfully
  ✅ Migrated: 22  |  ⚠️ Failed: 0  |  ❌ Run deleted: 2  |  ❌ No artifacts: 1
```

The DataFrame shows per-version:
- Source vs target params/metrics/artifacts counts
- Whether each dimension matches exactly
- Clear status for non-migratable versions

---

## Artifact Download Behavior

Two flags control what gets copied. Understanding the interaction is important for large migrations.

### How the flags interact

| `download_artifacts` | `include_run_artifacts` | Effect |
| --- | --- | --- |
| `True` | `True` | **Full clone** — model files + all other run artifacts (plots, SHAP values, data samples, evaluation CSVs, etc.) |
| `True` | `False` | Model files only (minimum for version registration) |
| `False` | `True` or `False` | Model files still copied (required for `create_model_version`), but non-model artifacts are skipped |

### What "model files" means

These are the files under the `model/` subdirectory of a run's artifact store — the minimum set required to register a model version:

```
artifacts/
└── model/
    ├── MLmodel              ← Model metadata (flavors, signature)
    ├── python_model.pkl     ← Serialized model object
    ├── conda.yaml           ← Conda environment spec
    ├── python_env.yaml      ← Python environment spec
    └── requirements.txt     ← Pip dependencies
```

These are **always copied** regardless of either flag, because `create_model_version` requires them to exist at the target path.

### What "non-model artifacts" means

Everything else under `artifacts/` that is NOT the model directory:

```
artifacts/
├── model/                   ← Always copied (see above)
├── feature_importance.png   ← Only copied if both flags are True
├── confusion_matrix.png     ← Only copied if both flags are True
├── shap_summary.html        ← Only copied if both flags are True
└── eval_results.csv         ← Only copied if both flags are True
```

### When to set `download_artifacts=False`

- **Large-scale migrations** (100+ models) where you only need working model versions in the target registry
- **Storage-sensitive environments** — non-model artifacts can be GBs per run (data snapshots, large plots)
- **Speed** — skipping non-model artifacts significantly reduces migration time

### When to keep `download_artifacts=True`

- You need a **complete replica** of the source workspace (audit, compliance)
- Downstream notebooks reference non-model artifacts (e.g., loading evaluation CSVs from runs)
- Experiment comparison workflows depend on logged plots/tables

---

## Concurrency Model

The framework uses **parallel execution** at multiple levels:

```
migrate_pending()
├── Query tracking table for PENDING models
├── _discover_models(pending_names)         ← targeted discovery
│   ├── ThreadPoolExecutor: fetch versions    ← parallel per model
│   └── ThreadPoolExecutor: fetch runs        ← parallel per experiment
├── _migrate_experiments()
│   └── for batch in chunked(experiments, batch_size):   ← batched
│       └── ThreadPoolExecutor(max_workers)              ← parallel
└── _migrate_models()
    └── for batch in chunked(models, batch_size):        ← batched
        └── ThreadPoolExecutor(max_workers)              ← parallel
            └── _migrate_single_registered_model(model)
        └── _update_tracking_table()                     ← main thread (Spark-safe)

generate_inventory_report()
└── ThreadPoolExecutor(max_workers=20)                  ← parallel per model
    └── _process_single_model(model)
        └── search_model_versions + get_run per version
        └── _parse_model_metadata (if include_metadata=True)
```

**Important**: Delta tracking table updates (`spark.sql(UPDATE ...)`) run on the **main thread** after each model’s future completes — Spark SQL is NOT thread-safe on serverless.

**Tuning guidance:**
- `max_workers=20` — good default for serverless (no cluster to saturate)
- `max_workers=6-8` — use if hitting 429 rate limits on source workspace
- `batch_size=20` — controls memory; larger batches hold more futures

---

## Edge Cases Handled

| Scenario | Behavior |
| --- | --- |
| Source run deleted | Reported in `discovery_comments`; version counted as blocked |
| Version with `run_id=None` | Counted as migratable (artifact-only registration) |
| Model artifacts purged | Caught before `create_model_version`; reported in `migration_comments` |
| Artifact with empty path | Skipped individually; other artifacts still copied |
| Rate limiting (429) | Automatic retry with exponential backoff (5 attempts) |
| Duplicate model version | Skipped when `skip_existing_model_versions=True` (tag-based) |
| Trailing whitespace in model names | Auto-stripped during allowlist matching |
| Thread-safety (env vars) | Global lock prevents source/target credential pollution |
| Thread-safety (Spark SQL) | Tracking updates run on main thread after future.result() |
| Transient server errors (500/503) | Retried automatically |
| Same-workspace test | Target counts only include tagged versions (excludes originals) |
| Cross-workspace env leak | `try/finally` in reporting guarantees env restoration |

---

## Delta Tracking Table

The migration state is persisted in a Unity Catalog Delta table (configured via `TRACKING_TABLE` in cell 5).

**Primary key**: `(source_host, model_name)` — multi-workspace safe.

### Key Columns

| Column | Populated by | Notes |
| --- | --- | --- |
| `source_host` | Discovery | Part of PK |
| `model_name` | Discovery | Part of PK |
| `readiness` | Discovery | ✅ READY / ⚠️ PARTIAL / ❌ BLOCKED |
| `source_versions` | Discovery | Total versions in source |
| `source_versions_migratable` | Discovery | Versions with accessible runs |
| `source_params` / `metrics` / `artifacts` | Discovery | Aggregate counts from source |
| `target_versions` | Migration | Versions with `source_model_version` tag in target |
| `target_params` / `metrics` / `artifacts` | Migration | Counts from migrated versions only |
| `migration_status` | Both | PENDING → COMPLETED / PARTIAL / FAILED / SKIPPED |
| `target_model_url` | Migration | Clickable URL to target model |
| `target_experiment_urls` | Migration | Pipe-separated experiment URLs |
| `discovery_comments` | Discovery | Per-version error messages |
| `migration_comments` | Migration | Per-version failure reasons |

### MERGE Behavior

- **Discovery** (cell 6): `WHEN MATCHED` updates only source-side columns. Never overwrites target-side data.
- **Migration** (cell 7): `UPDATE` only writes to target-side columns + `migration_status`.
- **Safe to re-run discovery** without losing migration progress.

### Migration Status Logic

| Status | Condition |
| --- | --- |
| `COMPLETED` | Target tagged versions ≥ source migratable versions |
| `PARTIAL` | Some versions migrated but not all |
| `FAILED` | Zero versions migrated + errors encountered |
| `SKIPPED` | Zero versions migrated, no errors (nothing to do) |

---

## Re-running After Failures

The framework is **idempotent** at multiple levels:

1. **Tracking table**: `migrate_pending()` only processes `PENDING` models — skips COMPLETED/PARTIAL/FAILED entirely.
2. **Version dedup**: `skip_existing_model_versions=True` checks for `source_model_version` tag in target.
3. **Run dedup**: Existing target runs detected by `source_run_id` tag — not re-created.
4. **Reset to retry failures**: `UPDATE tracking_table SET migration_status = 'PENDING' WHERE migration_status = 'FAILED'`

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `0 models discovered` | `extra_model_names` doesn't match source names | Check for typos/whitespace; set `[]` for all models |
| `⚠️ No models discovered` | Credentials not configured or empty | Fill in SOURCE_HOST + credentials in cell 4 |
| `KeyError: 'readiness'` | Empty inventory DataFrame (0 models) | Fixed: `print_inventory_summary` now handles empty DFs |
| All models show `❌ BLOCKED` | `generate_inventory_report` was using wrong client | Fixed: pass `source_context=migrator.source` |
| `source_runs = 1000` for all models | Pagination cap in `search_runs` | Fixed: now paginates fully (set `max_runs_per_experiment` to cap) |
| `expected string or bytes-like object, got NoneType` | Model version has `run_id=None` | Fixed: versions without runs counted as ready |
| `INVALID_PARAMETER_VALUE: Got an invalid source` | Old framework in memory | Re-run cell 3 to reload framework |
| `RESOURCE_DOES_NOT_EXIST` | Source run was permanently deleted | Reported in `discovery_comments`; version blocked |
| Migration hangs | Rate limiting on source workspace | Reduce `max_workers` to 6-8 |
| Duplicate version error | Version already exists in target | Set `skip_existing_model_versions=True` |
| Tracking table not updating | Spark SQL called from worker thread | Fixed: updates run on main thread after `future.result()` |
| Target counts > source counts | Same-workspace test (versions accumulate) | Expected; real cross-workspace migration won't have this |
