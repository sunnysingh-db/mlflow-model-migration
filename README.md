# MLflow Workspace Registry Migration Framework

Bulk-migrates MLflow **workspace registry** models, experiments, runs, and artifacts from a source Databricks workspace into the current (target) workspace.

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
    ├── framework.py                   ← Core migration logic
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

```
┌─────────────────────────────────────────────────────────────────┐
│  Cell 1-2: Documentation (no execution needed)                  │
├─────────────────────────────────────────────────────────────────┤
│  Cell 3: Setup Imports                                          │
│  • Adds project to sys.path                                     │
│  • Reloads framework module (picks up latest code changes)      │
├─────────────────────────────────────────────────────────────────┤
│  Cell 4: Source Workspace Credentials                           │
│  • Set AUTH_MODE = "pat" or "service_principal"                  │
│  • Configure SOURCE_HOST, SOURCE_TOKEN (or SP credentials)      │
│  • Validates connectivity                                       │
├─────────────────────────────────────────────────────────────────┤
│  Cell 5: Migration Options                                      │
│  • Configure prefixes, batch size, limits                       │
│  • Set extra_model_names allowlist (or [] for all models)       │
├─────────────────────────────────────────────────────────────────┤
│  Cell 6: Discover Source Assets                                 │
│  • Probes source workspace                                      │
│  • Filters inaccessible versions (deleted runs)                 │
│  • Returns inventory summary                                    │
├─────────────────────────────────────────────────────────────────┤
│  Cell 7: Run Bulk Migration                                     │
│  • Creates experiments under /Shared                            │
│  • Clones runs (params, metrics, tags, artifacts)               │
│  • Registers model versions with full dbfs:/ paths              │
│  • Returns MigrationSummary with skipped_versions detail        │
├─────────────────────────────────────────────────────────────────┤
│  Cell 8: Migration Report                                       │
│  • Generates per-version comparison DataFrame                   │
│  • Shows: status, params match, metrics match, artifacts match  │
│  • Surfaces skipped versions with failure reasons               │
└─────────────────────────────────────────────────────────────────┘
```

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
| `max_runs_per_experiment` | Max runs per experiment (`None` = all) | `None` |
| `skip_existing_model_versions` | Skip versions already in target | `False` |
| `download_artifacts` | Controls non-model artifact transfer (see below) | `True` |
| `include_run_artifacts` | Whether to invoke artifact copy at all | `True` |
| `include_deleted_runs` | Include soft-deleted runs | `False` |
| `batch_size` | Models/experiments per batch | `10` |
| `max_workers` | Thread pool concurrency | `10` |

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
migrate_all()
├── _migrate_experiments()
│   └── for batch in chunked(experiments, batch_size):     ← sequential across batches
│       └── ThreadPoolExecutor(max_workers)                ← parallel within batch
│           └── _migrate_single_experiment(exp)
│               └── ThreadPoolExecutor(max_workers)        ← parallel runs within experiment
│                   └── _clone_run(run)
│
└── _migrate_models()
    └── for batch in chunked(models, batch_size):          ← sequential across batches
        └── ThreadPoolExecutor(max_workers)                ← parallel within batch
            └── _migrate_single_registered_model(model)
                └── ThreadPoolExecutor(max_workers)        ← parallel versions within model
                    └── _clone_model_version(version)
```

**Tuning guidance:**
- `max_workers=10-12` — good default; balances throughput vs API rate limits
- `max_workers=4-6` — use if you're hitting 429 rate limits on source workspace
- `batch_size=8-10` — controls memory; larger batches hold more futures in memory

---

## Edge Cases Handled

| Scenario | Behavior |
| --- | --- |
| Source run deleted | Filtered during discovery; reported as `RUN_DELETED` |
| Model artifacts purged | Caught before `create_model_version`; reported as `NO_ARTIFACTS` |
| Artifact with empty path | Skipped individually; other artifacts still copied |
| Rate limiting (429) | Automatic retry with exponential backoff (5 attempts) |
| Duplicate model version | Skipped when `skip_existing_model_versions=True` |
| Trailing whitespace in model names | Auto-stripped during allowlist matching |
| Thread-safety (env vars) | Global lock prevents source/target credential pollution |
| Transient server errors (500/503) | Retried automatically |

---

## Re-running After Failures

The framework is **idempotent**:

1. Existing target runs are detected by `source_run_id` tag — not re-created.
2. Existing model versions are detected by `source_model_version` tag.
3. Set `skip_existing_model_versions=True` to skip already-migrated versions.
4. Set `skip_existing_model_versions=False` to force re-attempt of failed versions.

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `0 models discovered` | `extra_model_names` doesn't match source names | Check for typos/whitespace; set `[]` for all models |
| `INVALID_PARAMETER_VALUE: Got an invalid source` | Old framework version without dbfs path fix | Re-run cell 3 to reload framework |
| `RESOURCE_DOES_NOT_EXIST` | Source run was deleted | Cannot be recovered — reported in summary |
| `Parameter 'path' must be a non-empty string` | Corrupted artifact entry in source run | Handled automatically — skipped with warning |
| Migration hangs | Rate limiting on source workspace | Reduce `max_workers` to 4-6 |
| Duplicate version error | Version already exists in target | Set `skip_existing_model_versions=True` |
