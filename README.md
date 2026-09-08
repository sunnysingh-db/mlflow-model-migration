# 🚀 MLflow Model Migration Framework

Bulk-migrate MLflow models, experiments, runs & artifacts between Databricks workspaces.
Supports **4 migration paths**, tracks progress in Delta, and **resumes where it left off**.

---

## 📋 What Gets Migrated

| Asset | What's Copied |
| --- | --- |
| 🏷️ Registered Models | Name, description, tags |
| 📦 Model Versions | Source path, stage → alias mapping, tags, description |
| 🧪 Experiments | Name, tags (target: `/Shared/mlflow-workspace-migration/`) |
| 🏃 Runs | Params, full metric history, tags, status |
| 📁 Artifacts | Model files (MLmodel, pkl, conda.yaml…) + optionally all run artifacts |

---

## 🗺️ Supported Migration Scenarios

| # | Source → Target | Use Case | Key Config |
| --- | --- | --- | --- |
| 1️⃣ | **Workspace → Workspace** | Clone models to another workspace (or same with prefix) | `SOURCE_REGISTRY="workspace"`, `TARGET_REGISTRY="workspace"`, set `MODEL_NAME_PREFIX` |
| 2️⃣ | **Workspace → Unity Catalog** | Upgrade legacy WS models to UC | `SOURCE_REGISTRY="workspace"`, `TARGET_REGISTRY="uc"`, set `UC_TARGET_CATALOG` + `UC_TARGET_SCHEMA` |
| 3️⃣ | **Unity Catalog → Unity Catalog** | Move UC models across catalogs/schemas | `SOURCE_REGISTRY="uc"`, `TARGET_REGISTRY="uc"`, set `UC_TARGET_CATALOG` + `UC_TARGET_SCHEMA` |
| 4️⃣ | **Unity Catalog → Workspace** | Downgrade UC models back to legacy registry | `SOURCE_REGISTRY="uc"`, `TARGET_REGISTRY="workspace"`, set `MODEL_NAME_PREFIX` |

> 💡 **Same-workspace?** Set `SOURCE_HOST` to your current workspace URL. Use `MODEL_NAME_PREFIX` (WS targets) or a different `UC_TARGET_SCHEMA` (UC targets) to avoid overwriting source models.

---

## 📂 Project Structure

```
mlflow-model-migration/
├── README.md                              ← 📖 You are here
├── Workspace Registry Migration           ← 🎯 Main notebook (run this!)
└── workspace_registry_migrator/           ← ⚙️ Framework package
    ├── __init__.py
    ├── framework_v2.py                    ← Core migration engine
    ├── notebook_helpers.py                ← Notebook-friendly wrapper
    ├── rest_client.py                     ← REST API client with retry
    ├── reporting.py                       ← Delta tracking & URL generation
    ├── config.py                          ← Configuration helpers
    ├── clients.py                         ← Client wrappers
    ├── discovery.py                       ← Source discovery logic
    ├── migrate.py                         ← Migration orchestration
    └── utils.py                           ← Logging, chunking, temp dirs
```

---

## ✅ Prerequisites

### 1. 🌐 Network

- Target workspace must reach the source workspace REST API (port **443**).
- Private Link / VNet? → Ensure NSG/firewall allows HTTPS egress.
- Cross-region? → Verify VNet peering or public egress.

### 2. 🔑 Authentication (pick one)

| Method | What You Need |
| --- | --- |
| **PAT** (simplest) | Source workspace URL + [Personal Access Token](https://docs.databricks.com/en/dev-tools/auth/pat.html) |
| **Service Principal** | Source workspace URL + `client_id` + `client_secret` |

> 🔒 **Best practice:** Store credentials in a [Databricks secret scope](https://docs.databricks.com/en/security/secrets/secret-scopes.html) — never hardcode tokens!

### 3. 🛡️ Permissions

| Where | What |
| --- | --- |
| **Source workspace** | Read: MLflow Tracking (experiments, runs) + Model Registry (models, versions) + Download artifacts |
| **Target workspace** | Write: `/Shared` experiments + Create registered models & versions |
| **UC targets** | `USE CATALOG`, `USE SCHEMA`, `CREATE MODEL` on the target catalog/schema |

### 4. 💻 Compute

- ✅ **Serverless CPU** — works great, no special setup
- ✅ Any cluster with `mlflow` + `databricks-sdk` pre-installed
- ❌ No GPU needed

---

## 🏁 Step-by-Step: How to Run

The notebook has **3 cells**: Markdown header → Configuration → Run Migration.

### Step 1️⃣ — Open the Notebook

Open **`Workspace Registry Migration`** in your target workspace.

### Step 2️⃣ — Configure (Cell 2)

Edit the **Configuration** cell. Here's what to set for each scenario:

#### 🔀 Scenario A: Workspace → Workspace

```python
SOURCE_REGISTRY  = "workspace"
TARGET_REGISTRY  = "workspace"
SOURCE_HOST      = "https://<source-workspace>.azuredatabricks.net"
SOURCE_TOKEN     = dbutils.secrets.get("my-scope", "source-pat")
MODEL_NAMES      = ["my_model_1", "my_model_2"]   # [] = all models
MODEL_NAME_PREFIX = "migrated_"                     # avoid name collision
TRACKING_TABLE   = "catalog.schema.ws_tracking"     # Delta tracking table
```

#### ⬆️ Scenario B: Workspace → Unity Catalog

```python
SOURCE_REGISTRY  = "workspace"
TARGET_REGISTRY  = "uc"
SOURCE_HOST      = "https://<source-workspace>.azuredatabricks.net"
SOURCE_TOKEN     = dbutils.secrets.get("my-scope", "source-pat")
MODEL_NAMES      = ["my_model_1", "my_model_2"]
UC_TARGET_CATALOG = "my_catalog"
UC_TARGET_SCHEMA  = "my_schema"
TRACKING_TABLE    = "my_catalog.my_schema.migration_tracking"
```

#### 🔄 Scenario C: UC → UC (cross-catalog/schema)

```python
SOURCE_REGISTRY  = "uc"
TARGET_REGISTRY  = "uc"
# For same-workspace UC→UC, auto-detect credentials:
_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
SOURCE_HOST      = "https://" + _ctx.browserHostName().get()
SOURCE_TOKEN     = _ctx.apiToken().get()
MODEL_NAMES      = ["source_catalog.source_schema.my_model"]
UC_TARGET_CATALOG = "target_catalog"
UC_TARGET_SCHEMA  = "target_schema"
TRACKING_TABLE    = "target_catalog.target_schema.migration_tracking"
```

#### ⬇️ Scenario D: UC → Workspace

```python
SOURCE_REGISTRY  = "uc"
TARGET_REGISTRY  = "workspace"
SOURCE_HOST      = "https://<source-workspace>.azuredatabricks.net"
SOURCE_TOKEN     = dbutils.secrets.get("my-scope", "source-pat")
MODEL_NAMES      = ["catalog.schema.my_uc_model"]
MODEL_NAME_PREFIX = "from_uc_"    # target WS model name prefix
TRACKING_TABLE    = "catalog.schema.uc_ws_tracking"
```

### Step 3️⃣ — Run All

Hit **Run All** (or run cells 2 → 3 in order). The framework will:

```
📡 Phase 1 — Discovery
   └─ Scan source → list models, versions, experiments, runs
   └─ MERGE inventory into tracking table (all start as PENDING)

🚚 Phase 2 — Migration
   └─ Download artifacts from source → local staging
   └─ Create experiments + clone runs on target
   └─ Register model versions on target (sequential per model)
   └─ Update tracking table per model (COMPLETED / PARTIAL / FAILED)

✅ Phase 3 — Verification
   └─ Compare source vs target version counts
   └─ Report mismatches
```

### Step 4️⃣ — Review Results

The output table shows per-model:

| Column | Meaning |
| --- | --- |
| **Model** | Source model name |
| **Model URL** | 🔗 Clickable link to target model |
| **Versions Migrated** | Count of successfully created versions |
| **Status** | ✅ OK / ❌ FAILED |
| **Verified** | ✅ `[N] → [N]` match or ❌ mismatch |
| **Comments** | Error details (if any) |

### Step 5️⃣ — Resume (if needed)

Just **re-run** the same cells. The framework:
- ✅ Skips `COMPLETED` models (reads tracking table)
- ✅ Retries `PENDING` / `FAILED` models
- ✅ Skips already-created versions (tag-based dedup)

To force-retry failed models:
```sql
UPDATE my_catalog.my_schema.migration_tracking
SET migration_status = 'PENDING'
WHERE migration_status = 'FAILED'
```

---

## 📤📥 Export / Import Mode (Air-Gapped Migrations)

When source and target workspaces **cannot reach each other** (air-gapped, different clouds, strict firewall), use the **export → transfer → import** workflow instead of direct migration.

### 🔀 Three Migration Modes

| Mode | Runs On | What It Does |
| --- | --- | --- |
| `"direct"` | **Either** (needs access to both) | Standard end-to-end migration (default, unchanged behavior) |
| `"export"` | **Source** workspace | Discovers models → downloads artifacts → writes JSON manifests to `ARTIFACT_TEMP_DIR`. No target operations. |
| `"import"` | **Target** workspace | Reads manifests + artifacts from `ARTIFACT_TEMP_DIR` → creates experiments, runs, models & versions on target. No source API calls. |

### 🛫 Step-by-Step: Export / Import Workflow

#### Step 1️⃣ — Export from Source

On the **source** workspace, set:

```python
MIGRATION_MODE    = "export"
ARTIFACT_TEMP_DIR = "/tmp/ws_export_bundle"   # or /Volumes/..., /dbfs/tmp/...
# Configure SOURCE_HOST, SOURCE_TOKEN, MODEL_NAMES as usual
# TARGET settings are ignored in export mode
```

Run the notebook. The framework will:
```
📡 Phase 1 — Discovery
   └─ Scan source models, versions, experiments, runs

📦 Phase 2 — Export
   └─ Download all model artifacts to ARTIFACT_TEMP_DIR/artifacts/
   └─ Write JSON manifests (models, versions, experiments, runs)
   └─ ⏭️  Skip all target operations
   └─ Update tracking table with EXPORTED status
```

#### Step 2️⃣ — Transfer the Bundle

Copy the `ARTIFACT_TEMP_DIR` contents to the target workspace using any method:

```bash
# Azure Blob → Azure Blob
azcopy copy "/tmp/ws_export_bundle/*" \
  "https://<target-storage>.blob.core.windows.net/export-bundle/" --recursive

# Upload to Unity Catalog Volume
databricks fs cp -r /tmp/ws_export_bundle \
  dbfs:/Volumes/catalog/schema/volume/export-bundle/

# Or: zip + scp, S3 sync, GCS transfer, etc.
```

#### Step 3️⃣ — Import on Target

On the **target** workspace, set:

```python
MIGRATION_MODE    = "import"
ARTIFACT_TEMP_DIR = "/tmp/ws_export_bundle"   # same path (or wherever you placed the bundle)
# Configure target settings:
#   UC targets → UC_TARGET_CATALOG + UC_TARGET_SCHEMA
#   WS targets → MODEL_NAME_PREFIX
# SOURCE_HOST / SOURCE_TOKEN are ignored in import mode
```

Run the notebook. The framework will:
```
📥 Phase 1 — Load Manifests
   └─ Read models, versions, experiments, runs from JSON manifests

🚚 Phase 2 — Import
   └─ Create experiments + clone runs from manifest data
   └─ Upload artifacts from bundle directory
   └─ Register model versions on target

✅ Phase 3 — Verification
   └─ Compare manifest counts vs target counts
```

### 📁 Export Bundle Structure

```
ARTIFACT_TEMP_DIR/
├── manifests/
│   ├── models.json          ← Model metadata (name, description, tags)
│   ├── versions.json        ← Version details (stage, tags, run mapping)
│   ├── experiments.json     ← Experiment names + tags
│   └── runs.json            ← Run params, full metric history, tags, status
└── artifacts/
    └── <model_name>/
        └── v<version>/
            └── model/       ← MLmodel, pkl, conda.yaml, requirements.txt…
```

### 🔄 Export / Import Examples

#### UC → UC (air-gapped)

```python
# --- On SOURCE workspace ---
MIGRATION_MODE   = "export"
SOURCE_REGISTRY  = "uc"
TARGET_REGISTRY  = "uc"
MODEL_NAMES      = ["prod_catalog.ml.fraud_detector"]
ARTIFACT_TEMP_DIR = "/Volumes/prod_catalog/staging/export_bundle"
```

```python
# --- On TARGET workspace ---
MIGRATION_MODE   = "import"
SOURCE_REGISTRY  = "uc"
TARGET_REGISTRY  = "uc"
UC_TARGET_CATALOG = "new_catalog"
UC_TARGET_SCHEMA  = "ml"
ARTIFACT_TEMP_DIR = "/Volumes/new_catalog/staging/export_bundle"
```

#### WS → WS (cross-workspace)

```python
# --- On SOURCE workspace ---
MIGRATION_MODE   = "export"
SOURCE_REGISTRY  = "workspace"
TARGET_REGISTRY  = "workspace"
MODEL_NAMES      = ["my_model_1", "my_model_2"]
ARTIFACT_TEMP_DIR = "/tmp/ws_export_bundle"
```

```python
# --- On TARGET workspace ---
MIGRATION_MODE    = "import"
SOURCE_REGISTRY   = "workspace"
TARGET_REGISTRY   = "workspace"
MODEL_NAME_PREFIX = "imported_"
ARTIFACT_TEMP_DIR = "/tmp/ws_export_bundle"
```

### 💡 Export / Import Tips

- 🧪 **Test with 1 model first** — verify the full export → transfer → import cycle before bulk runs
- 📊 **Tracking table works in both modes** — export writes `EXPORTED` status; import writes `COMPLETED`
- 🔄 **Resumable** — re-running import skips already-imported models (same dedup logic as direct mode)
- 💾 **Bundle is portable** — any path both workspaces can access works (`/Volumes/...`, `/dbfs/...`, `/tmp/...`)
- ⚡ **Fastest transfer** — use `/Volumes/` on both sides with cloud-native copy (`azcopy`, `gsutil`, `aws s3 sync`)
- 🗂️ **Manifest = source of truth** — the JSON manifests capture the exact source state at export time
- ⚠️ **Don't mix modes** — run export completely before starting import; don't run both on the same workspace simultaneously

---

## 📤📥 Export / Import Mode (Air-Gapped Migrations)

When source and target workspaces **cannot reach each other** (air-gapped, different clouds, strict firewall), use the **export → transfer → import** workflow instead of direct migration.

### How It Works

| Mode | Runs On | What It Does |
| --- | --- | --- |
| `"direct"` | **Either** (needs both) | Standard end-to-end migration (default, unchanged behavior) |
| `"export"` | **Source** workspace | Discovers models → downloads artifacts → writes JSON manifests to `ARTIFACT_TEMP_DIR`. No target operations. |
| `"import"` | **Target** workspace | Reads manifests + artifacts from `ARTIFACT_TEMP_DIR` → creates experiments, runs, models & versions on target. No source API calls. |

### Step-by-Step: Export / Import Workflow

#### Step 1️⃣ — Export from Source

On the **source** workspace, set:

```python
MIGRATION_MODE    = "export"
ARTIFACT_TEMP_DIR = "/tmp/ws_export_bundle"   # or /Volumes/..., /dbfs/tmp/...
# Configure SOURCE_HOST, SOURCE_TOKEN, MODEL_NAMES as usual
# TARGET settings are ignored in export mode
```

Run the notebook. The framework will:
```
📡 Phase 1 — Discovery
   └─ Scan source models, versions, experiments, runs

📦 Phase 2 — Export
   └─ Download all artifacts to ARTIFACT_TEMP_DIR/artifacts/
   └─ Write JSON manifests to ARTIFACT_TEMP_DIR/manifests/
   └─ ⏭️  Skip all target operations
```

#### Step 2️⃣ — Transfer the Bundle

Copy the `ARTIFACT_TEMP_DIR` contents to the target workspace using any method:

```bash
# Azure Blob → Azure Blob
azcopy copy "/tmp/ws_export_bundle/*" \
  "https://<target-storage>.blob.core.windows.net/export-bundle/" --recursive

# Upload to Unity Catalog Volume
databricks fs cp -r /tmp/ws_export_bundle \
  dbfs:/Volumes/catalog/schema/volume/export-bundle/

# Or: zip + scp, S3 sync, GCS transfer, etc.
```

#### Step 3️⃣ — Import on Target

On the **target** workspace, set:

```python
MIGRATION_MODE    = "import"
ARTIFACT_TEMP_DIR = "/tmp/ws_export_bundle"   # same path (or wherever you placed the bundle)
# Configure target settings (UC_TARGET_CATALOG, UC_TARGET_SCHEMA, MODEL_NAME_PREFIX, etc.)
# SOURCE_HOST / SOURCE_TOKEN are ignored in import mode
```

Run the notebook. The framework will:
```
📥 Phase 1 — Load Manifests
   └─ Read JSON manifests from ARTIFACT_TEMP_DIR/manifests/

🚚 Phase 2 — Import
   └─ Create experiments + clone runs from manifest data
   └─ Upload artifacts from bundle directory
   └─ Register model versions on target

✅ Phase 3 — Verification
   └─ Compare manifest counts vs target counts
```

### 📁 Export Bundle Structure

```
ARTIFACT_TEMP_DIR/
├── manifests/
│   ├── models.json          ← Model metadata (name, description, tags)
│   ├── versions.json        ← Version details (stage, tags, run mapping)
│   ├── experiments.json     ← Experiment names + tags
│   └── runs.json            ← Run params, metrics, tags, status
└── artifacts/
    └── <model_name>/
        └── <version>/
            └── model/       ← MLmodel, pkl, conda.yaml, etc.
```

### 💡 Export / Import Tips

- 🧪 **Test with 1 model first** — verify the full export → transfer → import cycle before bulk runs
- 📊 **Tracking table works in both modes** — export writes `EXPORTED` status; import writes `COMPLETED`
- 🔄 **Resumable** — re-running import skips already-imported models (same dedup logic as direct)
- 💾 **Bundle is portable** — any path both workspaces can access works (`/Volumes/...`, `/dbfs/...`, `/tmp/...`)
- ⚡ **Fastest transfer** — use `/Volumes/` on both sides with cloud-native copy (azcopy, gsutil, aws s3 sync)

---

## ⚙️ Configuration Reference

| Option | Purpose | Default |
| --- | --- | --- |
| `MODEL_NAMES` | Specific models to migrate (`[]` = scan all) | `[]` |
| `MODEL_NAME_PREFIX` | Prefix for WS target names (avoids collisions) | `""` |
| `UC_TARGET_CATALOG` | Override target catalog for UC targets | `""` (mirror source) |
| `UC_TARGET_SCHEMA` | Override target schema for UC targets | `""` (mirror source) |
| `TRACKING_TABLE` | Delta table for progress tracking | `""` |
| `INCLUDE_ARTIFACTS` | Copy model artifacts | `True` |
| `CREATE_DUMMY_VERSIONS` | Placeholder versions for deleted source runs | `True` |
| `INCLUDE_DELETED` | Include soft-deleted runs | `False` |
| `BATCH_SIZE` | Models per parallel batch | `10` |
| `INCLUDE_CATALOGS` | Only scan these catalogs (bulk UC scan) | `[]` (all) |
| `EXCLUDE_CATALOGS` | Skip these catalogs | `[]` |
| `EXCLUDE_SCHEMAS` | Skip these schemas (`catalog.schema`) | `[]` |
| `MIGRATION_MODE` | `"direct"` / `"export"` / `"import"` | `"direct"` |
| `ARTIFACT_TEMP_DIR` | Staging dir for artifacts & export bundles | `"/tmp/ws_export_bundle"` |
| `MIGRATION_MODE` | `"direct"` / `"export"` / `"import"` | `"direct"` |
| `ARTIFACT_TEMP_DIR` | Staging dir for artifacts & export bundles | `"/tmp/ws_export_bundle"` |

---

## 📊 Tracking Table

The Delta tracking table is your migration's single source of truth.

**Primary key:** `(source_host, model_name)` — safe for multi-workspace migrations.

| Column | Set By | Description |
| --- | --- | --- |
| `source_host` | Discovery | Source workspace URL |
| `model_name` | Discovery | Source model name |
| `readiness` | Discovery | `READY` / `PARTIAL` / `BLOCKED` |
| `migration_status` | Both | `PENDING` → `COMPLETED` / `PARTIAL` / `FAILED` |
| `source_versions` | Discovery | Total version count |
| `target_versions` | Migration | Migrated version count |
| `target_runs` | Migration | Cloned run count |
| `target_model_url` | Migration | 🔗 Link to target model |
| `target_experiment_urls` | Migration | 🔗 Links to target experiments |
| `migration_comments` | Migration | Error details |
| `last_updated_at` | Both | Timestamp of last change |

**Status flow:**
```
PENDING → IN_PROGRESS → COMPLETED ✅
                      → PARTIAL   ⚠️  (some versions migrated)
                      → FAILED    ❌  (zero versions, errors occurred)
```

---

## 🔧 Troubleshooting

### 🚨 Common Errors

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ValueError: Provide either a PAT token or a service principal` | Empty `SOURCE_HOST` / `SOURCE_TOKEN` | ✏️ Fill in credentials in the Configuration cell |
| `0 models discovered` | `MODEL_NAMES` doesn't match source | 🔍 Check for typos/whitespace; use `[]` for all models |
| `RESOURCE_DOES_NOT_EXIST` | Source run was permanently deleted | ✅ Expected — framework creates placeholder if `CREATE_DUMMY_VERSIONS=True` |
| `INVALID_PARAMETER_VALUE: Got an invalid source` | Stale framework in memory | 🔄 Re-run Cell 3 (force-reloads all modules) |
| `404 Client Error: Not Found for url: .../runs/get` | Source run deleted after version was created | ✅ Placeholder version created automatically |
| `Failed to download artifacts from path 'model'` | Model artifacts purged from source | ✅ Version created as placeholder; noted in Comments |
| `expected string or bytes-like object, got NoneType` | Model version has `run_id=None` | ✅ Fixed — versions without runs counted as ready |

### 🐌 Performance Issues

| Symptom | Cause | Fix |
| --- | --- | --- |
| Migration hangs / very slow | Rate limiting (HTTP 429) on source | ⬇️ Reduce `BATCH_SIZE` to `5` — auto-retry handles 429s |
| `Rate limited on uc_search_model_versions` | UC API rate limits (stricter than WS) | ✅ Normal — framework auto-retries with backoff (up to 5 attempts) |
| Takes hours for large migrations | Too many artifacts per run | ⚡ Set `INCLUDE_ARTIFACTS=False` for model-files-only (much faster) |
| `source_runs = 1000` for all models | Pagination cap in `search_runs` | ✅ Fixed — now paginates fully |

### 📊 Tracking Table Issues

| Symptom | Cause | Fix |
| --- | --- | --- |
| Tracking table not created | Missing catalog/schema permissions | 🔐 Ensure `CREATE TABLE` permission on the target schema |
| Status stuck at `PENDING` | Migration failed before status update | 👀 Check Comments column; fix the error and re-run |
| All models re-migrating on re-run | `TRACKING_TABLE` not set | ✏️ Set a tracking table path to enable resume |
| Tracking table not updating from threads | Spark SQL called from worker thread | ✅ Fixed — updates run on main thread after `future.result()` |
| `'function' object has no attribute 'get'` | UC SDK bug — `ModelVersionSearch.tags` is a method, not a dict | ✅ **Fixed in v2** — uses `get_model_version` per version |
| `'function' object has no attribute 'items'` | Same UC SDK bug in tracking table update | ✅ **Fixed in v2** — same approach |

### ✅ Verification Mismatches

| Symptom | Cause | Fix |
| --- | --- | --- |
| `[N] → [2N] mismatch` | Re-ran migration without cleaning target | 🧹 Clean target models or ensure dedup is on |
| `[4] → [6] mismatch` | Source has non-READY versions; target has all READY | ✅ Expected when `CREATE_DUMMY_VERSIONS=True` — placeholders are READY |
| Target counts > source | Same-workspace test accumulates versions | ✅ Expected — real cross-workspace won't have this |

### 🔐 Permission Errors

| Symptom | Cause | Fix |
| --- | --- | --- |
| `PERMISSION_DENIED` on source | PAT/SP lacks MLflow read access | 🛡️ Grant `Can Read` on source experiments + models |
| `PERMISSION_DENIED` on target | Can't create models/experiments | 🛡️ Grant `Can Manage` on target model registry |
| `PERMISSION_DENIED` on UC target | Missing UC privileges | 🛡️ Grant `USE CATALOG`, `USE SCHEMA`, `CREATE MODEL` |
| `KeyError: 'readiness'` | Empty inventory DataFrame (0 models) | ✅ Fixed — `print_inventory_summary` now handles empty DFs |
| All models show `❌ BLOCKED` | Wrong client used for inventory | ✅ Fixed — pass `source_context=migrator.source` |

---

## 🔁 Concurrency Model

```
execute_migration()
├── 📡 discover()                               ← Phase 1
│   ├── ThreadPool: fetch model versions           (parallel per model)
│   ├── ThreadPool: fetch experiment runs           (parallel per experiment)
│   └── MERGE into tracking table                   (main thread, Spark-safe)
├── 🚚 _migrate_experiments()                    ← Phase 2a
│   └── batched ThreadPool                          (parallel per experiment)
├── 🚚 _migrate_models()                        ← Phase 2b
│   └── for batch in chunked(models, batch_size):
│       ├── Phase 1: ThreadPool download artifacts   (high concurrency)
│       └── Phase 2: ThreadPool register versions    (sequential per model)
│           └── UPDATE tracking table                (main thread after each model)
└── ✅ Verification                              ← Phase 3
    └── Compare source vs target version counts
```

> ⚠️ **Thread safety:** Delta tracking updates (`spark.sql`) always run on the **main thread** — Spark SQL is NOT thread-safe on serverless.

> 📤 **Export mode** runs only Phases 1–2a (discovery + artifact download + manifest write) — no target operations.
> 📥 **Import mode** skips Phase 1 discovery and reads from manifests instead — no source API calls.
>
> 📤 **Export mode** runs only discovery + artifact download + manifest write — no target operations.
>
> 📥 **Import mode** skips source discovery and reads from manifests instead — no source API calls.

**Tuning tips:**
- `BATCH_SIZE=10` — good default for serverless
- Hitting 429s? → Reduce to `5`
- Large workspace (1000+ models)? → Keep at `10`, the framework handles batching

---

## 💡 Tips & Tricks

- 🧪 **Test first** — run with 1-2 models + a prefix before bulk migration
- 📊 **Track everything** — always set `TRACKING_TABLE` for production migrations
- 🔄 **Idempotent by design** — safe to re-run; completed models are skipped
- 🧹 **Clean up tests** — use the Cleanup cell (Cell 4) to drop test schemas
- ⏱️ **Speed up large runs** — set `INCLUDE_ARTIFACTS=False` for model-files-only
- 🔒 **Never hardcode tokens** — use `dbutils.secrets.get()` in production
- 📤 **Air-gapped?** — use `MIGRATION_MODE="export"` on source, transfer the bundle, then `"import"` on target
- 📁 **Bundle path** — `ARTIFACT_TEMP_DIR` works with `/Volumes/`, `/dbfs/`, `/tmp/`, or cloud storage
- 📤 **Air-gapped?** — use `MIGRATION_MODE="export"` on source, transfer the bundle, then `"import"` on target
- 📁 **Bundle path** — `ARTIFACT_TEMP_DIR` works with `/Volumes/`, `/dbfs/`, `/tmp/`, or cloud storage paths
