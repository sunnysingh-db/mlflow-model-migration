# Workspace Registry MLflow Migration

This framework migrates Databricks workspace-registry MLflow assets from a source workspace into the current workspace.

## Scope

* Only workspace registry models are included.
* Unity Catalog models are intentionally excluded.
* Experiments are recreated under `/Shared`.
* Runs, params, metric history, tags, model versions, and artifacts are migrated.
* Parallel execution is controlled with `batch_size` and `max_workers`.
* Notebook progress uses plain prints and disables tqdm-based widget noise.

## Layout

* `framework.py` contains the reusable migration framework.
* `utils.py` contains shared helpers for chunking, temp directories, and logging.
* The root notebook is intentionally lean and only orchestrates inputs and execution.

## Notebook flow

1. Fill in the source workspace host and token.
2. Review migration options.
3. Run discovery to inspect source inventory.
4. Run migration when ready.

## Important behavior

* Source access uses explicit workspace credentials supplied by the user.
* Target access uses the current workspace runtime context.
* Registered model names remain workspace-registry names unless a prefix is provided.
* Existing target runs are detected by the `source_run_id` tag.
* Existing target model versions can be skipped using `skip_existing_model_versions=True`.
