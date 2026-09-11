"""Thread-safe REST client for Databricks MLflow operations.

All source-workspace operations — including artifact downloads — use
per-instance credentials carried in requests.Session headers.  Artifact
downloads go through presigned URLs (POST credentials-for-read), so
no environment variables are ever read or mutated.  Each instance is
fully isolated — safe for concurrent use from any number of threads.
"""

from __future__ import annotations

import functools
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("workspace_registry_migrator.rest_client")

# ---------------------------------------------------------------------------
# Retry decorator (same semantics as the original _retry_on_rate_limit)
# ---------------------------------------------------------------------------

def _retry_on_rate_limit(
    max_retries: int = 5,
    initial_backoff: float = 2.0,
    backoff_factor: float = 2.0,
    retryable_status_codes: tuple[int, ...] = (429, 500, 503),
):
    """Decorator that retries on HTTP rate-limit or transient server errors."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            backoff = initial_backoff
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except requests.HTTPError as exc:
                    status = exc.response.status_code if exc.response is not None else 0
                    if status not in retryable_status_codes or attempt == max_retries:
                        raise
                    sleep_time = backoff + (attempt * 0.5)
                    logger.warning(
                        f"Rate limited on {func.__name__}, retrying in {sleep_time:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries}) [HTTP {status}]"
                    )
                    time.sleep(sleep_time)
                    backoff *= backoff_factor
                except Exception as exc:
                    exc_str = str(exc)
                    is_retryable = any(tok in exc_str for tok in (
                        "429", "RESOURCE_EXHAUSTED", "Too Many Requests",
                        "503", "TEMPORARILY_UNAVAILABLE",
                    ))
                    if not is_retryable or attempt == max_retries:
                        raise
                    sleep_time = backoff + (attempt * 0.5)
                    logger.warning(
                        f"Transient error on {func.__name__}, retrying in {sleep_time:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(sleep_time)
                    backoff *= backoff_factor
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArtifactInfo:
    """Lightweight representation of an MLflow artifact entry."""
    path: str
    is_dir: bool
    file_size: int | None = None


@dataclass(frozen=True)
class MLflowVersionInfo:
    """MLflow version details from a workspace."""
    mlflow_version: str
    raw_response: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# REST Client
# ---------------------------------------------------------------------------

class DatabricksRestClient:
    """Thread-safe REST client for a single Databricks workspace.

    Each instance carries its own host + auth in a ``requests.Session`` with
    connection pooling. No environment variables are touched — safe for
    concurrent use from multiple threads.

    Supports:
    - PAT authentication (token)
    - Service principal OAuth M2M (client_id + client_secret)
    """

    def __init__(
        self,
        host: str,
        token: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        pool_connections: int = 20,
        pool_maxsize: int = 20,
    ) -> None:
        self.host = host.rstrip("/")
        self._token = token
        self._client_id = client_id
        self._client_secret = client_secret

        self._session = requests.Session()

        # Connection pooling
        adapter = HTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            max_retries=Retry(total=0),  # we handle retries ourselves
        )
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

        # Auth header
        self._auth_mode = "none"
        if token:
            self._session.headers["Authorization"] = f"Bearer {token}"
            self._auth_mode = "pat"
        elif client_id and client_secret:
            self._oauth_token: str | None = None
            self._oauth_expiry: float = 0.0
            self._auth_mode = "oauth"
        # If no credentials, allow construction for current-workspace use
        # (caller must ensure the host is reachable)

    # ---- OAuth M2M token refresh ----

    def _ensure_oauth_token(self) -> None:
        """Fetch/refresh OAuth token for service principal auth."""
        if self._auth_mode != "oauth":
            return  # PAT auth or no-auth — nothing to do
        if self._oauth_token and time.time() < self._oauth_expiry - 60:
            return  # Still valid (with 60s buffer)
        resp = requests.post(
            f"{self.host}/oidc/v1/token",
            data={
                "grant_type": "client_credentials",
                "scope": "all-apis",
            },
            auth=(self._client_id, self._client_secret),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self._oauth_token = data["access_token"]
        self._oauth_expiry = time.time() + data.get("expires_in", 3600)
        self._session.headers["Authorization"] = f"Bearer {self._oauth_token}"

    # ---- Low-level HTTP ----

    def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        json_body: dict | None = None,
        stream: bool = False,
        timeout: int = 120,
    ) -> requests.Response:
        self._ensure_oauth_token()
        url = f"{self.host}{path}"
        resp = self._session.request(
            method, url, params=params, json=json_body,
            stream=stream, timeout=timeout,
        )
        resp.raise_for_status()
        return resp

    def _get_json(self, path: str, params: dict | None = None) -> dict:
        return self._request("GET", path, params=params).json()

    def _post_json(self, path: str, json_body: dict | None = None) -> dict:
        return self._request("POST", path, json_body=json_body).json()

    # ---- MLflow Version ----

    @_retry_on_rate_limit()
    def get_mlflow_version(self) -> MLflowVersionInfo:
        """GET /api/2.0/mlflow/version — returns the MLflow tracking server version."""
        try:
            data = self._get_json("/api/2.0/mlflow/version")
            return MLflowVersionInfo(
                mlflow_version=data.get("version", "unknown"),
                raw_response=data,
            )
        except Exception:
            return MLflowVersionInfo(mlflow_version="unknown")

    # ---- Registered Models ----

    @_retry_on_rate_limit()
    def search_registered_models(
        self,
        max_results: int = 100,
        page_token: str | None = None,
        filter_string: str | None = None,
    ) -> dict:
        """GET /api/2.0/mlflow/registered-models/search"""
        params: dict[str, Any] = {"max_results": max_results}
        if page_token:
            params["page_token"] = page_token
        if filter_string:
            params["filter"] = filter_string
        return self._get_json("/api/2.0/mlflow/registered-models/search", params=params)

    def search_all_registered_models(self, filter_string: str | None = None) -> list[dict]:
        """Paginated search — returns ALL registered models."""
        models: list[dict] = []
        page_token: str | None = None
        while True:
            data = self.search_registered_models(
                max_results=100, page_token=page_token, filter_string=filter_string,
            )
            models.extend(data.get("registered_models", []))
            page_token = data.get("next_page_token")
            if not page_token:
                return models

    @_retry_on_rate_limit()
    def get_registered_model(self, name: str) -> dict:
        """GET /api/2.0/mlflow/registered-models/get"""
        return self._get_json("/api/2.0/mlflow/registered-models/get", params={"name": name})

    @_retry_on_rate_limit()
    def create_registered_model(
        self, name: str, description: str | None = None, tags: list[dict] | None = None,
    ) -> dict:
        """POST /api/2.0/mlflow/registered-models/create"""
        body: dict[str, Any] = {"name": name}
        if description:
            body["description"] = description
        if tags:
            body["tags"] = tags
        return self._post_json("/api/2.0/mlflow/registered-models/create", json_body=body)

    # ---- Model Versions ----

    @_retry_on_rate_limit()
    def search_model_versions(
        self,
        filter_string: str,
        max_results: int = 200,
        page_token: str | None = None,
    ) -> dict:
        """GET /api/2.0/mlflow/model-versions/search"""
        params: dict[str, Any] = {"filter": filter_string, "max_results": max_results}
        if page_token:
            params["page_token"] = page_token
        return self._get_json("/api/2.0/mlflow/model-versions/search", params=params)

    def search_all_model_versions(self, model_name: str) -> list[dict]:
        """Paginated search — returns ALL versions for a model."""
        escaped = model_name.replace("'", "''")
        filter_str = f"name='{escaped}'"
        versions: list[dict] = []
        page_token: str | None = None
        while True:
            data = self.search_model_versions(
                filter_string=filter_str, max_results=200, page_token=page_token,
            )
            versions.extend(data.get("model_versions", []))
            page_token = data.get("next_page_token")
            if not page_token:
                return versions

    @_retry_on_rate_limit()
    def create_model_version(
        self,
        name: str,
        source: str,
        run_id: str | None = None,
        description: str | None = None,
        tags: list[dict] | None = None,
    ) -> dict:
        """POST /api/2.0/mlflow/model-versions/create"""
        body: dict[str, Any] = {"name": name, "source": source}
        if run_id:
            body["run_id"] = run_id
        if description:
            body["description"] = description
        if tags:
            body["tags"] = tags
        return self._post_json("/api/2.0/mlflow/model-versions/create", json_body=body)

    @_retry_on_rate_limit()
    def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        """POST /api/2.0/mlflow/model-versions/set-tag"""
        self._post_json("/api/2.0/mlflow/model-versions/set-tag", json_body={
            "name": name, "version": version, "key": key, "value": value,
        })

    @_retry_on_rate_limit()
    def transition_model_version_stage(
        self, name: str, version: str, stage: str, archive_existing_versions: bool = False,
    ) -> dict:
        """POST /api/2.0/mlflow/model-versions/transition-stage"""
        return self._post_json("/api/2.0/mlflow/model-versions/transition-stage", json_body={
            "name": name, "version": version, "stage": stage,
            "archive_existing_versions": archive_existing_versions,
        })

    @_retry_on_rate_limit()
    def set_registered_model_alias(
        self, name: str, alias: str, version: str,
    ) -> None:
        """POST /api/2.0/mlflow/registered-models/alias (UC registry only)."""
        self._post_json("/api/2.0/mlflow/registered-models/alias", json_body={
            "name": name, "alias": alias, "version": version,
        })

    @_retry_on_rate_limit()
    def set_registered_model_tag(
        self, name: str, key: str, value: str,
    ) -> None:
        """POST /api/2.0/mlflow/registered-models/set-tag"""
        self._post_json("/api/2.0/mlflow/registered-models/set-tag", json_body={
            "name": name, "key": key, "value": value,
        })

    # ---- Unity Catalog Model Registry ----

    @_retry_on_rate_limit()
    def uc_get_registered_model(self, name: str) -> dict:
        """GET /api/2.0/mlflow/unity-catalog/registered-models/get"""
        return self._get_json("/api/2.0/mlflow/unity-catalog/registered-models/get", params={"name": name})

    @_retry_on_rate_limit()
    def uc_create_registered_model(
        self, name: str, description: str | None = None, tags: list[dict] | None = None,
    ) -> dict:
        """POST /api/2.0/mlflow/unity-catalog/registered-models/create"""
        body: dict[str, Any] = {"name": name}
        if description:
            body["description"] = description
        if tags:
            body["tags"] = tags
        return self._post_json("/api/2.0/mlflow/unity-catalog/registered-models/create", json_body=body)

    @_retry_on_rate_limit()
    def uc_search_model_versions(
        self, filter_string: str, max_results: int = 200, page_token: str | None = None,
    ) -> dict:
        """GET /api/2.0/mlflow/unity-catalog/model-versions/search"""
        params: dict[str, Any] = {"filter": filter_string, "max_results": max_results}
        if page_token:
            params["page_token"] = page_token
        return self._get_json("/api/2.0/mlflow/unity-catalog/model-versions/search", params=params)

    def uc_search_all_model_versions(self, model_name: str) -> list[dict]:
        """Paginated search for all UC model versions."""
        escaped = model_name.replace("'", "''")
        filter_str = f"name='{escaped}'"
        versions: list[dict] = []
        page_token: str | None = None
        while True:
            data = self.uc_search_model_versions(
                filter_string=filter_str, max_results=200, page_token=page_token,
            )
            versions.extend(data.get("model_versions", []))
            page_token = data.get("next_page_token")
            if not page_token:
                return versions

    @_retry_on_rate_limit()
    def uc_create_model_version(
        self,
        name: str,
        source: str,
        run_id: str | None = None,
        description: str | None = None,
        tags: list[dict] | None = None,
    ) -> dict:
        """POST /api/2.0/mlflow/unity-catalog/model-versions/create"""
        body: dict[str, Any] = {"name": name, "source": source}
        if run_id:
            body["run_id"] = run_id
        if description:
            body["description"] = description
        if tags:
            body["tags"] = tags
        return self._post_json("/api/2.0/mlflow/unity-catalog/model-versions/create", json_body=body)

    @_retry_on_rate_limit()
    def uc_set_model_version_tag(
        self, name: str, version: str, key: str, value: str,
    ) -> None:
        """POST /api/2.0/mlflow/unity-catalog/model-versions/set-tag"""
        self._post_json("/api/2.0/mlflow/unity-catalog/model-versions/set-tag", json_body={
            "name": name, "version": version, "key": key, "value": value,
        })

    @_retry_on_rate_limit()
    def uc_set_registered_model_alias(
        self, name: str, alias: str, version: str,
    ) -> None:
        """POST /api/2.0/mlflow/unity-catalog/registered-models/alias"""
        self._post_json("/api/2.0/mlflow/unity-catalog/registered-models/alias", json_body={
            "name": name, "alias": alias, "version": version,
        })

    # ---- Experiments ----

    @_retry_on_rate_limit()
    def get_experiment(self, experiment_id: str) -> dict:
        """GET /api/2.0/mlflow/experiments/get"""
        return self._get_json("/api/2.0/mlflow/experiments/get", params={"experiment_id": experiment_id})

    @_retry_on_rate_limit()
    def get_experiment_by_name(self, name: str) -> dict | None:
        """GET /api/2.0/mlflow/experiments/get-by-name"""
        try:
            return self._get_json("/api/2.0/mlflow/experiments/get-by-name", params={"experiment_name": name})
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            raise

    @_retry_on_rate_limit()
    def create_experiment(self, name: str, tags: list[dict] | None = None) -> dict:
        """POST /api/2.0/mlflow/experiments/create"""
        body: dict[str, Any] = {"name": name}
        if tags:
            body["tags"] = tags
        return self._post_json("/api/2.0/mlflow/experiments/create", json_body=body)

    @_retry_on_rate_limit()
    def set_experiment_tag(self, experiment_id: str, key: str, value: str) -> None:
        """POST /api/2.0/mlflow/experiments/set-experiment-tag"""
        self._post_json("/api/2.0/mlflow/experiments/set-experiment-tag", json_body={
            "experiment_id": experiment_id, "key": key, "value": value,
        })

    # ---- Runs ----

    @_retry_on_rate_limit()
    def get_run(self, run_id: str) -> dict:
        """GET /api/2.0/mlflow/runs/get"""
        return self._get_json("/api/2.0/mlflow/runs/get", params={"run_id": run_id})

    @_retry_on_rate_limit()
    def search_runs(
        self,
        experiment_ids: list[str],
        filter_string: str = "",
        max_results: int = 1000,
        order_by: list[str] | None = None,
        run_view_type: int = 1,
        page_token: str | None = None,
    ) -> dict:
        """POST /api/2.0/mlflow/runs/search"""
        body: dict[str, Any] = {
            "experiment_ids": experiment_ids,
            "max_results": max_results,
            "run_view_type": run_view_type,
        }
        if filter_string:
            body["filter"] = filter_string
        if order_by:
            body["order_by"] = order_by
        if page_token:
            body["page_token"] = page_token
        return self._post_json("/api/2.0/mlflow/runs/search", json_body=body)

    def search_all_runs(
        self,
        experiment_ids: list[str],
        filter_string: str = "",
        include_deleted: bool = False,
        max_runs: int | None = None,
    ) -> list[dict]:
        """Paginated search — returns ALL runs (or up to max_runs)."""
        all_runs: list[dict] = []
        page_token: str | None = None
        run_view = 3 if include_deleted else 1
        while True:
            data = self.search_runs(
                experiment_ids=experiment_ids,
                filter_string=filter_string,
                max_results=1000,
                order_by=["attributes.start_time DESC"],
                run_view_type=run_view,
                page_token=page_token,
            )
            all_runs.extend(data.get("runs", []))
            if max_runs and len(all_runs) >= max_runs:
                return all_runs[:max_runs]
            page_token = data.get("next_page_token")
            if not page_token:
                return all_runs

    @_retry_on_rate_limit()
    def create_run(
        self,
        experiment_id: str,
        start_time: int | None = None,
        tags: list[dict] | None = None,
        run_name: str | None = None,
    ) -> dict:
        """POST /api/2.0/mlflow/runs/create"""
        body: dict[str, Any] = {"experiment_id": experiment_id}
        if start_time:
            body["start_time"] = start_time
        if tags:
            body["tags"] = tags
        if run_name:
            body["run_name"] = run_name
        return self._post_json("/api/2.0/mlflow/runs/create", json_body=body)

    @_retry_on_rate_limit()
    def log_batch(
        self,
        run_id: str,
        params: list[dict] | None = None,
        metrics: list[dict] | None = None,
        tags: list[dict] | None = None,
    ) -> None:
        """POST /api/2.0/mlflow/runs/log-batch"""
        body: dict[str, Any] = {"run_id": run_id}
        if params:
            body["params"] = params
        if metrics:
            body["metrics"] = metrics
        if tags:
            body["tags"] = tags
        self._post_json("/api/2.0/mlflow/runs/log-batch", json_body=body)

    @_retry_on_rate_limit()
    def update_run(
        self, run_id: str, status: str, end_time: int | None = None,
    ) -> None:
        """POST /api/2.0/mlflow/runs/update"""
        body: dict[str, Any] = {"run_id": run_id, "status": status}
        if end_time:
            body["end_time"] = end_time
        self._post_json("/api/2.0/mlflow/runs/update", json_body=body)

    @_retry_on_rate_limit()
    def get_metric_history(self, run_id: str, metric_key: str) -> list[dict]:
        """GET /api/2.0/mlflow/metrics/get-history"""
        data = self._get_json("/api/2.0/mlflow/metrics/get-history", params={
            "run_id": run_id, "metric_key": metric_key,
        })
        return data.get("metrics", [])

    # ---- Artifacts ----

    @_retry_on_rate_limit()
    def list_artifacts(self, run_id: str, path: str = "") -> list[ArtifactInfo]:
        """GET /api/2.0/mlflow/artifacts/list"""
        params: dict[str, str] = {"run_id": run_id}
        if path:
            params["path"] = path
        data = self._get_json("/api/2.0/mlflow/artifacts/list", params=params)
        return [
            ArtifactInfo(
                path=f.get("path", ""),
                is_dir=f.get("is_dir", False),
                file_size=f.get("file_size"),
            )
            for f in data.get("files", [])
        ]

    # ---- Artifact Download via Presigned URLs ----

    @_retry_on_rate_limit()
    def get_artifact_presigned_urls(
        self, run_id: str, paths: list[str],
    ) -> list[dict[str, Any]]:
        """POST /api/2.0/mlflow/artifacts/credentials-for-read

        Returns credential_infos — each entry has ``path``, ``signed_uri``,
        and ``type`` (e.g. ``GCP_SIGNED_URL``, ``AWS_PRESIGNED_URL``,
        ``AZURE_SAS_URI``).
        """
        data = self._post_json(
            "/api/2.0/mlflow/artifacts/credentials-for-read",
            json_body={"run_id": run_id, "path": paths},
        )
        return data.get("credential_infos", [])

    def _list_artifact_files_recursive(
        self, run_id: str, root_path: str,
    ) -> list[str]:
        """Recursively enumerate all *file* paths under ``root_path``."""
        files: list[str] = []
        stack = [root_path]
        while stack:
            current = stack.pop()
            entries = self.list_artifacts(run_id, path=current)
            for entry in entries:
                if not entry.path:
                    continue
                if entry.is_dir:
                    stack.append(entry.path)
                else:
                    files.append(entry.path)
        return files

    def download_artifact_tree(
        self,
        run_id: str,
        artifact_path: str,
        dst_path: str,
        *,
        batch_size: int = 8,
        max_workers: int = 10,
    ) -> str:
        """Download an artifact tree via presigned URLs — fully thread-safe.

        1. Recursively lists all files under *artifact_path*.
        2. Batches the paths and fetches presigned URLs.
        3. Downloads each file with a plain ``requests.get()`` (no auth headers).
        4. Writes to *dst_path* preserving the directory structure.

        No environment variables are read or mutated — safe for concurrent
        use from any number of threads.

        Returns the local path to the downloaded artifact root.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 1. Enumerate files
        all_files = self._list_artifact_files_recursive(run_id, artifact_path)
        if not all_files:
            # artifact_path may itself be a single file
            all_files = [artifact_path]

        # 2. Fetch presigned URLs in batches
        url_map: dict[str, str] = {}  # artifact_path -> signed_uri
        for i in range(0, len(all_files), batch_size):
            batch = all_files[i : i + batch_size]
            cred_infos = self.get_artifact_presigned_urls(run_id, batch)
            for info in cred_infos:
                signed_uri = info.get("signed_uri", "")
                info_path = info.get("path", "")
                if signed_uri and info_path:
                    url_map[info_path] = signed_uri

        # 3. Download files in parallel
        errors: list[str] = []

        def _download_one(artifact_file_path: str, signed_uri: str) -> None:
            # Compute local destination preserving directory structure
            local_path = os.path.join(dst_path, artifact_file_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            resp = requests.get(signed_uri, stream=True, timeout=300)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_download_one, path, uri): path
                for path, uri in url_map.items()
            }
            for future in as_completed(futures):
                path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    errors.append(f"{path}: {exc}")
                    logger.warning(f"Presigned download failed for {path}: {exc}")

        if errors and len(errors) == len(url_map):
            raise RuntimeError(
                f"All {len(errors)} artifact downloads failed. "
                f"First error: {errors[0]}"
            )

        return os.path.join(dst_path, artifact_path)

    # ---- Registered Model Tags ----

    @_retry_on_rate_limit()
    def set_registered_model_tag(self, name: str, key: str, value: str) -> None:
        """POST /api/2.0/mlflow/registered-models/set-tag"""
        self._post_json("/api/2.0/mlflow/registered-models/set-tag", json_body={
            "name": name, "key": key, "value": value,
        })


# ---------------------------------------------------------------------------
# Artifact Upload (target workspace — no env var mutation)
# ---------------------------------------------------------------------------

class TargetArtifactUploader:
    """Uploads artifacts to the TARGET (current) workspace.

    Uses the native mlflow client — no env var changes needed since the
    target is always the current workspace where the notebook runs.

    This class is used in Phase 2 (register only) of the two-phase pipeline.
    """

    def __init__(self) -> None:
        import mlflow
        from mlflow import MlflowClient
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks")
        self._client = MlflowClient(tracking_uri="databricks", registry_uri="databricks")

    @_retry_on_rate_limit(max_retries=3)
    def log_artifact(self, run_id: str, local_path: str, artifact_path: str | None = None) -> None:
        self._client.log_artifact(run_id=run_id, local_path=local_path, artifact_path=artifact_path)

    @_retry_on_rate_limit(max_retries=3)
    def log_artifacts(self, run_id: str, local_dir: str, artifact_path: str | None = None) -> None:
        self._client.log_artifacts(run_id=run_id, local_dir=local_dir, artifact_path=artifact_path)

    def list_artifacts(self, run_id: str, path: str = "") -> list:
        return self._client.list_artifacts(run_id=run_id, path=path)

    def get_run(self, run_id: str):
        return self._client.get_run(run_id)


# ---------------------------------------------------------------------------
# Compatibility Checker
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompatReport:
    """Compatibility assessment between source and target workspaces."""
    source_mlflow_version: str
    target_mlflow_version: str
    source_uses_stages: bool
    target_uses_stages: bool
    stage_to_alias_needed: bool
    warnings: list[str]

    def summary(self) -> str:
        lines = [
            f"Source MLflow: {self.source_mlflow_version}",
            f"Target MLflow: {self.target_mlflow_version}",
        ]
        if self.stage_to_alias_needed:
            lines.append("Stage-to-alias mapping: REQUIRED (target uses aliases instead of stages)")
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


def check_compatibility(
    source_client: DatabricksRestClient,
    target_client: DatabricksRestClient,
) -> CompatReport:
    """Compare MLflow versions between source and target workspaces."""
    source_info = source_client.get_mlflow_version()
    target_info = target_client.get_mlflow_version()

    warnings: list[str] = []
    src_v = source_info.mlflow_version
    tgt_v = target_info.mlflow_version

    if src_v != tgt_v and src_v != "unknown" and tgt_v != "unknown":
        warnings.append(
            f"MLflow version mismatch: source={src_v}, target={tgt_v}. "
            "API behavior may differ (pagination tokens, log_batch format, model stages vs aliases)."
        )

    # Determine if stages vs aliases
    def _version_tuple(v: str) -> tuple[int, ...]:
        try:
            return tuple(int(x) for x in v.split(".")[:3])
        except (ValueError, AttributeError):
            return (0, 0, 0)

    src_vt = _version_tuple(src_v)
    tgt_vt = _version_tuple(tgt_v)

    # MLflow >= 2.9 deprecated model stages in favor of aliases
    source_uses_stages = src_vt < (2, 9, 0) or src_v == "unknown"
    target_uses_stages = tgt_vt < (2, 9, 0) or tgt_v == "unknown"
    stage_to_alias_needed = not target_uses_stages and source_uses_stages

    if stage_to_alias_needed:
        warnings.append(
            "Source workspace uses model stages; target uses aliases (MLflow >= 2.9). "
            "Stages will be mapped: Production → Champion, Staging → Challenger."
        )

    # Check for log_batch differences
    if src_vt < (2, 0, 0) and tgt_vt >= (2, 0, 0):
        warnings.append(
            "Major MLflow version boundary crossed (1.x → 2.x+). "
            "log_batch parameter formats may differ."
        )

    return CompatReport(
        source_mlflow_version=src_v,
        target_mlflow_version=tgt_v,
        source_uses_stages=source_uses_stages,
        target_uses_stages=target_uses_stages,
        stage_to_alias_needed=stage_to_alias_needed,
        warnings=warnings,
    )
