from __future__ import annotations

import contextlib
import os
import re
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def sanitize_name(value: str) -> str:
    """Convert arbitrary text into a stable path-safe token."""
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "item"


def chunked(items: list[Any], chunk_size: int) -> Iterable[list[Any]]:
    """Yield fixed-size chunks from a list."""
    for index in range(0, len(items), chunk_size):
        yield items[index : index + chunk_size]


def ensure_directory(path: str | Path) -> str:
    """Create a directory if needed and return its string path."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return str(target)


@contextlib.contextmanager
def temporary_directory(prefix: str, parent_dir: str | None = None) -> Iterable[str]:
    """Create and clean up a temporary directory."""
    root = parent_dir or tempfile.gettempdir()
    ensure_directory(root)
    temp_dir = tempfile.mkdtemp(prefix=prefix, dir=root)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class NotebookLogger:
    """Simple print-based logger without progress widgets."""

    def __init__(self, enabled: bool = True, show_warnings: bool = False) -> None:
        self.enabled = enabled
        self.show_warnings = show_warnings

    def info(self, message: str) -> None:
        if self.enabled:
            print(f"[INFO] {message}")

    def warning(self, message: str) -> None:
        if self.enabled and self.show_warnings:
            print(f"[WARN] {message}")

    def error(self, message: str) -> None:
        if self.enabled:
            print(f"[ERROR] {message}")


def env_or_value(env_name: str, explicit_value: str | None) -> str | None:
    """Prefer explicit value, then environment variable."""
    return explicit_value if explicit_value not in (None, "") else os.getenv(env_name)
