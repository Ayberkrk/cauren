"""Small, dependency-free helpers for reproducible local data builds."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a stable SHA-256 digest without reading the whole file into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, source_id: str) -> dict[str, object]:
    """Describe a data input by portable name, size, and content hash."""
    return {
        "source_id": source_id,
        "file_name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def package_versions(package_names: Iterable[str]) -> dict[str, str | None]:
    """Capture installed distribution versions, retaining missing packages."""
    versions: dict[str, str | None] = {}
    for package_name in package_names:
        try:
            versions[package_name] = version(package_name)
        except PackageNotFoundError:
            versions[package_name] = None
    return versions
