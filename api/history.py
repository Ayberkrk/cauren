"""Small local history store for single-operator Cauren deployments."""

from __future__ import annotations

import errno
import json
import logging
import os
import sqlite3
import stat
import time
from contextlib import closing
from pathlib import Path
from typing import Any

DEFAULT_RETENTION_DAYS = 90
logger = logging.getLogger(__name__)


def database_path() -> Path:
    configured = os.getenv("CAUREN_HISTORY_DB", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "share" / "cauren" / "history.sqlite3"


def retention_days() -> int:
    raw = os.getenv("CAUREN_HISTORY_RETENTION_DAYS", str(DEFAULT_RETENTION_DAYS))
    try:
        days = int(raw)
    except ValueError as exc:
        raise ValueError("CAUREN_HISTORY_RETENTION_DAYS must be a positive integer") from exc
    if days < 1:
        raise ValueError("CAUREN_HISTORY_RETENTION_DAYS must be a positive integer")
    return days


def _connect() -> sqlite3.Connection:
    path = database_path()
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
        created_parent = True
    except FileExistsError:
        created_parent = False

    # Resolve the parent once, then operate on the canonical basename. SQLite
    # follows symlinks, so checking the final component before chmod/connect is
    # not sufficient unless the parent also prevents another principal from
    # replacing it between those operations.
    parent = path.parent.resolve(strict=True)
    parent_stat = parent.stat()
    if os.name == "posix" and parent_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("CAUREN_HISTORY_DB parent directory must not be group- or world-writable")
    path = parent / path.name

    if os.name == "posix":
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        flags = os.O_CREAT | os.O_WRONLY | nofollow
        if created_parent:
            try:
                os.chmod(path.parent, 0o700)
            except OSError as exc:
                logger.warning("could not restrict history directory permissions: %s", exc.__class__.__name__)
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ValueError("CAUREN_HISTORY_DB must not be a symbolic link") from exc
            raise
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError("CAUREN_HISTORY_DB must be a regular file")
            if file_stat.st_nlink > 1:
                raise ValueError("CAUREN_HISTORY_DB must not be a hard link")
            try:
                os.fchmod(descriptor, 0o600)
            except OSError as exc:
                # Keep group-owned mounted volumes usable when their owner
                # controls the mode; the protected parent and no-follow open
                # still prevent another local user from swapping the path.
                logger.warning("could not restrict history file permissions: %s", exc.__class__.__name__)
        finally:
            os.close(descriptor)
    else:
        if path.is_symlink():
            raise ValueError("CAUREN_HISTORY_DB must not be a symbolic link")
        if path.exists() and not path.is_file():
            raise ValueError("CAUREN_HISTORY_DB must be a regular file")

    connection = sqlite3.connect(path, timeout=10.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 10000")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS diagnosis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id TEXT NOT NULL,
            recorded_at REAL NOT NULL,
            kind TEXT NOT NULL,
            result_json TEXT NOT NULL
        )"""
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_diagnosis_history_asset_time "
        "ON diagnosis_history(asset_id, recorded_at DESC)"
    )
    return connection


def persist_history(*, asset_id: str, kind: str, result: dict[str, Any], recorded_at: float | None = None) -> None:
    """Persist a response without the raw request (which can contain addresses)."""
    now = float(recorded_at if recorded_at is not None else time.time())
    cutoff = now - retention_days() * 24 * 60 * 60
    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    with closing(_connect()) as connection:
        with connection:
            connection.execute("DELETE FROM diagnosis_history WHERE recorded_at < ?", (cutoff,))
            connection.execute(
                "INSERT INTO diagnosis_history(asset_id, recorded_at, kind, result_json) VALUES (?, ?, ?, ?)",
                (asset_id, now, kind, encoded),
            )


def read_history(*, asset_id: str, limit: int = 100) -> list[dict[str, Any]]:
    now = time.time()
    cutoff = now - retention_days() * 24 * 60 * 60
    with closing(_connect()) as connection:
        with connection:
            connection.execute("DELETE FROM diagnosis_history WHERE recorded_at < ?", (cutoff,))
            rows = connection.execute(
                "SELECT id, asset_id, recorded_at, kind, result_json FROM diagnosis_history "
                "WHERE asset_id = ? AND recorded_at >= ? ORDER BY recorded_at DESC, id DESC LIMIT ?",
                (asset_id, cutoff, limit),
            ).fetchall()
    return [
        {
            "id": row["id"],
            "asset_id": row["asset_id"],
            "recorded_at": row["recorded_at"],
            "kind": row["kind"],
            "result": json.loads(row["result_json"]),
        }
        for row in rows
    ]
