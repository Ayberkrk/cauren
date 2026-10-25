"""Small local history store for single-operator Cauren deployments."""

from __future__ import annotations

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
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Another local user who can write to the parent directory could swap the
    # database for a link to their own file. Refusing shared parents is enough:
    # in an owner-only directory, only the owner can create such links.
    if os.name == "posix" and path.parent.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("CAUREN_HISTORY_DB parent directory must not be group- or world-writable")

    if os.name == "posix":
        # Create owner-only, and tighten an existing file before SQLite opens it.
        os.close(os.open(path, os.O_CREAT | os.O_WRONLY, 0o600))
        try:
            os.chmod(path, 0o600)
        except OSError as exc:
            logger.warning("could not restrict history file permissions: %s", exc.__class__.__name__)
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
    # Summary statistics only (no reference readings): one row per asset
    # feature, replaced whenever the asset's baseline is refitted.
    connection.execute(
        """CREATE TABLE IF NOT EXISTS asset_baselines (
            asset_id TEXT NOT NULL,
            feature TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            center REAL NOT NULL,
            scale REAL NOT NULL,
            n INTEGER NOT NULL,
            expected_variation INTEGER NOT NULL,
            unit TEXT NOT NULL,
            fitted_at REAL NOT NULL,
            PRIMARY KEY (asset_id, feature)
        )"""
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


def store_baseline(
    *,
    asset_id: str,
    agent_id: str,
    baselines: dict[str, dict[str, Any]],
    fitted_at: float | None = None,
) -> None:
    """Replace an asset's stored baseline. Baselines are configuration, so
    they are not subject to the history retention window."""
    now = float(fitted_at if fitted_at is not None else time.time())
    rows = [
        (
            asset_id,
            feature,
            agent_id,
            float(item["center"]),
            float(item["scale"]),
            int(item["n"]),
            1 if item.get("expected_variation", True) else 0,
            str(item.get("unit") or ""),
            now,
        )
        for feature, item in baselines.items()
    ]
    with closing(_connect()) as connection:
        with connection:
            connection.execute("DELETE FROM asset_baselines WHERE asset_id = ?", (asset_id,))
            connection.executemany(
                "INSERT INTO asset_baselines(asset_id, feature, agent_id, center, scale, n, "
                "expected_variation, unit, fitted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )


def read_baseline(*, asset_id: str) -> dict[str, Any] | None:
    with closing(_connect()) as connection:
        rows = connection.execute(
            "SELECT feature, agent_id, center, scale, n, expected_variation, unit, fitted_at "
            "FROM asset_baselines WHERE asset_id = ? ORDER BY feature",
            (asset_id,),
        ).fetchall()
    if not rows:
        return None
    return {
        "asset_id": asset_id,
        "agent_id": rows[0]["agent_id"],
        "fitted_at": max(row["fitted_at"] for row in rows),
        "features": {
            row["feature"]: {
                "center": row["center"],
                "scale": row["scale"],
                "n": row["n"],
                "expected_variation": bool(row["expected_variation"]),
                "unit": row["unit"],
            }
            for row in rows
        },
    }
