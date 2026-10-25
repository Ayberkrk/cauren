from __future__ import annotations

import os
import sqlite3
import stat
import time

import pytest

from api.history import persist_history, read_history


def test_history_prunes_records_older_than_configured_retention(tmp_path, monkeypatch):
    monkeypatch.setenv("CAUREN_HISTORY_DB", str(tmp_path / "history.sqlite3"))
    monkeypatch.setenv("CAUREN_HISTORY_RETENTION_DAYS", "1")
    now = time.time()

    persist_history(asset_id="a1", kind="diagnose", result={"risk_score": 0.1}, recorded_at=now - 2 * 86400)
    persist_history(asset_id="a1", kind="diagnose", result={"risk_score": 0.8}, recorded_at=now)

    records = read_history(asset_id="a1", limit=10)
    assert len(records) == 1
    assert records[0]["result"] == {"risk_score": 0.8}


def test_history_rejects_non_positive_retention(tmp_path, monkeypatch):
    monkeypatch.setenv("CAUREN_HISTORY_DB", str(tmp_path / "history.sqlite3"))
    monkeypatch.setenv("CAUREN_HISTORY_RETENTION_DAYS", "0")

    try:
        read_history(asset_id="a1")
    except ValueError as exc:
        assert "positive integer" in str(exc)
    else:
        raise AssertionError("invalid retention configuration should be rejected")


@pytest.mark.skipif(os.name != "posix", reason="POSIX file mode bits are not portable")
def test_new_history_path_is_owner_only(tmp_path, monkeypatch):
    db_path = tmp_path / "new-private-dir" / "history.sqlite3"
    monkeypatch.setenv("CAUREN_HISTORY_DB", str(db_path))

    persist_history(asset_id="a1", kind="diagnose", result={"risk_score": 0.5})

    assert stat.S_IMODE(db_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX file mode bits are not portable")
def test_existing_volume_directory_mode_is_preserved(tmp_path, monkeypatch):
    volume = tmp_path / "existing-volume"
    volume.mkdir(mode=0o750)
    os.chmod(volume, 0o750)
    db_path = volume / "history.sqlite3"
    with sqlite3.connect(db_path):
        pass
    os.chmod(db_path, 0o640)
    monkeypatch.setenv("CAUREN_HISTORY_DB", str(db_path))

    persist_history(asset_id="a1", kind="diagnose", result={"risk_score": 0.5})

    assert stat.S_IMODE(volume.stat().st_mode) == 0o750
    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
    assert read_history(asset_id="a1")[0]["result"] == {"risk_score": 0.5}


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink and mode semantics are required")
def test_history_rejects_symlink_without_mutating_target(tmp_path, monkeypatch):
    target = tmp_path / "unrelated.sqlite3"
    with sqlite3.connect(target) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
    os.chmod(target, 0o644)
    symlink = tmp_path / "history.sqlite3"
    symlink.symlink_to(target)
    monkeypatch.setenv("CAUREN_HISTORY_DB", str(symlink))

    with pytest.raises(ValueError, match="symbolic link"):
        persist_history(asset_id="a1", kind="diagnose", result={"risk_score": 0.5})

    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    with sqlite3.connect(target) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {"unrelated"}


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory mode bits are not portable")
def test_history_rejects_group_or_world_writable_parent(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o777)
    os.chmod(shared, 0o777)
    monkeypatch.setenv("CAUREN_HISTORY_DB", str(shared / "history.sqlite3"))

    with pytest.raises(ValueError, match="must not be group- or world-writable"):
        persist_history(asset_id="a1", kind="diagnose", result={"risk_score": 0.5})
