from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.memory_service import SharedMemoryService


def _sensor_row(value: float) -> list[float]:
    return [value, value, value, value, value, value, value]


def _ingest(service: SharedMemoryService, asset_id: str, dt_local: datetime, value: float, *, include: bool = True) -> None:
    service.ingest_sensor_window(
        asset_id=asset_id,
        timestamp=dt_local.timestamp(),
        sensor_matrix=[_sensor_row(value)],
        include_in_learning=include,
    )


def test_daypart_boundaries_follow_spec(tmp_path) -> None:
    service = SharedMemoryService(state_path=tmp_path / "memory_state.json", timezone_name="Europe/Istanbul")
    tz = service.timezone
    assert service._daypart(datetime(2026, 3, 15, 6, 0, tzinfo=tz)) == "morning"
    assert service._daypart(datetime(2026, 3, 15, 11, 0, tzinfo=tz)) == "morning"
    assert service._daypart(datetime(2026, 3, 15, 11, 1, tzinfo=tz)) == "midday"
    assert service._daypart(datetime(2026, 3, 15, 19, 0, tzinfo=tz)) == "midday"
    assert service._daypart(datetime(2026, 3, 15, 19, 1, tzinfo=tz)) == "night"
    assert service._daypart(datetime(2026, 3, 15, 5, 59, tzinfo=tz)) == "night"


def test_anomaly_windows_are_not_learned(tmp_path) -> None:
    service = SharedMemoryService(state_path=tmp_path / "memory_state.json", timezone_name="Europe/Istanbul")
    tz = service.timezone
    base = datetime(2026, 3, 1, 8, 0, tzinfo=tz)
    _ingest(service, "line_skip", base, 112.0, include=False)
    _ingest(service, "line_skip", base + timedelta(hours=1), 112.0, include=False)
    snap = service.get_norm_snapshot("line_skip")
    morning = snap["dimensions"]["electronic_temp"]["morning"]
    assert morning["hourly_window_means"] == []
    assert morning["normal_lower_bound"] is None
    assert morning["normal_upper_bound"] is None


def test_confidence_uses_recent8of10_and_7day_window(tmp_path) -> None:
    service = SharedMemoryService(state_path=tmp_path / "memory_state.json", timezone_name="Europe/Istanbul")
    tz = service.timezone
    base = datetime(2026, 3, 1, 8, 0, tzinfo=tz)

    # Build >= 7 day coverage with dominant 110-range readings.
    for day in range(10):
        _ingest(service, "line_conf", base + timedelta(days=day), 112.0)
        _ingest(service, "line_conf", base + timedelta(days=day, hours=1), 112.0)

    # Add two out-of-range finalized observations in recent window.
    _ingest(service, "line_conf", base + timedelta(days=10), 135.0)
    _ingest(service, "line_conf", base + timedelta(days=10, hours=1), 135.0)
    _ingest(service, "line_conf", base + timedelta(days=10, hours=2), 135.0)

    snap = service.get_norm_snapshot("line_conf")
    morning = snap["dimensions"]["electronic_temp"]["morning"]
    assert morning["days_covered"] >= 7
    assert morning["recent10_total_count"] == 10
    assert morning["recent10_in_range_count"] >= 8
    assert morning["high_confidence"] is True
    assert morning["ready"] is True
    assert 0.0 <= morning["confidence_score"] <= 1.0


def test_sliding_window_keeps_last_7_days_only(tmp_path) -> None:
    service = SharedMemoryService(state_path=tmp_path / "memory_state.json", timezone_name="Europe/Istanbul")
    tz = service.timezone
    start = datetime(2026, 1, 1, 8, 0, tzinfo=tz)

    for hour in range((8 * 24) + 2):
        ts = start + timedelta(hours=hour)
        value = 110.0 if hour % 3 else 120.0
        _ingest(service, "line_window", ts, value)

    snap = service.get_norm_snapshot("line_window")
    morning = snap["dimensions"]["electronic_temp"]["morning"]
    # Window means are derived from hourly records and should never exceed configured window_size.
    assert len(morning["hourly_window_means"]) <= 10
    # Service prunes older records; if pruning fails this would stay near 8 days.
    assert morning["days_covered"] <= 7


def test_calibration_memory_tracks_drift_summary(tmp_path) -> None:
    service = SharedMemoryService(state_path=tmp_path / "memory_state.json", timezone_name="Europe/Istanbul")
    correction = np.asarray(
        [
            [0.10, 0.20, 0.01, 0.02, 0.03, 0.04, 0.05],
            [0.12, 0.18, 0.02, 0.03, 0.01, 0.05, 0.06],
        ],
        dtype=np.float32,
    )
    summary = service.update_calibration_memory(
        asset_id="line_calib",
        timestamp=datetime(2026, 3, 15, 12, 0).timestamp(),
        correction_matrix=correction.tolist(),
    )
    assert summary["memory_status"] == "available"
    assert "electronic_temp" in summary["drift_summary"]
    assert summary["drift_summary"]["electronic_temp"]["last_mean_abs_drift"] is not None
