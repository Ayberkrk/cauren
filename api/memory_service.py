from __future__ import annotations

import json
import math
import threading
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core_dimensions import CANONICAL_DIMENSIONS, CORE_DIMENSIONS, dimension_class

MEMORY_DIMENSIONS: List[str] = list(CORE_DIMENSIONS)
CONTEXT_DIMENSIONS: List[str] = list(CANONICAL_DIMENSIONS)

DAYPARTS: List[str] = ["morning", "midday", "night"]


def _coerce_ts(value: float | str | int | None) -> datetime:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=ZoneInfo("UTC"))
    if isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
        return parsed.astimezone(ZoneInfo("UTC"))
    return datetime.now(tz=ZoneInfo("UTC"))


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(ZoneInfo("UTC")).isoformat()


class SharedMemoryService:
    """Independent cross-model memory layer for norm learning and drift history."""

    def __init__(
        self,
        *,
        state_path: str | Path,
        timezone_name: str = "Europe/Istanbul",
        dimensions: List[str] | None = None,
    ) -> None:
        self.state_path = Path(state_path)
        self.timezone_name = timezone_name
        self.timezone = ZoneInfo(timezone_name)
        self.dimensions = list(dimensions or MEMORY_DIMENSIONS)
        self.context_dimensions = list(CONTEXT_DIMENSIONS)
        self.dayparts = list(DAYPARTS)
        self.history_days = 7
        self.window_size = 10
        self._lock = threading.RLock()
        self._state = self._load_state()

    def _base_daypart_state(self) -> Dict[str, Any]:
        return {
            "hourly_records": [],
            "normal_lower_bound": None,
            "normal_upper_bound": None,
            "confidence_score": 0.0,
            "updated_at": None,
            "recent10_in_range_count": 0,
            "recent10_total_count": 0,
            "days_covered": 0,
            "ready": False,
            "high_confidence": False,
        }

    def _base_dimension_state(self) -> Dict[str, Any]:
        return {
            daypart: self._base_daypart_state()
            for daypart in self.dayparts
        }

    def _base_asset_state(self) -> Dict[str, Any]:
        return {
            "open_hour_buckets": {},
            "dimensions": {
                dim: self._base_dimension_state()
                for dim in self.dimensions
            },
            "context_dimensions": {
                dim: {
                    "last_value": None,
                    "hourly_records": [],
                    "observed_min": None,
                    "observed_max": None,
                    "recent_presence_total": 0,
                    "recent_presence_hits": 0,
                    "updated_at": None,
                }
                for dim in self.context_dimensions
            },
            "calibration_memory": {
                "drift_history": {dim: [] for dim in self.dimensions},
                "updated_at": None,
            },
        }

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {
                "version": 1,
                "timezone": self.timezone_name,
                "assets": {},
                "updated_at": _iso_utc(datetime.now(tz=ZoneInfo("UTC"))),
            }
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("invalid_state")
            payload.setdefault("assets", {})
            payload.setdefault("timezone", self.timezone_name)
            payload.setdefault("updated_at", _iso_utc(datetime.now(tz=ZoneInfo("UTC"))))
            return payload
        except Exception:
            backup = self.state_path.with_suffix(self.state_path.suffix + ".corrupt")
            try:
                self.state_path.replace(backup)
            except OSError:
                pass
            return {
                "version": 1,
                "timezone": self.timezone_name,
                "assets": {},
                "updated_at": _iso_utc(datetime.now(tz=ZoneInfo("UTC"))),
            }

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state["updated_at"] = _iso_utc(datetime.now(tz=ZoneInfo("UTC")))
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, ensure_ascii=True, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def _asset(self, asset_id: str) -> Dict[str, Any]:
        assets = self._state.setdefault("assets", {})
        if asset_id not in assets:
            assets[asset_id] = self._base_asset_state()
        asset = assets[asset_id]
        asset.setdefault("open_hour_buckets", {})
        asset.setdefault("dimensions", {})
        asset.setdefault("context_dimensions", {})
        for dim in self.dimensions:
            asset["dimensions"].setdefault(dim, self._base_dimension_state())
            for daypart in self.dayparts:
                asset["dimensions"][dim].setdefault(daypart, self._base_daypart_state())
        for dim in self.context_dimensions:
            asset["context_dimensions"].setdefault(
                dim,
                {
                    "last_value": None,
                    "hourly_records": [],
                    "observed_min": None,
                    "observed_max": None,
                    "recent_presence_total": 0,
                    "recent_presence_hits": 0,
                    "updated_at": None,
                },
            )
        asset.setdefault(
            "calibration_memory",
            {
                "drift_history": {dim: [] for dim in self.dimensions},
                "updated_at": None,
            },
        )
        drift_history = asset["calibration_memory"].setdefault("drift_history", {})
        for dim in self.dimensions:
            drift_history.setdefault(dim, [])
        return asset

    def _daypart(self, dt_local: datetime) -> str:
        minute_of_day = dt_local.hour * 60 + dt_local.minute
        if 6 * 60 <= minute_of_day <= 11 * 60:
            return "morning"
        if (11 * 60 + 1) <= minute_of_day <= 19 * 60:
            return "midday"
        return "night"

    def _finalize_bucket(
        self,
        *,
        asset: Dict[str, Any],
        dimension: str,
        bucket: Dict[str, Any],
        ref_ts_utc: datetime,
    ) -> None:
        count = float(bucket.get("count", 0.0))
        total = float(bucket.get("sum", 0.0))
        if count <= 0.0:
            return
        avg = total / count
        daypart = str(bucket.get("daypart", "night"))
        rec = {
            "hour_start": str(bucket.get("hour_start")),
            "mean": float(avg),
        }
        records = asset["dimensions"][dimension][daypart].setdefault("hourly_records", [])
        records.append(rec)
        records.sort(key=lambda row: str(row.get("hour_start", "")))
        cutoff = ref_ts_utc.astimezone(self.timezone) - timedelta(days=self.history_days)
        kept = []
        for row in records:
            try:
                row_dt = _coerce_ts(row.get("hour_start")).astimezone(self.timezone)
            except Exception:
                continue
            if row_dt > cutoff:
                kept.append(row)
        asset["dimensions"][dimension][daypart]["hourly_records"] = kept[-(self.history_days * 24 + 24):]

    def _recompute_daypart_stats(self, daypart_state: Dict[str, Any]) -> None:
        records = daypart_state.get("hourly_records", [])
        means = [float(row.get("mean")) for row in records if isinstance(row, dict)]
        if not means:
            daypart_state.update(
                {
                    "normal_lower_bound": None,
                    "normal_upper_bound": None,
                    "confidence_score": 0.0,
                    "recent10_in_range_count": 0,
                    "recent10_total_count": 0,
                    "days_covered": 0,
                    "ready": False,
                    "high_confidence": False,
                    "updated_at": daypart_state.get("updated_at"),
                }
            )
            return

        values = np.asarray(means, dtype=np.float64)
        if np.allclose(values.min(), values.max()):
            eps = max(abs(float(values.min())) * 0.01, 1e-3)
            lower = float(values.min() - eps)
            upper = float(values.max() + eps)
            dominant_count = len(values)
        else:
            bin_count = min(10, max(3, int(math.sqrt(len(values))) + 1))
            hist, edges = np.histogram(values, bins=bin_count)
            idx = int(np.argmax(hist))
            lower = float(edges[idx])
            upper = float(edges[idx + 1])
            dominant_count = int(hist[idx])

        recent = means[-self.window_size :]
        in_range = sum(1 for val in recent if lower <= float(val) <= upper)
        recent_total = len(recent)
        recent_ratio = (in_range / recent_total) if recent_total else 0.0
        window_ratio = (dominant_count / len(means)) if means else 0.0
        confidence_score = round(0.6 * recent_ratio + 0.4 * window_ratio, 4)
        days = {
            _coerce_ts(row.get("hour_start")).astimezone(self.timezone).date().isoformat()
            for row in records
            if isinstance(row, dict) and row.get("hour_start")
        }
        days_covered = min(self.history_days, len(days))
        ready = days_covered >= 7 and recent_total >= self.window_size
        daypart_state.update(
            {
                "normal_lower_bound": lower,
                "normal_upper_bound": upper,
                "confidence_score": confidence_score,
                "recent10_in_range_count": int(in_range),
                "recent10_total_count": int(recent_total),
                "days_covered": int(days_covered),
                "ready": bool(ready),
                "high_confidence": bool(recent_total > 0 and in_range >= 8),
                "updated_at": _iso_utc(datetime.now(tz=ZoneInfo("UTC"))),
            }
        )

    def _update_context_dimension(
        self,
        *,
        asset: Dict[str, Any],
        dimension: str,
        values: List[float],
        ts_utc: datetime,
    ) -> None:
        if not values:
            ctx = asset["context_dimensions"][dimension]
            ctx["recent_presence_total"] = int(ctx.get("recent_presence_total", 0)) + 1
            return
        ctx = asset["context_dimensions"][dimension]
        finite_values = [float(v) for v in values if np.isfinite(v)]
        if not finite_values:
            ctx["recent_presence_total"] = int(ctx.get("recent_presence_total", 0)) + 1
            return
        mean_val = float(np.mean(np.asarray(finite_values, dtype=np.float64)))
        hour_start = ts_utc.astimezone(self.timezone).replace(minute=0, second=0, microsecond=0).isoformat()
        rows = ctx.setdefault("hourly_records", [])
        rows.append({"hour_start": hour_start, "mean": mean_val})
        ctx["hourly_records"] = rows[-(self.history_days * 24 + 24):]
        ctx["last_value"] = float(finite_values[-1])
        cur_min = ctx.get("observed_min")
        cur_max = ctx.get("observed_max")
        ctx["observed_min"] = min(float(cur_min), float(np.min(finite_values))) if cur_min is not None else float(np.min(finite_values))
        ctx["observed_max"] = max(float(cur_max), float(np.max(finite_values))) if cur_max is not None else float(np.max(finite_values))
        ctx["recent_presence_total"] = int(ctx.get("recent_presence_total", 0)) + 1
        ctx["recent_presence_hits"] = int(ctx.get("recent_presence_hits", 0)) + 1
        ctx["updated_at"] = _iso_utc(datetime.now(tz=ZoneInfo("UTC")))

    def ingest_sensor_window(
        self,
        *,
        asset_id: str,
        timestamp: float | str | int,
        sensor_matrix: List[List[float]],
        dimension_series: Dict[str, List[float]] | None = None,
        include_in_learning: bool = True,
    ) -> Dict[str, Any]:
        ts_utc = _coerce_ts(timestamp)
        ts_local = ts_utc.astimezone(self.timezone)
        hour_start = ts_local.replace(minute=0, second=0, microsecond=0).isoformat()
        daypart = self._daypart(ts_local)
        series_map: Dict[str, List[float]] = {}
        explicit_series_payload = isinstance(dimension_series, dict)
        if isinstance(dimension_series, dict):
            for dim, values in dimension_series.items():
                if dim in self.context_dimensions and isinstance(values, list):
                    series_map[dim] = [float(v) for v in values if np.isfinite(float(v))]
        if sensor_matrix and not explicit_series_payload:
            for idx, dim in enumerate(self.dimensions):
                if dim in series_map:
                    continue
                values = [float(row[idx]) for row in sensor_matrix if isinstance(row, list) and idx < len(row)]
                if values:
                    series_map[dim] = values

        if not include_in_learning and not series_map:
            return {"learning_applied": False}

        with self._lock:
            asset = self._asset(asset_id)
            buckets = asset.setdefault("open_hour_buckets", {})
            if include_in_learning:
                for dim in self.dimensions:
                    values = list(series_map.get(dim, []))
                    if not values:
                        continue
                    bucket = buckets.get(dim)
                    if bucket and bucket.get("hour_start") != hour_start:
                        self._finalize_bucket(asset=asset, dimension=dim, bucket=bucket, ref_ts_utc=ts_utc)
                        bucket = None
                    if not bucket:
                        bucket = {
                            "hour_start": hour_start,
                            "daypart": daypart,
                            "sum": 0.0,
                            "count": 0.0,
                        }
                        buckets[dim] = bucket
                    bucket["sum"] = float(bucket.get("sum", 0.0)) + float(sum(values))
                    bucket["count"] = float(bucket.get("count", 0.0)) + float(len(values))

            for dim in self.context_dimensions:
                self._update_context_dimension(
                    asset=asset,
                    dimension=dim,
                    values=list(series_map.get(dim, [])),
                    ts_utc=ts_utc,
                )

            if include_in_learning:
                for dim in self.dimensions:
                    for dp in self.dayparts:
                        self._recompute_daypart_stats(asset["dimensions"][dim][dp])
            self._save_state()
        return {"learning_applied": bool(include_in_learning)}

    def update_calibration_memory(
        self,
        *,
        asset_id: str,
        timestamp: float | str | int,
        correction_matrix: List[List[float]],
    ) -> Dict[str, Any]:
        if not correction_matrix:
            return self.get_calibration_memory(asset_id)
        ts = _coerce_ts(timestamp)
        with self._lock:
            asset = self._asset(asset_id)
            calib = asset["calibration_memory"]
            drift_history = calib["drift_history"]
            for idx, dim in enumerate(self.dimensions):
                values = [abs(float(row[idx])) for row in correction_matrix if isinstance(row, list) and idx < len(row)]
                if not values:
                    continue
                rec = {
                    "timestamp": _iso_utc(ts),
                    "mean_abs_drift": float(np.mean(np.asarray(values, dtype=np.float64))),
                }
                drift_history[dim].append(rec)
                drift_history[dim] = drift_history[dim][-168:]
            calib["updated_at"] = _iso_utc(datetime.now(tz=ZoneInfo("UTC")))
            self._save_state()
        return self.get_calibration_memory(asset_id)

    def _build_daypart_view(self, state: Dict[str, Any]) -> Dict[str, Any]:
        records = state.get("hourly_records", [])
        hourly_window_means = [float(row.get("mean")) for row in records[-self.window_size :]]
        return {
            "hourly_window_means": hourly_window_means,
            "normal_lower_bound": state.get("normal_lower_bound"),
            "normal_upper_bound": state.get("normal_upper_bound"),
            "confidence_score": float(state.get("confidence_score", 0.0)),
            "updated_at": state.get("updated_at"),
            "recent10_in_range_count": int(state.get("recent10_in_range_count", 0)),
            "recent10_total_count": int(state.get("recent10_total_count", 0)),
            "days_covered": int(state.get("days_covered", 0)),
            "ready": bool(state.get("ready", False)),
            "high_confidence": bool(state.get("high_confidence", False)),
        }

    def get_norm_snapshot(self, asset_id: str) -> Dict[str, Any]:
        with self._lock:
            asset = self._asset(asset_id)
            dimensions_view: Dict[str, Any] = {}
            for dim in self.dimensions:
                dim_state = asset["dimensions"][dim]
                dimensions_view[dim] = {
                    daypart: self._build_daypart_view(dim_state[daypart])
                    for daypart in self.dayparts
                }
            return {
                "asset_id": asset_id,
                "timezone": self.timezone_name,
                "generated_at": _iso_utc(datetime.now(tz=ZoneInfo("UTC"))),
                "memory_status": "available",
                "norm_learning_dimensions": list(self.dimensions),
                "context_dimensions": list(self.context_dimensions),
                "dimensions": dimensions_view,
            }

    def get_context(
        self,
        *,
        asset_id: str,
        timestamp: float | str | int | None,
        current_values: Dict[str, float] | None,
    ) -> Dict[str, Any]:
        snap = self.get_norm_snapshot(asset_id)
        ts = _coerce_ts(timestamp).astimezone(self.timezone)
        daypart = self._daypart(ts)
        contexts: List[Dict[str, Any]] = []
        current_values = current_values or {}
        for dim in self.dimensions:
            stats = snap["dimensions"][dim][daypart]
            val = current_values.get(dim)
            low = stats.get("normal_lower_bound")
            high = stats.get("normal_upper_bound")
            in_range = None
            if val is not None and low is not None and high is not None:
                in_range = bool(float(low) <= float(val) <= float(high))
            contexts.append(
                {
                    "dimension": dim,
                    "daypart": daypart,
                    "current_value": float(val) if val is not None else None,
                    "normal_lower_bound": low,
                    "normal_upper_bound": high,
                    "in_normal_range": in_range,
                    "confidence_score": float(stats.get("confidence_score", 0.0)),
                    "recent10_in_range_count": int(stats.get("recent10_in_range_count", 0)),
                    "recent10_total_count": int(stats.get("recent10_total_count", 0)),
                    "days_covered": int(stats.get("days_covered", 0)),
                    "ready": bool(stats.get("ready", False)),
                    "high_confidence": bool(stats.get("high_confidence", False)),
                    "updated_at": stats.get("updated_at"),
                }
            )
        with self._lock:
            asset = self._asset(asset_id)
            for dim in self.context_dimensions:
                if dim in self.dimensions:
                    continue
                ctx = asset["context_dimensions"][dim]
                presence_total = int(ctx.get("recent_presence_total", 0))
                presence_hits = int(ctx.get("recent_presence_hits", 0))
                current_val = current_values.get(dim)
                contexts.append(
                    {
                        "dimension": dim,
                        "dimension_class": dimension_class(dim) or "aux",
                        "daypart": daypart,
                        "current_value": float(current_val) if current_val is not None else None,
                        "normal_lower_bound": None,
                        "normal_upper_bound": None,
                        "in_normal_range": None,
                        "confidence_score": 0.0,
                        "recent10_in_range_count": 0,
                        "recent10_total_count": 0,
                        "days_covered": 0,
                        "ready": presence_hits > 0,
                        "high_confidence": presence_hits >= 3,
                        "updated_at": ctx.get("updated_at"),
                        "last_value": ctx.get("last_value"),
                        "observed_min": ctx.get("observed_min"),
                        "observed_max": ctx.get("observed_max"),
                        "recent_presence_ratio": (
                            float(presence_hits) / float(presence_total)
                            if presence_total > 0
                            else 0.0
                        ),
                    }
                )
        for row in contexts:
            row.setdefault("dimension_class", dimension_class(str(row.get("dimension") or "")) or "core")
        return {
            "asset_id": asset_id,
            "timezone": self.timezone_name,
            "generated_at": _iso_utc(datetime.now(tz=ZoneInfo("UTC"))),
            "memory_status": "available",
            "daypart": daypart,
            "contexts": contexts,
        }

    def get_calibration_memory(self, asset_id: str) -> Dict[str, Any]:
        with self._lock:
            asset = self._asset(asset_id)
            calib = asset["calibration_memory"]
            drift_history = calib.get("drift_history", {})
            summary: Dict[str, Any] = {}
            for dim in self.dimensions:
                rows = drift_history.get(dim, [])
                recent = rows[-self.window_size :]
                last_value = recent[-1]["mean_abs_drift"] if recent else None
                trend = None
                if len(recent) >= 2:
                    trend = float(recent[-1]["mean_abs_drift"]) - float(recent[0]["mean_abs_drift"])
                summary[dim] = {
                    "last_mean_abs_drift": last_value,
                    "recent_points": len(recent),
                    "drift_trend": trend,
                    "updated_at": rows[-1]["timestamp"] if rows else None,
                }
            return {
                "asset_id": asset_id,
                "memory_status": "available",
                "updated_at": calib.get("updated_at"),
                "drift_summary": summary,
            }
