from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core_dimensions import AUX_DIMENSIONS, CANONICAL_DIMENSIONS, CORE_DIMENSIONS


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json_file(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def _decode_json(value: Any, *, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _day_bucket(timestamp_iso: str) -> str:
    try:
        parsed = datetime.fromisoformat(str(timestamp_iso).replace("Z", "+00:00"))
    except Exception:
        parsed = datetime.now(UTC)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    parsed = parsed.astimezone(UTC)
    hour = parsed.hour
    if hour < 8:
        return "night"
    if hour < 16:
        return "day"
    return "evening"


def _date_bucket(timestamp_iso: str) -> str:
    try:
        parsed = datetime.fromisoformat(str(timestamp_iso).replace("Z", "+00:00"))
    except Exception:
        parsed = datetime.now(UTC)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).date().isoformat()


def _slug(text: str, *, default: str = "") -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text or "").strip()).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or default


DEFAULT_PROFILE_REGISTRY = {
    "profiles": [
        {
            "asset_class": "generic",
            "critical_dimensions": list(CORE_DIMENSIONS),
            "preferred_dimensions": list(CANONICAL_DIMENSIONS),
            "dimension_norm_policy": "asset",
            "fallback_to_asset_class": True,
            "fallback_to_global": True,
        },
        {
            "asset_class": "chiller",
            "critical_dimensions": [
                "chiller_supply_temp_c",
                "chiller_return_temp_c",
                "cooling_water_flow_l_min",
                            "humidity_rh_pct",
                "material_temp",
                "electronic_temp",
            ],
            "preferred_dimensions": list(CANONICAL_DIMENSIONS),
            "dimension_norm_policy": "asset",
            "fallback_to_asset_class": True,
            "fallback_to_global": True,
        },
        {
            "asset_class": "tank",
            "critical_dimensions": [
                "internal_pressure",
                "external_pressure",
                "material_temp",
                "electronic_temp",
                "strain",
                "radiation",
            ],
            "preferred_dimensions": list(CANONICAL_DIMENSIONS),
            "dimension_norm_policy": "asset",
            "fallback_to_asset_class": True,
            "fallback_to_global": True,
        },
        {
            "asset_class": "power",
            "critical_dimensions": [
                "mains_voltage_l1_v",
                "mains_voltage_l2_v",
                "mains_voltage_l3_v",
                "line_current_l1_a",
                "line_current_l2_a",
                "line_current_l3_a",
                "ground_line_resistance_ohm",
                "electronic_temp",
            ],
            "preferred_dimensions": list(CANONICAL_DIMENSIONS),
            "dimension_norm_policy": "asset",
            "fallback_to_asset_class": True,
            "fallback_to_global": True,
        },
        {
            "asset_class": "habitat",
            "critical_dimensions": [
                "co2_ppm",
                "o2_pct",
                "cabin_pressure_total",
                "airflow_velocity_m_s",
                "trace_contaminant_index",
                "combustion_co_ppm",
                "humidity_rh_pct",
            ],
            "preferred_dimensions": list(CANONICAL_DIMENSIONS),
            "dimension_norm_policy": "asset",
            "fallback_to_asset_class": True,
            "fallback_to_global": True,
        },
        {
            "asset_class": "water_recovery",
            "critical_dimensions": [
                "potable_water_conductivity_us_cm",
                "potable_water_toc_ppb",
                "potable_water_microbe_cfu_ml",
                "surface_microbe_cfu_cm2",
                "water_storage_level_pct",
                "waste_storage_fill_pct",
            ],
            "preferred_dimensions": list(CANONICAL_DIMENSIONS),
            "dimension_norm_policy": "asset",
            "fallback_to_asset_class": True,
            "fallback_to_global": True,
        },
    ]
}


@dataclass(frozen=True)
class AssetNormSummary:
    asset_id: str
    asset_class: str
    site_id: str
    asset_norm_scope: str
    asset_norm_confidence: float
    norm_source: str
    asset_critical_dimensions: list[str]
    dimension_importance_by_asset: dict[str, str]
    candidate_dimensions: list[str]
    observing_dimensions: list[str]
    provisional_dimensions: list[str]
    approved_dimensions: list[str]
    dimension_lifecycle_status: dict[str, str]


class AssetNormStore:
    def __init__(
        self,
        *,
        sqlite_path: str | Path,
        profile_registry_path: str | Path,
        min_observation_days: int = 7,
        min_sample_count: int = 500,
    ):
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile_registry_path = Path(profile_registry_path)
        self.profile_registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.profile_registry_path.exists():
            self.profile_registry_path.write_text(
                json.dumps(DEFAULT_PROFILE_REGISTRY, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
        self.min_observation_days = max(1, int(min_observation_days))
        self.min_sample_count = max(10, int(min_sample_count))
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS asset_dimension_norms (
                    asset_id TEXT NOT NULL,
                    asset_class TEXT NOT NULL,
                    site_id TEXT NOT NULL,
                    dimension_name TEXT NOT NULL,
                    unit TEXT NOT NULL,
                    baseline_range_json TEXT NOT NULL,
                    learned_range_json TEXT NOT NULL,
                    volatility_band REAL NOT NULL,
                    seasonality_hint TEXT NOT NULL,
                    sample_count INTEGER NOT NULL,
                    observed_days INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    min_seen REAL,
                    max_seen REAL,
                    mean_seen REAL,
                    m2_seen REAL,
                    dayparts_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (asset_id, dimension_name)
                );
                CREATE TABLE IF NOT EXISTS asset_class_dimension_norms (
                    asset_class TEXT NOT NULL,
                    site_id TEXT NOT NULL,
                    dimension_name TEXT NOT NULL,
                    unit TEXT NOT NULL,
                    sample_count INTEGER NOT NULL,
                    min_seen REAL,
                    max_seen REAL,
                    mean_seen REAL,
                    m2_seen REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (asset_class, site_id, dimension_name)
                );
                CREATE TABLE IF NOT EXISTS dimension_lifecycle (
                    candidate_key TEXT PRIMARY KEY,
                    asset_id TEXT NOT NULL,
                    asset_class TEXT NOT NULL,
                    site_id TEXT NOT NULL,
                    dimension_name TEXT NOT NULL,
                    unit TEXT NOT NULL,
                    status TEXT NOT NULL,
                    sample_count INTEGER NOT NULL,
                    observed_days INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    value_min REAL,
                    value_max REAL,
                    dayparts_json TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    review_required INTEGER NOT NULL,
                    promoted_to_registry INTEGER NOT NULL
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _profiles(self) -> dict[str, dict[str, Any]]:
        payload = _load_json_file(self.profile_registry_path, default=DEFAULT_PROFILE_REGISTRY)
        profiles = payload.get("profiles") if isinstance(payload, dict) else None
        out: dict[str, dict[str, Any]] = {}
        if isinstance(profiles, list):
            for row in profiles:
                if not isinstance(row, dict):
                    continue
                key = str(row.get("asset_class") or "").strip().lower()
                if key:
                    out[key] = dict(row)
        if "generic" not in out:
            out["generic"] = dict(DEFAULT_PROFILE_REGISTRY["profiles"][0])
        return out

    def infer_asset_class(
        self,
        *,
        asset_id: str,
        site_id: str,
        line_id: str,
        machine_id: str,
        observed_dimensions: list[str],
        candidate_dimensions: list[str],
    ) -> str:
        probe = " ".join([asset_id, site_id, line_id, machine_id]).lower()
        dims = set(observed_dimensions) | set(candidate_dimensions)
        if any(token in probe for token in ("chiller", "cool", "hvac")) or {
            "chiller_supply_temp_c",
            "chiller_return_temp_c",
            "cooling_water_flow_l_min",
        } & dims:
            return "chiller"
        if any(token in probe for token in ("tank", "fluid", "pressure")) or {
            "internal_pressure",
            "external_pressure",
        } & dims:
            return "tank"
        if any(token in probe for token in ("power", "mains", "electrical")) or {
            "mains_voltage_l1_v",
            "line_current_l1_a",
            "ground_line_resistance_ohm",
        } & dims:
            return "power"
        fallback = _slug(line_id, default="")
        return fallback or "generic"

    def _profile_for(self, asset_class: str) -> dict[str, Any]:
        profiles = self._profiles()
        return profiles.get(asset_class.lower(), profiles["generic"])

    def _update_running_stats(
        self,
        *,
        sample_count: int,
        mean_seen: float,
        m2_seen: float,
        values: list[float],
    ) -> tuple[int, float, float]:
        count = int(sample_count)
        mean = float(mean_seen)
        m2 = float(m2_seen)
        for value in values:
            count += 1
            delta = value - mean
            mean += delta / float(max(1, count))
            delta2 = value - mean
            m2 += delta * delta2
        return count, mean, m2

    def _learned_range(
        self,
        *,
        mean_seen: float,
        m2_seen: float,
        sample_count: int,
        min_seen: float,
        max_seen: float,
    ) -> tuple[list[float], float, float]:
        if sample_count <= 1:
            return [float(min_seen), float(max_seen)], 0.0, 0.05
        variance = max(0.0, float(m2_seen) / float(max(1, sample_count - 1)))
        std = math.sqrt(variance)
        low = max(float(min_seen), float(mean_seen) - (2.5 * std))
        high = min(float(max_seen), float(mean_seen) + (2.5 * std))
        return [float(low), float(high)], float(std), min(1.0, sample_count / float(self.min_sample_count))

    def observe(
        self,
        *,
        asset_id: str,
        site_id: str,
        asset_class: str,
        observed_series: dict[str, list[float]],
        candidate_dimensions: list[str],
        accepted_dimensions: list[str],
        timestamp_iso: str,
    ) -> AssetNormSummary:
        now = _now_iso()
        day_bucket = _day_bucket(timestamp_iso)
        date_bucket = _date_bucket(timestamp_iso)
        observation_bucket = f"{date_bucket}:{day_bucket}"
        profile = self._profile_for(asset_class)
        critical_dimensions = [
            dim for dim in profile.get("critical_dimensions", []) if dim in accepted_dimensions or dim in candidate_dimensions
        ]
        conn = self._connect()
        try:
            cur = conn.cursor()
            dimension_lifecycle_status: dict[str, str] = {}
            observing_dimensions: list[str] = []
            provisional_dimensions: list[str] = []
            approved_dimensions: list[str] = []

            for dim_name, values in observed_series.items():
                if not values:
                    continue
                cur.execute(
                    """
                    SELECT * FROM asset_dimension_norms
                    WHERE asset_id = ? AND dimension_name = ?
                    """,
                    (asset_id, dim_name),
                )
                row = cur.fetchone()
                values_f = [float(v) for v in values]
                min_seen = min(values_f) if values_f else 0.0
                max_seen = max(values_f) if values_f else 0.0
                if row:
                    sample_count, mean_seen, m2_seen = self._update_running_stats(
                        sample_count=int(row["sample_count"]),
                        mean_seen=float(row["mean_seen"] or 0.0),
                        m2_seen=float(row["m2_seen"] or 0.0),
                        values=values_f,
                    )
                    min_seen = min(float(row["min_seen"] or min_seen), min_seen)
                    max_seen = max(float(row["max_seen"] or max_seen), max_seen)
                    dayparts = _decode_json(row["dayparts_json"], default={})
                else:
                    sample_count, mean_seen, m2_seen = self._update_running_stats(
                        sample_count=0,
                        mean_seen=0.0,
                        m2_seen=0.0,
                        values=values_f,
                    )
                    dayparts = {}
                dayparts[observation_bucket] = int(dayparts.get(observation_bucket, 0)) + 1
                observed_days = len({str(key).split(":", 1)[0] for key in dayparts.keys()})
                observed_dayparts = len(
                    {str(key).split(":", 1)[1] for key in dayparts.keys() if ":" in str(key)}
                )
                learned_range, volatility_band, sample_conf = self._learned_range(
                    mean_seen=mean_seen,
                    m2_seen=m2_seen,
                    sample_count=sample_count,
                    min_seen=min_seen,
                    max_seen=max_seen,
                )
                status = (
                    "approved"
                    if (
                        sample_count >= self.min_sample_count
                        and observed_days >= self.min_observation_days
                        and observed_dayparts >= 3
                    )
                    else "observing"
                )
                confidence = min(1.0, sample_conf * min(1.0, observed_days / float(self.min_observation_days)))
                baseline_range = learned_range if row else [float(min_seen), float(max_seen)]
                cur.execute(
                    """
                    INSERT INTO asset_dimension_norms (
                        asset_id, asset_class, site_id, dimension_name, unit, baseline_range_json, learned_range_json,
                        volatility_band, seasonality_hint, sample_count, observed_days, confidence, status,
                        min_seen, max_seen, mean_seen, m2_seen, dayparts_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(asset_id, dimension_name) DO UPDATE SET
                        asset_class=excluded.asset_class,
                        site_id=excluded.site_id,
                        baseline_range_json=excluded.baseline_range_json,
                        learned_range_json=excluded.learned_range_json,
                        volatility_band=excluded.volatility_band,
                        seasonality_hint=excluded.seasonality_hint,
                        sample_count=excluded.sample_count,
                        observed_days=excluded.observed_days,
                        confidence=excluded.confidence,
                        status=excluded.status,
                        min_seen=excluded.min_seen,
                        max_seen=excluded.max_seen,
                        mean_seen=excluded.mean_seen,
                        m2_seen=excluded.m2_seen,
                        dayparts_json=excluded.dayparts_json,
                        updated_at=excluded.updated_at
                    """,
                    (
                        asset_id,
                        asset_class,
                        site_id,
                        dim_name,
                        "",
                        json.dumps(baseline_range, ensure_ascii=True),
                        json.dumps(learned_range, ensure_ascii=True),
                        float(volatility_band),
                        day_bucket,
                        int(sample_count),
                        int(observed_days),
                        float(confidence),
                        status,
                        float(min_seen),
                        float(max_seen),
                        float(mean_seen),
                        float(m2_seen),
                        json.dumps(dayparts, ensure_ascii=True),
                        row["created_at"] if row else now,
                        now,
                    ),
                )
                dimension_lifecycle_status[dim_name] = status
                if status == "approved":
                    approved_dimensions.append(dim_name)
                else:
                    observing_dimensions.append(dim_name)

            for dim_name in candidate_dimensions:
                candidate_key = f"{asset_id}::{dim_name}"
                cur.execute("SELECT * FROM dimension_lifecycle WHERE candidate_key = ?", (candidate_key,))
                row = cur.fetchone()
                if row:
                    sample_count = int(row["sample_count"]) + 1
                    dayparts = _decode_json(row["dayparts_json"], default={})
                    value_min = row["value_min"]
                    value_max = row["value_max"]
                    first_seen_at = str(row["first_seen_at"])
                else:
                    sample_count = 1
                    dayparts = {}
                    value_min = None
                    value_max = None
                    first_seen_at = now
                dayparts[observation_bucket] = int(dayparts.get(observation_bucket, 0)) + 1
                observed_days = len({str(key).split(":", 1)[0] for key in dayparts.keys()})
                observed_dayparts = len(
                    {str(key).split(":", 1)[1] for key in dayparts.keys() if ":" in str(key)}
                )
                if (
                    sample_count >= self.min_sample_count
                    and observed_days >= self.min_observation_days
                    and observed_dayparts >= 3
                ):
                    status = "provisional"
                    confidence = min(0.95, 0.55 + (0.25 * min(1.0, sample_count / float(self.min_sample_count))) + (0.15 * min(1.0, observed_days / float(self.min_observation_days))))
                    provisional_dimensions.append(dim_name)
                elif sample_count > 1:
                    status = "observing"
                    confidence = min(0.75, 0.20 + (0.30 * min(1.0, sample_count / float(self.min_sample_count))))
                    observing_dimensions.append(dim_name)
                else:
                    status = "candidate"
                    confidence = 0.10
                cur.execute(
                    """
                    INSERT INTO dimension_lifecycle (
                        candidate_key, asset_id, asset_class, site_id, dimension_name, unit, status,
                        sample_count, observed_days, confidence, value_min, value_max, dayparts_json,
                        first_seen_at, last_seen_at, review_required, promoted_to_registry
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(candidate_key) DO UPDATE SET
                        asset_class=excluded.asset_class,
                        site_id=excluded.site_id,
                        status=excluded.status,
                        sample_count=excluded.sample_count,
                        observed_days=excluded.observed_days,
                        confidence=excluded.confidence,
                        dayparts_json=excluded.dayparts_json,
                        last_seen_at=excluded.last_seen_at
                    """,
                    (
                        candidate_key,
                        asset_id,
                        asset_class,
                        site_id,
                        dim_name,
                        "unknown",
                        status,
                        int(sample_count),
                        int(observed_days),
                        float(confidence),
                        value_min,
                        value_max,
                        json.dumps(dayparts, ensure_ascii=True),
                        first_seen_at,
                        now,
                        1,
                        0,
                    ),
                )
                dimension_lifecycle_status[dim_name] = status

            if not critical_dimensions:
                critical_dimensions = [dim for dim in CORE_DIMENSIONS if dim in accepted_dimensions]
            importance: dict[str, str] = {}
            for dim in accepted_dimensions:
                if dim in critical_dimensions:
                    importance[dim] = "critical_for_asset"
                elif dim in CORE_DIMENSIONS:
                    importance[dim] = "train_maturity_core"
                elif dim in AUX_DIMENSIONS:
                    importance[dim] = "train_maturity_context"
            for dim in candidate_dimensions:
                importance[dim] = "candidate_dimension"

            norm_source = "asset" if approved_dimensions else ("asset_class" if observed_series else "global")
            confidence_values = []
            for dim in list(observed_series.keys())[: min(5, len(observed_series))]:
                cur.execute(
                    "SELECT confidence FROM asset_dimension_norms WHERE asset_id = ? AND dimension_name = ?",
                    (asset_id, dim),
                )
                row = cur.fetchone()
                if row:
                    confidence_values.append(float(row["confidence"] or 0.0))
            asset_norm_confidence = sum(confidence_values) / float(len(confidence_values)) if confidence_values else 0.0
            conn.commit()
            return AssetNormSummary(
                asset_id=asset_id,
                asset_class=asset_class,
                site_id=site_id,
                asset_norm_scope="asset",
                asset_norm_confidence=float(round(asset_norm_confidence, 6)),
                norm_source=norm_source,
                asset_critical_dimensions=list(dict.fromkeys(critical_dimensions)),
                dimension_importance_by_asset=importance,
                candidate_dimensions=list(dict.fromkeys(candidate_dimensions)),
                observing_dimensions=list(dict.fromkeys(observing_dimensions)),
                provisional_dimensions=list(dict.fromkeys(provisional_dimensions)),
                approved_dimensions=list(dict.fromkeys(approved_dimensions)),
                dimension_lifecycle_status=dimension_lifecycle_status,
            )
        finally:
            conn.close()
