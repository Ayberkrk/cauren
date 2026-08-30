from __future__ import annotations

import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .contracts import AgentSchema, FeatureMetadata, SensorReading, SensorWindow


def _normalize_key(value: Any) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "").strip()).strip("_")


def _as_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "ok"}:
        return True
    if text in {"0", "false", "no", "n", "bad"}:
        return False
    return default


def _as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _as_timestamp(value: Any) -> float | None:
    parsed = _as_float(value)
    if parsed is not None:
        return parsed
    if isinstance(value, str) and value.strip():
        try:
            dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return float(dt.timestamp())
    return None


def parse_sensor_readings(rows: Iterable[Mapping[str, Any]]) -> tuple[list[SensorReading], list[dict[str, Any]]]:
    readings: list[SensorReading] = []
    rejected: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        sensor_id = str(row.get("sensor_id") or row.get("id") or row.get("source_key") or f"s{idx+1}").strip()
        name = str(
            row.get("name")
            or row.get("sensor_name")
            or row.get("tag")
            or row.get("dimension")
            or sensor_id
        ).strip()
        value = _as_float(row.get("value"))
        if value is None:
            rejected.append({"sensor_id": sensor_id, "name": name, "reason": "non_numeric_value"})
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        readings.append(
            SensorReading(
                sensor_id=sensor_id,
                name=name,
                unit=str(row.get("unit") or "").strip(),
                value=float(value),
                timestamp=_as_timestamp(row.get("timestamp")),
                quality=_as_bool(row.get("quality"), True),
                metadata=dict(metadata),
            )
        )
    return readings, rejected


def sensor_layout_key(rows: Iterable[Mapping[str, Any]]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        name = str(
            row.get("name")
            or row.get("sensor_name")
            or row.get("tag")
            or row.get("dimension")
            or row.get("sensor_id")
            or row.get("id")
            or ""
        ).strip()
        unit = str(row.get("unit") or "").strip()
        key = f"{_normalize_key(name)}|{_normalize_key(unit)}"
        counts[key] += 1
    if not counts:
        return "layout:empty"
    encoded = ",".join(f"{key}:{counts[key]}" for key in sorted(counts))
    return f"layout:{encoded}"


class AgentSchemaAdapter:
    """Maps arbitrary sensor names into an agent-owned feature layout."""

    def __init__(self, schema: AgentSchema):
        self.schema = schema
        self._alias_to_feature = self._build_alias_map(schema)

    @staticmethod
    def _build_alias_map(schema: AgentSchema) -> dict[str, str]:
        alias_map: dict[str, str] = {}
        for feature in schema.feature_order:
            alias_map[_normalize_key(feature)] = feature
            for alias in schema.aliases.get(feature, ()):
                alias_map[_normalize_key(alias)] = feature
        return alias_map

    def _feature_for(self, reading: SensorReading) -> str | None:
        for candidate in (
            reading.name,
            reading.sensor_id,
            reading.metadata.get("feature"),
            reading.metadata.get("normalized_name"),
        ):
            key = _normalize_key(candidate)
            if key in self._alias_to_feature:
                return self._alias_to_feature[key]
        return None

    @staticmethod
    def _normalize_unit(value: float, source_unit: str, target_unit: str) -> float:
        source = source_unit.strip().lower().replace(" ", "")
        target = target_unit.strip().lower().replace(" ", "")
        if not source or not target or source == target:
            return float(value)
        if target in {"c", "degc", "celsius"}:
            if source in {"k", "kelvin"}:
                return float(value) - 273.15
            if source in {"f", "fahrenheit"}:
                return (float(value) - 32.0) * 5.0 / 9.0
        if target in {"k", "kelvin"}:
            if source in {"c", "degc", "celsius"}:
                return float(value) + 273.15
        if target == "bar":
            if source == "pa":
                return float(value) / 100000.0
            if source == "kpa":
                return float(value) / 100.0
            if source == "mpa":
                return float(value) * 10.0
            if source == "psi":
                return float(value) * 0.0689475729
        if target in {"kw", "kilowatt"} and source in {"w", "watt"}:
            return float(value) / 1000.0
        if target in {"w", "watt"} and source in {"kw", "kilowatt"}:
            return float(value) * 1000.0
        if target in {"min", "minute", "minutes"}:
            if source in {"s", "sec", "second", "seconds"}:
                return float(value) / 60.0
            if source in {"ms", "millisecond", "milliseconds"}:
                return float(value) / 60000.0
        if target in {"s", "sec", "second", "seconds"}:
            if source in {"min", "minute", "minutes"}:
                return float(value) * 60.0
            if source in {"ms", "millisecond", "milliseconds"}:
                return float(value) / 1000.0
        if target in {"ms", "millisecond", "milliseconds"}:
            if source in {"s", "sec", "second", "seconds"}:
                return float(value) * 1000.0
            if source in {"min", "minute", "minutes"}:
                return float(value) * 60000.0
        return float(value)

    def _feature_value(
        self,
        rows: list[SensorReading],
        *,
        target_unit: str,
    ) -> tuple[float, bool, tuple[str, ...], float] | None:
        if not rows:
            return None
        good_rows = [row for row in rows if row.quality]
        source_rows = good_rows or rows
        converted = [
            self._normalize_unit(row.value, row.unit, target_unit)
            for row in source_rows
        ]
        return (
            float(sum(converted) / float(len(converted))),
            bool(good_rows),
            tuple(row.sensor_id for row in source_rows),
            float(len(good_rows)) / float(max(1, len(rows))),
        )

    def build_window(
        self,
        readings: Iterable[SensorReading],
        *,
        seq_len: int = 16,
        timestamp: float | None = None,
        inherited_rejections: Iterable[dict[str, Any]] = (),
    ) -> SensorWindow:
        grouped: dict[str, list[SensorReading]] = defaultdict(list)
        rejected = list(inherited_rejections)
        raw_count = 0
        for reading in readings:
            raw_count += 1
            feature = self._feature_for(reading)
            if feature is None:
                rejected.append(
                    {
                        "sensor_id": reading.sensor_id,
                        "name": reading.name,
                        "reason": "not_in_agent_schema",
                        "agent_id": self.schema.agent_id,
                    }
                )
                continue
            grouped[feature].append(reading)

        seq_len = max(1, int(seq_len))
        feature_order = self.schema.feature_order
        observed_timestamps = sorted(
            {
                float(row.timestamp)
                for rows in grouped.values()
                for row in rows
                if row.timestamp is not None
            }
        )
        use_temporal_window = bool(observed_timestamps)
        if use_temporal_window:
            slots = observed_timestamps[-seq_len:]
        else:
            slots = [float(timestamp if timestamp is not None else time.time())]

        rows_by_feature_ts: dict[str, dict[float, list[SensorReading]]] = {}
        for feature, rows in grouped.items():
            by_ts: dict[float, list[SensorReading]] = defaultdict(list)
            for row in rows:
                ts_key = float(row.timestamp) if row.timestamp is not None else float(slots[-1])
                by_ts[ts_key].append(row)
            rows_by_feature_ts[feature] = by_ts

        matrix_rows: list[tuple[float, ...]] = []
        mask_rows: list[tuple[bool, ...]] = []
        metadata_sources: dict[str, set[str]] = {feature: set() for feature in feature_order}
        metadata_quality_sum: dict[str, float] = {feature: 0.0 for feature in feature_order}
        metadata_quality_count: dict[str, int] = {feature: 0 for feature in feature_order}
        last_seen: dict[str, float] = {}
        for slot in slots:
            values: list[float] = []
            presence: list[bool] = []
            for feature in feature_order:
                target_unit = self.schema.units.get(feature, "")
                feature_rows = rows_by_feature_ts.get(feature, {}).get(float(slot), [])
                observed = self._feature_value(feature_rows, target_unit=target_unit)
                if observed is None:
                    values.append(float(last_seen.get(feature, 0.0)))
                    presence.append(False)
                    continue
                value, is_present, source_ids, quality = observed
                values.append(value)
                presence.append(is_present)
                last_seen[feature] = value
                metadata_sources[feature].update(source_ids)
                metadata_quality_sum[feature] += quality
                metadata_quality_count[feature] += 1
            matrix_rows.append(tuple(values))
            mask_rows.append(tuple(presence))

        observed_step_count = len(matrix_rows)
        while len(matrix_rows) < seq_len:
            matrix_rows.insert(0, matrix_rows[0] if matrix_rows else tuple(0.0 for _ in feature_order))
            mask_rows.insert(0, mask_rows[0] if mask_rows else tuple(False for _ in feature_order))
            slots.insert(0, slots[0] if slots else float(timestamp if timestamp is not None else time.time()))

        metadata: list[FeatureMetadata] = []
        for feature in feature_order:
            quality_count = metadata_quality_count[feature]
            quality = (
                metadata_quality_sum[feature] / float(quality_count)
                if quality_count > 0
                else 0.0
            )
            metadata.append(
                FeatureMetadata(
                    name=feature,
                    unit=self.schema.units.get(feature, ""),
                    source_sensor_ids=tuple(sorted(metadata_sources[feature])),
                    quality=float(quality),
                    metadata=dict(self.schema.feature_metadata.get(feature, {})),
                )
            )

        matrix = tuple(matrix_rows[-seq_len:])
        mask = tuple(mask_rows[-seq_len:])
        return SensorWindow(
            matrix=matrix,
            features=tuple(metadata),
            presence_mask=mask,
            timestamps=tuple(float(ts) for ts in slots[-seq_len:]),
            rejected_samples=tuple(rejected),
            raw_sensor_count=raw_count,
            observed_step_count=observed_step_count,
        )
