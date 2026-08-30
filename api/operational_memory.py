from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _now_ts() -> float:
    return float(time.time())


@dataclass(frozen=True)
class OperationalRouteMemory:
    asset_id: str
    sensor_layout_key: str
    selected_agent: str
    agent_schema_version: str
    agent_registry_version: str
    route_confidence: float
    route_reason: str
    route_needs_context: bool
    route_ambiguity_reason: str
    candidate_agents: tuple[dict[str, Any], ...]
    sector_score_breakdown: tuple[dict[str, Any], ...]
    selected_sector_score: float
    selected_sector_prior: float
    selection_strategy: str
    prior_conflict: bool
    prior_conflict_reason: str
    feature_map_summary: dict[str, Any]
    usage_count: int
    last_seen_ts: float
    last_success_ts: float
    last_fallback_reason: str
    rejected_sample_rate: float
    anomaly_family: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OperationalRouteMemory":
        return cls(
            asset_id=str(payload.get("asset_id") or ""),
            sensor_layout_key=str(payload.get("sensor_layout_key") or ""),
            selected_agent=str(payload.get("selected_agent") or ""),
            agent_schema_version=str(payload.get("agent_schema_version") or ""),
            agent_registry_version=str(payload.get("agent_registry_version") or ""),
            route_confidence=float(payload.get("route_confidence") or 0.0),
            route_reason=str(payload.get("route_reason") or ""),
            route_needs_context=bool(payload.get("route_needs_context", False)),
            route_ambiguity_reason=str(payload.get("route_ambiguity_reason") or ""),
            candidate_agents=tuple(
                dict(item) for item in payload.get("candidate_agents", ()) if isinstance(item, dict)
            ),
            sector_score_breakdown=tuple(
                dict(item) for item in payload.get("sector_score_breakdown", ()) if isinstance(item, dict)
            ),
            selected_sector_score=float(payload.get("selected_sector_score") or 0.0),
            selected_sector_prior=float(payload.get("selected_sector_prior") or 0.0),
            selection_strategy=str(payload.get("selection_strategy") or "prior_weighted_score_fusion"),
            prior_conflict=bool(payload.get("prior_conflict", False)),
            prior_conflict_reason=str(payload.get("prior_conflict_reason") or ""),
            feature_map_summary=dict(payload.get("feature_map_summary") or {}),
            usage_count=int(payload.get("usage_count") or 0),
            last_seen_ts=float(payload.get("last_seen_ts") or 0.0),
            last_success_ts=float(payload.get("last_success_ts") or 0.0),
            last_fallback_reason=str(payload.get("last_fallback_reason") or ""),
            rejected_sample_rate=float(payload.get("rejected_sample_rate") or 0.0),
            anomaly_family=str(payload.get("anomaly_family") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "sensor_layout_key": self.sensor_layout_key,
            "selected_agent": self.selected_agent,
            "agent_schema_version": self.agent_schema_version,
            "agent_registry_version": self.agent_registry_version,
            "route_confidence": round(float(self.route_confidence), 6),
            "route_reason": self.route_reason,
            "route_needs_context": bool(self.route_needs_context),
            "route_ambiguity_reason": self.route_ambiguity_reason,
            "candidate_agents": [dict(item) for item in self.candidate_agents],
            "sector_score_breakdown": [dict(item) for item in self.sector_score_breakdown],
            "selected_sector_score": round(float(self.selected_sector_score), 6),
            "selected_sector_prior": round(float(self.selected_sector_prior), 6),
            "selection_strategy": self.selection_strategy,
            "prior_conflict": bool(self.prior_conflict),
            "prior_conflict_reason": self.prior_conflict_reason,
            "feature_map_summary": dict(self.feature_map_summary),
            "usage_count": int(self.usage_count),
            "last_seen_ts": round(float(self.last_seen_ts), 6),
            "last_success_ts": round(float(self.last_success_ts), 6),
            "last_fallback_reason": self.last_fallback_reason,
            "rejected_sample_rate": round(float(self.rejected_sample_rate), 6),
            "anomaly_family": self.anomaly_family,
        }


class OperationalMemoryStore:
    def __init__(
        self,
        *,
        state_path: str | Path,
        ttl_sec: float = 21600.0,
        max_entries: int = 4096,
    ) -> None:
        self.state_path = Path(state_path)
        self.ttl_sec = float(max(60.0, ttl_sec))
        self.max_entries = int(max(32, max_entries))
        self._lock = threading.RLock()
        self._state = self._load_state()

    def _base_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "updated_at": _now_ts(),
            "entries": {},
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._base_state()
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("invalid_state")
            payload.setdefault("entries", {})
            payload.setdefault("updated_at", _now_ts())
            return payload
        except Exception:
            backup = self.state_path.with_suffix(self.state_path.suffix + ".corrupt")
            try:
                self.state_path.replace(backup)
            except OSError:
                pass
            return self._base_state()

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state["updated_at"] = _now_ts()
        tmp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self._state, ensure_ascii=True, indent=2), encoding="utf-8")
        tmp_path.replace(self.state_path)

    @staticmethod
    def _entry_key(asset_id: str, sensor_layout_key: str) -> str:
        return f"{asset_id}|{sensor_layout_key}"

    def _entries(self) -> dict[str, dict[str, Any]]:
        entries = self._state.setdefault("entries", {})
        if not isinstance(entries, dict):
            entries = {}
            self._state["entries"] = entries
        return entries

    def _evict_if_needed(self) -> None:
        entries = self._entries()
        if len(entries) <= self.max_entries:
            return
        ordered = sorted(
            entries.items(),
            key=lambda item: float((item[1] or {}).get("last_seen_ts", 0.0)),
        )
        trim_count = max(1, len(entries) - self.max_entries)
        for key, _ in ordered[:trim_count]:
            entries.pop(key, None)

    def lookup(
        self,
        *,
        asset_id: str,
        sensor_layout_key: str,
        agent_registry_version: str,
        requested_agent_id: str | None = None,
        requested_sector: str | None = None,
    ) -> tuple[OperationalRouteMemory | None, str]:
        with self._lock:
            key = self._entry_key(asset_id, sensor_layout_key)
            raw = self._entries().get(key)
            if not isinstance(raw, dict):
                return None, "miss"
            entry = OperationalRouteMemory.from_dict(raw)
            age_sec = max(0.0, _now_ts() - float(entry.last_success_ts))
            if age_sec > self.ttl_sec:
                self._entries().pop(key, None)
                self._save_state()
                return None, "expired"
            if entry.agent_registry_version != str(agent_registry_version):
                self._entries().pop(key, None)
                self._save_state()
                return None, "registry_version_mismatch"
            if requested_agent_id and str(requested_agent_id) != entry.selected_agent:
                return None, "requested_agent_mismatch"
            if requested_sector and requested_sector.strip():
                summary_sector = str(entry.feature_map_summary.get("sector") or "").strip()
                if summary_sector and summary_sector != str(requested_sector).strip():
                    return None, "requested_sector_mismatch"
            return entry, "hit"

    def remember_success(
        self,
        *,
        asset_id: str,
        sensor_layout_key: str,
        selected_agent: str,
        agent_schema_version: str,
        agent_registry_version: str,
        route_confidence: float,
        route_reason: str,
        route_needs_context: bool,
        route_ambiguity_reason: str,
        candidate_agents: list[dict[str, Any]],
        sector_score_breakdown: list[dict[str, Any]],
        selected_sector_score: float,
        selected_sector_prior: float,
        selection_strategy: str,
        prior_conflict: bool,
        prior_conflict_reason: str,
        feature_map_summary: dict[str, Any],
        rejected_sample_rate: float,
        anomaly_family: str,
    ) -> OperationalRouteMemory:
        with self._lock:
            key = self._entry_key(asset_id, sensor_layout_key)
            previous = self._entries().get(key)
            previous_usage = 0
            previous_last_fallback = ""
            if isinstance(previous, dict):
                previous_usage = int(previous.get("usage_count") or 0)
                previous_last_fallback = str(previous.get("last_fallback_reason") or "")
            now_ts = _now_ts()
            entry = OperationalRouteMemory(
                asset_id=str(asset_id),
                sensor_layout_key=str(sensor_layout_key),
                selected_agent=str(selected_agent),
                agent_schema_version=str(agent_schema_version),
                agent_registry_version=str(agent_registry_version),
                route_confidence=float(route_confidence),
                route_reason=str(route_reason),
                route_needs_context=bool(route_needs_context),
                route_ambiguity_reason=str(route_ambiguity_reason),
                candidate_agents=tuple(dict(item) for item in candidate_agents if isinstance(item, dict)),
                sector_score_breakdown=tuple(
                    dict(item) for item in sector_score_breakdown if isinstance(item, dict)
                ),
                selected_sector_score=float(selected_sector_score),
                selected_sector_prior=float(selected_sector_prior),
                selection_strategy=str(selection_strategy or "prior_weighted_score_fusion"),
                prior_conflict=bool(prior_conflict),
                prior_conflict_reason=str(prior_conflict_reason or ""),
                feature_map_summary=dict(feature_map_summary),
                usage_count=previous_usage,
                last_seen_ts=now_ts,
                last_success_ts=now_ts,
                last_fallback_reason=previous_last_fallback,
                rejected_sample_rate=float(rejected_sample_rate),
                anomaly_family=str(anomaly_family),
            )
            self._entries()[key] = entry.to_dict()
            self._evict_if_needed()
            self._save_state()
            return entry

    def mark_hot_path_success(self, *, asset_id: str, sensor_layout_key: str) -> OperationalRouteMemory | None:
        with self._lock:
            key = self._entry_key(asset_id, sensor_layout_key)
            raw = self._entries().get(key)
            if not isinstance(raw, dict):
                return None
            entry = OperationalRouteMemory.from_dict(raw)
            updated = OperationalRouteMemory(
                **{
                    **entry.__dict__,
                    "usage_count": int(entry.usage_count) + 1,
                    "last_seen_ts": _now_ts(),
                }
            )
            self._entries()[key] = updated.to_dict()
            self._save_state()
            return updated

    def note_fallback(
        self,
        *,
        asset_id: str,
        sensor_layout_key: str,
        reason: str,
        invalidate: bool = False,
    ) -> None:
        with self._lock:
            key = self._entry_key(asset_id, sensor_layout_key)
            raw = self._entries().get(key)
            if not isinstance(raw, dict):
                return
            if invalidate:
                self._entries().pop(key, None)
                self._save_state()
                return
            entry = OperationalRouteMemory.from_dict(raw)
            updated = OperationalRouteMemory(
                **{
                    **entry.__dict__,
                    "last_seen_ts": _now_ts(),
                    "last_fallback_reason": str(reason),
                }
            )
            self._entries()[key] = updated.to_dict()
            self._save_state()

    def get_entry(self, *, asset_id: str, sensor_layout_key: str) -> OperationalRouteMemory | None:
        with self._lock:
            raw = self._entries().get(self._entry_key(asset_id, sensor_layout_key))
            if not isinstance(raw, dict):
                return None
            return OperationalRouteMemory.from_dict(raw)
