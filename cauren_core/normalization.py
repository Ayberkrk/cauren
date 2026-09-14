from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from cauren_agents.registry import AgentRegistry, build_default_registry

from .contracts import (
    NormalizationDecision,
    NormalizationRegistryEntry,
    NormalizationTrace,
    NormalizedSensorReading,
    SensorReading,
)


def _normalize_key(value: Any) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "").strip()).strip("_")


def _tuple_of_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (str(value),)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item).strip())
    return ()


def _registry_dir() -> Path:
    return Path(__file__).resolve().parent / "normalization_registry"


def _load_registry(path: Path) -> tuple[NormalizationRegistryEntry, ...]:
    if not path.exists():
        return ()
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries", ()) if isinstance(data, dict) else ()
    loaded: list[NormalizationRegistryEntry] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        loaded.append(
            NormalizationRegistryEntry(
                canonical_feature=str(item.get("canonical_feature") or "").strip(),
                aliases=_tuple_of_strings(item.get("aliases")),
                agent_scope=_tuple_of_strings(item.get("agent_scope")),
                sector_scope=_tuple_of_strings(item.get("sector_scope")),
                unit_hints={
                    str(key): str(value)
                    for key, value in (item.get("unit_hints") or {}).items()
                    if str(key).strip()
                }
                if isinstance(item.get("unit_hints"), dict)
                else {},
                client_scope=_tuple_of_strings(item.get("client_scope")),
                priority=int(item.get("priority") or 0),
                deprecated_aliases=_tuple_of_strings(item.get("deprecated_aliases")),
            )
        )
    return tuple(loaded)


class FeatureNormalizer:
    def __init__(self, *, registry: AgentRegistry | None = None):
        self.registry = registry or build_default_registry()
        self._schema_feature_keys: dict[str, set[str]] = {}
        self._schema_alias_candidates: dict[str, set[str]] = {}
        self._feature_scopes: dict[str, tuple[str, str]] = {}
        self._global_registry = _load_registry(_registry_dir() / "global_feature_registry.json")
        self._client_registry = _load_registry(_registry_dir() / "client_feature_overrides.json")
        self._bootstrap_schema_indexes()

    def _bootstrap_schema_indexes(self) -> None:
        for agent in self.registry.all():
            schema = agent.schema
            for feature in schema.feature_order:
                self._schema_feature_keys.setdefault(_normalize_key(feature), set()).add(feature)
                self._feature_scopes[feature] = (schema.agent_id, schema.sector)
                keys = {_normalize_key(feature)}
                keys.update(_normalize_key(alias) for alias in schema.aliases.get(feature, ()) if str(alias).strip())
                for key in keys:
                    self._schema_alias_candidates.setdefault(key, set()).add(feature)

    def normalize_readings(
        self,
        readings: Iterable[SensorReading],
        *,
        requested_agent_id: str | None = None,
        sector: str | None = None,
        sector_hint: str | None = None,
        client_id: str | None = None,
        site_id: str | None = None,
        normalization_profile: str | None = None,
    ) -> tuple[list[SensorReading], NormalizationTrace, list[dict[str, Any]]]:
        normalized: list[SensorReading] = []
        normalized_pairs: list[NormalizedSensorReading] = []
        review_items: list[dict[str, Any]] = []
        inherited_rejections: list[dict[str, Any]] = []
        unknown_names: list[str] = []
        ambiguous_names: list[str] = []
        normalized_count = 0
        unknown_count = 0
        ambiguous_count = 0
        for reading in readings:
            decision = self._normalize_reading(
                reading,
                requested_agent_id=requested_agent_id,
                sector=sector,
                sector_hint=sector_hint,
                client_id=client_id,
                site_id=site_id,
                normalization_profile=normalization_profile,
            )
            if decision.normalized_name and decision.normalized_name != reading.name:
                normalized_count += 1
            if decision.normalization_review_required:
                review_payload = {
                    "sensor_id": reading.sensor_id,
                    "name": reading.name,
                    "reason": (
                        "ambiguous_feature_name"
                        if decision.candidate_features
                        else "unknown_feature_name"
                    ),
                    "candidate_features": list(decision.candidate_features),
                    "normalization_source": decision.normalization_source,
                    "normalization_confidence": round(float(decision.normalization_confidence), 6),
                    "normalization_reason": decision.normalization_reason,
                }
                review_items.append(review_payload)
                if decision.candidate_features:
                    ambiguous_count += 1
                    ambiguous_names.append(reading.name)
                else:
                    unknown_count += 1
                    unknown_names.append(reading.name)
                inherited_rejections.append(review_payload)
            elif decision.normalized_name is None:
                unknown_count += 1
                unknown_names.append(reading.name)
            normalized_reading = self._apply_decision(reading, decision)
            normalized.append(normalized_reading)
            normalized_pairs.append(NormalizedSensorReading(reading=normalized_reading, decision=decision))

        trace = NormalizationTrace(
            decisions=tuple(item.decision for item in normalized_pairs),
            normalized_sensor_count=normalized_count,
            unknown_sensor_count=unknown_count,
            ambiguous_sensor_count=ambiguous_count,
            unknown_sensor_names=tuple(dict.fromkeys(unknown_names)),
            ambiguous_sensor_names=tuple(dict.fromkeys(ambiguous_names)),
            review_items=tuple(review_items),
        )
        return normalized, trace, inherited_rejections

    def _apply_decision(self, reading: SensorReading, decision: NormalizationDecision) -> SensorReading:
        metadata = dict(reading.metadata)
        metadata.update(
            {
                "raw_name": reading.name,
                "normalized_name": decision.normalized_name or "",
                "normalization_source": decision.normalization_source,
                "normalization_confidence": float(decision.normalization_confidence),
                "normalization_reason": decision.normalization_reason,
                "normalization_review_required": bool(decision.normalization_review_required),
            }
        )
        if decision.normalized_name:
            metadata["feature"] = decision.normalized_name
        if decision.candidate_features:
            metadata["normalization_candidates"] = list(decision.candidate_features)
        return replace(reading, name=decision.normalized_name or reading.name, metadata=metadata)

    def _normalize_reading(
        self,
        reading: SensorReading,
        *,
        requested_agent_id: str | None,
        sector: str | None,
        sector_hint: str | None,
        client_id: str | None,
        site_id: str | None,
        normalization_profile: str | None,
    ) -> NormalizationDecision:
        raw_key = _normalize_key(reading.name)
        scope_agent = str(requested_agent_id or "").strip()
        scope_sector = str(sector or sector_hint or "").strip()
        scope_client = str(client_id or "").strip()
        scope_site = str(site_id or "").strip()
        scope_profile = str(normalization_profile or "").strip()

        # 1) exact canonical match
        exact_candidates = self._narrow_candidates(
            sorted(self._schema_feature_keys.get(raw_key, ()) or ()),
            scope_agent=scope_agent,
            scope_sector=scope_sector,
        )
        if exact_candidates:
            if len(exact_candidates) > 1:
                return NormalizationDecision(
                    raw_name=reading.name,
                    normalized_name=None,
                    normalization_source="exact_canonical",
                    normalization_confidence=0.25,
                    normalization_reason="multiple exact canonical candidates after scope filtering; leaving unresolved",
                    normalization_review_required=True,
                    candidate_features=tuple(exact_candidates),
                )
            return NormalizationDecision(
                raw_name=reading.name,
                normalized_name=exact_candidates[0],
                normalization_source="exact_canonical",
                normalization_confidence=1.0,
                normalization_reason="raw sensor name already matches canonical feature",
            )

        # 2) schema alias match
        schema_candidates = self._resolve_schema_alias(raw_key, scope_agent=scope_agent, scope_sector=scope_sector)
        if schema_candidates:
            if len(schema_candidates) == 1:
                return NormalizationDecision(
                    raw_name=reading.name,
                    normalized_name=schema_candidates[0],
                    normalization_source="schema_alias",
                    normalization_confidence=0.99,
                    normalization_reason="matched schema alias map",
                )
            return NormalizationDecision(
                raw_name=reading.name,
                normalized_name=None,
                normalization_source="schema_alias",
                normalization_confidence=0.2,
                normalization_reason="multiple schema alias candidates; leaving feature unresolved",
                normalization_review_required=True,
                candidate_features=tuple(schema_candidates),
            )

        # 3-5) registry / overrides / safe pattern
        registry_match = self._resolve_registry_match(
            raw_name=reading.name,
            raw_key=raw_key,
            scope_agent=scope_agent,
            scope_sector=scope_sector,
            scope_client=scope_client,
            scope_site=scope_site,
            scope_profile=scope_profile,
        )
        if registry_match is not None:
            return registry_match

        return NormalizationDecision(
            raw_name=reading.name,
            normalized_name=None,
            normalization_source="unknown",
            normalization_confidence=0.0,
            normalization_reason="no canonical, alias, registry, or safe pattern match found",
            normalization_review_required=True,
        )

    def _resolve_schema_alias(self, raw_key: str, *, scope_agent: str, scope_sector: str) -> list[str]:
        candidates = sorted(self._schema_alias_candidates.get(raw_key, ()))
        if not candidates:
            return []
        return self._narrow_candidates(candidates, scope_agent=scope_agent, scope_sector=scope_sector)

    def _resolve_registry_match(
        self,
        raw_name: str,
        raw_key: str,
        *,
        scope_agent: str,
        scope_sector: str,
        scope_client: str,
        scope_site: str,
        scope_profile: str,
    ) -> NormalizationDecision | None:
        client_matches = self._find_registry_candidates(
            entries=self._client_registry,
            raw_name=raw_name,
            raw_key=raw_key,
            scope_agent=scope_agent,
            scope_sector=scope_sector,
            scope_client=scope_client,
            scope_site=scope_site,
            scope_profile=scope_profile,
        )
        global_matches = self._find_registry_candidates(
            entries=self._global_registry,
            raw_name=raw_name,
            raw_key=raw_key,
            scope_agent=scope_agent,
            scope_sector=scope_sector,
            scope_client=scope_client,
            scope_site=scope_site,
            scope_profile=scope_profile,
        )
        if client_matches:
            chosen = client_matches[0]
            reason = "matched client-specific normalization registry"
            if global_matches and global_matches[0]["canonical_feature"] != chosen["canonical_feature"]:
                reason = (
                    "client override selected different canonical feature than global registry; "
                    "keeping client-scoped mapping"
                )
            return self._decision_from_registry(chosen, source="client_override", reason=reason)
        if global_matches:
            return self._decision_from_registry(
                global_matches[0],
                source=str(global_matches[0]["source"]),
                reason=str(global_matches[0]["reason"]),
            )
        return None

    def _decision_from_registry(self, match: dict[str, Any], *, source: str, reason: str) -> NormalizationDecision:
        candidates = tuple(str(item) for item in match.get("candidate_features", ()) if str(item).strip())
        if len(candidates) > 1:
            return NormalizationDecision(
                raw_name=str(match.get("raw_name") or ""),
                normalized_name=None,
                normalization_source=source,
                normalization_confidence=0.25,
                normalization_reason="multiple registry candidates after scope filtering; leaving unresolved",
                normalization_review_required=True,
                candidate_features=candidates,
            )
        confidence = 0.97 if source in {"global_registry", "client_override"} else 0.72
        return NormalizationDecision(
            raw_name=str(match.get("raw_name") or ""),
            normalized_name=str(match.get("canonical_feature") or ""),
            normalization_source=source,
            normalization_confidence=confidence,
            normalization_reason=reason,
        )

    def _find_registry_candidates(
        self,
        *,
        entries: tuple[NormalizationRegistryEntry, ...],
        raw_name: str,
        raw_key: str,
        scope_agent: str,
        scope_sector: str,
        scope_client: str,
        scope_site: str,
        scope_profile: str,
    ) -> list[dict[str, Any]]:
        exact: list[dict[str, Any]] = []
        pattern: list[dict[str, Any]] = []
        for entry in entries:
            if not self._entry_in_scope(
                entry,
                scope_agent=scope_agent,
                scope_sector=scope_sector,
                scope_client=scope_client,
                scope_site=scope_site,
                scope_profile=scope_profile,
            ):
                continue
            exact_aliases = []
            pattern_aliases = []
            for alias in (*entry.aliases, *entry.deprecated_aliases):
                if "*" in alias:
                    pattern_aliases.append(_normalize_key(str(alias).replace("*", "")))
                else:
                    exact_aliases.append(_normalize_key(alias))
            if raw_key in exact_aliases:
                exact.append(
                    {
                        "raw_name": raw_name,
                        "canonical_feature": entry.canonical_feature,
                        "candidate_features": (entry.canonical_feature,),
                        "priority": entry.priority,
                        "source": "global_registry",
                        "reason": "matched global synonym registry" if not entry.client_scope else "matched scoped synonym registry",
                    }
                )
                continue
            for alias in pattern_aliases:
                prefix = alias
                if prefix and raw_key.startswith(prefix):
                    pattern.append(
                        {
                            "raw_name": raw_name,
                            "canonical_feature": entry.canonical_feature,
                            "candidate_features": (entry.canonical_feature,),
                            "priority": entry.priority,
                            "source": "pattern",
                            "reason": "matched safe registry pattern",
                        }
                    )
                    break
        exact.sort(key=lambda item: (-int(item["priority"]), str(item["canonical_feature"])))
        pattern.sort(key=lambda item: (-int(item["priority"]), str(item["canonical_feature"])))
        return exact or pattern

    def _entry_in_scope(
        self,
        entry: NormalizationRegistryEntry,
        *,
        scope_agent: str,
        scope_sector: str,
        scope_client: str,
        scope_site: str,
        scope_profile: str,
    ) -> bool:
        if entry.agent_scope and scope_agent and scope_agent not in entry.agent_scope:
            return False
        if entry.agent_scope and not scope_agent and len(entry.agent_scope) == 1:
            # single-agent entries are still allowed; the alias is effectively unique
            pass
        if entry.sector_scope and scope_sector and scope_sector not in entry.sector_scope:
            return False
        if entry.client_scope:
            active_scopes = {scope_client, scope_site, scope_profile}
            if not any(scope and scope in entry.client_scope for scope in active_scopes):
                return False
        return True

    def _narrow_candidates(self, candidates: list[str], *, scope_agent: str, scope_sector: str) -> list[str]:
        if len(candidates) <= 1:
            if not candidates:
                return candidates
            if scope_agent and self._feature_scopes.get(candidates[0], ("", ""))[0] != scope_agent:
                return []
            if scope_sector and self._feature_scopes.get(candidates[0], ("", ""))[1] != scope_sector:
                return []
            return candidates
        if scope_agent:
            filtered = [feature for feature in candidates if self._feature_scopes.get(feature, ("", ""))[0] == scope_agent]
            if len(filtered) == 1:
                return filtered
            return filtered
        if scope_sector:
            filtered = [feature for feature in candidates if self._feature_scopes.get(feature, ("", ""))[1] == scope_sector]
            if len(filtered) == 1:
                return filtered
            return filtered
        return candidates
