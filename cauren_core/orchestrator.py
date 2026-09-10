from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from cauren_agents.registry import AgentRegistry, build_default_registry, build_registry
from cauren_agents.router import AgentRouter

from .adapter import AgentSchemaAdapter, parse_sensor_readings
from .contracts import AgentCandidate, CaurenDiagnosis, NormalizationTrace, SectorScoreBreakdown
from .normalization import FeatureNormalizer
from cauren_agents.base import AgentRoute
from .quality_control import evaluate_quality
from .runtime import CaurenCoreRuntime
from .uncertainty import estimate_uncertainty


class CaurenPipeline:
    def __init__(self, *, registry: AgentRegistry | None = None, core: CaurenCoreRuntime | None = None):
        self.registry = registry or build_default_registry()
        self.router = AgentRouter(self.registry)
        self.core = core or CaurenCoreRuntime()
        self.normalizer = FeatureNormalizer(registry=self.registry)
        self._adapter_cache: dict[str, AgentSchemaAdapter] = {}
        self.registry_version = "|".join(
            f"{agent.schema.agent_id}:{agent.schema.version}"
            for agent in sorted(self.registry.all(), key=lambda item: item.schema.agent_id)
        )

    @classmethod
    def from_default_registry(cls) -> "CaurenPipeline":
        return cls(registry=build_default_registry(), core=CaurenCoreRuntime())

    @classmethod
    def from_agent_ids(cls, agent_ids: list[str] | tuple[str, ...]) -> "CaurenPipeline":
        return cls(registry=build_registry(list(agent_ids)), core=CaurenCoreRuntime())

    def diagnose(self, payload: Mapping[str, Any]) -> CaurenDiagnosis:
        route, agent, window, normalization_trace = self._build_agent_window(payload)
        return self._diagnose_from_route(
            route=route,
            agent=agent,
            window=window,
            payload=payload,
            normalization_trace=normalization_trace,
        )

    def diagnose_with_preselected_route(
        self,
        payload: Mapping[str, Any],
        *,
        selected_agent_id: str,
        route_confidence: float,
        route_reason: str,
        route_needs_context: bool = False,
        route_ambiguity_reason: str = "",
        candidate_agents: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
        sector_score_breakdown: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
        selected_sector_score: float | None = None,
        selected_sector_prior: float | None = None,
        selection_strategy: str = "prior_weighted_score_fusion",
        prior_conflict: bool = False,
        prior_conflict_reason: str = "",
        context_gate_status: str = "ok",
        context_gate_reason: str = "",
        context_required_fields: list[str] | tuple[str, ...] = (),
    ) -> CaurenDiagnosis:
        route = AgentRoute(
            selected_agent_id=selected_agent_id,
            candidates=tuple(
                AgentCandidate(
                    agent_id=str(item.get("agent_id") or selected_agent_id),
                    score=float(item.get("score") or 0.0),
                    reason=str(item.get("reason") or route_reason),
                )
                for item in candidate_agents
                if isinstance(item, dict)
            )
            or (
                AgentCandidate(
                    agent_id=selected_agent_id,
                    score=float(route_confidence),
                    reason=str(route_reason),
                ),
            ),
            confidence=float(route_confidence),
            reason=str(route_reason),
            needs_context=bool(route_needs_context),
            ambiguity_reason=str(route_ambiguity_reason),
            sector_scores=tuple(
                SectorScoreBreakdown(
                    agent_id=str(item.get("agent_id") or selected_agent_id),
                    sector=str(item.get("sector") or ""),
                    model_evidence_score=float(item.get("model_evidence_score") or 0.0),
                    metadata_prior_score=float(item.get("metadata_prior_score") or 0.0),
                    context_prior_score=float(item.get("context_prior_score") or 0.0),
                    penalty_score=float(item.get("penalty_score") or 0.0),
                    final_sector_score=float(
                        item.get("final_sector_score")
                        or item.get("score")
                        or 0.0
                    ),
                    evidence_reasons=tuple(str(value) for value in item.get("evidence_reasons", ()) if str(value)),
                    prior_reasons=tuple(str(value) for value in item.get("prior_reasons", ()) if str(value)),
                    penalty_reasons=tuple(str(value) for value in item.get("penalty_reasons", ()) if str(value)),
                    ingress_tags=tuple(str(value) for value in item.get("ingress_tags", ()) if str(value)),
                    prior_conflict=bool(item.get("prior_conflict", False)),
                )
                for item in sector_score_breakdown
                if isinstance(item, dict)
            ),
            selected_sector_score=float(selected_sector_score if selected_sector_score is not None else route_confidence),
            selected_sector_prior=float(selected_sector_prior or 0.0),
            selection_strategy=str(selection_strategy or "prior_weighted_score_fusion"),
            prior_conflict=bool(prior_conflict),
            prior_conflict_reason=str(prior_conflict_reason or ""),
            context_gate_status=str(context_gate_status or "ok"),
            context_gate_reason=str(context_gate_reason or ""),
            context_required_fields=tuple(str(item) for item in context_required_fields if str(item)),
        )
        agent, window, normalization_trace = self._build_agent_window_preselected(payload, selected_agent_id=selected_agent_id)
        return self._diagnose_from_route(
            route=route,
            agent=agent,
            window=window,
            payload=payload,
            normalization_trace=normalization_trace,
        )

    def _diagnose_from_route(
        self,
        *,
        route,
        agent,
        window,
        payload: Mapping[str, Any],
        normalization_trace: NormalizationTrace,
    ) -> CaurenDiagnosis:
        core_output = self.core.classify(
            window,
            sampling_hz=float(payload.get("sampling_hz") or 1.0),
            runtime_mode=str(payload.get("runtime_mode") or "") or None,
            context={
                "selected_agent_id": agent.schema.agent_id,
                "site_context": payload.get("site_context") or {},
                "forecast_context": payload.get("forecast_context") or {},
                "scenario_metadata": payload.get("scenario_metadata") or {},
                "mission_phase": payload.get("mission_phase") or "",
                "client_id": payload.get("client_id") or "",
            },
        )
        physics_evidence = agent.physics.evaluate(
            window=window,
            schema=agent.schema,
            context={
                "client_id": payload.get("client_id"),
                "site_context": payload.get("site_context") or {},
                "forecast_context": payload.get("forecast_context") or {},
                "control_mode": payload.get("control_mode") or "guarded_auto",
                "sensor_schema_version": payload.get("sensor_schema_version") or agent.schema.version,
                "core_output": core_output,
                # Optional: the result of cauren_physics.oma.compare_to_baseline
                # (e.g. from tools/run_oma_identification.py), as a dict. Lets a
                # caller fold sensor-derived modal-frequency drift evidence into
                # the same diagnosis instead of running physics evaluation twice.
                "oma_frequency_drift": payload.get("oma_frequency_drift") or {},
            },
        )
        diagnosis = agent.compose_result(
            core_output=core_output,
            physics_evidence=physics_evidence,
            route=route,
            window=window,
        )
        quality_report = evaluate_quality(window, agent.schema, normalization_trace)
        uncertainty_report = estimate_uncertainty(
            risk_score=diagnosis.risk_score,
            core_confidence=core_output.confidence,
            quality_report=quality_report,
            schema=agent.schema,
            missing_features=physics_evidence.missing_features,
        )
        merged_outputs = dict(diagnosis.agent_outputs)
        merged_outputs.update(self._normalization_outputs(normalization_trace))
        merged_outputs["quality_control"] = quality_report.to_dict()
        merged_outputs["uncertainty_report"] = uncertainty_report.to_dict()
        return replace(diagnosis, agent_outputs=merged_outputs)

    def calibrate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        route, agent, window, normalization_trace = self._build_agent_window(payload)
        calibrated = self.core.calibrate(
            window,
            sampling_hz=float(payload.get("sampling_hz") or 1.0),
            runtime_mode=str(payload.get("runtime_mode") or "") or None,
        )
        quality_report = evaluate_quality(window, agent.schema, normalization_trace)
        return {
            "quality_control": quality_report.to_dict(),
            "selected_agent": agent.schema.agent_id,
            "candidate_agents": [
                {"agent_id": c.agent_id, "score": round(float(c.score), 6), "reason": c.reason}
                for c in route.candidates
            ],
            "sector_scores": [
                {
                    "agent_id": item.agent_id,
                    "sector": item.sector,
                    "score": round(float(item.final_sector_score), 6),
                }
                for item in route.sector_scores
            ],
            "sector_score_breakdown": [item.to_dict() for item in route.sector_scores],
            "selected_sector_score": round(float(route.selected_sector_score), 6),
            "selected_sector_prior": round(float(route.selected_sector_prior), 6),
            "sector_selection_strategy": route.selection_strategy,
            "sector_selection_needs_context": bool(route.needs_context),
            "sector_prior_conflict": bool(route.prior_conflict),
            "sector_prior_conflict_reason": route.prior_conflict_reason,
            "context_gate_status": route.context_gate_status,
            "context_gate_reason": route.context_gate_reason,
            "context_required_fields": list(route.context_required_fields),
            "confidence": round(float(route.confidence), 6),
            "reason": route.reason,
            "needs_context": bool(route.needs_context),
            "ambiguity_reason": route.ambiguity_reason,
            "calibrated_data": [list(row) for row in calibrated],
            "feature_names": [feature.name for feature in window.features],
            "feature_metadata": [
                {
                    "name": feature.name,
                    "unit": feature.unit,
                    "source_sensor_ids": list(feature.source_sensor_ids),
                    "quality": round(float(feature.quality), 6),
                    "metadata": dict(feature.metadata),
                }
                for feature in window.features
            ],
            "presence_mask": [list(row) for row in window.presence_mask],
            "timestamps": list(window.timestamps),
            "rejected_samples": list(window.rejected_samples),
            "raw_sensor_count": int(window.raw_sensor_count),
            **self._normalization_outputs(normalization_trace),
        }

    def _build_agent_window(self, payload: Mapping[str, Any]):
        sensors_payload = payload.get("sensors") or payload.get("sensor_readings") or []
        if not isinstance(sensors_payload, list):
            sensors_payload = []
        readings, rejected = parse_sensor_readings(sensors_payload)
        readings, normalization_trace, normalization_rejections = self.normalizer.normalize_readings(
            readings,
            requested_agent_id=str(payload.get("agent_id") or "").strip() or None,
            sector=str(payload.get("sector") or "").strip() or None,
            sector_hint=str(payload.get("sector_hint") or "").strip() or None,
            client_id=str(payload.get("client_id") or "").strip() or None,
            site_id=str(payload.get("site_id") or "").strip() or None,
            normalization_profile=str(payload.get("normalization_profile") or "").strip() or None,
        )
        rejected.extend(normalization_rejections)

        route = self.router.route(
            readings=readings,
            requested_agent_id=str(payload.get("agent_id") or "").strip() or None,
            sector=str(payload.get("sector") or "").strip() or None,
            site_context=payload.get("site_context") or {},
            ingress_tag=str(payload.get("ingress_tag") or "").strip() or None,
            ingress_tags=payload.get("ingress_tags") if isinstance(payload.get("ingress_tags"), list) else None,
            sector_hint=str(payload.get("sector_hint") or "").strip() or None,
            sector_hint_confidence=float(payload.get("sector_hint_confidence")) if payload.get("sector_hint_confidence") is not None else None,
            source_metadata=payload.get("source_metadata") if isinstance(payload.get("source_metadata"), dict) else None,
            identity_hints={
                "asset_id": payload.get("asset_id") or "",
                "site_id": payload.get("site_id") or "",
                "line_id": payload.get("line_id") or "",
                "machine_id": payload.get("machine_id") or "",
                "client_id": payload.get("client_id") or "",
            },
        )
        agent = self.registry.get(route.selected_agent_id)
        adapter = self._adapter_for(route.selected_agent_id)
        window = adapter.build_window(
            readings,
            seq_len=int(payload.get("seq_len") or 16),
            timestamp=payload.get("timestamp"),
            inherited_rejections=rejected,
        )
        return route, agent, window, normalization_trace

    def _build_agent_window_preselected(self, payload: Mapping[str, Any], *, selected_agent_id: str):
        sensors_payload = payload.get("sensors") or payload.get("sensor_readings") or []
        if not isinstance(sensors_payload, list):
            sensors_payload = []
        readings, rejected = parse_sensor_readings(sensors_payload)
        readings, normalization_trace, normalization_rejections = self.normalizer.normalize_readings(
            readings,
            requested_agent_id=str(selected_agent_id or "").strip() or None,
            sector=str(payload.get("sector") or "").strip() or None,
            sector_hint=str(payload.get("sector_hint") or "").strip() or None,
            client_id=str(payload.get("client_id") or "").strip() or None,
            site_id=str(payload.get("site_id") or "").strip() or None,
            normalization_profile=str(payload.get("normalization_profile") or "").strip() or None,
        )
        rejected.extend(normalization_rejections)
        agent = self.registry.get(selected_agent_id)
        adapter = self._adapter_for(selected_agent_id)
        window = adapter.build_window(
            readings,
            seq_len=int(payload.get("seq_len") or 16),
            timestamp=payload.get("timestamp"),
            inherited_rejections=rejected,
        )
        return agent, window, normalization_trace

    def _adapter_for(self, agent_id: str) -> AgentSchemaAdapter:
        adapter = self._adapter_cache.get(agent_id)
        if adapter is not None:
            return adapter
        agent = self.registry.get(agent_id)
        adapter = AgentSchemaAdapter(agent.schema)
        self._adapter_cache[agent_id] = adapter
        return adapter

    @staticmethod
    def _normalization_outputs(normalization_trace: NormalizationTrace) -> dict[str, Any]:
        return {
            "normalization_summary": normalization_trace.summary_dict(),
            "normalization_trace": normalization_trace.trace_dict(),
            "normalized_sensor_count": int(normalization_trace.normalized_sensor_count),
            "unknown_sensor_count": int(normalization_trace.unknown_sensor_count),
            "normalization_review_items": list(normalization_trace.review_items),
            "unknown_sensor_names": list(normalization_trace.unknown_sensor_names),
            "ambiguous_sensor_names": list(normalization_trace.ambiguous_sensor_names),
        }
