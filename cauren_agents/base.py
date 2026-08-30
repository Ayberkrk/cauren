from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cauren_agents.diagnostics import diagnose_taxonomy, _normalize_feature_hint_value
from cauren_agents.taxonomy_loader import load_agent_taxonomy
from cauren_core.contracts import (
    AgentCandidate,
    AgentSchema,
    CaurenDiagnosis,
    CoreOutput,
    PhysicsEvidence,
    SectorScoreBreakdown,
    SensorWindow,
)
from cauren_core.risk_calibration import calibrated_combined_risk, sector_threshold


@dataclass(frozen=True)
class AgentRoute:
    selected_agent_id: str
    candidates: tuple[AgentCandidate, ...]
    confidence: float
    reason: str
    needs_context: bool = False
    ambiguity_reason: str = ""
    sector_scores: tuple[SectorScoreBreakdown, ...] = ()
    selected_sector_score: float = 0.0
    selected_sector_prior: float = 0.0
    selection_strategy: str = "router"
    prior_conflict: bool = False
    prior_conflict_reason: str = ""
    context_gate_status: str = "ok"
    context_gate_reason: str = ""
    context_required_fields: tuple[str, ...] = ()


class AgentPhysics(Protocol):
    def evaluate(self, *, window: SensorWindow, schema: AgentSchema, context: dict) -> PhysicsEvidence:
        ...


class SectorAgent:
    schema: AgentSchema
    physics: AgentPhysics

    def __init__(self, *, schema: AgentSchema, physics: AgentPhysics):
        self.schema = schema
        self.physics = physics

    def compose_result(
        self,
        *,
        core_output: CoreOutput,
        physics_evidence: PhysicsEvidence,
        route: AgentRoute,
        window: SensorWindow,
    ) -> CaurenDiagnosis:
        raw_combined_risk, calibrated_risk = calibrated_combined_risk(
            agent_id=self.schema.agent_id,
            core_risk=float(core_output.risk_score),
            physics_risk=float(physics_evidence.risk_contribution),
            anomaly_type=core_output.anomaly_type,
            pattern_scores=core_output.pattern_scores,
        )
        taxonomy = load_agent_taxonomy(self.schema.agent_id)
        taxonomy_match = _match_taxonomy_diagnosis(taxonomy, core_output, physics_evidence, window)
        selected_family = str(core_output.anomaly_family)
        confidence = min(1.0, 0.65 * float(core_output.confidence) + 0.35 * float(route.confidence))
        reasoning = self._reasoning(core_output, physics_evidence)
        actions = self._actions(core_output, physics_evidence)
        return CaurenDiagnosis(
            anomaly_type=core_output.anomaly_type,
            anomaly_family=selected_family,
            risk_score=calibrated_risk,
            raw_risk_score=raw_combined_risk,
            calibrated_risk_score=calibrated_risk,
            confidence=confidence,
            selected_agent=self.schema.agent_id,
            candidate_agents=route.candidates,
            physics_evidence=physics_evidence,
            agent_reasoning=reasoning,
            recommended_actions=actions,
            core_output=core_output,
            feature_names=tuple(feature.name for feature in window.features),
            rejected_samples=window.rejected_samples,
            agent_outputs={
                **dict(physics_evidence.agent_outputs),
                **self._agent_outputs(
                    core_output,
                    physics_evidence,
                    route,
                    window,
                    taxonomy_match=taxonomy_match,
                ),
            },
        )

    def _reasoning(self, core_output: CoreOutput, physics_evidence: PhysicsEvidence) -> str:
        relation_count = len(physics_evidence.relations)
        return (
            f"{self.schema.display_name} agent selected. Core pattern={core_output.anomaly_type} "
            f"family={core_output.anomaly_family}; agent physics produced {relation_count} relation signals."
        )

    def _actions(self, core_output: CoreOutput, physics_evidence: PhysicsEvidence) -> tuple[str, ...]:
        if core_output.anomaly_type == "nominal_variation" and physics_evidence.risk_contribution < 0.1:
            return ("Continue passive monitoring with this agent schema.",)
        actions = ["Review agent-specific physics evidence before intervention."]
        if physics_evidence.missing_features:
            actions.append("Improve sensor coverage for missing agent features.")
        if physics_evidence.relations:
            actions.append("Inspect the highest-risk relation in physics_evidence.relations.")
        return tuple(actions)

    def _agent_outputs(
        self,
        core_output: CoreOutput,
        physics_evidence: PhysicsEvidence,
        route: AgentRoute,
        window: SensorWindow,
        *,
        taxonomy_match: dict | None = None,
    ) -> dict:
        taxonomy = load_agent_taxonomy(self.schema.agent_id)
        version = str(taxonomy[0].get("taxonomy_version") or "") if taxonomy else ""
        resolved_taxonomy_match = taxonomy_match if taxonomy_match is not None else _match_taxonomy_diagnosis(taxonomy, core_output, physics_evidence, window)
        outputs = {
            "route_confidence": round(float(route.confidence), 6),
            "route_reason": route.reason,
            "route_needs_context": bool(route.needs_context),
            "route_ambiguity_reason": route.ambiguity_reason,
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
            "sector_risk_threshold": round(float(sector_threshold(self.schema.agent_id)), 6),
            "agent_taxonomy_count": len(taxonomy),
            "agent_taxonomy_version": version,
            "agent_taxonomy_diagnosis": resolved_taxonomy_match,
        }
        return outputs


def series(window: SensorWindow, feature: str) -> list[float] | None:
    for idx, metadata in enumerate(window.features):
        if metadata.name == feature and any(row[idx] for row in window.presence_mask):
            return [float(row[idx]) for row in window.matrix]
    return None


def latest(window: SensorWindow, feature: str) -> float | None:
    values = series(window, feature)
    if values is None or len(values) == 0:
        return None
    return float(values[-1])


def relation(name: str, score: float, detail: str, **extra) -> dict:
    payload = {"name": name, "score": round(float(max(0.0, min(1.0, score))), 6), "detail": detail}
    payload.update(extra)
    return payload


def _match_taxonomy_diagnosis(
    taxonomy: tuple[dict, ...],
    core_output: CoreOutput,
    physics_evidence: PhysicsEvidence,
    window: SensorWindow,
) -> dict:
    return diagnose_taxonomy(
        taxonomy=taxonomy,
        core_output=core_output,
        physics_evidence=physics_evidence,
        window=window,
    )


def _window_latest_feature_values(window: SensorWindow) -> dict[str, float]:
    values: dict[str, float] = {}
    for feature in window.features:
        value = latest(window, feature.name)
        if value is not None:
            values[feature.name] = float(value)
    return values

