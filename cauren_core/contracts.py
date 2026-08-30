from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class SensorReading:
    sensor_id: str
    name: str
    unit: str
    value: float
    timestamp: float | None = None
    quality: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizationRegistryEntry:
    canonical_feature: str
    aliases: tuple[str, ...] = ()
    agent_scope: tuple[str, ...] = ()
    sector_scope: tuple[str, ...] = ()
    unit_hints: Mapping[str, str] = field(default_factory=dict)
    client_scope: tuple[str, ...] = ()
    priority: int = 0
    deprecated_aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class NormalizationDecision:
    raw_name: str
    normalized_name: str | None
    normalization_source: str
    normalization_confidence: float
    normalization_reason: str
    normalization_review_required: bool = False
    candidate_features: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_name": self.raw_name,
            "normalized_name": self.normalized_name,
            "normalization_source": self.normalization_source,
            "normalization_confidence": round(float(self.normalization_confidence), 6),
            "normalization_reason": self.normalization_reason,
            "normalization_review_required": bool(self.normalization_review_required),
            "candidate_features": list(self.candidate_features),
        }


@dataclass(frozen=True)
class NormalizedSensorReading:
    reading: SensorReading
    decision: NormalizationDecision


@dataclass(frozen=True)
class NormalizationTrace:
    decisions: tuple[NormalizationDecision, ...]
    normalized_sensor_count: int = 0
    unknown_sensor_count: int = 0
    ambiguous_sensor_count: int = 0
    unknown_sensor_names: tuple[str, ...] = ()
    ambiguous_sensor_names: tuple[str, ...] = ()
    review_items: tuple[dict[str, Any], ...] = ()

    def summary_dict(self) -> dict[str, Any]:
        return {
            "normalized_sensor_count": int(self.normalized_sensor_count),
            "unknown_sensor_count": int(self.unknown_sensor_count),
            "ambiguous_sensor_count": int(self.ambiguous_sensor_count),
            "unknown_sensor_names": list(self.unknown_sensor_names),
            "ambiguous_sensor_names": list(self.ambiguous_sensor_names),
            "review_item_count": len(self.review_items),
        }

    def trace_dict(self) -> list[dict[str, Any]]:
        return [decision.to_dict() for decision in self.decisions]


@dataclass(frozen=True)
class FeatureMetadata:
    name: str
    unit: str
    source_sensor_ids: tuple[str, ...] = ()
    quality: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SensorWindow:
    matrix: tuple[tuple[float, ...], ...]
    features: tuple[FeatureMetadata, ...]
    presence_mask: tuple[tuple[bool, ...], ...]
    timestamps: tuple[float, ...] = ()
    rejected_samples: tuple[dict[str, Any], ...] = ()
    raw_sensor_count: int = 0
    # Number of genuinely distinct observed time steps before the adapter
    # padded the window out to seq_len (e.g. by repeating a single
    # snapshot reading). 0 means "not reported" (older/other producers of
    # SensorWindow); consumers should treat that as "unknown" rather than
    # "zero real samples". Lets the core avoid reporting temporal-shape
    # anomalies (flatline, oscillation, drift, ...) that a single
    # snapshot can't actually support evidence for.
    observed_step_count: int = 0


@dataclass(frozen=True)
class AgentSchema:
    agent_id: str
    sector: str
    display_name: str
    required_features: tuple[str, ...]
    optional_features: tuple[str, ...] = ()
    aliases: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)
    feature_metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    version: str = "v1"

    @property
    def feature_order(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.required_features, *self.optional_features)))


@dataclass(frozen=True)
class CoreOutput:
    anomaly_type: str
    anomaly_family: str
    risk_score: float
    confidence: float
    pattern_scores: dict[str, float]
    reconstruction_error: float
    drift_score: float
    spike_score: float
    oscillation_score: float
    calibrated_matrix: tuple[tuple[float, ...], ...]
    raw_risk_score: float = 0.0
    calibrated_risk_score: float = 0.0
    backbone_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhysicsEvidence:
    agent_id: str
    risk_contribution: float
    relations: tuple[dict[str, Any], ...] = ()
    missing_features: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    agent_outputs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentCandidate:
    agent_id: str
    score: float
    reason: str


@dataclass(frozen=True)
class SectorPrior:
    agent_id: str
    score: float
    confidence: float
    provenance: tuple[str, ...] = ()
    ingress_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SectorScoreBreakdown:
    agent_id: str
    sector: str
    model_evidence_score: float
    metadata_prior_score: float
    context_prior_score: float
    penalty_score: float
    final_sector_score: float
    evidence_reasons: tuple[str, ...] = ()
    prior_reasons: tuple[str, ...] = ()
    penalty_reasons: tuple[str, ...] = ()
    ingress_tags: tuple[str, ...] = ()
    prior_conflict: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "sector": self.sector,
            "model_evidence_score": round(float(self.model_evidence_score), 6),
            "metadata_prior_score": round(float(self.metadata_prior_score), 6),
            "context_prior_score": round(float(self.context_prior_score), 6),
            "penalty_score": round(float(self.penalty_score), 6),
            "final_sector_score": round(float(self.final_sector_score), 6),
            "evidence_reasons": list(self.evidence_reasons),
            "prior_reasons": list(self.prior_reasons),
            "penalty_reasons": list(self.penalty_reasons),
            "ingress_tags": list(self.ingress_tags),
            "prior_conflict": bool(self.prior_conflict),
        }


@dataclass(frozen=True)
class SectorFusionDecision:
    selected_agent_id: str
    selected_sector: str
    selected_score: float
    selected_prior: float
    selection_strategy: str
    needs_context: bool = False
    prior_conflict: bool = False
    conflict_reason: str = ""


@dataclass(frozen=True)
class CaurenDiagnosis:
    anomaly_type: str
    anomaly_family: str
    risk_score: float
    raw_risk_score: float
    calibrated_risk_score: float
    confidence: float
    selected_agent: str
    candidate_agents: tuple[AgentCandidate, ...]
    physics_evidence: PhysicsEvidence
    agent_reasoning: str
    recommended_actions: tuple[str, ...]
    core_output: CoreOutput
    feature_names: tuple[str, ...]
    rejected_samples: tuple[dict[str, Any], ...] = ()
    agent_outputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_type": self.anomaly_type,
            "anomaly_family": self.anomaly_family,
            "risk_score": round(float(self.risk_score), 6),
            "raw_risk_score": round(float(self.raw_risk_score), 6),
            "calibrated_risk_score": round(float(self.calibrated_risk_score), 6),
            "confidence": round(float(self.confidence), 6),
            "selected_agent": self.selected_agent,
            "candidate_agents": [
                {"agent_id": c.agent_id, "score": round(float(c.score), 6), "reason": c.reason}
                for c in self.candidate_agents
            ],
            "physics_evidence": {
                "agent_id": self.physics_evidence.agent_id,
                "risk_contribution": round(float(self.physics_evidence.risk_contribution), 6),
                "relations": list(self.physics_evidence.relations),
                "missing_features": list(self.physics_evidence.missing_features),
                "notes": list(self.physics_evidence.notes),
                "agent_outputs": dict(self.physics_evidence.agent_outputs),
            },
            "agent_reasoning": self.agent_reasoning,
            "recommended_actions": list(self.recommended_actions),
            "core": {
                "anomaly_type": self.core_output.anomaly_type,
                "anomaly_family": self.core_output.anomaly_family,
                "risk_score": round(float(self.core_output.risk_score), 6),
                "raw_risk_score": round(float(self.core_output.raw_risk_score), 6),
                "calibrated_risk_score": round(float(self.core_output.calibrated_risk_score), 6),
                "confidence": round(float(self.core_output.confidence), 6),
                "pattern_scores": {
                    k: round(float(v), 6) for k, v in self.core_output.pattern_scores.items()
                },
                "reconstruction_error": round(float(self.core_output.reconstruction_error), 6),
                "drift_score": round(float(self.core_output.drift_score), 6),
                "spike_score": round(float(self.core_output.spike_score), 6),
                "oscillation_score": round(float(self.core_output.oscillation_score), 6),
                "backbone_metadata": dict(self.core_output.backbone_metadata),
            },
            "feature_names": list(self.feature_names),
            "rejected_samples": list(self.rejected_samples),
            **dict(self.agent_outputs),
        }
