from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cauren_core.contracts import AgentSchema, PhysicsEvidence, SensorWindow


def latest(window: SensorWindow, feature: str) -> float | None:
    for idx, metadata in enumerate(window.features):
        if metadata.name != feature:
            continue
        # The final row can be an unobserved padding step. Select only a value
        # whose presence mask says it came from an actual reading.
        for row_idx in range(min(len(window.matrix), len(window.presence_mask)) - 1, -1, -1):
            if idx < len(window.presence_mask[row_idx]) and window.presence_mask[row_idx][idx]:
                return float(window.matrix[row_idx][idx])
    return None


def build_relation(name: str, score: float, detail: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "name": name,
        "score": round(float(max(0.0, min(1.0, score))), 6),
        "detail": detail,
        # These relations are transparent prototype heuristics, not validated
        # design checks or engineering conclusions. An agent can replace this
        # status when a rule has documented, reviewed validation evidence.
        "validation_status": "unvalidated_decision_support_heuristic",
        "risk_contribution_enabled": False,
        "references": [],
    }
    payload.update(extra)
    return payload


def measurement_quality_penalty(window: SensorWindow) -> tuple[float, dict[str, Any]] | None:
    present_qualities = [float(feature.quality) for feature in window.features if feature.source_sensor_ids]
    if not present_qualities:
        return None
    avg_quality = sum(present_qualities) / float(len(present_qualities))
    rejected_ratio = len(window.rejected_samples) / max(1, int(window.raw_sensor_count))
    quality_drop = min(1.0, max(0.0, 0.98 - avg_quality) / 0.08)
    rejected_penalty = min(1.0, max(0.0, rejected_ratio - 0.08) / 0.10)
    score = min(1.0, 0.75 * quality_drop + 0.25 * rejected_penalty)
    return score, {
        "avg_feature_quality": round(avg_quality, 6),
        "rejected_ratio": round(rejected_ratio, 6),
    }


@dataclass(frozen=True)
class PhysicsKnowledge:
    physics_class: str
    equipment_families: tuple[str, ...]
    material_systems: tuple[str, ...]
    governing_principles: tuple[str, ...]
    failure_mechanisms: tuple[str, ...]


class DomainPhysics:
    knowledge: PhysicsKnowledge

    def evidence(
        self,
        *,
        schema: AgentSchema,
        risk: float,
        relations: list[dict[str, Any]],
        missing_features: list[str],
        notes: tuple[str, ...],
        subsystem_scores: dict[str, float] | None = None,
        subsystem_notes: dict[str, str] | None = None,
    ) -> PhysicsEvidence:
        validated_risk_rules = [
            relation
            for relation in relations
            if relation.get("validation_status") == "validated"
            and relation.get("risk_contribution_enabled") is True
        ]
        proposed_risk = max(0.0, min(1.0, float(risk)))
        # Prototype relations are retained for diagnostics but cannot change
        # the active diagnosis risk until a rule has both reviewed validation
        # evidence and an explicit opt-in. No current agent rule meets that
        # bar, so the physics layer is fail-closed by default.
        active_risk = proposed_risk if validated_risk_rules else 0.0
        outputs = {
            "physics_class": self.knowledge.physics_class,
            "equipment_families": list(self.knowledge.equipment_families),
            "material_systems": list(self.knowledge.material_systems),
            "governing_principles": list(self.knowledge.governing_principles),
            "failure_mechanisms": list(self.knowledge.failure_mechanisms),
            "relation_count": len(relations),
            "interpretation_status": "non_authoritative_decision_support",
            "engineering_validation_status": "not_established",
            "proposed_heuristic_risk_contribution": round(proposed_risk, 6),
            "risk_contribution_status": (
                "validated_rules_enabled" if validated_risk_rules else "suppressed_unvalidated_rules"
            ),
        }
        if subsystem_scores:
            outputs["subsystem_scores"] = {
                key: round(float(value), 6) for key, value in subsystem_scores.items()
            }
        if subsystem_notes:
            outputs["subsystem_notes"] = dict(subsystem_notes)
        return PhysicsEvidence(
            agent_id=schema.agent_id,
            risk_contribution=active_risk,
            relations=tuple(relations),
            missing_features=tuple(missing_features),
            notes=notes,
            agent_outputs=outputs,
        )
