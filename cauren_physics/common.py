from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cauren_core.contracts import AgentSchema, PhysicsEvidence, SensorWindow


def latest(window: SensorWindow, feature: str) -> float | None:
    for idx, metadata in enumerate(window.features):
        if metadata.name == feature and any(row[idx] for row in window.presence_mask):
            return float(window.matrix[-1][idx])
    return None


def build_relation(name: str, score: float, detail: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "name": name,
        "score": round(float(max(0.0, min(1.0, score))), 6),
        "detail": detail,
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
        outputs = {
            "physics_class": self.knowledge.physics_class,
            "equipment_families": list(self.knowledge.equipment_families),
            "material_systems": list(self.knowledge.material_systems),
            "governing_principles": list(self.knowledge.governing_principles),
            "failure_mechanisms": list(self.knowledge.failure_mechanisms),
            "relation_count": len(relations),
        }
        if subsystem_scores:
            outputs["subsystem_scores"] = {
                key: round(float(value), 6) for key, value in subsystem_scores.items()
            }
        if subsystem_notes:
            outputs["subsystem_notes"] = dict(subsystem_notes)
        return PhysicsEvidence(
            agent_id=schema.agent_id,
            risk_contribution=max(0.0, min(1.0, float(risk))),
            relations=tuple(relations),
            missing_features=tuple(missing_features),
            notes=notes,
            agent_outputs=outputs,
        )
