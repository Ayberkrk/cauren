from __future__ import annotations

from typing import Any, Mapping

from .contracts import CaurenDiagnosis, PhysicsEvidence


def render_physics_evidence(evidence: PhysicsEvidence | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(evidence, PhysicsEvidence):
        agent_id = evidence.agent_id
        risk_contribution = evidence.risk_contribution
        relations = list(evidence.relations)
        missing_features = list(evidence.missing_features)
        notes = list(evidence.notes)
    else:
        agent_id = str(evidence.get("agent_id") or "unknown-agent")
        risk_contribution = _as_float(evidence.get("risk_contribution"), 0.0)
        relations = [item for item in evidence.get("relations", []) if isinstance(item, dict)]
        missing_features = [str(item) for item in evidence.get("missing_features", [])]
        notes = [str(item) for item in evidence.get("notes", [])]

    ranked_relations = sorted(
        relations,
        key=lambda item: _as_float(item.get("score"), 0.0),
        reverse=True,
    )
    return {
        "renderer": "cauren_agent_evidence_v1",
        "agent_id": agent_id,
        "risk_contribution": round(float(risk_contribution), 6),
        "top_relations": ranked_relations[:5],
        "missing_features": missing_features,
        "notes": notes,
        "summary": _summary(agent_id, ranked_relations, missing_features),
        "status_taxonomy": "risk_score_only",
    }


def render_diagnosis_explanation(diagnosis: CaurenDiagnosis | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(diagnosis, CaurenDiagnosis):
        payload = diagnosis.to_dict()
    else:
        payload = dict(diagnosis)
    physics = payload.get("physics_evidence")
    if not isinstance(physics, dict):
        physics = {}
    return {
        "renderer": "cauren_xai_evidence_renderer_v1",
        "anomaly_type": str(payload.get("anomaly_type") or "unknown_pattern"),
        "anomaly_family": str(payload.get("anomaly_family") or "unknown_family"),
        "risk_score": round(_as_float(payload.get("risk_score"), 0.0), 6),
        "confidence": round(_as_float(payload.get("confidence"), 0.0), 6),
        "selected_agent": str(payload.get("selected_agent") or physics.get("agent_id") or "unknown-agent"),
        "physics_evidence": render_physics_evidence(physics),
        "agent_reasoning": str(payload.get("agent_reasoning") or ""),
        "recommended_actions": [str(item) for item in payload.get("recommended_actions", [])],
        "global_physics_bridge": False,
        "hard_limit_breach": None,
    }


def _summary(agent_id: str, relations: list[dict[str, Any]], missing_features: list[str]) -> str:
    if not relations:
        if missing_features:
            return f"{agent_id} produced limited evidence because required/optional features are missing."
        return f"{agent_id} produced no elevated relation evidence."
    top = relations[0]
    name = str(top.get("name") or "agent_relation")
    score = _as_float(top.get("score"), 0.0)
    return f"{agent_id} top relation is {name} with risk contribution {score:.3f}."


def _as_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return max(0.0, min(1.0, parsed))
