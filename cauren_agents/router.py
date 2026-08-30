from __future__ import annotations

from typing import Any

from cauren_core.contracts import AgentCandidate, SectorScoreBreakdown, SensorReading

from .base import AgentRoute
from .registry import AgentRegistry

CIVIL_HINT_TOKENS = {
    "address",
    "building",
    "cadastre",
    "cadastral",
    "civil",
    "construction",
    "foundation",
    "geotechnical",
    "ground",
    "hazard",
    "infrastructure",
    "inspection",
    "occupancy",
    "parcel",
    "permit",
    "risk",
    "safety",
    "settlement",
    "soil",
    "structural",
    "tucbs",
}


class AgentRouter:
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self._default_agent = self.registry.all()[0]

    def route(
        self,
        *,
        readings: list[SensorReading],
        requested_agent_id: str | None,
        sector: str | None,
        site_context: dict[str, Any],
        ingress_tag: str | None,
        ingress_tags: list[str] | None = None,
        sector_hint: str | None = None,
        sector_hint_confidence: float | None = None,
        source_metadata: dict[str, Any] | None = None,
        identity_hints: dict[str, Any] | None = None,
    ) -> AgentRoute:
        agent = self._default_agent
        identity_hints = identity_hints or {}
        source_metadata = source_metadata or {}
        ingress_tags = ingress_tags or []

        tokens = set()
        for reading in readings:
            tokens.update(_tokenize(reading.name))
            tokens.update(_tokenize(reading.sensor_id))
        for value in site_context.values():
            tokens.update(_tokenize(value))
        for value in source_metadata.values():
            tokens.update(_tokenize(value))
        for value in identity_hints.values():
            tokens.update(_tokenize(value))
        tokens.update(_tokenize(sector))
        tokens.update(_tokenize(sector_hint))
        tokens.update(_tokenize(ingress_tag))
        for tag in ingress_tags:
            tokens.update(_tokenize(tag))
        matched = sorted(token for token in tokens if token in CIVIL_HINT_TOKENS)

        requested_match = str(requested_agent_id or '').strip() == agent.schema.agent_id
        explicit_sector_match = str(sector or '').strip().lower() == agent.schema.sector
        explicit_hint_match = str(sector_hint or '').strip().lower() in {agent.schema.sector, agent.schema.agent_id}

        model_score = min(1.0, 0.45 + 0.05 * len(matched)) if matched else 0.55
        metadata_prior = 0.15 if requested_match or explicit_sector_match else 0.0
        context_prior = 0.10 if explicit_hint_match else 0.0
        final_score = min(0.99, model_score + metadata_prior + context_prior)
        reasons = list(dict.fromkeys([f"civil_token:{token}" for token in matched]))
        if requested_match:
            reasons.append("requested_agent:cauren-civil")
        if explicit_sector_match:
            reasons.append("payload_sector:civil")
        if explicit_hint_match:
            reasons.append("sector_hint:civil")
        if not reasons:
            reasons.append("single_agent_civil_default")

        breakdown = SectorScoreBreakdown(
            agent_id=agent.schema.agent_id,
            sector=agent.schema.sector,
            model_evidence_score=float(model_score),
            metadata_prior_score=float(metadata_prior),
            context_prior_score=float(context_prior),
            penalty_score=0.0,
            final_sector_score=float(final_score),
            evidence_reasons=tuple(reasons),
            prior_reasons=tuple(reason for reason in reasons if reason.startswith(("requested_agent", "payload_sector", "sector_hint"))),
            penalty_reasons=(),
            ingress_tags=tuple(tag for tag in ingress_tags if str(tag).strip()),
            prior_conflict=False,
        )
        return AgentRoute(
            selected_agent_id=agent.schema.agent_id,
            candidates=(AgentCandidate(agent_id=agent.schema.agent_id, score=float(final_score), reason='; '.join(reasons)),),
            confidence=float(final_score),
            reason='; '.join(reasons),
            needs_context=False,
            ambiguity_reason='',
            sector_scores=(breakdown,),
            selected_sector_score=float(final_score),
            selected_sector_prior=float(metadata_prior + context_prior),
            selection_strategy='single_civil_agent',
            prior_conflict=False,
            prior_conflict_reason='',
            context_gate_status='ok',
            context_gate_reason='single civil agent registry active',
            context_required_fields=(),
        )


def _tokenize(value: Any) -> set[str]:
    if value is None:
        return set()
    text = str(value).strip().lower()
    if not text:
        return set()
    return {token for token in ''.join(ch if ch.isalnum() else ' ' for ch in text).split() if token}
