from __future__ import annotations

from cauren_core.contracts import AgentSchema, SensorWindow

from .common import DomainPhysics, PhysicsKnowledge, build_relation, latest, measurement_quality_penalty


class BridgePhysics(DomainPhysics):
    """Bridge-vertical physics: deliberately narrow (3 features) because
    that's what real, public, per-bridge ground truth actually covers
    (FHWA National Bridge Inventory condition ratings + USGS/FEMA hazard
    layers) -- see data/cauren_bridge/. Unlike CivilPhysics's 8-feature
    building schema, this is not a placeholder subset; it's scoped to
    what can be trained and validated against a real 5-year deterioration
    outcome (deck_drop_5yr).
    """

    knowledge = PhysicsKnowledge(
        physics_class="BridgePhysics",
        equipment_families=("bridges_and_culverts", "foundations_and_scour_zones"),
        material_systems=("reinforced_concrete_deck", "structural_steel_superstructure", "substructure_piers"),
        governing_principles=("structural_reliability", "geotechnical_stability", "hazard_exposure_management"),
        failure_mechanisms=("deck_deterioration", "scour_induced_foundation_instability", "seismic_or_flood_damage"),
    )

    def evaluate(self, *, window: SensorWindow, schema: AgentSchema, context: dict) -> object:
        structural_risk = latest(window, "structural_risk_score")
        ground_stability = latest(window, "ground_stability_score")
        natural_hazard = latest(window, "natural_hazard_score")

        missing = [
            name
            for name, value in {
                "structural_risk_score": structural_risk,
                "ground_stability_score": ground_stability,
                "natural_hazard_score": natural_hazard,
            }.items()
            if value is None
        ]

        relations = []
        subsystem_scores = {"structural_reliability": 0.0, "hazard_exposure": 0.0}

        if structural_risk is not None and ground_stability is not None:
            # Scour (foundation/soil undermining) compounds deck/superstructure
            # deterioration -- a bridge with both a poor condition rating and
            # a scour-critical foundation is a materially worse combination
            # than either alone.
            score = min(1.0, 0.6 * float(structural_risk) + 0.4 * float(ground_stability))
            subsystem_scores["structural_reliability"] = score
            relations.append(
                build_relation(
                    "structural_scour_coupling",
                    score,
                    "Deck/superstructure condition and scour-critical foundation status are combined into a single structural reliability signal.",
                    structural_risk_score=round(float(structural_risk), 6),
                    ground_stability_score=round(float(ground_stability), 6),
                )
            )
        elif structural_risk is not None or ground_stability is not None:
            solo = structural_risk if structural_risk is not None else ground_stability
            solo_name = "structural_risk_score" if structural_risk is not None else "ground_stability_score"
            score = 0.5 * float(solo)
            subsystem_scores["structural_reliability"] = score
            relations.append(
                build_relation(
                    "structural_partial_evidence",
                    score,
                    "Only one of condition rating / scour status is available; dampened as weak evidence.",
                    partial_evidence=True,
                    **{solo_name: round(float(solo), 6)},
                )
            )

        if natural_hazard is not None and structural_risk is not None:
            score = min(1.0, 0.55 * float(natural_hazard) + 0.45 * float(structural_risk))
            subsystem_scores["hazard_exposure"] = score
            relations.append(
                build_relation(
                    "hazard_structural_exposure",
                    score,
                    "Seismic/flood hazard exposure combined with existing structural condition.",
                    natural_hazard_score=round(float(natural_hazard), 6),
                    structural_risk_score=round(float(structural_risk), 6),
                )
            )
        elif natural_hazard is not None:
            score = 0.5 * float(natural_hazard)
            subsystem_scores["hazard_exposure"] = score
            relations.append(
                build_relation(
                    "hazard_partial_evidence",
                    score,
                    "Hazard exposure without structural context is treated as weak evidence.",
                    partial_evidence=True,
                    natural_hazard_score=round(float(natural_hazard), 6),
                )
            )

        quality = measurement_quality_penalty(window)
        if quality is not None:
            score, payload = quality
            relations.append(
                build_relation(
                    "measurement_quality_penalty",
                    score,
                    "Missing or low-quality records add a data-confidence bound to the bridge risk read.",
                    **payload,
                )
            )

        risk = max(subsystem_scores.values(), default=0.0)
        return self.evidence(
            schema=schema,
            risk=risk,
            relations=relations,
            missing_features=missing,
            notes=(
                "BridgePhysics combines FHWA condition ratings with USGS seismic and FEMA flood hazard exposure.",
                "Purpose is operational decision support; it does not produce automated enforcement or a final engineering judgment.",
            ),
            subsystem_scores=subsystem_scores,
            subsystem_notes={
                "structural_reliability": "Deck/superstructure/substructure condition combined with scour-critical foundation status.",
                "hazard_exposure": "Seismic and flood hazard exposure fused with existing structural condition.",
            },
        )
