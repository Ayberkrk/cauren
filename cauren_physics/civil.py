from __future__ import annotations

import math

from cauren_core.contracts import AgentSchema, SensorWindow

from .common import DomainPhysics, PhysicsKnowledge, build_relation, latest, measurement_quality_penalty


def _usable_ratio(value: float | None) -> float | None:
    """Return a ratio feature only if it is finite and inside [0, 1].

    Quality control already reports out-of-range values as
    `value_out_of_range`; scoring them here would let a value the pipeline
    knows is invalid push a subsystem below zero or saturate it at 1.0, so
    the physics relations treat such a value as unavailable instead.
    """
    if value is None or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return None
    return value


def _finite_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class CivilPhysics(DomainPhysics):
    knowledge = PhysicsKnowledge(
        physics_class="CivilPhysics",
        equipment_families=(
            "buildings_and_superstructures",
            "foundations_and_subgrades",
            "construction_sites",
            "utility_interfaces",
            "inspection_and_permit_workflows",
        ),
        material_systems=(
            "reinforced_concrete",
            "structural_steel",
            "masonry_and_envelope_systems",
            "soil_and_geotechnical_media",
            "utility_connection_assets",
        ),
        governing_principles=(
            "structural_reliability",
            "construction_sequence_readiness",
            "geotechnical_stability",
            "hazard_exposure_management",
            "inspection_driven_decision_support",
        ),
        failure_mechanisms=(
            "structural_capacity_erosion",
            "foundation_instability",
            "construction_stall",
            "permit_compliance_gap",
            "utility_readiness_delay",
        ),
    )

    def evaluate(self, *, window: SensorWindow, schema: AgentSchema, context: dict) -> object:
        raw = {
            name: latest(window, name)
            for name in (
                "structural_risk_score",
                "inspection_finding_score",
                "permit_status_score",
                "construction_progress_pct",
                "infrastructure_connection_score",
                "natural_hazard_score",
                "occupancy_safety_score",
                "ground_stability_score",
            )
        }
        structural_risk = _usable_ratio(raw["structural_risk_score"])
        inspection_score = _usable_ratio(raw["inspection_finding_score"])
        permit_score = _usable_ratio(raw["permit_status_score"])
        progress_pct = _finite_or_none(raw["construction_progress_pct"])
        infrastructure_score = _usable_ratio(raw["infrastructure_connection_score"])
        natural_hazard = _usable_ratio(raw["natural_hazard_score"])
        occupancy_safety = _usable_ratio(raw["occupancy_safety_score"])
        ground_stability = _usable_ratio(raw["ground_stability_score"])

        # Missing means "not supplied". A supplied but unusable value is not
        # listed here; quality control reports it separately.
        missing = [name for name, value in raw.items() if value is None]

        relations = []
        subsystem_scores = {
            "structural_reliability": 0.0,
            "construction_readiness": 0.0,
            "hazard_exposure": 0.0,
            "occupancy_and_review": 0.0,
        }

        # Each subsystem's full relation needs its *whole* feature pair/triple,
        # so a real request that only supplies some of the 8 scores (the
        # common case -- most callers won't have all of them) would get
        # exactly zero physics signal even when the features it does have are
        # informative on their own. The elif branches fall back to a dampened
        # reading in that case, so partial-but-real evidence isn't discarded
        # outright. Dampened (0.5x) and explicitly flagged as partial so it
        # can never out-rank a fully-corroborated relation. A fallback only
        # runs when its full relation could not: a full relation that scored
        # 0.0 is a real "no risk" result, not missing evidence.
        if structural_risk is not None and ground_stability is not None:
            score = min(1.0, 0.6 * float(structural_risk) + 0.4 * float(ground_stability))
            subsystem_scores["structural_reliability"] = score
            relations.append(
                build_relation(
                    "structural_ground_coupling",
                    score,
                    "Yapisal risk ile zemin stabilitesi birlikte okunarak bina-butun sistem riski yorumlanir.",
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
                    "Yapisal guvenilirlik icin sadece tek bir gosterge mevcut; zayif kanit olarak dampen edilmis skor kullanildi.",
                    partial_evidence=True,
                    **{solo_name: round(float(solo), 6)},
                )
            )

        readiness_scores = {
            name: float(value)
            for name, value in {
                "permit_status_score": permit_score,
                "infrastructure_connection_score": infrastructure_score,
            }.items()
            if value is not None
        }
        if permit_score is not None and progress_pct is not None and infrastructure_score is not None:
            normalized_progress = max(0.0, min(1.0, float(progress_pct) / 100.0))
            delay_gap = max(0.0, normalized_progress - min(float(permit_score), float(infrastructure_score)))
            score = min(1.0, delay_gap / 0.35)
            subsystem_scores["construction_readiness"] = score
            relations.append(
                build_relation(
                    "construction_readiness_gap",
                    score,
                    "Ilerleme seviyesi izin ve altyapi hazirliginin onune gectiginde saha-risk uyumsuzlugu olusur.",
                    normalized_progress=round(normalized_progress, 6),
                    permit_status_score=round(float(permit_score), 6),
                    infrastructure_connection_score=round(float(infrastructure_score), 6),
                )
            )
        elif progress_pct is not None and readiness_scores:
            # Same progress-ahead-of-readiness gap as the full relation, read
            # against the one readiness indicator that is present. Progress
            # alone, or permit/infrastructure without progress, says nothing
            # about a gap, so those leave the subsystem at zero.
            normalized_progress = max(0.0, min(1.0, float(progress_pct) / 100.0))
            delay_gap = max(0.0, normalized_progress - min(readiness_scores.values()))
            score = 0.5 * min(1.0, delay_gap / 0.35)
            subsystem_scores["construction_readiness"] = score
            relations.append(
                build_relation(
                    "construction_readiness_partial_evidence",
                    score,
                    "Insaat hazirligi icin izin/ilerleme/altyapi uclusunun tamami yok; mevcut gostergelerle zayif kanit uretildi.",
                    partial_evidence=True,
                    normalized_progress=round(normalized_progress, 6),
                    **{name: round(value, 6) for name, value in readiness_scores.items()},
                )
            )

        if natural_hazard is not None and structural_risk is not None:
            score = min(1.0, 0.55 * float(natural_hazard) + 0.45 * float(structural_risk))
            subsystem_scores["hazard_exposure"] = score
            relations.append(
                build_relation(
                    "hazard_structural_exposure",
                    score,
                    "Dogal tehlike maruziyeti, mevcut yapisal risk seviyesiyle birlestirilerek karar destegi sinyali uretilir.",
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
                    "Yapisal risk baglami olmadan dogal tehlike maruziyeti tek basina zayif kanit olarak degerlendirildi.",
                    partial_evidence=True,
                    natural_hazard_score=round(float(natural_hazard), 6),
                )
            )

        if inspection_score is not None and occupancy_safety is not None:
            score = min(1.0, 0.5 * float(inspection_score) + 0.5 * float(occupancy_safety))
            subsystem_scores["occupancy_and_review"] = score
            relations.append(
                build_relation(
                    "inspection_occupancy_safety_alignment",
                    score,
                    "Saha bulgulari ile kullanim guvenligi birlikte okunarak inceleme-onceligi uretilir.",
                    inspection_finding_score=round(float(inspection_score), 6),
                    occupancy_safety_score=round(float(occupancy_safety), 6),
                )
            )
        elif inspection_score is not None or occupancy_safety is not None:
            solo = inspection_score if inspection_score is not None else occupancy_safety
            solo_name = "inspection_finding_score" if inspection_score is not None else "occupancy_safety_score"
            score = 0.5 * float(solo)
            subsystem_scores["occupancy_and_review"] = score
            relations.append(
                build_relation(
                    "occupancy_review_partial_evidence",
                    score,
                    "Inceleme-kullanim guvenligi icin sadece tek bir gosterge mevcut; zayif kanit olarak dampen edilmis skor kullanildi.",
                    partial_evidence=True,
                    **{solo_name: round(float(solo), 6)},
                )
            )

        quality = measurement_quality_penalty(window)
        if quality is not None:
            score, payload = quality
            relations.append(
                build_relation(
                    "measurement_quality_penalty",
                    score,
                    "Eksik veya dusuk kaliteli kayitlar civil risk yorumuna veri-guven siniri ekler.",
                    **payload,
                )
            )

        # Optional operational modal analysis (OMA) input: a caller that has
        # already run cauren_physics.oma.compare_to_baseline on the
        # structure's vibration data can pass its result dict here. A
        # dropping natural frequency versus baseline is a standard
        # structural-health early-warning sign (stiffness loss), so it
        # folds into structural_reliability rather than sitting off to the
        # side as an unrelated signal.
        oma_drift = context.get("oma_frequency_drift") if isinstance(context, dict) else None
        max_drop_pct = (
            _finite_or_none(oma_drift.get("max_drop_pct")) if isinstance(oma_drift, dict) else None
        )
        # A NaN would otherwise pass through max(0, min(1, x)) as 1.0, the
        # strongest possible evidence, so a non-finite or non-numeric drop is
        # ignored here. The API rejects such input before it gets this far.
        if isinstance(oma_drift, dict) and max_drop_pct is not None:
            severity = str(oma_drift.get("overall_severity") or "none")
            # 15% frequency drop maps to full-strength evidence; below that,
            # scale down linearly so a small drop reads as mild concern.
            score = max(0.0, min(1.0, max_drop_pct / 15.0))
            if score > 0.0:
                subsystem_scores["structural_reliability"] = max(
                    subsystem_scores["structural_reliability"], score
                )
                relations.append(
                    build_relation(
                        "modal_frequency_drift",
                        score,
                        "Yapinin dogal frekansinda taban degere gore dusus, olasi rijitlik kaybina isaret eder.",
                        max_drop_pct=round(max_drop_pct, 6),
                        severity=severity,
                    )
                )

        risk = max(subsystem_scores.values(), default=0.0)
        return self.evidence(
            schema=schema,
            risk=risk,
            relations=relations,
            missing_features=missing,
            notes=(
                "CivilPhysics bina, izin, saha incelemesi, altyapi ve dogal tehlike sinyallerini birlestirir.",
                "Amac operasyonel karar destegi saglamaktir; otomatik yaptirim veya nihai muhendislik karari uretmez.",
            ),
            subsystem_scores=subsystem_scores,
            subsystem_notes={
                "structural_reliability": "Yapisal risk ile zemin etkisi birlikte degerlendirilir.",
                "construction_readiness": "Ilerleme, izin ve altyapi olgunlugu arasindaki uyum izlenir.",
                "hazard_exposure": "Dogal tehlike skoru yapi riskiyle kaynastirilir.",
                "occupancy_and_review": "Inceleme bulgulari ile kullanim guvenligi review onceligine donusturulur.",
            },
        )
