from __future__ import annotations

from cauren_core.contracts import AgentSchema, SensorWindow

from .common import DomainPhysics, PhysicsKnowledge, build_relation, latest, measurement_quality_penalty


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
        structural_risk = latest(window, "structural_risk_score")
        inspection_score = latest(window, "inspection_finding_score")
        permit_score = latest(window, "permit_status_score")
        progress_pct = latest(window, "construction_progress_pct")
        infrastructure_score = latest(window, "infrastructure_connection_score")
        natural_hazard = latest(window, "natural_hazard_score")
        occupancy_safety = latest(window, "occupancy_safety_score")
        ground_stability = latest(window, "ground_stability_score")

        missing = [
            name
            for name, value in {
                "structural_risk_score": structural_risk,
                "inspection_finding_score": inspection_score,
                "permit_status_score": permit_score,
                "construction_progress_pct": progress_pct,
                "infrastructure_connection_score": infrastructure_score,
                "natural_hazard_score": natural_hazard,
                "occupancy_safety_score": occupancy_safety,
                "ground_stability_score": ground_stability,
            }.items()
            if value is None
        ]

        relations = []
        subsystem_scores = {
            "structural_reliability": 0.0,
            "construction_readiness": 0.0,
            "hazard_exposure": 0.0,
            "occupancy_and_review": 0.0,
        }

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

        if permit_score is not None and progress_pct is not None and infrastructure_score is not None:
            normalized_progress = max(0.0, min(1.0, float(progress_pct) / 100.0))
            delay_gap = max(0.0, normalized_progress - min(float(permit_score), float(infrastructure_score)))
            score = min(1.0, delay_gap / 0.35)
            subsystem_scores["construction_readiness"] = max(subsystem_scores["construction_readiness"], score)
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

        # Every relation above needs its *whole* feature pair/triple to
        # fire, so a real request that only supplies some of the 8 scores
        # (the common case -- most callers won't have all of them) gets
        # exactly zero physics signal even when the features it does have
        # are informative on their own. Fall back to a dampened
        # single-feature reading for any subsystem that a full relation
        # didn't already cover, so partial-but-real evidence isn't
        # discarded outright. Dampened (0.5x) and explicitly flagged as
        # partial so it can never out-rank a fully-corroborated relation.
        if subsystem_scores["structural_reliability"] == 0.0:
            solo = structural_risk if structural_risk is not None else ground_stability
            solo_name = "structural_risk_score" if structural_risk is not None else "ground_stability_score"
            if solo is not None:
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

        if subsystem_scores["construction_readiness"] == 0.0:
            available = {
                "permit_status_score": permit_score,
                "construction_progress_pct": (progress_pct / 100.0) if progress_pct is not None else None,
                "infrastructure_connection_score": infrastructure_score,
            }
            present = {name: value for name, value in available.items() if value is not None}
            if present:
                score = 0.5 * (sum(present.values()) / float(len(present)))
                subsystem_scores["construction_readiness"] = score
                relations.append(
                    build_relation(
                        "construction_readiness_partial_evidence",
                        score,
                        "Insaat hazirligi icin izin/ilerleme/altyapi uclusunun tamami yok; mevcut gostergelerle zayif kanit uretildi.",
                        partial_evidence=True,
                        **{name: round(float(value), 6) for name, value in present.items()},
                    )
                )

        if subsystem_scores["hazard_exposure"] == 0.0 and natural_hazard is not None:
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

        if subsystem_scores["occupancy_and_review"] == 0.0:
            solo = inspection_score if inspection_score is not None else occupancy_safety
            solo_name = "inspection_finding_score" if inspection_score is not None else "occupancy_safety_score"
            if solo is not None:
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
