from __future__ import annotations

from collections import defaultdict
from typing import Any

from cauren_core.contracts import CoreOutput, PhysicsEvidence, SensorWindow


GROUP_RELATION_HINTS: dict[str, dict[str, tuple[str, ...]]] = {
    "cauren-civil": {
        "structural_risk": ("structural_load_path_consistency", "ground_response_stability", "hazard_exposure_balance"),
        "construction_governance": ("permit_readiness_alignment", "construction_progress_consistency"),
        "inspection_and_safety": ("inspection_severity_alignment", "occupancy_safety_envelope"),
        "infrastructure_readiness": ("utility_connection_readiness", "construction_dependency_balance"),
    },
}

AGENT_ANOMALY_RELATION_HINTS: dict[str, dict[str, tuple[str, ...]]] = {
    "cauren-civil": {
        "structural_risk_escalation": ("structural_load_path_consistency", "hazard_exposure_balance"),
        "ground_stability_deterioration": ("ground_response_stability",),
        "hazard_exposure_accumulation": ("hazard_exposure_balance",),
        "permit_readiness_gap": ("permit_readiness_alignment",),
        "construction_progress_stall": ("construction_progress_consistency",),
        "construction_compliance_misalignment": ("permit_readiness_alignment", "construction_progress_consistency"),
        "inspection_finding_escalation": ("inspection_severity_alignment",),
        "occupancy_safety_exposure": ("occupancy_safety_envelope",),
        "review_queue_trigger": ("inspection_severity_alignment", "occupancy_safety_envelope"),
        "utility_readiness_delay": ("utility_connection_readiness",),
        "infrastructure_dependency_gap": ("utility_connection_readiness", "construction_dependency_balance"),
    },
}

AGENT_ANOMALY_FEATURE_HINTS: dict[str, dict[str, tuple[str, ...]]] = {
    "cauren-civil": {
        "structural_risk_escalation": ("structural_risk_score", "natural_hazard_score"),
        "ground_stability_deterioration": ("ground_stability_score",),
        "hazard_exposure_accumulation": ("natural_hazard_score",),
        "permit_readiness_gap": ("permit_status_score",),
        "construction_progress_stall": ("construction_progress_pct",),
        "construction_compliance_misalignment": ("permit_status_score", "construction_progress_pct"),
        "inspection_finding_escalation": ("inspection_finding_score",),
        "occupancy_safety_exposure": ("occupancy_safety_score",),
        "review_queue_trigger": ("inspection_finding_score", "occupancy_safety_score"),
        "utility_readiness_delay": ("infrastructure_connection_score",),
        "infrastructure_dependency_gap": ("infrastructure_connection_score", "construction_progress_pct"),
    },
}

AGENT_ANOMALY_PRIORS: dict[str, dict[str, float]] = {
    "cauren-civil": {
        "structural_risk_escalation": 0.88,
        "ground_stability_deterioration": 0.9,
        "hazard_exposure_accumulation": 0.82,
        "permit_readiness_gap": 0.78,
        "construction_progress_stall": 0.8,
        "construction_compliance_misalignment": 0.79,
        "inspection_finding_escalation": 0.86,
        "occupancy_safety_exposure": 0.87,
        "review_queue_trigger": 0.84,
        "utility_readiness_delay": 0.77,
        "infrastructure_dependency_gap": 0.76,
    },
}


def _group_score_weights(agent_id: str, group: str) -> dict[str, float]:
    return {
        "core_pattern": 0.28,
        "feature_overlap": 0.25,
        "relation_hint": 0.27,
        "case_token_similarity": 0.14,
        "context_similarity": 0.06,
        "semantic_bonus": 0.0,
    }


def _semantic_group_bonus(agent_id: str, group: str, relation_score_map: dict[str, float]) -> float:
    return 0.0


def _anomaly_score_weights(agent_id: str, anomaly_id: str) -> dict[str, float]:
    return {
        "group_score": 0.22,
        "core_pattern": 0.14,
        "feature_overlap": 0.12,
        "case_similarity": 0.10,
        "relation_support": 0.24,
        "feature_support": 0.16,
        "prior_bonus": 0.08,
    }


def diagnose_taxonomy(
    *,
    taxonomy: tuple[dict[str, Any], ...],
    core_output: CoreOutput,
    physics_evidence: PhysicsEvidence,
    window: SensorWindow,
) -> dict[str, Any]:
    if not taxonomy:
        return {}

    present_features = _present_features(window)
    evidence_tokens = _evidence_tokens(core_output, physics_evidence, window)
    relation_score_map = _relation_score_map(physics_evidence)
    latest_feature_values = _latest_feature_values(window)
    agent_id = str(taxonomy[0].get("agent_id") or "")
    group_scores = _score_groups(
        taxonomy,
        core_output,
        physics_evidence,
        present_features,
        evidence_tokens,
        relation_score_map,
    )
    best_group = group_scores[0] if group_scores else {}
    runner_up_group = group_scores[1] if len(group_scores) > 1 else {}
    group_id = str(best_group.get("group") or "")
    group_items = [item for item in taxonomy if item.get("group") == group_id] or list(taxonomy)
    candidate_rows = [
        _score_anomaly(
            item,
            core_output,
            present_features,
            evidence_tokens,
            float(best_group.get("score") or 0.0),
            relation_score_map,
            agent_id,
            latest_feature_values,
        )
        for item in group_items
    ]
    candidate_rows.sort(key=lambda item: item["confidence"], reverse=True)
    selected = candidate_rows[0]
    runner_up = candidate_rows[1] if len(candidate_rows) > 1 else None
    gap = selected["confidence"] - float(runner_up["confidence"] if runner_up else 0.0)
    resolution = "exact"
    if len(candidate_rows) > 1 and gap < 0.08:
        resolution = "group_ambiguous"
    low_evidence_floor = 0.45
    if selected["confidence"] < low_evidence_floor:
        resolution = "low_evidence"

    top_candidates = [
        {
            "anomaly_id": item["anomaly_id"],
            "human_label_tr": item["human_label_tr"],
            "confidence": round(float(item["confidence"]), 6),
            "case_similarity": round(float(item["case_similarity"]), 6),
            "token_hits": item["token_hits"][:8],
        }
        for item in candidate_rows[:5]
    ]
    return {
        "anomaly_id": selected["anomaly_id"],
        "group": group_id,
        "confidence": round(float(selected["confidence"]), 6),
        "human_label_tr": selected["human_label_tr"],
        "matched_core_pattern": core_output.anomaly_type,
        "taxonomy_version": selected["taxonomy_version"],
        "resolution": resolution,
        "diagnostic_method": "rules_plus_physics_evidence_plus_case_similarity",
        "group_confidence": round(float(best_group.get("score") or 0.0), 6),
        "group_ambiguity_gap": round(
            float(best_group.get("score") or 0.0) - float(runner_up_group.get("score") or 0.0),
            6,
        ),
        "relation_support": round(float(selected.get("relation_support") or 0.0), 6),
        "feature_support": round(float(selected.get("feature_support") or 0.0), 6),
        "hint_matches": list(selected.get("hint_matches") or ()),
        "feature_hint_matches": list(selected.get("feature_hint_matches") or ()),
        "group_candidates": [
            {
                "group": item["group"],
                "confidence": round(float(item["score"]), 6),
                "score_breakdown": item["score_breakdown"],
            }
            for item in group_scores[:5]
        ],
        "top_candidates": top_candidates,
        "score_breakdown": selected["score_breakdown"],
        "ambiguity_gap": round(float(gap), 6),
    }


def _score_groups(
    taxonomy: tuple[dict[str, Any], ...],
    core_output: CoreOutput,
    physics_evidence: PhysicsEvidence,
    present_features: set[str],
    evidence_tokens: set[str],
    relation_score_map: dict[str, float],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in taxonomy:
        grouped[str(item.get("group") or "general")].append(item)
    rows: list[dict[str, Any]] = []
    agent_id = str(taxonomy[0].get("agent_id") or "")
    relation_names = tuple(str(item.get("name") or "") for item in physics_evidence.relations if isinstance(item, dict))
    for group, items in grouped.items():
        required_features = set().union(*(set(item.get("required_features") or ()) for item in items))
        core_patterns = set().union(*(set(item.get("core_patterns") or ()) for item in items))
        optional_context = set().union(*(set(item.get("optional_context") or ()) for item in items))
        pattern_score = max((float(core_output.pattern_scores.get(pattern, 0.0)) for pattern in core_patterns), default=0.0)
        feature_overlap = len(required_features & present_features) / float(max(1, len(required_features)))
        relation_score = _relation_hint_score(agent_id, group, relation_names, evidence_tokens, relation_score_map)
        token_score = _jaccard(_token_set([group, *required_features, *optional_context]), evidence_tokens)
        context_score = _jaccard(_token_set(optional_context), evidence_tokens)
        weights = _group_score_weights(agent_id, group)
        semantic_bonus = _semantic_group_bonus(agent_id, group, relation_score_map)
        score = min(
            1.0,
            (weights["core_pattern"] * pattern_score)
            + (weights["feature_overlap"] * feature_overlap)
            + (weights["relation_hint"] * relation_score)
            + (weights["case_token_similarity"] * token_score)
            + (weights["context_similarity"] * context_score)
            + (weights["semantic_bonus"] * semantic_bonus),
        )
        rows.append(
            {
                "group": group,
                "score": score,
                "score_breakdown": {
                    "core_pattern": round(pattern_score, 6),
                    "feature_overlap": round(feature_overlap, 6),
                    "relation_hint": round(relation_score, 6),
                    "case_token_similarity": round(token_score, 6),
                    "context_similarity": round(context_score, 6),
                    "semantic_bonus": round(semantic_bonus, 6),
                },
            }
        )
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows


def _score_anomaly(
    item: dict[str, Any],
    core_output: CoreOutput,
    present_features: set[str],
    evidence_tokens: set[str],
    group_score: float,
    relation_score_map: dict[str, float],
    agent_id: str,
    latest_feature_values: dict[str, float],
) -> dict[str, Any]:
    core_patterns = tuple(item.get("core_patterns") or ())
    required = set(item.get("required_features") or ())
    optional_context = tuple(item.get("optional_context") or ())
    anomaly_tokens = _token_set([item.get("anomaly_id"), item.get("group"), *optional_context, *required])
    pattern_score = max((float(core_output.pattern_scores.get(pattern, 0.0)) for pattern in core_patterns), default=0.0)
    feature_overlap = len(required & present_features) / float(max(1, len(required)))
    case_similarity = _jaccard(anomaly_tokens, evidence_tokens)
    hint_matches, relation_support = _anomaly_relation_support(agent_id, str(item.get("anomaly_id") or ""), relation_score_map)
    feature_hint_matches, feature_support = _anomaly_feature_support(agent_id, str(item.get("anomaly_id") or ""), latest_feature_values)
    anomaly_prior = float(AGENT_ANOMALY_PRIORS.get(agent_id, {}).get(str(item.get("anomaly_id") or ""), 0.5))
    direct_hits = sorted(anomaly_tokens & evidence_tokens)
    weights = _anomaly_score_weights(agent_id, str(item.get("anomaly_id") or ""))
    confidence = min(
        1.0,
        (weights["group_score"] * group_score)
        + (weights["core_pattern"] * pattern_score)
        + (weights["feature_overlap"] * feature_overlap)
        + (weights["case_similarity"] * case_similarity)
        + (weights["relation_support"] * relation_support)
        + (weights["feature_support"] * feature_support),
    )
    confidence = min(1.0, confidence + weights["prior_bonus"] * anomaly_prior)
    return {
        "anomaly_id": item.get("anomaly_id"),
        "group": item.get("group"),
        "taxonomy_version": item.get("taxonomy_version"),
        "human_label_tr": item.get("human_label_tr"),
        "confidence": confidence,
        "case_similarity": case_similarity,
        "token_hits": direct_hits,
        "relation_support": relation_support,
        "hint_matches": hint_matches,
        "feature_support": feature_support,
        "feature_hint_matches": feature_hint_matches,
        "score_breakdown": {
            "group_score": round(group_score, 6),
            "core_pattern": round(pattern_score, 6),
            "feature_overlap": round(feature_overlap, 6),
            "case_similarity": round(case_similarity, 6),
            "relation_support": round(relation_support, 6),
            "feature_support": round(feature_support, 6),
            "anomaly_prior": round(anomaly_prior, 6),
        },
    }


def _relation_hint_score(
    agent_id: str,
    group: str,
    relation_names: tuple[str, ...],
    evidence_tokens: set[str],
    relation_score_map: dict[str, float],
) -> float:
    hints = GROUP_RELATION_HINTS.get(agent_id, {}).get(group, ())
    if not hints:
        return _jaccard(_token_set([group]), evidence_tokens)
    hit = 0.0
    for hint in hints:
        hint_tokens = _token_set([hint])
        if hint in relation_names:
            hit += float(relation_score_map.get(hint, 0.0))
        else:
            hit += min(1.0, _jaccard(hint_tokens, evidence_tokens) * 2.0)
    return min(1.0, hit / float(max(1, len(hints))))


def _relation_score_map(physics_evidence: PhysicsEvidence) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in physics_evidence.relations:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        out[name] = max(0.0, min(1.0, float(item.get("score") or 0.0)))
    return out


def _anomaly_relation_support(agent_id: str, anomaly_id: str, relation_score_map: dict[str, float]) -> tuple[list[str], float]:
    hints = AGENT_ANOMALY_RELATION_HINTS.get(agent_id, {}).get(anomaly_id, ())
    if not hints:
        return [], 0.0
    matches = [hint for hint in hints if hint in relation_score_map]
    if not matches:
        return [], 0.0
    score = sum(float(relation_score_map.get(hint, 0.0)) for hint in matches) / float(len(hints))
    return matches, max(0.0, min(1.0, score))


def _latest_feature_values(window: SensorWindow) -> dict[str, float]:
    latest: dict[str, float] = {}
    for idx, feature in enumerate(window.features):
        present = False
        last = 0.0
        for row_idx, row in enumerate(window.matrix):
            if row_idx < len(window.presence_mask) and idx < len(window.presence_mask[row_idx]) and window.presence_mask[row_idx][idx]:
                present = True
                last = float(row[idx])
        if present:
            latest[feature.name] = last
    return latest


def _anomaly_feature_support(agent_id: str, anomaly_id: str, latest_feature_values: dict[str, float]) -> tuple[list[str], float]:
    hints = AGENT_ANOMALY_FEATURE_HINTS.get(agent_id, {}).get(anomaly_id, ())
    if not hints:
        return [], 0.0
    matches: list[str] = []
    values: list[float] = []
    for hint in hints:
        if hint not in latest_feature_values:
            continue
        normalized = _normalize_feature_hint_value(hint, float(latest_feature_values[hint]))
        matches.append(hint)
        values.append(normalized)
    if not values:
        return [], 0.0
    return matches, max(0.0, min(1.0, sum(values) / float(len(hints))))


def _normalize_feature_hint_value(feature: str, value: float) -> float:
    v = float(value)
    if feature == "case_surface_temp_gradient_c":
        return _scaled(v, 6.0, 14.0)
    if feature == "coolant_delta_t_c":
        return _scaled(v, 4.0, 8.0)
    if feature == "cooling_channel_outlet_temp_c":
        return _scaled(v, 35.0, 20.0)
    if feature == "battery_enclosure_strain":
        return _scaled(v, 180.0, 720.0)
    if feature == "battery_mount_strain":
        return _scaled(v, 120.0, 560.0)
    if feature == "internal_resistance_increase_pct":
        return _scaled(v, 8.0, 20.0)
    if feature == "triax_vibration_rms_10_2000hz":
        return _scaled(v, 0.15, 0.75)
    if feature == "battery_pack_pressure_delta":
        return _scaled(v, 2.0, 23.0)
    if feature == "motor_bracket_stress":
        return _scaled(v, 80.0, 220.0)
    if feature == "motor_mount_accel":
        return _scaled(v, 0.18, 0.82)
    if feature == "motor_mount_relative_displacement":
        return _scaled(v, 0.5, 5.5)
    if feature == "flange_bolt_preload_loss_pct":
        return _scaled(v, 4.0, 21.0)
    if feature == "subframe_torsion_proxy":
        return _scaled(v, 0.04, 0.26)
    if feature == "regen_torque_event_rate":
        return _scaled(v, 2.0, 18.0)
    if feature == "modal_frequency_shift_pct":
        return _scaled(v, 1.0, 11.0)
    if feature == "damping_ratio_shift_pct":
        return _scaled(v, 1.0, 9.0)
    if feature == "structural_wear_index":
        return _scaled(v, 0.12, 0.88)
    if feature == "front_axle_vertical_load_var":
        return _scaled(v, 0.05, 0.30)
    if feature == "rear_axle_vertical_load_var":
        return _scaled(v, 0.05, 0.30)
    if feature == "route_roughness_iso8608_class":
        return _scaled(v, 3.0, 4.0)
    if feature == "curb_impact_event_count":
        return _scaled(v, 1.0, 8.0)
    if feature == "mixed_surface_exposure_ratio":
        return _scaled(v, 0.15, 0.80)
    if feature == "payload_mass_estimate":
        return _scaled(v, 7000.0, 9000.0)
    if feature == "door_cycle_count":
        return _scaled(v, 420.0, 720.0)
    if feature == "pantograph_charge_event_count":
        return _scaled(v, 1.0, 8.0)
    if feature == "depot_charge_event_count":
        return _scaled(v, 1.0, 8.0)
    if feature == "second_life_suitability_score":
        return _scaled(1.0 - v, 0.0, 1.0)
    if feature == "mechanical_stress_history_index":
        return _scaled(v, 0.15, 0.80)
    if feature == "thermal_stress_history_index":
        return _scaled(v, 0.15, 0.80)
    if feature == "usage_phase_co2_kg":
        return _scaled(v, 40.0, 120.0)
    if feature == "battery_soce":
        return _scaled(0.90 - v, 0.0, 0.25)
    if feature == "battery_state_of_health":
        return _scaled(0.95 - v, 0.0, 0.25)
    if feature == "charge_discharge_cycle_count":
        return _scaled(v, 800.0, 3200.0)
    if feature == "negative_event_count":
        return _scaled(v, 1.0, 8.0)
    if feature == "battery_status_class":
        return 1.0 if int(v) >= 2 else 0.0
    if feature == "passport_traceability_score":
        return _scaled(1.0 - v, 0.0, 1.0)
    if feature == "article14_record_completeness":
        return _scaled(1.0 - v, 0.0, 1.0)
    if feature == "annex_iv_record_completeness":
        return _scaled(1.0 - v, 0.0, 1.0)
    if feature == "passport_event_linkage_score":
        return _scaled(1.0 - v, 0.0, 1.0)
    if feature == "pantograph_contact_resistance_mohm":
        return _scaled(v, 2.0, 5.0)
    if feature == "charge_power_peak_kw":
        return _scaled(v, 180.0, 220.0)
    if feature == "charge_session_duration_min":
        return _scaled(v, 8.0, 22.0)
    if feature == "charge_turnaround_gap_min":
        return _scaled(12.0 - v, 0.0, 12.0)
    if feature == "post_charge_cooldown_slope_c_per_min":
        return _scaled(v + 0.25, 0.0, 0.75)
    if feature == "fast_charge_share_ratio":
        return _scaled(v, 0.25, 0.75)
    if feature == "second_life_evidence_coverage_ratio":
        return _scaled(1.0 - v, 0.0, 1.0)
    if feature == "cell_delta_ocv_mv":
        return _scaled(v, 12.0, 20.0)
    return _scaled(v, 0.0, 1.0)


def _scaled(value: float, offset: float, span: float) -> float:
    if span <= 0.0:
        return 0.0
    return max(0.0, min(1.0, (float(value) - offset) / span))


def _present_features(window: SensorWindow) -> set[str]:
    return {
        feature.name
        for idx, feature in enumerate(window.features)
        if any(row[idx] for row in window.presence_mask)
    }


def _evidence_tokens(core_output: CoreOutput, physics_evidence: PhysicsEvidence, window: SensorWindow) -> set[str]:
    values: list[Any] = [
        core_output.anomaly_type,
        core_output.anomaly_family,
        *[feature.name for feature in window.features],
        *physics_evidence.missing_features,
        *physics_evidence.notes,
    ]
    for relation in physics_evidence.relations:
        if isinstance(relation, dict):
            values.extend([relation.get("name"), relation.get("detail")])
            values.extend(relation.keys())
    values.extend(physics_evidence.agent_outputs.keys())
    return _token_set(values)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / float(len(left | right))


def _token_set(values) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        for part in str(value or "").replace("-", "_").replace("/", "_").split("_"):
            part = "".join(ch for ch in part.strip().lower() if ch.isalnum())
            if part:
                tokens.add(part)
    return tokens
