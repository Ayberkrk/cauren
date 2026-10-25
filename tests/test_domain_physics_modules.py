from cauren_agents.civil.agent import build_civil_agent
from cauren_core.adapter import AgentSchemaAdapter, parse_sensor_readings
from cauren_core.contracts import FeatureMetadata, SensorWindow
from cauren_physics.common import build_relation, latest



def test_civil_agent_emits_structural_and_readiness_relations():
    agent = build_civil_agent()
    readings, rejected = parse_sensor_readings(
        [
            {"sensor_id": "r1", "name": "structural_risk_score", "unit": "ratio", "value": 0.81, "timestamp": 1.0},
            {"sensor_id": "r2", "name": "inspection_finding_score", "unit": "ratio", "value": 0.78, "timestamp": 1.0},
            {"sensor_id": "r3", "name": "permit_status_score", "unit": "ratio", "value": 0.42, "timestamp": 1.0},
            {"sensor_id": "r4", "name": "construction_progress_pct", "unit": "%", "value": 82.0, "timestamp": 1.0},
            {"sensor_id": "r5", "name": "infrastructure_connection_score", "unit": "ratio", "value": 0.35, "timestamp": 1.0},
            {"sensor_id": "r6", "name": "natural_hazard_score", "unit": "ratio", "value": 0.62, "timestamp": 1.0},
            {"sensor_id": "r7", "name": "occupancy_safety_score", "unit": "ratio", "value": 0.71, "timestamp": 1.0},
            {"sensor_id": "r8", "name": "ground_stability_score", "unit": "ratio", "value": 0.58, "timestamp": 1.0},
        ]
    )
    assert not rejected
    adapter = AgentSchemaAdapter(agent.schema)
    window = adapter.build_window(readings, inherited_rejections=rejected)
    evidence = agent.physics.evaluate(window=window, schema=agent.schema, context={})

    assert evidence.agent_id == "cauren-civil"
    assert evidence.risk_contribution == 0.0
    assert evidence.agent_outputs["proposed_heuristic_risk_contribution"] > 0.0
    assert evidence.agent_outputs["risk_contribution_status"] == "suppressed_unvalidated_rules"
    assert evidence.agent_outputs["physics_class"] == "CivilPhysics"
    assert evidence.agent_outputs["interpretation_status"] == "non_authoritative_decision_support"
    assert evidence.agent_outputs["engineering_validation_status"] == "not_established"
    names = {item["name"] for item in evidence.relations}
    assert "structural_ground_coupling" in names
    assert "construction_readiness_gap" in names
    relation = next(item for item in evidence.relations if item["name"] == "structural_ground_coupling")
    assert relation["validation_status"] == "unvalidated_decision_support_heuristic"
    assert relation["risk_contribution_enabled"] is False
    assert relation["references"] == []


def test_unvalidated_relation_score_is_retained_but_cannot_raise_active_physics_risk():
    agent = build_civil_agent()
    relation = build_relation("prototype_rule", 0.9, "diagnostic-only evidence")

    evidence = agent.physics.evidence(
        schema=agent.schema,
        risk=0.9,
        relations=[relation],
        missing_features=[],
        notes=(),
    )

    assert evidence.relations[0]["score"] == 0.9
    assert evidence.risk_contribution == 0.0
    assert evidence.agent_outputs["proposed_heuristic_risk_contribution"] == 0.9
    assert evidence.agent_outputs["risk_contribution_status"] == "suppressed_unvalidated_rules"


def test_empty_physics_result_suppresses_proposed_risk_safely():
    agent = build_civil_agent()

    evidence = agent.physics.evidence(
        schema=agent.schema,
        risk=0.7,
        relations=[],
        missing_features=list(agent.schema.required_features),
        notes=(),
    )

    assert evidence.risk_contribution == 0.0
    assert evidence.agent_outputs["proposed_heuristic_risk_contribution"] == 0.7
    assert evidence.agent_outputs["risk_contribution_status"] == "suppressed_unvalidated_rules"


def test_civil_agent_folds_in_oma_frequency_drift_evidence():
    agent = build_civil_agent()
    readings, rejected = parse_sensor_readings(
        [
            {"sensor_id": "r1", "name": "structural_risk_score", "unit": "ratio", "value": 0.2, "timestamp": 1.0},
            {"sensor_id": "r2", "name": "ground_stability_score", "unit": "ratio", "value": 0.2, "timestamp": 1.0},
        ]
    )
    adapter = AgentSchemaAdapter(agent.schema)
    window = adapter.build_window(readings, inherited_rejections=rejected)

    evidence = agent.physics.evaluate(
        window=window,
        schema=agent.schema,
        context={"oma_frequency_drift": {"max_drop_pct": 12.0, "overall_severity": "alarm"}},
    )

    relation = next(item for item in evidence.relations if item["name"] == "modal_frequency_drift")
    assert relation["max_drop_pct"] == 12.0
    assert relation["severity"] == "alarm"
    assert relation["severity_validation_status"] == "caller_supplied_unvalidated_threshold"
    assert relation["risk_mapping_status"] == "not_established"
    assert relation["informational_only"] is True
    # The supplied physical scores contribute 0.2; an unvalidated OMA
    # percentage-to-risk mapping must not inflate that contribution.
    assert evidence.agent_outputs["subsystem_scores"]["structural_reliability"] == 0.2
    assert evidence.risk_contribution == 0.0


def test_latest_uses_last_observed_value_instead_of_unobserved_padding():
    window = SensorWindow(
        matrix=((0.2,), (0.9,)),
        features=(FeatureMetadata(name="structural_risk_score", unit="ratio"),),
        presence_mask=((True,), (False,)),
    )

    assert latest(window, "structural_risk_score") == 0.2


def test_latest_returns_none_when_feature_has_no_observed_evidence():
    window = SensorWindow(
        matrix=((0.2,),),
        features=(FeatureMetadata(name="structural_risk_score", unit="ratio"),),
        presence_mask=((False,),),
    )

    assert latest(window, "structural_risk_score") is None


def _civil_evidence(values: dict[str, float]):
    agent = build_civil_agent()
    readings, rejected = parse_sensor_readings(
        [
            {
                "sensor_id": name,
                "name": name,
                "unit": "%" if name.endswith("_pct") else "ratio",
                "value": value,
                "timestamp": 1.0,
            }
            for name, value in values.items()
        ]
    )
    assert not rejected
    window = AgentSchemaAdapter(agent.schema).build_window(readings, inherited_rejections=rejected)
    return agent.physics.evaluate(window=window, schema=agent.schema, context={})


def test_civil_zero_risk_full_relation_is_not_overridden_by_partial_fallback():
    evidence = _civil_evidence(
        {
            "structural_risk_score": 0.0,
            "inspection_finding_score": 0.0,
            "permit_status_score": 1.0,
            "construction_progress_pct": 100.0,
            "infrastructure_connection_score": 1.0,
            "natural_hazard_score": 0.0,
            "occupancy_safety_score": 0.0,
            "ground_stability_score": 0.0,
        }
    )

    assert evidence.risk_contribution == 0.0
    assert set(evidence.agent_outputs["subsystem_scores"].values()) == {0.0}
    assert not [item["name"] for item in evidence.relations if item.get("partial_evidence")]


def test_civil_lone_readiness_indicator_is_not_scored_as_risk():
    for values in (
        {"permit_status_score": 1.0},
        {"infrastructure_connection_score": 1.0},
        {"construction_progress_pct": 100.0},
        {"permit_status_score": 1.0, "infrastructure_connection_score": 1.0},
    ):
        evidence = _civil_evidence(values)
        assert evidence.agent_outputs["subsystem_scores"]["construction_readiness"] == 0.0, values


def test_civil_partial_readiness_follows_the_progress_gap_direction():
    ahead = _civil_evidence({"construction_progress_pct": 90.0, "permit_status_score": 0.2})
    ready = _civil_evidence({"construction_progress_pct": 90.0, "permit_status_score": 1.0})

    relation = next(
        item for item in ahead.relations if item["name"] == "construction_readiness_partial_evidence"
    )
    assert relation["partial_evidence"] is True
    # Dampened to half of the full relation's saturated score.
    assert ahead.agent_outputs["subsystem_scores"]["construction_readiness"] == 0.5
    assert ready.agent_outputs["subsystem_scores"]["construction_readiness"] == 0.0


def _civil_evidence_with_context(values: dict[str, float], context: dict):
    agent = build_civil_agent()
    readings, rejected = parse_sensor_readings(
        [
            {"sensor_id": name, "name": name, "unit": "ratio", "value": value, "timestamp": 1.0}
            for name, value in values.items()
        ]
    )
    assert not rejected
    window = AgentSchemaAdapter(agent.schema).build_window(readings, inherited_rejections=rejected)
    return agent.physics.evaluate(window=window, schema=agent.schema, context=context)


def test_civil_out_of_range_ratio_is_not_scored_by_physics():
    for value in (-3.0, 7.5):
        evidence = _civil_evidence_with_context(
            {"structural_risk_score": value, "ground_stability_score": 0.1}, context={}
        )
        scores = evidence.agent_outputs["subsystem_scores"]
        assert all(0.0 <= score <= 1.0 for score in scores.values())
        # Only the in-range ground stability reading is usable, so the
        # subsystem falls back to the dampened single-indicator relation.
        assert scores["structural_reliability"] == 0.05
        names = {item["name"] for item in evidence.relations}
        assert "structural_ground_coupling" not in names
        assert "structural_partial_evidence" in names
        assert "structural_risk_score" not in evidence.missing_features


def test_civil_ignores_non_finite_or_non_numeric_oma_drop():
    values = {"structural_risk_score": 0.1, "ground_stability_score": 0.1}
    baseline = _civil_evidence_with_context(values, context={})
    for drop in (float("nan"), float("inf"), "abc"):
        evidence = _civil_evidence_with_context(
            values, context={"oma_frequency_drift": {"max_drop_pct": drop}}
        )
        assert "modal_frequency_drift" not in {item["name"] for item in evidence.relations}
        assert (
            evidence.agent_outputs["subsystem_scores"]["structural_reliability"]
            == baseline.agent_outputs["subsystem_scores"]["structural_reliability"]
        )
