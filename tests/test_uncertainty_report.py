from cauren_agents.civil.agent import build_civil_agent
from cauren_core import CaurenPipeline
from cauren_core.quality_control import evaluate_quality
from cauren_core.uncertainty import estimate_uncertainty


def _full_civil_sensors():
    return [
        {"sensor_id": "r1", "name": "structural_risk_score", "unit": "ratio", "value": 0.81, "timestamp": 1.0},
        {"sensor_id": "r2", "name": "inspection_finding_score", "unit": "ratio", "value": 0.78, "timestamp": 1.0},
        {"sensor_id": "r3", "name": "permit_status_score", "unit": "ratio", "value": 0.42, "timestamp": 1.0},
        {"sensor_id": "r4", "name": "construction_progress_pct", "unit": "%", "value": 82.0, "timestamp": 1.0},
        {"sensor_id": "r5", "name": "infrastructure_connection_score", "unit": "ratio", "value": 0.35, "timestamp": 1.0},
        {"sensor_id": "r6", "name": "natural_hazard_score", "unit": "ratio", "value": 0.62, "timestamp": 1.0},
        {"sensor_id": "r7", "name": "occupancy_safety_score", "unit": "ratio", "value": 0.71, "timestamp": 1.0},
        {"sensor_id": "r8", "name": "ground_stability_score", "unit": "ratio", "value": 0.58, "timestamp": 1.0},
    ]


def test_estimate_uncertainty_is_low_for_clean_high_confidence_input():
    agent = build_civil_agent()
    from cauren_core.adapter import AgentSchemaAdapter, parse_sensor_readings

    readings, rejected = parse_sensor_readings(_full_civil_sensors())
    window = AgentSchemaAdapter(agent.schema).build_window(readings, inherited_rejections=rejected)
    quality_report = evaluate_quality(window, agent.schema)

    report = estimate_uncertainty(
        risk_score=0.6,
        core_confidence=0.95,
        quality_report=quality_report,
        schema=agent.schema,
        missing_features=(),
    )

    assert report.uncertainty_score < 0.1
    assert report.lower_bound <= report.point_estimate <= report.upper_bound
    assert (report.upper_bound - report.lower_bound) < 0.1


def test_estimate_uncertainty_widens_with_missing_features():
    agent = build_civil_agent()
    from cauren_core.adapter import AgentSchemaAdapter, parse_sensor_readings

    readings, rejected = parse_sensor_readings(_full_civil_sensors())
    window = AgentSchemaAdapter(agent.schema).build_window(readings, inherited_rejections=rejected)
    quality_report = evaluate_quality(window, agent.schema)

    narrow = estimate_uncertainty(
        risk_score=0.6,
        core_confidence=0.9,
        quality_report=quality_report,
        schema=agent.schema,
        missing_features=(),
    )
    wide = estimate_uncertainty(
        risk_score=0.6,
        core_confidence=0.9,
        quality_report=quality_report,
        schema=agent.schema,
        missing_features=("structural_risk_score", "ground_stability_score", "natural_hazard_score"),
    )

    assert wide.uncertainty_score > narrow.uncertainty_score
    assert (wide.upper_bound - wide.lower_bound) > (narrow.upper_bound - narrow.lower_bound)


def test_estimate_uncertainty_widens_with_low_model_confidence():
    agent = build_civil_agent()
    from cauren_core.adapter import AgentSchemaAdapter, parse_sensor_readings

    readings, rejected = parse_sensor_readings(_full_civil_sensors())
    window = AgentSchemaAdapter(agent.schema).build_window(readings, inherited_rejections=rejected)
    quality_report = evaluate_quality(window, agent.schema)

    confident = estimate_uncertainty(
        risk_score=0.6, core_confidence=0.95, quality_report=quality_report, schema=agent.schema,
    )
    unsure = estimate_uncertainty(
        risk_score=0.6, core_confidence=0.2, quality_report=quality_report, schema=agent.schema,
    )

    assert unsure.uncertainty_score > confident.uncertainty_score


def test_estimate_uncertainty_bounds_stay_within_zero_one():
    agent = build_civil_agent()
    from cauren_core.adapter import AgentSchemaAdapter, parse_sensor_readings

    readings, rejected = parse_sensor_readings([])
    window = AgentSchemaAdapter(agent.schema).build_window(readings, inherited_rejections=rejected)
    quality_report = evaluate_quality(window, agent.schema)

    report = estimate_uncertainty(
        risk_score=0.05,
        core_confidence=0.0,
        quality_report=quality_report,
        schema=agent.schema,
        missing_features=tuple(agent.schema.required_features),
    )

    assert report.uncertainty_score == 1.0
    assert 0.0 <= report.lower_bound <= report.point_estimate
    assert report.point_estimate <= report.upper_bound <= 1.0


def test_pipeline_diagnosis_includes_uncertainty_report():
    pipeline = CaurenPipeline.from_default_registry()
    payload = {"agent_id": "cauren-civil", "sensors": _full_civil_sensors()}

    diagnosis = pipeline.diagnose(payload)

    assert "uncertainty_report" in diagnosis.agent_outputs
    uncertainty = diagnosis.agent_outputs["uncertainty_report"]
    assert 0.0 <= uncertainty["uncertainty_score"] <= 1.0
    assert uncertainty["lower_bound"] <= uncertainty["point_estimate"] <= uncertainty["upper_bound"]
    assert "quality_control" in diagnosis.agent_outputs
