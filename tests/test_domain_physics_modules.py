from cauren_agents.civil.agent import build_civil_agent
from cauren_core.adapter import AgentSchemaAdapter, parse_sensor_readings



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
    assert evidence.risk_contribution > 0.0
    assert evidence.agent_outputs["physics_class"] == "CivilPhysics"
    names = {item["name"] for item in evidence.relations}
    assert "structural_ground_coupling" in names
    assert "construction_readiness_gap" in names
