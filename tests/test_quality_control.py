from cauren_agents.civil.agent import build_civil_agent
from cauren_core.adapter import AgentSchemaAdapter, parse_sensor_readings
from cauren_core.contracts import NormalizationTrace
from cauren_core.quality_control import evaluate_quality


def _full_civil_readings(**overrides):
    base = {
        "structural_risk_score": 0.81,
        "inspection_finding_score": 0.78,
        "permit_status_score": 0.42,
        "construction_progress_pct": 82.0,
        "infrastructure_connection_score": 0.35,
        "natural_hazard_score": 0.62,
        "occupancy_safety_score": 0.71,
        "ground_stability_score": 0.58,
    }
    base.update(overrides)
    unit_by_feature = {"construction_progress_pct": "%"}
    return [
        {
            "sensor_id": f"r{idx}",
            "name": name,
            "unit": unit_by_feature.get(name, "ratio"),
            "value": value,
            "timestamp": 1.0,
        }
        for idx, (name, value) in enumerate(base.items())
    ]


def test_evaluate_quality_passes_on_complete_in_range_window():
    agent = build_civil_agent()
    readings, rejected = parse_sensor_readings(_full_civil_readings())
    adapter = AgentSchemaAdapter(agent.schema)
    window = adapter.build_window(readings, inherited_rejections=rejected)

    report = evaluate_quality(window, agent.schema)

    assert report.status == "pass"
    assert report.score == 1.0
    assert report.findings == ()
    assert report.feature_completeness["structural_risk_score"] == 1.0


def test_evaluate_quality_flags_missing_required_feature():
    agent = build_civil_agent()
    rows = [row for row in _full_civil_readings() if row["name"] != "ground_stability_score"]
    readings, rejected = parse_sensor_readings(rows)
    adapter = AgentSchemaAdapter(agent.schema)
    window = adapter.build_window(readings, inherited_rejections=rejected)

    report = evaluate_quality(window, agent.schema)

    assert report.status == "fail"
    codes = {finding.code for finding in report.findings}
    assert "incomplete_required_feature" in codes
    assert report.feature_completeness["ground_stability_score"] == 0.0


def test_evaluate_quality_flags_out_of_range_ratio_value():
    agent = build_civil_agent()
    readings, rejected = parse_sensor_readings(_full_civil_readings(structural_risk_score=1.4))
    adapter = AgentSchemaAdapter(agent.schema)
    window = adapter.build_window(readings, inherited_rejections=rejected)

    report = evaluate_quality(window, agent.schema)

    assert report.status == "warn"
    matches = [f for f in report.findings if f.code == "value_out_of_range" and f.feature == "structural_risk_score"]
    assert matches


def test_evaluate_quality_flags_flatline_signal_across_repeated_window():
    agent = build_civil_agent()
    readings, rejected = parse_sensor_readings(
        [
            {
                "sensor_id": f"r{step}",
                "name": "structural_risk_score",
                "unit": "ratio",
                "value": 0.5,
                "timestamp": float(step),
            }
            for step in range(6)
        ]
        + [row for row in _full_civil_readings() if row["name"] != "structural_risk_score"]
    )
    adapter = AgentSchemaAdapter(agent.schema)
    window = adapter.build_window(readings, seq_len=6, inherited_rejections=rejected)

    report = evaluate_quality(window, agent.schema)

    codes = {finding.code for finding in report.findings if finding.feature == "structural_risk_score"}
    assert "flatline_signal" in codes


def test_flatline_finding_counts_only_genuinely_observed_steps():
    """Regression: the adapter pads a short window out to seq_len by
    repeating its earliest row, and those padded rows carry presence=True.
    Counting them made the finding claim far more "observed steps" than
    the caller actually sent.
    """
    agent = build_civil_agent()
    observed_steps = 5
    readings, rejected = parse_sensor_readings(
        [
            {
                "sensor_id": f"r{step}",
                "name": "structural_risk_score",
                "unit": "ratio",
                "value": 0.5,
                "timestamp": float(step),
            }
            for step in range(observed_steps)
        ]
    )
    adapter = AgentSchemaAdapter(agent.schema)
    window = adapter.build_window(readings, seq_len=16, inherited_rejections=rejected)

    assert window.observed_step_count == observed_steps
    assert len(window.matrix) == 16  # padded well beyond the real readings

    report = evaluate_quality(window, agent.schema)
    finding = next(
        item
        for item in report.findings
        if item.code == "flatline_signal" and item.feature == "structural_risk_score"
    )

    assert f"across {observed_steps} observed steps" in finding.message
    assert "16" not in finding.message


def test_evaluate_quality_flags_high_unknown_sensor_ratio():
    agent = build_civil_agent()
    readings, rejected = parse_sensor_readings(_full_civil_readings())
    adapter = AgentSchemaAdapter(agent.schema)
    window = adapter.build_window(readings, inherited_rejections=rejected)

    trace = NormalizationTrace(
        decisions=(),
        normalized_sensor_count=2,
        unknown_sensor_count=3,
        ambiguous_sensor_count=1,
        unknown_sensor_names=("weird_sensor_1", "weird_sensor_2", "weird_sensor_3"),
        ambiguous_sensor_names=("dual_match_sensor",),
    )

    report = evaluate_quality(window, agent.schema, trace)

    codes = {finding.code for finding in report.findings}
    assert "high_unknown_sensor_ratio" in codes
    assert "ambiguous_sensor_mapping" in codes


def test_quality_control_report_to_dict_roundtrip():
    agent = build_civil_agent()
    readings, rejected = parse_sensor_readings(_full_civil_readings())
    adapter = AgentSchemaAdapter(agent.schema)
    window = adapter.build_window(readings, inherited_rejections=rejected)

    report = evaluate_quality(window, agent.schema)
    payload = report.to_dict()

    assert payload["status"] == "pass"
    assert payload["checks_run"] == [
        "completeness",
        "range_sanity",
        "flatline",
        "normalization_health",
        "rejected_samples",
    ]
    assert isinstance(payload["feature_completeness"], dict)
