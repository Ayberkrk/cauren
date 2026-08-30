from cauren_core import CaurenPipeline
from cauren_core.contracts import SensorReading



def test_civil_normalizer_maps_common_synonyms_to_canonical_features():
    pipeline = CaurenPipeline.from_default_registry()
    normalizer = pipeline.normalizer
    readings = [
        SensorReading(sensor_id="a", name="building_risk_score", unit="ratio", value=0.72),
        SensorReading(sensor_id="b", name="finding_score", unit="ratio", value=0.64),
        SensorReading(sensor_id="c", name="approval_score", unit="ratio", value=0.51),
        SensorReading(sensor_id="d", name="readiness_score", unit="ratio", value=0.44),
    ]

    normalized, trace, rejected = normalizer.normalize_readings(
        readings,
        requested_agent_id="cauren-civil",
        sector="civil",
    )

    assert not rejected
    assert [item.name for item in normalized] == [
        "structural_risk_score",
        "inspection_finding_score",
        "permit_status_score",
        "infrastructure_connection_score",
    ]
    assert trace.normalized_sensor_count == 4
    assert trace.unknown_sensor_count == 0



def test_unknown_civil_feature_stays_unresolved_without_match():
    pipeline = CaurenPipeline.from_default_registry()
    normalizer = pipeline.normalizer
    normalized, trace, rejected = normalizer.normalize_readings(
        [SensorReading(sensor_id="x", name="fleet_queue_eta", unit="min", value=14.0)],
        requested_agent_id="cauren-civil",
        sector="civil",
    )

    assert normalized[0].name == "fleet_queue_eta"
    assert trace.unknown_sensor_count == 1
    assert rejected[0]["reason"] == "unknown_feature_name"



def test_pipeline_routes_civil_payload_with_synonym_variants_and_exposes_trace():
    pipeline = CaurenPipeline.from_default_registry()
    sensors = []
    for idx in range(1, 7):
        sensors.extend(
            [
                {"sensor_id": f"risk-{idx}", "name": "building_risk_score", "unit": "ratio", "value": 0.74, "timestamp": idx},
                {"sensor_id": f"inspect-{idx}", "name": "finding_score", "unit": "ratio", "value": 0.68, "timestamp": idx},
                {"sensor_id": f"permit-{idx}", "name": "approval_score", "unit": "ratio", "value": 0.47, "timestamp": idx},
                {"sensor_id": f"progress-{idx}", "name": "progress_pct", "unit": "%", "value": 82.0, "timestamp": idx},
                {"sensor_id": f"infra-{idx}", "name": "readiness_score", "unit": "ratio", "value": 0.42, "timestamp": idx},
                {"sensor_id": f"ground-{idx}", "name": "geotechnical_risk_score", "unit": "ratio", "value": 0.59, "timestamp": idx},
            ]
        )
    result = pipeline.diagnose(
        {
            "seq_len": 6,
            "site_context": {
                "asset_type": "building construction parcel permit inspection",
                "site": "civil construction site",
            },
            "sector_hint": "civil",
            "sensors": sensors,
        }
    )
    body = result.to_dict()

    assert result.selected_agent == "cauren-civil"
    assert body["normalized_sensor_count"] >= 6
    assert body["unknown_sensor_count"] == 0
    assert any(
        item["raw_name"] == "building_risk_score" and item["normalized_name"] == "structural_risk_score"
        for item in body["normalization_trace"]
    )
