import os
import sys
import time

import pytest
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.app import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _sensor_payload(asset_id: str = "asset-1") -> dict:
    return {
        "asset_id": asset_id,
        "site_id": "site-01",
        "line_id": "zone-a",
        "machine_id": "node-01",
        "agent_id": "cauren-civil",
        "sector": "civil",
        "timestamp": time.time(),
        "seq_len": 16,
        "sampling_hz": 1.0,
        "sensors": [
            {"sensor_id": "structural-risk", "name": "structural_risk_score", "unit": "score", "value": 0.72, "quality": True},
            {"sensor_id": "permit-status", "name": "permit_status_score", "unit": "score", "value": 0.81, "quality": True},
            {"sensor_id": "inspection", "name": "inspection_finding_score", "unit": "score", "value": 0.36, "quality": True},
            {"sensor_id": "progress", "name": "construction_progress_pct", "unit": "pct", "value": 64.0, "quality": True},
            {"sensor_id": "ground", "name": "ground_stability_score", "unit": "score", "value": 0.58, "quality": True},
        ],
    }


def _cbs_record(building_id: str = "bina-001") -> dict:
    return {
        "building_id": building_id,
        "geometry": {"type": "Point", "coordinates": [32.85, 39.92]},
        "address": {"province": "Ankara", "district": "Cankaya"},
        "administrative_unit": {"country": "TR", "province_code": "06"},
        "project_permit": {"permit_status_score": 0.4, "permit_no": "P-1"},
        "construction_status": {"progress_pct": 82},
        "infrastructure_connections": {"readiness_score": 0.45},
        "risk_assessments": {
            "structural_risk_score": 0.82,
            "natural_hazard_score": 0.65,
            "ground_stability_score": 0.52,
        },
        "inspection_findings": [{"finding_score": 0.78, "note": "field review required"}],
    }


def test_health_endpoints_are_available(client):
    liveness = client.get("/health/liveness")
    readiness = client.get("/health/readiness")
    assert liveness.status_code == 200
    assert readiness.status_code == 200
    assert liveness.json()["status"] == "ok"
    body = readiness.json()
    assert body["ready"] is True
    assert body["agent_count"] == 1
    assert body["agents"] == ["cauren-civil"]


def test_all_routes_are_reachable_with_no_authentication(client):
    """This API ships with no auth layer: it's a research prototype meant to
    run locally or behind whatever the operator puts in front of it, not a
    multi-tenant service with its own access control. A plain request with
    no headers must succeed.
    """
    resp = client.get("/agents")
    assert resp.status_code == 200


def test_agents_route_is_single_civil_agent(client):
    resp = client.get("/agents")
    assert resp.status_code == 200
    body = resp.json()
    assert body["architecture"] == "cauren_core_civil_agent"
    assert [agent["agent_id"] for agent in body["agents"]] == ["cauren-civil"]


def test_calibrate_requires_sensors(client):
    payload = _sensor_payload()
    payload.pop("sensors")
    resp = client.post("/calibrate", json=payload)
    assert resp.status_code == 422


def test_calibrate_accepts_civil_sensor_payload(client):
    resp = client.post("/calibrate", json=_sensor_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["selected_agent"] == "cauren-civil"
    assert body["feature_names"]
    assert body["meta"]["asset_id"] == "asset-1"


def test_diagnose_requires_sensors_or_building_payload(client):
    payload = _sensor_payload()
    payload.pop("sensors")
    payload.pop("agent_id")
    payload.pop("sector")
    resp = client.post("/diagnose", json=payload)
    assert resp.status_code == 422


def test_diagnose_accepts_civil_sensor_payload(client):
    resp = client.post("/diagnose", json=_sensor_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["selected_agent"] == "cauren-civil"
    assert body["calibration"]["mode"] == "cauren_core_agent_schema"
    assert body["meta"]["asset_id"] == "asset-1"


def test_diagnose_response_carries_quality_control_and_uncertainty(client):
    """Both layers are computed in the pipeline; this pins that they survive
    the API's response shaping and actually reach HTTP clients.
    """
    resp = client.post("/diagnose", json=_sensor_payload())
    assert resp.status_code == 200
    body = resp.json()

    quality = body["quality_control"]
    assert quality["status"] in {"pass", "warn", "fail"}
    assert 0.0 <= quality["score"] <= 1.0
    assert isinstance(quality["findings"], list)
    assert quality["feature_completeness"]

    uncertainty = body["uncertainty_report"]
    assert 0.0 <= uncertainty["uncertainty_score"] <= 1.0
    assert uncertainty["lower_bound"] <= uncertainty["point_estimate"] <= uncertainty["upper_bound"]
    assert {factor["name"] for factor in uncertainty["factors"]} == {
        "data_quality",
        "model_confidence",
        "evidence_coverage",
    }


def test_calibrate_response_carries_quality_control(client):
    """/calibrate exists to inspect what the pipeline made of a payload
    without running physics, so the input-quality verdict belongs in it too.
    """
    resp = client.post("/calibrate", json=_sensor_payload())
    assert resp.status_code == 200
    quality = resp.json()["quality_control"]
    assert quality["status"] in {"pass", "warn", "fail"}
    assert quality["checks_run"]


def test_diagnose_building_payload_forces_civil_agent(client):
    payload = {
        **_cbs_record("bina-002"),
        "timestamp": time.time(),
        "seq_len": 16,
        "sampling_hz": 1.0,
    }
    resp = client.post("/diagnose", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["selected_agent"] == "cauren-civil"
    assert body["meta"]["cbs_building_payload"] is True


def test_diagnose_defaults_asset_id_and_timestamp_when_omitted(client):
    """No institutional asset-identity resolver behind this anymore: a
    missing asset_id/timestamp should just fall back to sane defaults
    instead of requiring a resolution service.
    """
    payload = _sensor_payload()
    payload.pop("asset_id")
    payload.pop("timestamp")
    resp = client.post("/diagnose", json=payload)
    assert resp.status_code == 200
    assert resp.json()["meta"]["asset_id"] == "cauren_asset"
