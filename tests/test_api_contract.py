import os
import sys
import time

import pytest
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# api/app.py no longer ships a hardcoded default bearer token (see
# GOV_PILOT_API_TOKEN handling) -- tests provide their own before import
# so the security gate has something to check requests against.
TEST_AUTH_TOKEN = "test-only-cauren-pilot-token"
os.environ.setdefault("GOV_PILOT_API_TOKEN", TEST_AUTH_TOKEN)

from api.app import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        c.headers.update(
            {
                "Authorization": f"Bearer {TEST_AUTH_TOKEN}",
                "X-Cauren-Role": "operator",
            }
        )
        yield c


def _sensor_payload(asset_id: str = "asset-1") -> dict:
    return {
        "asset_id": asset_id,
        "site_id": "pilot-site",
        "line_id": "zone-a",
        "machine_id": "node-01",
        "agent_id": "cauren-civil",
        "sector": "civil",
        "timestamp": time.time(),
        "mission_phase": "field_review",
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
        "unexpected_tucbs_field": {"preserve": True},
    }


def test_health_endpoints_are_available(client):
    liveness = client.get("/health/liveness")
    readiness = client.get("/health/readiness")
    assert liveness.status_code == 200
    assert readiness.status_code == 200
    assert liveness.json()["status"] == "alive"
    body = readiness.json()
    assert body["runtime_architecture"] == "cauren_core_civil_agent"
    assert body["agent_count"] == 1
    assert body["agents"] == ["cauren-civil"]


def test_security_rejects_unauthenticated_non_health_request():
    with TestClient(app) as c:
        resp = c.get("/metrics")
    assert resp.status_code == 401


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
    assert body["meta"]["architecture"] == "cauren_core_civil_agent"


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
    assert body["meta"]["architecture"] == "cauren_core_civil_agent"


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


def test_cbs_building_batch_processes_and_deduplicates(client):
    payload = {
        "idempotency_key": f"batch-{time.time_ns()}",
        "source": "tucbs",
        "records": [_cbs_record("bina-001"), {"building_id": "bad-1"}],
    }
    first = client.post("/cbs/buildings/batch", json=payload)
    assert first.status_code == 200
    body = first.json()
    assert body["submitted_records"] == 2
    assert body["accepted_records"] == 1
    assert body["rejected_records"] == 1
    assert body["sample_results"][0]["selected_agent"] == "cauren-civil"
    assert body["contract"]["agent"] == "cauren-civil"

    status = client.get(f"/cbs/jobs/{body['job_id']}")
    assert status.status_code == 200
    assert status.json()["job_id"] == body["job_id"]

    replay = client.post("/cbs/buildings/batch", json=payload)
    assert replay.status_code == 200
    assert replay.json()["job_id"] == body["job_id"]
    assert replay.json()["idempotent_replay"] is True
