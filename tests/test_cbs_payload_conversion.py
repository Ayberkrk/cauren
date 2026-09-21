import os
import sys
import time

import pytest
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.app import (
    DiagnoseIn,
    _building_site_context,
    _building_sensors_from_payload,
    _inspection_score,
    _score_from_mapping,
    _uses_cbs_building_payload,
    app,
)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _cbs_record(building_id: str = "bina-001") -> dict:
    """Same shape as tests/test_api_contract.py's fixture of the same name."""
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


def _sensor_values(payload: DiagnoseIn) -> dict[str, float]:
    return {row["name"]: row["value"] for row in _building_sensors_from_payload(payload)}


# ---------------------------------------------------------------------------
# _score_from_mapping / _inspection_score (the shared helpers)
# ---------------------------------------------------------------------------


def test_score_from_mapping_uses_first_matching_key():
    data = {"structural_risk": 0.6, "risk_score": 0.9}
    value = _score_from_mapping(data, ("structural_risk_score", "structural_risk", "building_risk_score", "risk_score"), 0.25)
    assert value == 0.6


def test_score_from_mapping_falls_back_through_alias_chain():
    # Only the last alias in the chain is present.
    data = {"risk_score": 0.44}
    value = _score_from_mapping(data, ("structural_risk_score", "structural_risk", "building_risk_score", "risk_score"), 0.25)
    assert value == 0.44


def test_score_from_mapping_returns_default_when_no_key_matches():
    assert _score_from_mapping({"unrelated": 1.0}, ("a", "b"), 0.42) == 0.42


def test_score_from_mapping_returns_default_when_not_a_dict():
    assert _score_from_mapping(None, ("a",), 0.5) == 0.5
    assert _score_from_mapping("not-a-dict", ("a",), 0.5) == 0.5


def test_score_from_mapping_maps_booleans_to_one_or_zero():
    assert _score_from_mapping({"approved": True}, ("approved",), 0.5) == 1.0
    assert _score_from_mapping({"ready": False}, ("ready",), 0.5) == 0.0


def test_score_from_mapping_clamps_out_of_range_values():
    assert _score_from_mapping({"score": 5.0}, ("score",), 0.0) == 1.0
    assert _score_from_mapping({"score": -3.0}, ("score",), 0.0) == 0.0


def test_score_from_mapping_falls_back_on_unparseable_value():
    assert _score_from_mapping({"score": "not-a-number"}, ("score",), 0.33) == 0.33


def test_inspection_score_picks_max_across_items_and_key_aliases():
    items = [
        {"finding_score": 0.2},
        {"severity_score": 0.9},
        {"score": 0.5},
    ]
    assert _inspection_score(items) == 0.9


def test_inspection_score_default_when_empty_or_missing():
    assert _inspection_score([]) == 0.2
    assert _inspection_score(None) == 0.2


def test_inspection_score_ignores_non_dict_items():
    assert _inspection_score([{"finding_score": 0.6}, "not-a-dict"]) == 0.6


# ---------------------------------------------------------------------------
# _building_sensors_from_payload (full conversion)
# ---------------------------------------------------------------------------


def test_building_sensors_from_full_cbs_record_match_expected_values():
    payload = DiagnoseIn(**_cbs_record("bina-002"))
    values = _sensor_values(payload)

    assert values["structural_risk_score"] == 0.82
    assert values["inspection_finding_score"] == 0.78
    assert values["permit_status_score"] == 0.4
    # Percentage inputs must remain percentages rather than being clamped
    # to 1.0 before conversion.
    assert values["construction_progress_pct"] == 82.0
    assert values["infrastructure_connection_score"] == 0.45
    assert values["natural_hazard_score"] == 0.65
    # occupancy_safety_score has no matching key in risk_assessments here.
    assert values["occupancy_safety_score"] == 0.2
    assert values["ground_stability_score"] == 0.52


def test_construction_progress_pct_accepts_fractional_and_percentage_inputs():
    for raw_value, expected in ((0.82, 82.0), (82, 82.0), (-3, 0.0), (140, 100.0)):
        payload = DiagnoseIn(building_id="b1", construction_status={"progress_pct": raw_value})
        values = _sensor_values(payload)
        assert values["construction_progress_pct"] == expected


def test_building_sensors_use_documented_defaults_when_subdicts_are_missing():
    payload = DiagnoseIn(building_id="b1")
    values = _sensor_values(payload)
    assert values["structural_risk_score"] == 0.25
    assert values["inspection_finding_score"] == 0.2
    assert values["permit_status_score"] == 0.5
    assert values["construction_progress_pct"] == 0.0
    assert values["infrastructure_connection_score"] == 0.5
    assert values["natural_hazard_score"] == 0.2
    assert values["occupancy_safety_score"] == 0.2
    assert values["ground_stability_score"] == 0.2


def test_building_sensors_alias_keys_resolve_to_canonical_names():
    payload = DiagnoseIn(
        building_id="b1",
        risk_assessments={"structural_risk": 0.55, "flood_risk_score": 0.71, "life_safety_score": 0.61, "ground_risk_score": 0.48},
        project_permit={"approval_score": 0.66},
        infrastructure_connections={"utility_connection_score": 0.39},
    )
    values = _sensor_values(payload)
    assert values["structural_risk_score"] == 0.55
    assert values["natural_hazard_score"] == 0.71
    assert values["occupancy_safety_score"] == 0.61
    assert values["ground_stability_score"] == 0.48
    assert values["permit_status_score"] == 0.66
    assert values["infrastructure_connection_score"] == 0.39


def test_building_sensors_boolean_permit_and_infrastructure_fields():
    payload = DiagnoseIn(
        building_id="b1",
        project_permit={"approved": True},
        infrastructure_connections={"ready": False},
    )
    values = _sensor_values(payload)
    assert values["permit_status_score"] == 1.0
    assert values["infrastructure_connection_score"] == 0.0


def test_building_sensors_sensor_id_and_unit_shape():
    payload = DiagnoseIn(building_id="tower-7")
    rows = _building_sensors_from_payload(payload)
    by_name = {row["name"]: row for row in rows}

    assert len(rows) == 8
    for name, row in by_name.items():
        assert row["sensor_id"] == f"tower-7_{name}"
        expected_unit = "%" if name == "construction_progress_pct" else "ratio"
        assert row["unit"] == expected_unit
        assert row["quality"] is True


# ---------------------------------------------------------------------------
# _uses_cbs_building_payload / _building_site_context
# ---------------------------------------------------------------------------


def test_uses_cbs_building_payload_requires_building_id_and_no_sensors():
    assert _uses_cbs_building_payload(DiagnoseIn(building_id="b1")) is True
    assert _uses_cbs_building_payload(DiagnoseIn(building_id="")) is False
    assert _uses_cbs_building_payload(DiagnoseIn()) is False
    assert _uses_cbs_building_payload(DiagnoseIn(building_id="b1", sensors=[{"name": "x", "value": 1}])) is False


def test_building_site_context_collects_only_present_fields():
    assert _building_site_context(DiagnoseIn(building_id="b1")) == {}

    payload = DiagnoseIn(
        building_id="b1",
        geometry={"type": "Point", "coordinates": [1.0, 2.0]},
        address={"province": "Ankara"},
    )
    context = _building_site_context(payload)
    assert context == {"geometry": {"type": "Point", "coordinates": [1.0, 2.0]}, "address": {"province": "Ankara"}}
    assert "administrative_unit" not in context


# ---------------------------------------------------------------------------
# Through the API: behavior that lives in _sync_diagnose, not the helpers
# ---------------------------------------------------------------------------


def test_diagnose_prefers_sensors_over_cbs_fields_when_both_are_present(client):
    payload = {
        **_cbs_record("bina-003"),
        "sensors": [
            {"sensor_id": "s1", "name": "structural_risk_score", "unit": "ratio", "value": 0.9, "quality": True},
        ],
        "timestamp": time.time(),
    }
    resp = client.post("/diagnose", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    # The plain-sensors path won, so this is not treated as a CBS payload
    # even though building_id/risk_assessments/etc. were also supplied.
    assert body["meta"]["cbs_building_payload"] is False


def test_diagnose_overrides_conflicting_agent_id_and_sector_for_cbs_payloads(client):
    payload = {
        **_cbs_record("bina-004"),
        "agent_id": "cauren-bridge",
        "sector": "not-civil",
        "timestamp": time.time(),
    }
    resp = client.post("/diagnose", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["selected_agent"] == "cauren-civil"


def test_diagnose_propagates_cbs_site_context_fields(client):
    payload = {
        **_cbs_record("bina-005"),
        "timestamp": time.time(),
    }
    resp = client.post("/diagnose", json=payload)
    assert resp.status_code == 200
    # site_context isn't echoed back verbatim in the response today, so the
    # regression this guards against is a crash/500 when geometry/address/
    # administrative_unit are present -- a real assertion on the resulting
    # site_context requires either exposing it in the response or testing
    # _building_site_context directly (covered above).
