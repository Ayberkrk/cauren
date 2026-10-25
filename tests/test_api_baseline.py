import os
import random
import sqlite3
import sys

import pytest
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.app import app

FEATURE = "structural_risk_score"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CAUREN_HISTORY_DB", str(tmp_path / "history.sqlite3"))
    monkeypatch.setenv("CAUREN_HISTORY_RETENTION_DAYS", "90")
    monkeypatch.delenv("CAUREN_API_KEY", raising=False)
    with TestClient(app) as c:
        yield c


def _rows(values, name=FEATURE, unit="ratio"):
    return [
        {"sensor_id": f"{name}-{idx}", "name": name, "unit": unit, "value": value, "timestamp": float(idx)}
        for idx, value in enumerate(values)
    ]


def _diagnose_body(asset_id="bridge-7", **extra):
    body = {"asset_id": asset_id, "agent_id": "cauren-civil", "sensors": _rows([0.4 + 0.01 * (i % 3) for i in range(16)])}
    body.update(extra)
    return body


def _reference_window(seed=1, n=60):
    rng = random.Random(seed)
    return _rows([0.4 + rng.gauss(0.0, 0.03) for _ in range(n)]) + _rows([0.85] * n, name="permit_status_score")


def test_fit_stores_summary_statistics_only(client, tmp_path):
    response = client.post(
        "/baseline/bridge-7",
        json={"agent_id": "cauren-civil", "sensors": _reference_window(), "min_scale": {"permit_status_score": 0.01}},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["agent_id"] == "cauren-civil"
    assert body["features"][FEATURE]["n"] == 60
    assert body["features"][FEATURE]["expected_variation"] is True
    assert body["features"][FEATURE]["scale"] == pytest.approx(0.03, rel=0.35)
    assert body["features"]["permit_status_score"] == {
        "center": 0.85,
        "scale": 0.01,
        "n": 60,
        "expected_variation": False,
        "unit": "ratio",
    }

    with sqlite3.connect(tmp_path / "history.sqlite3") as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(asset_baselines)")]
        count = connection.execute("SELECT COUNT(*) FROM asset_baselines").fetchone()[0]
    assert count == 2
    assert columns == ["asset_id", "feature", "agent_id", "center", "scale", "n", "expected_variation", "unit", "fitted_at"]

    assert client.get("/baseline/bridge-7").json()["features"] == body["features"]


def test_fit_reports_features_it_could_not_fit(client):
    response = client.post("/baseline/bridge-7", json={"agent_id": "cauren-civil", "sensors": _reference_window()})
    assert response.status_code == 200
    assert "supply min_scale" in response.json()["skipped"]["permit_status_score"]

    too_short = client.post("/baseline/bridge-8", json={"agent_id": "cauren-civil", "sensors": _rows([0.4] * 10)})
    assert too_short.status_code == 422
    assert client.get("/baseline/bridge-8").status_code == 404


def test_fit_requires_timestamps(client):
    rows = _rows([0.4] * 40)
    rows[3].pop("timestamp")
    response = client.post("/baseline/bridge-7", json={"agent_id": "cauren-civil", "sensors": rows})
    assert response.status_code == 422


def test_diagnose_resolves_payload_then_stored_then_none(client):
    assert client.post("/diagnose", json=_diagnose_body()).json()["reference"] == "none"

    client.post("/baseline/bridge-7", json={"agent_id": "cauren-civil", "sensors": _reference_window()})
    stored = client.post("/diagnose", json=_diagnose_body()).json()
    assert stored["reference"] == "stored"
    assert stored["reference_coverage"]["features_with_reference"] == [FEATURE]

    inline = {FEATURE: {"center": 0.4, "scale": 0.02, "n": 50, "expected_variation": True, "unit": "ratio"}}
    assert client.post("/diagnose", json=_diagnose_body(baseline=inline)).json()["reference"] == "payload"

    # A stored baseline belongs to a named asset; an anonymous request never
    # picks one up.
    anonymous = _diagnose_body()
    anonymous.pop("asset_id")
    assert client.post("/diagnose", json=anonymous).json()["reference"] == "none"


@pytest.mark.parametrize(
    "item",
    [
        {"center": 0.4, "scale": 0.0, "n": 50},
        {"center": 0.4, "scale": 0.02, "n": 10},
        {"center": "nan", "scale": 0.02, "n": 50},
        {"center": 0.4, "scale": 0.02, "n": 50, "expected_variation": "sometimes"},
    ],
)
def test_diagnose_rejects_invalid_baselines(client, item):
    response = client.post("/diagnose", json=_diagnose_body(baseline={FEATURE: item}))
    assert response.status_code == 422


def test_baseline_route_validates_asset_id(client):
    assert client.post("/baseline/bad id!", json={"sensors": _reference_window()}).status_code in {404, 422}
    assert client.get("/baseline/" + "x" * 65).status_code == 422
