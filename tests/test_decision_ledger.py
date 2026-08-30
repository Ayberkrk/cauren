from __future__ import annotations

import json
from pathlib import Path

from api.decision_ledger import DecisionLedgerStore, build_model_decision_record


def test_build_model_decision_record_and_store(tmp_path: Path) -> None:
    resp = {
        "diagnostics": {
            "event_id": "evt-1",
            "timestamp": "2026-04-01T10:00:00Z",
            "status": "WARNING",
            "classifier_severity": "WARNING",
            "fault_label": "Pressure Instability",
            "fault_family_id": "pressure_instability",
            "fault_family_label": "Pressure Instability",
            "fault_subtype_id": "valve_lag",
            "fault_subtype_label": "Valve Lag",
            "fault_signature_id": "sig-123",
            "fault_signature_text": "VALVE_LAG__internal_pressure__stable",
            "known_status": "known",
            "similar_family_candidates": [{"family_id": "pressure_instability", "family_label": "Pressure Instability", "confidence": 0.81}],
            "primary_fault_dimension": "internal_pressure",
            "failure_type": "pressure_drift",
            "fault_name": "Pressure drift",
            "root_cause_label": "Valve lag",
            "root_cause_confidence": 0.81,
            "top3_candidates": [{"label": "Valve lag", "confidence": 0.81}],
            "ttf_state": "soon",
            "ttf_seconds": 120.0,
            "ttf_confidence": 0.77,
        },
        "metrics": {
            "coverage_score": 0.83,
            "missing_dimensions": ["radiation"],
            "active_dimensions": ["internal_pressure", "strain"],
            "core_active_dimensions": ["internal_pressure", "strain"],
            "aux_active_dimensions": [],
        },
        "meta": {
            "asset_id": "AST-001",
            "module_id": "MOD-001",
            "site_id": "SITE-001",
            "model_version": "run-1",
            "data_policy": "250k_beta25_only",
        },
    }
    record = build_model_decision_record(resp=resp)
    assert record["source_layer"] == "model"
    assert record["decision_type"] == "diagnose"
    assert record["severity"] == "warning"
    assert record["module_id"] == "MOD-001"
    assert record["diagnostics_snapshot"]["fault_family_id"] == "pressure_instability"

    sqlite_path = tmp_path / "decision_ledger.sqlite3"
    jsonl_path = tmp_path / "decision_ledger.jsonl"
    store = DecisionLedgerStore(sqlite_path=sqlite_path, jsonl_path=jsonl_path, retention_days=365)
    stored = store.append(record)

    assert stored["hash_self"]
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    parsed = json.loads(lines[0])
    assert parsed["event_id"] == "evt-1"
