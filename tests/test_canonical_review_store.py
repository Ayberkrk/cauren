from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from api.canonical_review_store import CanonicalReviewStore


def test_canonical_review_store_submits_and_merges_model_event(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "quarantine.sqlite3"
    store = CanonicalReviewStore(sqlite_path)
    entry = {
        "event_id": "evt-model-1",
        "mission_phase": "coast",
        "status": "WARNING",
        "fault_family_id": "pressure_instability",
        "fault_family_label": "Pressure Instability",
        "fault_subtype_id": "valve_lag",
        "fault_subtype_label": "Valve Lag",
        "fault_signature_id": "sig-123",
        "fault_signature_text": "VALVE_LAG__internal_pressure",
        "known_status": "known",
        "similar_family_candidates": [{"family_id": "pressure_instability", "family_label": "Pressure Instability", "confidence": 0.81}],
        "review_required": True,
        "asset_class": "tank",
        "asset_norm_scope": "asset",
        "asset_norm_confidence": 0.73,
        "norm_source": "asset",
        "candidate_dimensions": ["mystery_signal_alpha"],
        "observing_dimensions": ["internal_pressure"],
        "provisional_dimensions": [],
        "asset_critical_dimensions": ["internal_pressure", "material_temp"],
        "dimension_importance_by_asset": {"internal_pressure": "critical_for_asset"},
        "dimension_lifecycle_status": {"mystery_signal_alpha": "candidate"},
        "root_cause_label": "Valve lag",
        "root_cause_confidence": 0.42,
        "unknown_reason": "low_confidence",
        "procedure_ref": "PROC-PRESSURE-DIAGNOSTIC-001",
    }

    first = store.submit_model_case(entry, decision_id="model-evt-model-1-diagnose")
    second = store.submit_model_case(entry, decision_id="llm-evt-model-1-advisory-1")

    assert first["queue_id"].startswith("UNK-")
    assert second["queue_id"] == first["queue_id"]

    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM unknown_cases WHERE unk_code = ?", (first["queue_id"],)).fetchone()
        assert row is not None
        draft = json.loads(row["draft_json"])
        assert draft["fault_family_label"] == "Pressure Instability"
        assert sorted(draft["decision_ids"]) == ["llm-evt-model-1-advisory-1", "model-evt-model-1-diagnose"]
        assert draft["asset_class"] == "tank"
        assert draft["candidate_dimensions"] == ["mystery_signal_alpha"]
        assert draft["dimension_lifecycle_status"]["mystery_signal_alpha"] == "candidate"
    finally:
        conn.close()
