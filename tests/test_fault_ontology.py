from __future__ import annotations

from api.fault_ontology import canonicalize_fault


def test_fault_ontology_maps_known_pressure_fault_to_family_and_subtype() -> None:
    result = canonicalize_fault(
        root_cause_label="FAULT_PRESSURE_TRANSIENT",
        failure_type="Physics Violation",
        primary_dimension="internal_pressure",
        fault_name="Internal pressure instability",
        fault_descriptor="pressure transient with regulator lag",
        top3_candidates=[{"label": "FAULT_PRESSURE_TRANSIENT", "probability": 0.82}],
    )

    assert result.fault_family_id == "pressure_instability"
    assert result.fault_family_label == "Pressure Instability"
    assert result.fault_subtype_id == "fault_pressure_transient"
    assert result.fault_subtype_label == "Pressure Transient"
    assert result.known_status == "known"
    assert result.review_required is False


def test_fault_ontology_marks_unknown_with_similarity_when_candidates_exist() -> None:
    result = canonicalize_fault(
        root_cause_label="UNKNOWN_FALLBACK",
        failure_type="Pattern Drift",
        primary_dimension="magnetic",
        fault_name="unmapped pattern",
        fault_descriptor="oscillating magnetic deviation",
        top3_candidates=[{"label": "VALVE-LEAK", "probability": 0.41}],
        is_unknown=True,
    )

    assert result.fault_family_id == "unknown_anomaly"
    assert result.known_status == "unknown_with_similarity"
    assert result.review_required is True
    assert result.similar_family_candidates
