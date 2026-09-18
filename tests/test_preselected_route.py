from cauren_core import CaurenPipeline


def _payload() -> dict:
    return {
        "agent_id": "cauren-civil",
        "sensors": [
            {
                "sensor_id": "structural-risk",
                "name": "structural_risk_score",
                "unit": "ratio",
                "value": 0.25,
                "timestamp": 1.0,
            }
        ],
    }


def _diagnose_with_breakdown(breakdown: dict) -> dict:
    pipeline = CaurenPipeline.from_default_registry()
    diagnosis = pipeline.diagnose_with_preselected_route(
        _payload(),
        selected_agent_id="cauren-civil",
        route_confidence=0.8,
        route_reason="preselected for regression test",
        sector_score_breakdown=[breakdown],
    )
    return diagnosis.to_dict()


def test_preselected_route_preserves_explicit_zero_final_score() -> None:
    result = _diagnose_with_breakdown(
        {
            "agent_id": "cauren-civil",
            "sector": "civil",
            "final_sector_score": 0.0,
            "score": 0.75,
        }
    )

    assert result["sector_scores"][0]["score"] == 0.0
    assert result["sector_score_breakdown"][0]["final_sector_score"] == 0.0


def test_preselected_route_uses_legacy_score_when_final_score_is_missing() -> None:
    result = _diagnose_with_breakdown(
        {
            "agent_id": "cauren-civil",
            "sector": "civil",
            "score": 0.42,
        }
    )

    assert result["sector_scores"][0]["score"] == 0.42
    assert result["sector_score_breakdown"][0]["final_sector_score"] == 0.42
