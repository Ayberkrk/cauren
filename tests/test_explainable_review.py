from pathlib import Path

import pytest

from cauren_physics.oma import identify_modal_parameters
from cauren_agents.civil.agent import build_civil_agent
from cauren_core import CaurenPipeline
from cauren_core.explanations import render_diagnosis_explanation
from tools.run_explainable_review import (
    build_oma_frequency_drift,
    load_sensors_json,
    load_vibration_columns,
    load_vibration_series,
    load_wide_csv_rows,
    run_review,
    sensors_from_wide_row,
)

REAL_PUBLIC_CSV = Path("data/public_sources/normalized/civil_public_core_assets.csv")


def test_render_diagnosis_explanation_includes_data_trust_when_available():
    payload = {
        "anomaly_type": "drift",
        "anomaly_family": "structural_shift",
        "risk_score": 0.7,
        "confidence": 0.6,
        "selected_agent": "cauren-civil",
        "physics_evidence": {"agent_id": "cauren-civil", "risk_contribution": 0.7, "relations": [], "missing_features": [], "notes": []},
        "agent_reasoning": "test reasoning",
        "recommended_actions": [],
        "quality_control": {
            "status": "warn",
            "score": 0.8,
            "findings": [{"code": "value_out_of_range", "severity": "warn", "feature": "x", "message": "x is out of range"}],
        },
        "uncertainty_report": {
            "uncertainty_score": 0.3,
            "lower_bound": 0.5,
            "upper_bound": 0.9,
            "factors": [
                {"name": "data_quality", "contribution": 0.2},
                {"name": "model_confidence", "contribution": 0.1},
            ],
        },
    }

    explanation = render_diagnosis_explanation(payload)

    assert explanation["data_trust"]["available"] is True
    assert explanation["data_trust"]["quality_status"] == "warn"
    assert explanation["data_trust"]["uncertainty_score"] == 0.3
    assert explanation["data_trust"]["risk_score_band"] == [0.5, 0.9]
    assert explanation["data_trust"]["leading_uncertainty_factor"] == "data_quality"
    assert explanation["data_trust"]["top_quality_findings"] == ["x is out of range"]


def test_render_diagnosis_explanation_marks_data_trust_unavailable_when_absent():
    payload = {
        "anomaly_type": "drift",
        "anomaly_family": "structural_shift",
        "risk_score": 0.5,
        "confidence": 0.5,
        "selected_agent": "cauren-civil",
        "physics_evidence": {"agent_id": "cauren-civil", "risk_contribution": 0.5, "relations": [], "missing_features": [], "notes": []},
        "agent_reasoning": "",
        "recommended_actions": [],
    }

    explanation = render_diagnosis_explanation(payload)

    assert explanation["data_trust"] == {"available": False}


def test_sensors_from_wide_row_uses_schema_feature_order_and_units():
    schema = build_civil_agent().schema
    row = {
        "asset_id": "row-1",
        "structural_risk_score": "0.82",
        "construction_progress_pct": "41",
        "footprint_area_m2": "",  # blank optional column -> skipped
    }

    sensors = sensors_from_wide_row(row, schema, timestamp=1.0)
    by_name = {item["name"]: item for item in sensors}

    assert by_name["structural_risk_score"]["value"] == 0.82
    assert by_name["structural_risk_score"]["unit"] == "ratio"
    assert by_name["construction_progress_pct"]["unit"] == "%"
    assert "footprint_area_m2" not in by_name


def test_load_vibration_series_reads_a_clean_numeric_column(tmp_path):
    path = tmp_path / "vibration.csv"
    path.write_text("value\n1.0\n2.5\n-3\n", encoding="utf-8")

    assert load_vibration_series(path, "value") == [1.0, 2.5, -3.0]


def test_load_vibration_series_rejects_blank_single_column_sample(tmp_path):
    path = tmp_path / "single.csv"
    path.write_text("value\n1.0\n\n3.0\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        load_vibration_series(path, "value")

    assert f"{path}:3:" in str(exc.value)


def test_load_vibration_columns_rejects_blank_multicolumn_cell(tmp_path):
    path = tmp_path / "multiple.csv"
    path.write_text("left,right\n1.0,2.0\n3.0,\n5.0,6.0\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        load_vibration_columns(path, ["right", "left"])

    assert f"{path}:3:" in str(exc.value)


def test_load_vibration_series_rejects_non_finite_values_with_line_number(tmp_path):
    path = tmp_path / "non_finite.csv"
    path.write_text("value\n1.0\ninf\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        load_vibration_series(path, "value")

    assert f"{path}:3:" in str(exc.value)


def test_load_vibration_columns_keeps_requested_channel_order(tmp_path):
    path = tmp_path / "columns.csv"
    path.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")

    assert load_vibration_columns(path, ["c", "a"]) == [[3.0, 6.0], [1.0, 4.0]]


def test_multichannel_review_requires_timoshenko_when_engine_is_unavailable(tmp_path, monkeypatch):
    import tools.run_explainable_review as review

    path = tmp_path / "channels.csv"
    path.write_text("a,b\n1,2\n2,3\n", encoding="utf-8")
    monkeypatch.setattr(review, "identify_multichannel", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit, match="multi-channel FDD requires timoshenko-engine 2.0 or newer"):
        build_oma_frequency_drift(
            path,
            columns=["a", "b"],
            sampling_hz=100.0,
            baseline_frequencies_hz=None,
        )


def test_multichannel_baseline_comparison_keeps_drift_report_separate(tmp_path, monkeypatch):
    import tools.run_explainable_review as review

    path = tmp_path / "channels.csv"
    path.write_text("a,b\n1,2\n2,3\n", encoding="utf-8")
    result = identify_modal_parameters([0.0, 1.0, 0.0, -1.0] * 64, 100.0)
    result = result.__class__(
        result.modal_parameters,
        result.sampling_hz,
        result.num_samples,
        "fdd",
        result.frequency_resolution_hz,
        mode_shapes=({"frequency_hz": 25.0, "shape": {}},),
    )
    monkeypatch.setattr(review, "identify_multichannel", lambda *args, **kwargs: result)

    report = build_oma_frequency_drift(
        path,
        columns=["a", "b"],
        sampling_hz=100.0,
        baseline_frequencies_hz=[25.0],
    )

    assert report["oma_result"]["method"] == "fdd"
    assert report["oma_result"]["mode_shapes"]
    assert "findings" in report


def test_load_vibration_series_reports_the_offending_line_for_bad_data(tmp_path):
    path = tmp_path / "vibration.csv"
    path.write_text("value\n1.0\nBROKEN\n3.0\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        load_vibration_series(path, "value")

    message = str(exc.value)
    assert ":3:" in message, "the error should point at the offending CSV line"
    assert "BROKEN" in message


def test_load_vibration_series_rejects_a_missing_or_empty_column(tmp_path):
    missing = tmp_path / "missing.csv"
    missing.write_text("other\n1.0\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        load_vibration_series(missing, "value")
    assert "not found" in str(exc.value)

    empty = tmp_path / "empty.csv"
    empty.write_text("value\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        load_vibration_series(empty, "value")
    assert "no numeric samples" in str(exc.value)


def test_load_sensors_json_accepts_both_a_bare_list_and_a_wrapped_object(tmp_path):
    reading = {"sensor_id": "s1", "name": "structural_risk_score", "unit": "ratio", "value": 0.5}

    bare = tmp_path / "bare.json"
    bare.write_text(f"[{reading!r}]".replace("'", '"'), encoding="utf-8")
    assert load_sensors_json(bare) == [reading]

    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(('{"sensors": [' + f"{reading!r}" + "]}").replace("'", '"'), encoding="utf-8")
    assert load_sensors_json(wrapped) == [reading]


def test_load_sensors_json_rejects_an_unusable_shape(tmp_path):
    path = tmp_path / "wrong.json"
    path.write_text('{"not_sensors": 1}', encoding="utf-8")

    with pytest.raises(SystemExit):
        load_sensors_json(path)


def test_run_explainable_review_on_real_public_dataset_rows_reacts_to_real_data():
    """Runs the general review pipeline (not a fixed scenario) against real
    rows from data/public_sources/normalized/civil_public_core_assets.csv --
    a higher-risk bridge asset and a lower-risk building asset should come
    out with genuinely different diagnoses, proving the pipeline is reading
    the uploaded data rather than reproducing one canned output.
    """
    assert REAL_PUBLIC_CSV.exists(), "expected the repo's real public civil dataset to be present"
    rows = {row["asset_id"]: row for row in load_wide_csv_rows(REAL_PUBLIC_CSV)}
    assert "us-bridge-0001" in rows and "us-building-0002" in rows

    pipeline = CaurenPipeline.from_default_registry()
    schema = build_civil_agent().schema

    bridge_report = run_review(
        pipeline,
        sensors_from_wide_row(rows["us-bridge-0001"], schema),
        asset_id="us-bridge-0001",
    )
    building_report = run_review(
        pipeline,
        sensors_from_wide_row(rows["us-building-0002"], schema),
        asset_id="us-building-0002",
    )

    bridge_diag = bridge_report["diagnosis"]
    building_diag = building_report["diagnosis"]

    # Real, different inputs -> genuinely different risk scores, not a
    # fixed/repeated number.
    assert bridge_diag["risk_score"] != building_diag["risk_score"]
    # The bridge row's high structural_risk_score (0.82) + natural_hazard
    # (0.66) should read as materially riskier than the building row's low
    # scores across the board.
    assert bridge_diag["risk_score"] > building_diag["risk_score"]

    for report in (bridge_report, building_report):
        assert "quality_control" in report["diagnosis"]
        assert "uncertainty_report" in report["diagnosis"]
        band = report["diagnosis"]["uncertainty_report"]
        assert band["lower_bound"] <= band["point_estimate"] <= band["upper_bound"]
        assert report["explainable_review"]["data_trust"]["available"] is True
