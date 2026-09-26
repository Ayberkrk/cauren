import json
import random

import pytest

from tools.bridge_model_metrics import score_metrics


def test_average_precision_matches_the_step_definition():
    # Ranked: 1, 0, 1, 0 -> recall steps 0.5 at precision 1 and 0.5 at precision 2/3.
    result = score_metrics([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1])
    assert result["pr_auc_average_precision"] == pytest.approx(0.5 * 1.0 + 0.5 * 2 / 3, abs=1e-6)


def test_average_precision_groups_tied_scores():
    assert score_metrics([1, 0], [0.5, 0.5])["pr_auc_average_precision"] == pytest.approx(0.5)
    assert score_metrics([0, 1], [0.5, 0.5])["pr_auc_average_precision"] == pytest.approx(0.5)


def test_brier_and_threshold_metrics():
    result = score_metrics([1, 0, 1, 0], [0.8, 0.6, 0.4, 0.2], threshold=0.5)
    assert result["brier_score"] == pytest.approx((0.04 + 0.36 + 0.36 + 0.04) / 4, abs=1e-6)
    threshold = result["threshold_metrics"]
    assert (threshold["true_positives"], threshold["false_alarms"], threshold["missed_deteriorations"]) == (1, 1, 1)


def test_calibration_error_of_a_constant_predictor_does_not_depend_on_row_order():
    labels = [1] * 26 + [0] * 74
    shuffled = labels[:]
    random.Random(0).shuffle(shuffled)
    for order in (labels, labels[::-1], shuffled):
        ece = score_metrics(order, [0.30] * len(order))["expected_calibration_error_10_equal_frequency_bins"]
        assert ece == pytest.approx(0.04, abs=1e-6)


def test_calibration_error_is_zero_for_a_perfectly_calibrated_predictor():
    labels = [0] * 50 + [1] * 50
    scores = [0.0] * 50 + [1.0] * 50
    assert score_metrics(labels, scores)["expected_calibration_error_10_equal_frequency_bins"] == 0.0


def test_capacity_cut_inside_a_tie_reports_expected_captures():
    labels = [1, 0] + [0] * 8
    scores = [0.9, 0.9] + [0.1] * 8
    top = score_metrics(labels, scores)["inspection_capacity"]["top_10_percent"]
    assert top["bridges_reviewed"] == 1
    assert top["deteriorations_captured_expected"] == pytest.approx(0.5)
    assert top["false_alarms_expected"] == pytest.approx(0.5)


def test_non_finite_scores_are_excluded_and_lengths_must_match():
    assert score_metrics([1, 0], [0.9, float("nan")])["labeled_windows"] == 1
    with pytest.raises(ValueError):
        score_metrics([1, 0], [0.5])


def _write_readings(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "window_id,timestamp,sensor_name,value,quality,metadata_json\n"
    lines = []
    for window_id, year, value, anchor_year in rows:
        metadata = json.dumps({"anchor_year": anchor_year, "year_built": 1980.0}).replace('"', '""')
        lines.append(f'{window_id},{year}-07-01T00:00:00Z,structural_risk_score,{value},true,"{metadata}"\n')
    path.write_text(header + "".join(lines), encoding="utf-8")


def test_benchmark_summarizes_each_window_up_to_its_anchor_year(tmp_path):
    pytest.importorskip("numpy")
    from tools.benchmark_cauren_bridge_models import _load_core_history

    readings = tmp_path / "agents" / "cauren-bridge" / "raw_sensor_readings.csv"
    _write_readings(readings, [("w1", 2010, 0.2, 2012), ("w1", 2012, 0.4, 2012)])
    names, vectors = _load_core_history(tmp_path, {"w1": {"seq_len": 2}}, ("structural_risk_score",))
    summary = dict(zip(names, vectors["w1"]))
    assert summary["structural_risk_score__last"] == pytest.approx(0.4)
    assert summary["structural_risk_score__slope_per_year"] == pytest.approx(0.1)
    assert summary["bridge_age_at_anchor_years"] == pytest.approx(32.0)


def test_benchmark_rejects_readings_after_the_anchor_year(tmp_path):
    pytest.importorskip("numpy")
    from tools.benchmark_cauren_bridge_models import _load_core_history

    readings = tmp_path / "agents" / "cauren-bridge" / "raw_sensor_readings.csv"
    _write_readings(readings, [("w1", 2010, 0.2, 2012), ("w1", 2014, 0.4, 2012)])
    with pytest.raises(ValueError, match="after its anchor year"):
        _load_core_history(tmp_path, {"w1": {"seq_len": 2}}, ("structural_risk_score",))


def test_benchmark_rejects_non_contiguous_window_rows(tmp_path):
    pytest.importorskip("numpy")
    from tools.benchmark_cauren_bridge_models import _load_core_history

    readings = tmp_path / "agents" / "cauren-bridge" / "raw_sensor_readings.csv"
    # Each fragment of w1 passes the per-window year count check on its own,
    # so only the contiguity check can notice that w1 was split.
    _write_readings(readings, [("w1", 2010, 0.2, 2012), ("w2", 2010, 0.3, 2012), ("w1", 2010, 0.4, 2012)])
    with pytest.raises(ValueError, match="not contiguous"):
        _load_core_history(tmp_path, {"w1": {"seq_len": 1}, "w2": {"seq_len": 1}}, ("structural_risk_score",))
