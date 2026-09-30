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


def test_average_precision_matches_scikit_learn_with_ties():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = random.Random(7)
    labels = [int(rng.random() < 0.3) for _ in range(400)]
    # Two-decimal scores force many tied blocks.
    scores = [round(rng.random(), 2) for _ in labels]
    expected = sklearn_metrics.average_precision_score(labels, scores)
    assert score_metrics(labels, scores)["pr_auc_average_precision"] == pytest.approx(expected, abs=1e-6)


def test_calibration_fit_recovers_identity_for_calibrated_scores():
    # Each score block's observed rate equals its score, so the maximum
    # likelihood fit of label ~ logit(score) is intercept 0 and slope 1.
    labels = [1] * 2 + [0] * 8 + [1] * 5 + [0] * 5 + [1] * 8 + [0] * 2
    scores = [0.2] * 10 + [0.5] * 10 + [0.8] * 10
    result = score_metrics(labels, scores)
    assert result["calibration_fit_status"] == "fitted"
    assert result["calibration_intercept"] == pytest.approx(0.0, abs=1e-5)
    assert result["calibration_slope"] == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize(
    ("labels", "scores", "status"),
    [
        ([1, 1, 1], [0.2, 0.5, 0.9], "not_identifiable_single_class"),
        ([1, 0, 1], [0.4, 0.4, 0.4], "not_identifiable_constant_scores"),
        ([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], "not_identifiable_perfect_separation"),
    ],
)
def test_calibration_fit_reports_non_identifiable_cases_as_null(labels, scores, status):
    result = score_metrics(labels, scores)
    assert result["calibration_fit_status"] == status
    assert result["calibration_intercept"] is None
    assert result["calibration_slope"] is None


def test_prediction_csv_has_one_row_per_model_and_test_window(tmp_path):
    np = pytest.importorskip("numpy")
    from tools.benchmark_cauren_bridge_models import _write_test_predictions

    metadata = [
        {"asset_id": "CA_1", "state": "CA", "split": "train"},
        {"asset_id": "CA_2", "state": "CA", "split": "test"},
        {"asset_id": "IA_1", "state": "IA", "split": "test"},
    ]
    test_mask = np.array([False, True, True])
    hybrid = np.array([np.nan, 0.7, 0.2])
    output = tmp_path / "predictions.csv"
    _write_test_predictions(
        output,
        ["w0", "w1", "w2"],
        metadata,
        np.array([0.0, 1.0, 0.0]),
        test_mask,
        {"train_prevalence": np.full(3, 0.3), "saved_hybrid_checkpoint": hybrid},
    )
    rows = output.read_text(encoding="utf-8").splitlines()[1:]
    keys = [(row.split(",")[0], row.split(",")[5]) for row in rows]
    assert sorted(keys) == sorted(
        (window, model) for window in ("w1", "w2") for model in ("train_prevalence", "saved_hybrid_checkpoint")
    )


def test_same_input_slice_refuses_a_reordered_feature_matrix():
    pytest.importorskip("numpy")
    from tools.benchmark_cauren_bridge_models import FEATURE_SUMMARIES, _core_history_width

    features = ("structural_risk_score", "inspection_finding_score")
    names = [f"{feature}__{summary}" for feature in features for summary in FEATURE_SUMMARIES]
    assert _core_history_width(names + ["bridge_age_at_anchor_years"], features) == len(names)
    with pytest.raises(ValueError, match="core feature summaries"):
        _core_history_width(["bridge_age_at_anchor_years"] + names, features)


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
