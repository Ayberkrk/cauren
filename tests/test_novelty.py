"""Synthetic checks for the shared novelty-detection protocol.

Every test uses generated data with a known answer, so none needs a
monitoring dataset.
"""

from __future__ import annotations

import datetime as dt

import pytest

np = pytest.importorskip("numpy", reason="cauren_physics/novelty.py is numpy based")

from cauren_physics import novelty

STEEL_DF_OVER_F_PER_K = -1.5e-4


def test_select_modes_uses_reference_availability_only():
    frequencies = np.ones((10, 3))
    frequencies[:5, 1] = np.nan  # half missing in the reference rows
    frequencies[5:, 2] = np.nan  # missing only outside the reference rows
    reference = np.arange(10) < 5
    assert novelty.select_modes(frequencies, reference, max_missing=0.1) == [0, 2]


def test_temperature_features_are_bilinear_at_the_hinge():
    features = novelty.temperature_features([-4.0, 0.0, 6.0], hinge_c=0.0)
    np.testing.assert_allclose(features, [[-4.0, -4.0], [0.0, 0.0], [6.0, 0.0]])


def test_temperature_baseline_recovers_bilinear_slopes():
    rng = np.random.default_rng(0)
    temperature = rng.uniform(-5.0, 20.0, 400)
    features = novelty.temperature_features(temperature)
    # Mode 1 at 2 Hz: -0.0004 Hz/K above 0 C and -0.01 Hz/K below it.
    frequencies = np.column_stack(
        (
            2.0 - 4e-4 * temperature - 9.6e-3 * np.minimum(temperature, 0.0),
            5.0 - 1e-3 * temperature,
        )
    )
    baseline = novelty.fit_temperature_baseline(features, frequencies + rng.normal(0.0, 1e-5, frequencies.shape))
    np.testing.assert_allclose(baseline.coefficients, [[-4e-4, -9.6e-3], [-1e-3, 0.0]], atol=2e-5)
    np.testing.assert_allclose(baseline.residuals(features, frequencies), 0.0, atol=5e-5)

    checks = novelty.temperature_sensitivity_check(baseline, expected_df_over_f=STEEL_DF_OVER_F_PER_K)
    assert all(row["sign_ok"] for row in checks)
    assert checks[0]["df_over_f_per_k_below"] < checks[0]["df_over_f_per_k_above"] < 0.0


def test_temperature_sensitivity_flags_a_stiffening_slope():
    baseline = novelty.LinearBaseline([10.0, 0.0], [3.0], [[3e-4, 0.0]], "test")
    (row,) = novelty.temperature_sensitivity_check(baseline, expected_df_over_f=STEEL_DF_OVER_F_PER_K)
    assert row["sign_ok"] is False
    assert row["ratio_to_material_only"] == pytest.approx(1e-4 / STEEL_DF_OVER_F_PER_K)


def test_ledoit_wolf_covariance_is_invertible_with_few_samples():
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(4, 6))
    covariance = novelty.ledoit_wolf_covariance(samples)
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)
    np.testing.assert_allclose(covariance, covariance.T)


def test_ledoit_wolf_covariance_approaches_sample_covariance_with_many_samples():
    rng = np.random.default_rng(2)
    true = np.array([[1.0, 0.8], [0.8, 1.0]])
    samples = rng.multivariate_normal([0.0, 0.0], true, size=20000)
    np.testing.assert_allclose(novelty.ledoit_wolf_covariance(samples), true, atol=0.05)


def test_mahalanobis_detector_scores_shifted_samples_higher():
    rng = np.random.default_rng(3)
    training = rng.normal(size=(500, 3))
    detector = novelty.MahalanobisDetector(training)
    healthy, shifted = detector.score(np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]))
    assert healthy < 0.2 < 3.5 < shifted


def test_pca_residual_detector_separates_pattern_breaks_from_common_variation():
    rng = np.random.default_rng(4)
    factor = rng.normal(size=400)
    training = np.column_stack((factor, 2.0 * factor, -factor)) + rng.normal(0.0, 0.05, (400, 3))
    detector = novelty.PCAResidualDetector(training, retained_variance=0.9)
    assert detector.retained == 1
    along, across = detector.score(np.array([[3.0, 6.0, -3.0], [1.0, -2.0, 1.0]]))
    assert along < 0.3 < 1.5 < across


def test_daily_medians_skip_sparse_days_and_alarm_summary_counts_delay():
    start = dt.datetime(2019, 9, 28)
    times = [start + dt.timedelta(hours=hour) for hour in range(72)]
    scores = [1.0] * 24 + [5.0] * 24 + [5.0] * 3 + [float("nan")] * 21
    daily = novelty.daily_medians(times, scores, min_hours=6)
    assert list(daily) == [start.date(), start.date() + dt.timedelta(days=1)]

    # 199 reference scores of 0..1.98: the conformal limit at alpha 0.01 is
    # the second largest, so 1.0 stays quiet and 5.0 alarms.
    reference = [index / 100.0 for index in range(199)]
    summary = novelty.alarm_summary(daily, reference, 0.01, (start, start + dt.timedelta(days=3)))
    assert summary["days"] == 2
    assert summary["alarmed_days"] == 1
    assert summary["delay_days"] == 1
    assert summary["min_p_value"] == pytest.approx(1 / 200)
    assert summary["max_over_limit"] == pytest.approx(5.0 / 1.97)


def test_frequency_ratio_check_distinguishes_global_from_local_change():
    reference = np.array([2.0, 3.0, 4.0])
    uniform = novelty.frequency_ratio_check(reference, reference * 1.01)
    assert uniform["pattern"] == "global"
    np.testing.assert_allclose(uniform["ratio_changes"], 0.0, atol=1e-12)

    local = novelty.frequency_ratio_check(reference, reference * np.array([0.997, 1.02, 0.993]))
    assert local["pattern"] == "local"
    assert local["same_sign"] is False


def test_run_protocol_detects_a_local_change_smaller_than_the_seasonal_swing():
    rng = np.random.default_rng(5)
    start = dt.datetime(2020, 1, 1)
    times = [start + dt.timedelta(hours=hour) for hour in range(24 * 200)]
    days = np.arange(len(times)) / 24.0
    temperature = 8.0 + 8.0 * np.sin(2.0 * np.pi * days / 60.0) + rng.normal(0.0, 0.5, len(times))
    base = np.array([2.0, 3.5, 5.0])
    frequencies = base * (1.0 - 4e-4 * temperature[:, None]) + rng.normal(0.0, 5e-4, (len(times), 3))
    protocol = novelty.StudyProtocol(
        reference=(start, start + dt.timedelta(days=120)),
        holdout=(start + dt.timedelta(days=120), start + dt.timedelta(days=160)),
        changed=(start + dt.timedelta(days=160), start + dt.timedelta(days=200)),
    )
    changed = np.array([stamp >= protocol.changed[0] for stamp in times])
    # A 0.5% drop of one mode is smaller than the 1.3% seasonal swing.
    frequencies[changed, 1] *= 0.995
    frequencies[5, 0] = np.nan

    result = novelty.run_protocol(
        times, frequencies, temperature, protocol, expected_df_over_f=STEEL_DF_OVER_F_PER_K
    )
    with pytest.raises(ValueError, match="needs at least 999 reference days"):
        novelty.run_protocol(
            times, frequencies, temperature, protocol, expected_df_over_f=STEEL_DF_OVER_F_PER_K, alpha=0.001
        )

    assert result["reference_days"] == 120
    assert result["p_value_floor"] == pytest.approx(1 / 121)
    assert result["modes"] == [0, 1, 2]
    assert result["complete_hours"]["reference"] == 24 * 120 - 1
    normalized = result["arms"]["temperature_normalized"]
    assert normalized["holdout"]["alarmed_days"] == 0
    assert normalized["changed"]["alarm_rate"] == 1.0
    assert normalized["changed"]["delay_days"] == 0
    assert normalized["changed"]["min_p_value"] == pytest.approx(1 / 121)
    assert len(normalized["daily"]) == 200
    assert {"cointegration", "knn", "gp_normalized", "daily_mean_normalized"} <= set(result["arms"])
    assert "environment_normalized" not in result["arms"]
    assert result["arms"]["cointegration"]["cusum"]["threshold"] == result["cusum_threshold"]

    humidity = rng.uniform(40.0, 90.0, len(times))
    with_environment = novelty.run_protocol(
        times, frequencies, temperature, protocol, expected_df_over_f=STEEL_DF_OVER_F_PER_K, environment=humidity
    )
    assert with_environment["arms"]["environment_normalized"]["changed"]["alarm_rate"] == 1.0
    assert result["frequency_ratio"]["pattern"] == "local"
    assert all(row["sign_ok"] for row in result["temperature_sensitivity"])
    assert "Post-change days alarmed" in novelty.markdown_table(result)


def test_conformal_p_values_count_reference_scores_at_or_above():
    reference = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    p_values = novelty.conformal_p_values(reference, [10.0, 9.0, 5.0, 0.0])
    np.testing.assert_allclose(p_values, [1 / 10, 2 / 10, 6 / 10, 1.0])


def test_conformal_limit_alarms_exactly_where_p_is_at_most_alpha():
    rng = np.random.default_rng(6)
    reference = rng.normal(size=150)
    scores = np.concatenate((rng.normal(size=500), np.sort(reference)))
    for alpha in (0.01, 0.05, 0.2):
        limit = novelty.conformal_limit(reference, alpha)
        by_p = novelty.conformal_p_values(reference, scores) <= alpha + 1e-12
        np.testing.assert_array_equal(scores > limit, by_p)
    assert novelty.conformal_limit(np.arange(1.0, 100.0), 0.01) == 99.0
    assert novelty.conformal_limit(np.arange(1.0, 100.0), 0.05) == 95.0


def test_conformal_p_values_hold_the_false_alarm_rate_on_exchangeable_days():
    rng = np.random.default_rng(7)
    trials = 20000
    reference = rng.normal(size=(trials, 99))
    new_day = rng.normal(size=trials)
    alarms = [novelty.conformal_p_values(row, [value])[0] <= 0.01 for row, value in zip(reference, new_day)]
    # The exact rate is 1/100; the bound allows for Monte Carlo noise.
    assert np.mean(alarms) < 0.013


def test_check_alpha_rejects_a_reference_period_too_short_for_alpha():
    novelty.check_alpha(99, 0.01)
    with pytest.raises(ValueError, match="needs at least 99 reference days"):
        novelty.check_alpha(98, 0.01)
    with pytest.raises(ValueError, match="alpha must be in"):
        novelty.check_alpha(500, 0.0)


def test_leave_one_out_scores_exceed_in_sample_scores_on_a_small_reference():
    rng = np.random.default_rng(8)
    reference = rng.normal(size=(25, 5))
    in_sample = novelty.MahalanobisDetector(reference).score(reference)
    held_out = novelty.leave_one_out_scores(novelty.MahalanobisDetector, reference)
    assert held_out.shape == (25,)
    assert np.all(held_out > in_sample)


def test_evaluate_states_separates_a_shifted_state_and_keeps_holdout_quiet():
    rng = np.random.default_rng(9)
    covariance = np.array([[1.0, 0.6, 0.0], [0.6, 1.0, 0.0], [0.0, 0.0, 1.0]])
    reference = rng.multivariate_normal(np.zeros(3), covariance, size=40)
    holdout = rng.multivariate_normal(np.zeros(3), covariance, size=400)
    shifted = rng.multivariate_normal([0.0, 0.0, 5.0], covariance, size=30)
    # Breaks the correlation of the first two features without moving either far.
    pattern_break = rng.multivariate_normal([3.0, -3.0, 0.0], covariance, size=30)

    result = novelty.evaluate_states(
        reference, holdout, {"shifted": shifted, "pattern_break": pattern_break}, alpha=0.05
    )

    assert result["reference_samples"] == 40
    assert result["p_value_floor"] == pytest.approx(1 / 41)
    raw = result["arms"]["raw_frequencies"]
    assert raw["holdout"]["alarm_rate"] < 0.10
    assert raw["shifted"]["alarm_rate"] > 0.9
    assert raw["pattern_break"]["alarm_rate"] > 0.9
    assert raw["shifted"]["median_p_value"] == pytest.approx(1 / 41)
    with pytest.raises(ValueError, match="needs at least 99 reference days"):
        novelty.evaluate_states(reference, holdout, {}, alpha=0.01)


def test_knn_detector_follows_a_curved_healthy_pattern_that_mahalanobis_cannot():
    rng = np.random.default_rng(10)
    x = rng.uniform(-2.0, 2.0, 600)
    training = np.column_stack((x, x**2)) + rng.normal(0.0, 0.05, (600, 2))
    on_curve_far = np.array([[1.9, 3.61]])
    off_curve_centre = np.array([[0.0, 1.5]])
    knn = novelty.KNNDetector(training)
    mahalanobis = novelty.MahalanobisDetector(training)
    assert knn.score(off_curve_centre)[0] > 5.0 * knn.score(on_curve_far)[0]
    assert mahalanobis.score(off_curve_centre)[0] < mahalanobis.score(on_curve_far)[0]


def test_gaussian_process_baseline_learns_the_freezing_bend():
    rng = np.random.default_rng(11)
    temperature = rng.uniform(-6.0, 22.0, 3000)

    def truth(t):
        return 2.0 - 4e-4 * t - 9.6e-3 * np.minimum(t, 0.0)

    frequencies = truth(temperature)[:, None] + rng.normal(0.0, 2e-3, (3000, 1))
    baseline = novelty.GaussianProcessBaseline(temperature, frequencies)
    grid = np.array([-5.0, -2.0, 0.0, 5.0, 15.0])
    # A smooth kernel rounds the kink at 0 C slightly; the error stays within the noise level.
    np.testing.assert_allclose(baseline.predict(grid)[:, 0], truth(grid), atol=2e-3)
    assert np.std(baseline.residuals(temperature, frequencies)) == pytest.approx(2e-3, rel=0.1)


def test_identification_uncertainty_separates_hourly_noise_from_slow_drift():
    rng = np.random.default_rng(12)
    start = dt.datetime(2020, 1, 1)
    times = [start + dt.timedelta(hours=hour) for hour in range(5000) if hour % 97 != 0]
    slow = 0.004 * np.sin(2.0 * np.pi * np.arange(len(times)) / 500.0)
    noise = rng.normal(0.0, 0.002, len(times))
    result = novelty.identification_uncertainty(times, (slow + noise)[:, None], [2.0])
    assert result["identification_sigma_hz"][0] == pytest.approx(0.002, rel=0.05)
    assert result["identification_sigma_relative"][0] == pytest.approx(0.001, rel=0.05)
    # Noise variance 4e-6 against a total of 4e-6 + 8e-6.
    assert result["noise_share"][0] == pytest.approx(1 / 3, abs=0.03)
    assert result["hourly_pairs"] < len(times) - 1


def test_cointegration_detector_follows_a_common_trend_and_flags_a_broken_relation():
    rng = np.random.default_rng(13)
    trend = np.cumsum(rng.normal(0.0, 0.05, 3001))
    levels = np.column_stack((trend, 2.0 * trend)) + rng.normal(0.0, 0.01, (3001, 2))
    detector = novelty.CointegrationDetector(levels[:-1], levels[1:])
    far_along_trend = np.array([[trend.max() + 1.0, 2.0 * (trend.max() + 1.0)]])
    # The stationary combination y2 - 2 y1 has a noise sigma of about 0.022.
    broken = np.array([[trend.mean(), 2.0 * trend.mean() + 0.4]])
    assert detector.score(far_along_trend)[0] < 3.0 < 10.0 < detector.score(broken)[0]
    assert novelty.MahalanobisDetector(levels).score(far_along_trend)[0] > 3.0


def test_cusum_threshold_and_first_alarm():
    threshold = novelty.cusum_threshold(130, runs=500)
    assert 0.0 < threshold < novelty.cusum_threshold(130, arl0_days=1000, runs=500)
    start = dt.datetime(2020, 1, 1)
    days = [start.date() + dt.timedelta(days=offset) for offset in range(30)]
    quiet = novelty.cusum_first_alarm(dict.fromkeys(days, 0.5), (start, start + dt.timedelta(days=30)), threshold)
    assert quiet["first_alarm"] is None
    # Each day at the p-value floor adds -ln(1/131) - 1.5, about 3.4.
    strong = novelty.cusum_first_alarm(dict.fromkeys(days, 1 / 131), (start, start + dt.timedelta(days=30)), 5.0)
    assert strong["delay_days"] == 1


def test_max_deviation_detector_sees_a_change_in_one_of_many_features():
    rng = np.random.default_rng(14)
    training = rng.normal(size=(40, 30))
    local = np.zeros((1, 30))
    local[0, 7] = 6.0
    spread = np.full((1, 30), 1.2)
    detector = novelty.MaxDeviationDetector(training)
    assert detector.score(local)[0] > 4.0 > detector.score(spread)[0]


def test_segment_mask_is_half_open():
    times = [
        dt.datetime(2024, 1, 1, 0, 0),  # before the segment
        dt.datetime(2024, 1, 1, 5, 0),  # exactly start: included
        dt.datetime(2024, 1, 1, 6, 0),  # inside
        dt.datetime(2024, 1, 2, 0, 0),  # exactly end: excluded
        dt.datetime(2024, 1, 2, 3, 0),  # after the segment
    ]
    mask = novelty.segment_mask(times, (times[1], times[3]))
    assert mask.dtype == bool
    assert mask.tolist() == [False, True, True, False, False]


def test_segment_mask_empty_times_gives_empty_mask():
    mask = novelty.segment_mask([], (dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)))
    assert mask.dtype == bool
    assert mask.shape == (0,)


def test_segment_mask_accepts_datetime64_arrays():
    times = np.array(
        ["2024-01-01T05:00", "2024-01-01T06:00", "2024-01-02T00:00", "2024-01-02T01:00"],
        dtype="datetime64[m]",
    )
    segment = (dt.datetime(2024, 1, 1, 6, 0), dt.datetime(2024, 1, 2, 0, 0))
    as_datetime64 = novelty.segment_mask(times, segment)
    as_datetimes = [stamp.astype("datetime64[us]").item() for stamp in times]
    assert as_datetime64.dtype == bool
    assert as_datetime64.tolist() == [False, True, False, False]
    assert novelty.segment_mask(as_datetimes, segment).tolist() == as_datetime64.tolist()
