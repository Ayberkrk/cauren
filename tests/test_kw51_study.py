"""Synthetic checks for the KW51 case-study helpers.

The real study reads the Zenodo KW51 record, which is not bundled; these
tests cover the helper functions on generated data with known answers.
"""

from __future__ import annotations

import datetime as dt

import pytest

np = pytest.importorskip("numpy", reason="tools/kw51_study.py is numpy based")

from tools import kw51_study as study


def test_datenum_conversion_matches_matlab_epoch():
    # datenum(2018, 10, 1) is 737334 in MATLAB; 0.5 is noon.
    times = study.datenum_to_datetime([737334.0, 737334.5, 737334.0 + 23 / 24 - 1e-9])
    assert times == [dt.datetime(2018, 10, 1), dt.datetime(2018, 10, 1, 12), dt.datetime(2018, 10, 1, 23)]


def test_select_modes_uses_reference_availability_only():
    frequencies = np.ones((10, 3))
    frequencies[:5, 1] = np.nan  # half missing in the reference rows
    frequencies[5:, 2] = np.nan  # missing only outside the reference rows
    reference = np.arange(10) < 5
    assert study.select_modes(frequencies, reference, max_missing=0.1) == [0, 2]


def test_temperature_features_are_bilinear_at_the_hinge():
    features = study.temperature_features([-4.0, 0.0, 6.0], hinge_c=0.0)
    np.testing.assert_allclose(features, [[-4.0, -4.0], [0.0, 0.0], [6.0, 0.0]])


def test_temperature_baseline_recovers_bilinear_slopes():
    rng = np.random.default_rng(0)
    temperature = rng.uniform(-5.0, 20.0, 400)
    features = study.temperature_features(temperature)
    # Mode 1 at 2 Hz: -0.0004 Hz/K above 0 C and -0.01 Hz/K below it.
    frequencies = np.column_stack(
        (
            2.0 - 4e-4 * temperature - 9.6e-3 * np.minimum(temperature, 0.0),
            5.0 - 1e-3 * temperature,
        )
    )
    baseline = study.fit_temperature_baseline(features, frequencies + rng.normal(0.0, 1e-5, frequencies.shape))
    np.testing.assert_allclose(baseline.coefficients, [[-4e-4, -9.6e-3], [-1e-3, 0.0]], atol=2e-5)
    np.testing.assert_allclose(baseline.residuals(features, frequencies), 0.0, atol=5e-5)

    checks = study.temperature_sensitivity_check(baseline)
    assert all(row["sign_ok"] for row in checks)
    assert checks[0]["df_over_f_per_k_below"] < checks[0]["df_over_f_per_k_above"] < 0.0


def test_temperature_sensitivity_flags_a_stiffening_slope():
    baseline = study.LinearBaseline([10.0, 0.0], [3.0], [[3e-4, 0.0]], "test")
    (row,) = study.temperature_sensitivity_check(baseline)
    assert row["sign_ok"] is False
    assert row["ratio_to_material_only"] == pytest.approx(1e-4 / study.EXPECTED_DF_OVER_F_PER_K)


def test_ledoit_wolf_covariance_is_invertible_with_few_samples():
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(4, 6))
    covariance = study.ledoit_wolf_covariance(samples)
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)
    np.testing.assert_allclose(covariance, covariance.T)


def test_ledoit_wolf_covariance_approaches_sample_covariance_with_many_samples():
    rng = np.random.default_rng(2)
    true = np.array([[1.0, 0.8], [0.8, 1.0]])
    samples = rng.multivariate_normal([0.0, 0.0], true, size=20000)
    np.testing.assert_allclose(study.ledoit_wolf_covariance(samples), true, atol=0.05)


def test_mahalanobis_detector_scores_shifted_samples_higher():
    rng = np.random.default_rng(3)
    training = rng.normal(size=(500, 3))
    detector = study.MahalanobisDetector(training)
    healthy, shifted = detector.score(np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]))
    assert healthy < 0.2 < 3.5 < shifted


def test_pca_residual_detector_separates_pattern_breaks_from_common_variation():
    rng = np.random.default_rng(4)
    factor = rng.normal(size=400)
    training = np.column_stack((factor, 2.0 * factor, -factor)) + rng.normal(0.0, 0.05, (400, 3))
    detector = study.PCAResidualDetector(training, retained_variance=0.9)
    assert detector.retained == 1
    along, across = detector.score(np.array([[3.0, 6.0, -3.0], [1.0, -2.0, 1.0]]))
    assert along < 0.3 < 1.5 < across


def test_daily_medians_skip_sparse_days_and_alarm_summary_counts_delay():
    start = dt.datetime(2019, 9, 28)
    times = [start + dt.timedelta(hours=hour) for hour in range(72)]
    scores = [1.0] * 24 + [5.0] * 24 + [5.0] * 3 + [float("nan")] * 21
    daily = study.daily_medians(times, scores, min_hours=6)
    assert list(daily) == [start.date(), start.date() + dt.timedelta(days=1)]

    reference = {dt.date(2019, 1, day): float(day) for day in range(1, 31)}
    limit = study.control_limit(reference, percentile=50.0)
    assert limit == pytest.approx(15.5)

    summary = study.alarm_summary(daily, 2.0, (start, start + dt.timedelta(days=3)))
    assert summary["days"] == 2
    assert summary["alarmed_days"] == 1
    assert summary["delay_days"] == 1
    assert summary["max_over_limit"] == pytest.approx(2.5)


def test_frequency_ratio_check_distinguishes_global_from_local_change():
    reference = np.array([2.0, 3.0, 4.0])
    uniform = study.frequency_ratio_check(reference, reference * 1.01)
    assert uniform["pattern"] == "global"
    np.testing.assert_allclose(uniform["ratio_changes"], 0.0, atol=1e-12)

    local = study.frequency_ratio_check(reference, reference * np.array([0.997, 1.02, 0.993]))
    assert local["pattern"] == "local"
    assert local["same_sign"] is False
