import math
import random

import pytest

from cauren_core import CaurenPipeline
from cauren_core.risk_calibration import (
    HOLM_ALPHA,
    calibrated_combined_risk,
    chi2_lower_p,
    holm_adjust,
    max_abs_t_p,
    p_value_to_risk,
    sector_threshold,
    student_t_two_sided_p,
)

FEATURES = {
    "structural_risk_score": (0.40, 0.03),
    "inspection_finding_score": (0.30, 0.04),
    "ground_stability_score": (0.50, 0.03),
}
TARGET = "structural_risk_score"


def _baseline(expected_variation=True):
    return {
        name: {"center": center, "scale": scale, "n": 120, "expected_variation": expected_variation, "unit": "ratio"}
        for name, (center, scale) in FEATURES.items()
    }


def _payload(change=None, *, seed=3, baseline=True, expected_variation=True):
    rng = random.Random(seed)
    sensors = []
    for step in range(16):
        for name, (center, scale) in FEATURES.items():
            value = center + rng.gauss(0.0, scale)
            if name == TARGET and change is not None:
                value = change(step, value, center, scale)
            sensors.append({"sensor_id": f"{name}-{step}", "name": name, "unit": "ratio", "value": value, "timestamp": float(step)})
    payload = {"agent_id": "cauren-civil", "sensors": sensors}
    if baseline:
        payload["baseline"] = _baseline(expected_variation)
    return payload


def _diagnose(*args, **kwargs):
    return CaurenPipeline.from_default_registry().diagnose(_payload(*args, **kwargs))


def test_risk_mapping_puts_the_sector_threshold_at_the_holm_alpha():
    threshold = sector_threshold("cauren-civil")
    assert p_value_to_risk(HOLM_ALPHA) == pytest.approx(threshold)
    assert p_value_to_risk(HOLM_ALPHA * 1.01) < threshold < p_value_to_risk(HOLM_ALPHA * 0.99)
    assert p_value_to_risk(1.0) == 0.0 and p_value_to_risk(0.0) == 1.0
    assert p_value_to_risk(HOLM_ALPHA**2) == pytest.approx(1.0 - (1.0 - threshold) ** 2)


def test_holm_adjustment_matches_the_step_down_definition():
    assert holm_adjust([0.01, 0.04, 0.03, 0.5]) == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert holm_adjust([]) == []


def test_tail_probabilities_match_reference_values():
    assert student_t_two_sided_p(2.0, 10) == pytest.approx(0.073388, abs=1e-6)
    assert student_t_two_sided_p(1.0, 1) == pytest.approx(0.5)
    assert student_t_two_sided_p(1.959964, math.inf) == pytest.approx(0.05, abs=1e-6)
    assert chi2_lower_p(3.0, 4) == pytest.approx(0.442175, abs=1e-6)
    assert max_abs_t_p(3.0, 1, math.inf) == pytest.approx(student_t_two_sided_p(3.0, math.inf))
    assert max_abs_t_p(3.0, 16, math.inf) == pytest.approx(1.0 - (1.0 - 0.0026998) ** 16, abs=1e-6)


def test_suppressed_physics_does_not_dilute_core_risk():
    raw, calibrated = calibrated_combined_risk(
        agent_id="cauren-civil", core_risk=0.34, physics_risk=0.0, anomaly_type="spike", pattern_scores={}
    )
    assert raw == pytest.approx(0.34) and calibrated == pytest.approx(0.34)


def test_alarm_depends_on_magnitude_in_reference_scales():
    def step(size):
        return lambda t, value, center, scale: value + (size * scale if t >= 8 else 0.0)

    small = _diagnose(step(0.3))
    large = _diagnose(step(5.0))
    assert small.risk_score < 0.34
    assert large.risk_score >= 0.34
    assert large.anomaly_type == "level_shift"
    assert large.anomaly_family == "change_point_deviation"
    assert large.core_output.evidence_status == "reference_tested"


def test_large_spike_and_drift_are_named():
    spike = _diagnose(lambda t, value, center, scale: value + (8.0 * scale if t == 11 else 0.0))
    assert spike.risk_score >= 0.34 and spike.anomaly_type == "spike"

    drift = _diagnose(lambda t, value, center, scale: value + 6.0 * scale * t / 15.0)
    assert drift.risk_score >= 0.34 and drift.anomaly_type == "drift"
    assert drift.anomaly_family == "progressive_deviation"


def test_flatline_needs_a_feature_that_is_expected_to_vary():
    def stuck(t, value, center, scale):
        return center

    expected = _diagnose(stuck)
    assert expected.anomaly_type == "flatline" and expected.risk_score >= 0.34

    natural = _diagnose(stuck, expected_variation=False)
    assert natural.core_output.pattern_scores["flatline"] == 0.0
    assert natural.risk_score < 0.34


def test_pattern_score_keys_are_unchanged():
    keys = set(_diagnose().core_output.pattern_scores)
    assert keys == {
        "drift", "spike", "dip", "oscillation", "level_shift", "variance_change", "flatline",
        "sensor_dropout", "correlation_break", "repeating_burst", "collective_anomaly",
        "multisensor_residual", "calibration_bias", "slow_degradation", "contextual_anomaly",
        "seasonal_deviation",
    }


def test_no_reference_means_insufficient_evidence_not_an_alarm():
    stuck = _diagnose(lambda t, value, center, scale: value + 10.0 * scale, baseline=False)
    assert stuck.risk_score == 0.0
    assert stuck.risk_score_status == "insufficient_validated_evidence"
    assert stuck.core_output.evidence_status == "insufficient_reference"
    assert stuck.to_dict()["core"]["evidence_status"] == "insufficient_reference"
