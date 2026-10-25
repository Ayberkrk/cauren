import math
import random
import threading

import pytest

from cauren_core import CaurenPipeline
from cauren_core.baseline import fit_feature_baseline, parse_baselines
from cauren_core.runtime import CaurenCoreRuntime

FEATURE = "structural_risk_score"


def _reference(**overrides):
    item = {"center": 0.4, "scale": 0.03, "n": 120, "expected_variation": True, "unit": "ratio"}
    item.update(overrides)
    return {FEATURE: item}


def _payload(base_value: float = 0.4, baseline=None):
    payload = {
        "agent_id": "cauren-civil",
        "sensors": [
            {
                "sensor_id": f"s{step}",
                "name": FEATURE,
                "unit": "ratio",
                "value": base_value + 0.01 * ((step * 7) % 5),
                "timestamp": float(step),
            }
            for step in range(16)
        ],
    }
    if baseline is not None:
        payload["baseline"] = baseline
    return payload


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"center": math.nan}, "center must be a finite number"),
        ({"center": "high"}, "center must be a finite number"),
        ({"scale": 0.0}, "scale must be > 0"),
        ({"scale": -1.0}, "scale must be > 0"),
        ({"scale": math.inf}, "scale must be a finite number"),
        ({"n": 29}, "n must be an integer >= 30"),
        ({"n": 30.5}, "n must be an integer >= 30"),
        ({"n": True}, "n must be a finite number"),
        ({"expected_variation": "yes"}, "expected_variation must be a boolean"),
    ],
)
def test_invalid_baselines_are_rejected_with_the_offending_field(overrides, message):
    with pytest.raises(ValueError, match=message):
        parse_baselines(_reference(**overrides))


def test_valid_baseline_round_trips():
    parsed = parse_baselines(_reference())
    assert parsed[FEATURE].to_dict() == _reference()[FEATURE]


def test_fit_uses_median_and_normal_consistent_mad():
    values = [10.0 + 0.5 * ((idx * 13) % 9 - 4) for idx in range(45)]
    fitted = fit_feature_baseline(values, unit="mm")
    deviations = sorted(abs(value - fitted.center) for value in values)
    assert fitted.center == sorted(values)[len(values) // 2]
    assert fitted.scale == pytest.approx(1.4826 * deviations[len(deviations) // 2])
    assert fitted.n == 45 and fitted.expected_variation is True and fitted.unit == "mm"


def test_fit_recovers_gaussian_scale():
    rng = random.Random(4)
    fitted = fit_feature_baseline([rng.gauss(5.0, 2.0) for _ in range(4000)])
    assert fitted.center == pytest.approx(5.0, abs=0.1)
    assert fitted.scale == pytest.approx(2.0, rel=0.05)


def test_fit_needs_enough_samples_and_a_scale_for_a_constant_reference():
    with pytest.raises(ValueError, match="at least 30 samples"):
        fit_feature_baseline([1.0] * 29)
    with pytest.raises(ValueError, match="supply min_scale"):
        fit_feature_baseline([0.85] * 60)
    constant = fit_feature_baseline([0.85] * 60, min_scale=0.01)
    assert (constant.center, constant.scale, constant.expected_variation) == (0.85, 0.01, False)


def test_fit_falls_back_to_mean_absolute_deviation_when_mad_is_zero():
    values = [1.0] * 40 + [2.0] * 5
    fitted = fit_feature_baseline(values)
    assert fitted.center == 1.0
    assert fitted.scale == pytest.approx(math.sqrt(math.pi / 2.0) * 5.0 / 45.0)


class _SpyCore(CaurenCoreRuntime):
    def __init__(self):
        super().__init__(backbone=None)
        self.contexts = []

    def classify(self, window, **kwargs):
        self.contexts.append(kwargs.get("context"))
        return super().classify(window, **kwargs)


def test_payload_baseline_reaches_the_core_context_per_call():
    core = _SpyCore()
    pipeline = CaurenPipeline(core=core)

    with_reference = pipeline.diagnose(_payload(baseline=_reference()))
    without_reference = pipeline.diagnose(_payload())

    assert core.contexts[0]["feature_baselines"] == _reference()
    assert core.contexts[0]["reference_source"] == "payload"
    assert core.contexts[1]["feature_baselines"] == {}
    assert core.contexts[1]["reference_source"] == "none"
    assert with_reference.agent_outputs["reference"] == "payload"
    assert without_reference.agent_outputs["reference"] == "none"
    assert without_reference.to_dict()["reference"] == "none"
    assert without_reference.core_output.confidence < with_reference.core_output.confidence


def test_reference_coverage_names_features_without_and_outside_the_schema():
    baseline = {**_reference(), "not_a_civil_feature": _reference()[FEATURE]}
    coverage = CaurenPipeline().diagnose(_payload(baseline=baseline)).agent_outputs["reference_coverage"]
    assert coverage["features_with_reference"] == [FEATURE]
    assert "ground_stability_score" in coverage["features_without_reference"]
    assert coverage["ignored_reference_features"] == ["not_a_civil_feature"]


def test_pipeline_rejects_an_invalid_baseline():
    with pytest.raises(ValueError, match="scale must be > 0"):
        CaurenPipeline().diagnose(_payload(baseline=_reference(scale=0.0)))


def test_concurrent_requests_with_different_baselines_do_not_share_them():
    pipeline = CaurenPipeline.from_default_registry()
    near = _reference()
    far = _reference(center=0.9)
    expected = {
        "near": pipeline.diagnose(_payload(baseline=near)).core_output.pattern_scores,
        "far": pipeline.diagnose(_payload(baseline=far)).core_output.pattern_scores,
    }
    mismatches = []
    lock = threading.Lock()

    def worker(name, baseline):
        for _ in range(25):
            scores = pipeline.diagnose(_payload(baseline=baseline)).core_output.pattern_scores
            if scores != expected[name]:
                with lock:
                    mismatches.append(name)

    threads = [
        threading.Thread(target=worker, args=item)
        for item in (("near", near), ("far", far), ("near", near), ("far", far))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert expected["near"] != expected["far"]
    assert not mismatches
