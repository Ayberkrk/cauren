import random

import pytest

np = pytest.importorskip("numpy")

from tools.bridge_cluster_bootstrap import paired_cluster_bootstrap
from tools.bridge_model_metrics import score_metrics


def _sample(seed: int = 3, bridges: int = 60, windows_per_bridge: int = 4):
    rng = random.Random(seed)
    labels, clusters, good, weak = [], [], [], []
    for bridge in range(bridges):
        risk = rng.random()
        for _ in range(windows_per_bridge):
            label = int(rng.random() < risk)
            labels.append(label)
            clusters.append(f"CA_{bridge}")
            good.append(min(0.99, max(0.01, 0.6 * risk + 0.3 * label + 0.05)))
            weak.append(round(rng.random(), 1))
    return labels, clusters, {"saved_hybrid_checkpoint": good, "comparator": weak}


def test_point_estimates_match_the_shared_metric_definitions():
    labels, clusters, predictions = _sample()
    result = paired_cluster_bootstrap(labels, clusters, predictions, replicates=100)
    assert (result["cluster_count"], result["row_count"]) == (60, 240)
    for name, scores in predictions.items():
        expected = score_metrics(labels, scores)
        point = result["model_metrics"][name]["point_estimate"]
        assert point["pr_auc_average_precision"] == pytest.approx(expected["pr_auc_average_precision"], abs=1e-6)
        assert point["brier_score"] == pytest.approx(expected["brier_score"], abs=1e-6)


def test_identical_models_have_zero_width_paired_differences():
    labels, clusters, predictions = _sample()
    same = {"saved_hybrid_checkpoint": predictions["saved_hybrid_checkpoint"]}
    same["copy"] = list(same["saved_hybrid_checkpoint"])
    result = paired_cluster_bootstrap(labels, clusters, same, replicates=100)
    intervals = result["paired_differences_vs_reference"]["copy"]["confidence_intervals"]
    assert intervals["pr_auc_average_precision"] == [0.0, 0.0]
    assert intervals["brier_score"] == [0.0, 0.0]


def test_a_fixed_seed_reproduces_the_intervals():
    labels, clusters, predictions = _sample()
    first = paired_cluster_bootstrap(labels, clusters, predictions, replicates=100, seed=11)
    second = paired_cluster_bootstrap(labels, clusters, predictions, replicates=100, seed=11)
    assert first["model_metrics"] == second["model_metrics"]
    assert first["paired_differences_vs_reference"] == second["paired_differences_vs_reference"]


@pytest.mark.parametrize(
    ("labels", "predictions", "message"),
    [
        ([1, 0, 1, 0], {"other": [0.5] * 4}, "reference model"),
        ([1, 1, 1, 1], {"saved_hybrid_checkpoint": [0.5] * 4}, "both outcome classes"),
        ([1, 0, 1, 0], {"saved_hybrid_checkpoint": [0.5, 0.5, 1.5, 0.5]}, "finite probabilities"),
        ([1, 0, 1, 0], {"saved_hybrid_checkpoint": [0.5] * 3}, "not aligned"),
    ],
)
def test_invalid_inputs_are_rejected(labels, predictions, message):
    with pytest.raises(ValueError, match=message):
        paired_cluster_bootstrap(labels, ["A", "A", "B", "B"], predictions, replicates=100)
