"""Deterministic coverage for compute_classification_metrics on imbalanced labels.

No torch/numpy needed -- tools/evaluate_cauren_bridge_checkpoint.py's
compute_classification_metrics is pure Python by design (see its
docstring), so this stays in the minimal-dependency test set even though
actually running the checkpoint evaluation itself needs the `backbone`
and `dataset` extras.
"""

from __future__ import annotations

import pytest

from tools.evaluate_cauren_bridge_checkpoint import PREDICTION_THRESHOLD, compute_classification_metrics


def test_majority_class_predictor_on_imbalanced_labels_scores_high_accuracy_but_zero_recall():
    # 25% positive rate, matching deck_drop_5yr's real imbalance -- a
    # predictor that always says "no drop" is exactly the failure mode
    # issue #14 is about: it looks good on accuracy alone.
    true_labels = [1, 0, 0, 0] * 5  # 20 labels, 5 positive (25%)
    predicted_labels = [0] * 20

    metrics = compute_classification_metrics(true_labels=true_labels, predicted_labels=predicted_labels)

    assert metrics["confusion_matrix"] == {
        "true_positives": 0,
        "true_negatives": 15,
        "false_positives": 0,
        "false_negatives": 5,
    }
    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    # Balanced accuracy is the real tell: 0.5 (chance) despite 75% raw
    # accuracy, because it averages recall (0.0) and specificity (1.0).
    assert metrics["balanced_accuracy"] == 0.5


def test_perfect_predictor_scores_one_on_every_metric():
    true_labels = [1, 0, 1, 0, 1]
    predicted_labels = [1, 0, 1, 0, 1]

    metrics = compute_classification_metrics(true_labels=true_labels, predicted_labels=predicted_labels)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["confusion_matrix"] == {
        "true_positives": 3,
        "true_negatives": 2,
        "false_positives": 0,
        "false_negatives": 0,
    }


def test_mixed_confusion_matrix_matches_hand_computed_values():
    # 10 labels: 4 real positives, 6 real negatives.
    true_labels = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    # 3 true positives, 1 false negative, 2 false positives, 4 true negatives.
    predicted_labels = [1, 1, 1, 0, 1, 1, 0, 0, 0, 0]

    metrics = compute_classification_metrics(true_labels=true_labels, predicted_labels=predicted_labels)

    assert metrics["confusion_matrix"] == {
        "true_positives": 3,
        "true_negatives": 4,
        "false_positives": 2,
        "false_negatives": 1,
    }
    assert metrics["precision"] == 0.6  # 3 / (3 + 2)
    assert metrics["recall"] == 0.75  # 3 / (3 + 1)
    assert metrics["f1"] == round(2 * 0.6 * 0.75 / (0.6 + 0.75), 6)
    # specificity = 4 / (4 + 2) = 0.666667; balanced_accuracy = (0.75 + 0.666667) / 2
    assert abs(metrics["balanced_accuracy"] - 0.708333) < 1e-5


def test_no_positive_predictions_gives_zero_precision_not_a_crash():
    true_labels = [1, 0, 0]
    predicted_labels = [0, 0, 0]
    metrics = compute_classification_metrics(true_labels=true_labels, predicted_labels=predicted_labels)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_no_real_positives_gives_zero_recall_not_a_crash():
    true_labels = [0, 0, 0]
    predicted_labels = [1, 0, 0]
    metrics = compute_classification_metrics(true_labels=true_labels, predicted_labels=predicted_labels)
    assert metrics["recall"] == 0.0
    assert metrics["precision"] == 0.0


def test_empty_input_does_not_crash():
    metrics = compute_classification_metrics(true_labels=[], predicted_labels=[])
    assert metrics["accuracy"] == 0.0
    assert metrics["confusion_matrix"] == {
        "true_positives": 0,
        "true_negatives": 0,
        "false_positives": 0,
        "false_negatives": 0,
    }


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        compute_classification_metrics(true_labels=[1, 0], predicted_labels=[1])


def test_prediction_threshold_is_documented_and_reasonable():
    assert 0.0 < PREDICTION_THRESHOLD < 1.0
