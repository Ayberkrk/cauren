"""Small dependency-free metrics shared by bridge benchmark scripts."""

from __future__ import annotations

import math
from collections.abc import Sequence


def score_metrics(
    true_labels: Sequence[float],
    scores: Sequence[float],
    *,
    threshold: float = 0.5,
    capacities: Sequence[float] = (0.05, 0.10),
) -> dict:
    """Report ranking, threshold, capacity, and calibration metrics.

    Average precision groups equal scores before accumulating precision, as
    scikit-learn's average_precision_score does. Capacity cutoffs use ceil so
    a fractional review slot is rounded up to an actionable bridge count.
    """
    if len(true_labels) != len(scores):
        raise ValueError("true_labels and scores must have the same length")
    pairs = [(int(float(y) >= 0.5), float(score)) for y, score in zip(true_labels, scores)]
    pairs = [(label, score) for label, score in pairs if math.isfinite(score)]
    if not pairs:
        return {"labeled_windows": 0}

    positives = sum(label for label, _score in pairs)
    negatives = len(pairs) - positives
    prevalence = positives / len(pairs)
    ordered = sorted(pairs, key=lambda item: item[1], reverse=True)

    average_precision = 0.0
    cumulative_tp = 0
    cumulative_fp = 0
    previous_recall = 0.0
    index = 0
    while index < len(ordered):
        score = ordered[index][1]
        group_tp = 0
        group_fp = 0
        while index < len(ordered) and ordered[index][1] == score:
            if ordered[index][0]:
                group_tp += 1
            else:
                group_fp += 1
            index += 1
        cumulative_tp += group_tp
        cumulative_fp += group_fp
        recall = cumulative_tp / positives if positives else 0.0
        precision = cumulative_tp / max(1, cumulative_tp + cumulative_fp)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall

    brier = sum((score - label) ** 2 for label, score in pairs) / len(pairs)
    # Equal-frequency bins avoid empty fixed-width bins on highly imbalanced
    # predictions. A block of tied scores always goes into a single bin (the
    # bin of its first position): splitting it would let input row order
    # decide which labels land in which bin, and a constant predictor would
    # then report a calibration error it does not have.
    calibration_order = sorted(pairs, key=lambda item: item[1])
    bin_count = min(10, len(calibration_order))
    bins: dict[int, list[tuple[int, float]]] = {}
    index = 0
    while index < len(calibration_order):
        end = index + 1
        while end < len(calibration_order) and calibration_order[end][1] == calibration_order[index][1]:
            end += 1
        bin_index = index * bin_count // len(calibration_order)
        bins.setdefault(bin_index, []).extend(calibration_order[index:end])
        index = end
    ece = 0.0
    for bucket in bins.values():
        observed = sum(label for label, _score in bucket) / len(bucket)
        predicted = sum(score for _label, score in bucket) / len(bucket)
        ece += len(bucket) / len(pairs) * abs(observed - predicted)

    predicted_positive = [(label, score) for label, score in pairs if score >= threshold]
    tp = sum(label for label, _score in predicted_positive)
    fp = len(predicted_positive) - tp
    fn = positives - tp
    tn = negatives - fp
    result = {
        "labeled_windows": len(pairs),
        "positive_rate": round(prevalence, 6),
        "pr_auc_average_precision": round(average_precision, 6),
        "brier_score": round(brier, 6),
        "expected_calibration_error_10_equal_frequency_bins": round(ece, 6),
        "threshold": threshold,
        "threshold_metrics": {
            "precision": round(tp / max(1, tp + fp), 6),
            "recall": round(tp / max(1, positives), 6),
            "false_alarms": fp,
            "missed_deteriorations": fn,
            "true_positives": tp,
            "true_negatives": tn,
            "accuracy": round((tp + tn) / len(pairs), 6),
        },
        "inspection_capacity": {},
    }
    for fraction in capacities:
        selected_count = min(len(ordered), max(1, math.ceil(len(ordered) * fraction)))
        remaining = selected_count
        selected_tp = 0.0
        group_start = 0
        while group_start < len(ordered) and remaining > 0:
            score = ordered[group_start][1]
            group_end = group_start + 1
            while group_end < len(ordered) and ordered[group_end][1] == score:
                group_end += 1
            group = ordered[group_start:group_end]
            take = min(remaining, len(group))
            group_positives = sum(label for label, _score in group)
            # A capacity cutoff inside a tied-score block has no defensible
            # deterministic order; report its expected captures over random
            # tie breaks instead of depending on CSV row order.
            selected_tp += take * group_positives / len(group)
            remaining -= take
            group_start = group_end
        key = f"top_{round(fraction * 100):g}_percent"
        result["inspection_capacity"][key] = {
            "bridges_reviewed": selected_count,
            "deteriorations_captured_expected": round(selected_tp, 3),
            "deterioration_recall": round(selected_tp / max(1, positives), 6),
            "false_alarms_expected": round(selected_count - selected_tp, 3),
            "precision": round(selected_tp / selected_count, 6),
        }
    return result
