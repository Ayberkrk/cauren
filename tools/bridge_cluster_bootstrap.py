"""Paired bridge-cluster bootstrap intervals for saved test predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _prepare_average_precision(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    group_starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    group_ids = np.cumsum(np.r_[0, sorted_scores[1:] != sorted_scores[:-1]])
    return order, group_ids, len(group_starts)


def _weighted_average_precision(
    labels: np.ndarray,
    order: np.ndarray,
    group_ids: np.ndarray,
    group_count: int,
    row_weights: np.ndarray,
) -> float:
    sorted_weights = row_weights[order]
    sorted_labels = labels[order]
    group_tp = np.bincount(
        group_ids, weights=sorted_weights * sorted_labels, minlength=group_count
    )
    group_fp = np.bincount(
        group_ids, weights=sorted_weights * (1.0 - sorted_labels), minlength=group_count
    )
    total_positive = float(group_tp.sum())
    if total_positive == 0.0:
        return float("nan")
    cumulative_tp = np.cumsum(group_tp)
    cumulative_fp = np.cumsum(group_fp)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1.0)
    return float(np.sum(group_tp / total_positive * precision))


def _percentile_interval(values: np.ndarray, confidence: float) -> list[float]:
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return [float(low), float(high)]


def paired_cluster_bootstrap(
    labels: Sequence[int | float],
    cluster_ids: Sequence[str],
    predictions: Mapping[str, Sequence[float]],
    *,
    reference_model: str = "saved_hybrid_checkpoint",
    replicates: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Return percentile intervals using paired resampling of bridge clusters.

    A sampled bridge contributes all of its rows, including repeated copies
    when drawn more than once. The same cluster draw is used for every model,
    so model differences retain their paired structure.
    """
    y = np.asarray(labels, dtype=np.float64)
    clusters = np.asarray(cluster_ids, dtype=str)
    if y.ndim != 1 or clusters.ndim != 1 or len(y) != len(clusters):
        raise ValueError("labels and cluster_ids must be aligned one-dimensional arrays")
    if len(y) == 0 or not np.isfinite(y).all() or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("bootstrap labels must be non-empty, finite binary values")
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if reference_model not in predictions:
        raise ValueError(f"reference model {reference_model!r} is missing")
    if len(np.unique(y)) != 2:
        raise ValueError("the fixed test sample must contain both outcome classes")

    model_names = list(predictions)
    score_arrays: dict[str, np.ndarray] = {}
    prepared: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for name, raw_scores in predictions.items():
        scores = np.asarray(raw_scores, dtype=np.float64)
        if scores.ndim != 1 or len(scores) != len(y):
            raise ValueError(f"predictions for {name!r} are not aligned with labels")
        if not np.isfinite(scores).all() or ((scores < 0.0) | (scores > 1.0)).any():
            raise ValueError(f"predictions for {name!r} must be finite probabilities")
        score_arrays[name] = scores
        prepared[name] = _prepare_average_precision(y, scores)

    unique_clusters, cluster_inverse = np.unique(clusters, return_inverse=True)
    if len(unique_clusters) < 2:
        raise ValueError("at least two bridge clusters are required")

    rng = np.random.default_rng(seed)
    sample_count = len(unique_clusters)
    metric_samples = {
        name: {"pr_auc_average_precision": np.empty(replicates), "brier_score": np.empty(replicates)}
        for name in model_names
    }
    difference_samples = {
        name: {"pr_auc_average_precision": np.empty(replicates), "brier_score": np.empty(replicates)}
        for name in model_names
        if name != reference_model
    }
    squared_errors = {name: (scores - y) ** 2 for name, scores in score_arrays.items()}

    for replicate in range(replicates):
        sampled_clusters = rng.integers(0, sample_count, size=sample_count)
        cluster_multiplicity = np.bincount(sampled_clusters, minlength=sample_count)
        row_weights = cluster_multiplicity[cluster_inverse].astype(np.float64, copy=False)
        total_weight = float(row_weights.sum())
        replicate_values: dict[str, tuple[float, float]] = {}
        for name in model_names:
            order, group_ids, group_count = prepared[name]
            average_precision = _weighted_average_precision(
                y, order, group_ids, group_count, row_weights
            )
            brier = float(np.dot(row_weights, squared_errors[name]) / total_weight)
            replicate_values[name] = (average_precision, brier)
            metric_samples[name]["pr_auc_average_precision"][replicate] = average_precision
            metric_samples[name]["brier_score"][replicate] = brier
        reference_values = replicate_values[reference_model]
        for name in difference_samples:
            values = replicate_values[name]
            difference_samples[name]["pr_auc_average_precision"][replicate] = (
                reference_values[0] - values[0]
            )
            difference_samples[name]["brier_score"][replicate] = reference_values[1] - values[1]

    model_intervals: dict[str, Any] = {}
    for name in model_names:
        order, group_ids, group_count = prepared[name]
        point_ap = _weighted_average_precision(y, order, group_ids, group_count, np.ones(len(y)))
        point_brier = float(np.mean(squared_errors[name]))
        model_intervals[name] = {
            "point_estimate": {
                "pr_auc_average_precision": point_ap,
                "brier_score": point_brier,
            },
            "confidence_intervals": {
                metric: _percentile_interval(values, confidence)
                for metric, values in metric_samples[name].items()
            },
        }

    paired_differences: dict[str, Any] = {}
    for name, metrics in difference_samples.items():
        paired_differences[name] = {
            "direction": "saved hybrid minus comparator; positive PR-AUC and negative Brier favor the hybrid",
            "point_difference": {
                metric: model_intervals[reference_model]["point_estimate"][metric]
                - model_intervals[name]["point_estimate"][metric]
                for metric in metrics
            },
            "confidence_intervals": {
                metric: _percentile_interval(values, confidence)
                for metric, values in metrics.items()
            },
        }

    return {
        "status": "completed",
        "method": "paired percentile bootstrap resampling bridges (asset_id clusters) with replacement",
        "cluster_count": int(len(unique_clusters)),
        "row_count": int(len(y)),
        "replicates": int(replicates),
        "seed": int(seed),
        "confidence_level": confidence,
        "reference_model": reference_model,
        "resample_shared_across_models": True,
        "model_metrics": model_intervals,
        "paired_differences_vs_reference": paired_differences,
        "interpretation": "These are descriptive percentile intervals, not formal hypothesis-test p-values.",
    }
