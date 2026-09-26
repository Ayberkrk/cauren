"""Evaluate a trained cauren-bridge checkpoint once on the untouched test split.

Training (tools/train_cauren_core.py / cauren_core/training.py) only ever
reports validation metrics -- the split used for early stopping and
checkpoint selection. Reporting a number from that split as "the" accuracy
risks overstating how well the model generalizes, since the checkpoint was
explicitly chosen to do well on it. This script runs the already-trained,
already-selected checkpoint's forward pass once on the test split (which
never influenced training or checkpoint selection) and reports accuracy and
BCE there, matching this project's own stated leakage-consciousness (see
README.md's Evaluation section).

deck_drop_5yr is imbalanced (~25% positive), so accuracy alone can hide a
model that mostly predicts the majority (no-drop) class -- a model that
always predicts "no drop" already scores ~74-75% accuracy without finding
a single true deterioration case. Alongside accuracy/BCE, this script also
reports precision, recall, F1, balanced accuracy, and the raw confusion
matrix, all computed at the same PREDICTION_THRESHOLD used to turn the
model's sigmoid output into a 0/1 prediction (see PREDICTION_THRESHOLD
below; also documented in README.md's Evaluation section).

Usage:
    python3 tools/evaluate_cauren_bridge_checkpoint.py \\
        --dataset-dir data/cauren_bridge \\
        --checkpoint cauren_core/checkpoints/cauren_bridge_backbone_bundle.pt \\
        --agent cauren-bridge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cauren_agents.registry import build_default_registry, build_registry
from cauren_core.backbone import CaurenHybridBackbone
from cauren_core.training import (
    _build_label_tensors,
    _import_torch,
    _load_agent_tensors,
    _load_sampling_hz,
    _load_supervised_target_column,
    _load_window_labels,
    _select_window_ids,
    _stack_tensors,
)
from tools.bridge_model_metrics import score_metrics

# Sigmoid output at or above this counts as a positive (deterioration)
# prediction. Documented here rather than left as a bare literal because
# issue #14 flagged that a threshold silently baked into a comparison is
# easy to lose track of when interpreting precision/recall.
PREDICTION_THRESHOLD = 0.5


def compute_classification_metrics(*, true_labels: Sequence[float], predicted_labels: Sequence[float]) -> dict:
    """Confusion-matrix-derived metrics for binary 0/1 labels.

    Pure Python, no torch/numpy -- both arguments are already-thresholded
    0/1 values (see PREDICTION_THRESHOLD), so this is testable without the
    `backbone`/`dataset` extras a real checkpoint evaluation needs.
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have the same length")

    true_positives = true_negatives = false_positives = false_negatives = 0
    for true, predicted in zip(true_labels, predicted_labels):
        is_positive = true >= 0.5
        predicted_positive = predicted >= 0.5
        if is_positive and predicted_positive:
            true_positives += 1
        elif is_positive and not predicted_positive:
            false_negatives += 1
        elif not is_positive and predicted_positive:
            false_positives += 1
        else:
            true_negatives += 1

    total = true_positives + true_negatives + false_positives + false_negatives
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    accuracy = (true_positives + true_negatives) / total if total else 0.0

    return {
        "confusion_matrix": {
            "true_positives": true_positives,
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        },
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "balanced_accuracy": round(balanced_accuracy, 6),
    }


def evaluate_on_test_split(*, dataset_dir: Path, checkpoint_path: Path, agent_id: str) -> dict:
    torch = _import_torch()
    registry = build_registry([agent_id]) if agent_id else build_default_registry()
    agent = registry.get(agent_id)
    feature_names = tuple(agent.schema.feature_order)
    dataset_sampling_hz = _load_sampling_hz(dataset_dir)

    bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    layout_key = CaurenHybridBackbone.layout_key(feature_names)
    layout = bundle["layouts"].get(layout_key)
    if layout is None:
        raise SystemExit(
            f"Checkpoint {checkpoint_path} has no layout for {agent_id}'s feature set "
            f"({feature_names}); available layouts: {list(bundle['layouts'])}"
        )

    from cauren_core.backbone_model import create_cauren_backbone_model

    model = create_cauren_backbone_model(torch, input_dim=len(feature_names))
    model.load_state_dict(layout["model_state_dict"], strict=False)
    model.eval()

    mean = torch.tensor(layout["normalization"]["mean"], dtype=torch.float32)
    std = torch.tensor(layout["normalization"]["std"], dtype=torch.float32)

    selected = _select_window_ids(
        dataset_dir=dataset_dir,
        agent_id=agent_id,
        splits=["test"],
        training_roles=["mae_reconstruction"],
        require_nominal_for_mae=False,
        max_windows=None,
    )
    if not selected:
        raise SystemExit(f"No test-split windows found for {agent_id} under {dataset_dir}")

    tensors = _load_agent_tensors(
        dataset_dir=dataset_dir, agent_id=agent_id, feature_names=feature_names, selected_window_ids=selected
    )
    matrix, mask = _stack_tensors(torch, tensors, device="cpu")
    normalized = (matrix - mean.view(1, 1, -1)) / std.clamp_min(1e-6).view(1, 1, -1)

    target_column = _load_supervised_target_column(dataset_dir)
    label_lookup = _load_window_labels(dataset_dir=dataset_dir, agent_id=agent_id, target_column=target_column)
    label_values, label_mask = _build_label_tensors(torch, tensors, label_lookup, device="cpu")

    with torch.no_grad():
        output = model(
            normalized,
            return_intermediates=True,
            presence_mask=mask,
            sampling_hz=dataset_sampling_hz,
            runtime_mode="validation",
        )
        predicted = (torch.sigmoid(output["risk_logit"]) >= PREDICTION_THRESHOLD).float()
        labeled_total = int(label_mask.sum().item())
        bce = torch.nn.functional.binary_cross_entropy_with_logits(output["risk_logit"], label_values, reduction="none")
        bce_mean = float((bce * label_mask).sum().item() / max(1, labeled_total))
        keep = label_mask.bool()
        true_labels = label_values[keep].tolist()
        predicted_labels = predicted[keep].tolist()
        probabilities = torch.sigmoid(output["risk_logit"])[keep].tolist()

    metrics = compute_classification_metrics(true_labels=true_labels, predicted_labels=predicted_labels)
    score_summary = score_metrics(true_labels, probabilities, threshold=PREDICTION_THRESHOLD)
    positive_rate = sum(true_labels) / len(true_labels) if true_labels else None
    majority_baseline = max(positive_rate, 1.0 - positive_rate) if positive_rate is not None else None
    return {
        "agent_id": agent_id,
        "target_column": target_column,
        "test_windows": len(tensors),
        "labeled_windows": labeled_total,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "test_accuracy": metrics["accuracy"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_f1": metrics["f1"],
        "test_balanced_accuracy": metrics["balanced_accuracy"],
        "test_confusion_matrix": metrics["confusion_matrix"],
        "test_bce": round(bce_mean, 6),
        "test_majority_baseline": round(majority_baseline, 6) if majority_baseline is not None else None,
        "test_pr_auc_average_precision": score_summary.get("pr_auc_average_precision"),
        "test_brier_score": score_summary.get("brier_score"),
        "test_expected_calibration_error_10_equal_frequency_bins": score_summary.get(
            "expected_calibration_error_10_equal_frequency_bins"
        ),
        "test_false_alarms_at_threshold": score_summary.get("threshold_metrics", {}).get("false_alarms"),
        "test_inspection_capacity": score_summary.get("inspection_capacity", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", default="data/cauren_bridge")
    parser.add_argument("--checkpoint", default="cauren_core/checkpoints/cauren_bridge_backbone_bundle.pt")
    parser.add_argument("--agent", default="cauren-bridge")
    args = parser.parse_args()
    result = evaluate_on_test_split(
        dataset_dir=Path(args.dataset_dir), checkpoint_path=Path(args.checkpoint), agent_id=args.agent
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
