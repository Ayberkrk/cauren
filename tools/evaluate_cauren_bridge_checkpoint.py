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
        predicted = (torch.sigmoid(output["risk_logit"]) > 0.5).float()
        hits = ((predicted == label_values).float() * label_mask).sum().item()
        labeled_total = int(label_mask.sum().item())
        bce = torch.nn.functional.binary_cross_entropy_with_logits(output["risk_logit"], label_values, reduction="none")
        bce_mean = float((bce * label_mask).sum().item() / max(1, labeled_total))

    accuracy = hits / max(1, labeled_total)
    positive_rate = float(label_values[label_mask.bool()].mean().item()) if labeled_total else None
    majority_baseline = max(positive_rate, 1.0 - positive_rate) if positive_rate is not None else None
    return {
        "agent_id": agent_id,
        "target_column": target_column,
        "test_windows": len(tensors),
        "labeled_windows": labeled_total,
        "test_accuracy": round(accuracy, 6),
        "test_bce": round(bce_mean, 6),
        "test_majority_baseline": round(majority_baseline, 6) if majority_baseline is not None else None,
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
