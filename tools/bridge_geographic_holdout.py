"""Strict leave-one-state-out evaluation for the bridge benchmark.

Each fold excludes every bridge from one state, including its original train,
validation, and test windows. The neural model is retrained from scratch using
the other states' official train windows and validation windows. The held-out
state is used only for scoring.
"""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from tools.benchmark_cauren_bridge_models import (
    _core_history_width,
    _fit_hist_gradient_boosting,
    _fit_logistic,
    _load_condition_history,
    _load_core_history,
    _matrix,
    _predict_logistic,
    _read_windows,
)
from tools.bridge_model_metrics import score_metrics


def _make_fold_view(dataset_dir: Path, fold_dir: Path, held_out_state: str) -> Path:
    target = fold_dir / "dataset"
    agent_source = dataset_dir / "agents" / "cauren-bridge"
    agent_target = target / "agents" / "cauren-bridge"
    agent_target.mkdir(parents=True, exist_ok=True)
    for name in ("raw_sensor_readings.csv", "asset_metadata.csv", "labels.csv"):
        (agent_target / name).symlink_to((agent_source / name).resolve())

    windows_out = []
    split_ids = {"train": [], "validation": [], "test": []}
    with (agent_source / "windows.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for row in reader:
            state = row["asset_id"].strip().split("_", 1)[0]
            original_split = row["split"].strip()
            if state == held_out_state:
                fold_split = "test"
            elif original_split in {"train", "validation"}:
                fold_split = original_split
            else:
                fold_split = "unused"
            row["split"] = fold_split
            windows_out.append(row)
            if fold_split in split_ids:
                split_ids[fold_split].append(row["window_id"])
    with (agent_target / "windows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(windows_out)

    shutil.copy2(dataset_dir / "dataset_summary.json", target / "dataset_summary.json")
    split_dir = target / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, ids in split_ids.items():
        (split_dir / f"{split}_windows.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    return target


def run_geographic_holdouts(
    *, dataset_dir: Path, checkpoint_root: Path, epochs: int = 20, batch_size: int = 64,
    learning_rate: float = 0.0004,
) -> dict[str, Any]:
    windows, labels = _read_windows(dataset_dir)
    summary = json.loads((dataset_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    feature_names = tuple(summary["features"])
    base_names, base_vectors = _load_core_history(dataset_dir, windows, feature_names)
    history_path = dataset_dir / "agents" / "cauren-bridge" / "condition_history.csv"
    history_names, history_vectors = _load_condition_history(history_path, windows)
    names, x, y, metadata = _matrix(windows, labels, base_names, base_vectors, history_names, history_vectors)

    splits = np.asarray([row["split"] for row in metadata])
    states = np.asarray([row["state"] for row in metadata])
    structural_column = names.index("structural_risk_score__last")
    core_width = _core_history_width(names, feature_names)
    core_x = x[:, :core_width]
    fold_reports: dict[str, Any] = {}

    try:
        from cauren_core.training import CaurenCoreTrainConfig, train_cauren_core
    except ImportError:
        train_cauren_core = None
        CaurenCoreTrainConfig = None

    with tempfile.TemporaryDirectory(prefix="cauren-bridge-losostate-") as temporary:
        temporary_root = Path(temporary)
        for state in sorted(set(states.tolist())):
            held_out = states == state
            train = (splits == "train") & ~held_out
            validation = (splits == "validation") & ~held_out
            if not train.any() or not validation.any() or not held_out.any():
                continue
            prior = float(np.mean(y[train]))
            structural = _predict_logistic(
                _fit_logistic(x[train][:, [structural_column]], y[train]), x[:, [structural_column]]
            )
            same_input = _predict_logistic(_fit_logistic(core_x[train], y[train]), core_x)
            expanded = _predict_logistic(_fit_logistic(x[train], y[train]), x)
            held_labels = y[held_out].tolist()
            heldout_models: dict[str, Any] = {
                "train_prevalence": score_metrics(held_labels, np.full(int(held_out.sum()), prior).tolist()),
                "last_structural_score_logistic": score_metrics(held_labels, structural[held_out].tolist()),
                "same_input_additive_logistic": score_metrics(held_labels, same_input[held_out].tolist()),
                "expanded_additive_logistic": score_metrics(held_labels, expanded[held_out].tolist()),
            }
            for name, matrix in (("hist_gradient_boosting", x), ("same_input_hist_gradient_boosting", core_x)):
                fitted = _fit_hist_gradient_boosting(
                    matrix, y, train, {"held_out_state": held_out}
                )
                heldout_models[name] = (
                    fitted["metrics"]["held_out_state"]
                    if fitted.get("status") == "completed"
                    else {"status": "skipped", "reason": fitted.get("reason")}
                )

            hybrid_result: dict[str, Any]
            if train_cauren_core is None or CaurenCoreTrainConfig is None:
                hybrid_result = {"status": "skipped", "reason": "training module is unavailable"}
            else:
                fold_dir = temporary_root / f"state-{state}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                fold_dataset = _make_fold_view(dataset_dir, fold_dir, state)
                fold_checkpoint = checkpoint_root / f"cauren_bridge_without_state_{state}.pt"
                try:
                    training_result = train_cauren_core(
                        CaurenCoreTrainConfig(
                            dataset_dir=fold_dataset,
                            output_path=fold_checkpoint,
                            agents=("cauren-bridge",),
                            training_roles=("mae_reconstruction",),
                            splits=("train",),
                            validation_splits=("validation",),
                            epochs=epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            seed=42,
                            early_stopping_patience=5,
                            min_epochs=5,
                        )
                    )
                    from tools.evaluate_cauren_bridge_checkpoint import evaluate_on_test_split

                    hybrid_result = {
                        "status": "completed",
                        "metrics": evaluate_on_test_split(
                            dataset_dir=fold_dataset,
                            checkpoint_path=fold_checkpoint,
                            agent_id="cauren-bridge",
                        ),
                        "training_summary": training_result.get("summaries", {}).get("cauren-bridge", {}),
                        "checkpoint_retained": False,
                    }
                except RuntimeError as error:
                    hybrid_result = {"status": "skipped", "reason": str(error)}
                finally:
                    fold_checkpoint.unlink(missing_ok=True)

            fold_reports[state] = {
                "train_windows": int(train.sum()),
                "validation_windows": int(validation.sum()),
                "held_out_windows": int(held_out.sum()),
                "models": heldout_models,
                "saved_hybrid_architecture_retrained_without_held_out_state": hybrid_result,
            }

    return {
        "protocol": {
            "type": "strict_leave_one_state_out",
            "held_out_state_excluded_from_train_and_validation": True,
            "held_out_state_scored_across_all_original_splits": True,
            "hybrid_retrained_from_scratch_per_fold": True,
            "tabular_fit_uses_official_train_rows_from_remaining_states": True,
            "hybrid_training": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "validation_selection": "validation supervised BCE on non-held-out states",
            },
        },
        "folds": fold_reports,
    }
