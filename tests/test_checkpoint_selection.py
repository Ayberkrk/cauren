"""Checkpoint selection should track the actual training objective.

Before this fix, `train_cauren_core` always selected the checkpoint with
the lowest validation *reconstruction* loss, even for datasets (like
cauren-bridge) that add a real supervised loss term and are meant to be
evaluated on that forecasting task -- so a better-reconstructing epoch
could be kept over one that predicted the outcome better.

The pure decision helpers (`_checkpoint_selection_metric_name`,
`_checkpoint_selection_metric_value`) are unit tested directly. The
end-to-end tests then run a couple of epochs of real training against a
tiny synthetic dataset for both an unsupervised agent (no real target --
selection stays reconstruction-based) and a supervised one (a real target
-- selection switches to validation supervised BCE), and check that the
reported `best_epoch`/`best_selection_metric_value` actually correspond to
the metric `summary["selection_metric"]` says was used, rather than
asserting a particular BCE value (which would be flaky under real
gradient descent).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="cauren_core.training trains the real backbone, which needs torch")

from cauren_core.training import (  # noqa: E402
    CaurenCoreTrainConfig,
    _checkpoint_selection_metric_name,
    _checkpoint_selection_metric_value,
    train_cauren_core,
)

FEATURES = ("structural_risk_score", "ground_stability_score", "natural_hazard_score")
SEQ_LEN = 4


def test_selection_metric_name_is_reconstruction_without_supervision():
    assert _checkpoint_selection_metric_name(has_supervision=False) == "validation_reconstruction_loss"


def test_selection_metric_name_is_supervised_bce_with_supervision():
    assert _checkpoint_selection_metric_name(has_supervision=True) == "validation_supervised_bce"


def test_selection_metric_value_picks_reconstruction_loss_when_unsupervised():
    metric_name = _checkpoint_selection_metric_name(has_supervision=False)
    value = _checkpoint_selection_metric_value(metric_name=metric_name, val_avg_loss=0.42, val_supervised_bce=0.05)
    assert value == 0.42


def test_selection_metric_value_picks_supervised_bce_when_supervised():
    metric_name = _checkpoint_selection_metric_name(has_supervision=True)
    value = _checkpoint_selection_metric_value(metric_name=metric_name, val_avg_loss=0.42, val_supervised_bce=0.05)
    assert value == 0.05


def test_selection_metric_value_is_none_when_supervised_batch_has_no_labels():
    # A dataset-level supervised run whose validation batch happened to
    # have zero labeled windows this epoch must not silently fall back to
    # comparing a reconstruction-loss value against a BCE running best --
    # those are different scales, so it should skip the epoch instead.
    metric_name = _checkpoint_selection_metric_name(has_supervision=True)
    value = _checkpoint_selection_metric_value(metric_name=metric_name, val_avg_loss=0.42, val_supervised_bce=None)
    assert value is None


def _write_window_row(writer, *, window_id: str, agent_id: str, split: str, training_role: str) -> None:
    writer.writerow(
        {
            "window_id": window_id,
            "agent_id": agent_id,
            "sector": "civil",
            "asset_id": window_id,
            "client_id": "synthetic",
            "site_id": "synthetic",
            "window_start": "2000-01-01T00:00:00Z",
            "window_end": "2001-01-01T00:00:00Z",
            "sampling_hz": 1.0,
            "seq_len": SEQ_LEN,
            "sensor_count": len(FEATURES),
            "missing_ratio": 0.0,
            "quality_score": 1.0,
            "split": split,
            "training_role": training_role,
            "real_step_count": SEQ_LEN,
        }
    )


def _write_reading_rows(writer, *, window_id: str, agent_id: str, values: list[list[float]]) -> None:
    for seq_index, step_values in enumerate(values):
        for feature_name, value in zip(FEATURES, step_values):
            writer.writerow(
                {
                    "window_id": window_id,
                    "agent_id": agent_id,
                    "sector": "civil",
                    "asset_id": window_id,
                    "client_id": "synthetic",
                    "site_id": "synthetic",
                    "timestamp": f"2000-{seq_index + 1:02d}-01T00:00:00Z",
                    "seq_index": seq_index,
                    "sensor_id": f"{window_id}:{feature_name}",
                    "sensor_name": feature_name,
                    "unit": "ratio",
                    "value": value,
                    "quality": "true",
                    "metadata_json": "{}",
                }
            )


def _build_synthetic_dataset(dataset_dir: Path, *, agent_id: str, supervised: bool) -> None:
    agents_dir = dataset_dir / "agents" / agent_id
    agents_dir.mkdir(parents=True, exist_ok=True)

    window_fields = [
        "window_id", "agent_id", "sector", "asset_id", "client_id", "site_id",
        "window_start", "window_end", "sampling_hz", "seq_len", "sensor_count",
        "missing_ratio", "quality_score", "split", "training_role", "real_step_count",
    ]
    reading_fields = [
        "window_id", "agent_id", "sector", "asset_id", "client_id", "site_id",
        "timestamp", "seq_index", "sensor_id", "sensor_name", "unit", "value",
        "quality", "metadata_json",
    ]
    label_fields = [
        "window_id", "agent_id", "sector", "anomaly_type", "anomaly_family",
        "risk_score", "confidence", "label_source", "event_start", "event_end",
        "root_cause_hint", "notes", "is_anomaly", "deck_drop_5yr",
    ]

    windows_fh = (agents_dir / "windows.csv").open("w", newline="", encoding="utf-8")
    readings_fh = (agents_dir / "raw_sensor_readings.csv").open("w", newline="", encoding="utf-8")
    labels_fh = (agents_dir / "labels.csv").open("w", newline="", encoding="utf-8")
    windows_writer = csv.DictWriter(windows_fh, fieldnames=window_fields)
    readings_writer = csv.DictWriter(readings_fh, fieldnames=reading_fields)
    labels_writer = csv.DictWriter(labels_fh, fieldnames=label_fields)
    windows_writer.writeheader()
    readings_writer.writeheader()
    labels_writer.writeheader()

    def make_values(window_idx: int, positive: bool) -> list[list[float]]:
        base = 0.75 if positive else 0.2
        return [[base + 0.01 * step, base + 0.02 * step, base - 0.01 * step] for step in range(SEQ_LEN)]

    for split, count in (("train", 24), ("validation", 12)):
        for i in range(count):
            window_id = f"{agent_id}-{split}-{i:03d}"
            positive = bool(i % 2)
            _write_window_row(windows_writer, window_id=window_id, agent_id=agent_id, split=split, training_role="mae_reconstruction")
            _write_reading_rows(readings_writer, window_id=window_id, agent_id=agent_id, values=make_values(i, positive))
            labels_writer.writerow(
                {
                    "window_id": window_id,
                    "agent_id": agent_id,
                    "sector": "civil",
                    "anomaly_type": "" if not positive else "structural_deterioration",
                    "anomaly_family": "" if not positive else "structural_deterioration",
                    "risk_score": 0.8 if positive else 0.2,
                    "confidence": 1.0,
                    "label_source": "synthetic",
                    "event_start": "",
                    "event_end": "",
                    "root_cause_hint": "",
                    "notes": "synthetic_test_fixture",
                    # All windows are nominal so the unsupervised run's
                    # require_nominal_for_mae filter doesn't drop them.
                    "is_anomaly": "false",
                    "deck_drop_5yr": (1.0 if positive else 0.0) if supervised else "",
                }
            )

    windows_fh.close()
    readings_fh.close()
    labels_fh.close()

    summary = {
        "dataset_id": "synthetic_checkpoint_selection_fixture",
        "agent_id": agent_id,
        "sector": "civil",
        "sampling_hz": 1.0,
    }
    if supervised:
        summary["supervised_target"] = "deck_drop_5yr"
    (dataset_dir / "dataset_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _tiny_model_configs() -> dict[str, dict]:
    tiny = {"mae_config": {"d_model": 8, "n_encoder_layers": 1, "n_heads": 2, "d_ff_encoder": 16, "d_ff_decoder": 8, "dropout": 0.0}}
    tiny["rnn_config"] = {"hidden_dim": 8, "num_layers": 1}
    tiny["scnn_config"] = {"hidden_channels": 4, "kernel_size": 3}
    return tiny


def test_unsupervised_run_selects_checkpoint_by_reconstruction_loss(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    _build_synthetic_dataset(dataset_dir, agent_id="cauren-bridge", supervised=False)
    output_path = tmp_path / "checkpoint.pt"
    config = CaurenCoreTrainConfig(
        dataset_dir=dataset_dir,
        output_path=output_path,
        agents=("cauren-bridge",),
        epochs=3,
        batch_size=32,
        min_epochs=3,
        early_stopping_patience=10,
        **_tiny_model_configs(),
    )
    result = train_cauren_core(config)
    summary = result["summaries"]["cauren-bridge"]

    assert summary["selection_metric"] == "validation_reconstruction_loss"
    assert summary["best_validation_supervised_accuracy"] is None
    recon_history = {row["epoch"]: row["loss"] for row in summary["validation_loss_history"]}
    assert recon_history[summary["best_epoch"]] == pytest.approx(min(recon_history.values()))
    assert summary["best_selection_metric_value"] == pytest.approx(summary["best_validation_loss"])


def test_supervised_run_selects_checkpoint_by_supervised_bce(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    _build_synthetic_dataset(dataset_dir, agent_id="cauren-bridge", supervised=True)
    output_path = tmp_path / "checkpoint.pt"
    config = CaurenCoreTrainConfig(
        dataset_dir=dataset_dir,
        output_path=output_path,
        agents=("cauren-bridge",),
        epochs=4,
        batch_size=32,
        min_epochs=4,
        early_stopping_patience=10,
        **_tiny_model_configs(),
    )
    result = train_cauren_core(config)
    summary = result["summaries"]["cauren-bridge"]

    assert summary["selection_metric"] == "validation_supervised_bce"
    bce_history = {row["epoch"]: row["bce"] for row in summary["validation_supervised_accuracy_history"] if row["bce"] is not None}
    assert bce_history, "expected at least one epoch with a real supervised BCE reading"
    # The selected epoch must be the one with the lowest supervised BCE,
    # not necessarily the one with the lowest reconstruction loss.
    assert bce_history[summary["best_epoch"]] == pytest.approx(min(bce_history.values()))
    assert summary["best_selection_metric_value"] == pytest.approx(bce_history[summary["best_epoch"]])
    # best_validation_supervised_accuracy must be the accuracy *at the
    # selected (BCE-best) epoch*, not at whatever epoch had the lowest
    # reconstruction loss.
    acc_history = {row["epoch"]: row["accuracy"] for row in summary["validation_supervised_accuracy_history"]}
    assert summary["best_validation_supervised_accuracy"] == pytest.approx(acc_history[summary["best_epoch"]])
