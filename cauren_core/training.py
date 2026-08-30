from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from cauren_agents.registry import build_default_registry, build_registry
from cauren_core.backbone import CaurenHybridBackbone
from cauren_core.backbone_model import create_cauren_backbone_model


DEFAULT_TRAINING_ROLES = ("mae_reconstruction",)
DEFAULT_SPLITS = ("train",)
DEFAULT_VALIDATION_SPLITS = ("validation",)


@dataclass(frozen=True)
class CaurenCoreTrainConfig:
    dataset_dir: Path
    output_path: Path
    agents: tuple[str, ...] = ()
    training_roles: tuple[str, ...] = DEFAULT_TRAINING_ROLES
    splits: tuple[str, ...] = DEFAULT_SPLITS
    validation_splits: tuple[str, ...] = DEFAULT_VALIDATION_SPLITS
    epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_windows_per_agent: int | None = None
    device: str = "cpu"
    seed: int = 42
    dry_run: bool = False
    early_stopping_patience: int = 4
    min_epochs: int = 4
    mae_config: dict[str, Any] = field(default_factory=dict)
    rnn_config: dict[str, Any] = field(default_factory=dict)
    scnn_config: dict[str, Any] = field(default_factory=dict)


def train_cauren_core(config: CaurenCoreTrainConfig) -> dict[str, Any]:
    torch = None if config.dry_run else _import_torch()
    if torch is not None:
        _set_seed(torch, config.seed)
    registry = build_registry(config.agents) if config.agents else build_default_registry()
    agents = config.agents or tuple(registry.ids())
    dataset_sampling_hz = _load_sampling_hz(config.dataset_dir)
    summaries: dict[str, Any] = {}
    layouts: dict[str, Any] = {}
    started = time.time()

    for agent_id in agents:
        agent = registry.get(agent_id)
        feature_names = tuple(agent.schema.feature_order)
        # "Nominal-only" window selection makes sense for pure unsupervised
        # MAE training (an autoencoder learns a clean baseline, then
        # reconstruction error on unseen anomalies becomes the anomaly
        # signal). It makes no sense for a dataset with a real supervised
        # target: it would silently drop every positive-labeled window
        # from both training and evaluation, leaving the classifier head
        # trained and validated on negatives only -- which trivially
        # "succeeds" at 100% accuracy by always predicting the negative
        # class, without having ever seen a real positive example.
        target_column_for_selection = _load_supervised_target_column(config.dataset_dir)
        require_nominal_for_mae = ("mae_reconstruction" in set(config.training_roles)) and not target_column_for_selection
        selected = _select_window_ids(
            dataset_dir=config.dataset_dir,
            agent_id=agent_id,
            splits=config.splits,
            training_roles=config.training_roles,
            require_nominal_for_mae=require_nominal_for_mae,
            max_windows=config.max_windows_per_agent,
        )
        validation_selected = _select_window_ids(
            dataset_dir=config.dataset_dir,
            agent_id=agent_id,
            splits=config.validation_splits,
            training_roles=config.training_roles,
            require_nominal_for_mae=require_nominal_for_mae,
            max_windows=max(1, int(config.max_windows_per_agent // 3)) if config.max_windows_per_agent else None,
        )
        if not selected:
            summaries[agent_id] = {"status": "skipped", "reason": "no_selected_windows"}
            continue
        tensors = _load_agent_tensors(
            dataset_dir=config.dataset_dir,
            agent_id=agent_id,
            feature_names=feature_names,
            selected_window_ids=selected,
        )
        validation_tensors = _load_agent_tensors(
            dataset_dir=config.dataset_dir,
            agent_id=agent_id,
            feature_names=feature_names,
            selected_window_ids=validation_selected,
        ) if validation_selected else []
        if not tensors:
            summaries[agent_id] = {"status": "skipped", "reason": "no_tensor_windows"}
            continue

        if config.dry_run:
            first_matrix, _first_mask, _first_id = tensors[0]
            summaries[agent_id] = {
                "status": "dry_run",
                "selected_windows": len(selected),
                "validation_windows": len(validation_selected),
                "tensor_windows": len(tensors),
                "validation_tensor_windows": len(validation_tensors),
                "seq_len": len(first_matrix),
                "input_dim": len(feature_names),
                "feature_names": list(feature_names),
                "training_roles": list(config.training_roles),
                "splits": list(config.splits),
                "validation_splits": list(config.validation_splits),
            }
            continue

        assert torch is not None
        matrix, mask = _stack_tensors(torch, tensors, device=config.device)
        mean, std = _masked_mean_std(torch, matrix, mask)
        normalized = (matrix - mean.view(1, 1, -1)) / std.clamp_min(1e-6).view(1, 1, -1)
        validation_normalized = None
        validation_mask = None
        if validation_tensors:
            validation_matrix, validation_mask = _stack_tensors(torch, validation_tensors, device=config.device)
            validation_normalized = (validation_matrix - mean.view(1, 1, -1)) / std.clamp_min(1e-6).view(1, 1, -1)

        # Real, independently-observed supervised target (e.g. deck_drop_5yr
        # for cauren-bridge), if this dataset has one. See
        # _load_supervised_target_column's docstring for why this is
        # opt-in per dataset rather than assumed.
        target_column = _load_supervised_target_column(config.dataset_dir)
        label_lookup = _load_window_labels(dataset_dir=config.dataset_dir, agent_id=agent_id, target_column=target_column)
        label_values, label_mask = _build_label_tensors(torch, tensors, label_lookup, device=config.device)
        validation_label_values, validation_label_mask = _build_label_tensors(
            torch, validation_tensors, label_lookup, device=config.device
        ) if validation_tensors else (None, None)
        has_supervision = bool(target_column) and bool(label_mask.any().item())

        summary = {
            "status": "dry_run" if config.dry_run else "trained",
            "selected_windows": int(matrix.shape[0]),
            "validation_windows": int(validation_normalized.shape[0]) if validation_normalized is not None else 0,
            "seq_len": int(matrix.shape[1]),
            "input_dim": int(matrix.shape[2]),
            "feature_names": list(feature_names),
            "training_roles": list(config.training_roles),
            "splits": list(config.splits),
            "validation_splits": list(config.validation_splits),
            "loss_history": [],
            "validation_loss_history": [],
            "supervised_target": target_column or None,
            "supervised_labeled_train_windows": int(label_mask.sum().item()) if has_supervision else 0,
        }
        model = _create_backbone_model(
            input_dim=len(feature_names),
            mae_config=config.mae_config,
            rnn_config=config.rnn_config,
            scnn_config=config.scnn_config,
            device=config.device,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate), weight_decay=1e-4)
        model.train()
        count = int(normalized.shape[0])
        batch_size = max(1, int(config.batch_size))
        best_val_loss = float("inf")
        best_epoch = 0
        best_state_dict = None
        epochs_without_improve = 0
        for epoch in range(max(1, int(config.epochs))):
            order = torch.randperm(count, device=normalized.device)
            epoch_losses: list[float] = []
            for start in range(0, count, batch_size):
                idx = order[start : start + batch_size]
                x = normalized.index_select(0, idx)
                batch_mask = mask.index_select(0, idx)
                output = model(
                    x,
                    return_intermediates=True,
                    presence_mask=batch_mask,
                    sampling_hz=dataset_sampling_hz,
                    runtime_mode="training",
                )
                calibrated = output["calibrated"]
                target = x[:, : calibrated.shape[1], :]
                target_mask = batch_mask[:, : calibrated.shape[1], :]
                recon = (calibrated - target).abs() * target_mask
                denom = target_mask.sum().clamp_min(1.0)
                recon_loss = recon.sum() / denom
                sub_losses = output.get("losses", {})
                aux_loss = _safe_tensor(torch, sub_losses.get("mae_loss"), device=config.device)
                aux_loss = aux_loss + (0.1 * _safe_tensor(torch, sub_losses.get("rnn_consistency_loss"), device=config.device))
                loss = recon_loss + (0.05 * aux_loss)
                if has_supervision:
                    batch_label_values = label_values.index_select(0, idx)
                    batch_label_mask = label_mask.index_select(0, idx)
                    if bool(batch_label_mask.any().item()):
                        bce = torch.nn.functional.binary_cross_entropy_with_logits(
                            output["risk_logit"], batch_label_values, reduction="none"
                        )
                        supervised_loss = (bce * batch_label_mask).sum() / batch_label_mask.sum().clamp_min(1.0)
                        loss = loss + (0.3 * supervised_loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))
            epoch_avg_loss = float(sum(epoch_losses) / float(max(1, len(epoch_losses))))
            val_avg_loss = None
            if validation_normalized is not None and validation_mask is not None and int(validation_normalized.shape[0]) > 0:
                model.eval()
                with torch.no_grad():
                    validation_losses: list[float] = []
                    correct = 0
                    labeled_total = 0
                    val_count = int(validation_normalized.shape[0])
                    for start in range(0, val_count, batch_size):
                        x = validation_normalized[start : start + batch_size]
                        batch_mask = validation_mask[start : start + batch_size]
                        output = model(
                            x,
                            return_intermediates=True,
                            presence_mask=batch_mask,
                            sampling_hz=dataset_sampling_hz,
                            runtime_mode="validation",
                        )
                        calibrated = output["calibrated"]
                        target = x[:, : calibrated.shape[1], :]
                        target_mask = batch_mask[:, : calibrated.shape[1], :]
                        recon = (calibrated - target).abs() * target_mask
                        denom = target_mask.sum().clamp_min(1.0)
                        validation_losses.append(float((recon.sum() / denom).detach().cpu().item()))
                        if has_supervision and validation_label_mask is not None:
                            batch_label_values = validation_label_values[start : start + batch_size]
                            batch_label_mask = validation_label_mask[start : start + batch_size]
                            predicted = (torch.sigmoid(output["risk_logit"]) > 0.5).float()
                            hits = ((predicted == batch_label_values).float() * batch_label_mask).sum().item()
                            correct += int(hits)
                            labeled_total += int(batch_label_mask.sum().item())
                model.train()
                val_avg_loss = float(sum(validation_losses) / float(max(1, len(validation_losses))))
                if has_supervision and labeled_total > 0:
                    summary.setdefault("validation_supervised_accuracy_history", []).append(
                        {"epoch": epoch + 1, "accuracy": round(correct / labeled_total, 6), "labeled_windows": labeled_total}
                    )
            summary["loss_history"].append(
                {
                    "epoch": epoch + 1,
                    "loss": epoch_avg_loss,
                }
            )
            if val_avg_loss is not None:
                summary["validation_loss_history"].append({"epoch": epoch + 1, "loss": val_avg_loss})
                if val_avg_loss < best_val_loss - 1e-6:
                    best_val_loss = val_avg_loss
                    best_epoch = epoch + 1
                    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1
                acc_history = summary.get("validation_supervised_accuracy_history") or []
                acc_suffix = f" | Val Accuracy ({target_column}): {acc_history[-1]['accuracy']:.4f}" if acc_history and acc_history[-1]["epoch"] == epoch + 1 else ""
                print(
                    f"Agent: {agent_id} | Epoch: {epoch + 1}/{max(1, int(config.epochs))} | "
                    f"Train Loss: {epoch_avg_loss:.6f} | Val Loss: {val_avg_loss:.6f}{acc_suffix}",
                    flush=True,
                )
                if (epoch + 1) >= max(1, int(config.min_epochs)) and epochs_without_improve >= max(1, int(config.early_stopping_patience)):
                    print(f"Agent: {agent_id} | Early stopping at epoch {epoch + 1} (best val epoch {best_epoch})", flush=True)
                    break
            else:
                print(f"Agent: {agent_id} | Epoch: {epoch + 1}/{max(1, int(config.epochs))} | Loss: {epoch_avg_loss:.6f}", flush=True)

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict, strict=False)
        summary["best_epoch"] = int(best_epoch or len(summary["loss_history"]))
        summary["best_validation_loss"] = float(best_val_loss) if best_val_loss != float("inf") else None
        acc_at_best = next(
            (row["accuracy"] for row in summary.get("validation_supervised_accuracy_history", []) if row["epoch"] == summary["best_epoch"]),
            None,
        )
        summary["best_validation_supervised_accuracy"] = acc_at_best

        layout_key = CaurenHybridBackbone.layout_key(feature_names)
        layouts[layout_key] = {
            "agent_id": agent_id,
            "feature_names": list(feature_names),
            "input_dim": len(feature_names),
            "normalization": {
                "mean": [float(x) for x in mean.detach().cpu().tolist()],
                "std": [float(x) for x in std.detach().cpu().tolist()],
            },
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "train_summary": summary,
        }
        summaries[agent_id] = summary

    payload = {
        "format": "cauren_core_backbone_bundle_v1",
        "created_at_unix": int(time.time()),
        "dataset_dir": str(config.dataset_dir),
        "agents": list(agents),
        "config": {
            "training_roles": list(config.training_roles),
            "splits": list(config.splits),
            "validation_splits": list(config.validation_splits),
            "epochs": int(config.epochs),
            "batch_size": int(config.batch_size),
            "learning_rate": float(config.learning_rate),
            "max_windows_per_agent": config.max_windows_per_agent,
            "early_stopping_patience": int(config.early_stopping_patience),
            "min_epochs": int(config.min_epochs),
            "dry_run": bool(config.dry_run),
        },
        "layouts": layouts,
        "summaries": summaries,
        "duration_seconds": round(time.time() - started, 3),
    }
    if not config.dry_run:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, config.output_path)
    return {
        "output_path": str(config.output_path),
        "dry_run": bool(config.dry_run),
        "layout_count": len(layouts),
        "summaries": summaries,
        "duration_seconds": payload["duration_seconds"],
    }


def _load_supervised_target_column(dataset_dir: Path) -> str:
    """Which labels.csv column, if any, is a real supervised target.

    Set by the dataset builder in dataset_summary.json (see
    data/cauren_bridge/dataset_summary.json's "supervised_target":
    "deck_drop_5yr"). Datasets without one (e.g. data/cauren_civil/,
    whose labels are a heuristic function of the input features, not an
    independent outcome) return "" and the supervised loss below is
    simply skipped.
    """
    summary_path = dataset_dir / "dataset_summary.json"
    if not summary_path.exists():
        return ""
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(payload.get("supervised_target") or "").strip()


def _load_window_labels(
    *,
    dataset_dir: Path,
    agent_id: str,
    target_column: str,
) -> dict[str, float]:
    if not target_column:
        return {}
    labels_path = dataset_dir / "agents" / agent_id / "labels.csv"
    if not labels_path.exists():
        return {}
    labels: dict[str, float] = {}
    with labels_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            window_id = str(row.get("window_id", "")).strip()
            raw = str(row.get(target_column, "")).strip()
            if not window_id or not raw:
                continue
            try:
                labels[window_id] = float(raw)
            except ValueError:
                continue
    return labels


def _load_sampling_hz(dataset_dir: Path) -> float:
    summary_path = dataset_dir / "dataset_summary.json"
    if not summary_path.exists():
        return 1.0
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 1.0
    try:
        return max(1.0, float(payload.get("sampling_hz", 1.0)))
    except (TypeError, ValueError):
        return 1.0


def _select_window_ids(
    *,
    dataset_dir: Path,
    agent_id: str,
    splits: Iterable[str],
    training_roles: Iterable[str],
    require_nominal_for_mae: bool,
    max_windows: int | None,
) -> set[str]:
    split_set = {str(item) for item in splits}
    role_set = {str(item) for item in training_roles}
    windows_path = dataset_dir / "agents" / agent_id / "windows.csv"
    labels_path = dataset_dir / "agents" / agent_id / "labels.csv"
    nominal_ids: set[str] = set()
    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                is_anomaly = str(row.get("is_anomaly", "")).strip().lower() in {"true", "1", "yes"}
                if not is_anomaly:
                    nominal_ids.add(str(row.get("window_id", "")).strip())

    selected: set[str] = set()
    with windows_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            window_id = str(row.get("window_id", "")).strip()
            if not window_id:
                continue
            if split_set and str(row.get("split", "")).strip() not in split_set:
                continue
            training_role = str(row.get("training_role", "")).strip()
            effective_role = training_role or "mae_reconstruction"
            if role_set and effective_role not in role_set:
                continue
            if require_nominal_for_mae and window_id not in nominal_ids:
                continue
            selected.add(window_id)
            if max_windows is not None and len(selected) >= int(max_windows):
                break
    return selected


def _load_agent_tensors(
    *,
    dataset_dir: Path,
    agent_id: str,
    feature_names: tuple[str, ...],
    selected_window_ids: set[str],
) -> list[tuple[list[list[float]], list[list[float]], str]]:
    raw_path = dataset_dir / "agents" / agent_id / "raw_sensor_readings.csv"
    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    windows: list[tuple[list[list[float]], list[list[float]], str]] = []
    current_id = None
    rows: list[dict[str, str]] = []
    with raw_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            window_id = str(row.get("window_id", "")).strip()
            if current_id is None:
                current_id = window_id
            if window_id != current_id:
                if current_id in selected_window_ids:
                    packed = _pack_window_rows(rows, feature_index)
                    if packed is not None:
                        windows.append((packed[0], packed[1], current_id))
                    if len(windows) >= len(selected_window_ids):
                        break
                rows = []
                current_id = window_id
            if window_id in selected_window_ids:
                rows.append(row)
        if current_id in selected_window_ids and len(windows) < len(selected_window_ids):
            packed = _pack_window_rows(rows, feature_index)
            if packed is not None:
                windows.append((packed[0], packed[1], current_id))
    return windows


def _pack_window_rows(
    rows: list[dict[str, str]],
    feature_index: dict[str, int],
) -> tuple[list[list[float]], list[list[float]]] | None:
    if not rows:
        return None
    has_explicit_seq = any(str(row.get("seq_index", "")).strip() not in {"", "None"} for row in rows)
    if has_explicit_seq:
        seq_keys = sorted({str(int(float(row.get("seq_index", 0) or 0))) for row in rows})
    else:
        seq_keys = sorted({str(row.get("timestamp", "")).strip() for row in rows if str(row.get("timestamp", "")).strip()})
    if not seq_keys:
        return None
    seq_to_row = {seq: pos for pos, seq in enumerate(seq_keys)}
    width = len(feature_index)
    matrix = [[0.0 for _ in range(width)] for _ in seq_keys]
    mask = [[0.0 for _ in range(width)] for _ in seq_keys]
    for row in rows:
        name = str(row.get("sensor_name") or row.get("name") or "").strip()
        if name not in feature_index:
            continue
        try:
            value = float(row.get("value", 0.0) or 0.0)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        quality = str(row.get("quality", "true")).strip().lower() in {"true", "1", "yes", "ok"}
        if has_explicit_seq:
            seq = str(int(float(row.get("seq_index", 0) or 0)))
        else:
            seq = str(row.get("timestamp", "")).strip()
        row_idx = seq_to_row.get(seq)
        if row_idx is None:
            continue
        col_idx = feature_index[name]
        matrix[row_idx][col_idx] = value
        mask[row_idx][col_idx] = 1.0 if quality else 0.0
    if not any(value for row in mask for value in row):
        # Real assets sometimes have zero genuine coverage for every
        # feature across every timestep in a window (e.g. a bridge whose
        # scour rating and hazard join both happened to be missing for
        # its whole reported history). Feeding an all-masked sequence
        # into the transformer encoder's key_padding_mask produces NaN
        # (softmax over an entirely-masked-out row), so drop it here
        # rather than let it poison the batch.
        return None
    return matrix, mask


def _stack_tensors(torch, tensors, *, device: str):
    # Real assets don't all have the same amount of real history (e.g. a
    # bridge with only 9 years of NBI inspections vs. one with the full
    # 16) -- torch.tensor(...) requires uniform shape, so left-pad every
    # window to the batch's longest one with mask=0 rows. The model
    # already treats mask=0 as "not observed" (see backbone_model.py's
    # masked mae_loss/rnn_consistency_loss), so padded steps are excluded
    # from the loss rather than silently treated as real zero-valued
    # observations.
    max_len = max((len(matrix) for matrix, _mask, _wid in tensors), default=0)
    width = len(tensors[0][0][0]) if tensors and tensors[0][0] else 0
    padded_matrices = []
    padded_masks = []
    for matrix, mask, _window_id in tensors:
        pad_amount = max_len - len(matrix)
        if pad_amount > 0:
            pad_row = [0.0] * width
            matrix = [list(pad_row) for _ in range(pad_amount)] + list(matrix)
            mask = [list(pad_row) for _ in range(pad_amount)] + list(mask)
        padded_matrices.append(matrix)
        padded_masks.append(mask)
    return (
        torch.tensor(padded_matrices, dtype=torch.float32, device=device),
        torch.tensor(padded_masks, dtype=torch.float32, device=device),
    )


def _build_label_tensors(torch, tensors, label_lookup: dict[str, float], *, device: str):
    """Align labels.csv's supervised target with `tensors`' window order.

    Returns (values, mask): mask=1 where a real, independently-observed
    label exists for that window; values is 0.0 (arbitrary, ignored via
    the mask) where it doesn't. Windows with no label are excluded from
    the supervised loss below, not treated as a negative example.
    """
    values = [float(label_lookup.get(window_id, 0.0)) for _matrix, _mask, window_id in tensors]
    present = [1.0 if window_id in label_lookup else 0.0 for _matrix, _mask, window_id in tensors]
    return (
        torch.tensor(values, dtype=torch.float32, device=device),
        torch.tensor(present, dtype=torch.float32, device=device),
    )


def _masked_mean_std(torch, matrix, mask):
    denom = mask.sum(dim=(0, 1)).clamp_min(1.0)
    mean = (matrix * mask).sum(dim=(0, 1)) / denom
    variance = (((matrix - mean.view(1, 1, -1)) ** 2) * mask).sum(dim=(0, 1)) / denom
    return mean, variance.sqrt().clamp_min(1e-3)


def _safe_tensor(torch, value, *, device: str):
    if hasattr(value, "detach"):
        return value
    return torch.tensor(float(value or 0.0), dtype=torch.float32, device=device)


def _import_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError("PyTorch is required for Cauren Core training.") from exc
    return torch


def _create_backbone_model(*, input_dim: int, mae_config: dict[str, Any], rnn_config: dict[str, Any], scnn_config: dict[str, Any], device: str):
    default_mae = {
        "d_model": 64,
        "n_encoder_layers": 2,
        "n_heads": 4,
        "d_ff_encoder": 128,
        "d_ff_decoder": 64,
    }
    default_rnn = {"hidden_dim": 128, "num_layers": 1}
    default_scnn = {"hidden_channels": 32}
    torch = _import_torch()
    model = create_cauren_backbone_model(
        torch,
        input_dim=int(input_dim),
        mae_config={**default_mae, **dict(mae_config or {})},
        rnn_config={**default_rnn, **dict(rnn_config or {})},
        scnn_config={**default_scnn, **dict(scnn_config or {})},
    )
    model.to(device)
    return model


def _set_seed(torch, seed: int) -> None:
    torch.manual_seed(int(seed))
    try:
        torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass
    os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
