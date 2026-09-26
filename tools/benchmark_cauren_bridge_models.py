"""Compare low-memory bridge deterioration baselines on fixed bridge splits.

The 836 MB readings CSV is streamed once. Feature summaries and all fits stay
in one process; BLAS/OpenMP thread counts are pinned to one before NumPy is
imported. The script needs NumPy. Scikit-learn is optional and enables a
HistGradientBoosting comparator. PyTorch is optional and enables evaluation of
the saved hybrid checkpoint through the existing checkpoint evaluator.

Example:
    python3 tools/benchmark_cauren_bridge_models.py \
        --dataset-dir data/cauren_bridge \
        --checkpoint cauren_core/checkpoints/cauren_bridge_backbone_bundle.pt

Metrics are reported on the existing bridge-disjoint test set. A second
leave-one-state-out analysis fits the additive logistic model using only the
official training windows from the other states and scores every window in
the held-out state. No split is reassigned and the test labels do not affect
feature scaling or model fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# A small, single-process benchmark should not fan out into BLAS/OpenMP worker
# pools, especially on the 8 GB machine used to build this local dataset.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = "1"

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.bridge_model_metrics import score_metrics


FEATURE_SUMMARIES = ("last", "first", "mean", "std", "minimum", "maximum", "slope_per_year", "missing_ratio")
CONDITION_FEATURES = (
    "deck_condition_risk_score",
    "superstructure_condition_risk_score",
    "substructure_condition_risk_score",
)


@dataclass
class MomentSummary:
    total: int = 0
    count: int = 0
    first_year: float = math.nan
    last_year: float = math.nan
    first_value: float = math.nan
    last_value: float = math.nan
    value_sum: float = 0.0
    value_squared_sum: float = 0.0
    year_sum: float = 0.0
    year_squared_sum: float = 0.0
    year_value_sum: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, year: float, raw_value: str, quality: str = "true") -> None:
        self.total += 1
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value) or quality.strip().lower() not in {"true", "1", "yes", "ok"}:
            return
        if self.count == 0:
            self.first_year = year
            self.first_value = value
        self.last_year = year
        self.last_value = value
        self.count += 1
        self.value_sum += value
        self.value_squared_sum += value * value
        self.year_sum += year
        self.year_squared_sum += year * year
        self.year_value_sum += year * value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def values(self) -> list[float]:
        if self.count == 0:
            return [math.nan] * 7 + [1.0]
        mean = self.value_sum / self.count
        variance = max(0.0, self.value_squared_sum / self.count - mean * mean)
        denominator = self.year_squared_sum - self.year_sum * self.year_sum / self.count
        slope = 0.0 if denominator <= 0.0 else (
            self.year_value_sum - self.year_sum * self.value_sum / self.count
        ) / denominator
        missing_ratio = 1.0 - self.count / max(1, self.total)
        return [
            self.last_value,
            self.first_value,
            mean,
            math.sqrt(variance),
            self.minimum,
            self.maximum,
            slope,
            missing_ratio,
        ]


def _header_index(header: list[str]) -> dict[str, int]:
    return {name: index for index, name in enumerate(header)}


def _read_windows(dataset_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    agent_dir = dataset_dir / "agents" / "cauren-bridge"
    windows: dict[str, dict[str, Any]] = {}
    asset_split: dict[str, str] = {}
    with (agent_dir / "windows.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            window_id = row["window_id"].strip()
            asset_id = row["asset_id"].strip()
            split = row["split"].strip()
            if split not in {"train", "validation", "test"}:
                raise ValueError(f"Unexpected split {split!r} for {window_id}")
            if asset_id in asset_split and asset_split[asset_id] != split:
                raise ValueError(f"Bridge {asset_id} occurs in multiple splits")
            asset_split[asset_id] = split
            windows[window_id] = {
                "asset_id": asset_id,
                "state": asset_id.split("_", 1)[0],
                "split": split,
                "seq_len": int(row.get("real_step_count") or row.get("seq_len") or 0),
            }

    labels: dict[str, float] = {}
    with (agent_dir / "labels.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = row.get("deck_drop_5yr", "").strip()
            if raw:
                labels[row["window_id"].strip()] = float(raw)
    missing = set(windows) - set(labels)
    if missing:
        raise ValueError(f"{len(missing)} windows have no deck_drop_5yr label")
    if set(labels) - set(windows):
        raise ValueError("labels.csv contains window ids absent from windows.csv")
    return windows, labels


def _empty_accumulators(feature_names: tuple[str, ...]) -> dict[str, MomentSummary]:
    return {feature: MomentSummary() for feature in feature_names}


def _summarize_window(
    window_id: str,
    accumulators: dict[str, MomentSummary],
    years: set[int],
    metadata_json: str,
    window: dict[str, Any],
    feature_names: tuple[str, ...],
) -> tuple[list[str], list[float]]:
    metadata: dict[str, Any] = {}
    if metadata_json:
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError:
            pass
    year_built = metadata.get("year_built")
    anchor_year = metadata.get("anchor_year")
    if anchor_year is None and years:
        anchor_year = max(years)
    elif anchor_year is not None and years and max(years) > int(anchor_year):
        # An input observation after the label's anchor year would leak the
        # outcome being predicted; refuse the dataset instead of scoring it.
        raise ValueError(f"{window_id} has readings from {max(years)}, after its anchor year {anchor_year}")
    try:
        bridge_age = float(anchor_year) - float(year_built)
    except (TypeError, ValueError):
        bridge_age = math.nan

    ordered_years = sorted(years)
    gaps = [float(right - left) for left, right in zip(ordered_years, ordered_years[1:])]
    gap_mean = sum(gaps) / len(gaps) if gaps else math.nan
    gap_std = math.sqrt(sum((gap - gap_mean) ** 2 for gap in gaps) / len(gaps)) if gaps else math.nan
    values: list[float] = []
    names: list[str] = []
    for feature in feature_names:
        summary_values = accumulators[feature].values()
        names.extend(f"{feature}__{name}" for name in FEATURE_SUMMARIES)
        values.extend(summary_values)
    timing = {
        "bridge_age_at_anchor_years": bridge_age,
        "inspection_count": float(len(ordered_years)),
        "inspection_span_years": float(ordered_years[-1] - ordered_years[0]) if len(ordered_years) > 1 else 0.0,
        "inspection_gap_mean_years": gap_mean,
        "inspection_gap_std_years": gap_std,
        "inspection_gap_max_years": max(gaps) if gaps else math.nan,
    }
    for name, value in timing.items():
        names.append(name)
        values.append(value)
    if len(ordered_years) != window["seq_len"]:
        raise ValueError(
            f"{window_id} has {len(ordered_years)} distinct inspection years in readings but "
            f"windows.csv says {window['seq_len']}"
        )
    return names, values


def _load_core_history(
    dataset_dir: Path,
    windows: dict[str, dict[str, Any]],
    feature_names: tuple[str, ...],
) -> tuple[list[str], dict[str, list[float]]]:
    path = dataset_dir / "agents" / "cauren-bridge" / "raw_sensor_readings.csv"
    print(f"Streaming bridge readings from {path} (single pass)", flush=True)
    feature_index = set(feature_names)
    vectors: dict[str, list[float]] = {}
    current_id: str | None = None
    accumulators = _empty_accumulators(feature_names)
    years: set[int] = set()
    metadata_json = ""
    rows_seen = 0
    expected_names: list[str] | None = None

    def flush(window_id: str | None) -> None:
        nonlocal expected_names
        if window_id is None:
            return
        if window_id not in windows:
            raise ValueError(f"Readings contain unknown window id {window_id}")
        if window_id in vectors:
            # The single-pass summary needs each window's rows to be contiguous.
            raise ValueError(f"Readings for {window_id} are not contiguous in raw_sensor_readings.csv")
        names, values = _summarize_window(
            window_id, accumulators, years, metadata_json, windows[window_id], feature_names
        )
        if expected_names is None:
            expected_names = names
        vectors[window_id] = values

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = _header_index(next(reader))
        required = {"window_id", "timestamp", "sensor_name", "value", "quality", "metadata_json"}
        if required - header.keys():
            raise ValueError(f"raw_sensor_readings.csv lacks columns: {sorted(required - header.keys())}")
        for row in reader:
            rows_seen += 1
            window_id = row[header["window_id"]].strip()
            if window_id != current_id:
                flush(current_id)
                current_id = window_id
                accumulators = _empty_accumulators(feature_names)
                years = set()
                metadata_json = row[header["metadata_json"]]
            sensor_name = row[header["sensor_name"]].strip()
            if sensor_name not in feature_index:
                continue
            raw_timestamp = row[header["timestamp"]]
            try:
                year = int(raw_timestamp[:4])
            except (TypeError, ValueError):
                continue
            years.add(year)
            accumulators[sensor_name].add(year, row[header["value"]], row[header["quality"]])
            if rows_seen % 250_000 == 0:
                print(f"  parsed {rows_seen:,} readings; completed {len(vectors):,} windows", flush=True)
    flush(current_id)

    if set(vectors) != set(windows):
        missing = len(set(windows) - set(vectors))
        extra = len(set(vectors) - set(windows))
        raise ValueError(f"Readings/window mismatch: {missing} missing windows, {extra} unexpected windows")
    assert expected_names is not None
    return expected_names, vectors


def _load_condition_history(path: Path, windows: dict[str, dict[str, Any]]) -> tuple[list[str], dict[str, list[float]]]:
    if not path.exists():
        return [], {}
    accumulators: dict[str, dict[str, MomentSummary]] = {}
    years_by_window: dict[str, set[int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"window_id", "inspection_year", *CONDITION_FEATURES}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        for row in reader:
            window_id = row["window_id"].strip()
            if window_id not in windows:
                raise ValueError(f"Condition history contains unknown window id {window_id}")
            year = int(row["inspection_year"])
            years_by_window.setdefault(window_id, set()).add(year)
            bucket = accumulators.setdefault(window_id, _empty_accumulators(CONDITION_FEATURES))
            for feature in CONDITION_FEATURES:
                bucket[feature].add(year, row.get(feature, ""), "true")
    if set(accumulators) != set(windows):
        raise ValueError("condition_history.csv must cover every labeled bridge window")
    for window_id, years in years_by_window.items():
        if len(years) != windows[window_id]["seq_len"]:
            raise ValueError(f"{window_id} component history does not match its observation count")
    names = [f"{feature}__{summary}" for feature in CONDITION_FEATURES for summary in FEATURE_SUMMARIES]
    vectors: dict[str, list[float]] = {}
    for window_id, feature_bucket in accumulators.items():
        values = [value for feature in CONDITION_FEATURES for value in feature_bucket[feature].values()]
        vectors[window_id] = values
    return names, vectors


def _matrix(
    windows: dict[str, dict[str, Any]],
    labels: dict[str, float],
    base_names: list[str],
    base_vectors: dict[str, list[float]],
    history_names: list[str],
    history_vectors: dict[str, list[float]],
) -> tuple[list[str], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    names = base_names + history_names
    rows = []
    targets = []
    metadata = []
    for window_id, window in windows.items():
        values = base_vectors[window_id] + history_vectors.get(window_id, [])
        rows.append(values)
        targets.append(labels[window_id])
        metadata.append(window)
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape[1] != len(names):
        raise ValueError(f"Feature matrix width {matrix.shape[1]} does not match {len(names)} names")
    return names, matrix, np.asarray(targets, dtype=np.float64), metadata


def _fit_logistic(x: np.ndarray, y: np.ndarray, *, l2: float = 0.2, max_iter: int = 60) -> dict[str, Any]:
    if len(x) == 0 or len(y) != len(x):
        raise ValueError("logistic regression requires non-empty aligned training data")
    full_means = np.nanmean(x, axis=0)
    full_means = np.where(np.isfinite(full_means), full_means, 0.0)
    imputed = np.where(np.isfinite(x), x, full_means)
    scales = np.std(imputed, axis=0)
    keep = scales > 1e-8
    means = full_means[keep]
    selected_scales = scales[keep]
    standardized = np.clip((imputed[:, keep] - means) / selected_scales, -20.0, 20.0)
    design = np.column_stack((standardized, np.ones(len(standardized))))
    prior = float(np.clip(np.mean(y), 1e-5, 1.0 - 1e-5))
    theta = np.zeros(design.shape[1], dtype=np.float64)
    theta[-1] = math.log(prior / (1.0 - prior))
    regularizer = np.eye(design.shape[1], dtype=np.float64) * l2
    regularizer[-1, -1] = 0.0
    for _ in range(max_iter):
        logits = np.clip(design @ theta, -35.0, 35.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        residual = probabilities - y
        gradient = design.T @ residual + regularizer @ theta
        weights = np.maximum(probabilities * (1.0 - probabilities), 1e-7)
        hessian = design.T @ (design * weights[:, None]) + regularizer
        step = np.linalg.solve(hessian, gradient)
        theta -= step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return {
        "means": means,
        "full_means": full_means,
        "scales": selected_scales,
        "keep": keep,
        "theta": theta,
        "prior": prior,
    }


def _predict_logistic(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    # Imputation values and scaling statistics are fitted on training rows only.
    imputed = np.where(np.isfinite(x), x, model["full_means"])
    standardized = np.clip((imputed[:, model["keep"]] - model["means"]) / model["scales"], -20.0, 20.0)
    design = np.column_stack((standardized, np.ones(len(standardized))))
    logits = np.clip(design @ model["theta"], -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _fit_and_score_logistic(
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    score_masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    model = _fit_logistic(x[train_mask], y[train_mask])
    predictions = _predict_logistic(model, x)
    return {
        split: score_metrics(y[mask].tolist(), predictions[mask].tolist())
        for split, mask in score_masks.items()
    }


def _fit_hist_gradient_boosting(
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    score_masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError:
        return {"status": "skipped", "reason": "scikit-learn is not installed; install the benchmark extra"}
    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=160,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=42,
    )
    model.fit(x[train_mask], y[train_mask].astype(int))
    predictions = model.predict_proba(x)[:, 1]
    return {
        "status": "completed",
        "configuration": {
            "learning_rate": 0.06,
            "max_iter": 160,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 100,
            "l2_regularization": 2.0,
        },
        "metrics": {
            split: score_metrics(y[mask].tolist(), predictions[mask].tolist())
            for split, mask in score_masks.items()
        },
    }


def _run_hybrid(dataset_dir: Path, checkpoint: Path) -> dict[str, Any]:
    if not checkpoint.exists():
        return {"status": "skipped", "reason": f"checkpoint not found: {checkpoint}"}
    try:
        from tools.evaluate_cauren_bridge_checkpoint import evaluate_on_test_split

        metrics = evaluate_on_test_split(
            dataset_dir=dataset_dir,
            checkpoint_path=checkpoint,
            agent_id="cauren-bridge",
        )
    except RuntimeError:
        return {"status": "skipped", "reason": "PyTorch is not installed in this Python environment"}
    return {"status": "completed", "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/cauren_bridge"))
    parser.add_argument("--checkpoint", type=Path, default=Path("cauren_core/checkpoints/cauren_bridge_backbone_bundle.pt"))
    parser.add_argument("--skip-hybrid", action="store_true", help="Do not make a second streamed pass for the saved neural checkpoint")
    parser.add_argument("--output-json", type=Path, help="Optional path for the complete machine-readable report")
    args = parser.parse_args()

    windows, labels = _read_windows(args.dataset_dir)
    summary_path = args.dataset_dir / "dataset_summary.json"
    dataset_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    feature_names = tuple(dataset_summary["features"])
    base_names, base_vectors = _load_core_history(args.dataset_dir, windows, feature_names)
    history_path = args.dataset_dir / "agents" / "cauren-bridge" / "condition_history.csv"
    history_names, history_vectors = _load_condition_history(history_path, windows)
    names, x, y, metadata = _matrix(windows, labels, base_names, base_vectors, history_names, history_vectors)

    splits = np.asarray([row["split"] for row in metadata])
    states = np.asarray([row["state"] for row in metadata])
    train_mask = splits == "train"
    validation_mask = splits == "validation"
    test_mask = splits == "test"
    if not train_mask.any() or not test_mask.any():
        raise ValueError("train and test splits must be non-empty")

    train_rate = float(np.mean(y[train_mask]))
    prior_predictions = np.full(len(y), train_rate)
    structural_feature = names.index("structural_risk_score__last")
    last_score_model = _fit_logistic(x[train_mask][:, [structural_feature]], y[train_mask])
    last_score_predictions = _predict_logistic(last_score_model, x[:, [structural_feature]])

    score_masks = {"validation": validation_mask, "test": test_mask}
    full_logistic_metrics = _fit_and_score_logistic(x, y, train_mask, score_masks)
    persistence_metrics = {
        split: score_metrics(y[mask].tolist(), last_score_predictions[mask].tolist())
        for split, mask in score_masks.items()
    }
    prior_metrics = {
        split: score_metrics(y[mask].tolist(), prior_predictions[mask].tolist())
        for split, mask in score_masks.items()
    }

    state_holdouts = {}
    for state in sorted(set(states.tolist())):
        state_test = states == state
        state_train = (splits == "train") & ~state_test
        if not state_train.any() or not state_test.any():
            continue
        state_model = _fit_logistic(x[state_train], y[state_train])
        state_predictions = _predict_logistic(state_model, x[state_test])
        state_holdouts[state] = score_metrics(y[state_test].tolist(), state_predictions.tolist())

    report: dict[str, Any] = {
        "dataset": {
            "dataset_id": dataset_summary.get("dataset_id"),
            "windows": len(windows),
            "positive_rate": dataset_summary.get("positive_rate"),
            "split_counts": {
                split: int(np.sum(splits == split)) for split in ("train", "validation", "test")
            },
            "states": {state: int(np.sum(states == state)) for state in sorted(set(states.tolist()))},
            "bridge_disjoint_splits_verified": True,
            "input_rows_truncated_at_anchor_year": True,
        },
        "feature_audit": {
            "tabular_feature_count": len(names),
            "features": names,
            "component_rating_history_available": bool(history_names),
            "component_history_path": str(history_path) if history_names else None,
            "note": (
                "Bridge age and actual inspection-year gaps are included. Separate component-rating "
                "history is included only when condition_history.csv exists; no values are inferred "
                "from the composite structural score."
            ),
        },
        "models": {
            "majority_prevalence_probability": {
                "train_positive_rate": round(train_rate, 6),
                "metrics": prior_metrics,
            },
            "current_structural_score_logistic": {
                "features": ["structural_risk_score__last"],
                "metrics": persistence_metrics,
            },
            "additive_logistic_with_time_and_age": {
                "features": len(names),
                "l2": 0.2,
                "metrics": full_logistic_metrics,
            },
            "hist_gradient_boosting": _fit_hist_gradient_boosting(x, y, train_mask, score_masks),
        },
        "leave_one_state_out_additive_logistic": state_holdouts,
        "saved_hybrid_checkpoint": (
            {"status": "skipped", "reason": "disabled by --skip-hybrid"}
            if args.skip_hybrid
            else _run_hybrid(args.dataset_dir, args.checkpoint)
        ),
        "protocol": {
            "model_fitting_split": "train only",
            "threshold": 0.5,
            "threshold_tuned_on_test": False,
            "capacity_metrics": ["top_5_percent", "top_10_percent"],
            "linear_model": "L2-regularized additive logistic regression fit by Newton iterations",
            "execution": "single process; BLAS/OpenMP thread counts pinned to one",
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
