"""Post-hoc calibration and conformal risk sets for bridge deterioration models.

The v3 benchmark showed that probabilities transfer poorly across states
(leave-one-state-out ECE 0.07-0.25). This tool measures three remedies on
the per-window predictions written by tools/bridge_hazard.py:

1. In-distribution recalibration: Platt or isotonic maps fitted on the
   validation split, pooled or per state, scored on the test split.
2. Leave-one-state-out plus k target labels: a model that never saw the
   held-out state is recalibrated on k labelled windows from that state's
   non-test bridges and scored on its test bridges. This is reported as
   "LOSO + k labels", never as pure transfer.
3. Mondrian split-conformal prediction sets per state at a nominal coverage,
   using nonconformity 1 - p(observed outcome). Sets are {0}, {1} or {0, 1};
   a two-element set means the model cannot separate the outcomes at that
   confidence for that state.

    python3 tools/bridge_calibration.py --predictions hazard/hazard_predictions.csv \\
        --model v4_hazard_end2023 --output-json hazard/calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from tools.bridge_model_metrics import score_metrics

EPSILON = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPSILON, 1.0 - EPSILON)
    return np.log(p / (1.0 - p))


def fit_platt(p: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Logistic recalibration on the logit of the raw probability."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1e4, max_iter=1000)
    model.fit(_logit(np.asarray(p, dtype=float)).reshape(-1, 1), np.asarray(y, dtype=float))
    return lambda values: model.predict_proba(_logit(np.asarray(values, dtype=float)).reshape(-1, 1))[:, 1]


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Monotone, non-decreasing step map from raw score to observed frequency."""
    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(y_min=EPSILON, y_max=1.0 - EPSILON, out_of_bounds="clip")
    model.fit(np.asarray(p, dtype=float), np.asarray(y, dtype=float))
    return lambda values: model.predict(np.asarray(values, dtype=float))


CALIBRATORS = {"platt": fit_platt, "isotonic": fit_isotonic}


def _summary(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    metrics = score_metrics(y.tolist(), p.tolist())
    return {
        "n": int(len(y)),
        "pr_auc": metrics.get("pr_auc_average_precision"),
        "brier": metrics.get("brier_score"),
        "ece": metrics.get("expected_calibration_error_10_equal_frequency_bins"),
        "calibration_slope": metrics.get("calibration_slope"),
    }


def in_distribution(frame: pd.DataFrame) -> dict[str, Any]:
    """Fit on validation, score on test; pooled and per-state (Mondrian) maps."""
    valid = frame[frame["split"].eq("validation")]
    test = frame[frame["split"].eq("test")]
    y_test, p_test = test["label"].to_numpy(float), test["probability"].to_numpy(float)
    report: dict[str, Any] = {"raw": _summary(y_test, p_test)}
    for name, fit in CALIBRATORS.items():
        pooled = fit(valid["probability"].to_numpy(float), valid["label"].to_numpy(float))
        report[f"{name}_pooled"] = _summary(y_test, pooled(p_test))
        per_state = np.empty_like(p_test)
        for state in test["state"].unique():
            target = test["state"].eq(state).to_numpy()
            source = valid[valid["state"].eq(state)]
            mapping = fit(source["probability"].to_numpy(float), source["label"].to_numpy(float))
            per_state[target] = mapping(p_test[target])
        report[f"{name}_per_state"] = _summary(y_test, per_state)
    return report


def loso_with_target_labels(
    frame: pd.DataFrame, *, k_values: tuple[int, ...] = (100, 300, 500), repeats: int = 20, seed: int = 42,
) -> dict[str, Any]:
    """Recalibrate held-out-state predictions on k labels from that state's non-test bridges."""
    rng = np.random.default_rng(seed)
    report: dict[str, Any] = {}
    for state in sorted(frame["state"].unique()):
        rows = frame[frame["state"].eq(state)]
        pool = rows[~rows["split"].eq("test")]
        test = rows[rows["split"].eq("test")]
        y_test, p_test = test["label"].to_numpy(float), test["probability"].to_numpy(float)
        state_report: dict[str, Any] = {"raw": _summary(y_test, p_test)}
        for k in k_values:
            if k > len(pool):
                continue
            for name, fit in CALIBRATORS.items():
                eces, briers = [], []
                for _ in range(repeats):
                    sample = pool.iloc[rng.choice(len(pool), size=k, replace=False)]
                    mapping = fit(sample["probability"].to_numpy(float), sample["label"].to_numpy(float))
                    summary = _summary(y_test, mapping(p_test))
                    eces.append(summary["ece"])
                    briers.append(summary["brier"])
                state_report[f"{name}_k{k}"] = {
                    "ece_mean": float(np.mean(eces)), "ece_p90": float(np.percentile(eces, 90)),
                    "brier_mean": float(np.mean(briers)), "repeats": repeats,
                }
        report[state] = state_report
    return report


def conformal_sets(frame: pd.DataFrame, *, coverage: float = 0.90) -> dict[str, Any]:
    """Mondrian (per-state) split-conformal sets; calibration is the validation split."""
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between 0 and 1")
    alpha = 1.0 - coverage
    report: dict[str, Any] = {"nominal_coverage": coverage}
    for state in sorted(frame["state"].unique()):
        rows = frame[frame["state"].eq(state)]
        calib = rows[rows["split"].eq("validation")]
        test = rows[rows["split"].eq("test")]
        y_c, p_c = calib["label"].to_numpy(float), calib["probability"].to_numpy(float)
        scores = np.where(y_c == 1.0, 1.0 - p_c, p_c)
        n = len(scores)
        if n == 0 or len(test) == 0:
            continue
        rank = min(n, math.ceil((n + 1) * (1.0 - alpha)))
        threshold = float(np.sort(scores)[rank - 1])
        p_t, y_t = test["probability"].to_numpy(float), test["label"].to_numpy(float)
        include_one = (1.0 - p_t) <= threshold
        include_zero = p_t <= threshold
        covered = np.where(y_t == 1.0, include_one, include_zero)
        size = include_one.astype(int) + include_zero.astype(int)
        report[state] = {
            "calibration_n": int(n),
            "test_n": int(len(test)),
            "threshold": threshold,
            "empirical_coverage": float(covered.mean()),
            "singleton_rate": float((size == 1).mean()),
            "both_outcomes_rate": float((size == 2).mean()),
            "empty_rate": float((size == 0).mean()),
        }
    return report


def run(predictions: Path, model: str, *, loso_prefix: str = "loso_v4_hazard_heldout_") -> dict[str, Any]:
    frame = pd.read_csv(predictions, dtype={"window_id": str, "asset_id": str, "state": str})
    chosen = frame[frame["model"].eq(model)]
    if chosen.empty:
        raise SystemExit(f"model {model!r} not found in {predictions}")
    report: dict[str, Any] = {
        "model": model,
        "in_distribution": in_distribution(chosen),
        "conformal_per_state": conformal_sets(chosen),
    }
    loso = frame[frame["model"].str.startswith(loso_prefix)]
    if not loso.empty:
        report["loso_plus_k_target_labels"] = loso_with_target_labels(loso)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    report = run(args.predictions, args.model)
    text = json.dumps(report, indent=2)
    if args.output_json:
        args.output_json.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
