"""Discrete-time deck deterioration hazard model on the v4 bridge-year panel.

The v3 benchmark models predict ``deck_drop_5yr`` from one window per bridge.
This tool instead fits the one-year hazard of a deck rating drop on every
observed bridge-year interval (``label_deck_drop_next``), then turns it into
a five-year probability by rolling the bridge's clock forward:

    P(drop within H years | x_t) = 1 - prod_{k=0}^{H-1} (1 - h(x_{t+k}))

Rolling forward advances age, time at the current rating, time since
reconstruction and the inspection schedule (item 091), and assumes the
rating has not yet changed, which is exactly the survival condition. All
other covariates stay at their anchor values. Because every factor is a
probability, the cumulative risk can only grow with the horizon: the model
can describe deterioration but never an implied repair.

NBI ratings only change when an inspection happens (in the v4 panel the
one-year drop rate is about 0.1% without a new inspection and about 14%
with one), so the inspection clock is a first-class covariate.

Scoring uses the exact v3 test windows (asset id, anchor year and label from
data/cauren_bridge), so results line up with
tools/benchmark_cauren_bridge_models.py predictions and its paired
bridge-cluster bootstrap.

    python3 tools/bridge_hazard.py --panel data/cauren_bridge_v4/bridge_years.parquet \\
        --v3-dataset-dir data/cauren_bridge --output-dir data/cauren_bridge_v4/hazard
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools.build_bridge_years_v4 import CATEGORICAL_FEATURES, MODEL_FEATURES

HORIZON_YEARS = 5
# Covariates where higher values should never lower the hazard. Ratings
# themselves get no constraint: a high rating can drop sooner than a low one.
MONOTONE_INCREASING = ("age_years", "years_since_reconstruction", "log_adt", "truck_adt")
HAZARD_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "verbose": -1,
    "seed": 42,
    "deterministic": True,
    "num_threads": 0,
}


def _lgb():
    try:
        import lightgbm
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise SystemExit("bridge_hazard requires lightgbm: pip install lightgbm") from exc
    return lightgbm


def feature_columns() -> list[str]:
    return [*MODEL_FEATURES, *CATEGORICAL_FEATURES]


def monotone_vector(columns: list[str]) -> list[int]:
    return [1 if column in MONOTONE_INCREASING else 0 for column in columns]


def read_v3_windows(dataset_dir: Path) -> pd.DataFrame:
    """Return v3 windows with asset id, anchor year, split, state and label."""
    agent_dir = dataset_dir / "agents" / "cauren-bridge"
    windows = pd.read_csv(agent_dir / "windows.csv", usecols=["window_id", "asset_id", "window_end", "split"], dtype=str)
    labels = pd.read_csv(agent_dir / "labels.csv", usecols=["window_id", "deck_drop_5yr"], dtype={"window_id": str})
    frame = windows.merge(labels, on="window_id", how="inner")
    frame["anchor_year"] = frame["window_end"].str[:4].astype(int)
    frame["state"] = frame["asset_id"].str.split("_", n=1).str[0]
    frame = frame.rename(columns={"deck_drop_5yr": "label"})
    return frame[frame["label"].notna()].reset_index(drop=True)


def hazard_rows(panel: pd.DataFrame, bridge_ids: set[str], *, last_interval_end: int) -> pd.DataFrame:
    """Observed one-year intervals for the given bridges ending by a cutoff year."""
    mask = (
        panel["bridge_id"].isin(bridge_ids)
        & panel["label_deck_drop_next"].notna()
        & ((panel["year"] + 1) <= last_interval_end)
    )
    return panel.loc[mask]


def roll_forward(anchor: pd.DataFrame, step: int) -> pd.DataFrame:
    """Advance anchor rows by ``step`` years assuming no rating change yet."""
    if step == 0:
        return anchor
    rolled = anchor.copy()
    for column in ("age_years", "years_since_reconstruction", "years_at_deck_rating"):
        rolled[column] = rolled[column] + step
    rolled["deck_prev"] = rolled["deck_cond"]
    rolled["deck_delta"] = np.float32(0.0)
    period = (rolled["inspect_freq_months"].where(rolled["inspect_freq_months"] > 0) / 12.0).fillna(2.0)
    elapsed = rolled["years_since_inspection"].fillna(0.0)
    before = np.floor((elapsed + step - 1) / period)
    after = np.floor((elapsed + step) / period)
    rolled["inspected_this_year"] = (after > before).astype(np.float32)
    rolled["years_since_inspection"] = ((elapsed + step) - after * period).astype(np.float32)
    return rolled


def cumulative_risk(booster: Any, anchor: pd.DataFrame, horizon: int = HORIZON_YEARS) -> np.ndarray:
    """Probability of at least one deck drop within ``horizon`` years."""
    columns = feature_columns()
    survival = np.ones(len(anchor))
    for step in range(horizon):
        hazard = booster.predict(roll_forward(anchor, step)[columns])
        survival *= 1.0 - np.clip(hazard, 0.0, 1.0)
    return 1.0 - survival


def risk_by_horizon(booster: Any, anchor: pd.DataFrame, horizon: int = HORIZON_YEARS) -> np.ndarray:
    """Cumulative risk for horizons 1..H as an (n, H) array."""
    columns = feature_columns()
    survival = np.ones(len(anchor))
    curves = np.empty((len(anchor), horizon))
    for step in range(horizon):
        survival = survival * (1.0 - np.clip(booster.predict(roll_forward(anchor, step)[columns]), 0.0, 1.0))
        curves[:, step] = 1.0 - survival
    return curves


def top_features(booster: Any, count: int = 15) -> list[dict[str, float]]:
    gains = booster.feature_importance(importance_type="gain")
    total = float(gains.sum()) or 1.0
    order = np.argsort(gains)[::-1][:count]
    names = booster.feature_name()
    return [{"feature": names[i], "gain_share": round(float(gains[i]) / total, 4)} for i in order]


def _dataset(lgb: Any, rows: pd.DataFrame, label: str, reference: Any = None) -> Any:
    return lgb.Dataset(
        rows[feature_columns()], label=rows[label].to_numpy(), categorical_feature=CATEGORICAL_FEATURES,
        reference=reference, free_raw_data=True,
    )


def fit_booster(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    label: str,
    *,
    monotone: bool,
    max_rounds: int = 2000,
) -> Any:
    lgb = _lgb()
    params = dict(HAZARD_PARAMS)
    if monotone:
        params["monotone_constraints"] = monotone_vector(feature_columns())
        params["monotone_constraints_method"] = "advanced"
    train_set = _dataset(lgb, train, label)
    valid_set = _dataset(lgb, valid, label, reference=train_set)
    return lgb.train(
        params, train_set, num_boost_round=max_rounds, valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )


def anchors_for(panel: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Panel rows at each v3 window's anchor, aligned to ``windows`` order."""
    keyed = panel.set_index(["bridge_id", "year"])
    index = pd.MultiIndex.from_arrays([windows["asset_id"], windows["anchor_year"]])
    missing = ~index.isin(keyed.index)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} v3 windows have no matching v4 panel row")
    return keyed.loc[index].reset_index()


def _prediction_rows(windows: pd.DataFrame, model: str, probability: np.ndarray) -> list[tuple]:
    return [
        (row.window_id, row.asset_id, row.state, row.split, int(float(row.label) >= 0.5), model, f"{float(p):.10f}")
        for row, p in zip(windows.itertuples(index=False), probability)
    ]


def run(
    panel_path: Path,
    v3_dataset_dir: Path,
    output_dir: Path,
    *,
    interval_end_years: tuple[int, ...] = (2018, 2023),
    run_loso: bool = True,
    run_loso_hazard: bool = True,
) -> dict[str, Any]:
    from tools.bridge_model_metrics import score_metrics

    output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(panel_path)
    windows = read_v3_windows(v3_dataset_dir)
    anchors = anchors_for(panel, windows)
    agreement = float((anchors["label_deck_drop_5yr"] == windows["label"].astype(float)).mean())

    test = windows["split"].eq("test").to_numpy()
    # Bridges too short for a v3 window still contribute hazard intervals; the
    # panel split reuses v3 assignments and hash-splits the rest, so no test
    # bridge enters training.
    bridge_split = panel.drop_duplicates("bridge_id").set_index("bridge_id")["split"]
    train_bridges = set(bridge_split.index[bridge_split.eq("train")])
    valid_bridges = set(bridge_split.index[bridge_split.eq("validation")])
    predictions: list[tuple] = []
    results: dict[str, Any] = {"v4_label_agreement_with_v3": agreement, "models": {}}

    def record(name: str, probability: np.ndarray, extra: dict[str, Any]) -> None:
        metrics = score_metrics(windows.loc[test, "label"].astype(float).tolist(), probability[test].tolist())
        print(f"{name}: test PR-AUC {metrics.get('pr_auc_average_precision'):.4f}, "
              f"Brier {metrics.get('brier_score'):.4f}", flush=True)
        results["models"][name] = {**extra, "test_metrics": metrics}
        # Validation rows are kept for post-hoc calibration (tools/bridge_calibration.py);
        # the paired bootstrap only uses windows every model scored, i.e. test.
        predictions.extend(_prediction_rows(windows, name, probability))

    for end_year in interval_end_years:
        train_rows = hazard_rows(panel, train_bridges, last_interval_end=end_year)
        valid_rows = hazard_rows(panel, valid_bridges, last_interval_end=end_year)
        for monotone in (False, True):
            booster = fit_booster(train_rows, valid_rows, "label_deck_drop_next", monotone=monotone)
            name = f"v4_hazard_end{end_year}{'_monotone' if monotone else ''}"
            curves = risk_by_horizon(booster, anchors)
            if not np.all(np.diff(curves, axis=1) >= -1e-12):
                raise AssertionError("cumulative risk decreased with horizon")
            record(name, curves[:, -1], {
                "formulation": "discrete_time_hazard",
                "last_interval_end_year": end_year,
                "monotone_constraints": list(MONOTONE_INCREASING) if monotone else [],
                "training_intervals": int(len(train_rows)),
                "best_iteration": int(booster.best_iteration),
            })
            booster.save_model(str(output_dir / f"{name}.txt"))

    # The same v4 covariates in the v3 single-window formulation, trained on
    # the v3 anchor of every train bridge.
    is_train = windows["split"].eq("train").to_numpy()
    is_valid = windows["split"].eq("validation").to_numpy()
    labels = windows["label"].astype(float).to_numpy()
    train_anchor = anchors[is_train].assign(y=labels[is_train])
    valid_anchor = anchors[is_valid].assign(y=labels[is_valid])
    for monotone in (False, True):
        single = fit_booster(train_anchor, valid_anchor, "y", monotone=monotone)
        name = f"v4_single_window_boosting{'_monotone' if monotone else ''}"
        record(name, single.predict(anchors[feature_columns()]), {
            "formulation": "single_window_classification",
            "monotone_constraints": list(MONOTONE_INCREASING) if monotone else [],
            "training_windows": int(len(train_anchor)),
            "best_iteration": int(single.best_iteration),
            "top_features_by_gain": top_features(single),
        })
        single.save_model(str(output_dir / f"{name}.txt"))

    # Out of time: train only on bridge-years whose five-year outcome was known
    # by the test anchors (label year <= first test anchor year), then score the
    # test anchors. This is the deployment setting: learn from the past,
    # forecast the next five years.
    anchor_year = int(windows.loc[test, "anchor_year"].mode().iloc[0])
    cutoff = anchor_year - HORIZON_YEARS
    labeled = panel["label_deck_drop_5yr"].notna() & (panel["year"] <= cutoff)
    past_train = panel.loc[labeled & panel["bridge_id"].isin(train_bridges)].rename(columns={"label_deck_drop_5yr": "y"})
    past_valid = panel.loc[labeled & panel["bridge_id"].isin(valid_bridges)].rename(columns={"label_deck_drop_5yr": "y"})
    past = fit_booster(past_train, past_valid, "y", monotone=False)
    past_probability = past.predict(anchors[feature_columns()])
    same_period = results["models"]["v4_single_window_boosting"]
    modal = test & windows["anchor_year"].eq(anchor_year).to_numpy()
    record("v4_single_window_out_of_time", past_probability, {
        "formulation": "single_window_classification_out_of_time",
        "training_rows": int(len(past_train)),
        "last_anchor_year_in_training": cutoff,
        "best_iteration": int(past.best_iteration),
        "test_metrics_modal_anchor_year_only": {
            "anchor_year": anchor_year,
            "out_of_time": score_metrics(labels[modal].tolist(), past_probability[modal].tolist()),
        },
    })
    same_period["test_metrics_modal_anchor_year_only"] = {"anchor_year": anchor_year}

    if run_loso:
        results["leave_one_state_out"] = {}
        for state in sorted(windows["state"].unique()):
            print(f"leave-one-state-out fold {state}", flush=True)
            held = windows["state"].eq(state).to_numpy()
            others = ~windows["state"].eq(state).to_numpy()
            fold_train = {b for b in train_bridges if not b.startswith(state + "_")}
            fold_valid = {b for b in valid_bridges if not b.startswith(state + "_")}
            fold: dict[str, Any] = {}
            single = fit_booster(
                anchors[is_train & others].assign(y=labels[is_train & others]),
                anchors[is_valid & others].assign(y=labels[is_valid & others]),
                "y", monotone=False,
            )
            probability = single.predict(anchors.loc[held, feature_columns()])
            fold["v4_single_window_boosting"] = score_metrics(labels[held].tolist(), probability.tolist())
            predictions.extend(_prediction_rows(windows.loc[held], f"loso_v4_single_window_heldout_{state}", probability))
            if run_loso_hazard:
                booster = fit_booster(
                    hazard_rows(panel, fold_train, last_interval_end=max(interval_end_years)),
                    hazard_rows(panel, fold_valid, last_interval_end=max(interval_end_years)),
                    "label_deck_drop_next", monotone=False,
                )
                probability = cumulative_risk(booster, anchors[held])
                fold["v4_hazard"] = score_metrics(labels[held].tolist(), probability.tolist())
                predictions.extend(_prediction_rows(windows.loc[held], f"loso_v4_hazard_heldout_{state}", probability))
            results["leave_one_state_out"][state] = fold

    with open(output_dir / "hazard_predictions.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("window_id", "asset_id", "state", "split", "label", "model", "probability"))
        writer.writerows(predictions)
    with open(output_dir / "hazard_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=lambda value: None if isinstance(value, float) and math.isnan(value) else value)
    return results


def compare(prediction_files: list[Path], reference_model: str, replicates: int = 2000) -> dict[str, Any]:
    """Paired bridge-cluster bootstrap across prediction files on one test set."""
    from tools.bridge_cluster_bootstrap import paired_cluster_bootstrap

    frame = pd.concat([pd.read_csv(path, dtype={"window_id": str, "asset_id": str}) for path in prediction_files])
    frame = frame[frame["split"].eq("test") & ~frame["model"].str.startswith("loso_")]
    wide = frame.pivot_table(index=["window_id", "asset_id", "label"], columns="model", values="probability").dropna()
    wide = wide.reset_index()
    models = [column for column in wide.columns if column not in {"window_id", "asset_id", "label"}]
    return paired_cluster_bootstrap(
        wide["label"].tolist(), wide["asset_id"].tolist(), {name: wide[name].tolist() for name in models},
        reference_model=reference_model, replicates=replicates,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--v3-dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-loso", action="store_true")
    parser.add_argument("--hazard-end-years", type=int, nargs="*", default=[2018, 2023],
                        help="hazard training cutoffs; pass none to skip the hazard arm")
    parser.add_argument("--compare-with", type=Path, default=None,
                        help="benchmark model_predictions.csv to bootstrap against")
    parser.add_argument("--reference-model", default="hist_gradient_boosting")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = run(args.panel, args.v3_dataset_dir, args.output_dir, run_loso=not args.skip_loso,
                  interval_end_years=tuple(args.hazard_end_years), run_loso_hazard=bool(args.hazard_end_years))
    if args.compare_with is not None:
        results["paired_bridge_cluster_bootstrap"] = compare(
            [args.compare_with, args.output_dir / "hazard_predictions.csv"], args.reference_model)
        with open(args.output_dir / "hazard_results.json", "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
