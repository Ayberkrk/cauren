import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from tools.bridge_calibration import conformal_sets, fit_isotonic, fit_platt, in_distribution, loso_with_target_labels


def _frame(n=4000, distortion=2.5, seed=0):
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.02, 0.6, size=n)
    labels = (rng.uniform(size=n) < true_p).astype(float)
    # Overconfident raw scores: a monotone distortion of the true probability.
    logit = np.log(true_p / (1 - true_p)) * distortion
    raw = 1 / (1 + np.exp(-logit))
    split = np.where(np.arange(n) % 2 == 0, "validation", "test")
    state = np.where(np.arange(n) % 4 < 2, "06", "19")
    return pd.DataFrame({"label": labels, "probability": raw, "split": split, "state": state})


def test_platt_recovers_a_logit_scale_distortion():
    frame = _frame()
    valid = frame[frame["split"].eq("validation")]
    mapping = fit_platt(valid["probability"].to_numpy(), valid["label"].to_numpy())
    p = np.array([0.1, 0.5, 0.9])
    calibrated = mapping(p)
    assert np.all(np.diff(calibrated) > 0)
    assert calibrated[0] > 0.1 and calibrated[2] < 0.9


def test_isotonic_map_is_monotone():
    frame = _frame()
    mapping = fit_isotonic(frame["probability"].to_numpy(), frame["label"].to_numpy())
    grid = np.linspace(0, 1, 50)
    assert np.all(np.diff(mapping(grid)) >= 0)


def test_recalibration_lowers_ece_on_overconfident_scores():
    report = in_distribution(_frame())
    assert report["platt_pooled"]["ece"] < report["raw"]["ece"]
    assert report["isotonic_per_state"]["ece"] < report["raw"]["ece"]


def test_loso_target_labels_report_per_k():
    frame = _frame(n=2000).assign(split=lambda f: np.where(np.arange(len(f)) % 3 == 0, "test", "train"))
    report = loso_with_target_labels(frame, k_values=(100,), repeats=3)
    assert set(report) == {"06", "19"}
    assert report["06"]["platt_k100"]["ece_mean"] < report["06"]["raw"]["ece"]


def test_conformal_sets_reach_nominal_coverage():
    report = conformal_sets(_frame(n=8000), coverage=0.9)
    for state in ("06", "19"):
        assert report[state]["empirical_coverage"] >= 0.88
        assert report[state]["empty_rate"] <= 0.12
