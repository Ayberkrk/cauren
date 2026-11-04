"""Environmental normalization and modal novelty detection under one protocol.

Every bridge study scores its identified natural frequencies with the same
code so the results are comparable across structures:

1. Monitored modes are chosen from the reference period only.
2. A per-mode linear temperature model, fitted on the reference period,
   removes the environmental part of the frequency variation.
3. Three detectors are fitted on the reference period: Mahalanobis distance
   of the raw frequencies, Mahalanobis distance of the temperature residuals,
   and the residual outside the leading principal components.
4. Hourly scores are reduced to daily medians. Each day gets a conformal
   p-value against the reference days, and a day alarms when p <= alpha.
   Alarms are counted in a later healthy holdout (false alarms) and after
   the known change (detections).

A study supplies the hourly times, frequencies and temperature plus its
segment dates; nothing here is specific to one bridge. Unlike
``cauren_physics.oma``, this module needs numpy.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import numpy as np

# Mode selection only looks at the reference period so the post-change data
# cannot influence which modes are monitored.
MAX_REFERENCE_MISSING = 0.10
HINGE_TEMPERATURE_C = 0.0
# Target false-alarm rate per healthy day. Needs at least 99 reference days.
ALPHA = 0.01
MIN_HOURS_PER_DAY = 6
PCA_RETAINED_VARIANCE = 0.90

Segment = tuple[dt.datetime, dt.datetime]


@dataclass(frozen=True)
class StudyProtocol:
    """Half-open ``[start, end)`` segments of one study.

    ``reference`` fits every model and control limit, ``holdout`` is later
    healthy data scored for false alarms, and ``changed`` is the structure
    after the known change, scored for detections. Anything outside the
    three segments (for example the works period itself) is ignored.
    """

    reference: Segment
    holdout: Segment
    changed: Segment


def segment_mask(times, segment: Segment) -> np.ndarray:
    start, end = segment
    return np.array([start <= stamp < end for stamp in times], dtype=bool)


def select_modes(frequencies_hz: np.ndarray, reference: np.ndarray, max_missing: float = MAX_REFERENCE_MISSING):
    """Indices of modes identified in at least ``1 - max_missing`` of reference hours."""
    missing = np.isnan(frequencies_hz[reference]).mean(axis=0)
    return [int(index) for index in np.flatnonzero(missing <= max_missing)]


def temperature_features(temperature_c, hinge_c: float = HINGE_TEMPERATURE_C) -> np.ndarray:
    """Bilinear temperature covariates ``[T, min(T - hinge, 0)]``.

    The second column lets the slope change below the hinge, where frozen
    ballast, asphalt and bearings stiffen a deck far more than the material
    modulus alone would.
    """
    temperature = np.asarray(temperature_c, dtype=float)
    return np.column_stack((temperature, np.minimum(temperature - hinge_c, 0.0)))


class LinearBaseline:
    """Per-mode linear frequency model ``f = f_ref + C (x - x_ref)``."""

    def __init__(self, reference_environment, reference_frequencies_hz, coefficients_hz_per_unit, source: str):
        self.reference_environment = np.asarray(reference_environment, dtype=float)
        self.reference_frequencies_hz = np.asarray(reference_frequencies_hz, dtype=float)
        self.coefficients = np.asarray(coefficients_hz_per_unit, dtype=float)
        self.source = source

    def predict(self, covariates) -> np.ndarray:
        delta = np.asarray(covariates, dtype=float) - self.reference_environment
        return self.reference_frequencies_hz + delta @ self.coefficients.T

    def residuals(self, covariates, frequencies_hz) -> np.ndarray:
        return np.asarray(frequencies_hz, dtype=float) - self.predict(covariates)


def fit_temperature_baseline(covariates, frequencies_hz, feature_names=("temperature_c", "below_hinge_c")):
    """Fit the environmental baseline with Timoshenko when available."""
    try:
        from timoshenko import fit_environmental_baseline
    except ImportError:
        fit_environmental_baseline = None
    if fit_environmental_baseline is not None:
        fitted = fit_environmental_baseline(covariates, frequencies_hz, feature_names=feature_names)
        return LinearBaseline(
            fitted.reference_environment,
            fitted.reference_frequencies_hz,
            fitted.coefficients_hz_per_unit,
            "timoshenko.fit_environmental_baseline",
        )
    x = np.asarray(covariates, dtype=float)
    y = np.asarray(frequencies_hz, dtype=float)
    reference = x.mean(axis=0)
    design = np.column_stack((np.ones(len(x)), x - reference))
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("temperature covariates are rank-deficient")
    return LinearBaseline(reference, coefficients[0], coefficients[1:].T, "numpy least squares")


def ledoit_wolf_covariance(samples: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf (2004) shrinkage of the sample covariance toward a scaled identity."""
    x = np.asarray(samples, dtype=float)
    n, p = x.shape
    centered = x - x.mean(axis=0)
    sample = centered.T @ centered / n
    mu = np.trace(sample) / p
    target = mu * np.eye(p)
    delta = np.sum((sample - target) ** 2) / p
    beta = sum(np.sum((np.outer(row, row) - sample) ** 2) for row in centered) / (n**2 * p)
    shrinkage = 0.0 if delta == 0.0 else min(1.0, beta / delta)
    return shrinkage * target + (1.0 - shrinkage) * sample


class MahalanobisDetector:
    def __init__(self, training: np.ndarray):
        self.mean = training.mean(axis=0)
        self.precision = np.linalg.inv(ledoit_wolf_covariance(training))

    def score(self, samples: np.ndarray) -> np.ndarray:
        centered = np.asarray(samples, dtype=float) - self.mean
        return np.sqrt(np.einsum("ij,jk,ik->i", centered, self.precision, centered))


class PCAResidualDetector:
    """Squared prediction error outside the leading principal components.

    The leading components absorb the dominant (mostly environmental)
    co-variation of the frequencies; a change that breaks that pattern shows
    up in the discarded subspace.
    """

    def __init__(self, training: np.ndarray, retained_variance: float = PCA_RETAINED_VARIANCE):
        self.mean = training.mean(axis=0)
        self.scale = training.std(axis=0)
        standardized = (training - self.mean) / self.scale
        _, singular, vt = np.linalg.svd(standardized, full_matrices=False)
        explained = np.cumsum(singular**2) / np.sum(singular**2)
        self.retained = min(int(np.searchsorted(explained, retained_variance) + 1), training.shape[1] - 1)
        self.loadings = vt[: self.retained].T

    def score(self, samples: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(samples, dtype=float) - self.mean) / self.scale
        residual = standardized - standardized @ self.loadings @ self.loadings.T
        return np.sqrt(np.sum(residual**2, axis=1))


def daily_medians(times, scores, min_hours: int = MIN_HOURS_PER_DAY) -> dict[dt.date, float]:
    """Median hourly score per calendar day, skipping days with too few hours."""
    by_day: dict[dt.date, list[float]] = {}
    for stamp, value in zip(times, scores):
        if np.isfinite(value):
            by_day.setdefault(stamp.date(), []).append(float(value))
    return {day: float(np.median(values)) for day, values in sorted(by_day.items()) if len(values) >= min_hours}


def conformal_p_values(reference_scores, scores) -> np.ndarray:
    """Conformal p-value of each score against the reference scores.

    p = (1 + number of reference scores >= s) / (n + 1). If a healthy day's
    score is exchangeable with the n reference days, P(p <= alpha) <= alpha
    for every alpha, so the p-value is the evidence a day gives against
    "still healthy". The smallest attainable value is 1 / (n + 1).
    """
    reference = np.sort(np.asarray(reference_scores, dtype=float))
    values = np.asarray(scores, dtype=float)
    at_or_above = len(reference) - np.searchsorted(reference, values, side="left")
    return (1.0 + at_or_above) / (len(reference) + 1.0)


def check_alpha(reference_days: int, alpha: float) -> None:
    """Refuse a false-alarm rate the reference period is too short to support.

    No day can reach a p-value below 1 / (n + 1), so with fewer than
    1 / alpha - 1 reference days the detector could never alarm. That is a
    flaw in the study design, so it is reported before anything is scored.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    floor = 1.0 / (reference_days + 1.0)
    if alpha < floor:
        needed = math.ceil(1.0 / alpha - 1.0)
        raise ValueError(
            f"alpha={alpha} needs at least {needed} reference days, but the reference period has "
            f"{reference_days}; the smallest attainable p-value is {floor:.4g}"
        )


def conformal_limit(reference_scores, alpha: float) -> float:
    """Highest score that does not alarm: the k-th smallest reference score.

    k = ceil((n + 1)(1 - alpha)). A score above it has p <= alpha, so the
    ratio of a score to this limit shows the margin to alarming.
    """
    reference = np.sort(np.asarray(reference_scores, dtype=float))
    check_alpha(len(reference), alpha)
    k = math.ceil((len(reference) + 1) * (1.0 - alpha) - 1e-9)
    return float(reference[k - 1])


def alarm_summary(daily: dict[dt.date, float], reference_scores, alpha: float, segment: Segment) -> dict:
    """Alarm counts within a segment and the delay to its first alarmed day.

    A day alarms when its conformal p-value against ``reference_scores`` is
    at most ``alpha``. The score-to-limit ratios show the margin: below 1
    stays quiet, above 1 alarms.
    """
    start, end = segment
    days = [day for day in daily if start.date() <= day < end.date()]
    values = np.array([daily[day] for day in days], dtype=float)
    p_values = conformal_p_values(reference_scores, values)
    alarmed = [day for day, p_value in zip(days, p_values) if p_value <= alpha + 1e-12]
    limit = conformal_limit(reference_scores, alpha)
    return {
        "days": len(days),
        "alarmed_days": len(alarmed),
        "alarm_rate": len(alarmed) / len(days) if days else float("nan"),
        "first_alarm": alarmed[0].isoformat() if alarmed else None,
        "delay_days": (alarmed[0] - start.date()).days if alarmed else None,
        "min_p_value": float(p_values.min()) if days else float("nan"),
        "min_over_limit": float(values.min()) / limit if days else float("nan"),
        "max_over_limit": float(values.max()) / limit if days else float("nan"),
    }


def temperature_sensitivity_check(baseline: LinearBaseline, expected_df_over_f: float) -> list[dict]:
    """Relative frequency change per kelvin above and below the hinge, against a material-only estimate.

    Above the hinge the slope is the first coefficient; below it the hinge
    term adds to it. Frequencies should fall as temperature rises, so a
    physically plausible slope is negative. ``expected_df_over_f`` is the
    study's material-only sensitivity, about half of the relative change of
    Young's modulus per kelvin since f scales with sqrt(E).
    """
    rows = []
    for frequency, (above, hinge) in zip(baseline.reference_frequencies_hz, baseline.coefficients):
        above_rel = above / frequency
        rows.append(
            {
                "reference_hz": float(frequency),
                "df_over_f_per_k_above": float(above_rel),
                "df_over_f_per_k_below": float((above + hinge) / frequency),
                "sign_ok": bool(above_rel < 0.0),
                "ratio_to_material_only": float(above_rel / expected_df_over_f),
            }
        )
    return rows


def frequency_ratio_check(reference_hz, changed_hz, relative_tolerance: float = 0.5) -> dict:
    """Is a change in normalized frequencies uniform across modes (global) or not (local)?

    A uniform stiffness or mass change scales every frequency by the same
    factor, so all frequency ratios stay fixed. A local change, such as
    connection strengthening or member damage, moves modes by different
    amounts, even in different directions. The change counts as global when
    every mode moves the same way and the spread of relative shifts is within
    ``relative_tolerance`` of their mean magnitude.
    """
    reference = np.asarray(reference_hz, dtype=float)
    changed = np.asarray(changed_hz, dtype=float)
    shifts = (changed - reference) / reference
    same_sign = bool(np.all(shifts > 0.0) or np.all(shifts < 0.0))
    spread = float(shifts.max() - shifts.min())
    mean_magnitude = float(np.mean(np.abs(shifts)))
    is_global = same_sign and spread <= relative_tolerance * mean_magnitude
    ratios_reference = reference / reference[0]
    ratios_changed = changed / changed[0]
    return {
        "relative_shifts": [float(value) for value in shifts],
        "ratio_changes": [float(value) for value in ratios_changed / ratios_reference - 1.0],
        "same_sign": same_sign,
        "spread": spread,
        "mean_abs_shift": mean_magnitude,
        "pattern": "global" if is_global else "local",
    }


def run_protocol(
    times,
    frequencies_hz: np.ndarray,
    temperature_c: np.ndarray,
    protocol: StudyProtocol,
    *,
    expected_df_over_f: float,
    modes: list[int] | None = None,
    alpha: float = ALPHA,
) -> dict:
    """Run all three detector arms on hourly frequencies and temperature.

    ``modes`` overrides the reference-only mode selection; a study that
    does so must report the result as post hoc. Raises ``ValueError``
    before scoring when the reference period has too few days for ``alpha``.
    """
    reference = segment_mask(times, protocol.reference)
    modes = select_modes(frequencies_hz, reference) if modes is None else list(modes)
    selected = frequencies_hz[:, modes]
    complete = np.all(np.isfinite(selected), axis=1) & np.isfinite(temperature_c)
    covariates = temperature_features(np.where(np.isfinite(temperature_c), temperature_c, 0.0))

    train = reference & complete
    reference_days = daily_medians([stamp for stamp, used in zip(times, train) if used], np.zeros(int(train.sum())))
    check_alpha(len(reference_days), alpha)
    baseline = fit_temperature_baseline(covariates[train], selected[train])
    residuals = baseline.residuals(covariates, np.where(complete[:, None], selected, np.nan))

    detectors = {
        "raw_frequencies": (MahalanobisDetector(selected[train]), selected),
        "temperature_normalized": (MahalanobisDetector(residuals[train]), residuals),
        "pca_residual": (PCAResidualDetector(selected[train]), selected),
    }
    reference_start, reference_end = (stamp.date() for stamp in protocol.reference)
    arms = {}
    for name, (detector, inputs) in detectors.items():
        scores = np.full(len(times), np.nan)
        scores[complete] = detector.score(inputs[complete])
        daily = daily_medians(times, scores)
        reference_scores = [value for day, value in daily.items() if reference_start <= day < reference_end]
        arms[name] = {
            "control_limit": conformal_limit(reference_scores, alpha),
            "reference": alarm_summary(daily, reference_scores, alpha, protocol.reference),
            "holdout": alarm_summary(daily, reference_scores, alpha, protocol.holdout),
            "changed": alarm_summary(daily, reference_scores, alpha, protocol.changed),
            "daily": [
                {"date": day.isoformat(), "score": value, "p_value": float(p_value)}
                for (day, value), p_value in zip(
                    daily.items(), conformal_p_values(reference_scores, list(daily.values()))
                )
            ],
        }
        if isinstance(detector, PCAResidualDetector):
            arms[name]["retained_components"] = detector.retained

    holdout = segment_mask(times, protocol.holdout) & complete
    changed = segment_mask(times, protocol.changed) & complete
    normalized = residuals + baseline.reference_frequencies_hz
    return {
        "alpha": alpha,
        "reference_days": len(reference_days),
        "p_value_floor": 1.0 / (len(reference_days) + 1.0),
        "modes": modes,
        "reference_hz": [float(value) for value in baseline.reference_frequencies_hz],
        "baseline_source": baseline.source,
        "complete_hours": {"reference": int(train.sum()), "holdout": int(holdout.sum()), "changed": int(changed.sum())},
        "temperature_range_c": {
            name: [float(temperature_c[mask].min()), float(temperature_c[mask].max())]
            for name, mask in (("reference", train), ("holdout", holdout), ("changed", changed))
        },
        "arms": arms,
        "temperature_sensitivity": temperature_sensitivity_check(baseline, expected_df_over_f),
        "frequency_ratio": frequency_ratio_check(
            np.median(normalized[train], axis=0), np.median(normalized[changed], axis=0)
        ),
    }


def markdown_table(result: dict, changed_label: str = "Post-change") -> str:
    lines = [
        f"| Arm | Holdout false-alarm days | {changed_label} days alarmed | First alarm (delay) "
        f"| Holdout min p | Holdout max / limit | {changed_label} min / limit |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, arm in result["arms"].items():
        holdout, changed = arm["holdout"], arm["changed"]
        delay = f"{changed['first_alarm']} ({changed['delay_days']} d)" if changed["first_alarm"] else "none"
        lines.append(
            f"| {name} | {holdout['alarmed_days']}/{holdout['days']} ({100 * holdout['alarm_rate']:.1f}%) "
            f"| {changed['alarmed_days']}/{changed['days']} ({100 * changed['alarm_rate']:.1f}%) | {delay} "
            f"| {holdout['min_p_value']:.3f} | {holdout['max_over_limit']:.2f} | {changed['min_over_limit']:.2f} |"
        )
    return "\n".join(lines)
