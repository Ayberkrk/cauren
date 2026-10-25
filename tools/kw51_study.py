"""KW51 railway bridge case study: environmental normalization before novelty detection.

Uses the hourly identified natural frequencies and deck temperature published
by Maes and Lombaert (KU Leuven) for steel bowstring railway bridge KW51 in
Leuven, Belgium (Zenodo record 3745914, CC BY-NC-SA 4.0). The diagonal
connections were strengthened between 2019-05-15 and 2019-09-27, so the
structure after the retrofit is a known, real change of state.

Three detectors share one protocol: fit on the reference period, set the
control limit at the 99th percentile of the reference daily scores, then
count alarmed days in a later healthy holdout (false alarms) and after the
retrofit (detections). This is a single-bridge case study, not a validation
of Cauren's anomaly detection in general.

The data is not bundled. ``--download`` fetches ``trackedmodes.zip`` (about
13 MB) into ``data/kw51/raw/``, which is gitignored. Needs numpy and scipy
(``pip install -e '.[dataset]'``). Timoshenko's ``fit_environmental_baseline``
is used for the temperature fit when it is installed; otherwise an identical
ordinary least-squares fit runs here.

Usage:
    python3 tools/kw51_study.py --download
    python3 tools/kw51_study.py --json-out runtime_outputs/kw51_study.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "kw51" / "raw"

ZENODO_RECORD = "3745914"
ZENODO_DOI = "10.5281/zenodo.3745914"
DATA_LICENSE = "CC BY-NC-SA 4.0"
TRACKED_MODES_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/trackedmodes.zip/content"
TRACKED_MODES_MD5 = "5cdb446519d1150228fab51b3a002239"

# Segment boundaries are half-open [start, end) in local hours as published.
REFERENCE = (dt.datetime(2018, 10, 1), dt.datetime(2019, 4, 1))
HOLDOUT = (dt.datetime(2019, 4, 1), dt.datetime(2019, 5, 15))
RETROFIT = (dt.datetime(2019, 5, 15), dt.datetime(2019, 9, 28))
CHANGED = (dt.datetime(2019, 9, 28), dt.datetime(2020, 1, 16))

# Mode selection only looks at the reference period so the post-retrofit
# data cannot influence which modes are monitored.
MAX_REFERENCE_MISSING = 0.10
HINGE_TEMPERATURE_C = 0.0
CONTROL_PERCENTILE = 99.0
MIN_HOURS_PER_DAY = 6
PCA_RETAINED_VARIANCE = 0.90

# Structural steel loses roughly 0.03 percent of its Young's modulus per
# kelvin near room temperature. With f proportional to sqrt(E), the
# material-only frequency sensitivity is about half of that.
STEEL_DE_OVER_E_PER_K = -3.0e-4
EXPECTED_DF_OVER_F_PER_K = 0.5 * STEEL_DE_OVER_E_PER_K


def download_tracked_modes(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Download and unpack ``trackedmodes.mat``, verifying the published MD5."""
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "trackedmodes.zip"
    if not archive.exists():
        urllib.request.urlretrieve(TRACKED_MODES_URL, archive)
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    if digest != TRACKED_MODES_MD5:
        raise ValueError(f"{archive} has MD5 {digest}, expected {TRACKED_MODES_MD5}")
    with zipfile.ZipFile(archive) as bundle:
        member = next(name for name in bundle.namelist() if name.endswith("trackedmodes.mat"))
        target = data_dir / "trackedmodes.mat"
        target.write_bytes(bundle.read(member))
    return target


def datenum_to_datetime(datenums) -> list[dt.datetime]:
    """Convert MATLAB datenums (day 1 is 0000-01-01) to rounded hourly datetimes."""
    out = []
    for value in np.asarray(datenums, dtype=float):
        stamp = dt.datetime.fromordinal(int(value)) + dt.timedelta(days=float(value) % 1.0)
        stamp -= dt.timedelta(days=366)
        out.append((stamp + dt.timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0))
    return out


def load_tracked_modes(path: Path) -> dict:
    """Read hourly frequencies (Hz) and deck steel temperature (C) from the MAT file."""
    from scipy.io import loadmat

    modes = loadmat(path, squeeze_me=True, struct_as_record=False)["modes"]
    env_labels = [str(label) for label in modes.labels_env]
    return {
        "times": datenum_to_datetime(modes.sdn),
        "frequencies_hz": np.asarray(modes.f, dtype=float),
        "deck_temperature_c": np.asarray(modes.env[:, env_labels.index("tBD31A")], dtype=float),
    }


def segment_mask(times, segment: tuple[dt.datetime, dt.datetime]) -> np.ndarray:
    start, end = segment
    return np.array([start <= stamp < end for stamp in times], dtype=bool)


def select_modes(frequencies_hz: np.ndarray, reference: np.ndarray, max_missing: float = MAX_REFERENCE_MISSING):
    """Indices of modes identified in at least ``1 - max_missing`` of reference hours."""
    missing = np.isnan(frequencies_hz[reference]).mean(axis=0)
    return [int(index) for index in np.flatnonzero(missing <= max_missing)]


def temperature_features(temperature_c, hinge_c: float = HINGE_TEMPERATURE_C) -> np.ndarray:
    """Bilinear temperature covariates ``[T, min(T - hinge, 0)]``.

    The second column lets the slope change below the hinge, where frozen
    ballast and bearings stiffen the deck far more than the steel modulus
    alone would.
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


def control_limit(reference_daily: dict[dt.date, float], percentile: float = CONTROL_PERCENTILE) -> float:
    return float(np.percentile(list(reference_daily.values()), percentile))


def alarm_summary(daily: dict[dt.date, float], limit: float, segment: tuple[dt.datetime, dt.datetime]) -> dict:
    """Alarm counts within a segment and the delay to its first alarmed day."""
    start, end = segment
    days = [day for day in daily if start.date() <= day < end.date()]
    alarmed = [day for day in days if daily[day] > limit]
    values = [daily[day] for day in days]
    return {
        "days": len(days),
        "alarmed_days": len(alarmed),
        "alarm_rate": len(alarmed) / len(days) if days else float("nan"),
        "first_alarm": alarmed[0].isoformat() if alarmed else None,
        "delay_days": (alarmed[0] - start.date()).days if alarmed else None,
        "min_over_limit": min(values) / limit if values else float("nan"),
        "max_over_limit": max(values) / limit if values else float("nan"),
    }


def temperature_sensitivity_check(
    baseline: LinearBaseline, expected_df_over_f: float = EXPECTED_DF_OVER_F_PER_K
) -> list[dict]:
    """Relative frequency change per kelvin above and below the hinge, against steel softening.

    Above the hinge the slope is the first coefficient; below it the hinge
    term adds to it. Frequencies should fall as temperature rises, so a
    physically plausible slope is negative.
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
    factor, so all frequency ratios stay fixed. Connection strengthening is
    local and moves modes by different amounts, even in different directions.
    The change counts as global when every mode moves the same way and the
    spread of relative shifts is within ``relative_tolerance`` of their mean
    magnitude.
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


def run_study(data: dict, modes: list[int] | None = None) -> dict:
    """Run all three arms. ``modes`` overrides the reference-only mode selection."""
    times = data["times"]
    frequencies = data["frequencies_hz"]
    temperature = data["deck_temperature_c"]
    reference = segment_mask(times, REFERENCE)
    modes = select_modes(frequencies, reference) if modes is None else list(modes)
    selected = frequencies[:, modes]
    complete = np.all(np.isfinite(selected), axis=1) & np.isfinite(temperature)
    covariates = temperature_features(np.where(np.isfinite(temperature), temperature, 0.0))

    train = reference & complete
    baseline = fit_temperature_baseline(covariates[train], selected[train])
    residuals = baseline.residuals(covariates, np.where(complete[:, None], selected, np.nan))

    detectors = {
        "raw_frequencies": (MahalanobisDetector(selected[train]), selected),
        "temperature_normalized": (MahalanobisDetector(residuals[train]), residuals),
        "pca_residual": (PCAResidualDetector(selected[train]), selected),
    }
    arms = {}
    for name, (detector, inputs) in detectors.items():
        scores = np.full(len(times), np.nan)
        scores[complete] = detector.score(inputs[complete])
        daily = daily_medians(times, scores)
        reference_daily = {
            day: value for day, value in daily.items() if REFERENCE[0].date() <= day < REFERENCE[1].date()
        }
        limit = control_limit(reference_daily)
        arms[name] = {
            "control_limit": limit,
            "reference": alarm_summary(daily, limit, REFERENCE),
            "holdout": alarm_summary(daily, limit, HOLDOUT),
            "changed": alarm_summary(daily, limit, CHANGED),
        }
        if isinstance(detector, PCAResidualDetector):
            arms[name]["retained_components"] = detector.retained

    holdout = segment_mask(times, HOLDOUT) & complete
    changed = segment_mask(times, CHANGED) & complete
    normalized = residuals + baseline.reference_frequencies_hz
    return {
        "source": {"zenodo_doi": ZENODO_DOI, "license": DATA_LICENSE},
        "modes": modes,
        "reference_hz": [float(value) for value in baseline.reference_frequencies_hz],
        "baseline_source": baseline.source,
        "complete_hours": {"reference": int(train.sum()), "holdout": int(holdout.sum()), "changed": int(changed.sum())},
        "deck_temperature_range_c": {
            name: [float(temperature[mask].min()), float(temperature[mask].max())]
            for name, mask in (("reference", train), ("holdout", holdout), ("changed", changed))
        },
        "arms": arms,
        "temperature_sensitivity": temperature_sensitivity_check(baseline),
        "frequency_ratio": frequency_ratio_check(
            np.median(normalized[train], axis=0), np.median(normalized[changed], axis=0)
        ),
    }


def markdown_table(result: dict) -> str:
    lines = [
        "| Arm | Holdout false-alarm days | Post-retrofit days alarmed | First alarm (delay) "
        "| Holdout max / limit | Post-retrofit min / limit |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, arm in result["arms"].items():
        holdout, changed = arm["holdout"], arm["changed"]
        delay = f"{changed['first_alarm']} ({changed['delay_days']} d)" if changed["first_alarm"] else "none"
        lines.append(
            f"| {name} | {holdout['alarmed_days']}/{holdout['days']} ({100 * holdout['alarm_rate']:.1f}%) "
            f"| {changed['alarmed_days']}/{changed['days']} ({100 * changed['alarm_rate']:.1f}%) | {delay} "
            f"| {holdout['max_over_limit']:.2f} | {changed['min_over_limit']:.2f} |"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--download", action="store_true", help="fetch trackedmodes.zip from Zenodo first")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--modes",
        type=int,
        nargs="+",
        help="0-based mode indices to monitor instead of the reference-only selection",
    )
    args = parser.parse_args(argv)

    mat_path = args.data_dir / "trackedmodes.mat"
    if args.download:
        mat_path = download_tracked_modes(args.data_dir)
    if not mat_path.exists():
        print(f"{mat_path} not found; rerun with --download", file=sys.stderr)
        return 2

    result = run_study(load_tracked_modes(mat_path), modes=args.modes)
    print(markdown_table(result))
    summary_keys = ("modes", "baseline_source", "complete_hours", "deck_temperature_range_c", "frequency_ratio")
    print(json.dumps({key: result[key] for key in summary_keys}, indent=2))
    for row in result["temperature_sensitivity"]:
        print(
            f"{row['reference_hz']:.3f} Hz: df/f per K {row['df_over_f_per_k_above']:.2e} above hinge, "
            f"{row['df_over_f_per_k_below']:.2e} below, {row['ratio_to_material_only']:.1f}x material-only"
        )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
