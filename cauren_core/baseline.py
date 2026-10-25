"""Per-asset feature baselines (reference center and scale).

A baseline describes how one feature of one asset behaves when nothing is
wrong: a robust center, a robust scale, how many reference samples they
came from, and whether the feature is expected to vary at all. The core
scores a window against it, so a change is measured in the asset's own
units of normal variation instead of the window's own spread.

Baselines are passed per request (or loaded per request from the API's
history store) and never cached on CaurenPipeline or CaurenCoreRuntime,
which are shared across concurrent requests.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

MIN_REFERENCE_SAMPLES = 30
# Scales MAD to the standard deviation of a normal distribution.
MAD_TO_SIGMA = 1.4826
# Scales the mean absolute deviation to the standard deviation of a normal.
MEAN_ABS_TO_SIGMA = math.sqrt(math.pi / 2.0)

REFERENCE_SOURCES = ("payload", "stored", "none")


@dataclass(frozen=True)
class FeatureBaseline:
    center: float
    scale: float
    n: int
    expected_variation: bool = True
    unit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def parse_feature_baseline(name: str, raw: Any) -> FeatureBaseline:
    if isinstance(raw, FeatureBaseline):
        raw = raw.to_dict()
    if not isinstance(raw, Mapping):
        raise ValueError(f"baseline.{name} must be an object")
    center = _finite(raw.get("center"), f"baseline.{name}.center")
    scale = _finite(raw.get("scale"), f"baseline.{name}.scale")
    if scale <= 0.0:
        raise ValueError(f"baseline.{name}.scale must be > 0")
    n_value = _finite(raw.get("n"), f"baseline.{name}.n")
    if n_value != int(n_value) or int(n_value) < MIN_REFERENCE_SAMPLES:
        raise ValueError(f"baseline.{name}.n must be an integer >= {MIN_REFERENCE_SAMPLES}")
    expected_variation = raw.get("expected_variation", True)
    if not isinstance(expected_variation, bool):
        raise ValueError(f"baseline.{name}.expected_variation must be a boolean")
    unit = raw.get("unit", "")
    if unit is None:
        unit = ""
    if not isinstance(unit, str):
        raise ValueError(f"baseline.{name}.unit must be a string")
    return FeatureBaseline(
        center=center,
        scale=scale,
        n=int(n_value),
        expected_variation=expected_variation,
        unit=unit,
    )


def parse_baselines(raw: Any) -> dict[str, FeatureBaseline]:
    """Validate a {feature: {center, scale, n, expected_variation, unit}} map.

    Raises ValueError naming the first offending field.
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("baseline must be an object keyed by feature name")
    return {str(name): parse_feature_baseline(str(name), value) for name, value in raw.items()}


def fit_feature_baseline(
    values: Iterable[float],
    *,
    unit: str = "",
    expected_variation: bool | None = None,
    min_scale: float | None = None,
) -> FeatureBaseline:
    """Fit a robust center (median) and scale (MAD * 1.4826).

    When more than half the reference samples are identical the MAD is
    zero; the mean absolute deviation is used instead. A reference with no
    variation at all has no scale to measure against, so the caller must
    supply `min_scale` (for example the measurement resolution) and the
    feature defaults to expected_variation=False.
    """
    observed = [_finite(value, "reference value") for value in values]
    if len(observed) < MIN_REFERENCE_SAMPLES:
        raise ValueError(
            f"a reference window needs at least {MIN_REFERENCE_SAMPLES} samples per feature, got {len(observed)}"
        )
    center = float(statistics.median(observed))
    deviations = [abs(value - center) for value in observed]
    scale = MAD_TO_SIGMA * float(statistics.median(deviations))
    if scale <= 0.0:
        scale = MEAN_ABS_TO_SIGMA * (sum(deviations) / float(len(deviations)))
    constant_reference = scale <= 0.0
    if min_scale is not None:
        floor = _finite(min_scale, "min_scale")
        if floor <= 0.0:
            raise ValueError("min_scale must be > 0")
        scale = max(scale, floor)
    if scale <= 0.0:
        raise ValueError("the reference window is constant; supply min_scale (e.g. the measurement resolution)")
    if expected_variation is None:
        expected_variation = not constant_reference
    return FeatureBaseline(
        center=center,
        scale=float(scale),
        n=len(observed),
        expected_variation=bool(expected_variation),
        unit=unit,
    )


def reference_summary(source: str, baselines: Mapping[str, FeatureBaseline], feature_names: Iterable[str]) -> dict:
    """What the diagnosis was scored against, for the response body."""
    names = list(feature_names)
    covered = [name for name in names if name in baselines]
    return {
        "source": source if covered else "none",
        "features_with_reference": covered,
        "features_without_reference": [name for name in names if name not in baselines],
        "ignored_reference_features": sorted(name for name in baselines if name not in names),
    }
