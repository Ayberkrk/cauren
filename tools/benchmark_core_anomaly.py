"""Synthetic alarm-rate benchmark for the statistical anomaly core.

Every scenario builds a 16-step window over the eight required civil
features (cauren_agents/civil/agent.py), draws ordinary noise for each
feature around a per-asset reference (center, scale), and then applies
at most ONE controlled change to ONE feature. Magnitudes are expressed in
units of that feature's reference scale ("sigma"), so a 3 sigma step means
the same thing for a 0-1 ratio and a 0-100 percentage.

The benchmark measures how often the full pipeline raises an alarm,
defined as diagnosis.risk_score >= the civil decision threshold (0.34),
through CaurenPipeline.from_default_registry().diagnose(...). It reports
each rate with a Wilson 95% interval so a small run is not mistaken for a
precise one.

The reference is sent in the payload under "baseline". A core that does
not read it simply ignores the key, which is how the "before" rates in
docs/core-anomaly-benchmark.md were recorded.

Usage:
    python3 tools/benchmark_core_anomaly.py
    python3 tools/benchmark_core_anomaly.py --trials 200 --seed 7 --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cauren_core import CaurenPipeline  # noqa: E402
from cauren_core.risk_calibration import sector_threshold  # noqa: E402

AGENT_ID = "cauren-civil"
SEQ_LEN = 16
DEFAULT_TRIALS = 500
DEFAULT_SEED = 20261025
REFERENCE_N = 120


@dataclass(frozen=True)
class FeatureReference:
    center: float
    scale: float
    unit: str
    expected_variation: bool = True


# Plausible per-asset operating points for the civil schema. Only the
# expected_variation flag of permit_status_score is special: a permit
# status legitimately stays put for long stretches, so a constant value
# there is not a sensor fault.
REFERENCE: dict[str, FeatureReference] = {
    "structural_risk_score": FeatureReference(0.40, 0.03, "ratio"),
    "inspection_finding_score": FeatureReference(0.30, 0.04, "ratio"),
    "permit_status_score": FeatureReference(0.85, 0.02, "ratio", expected_variation=False),
    "construction_progress_pct": FeatureReference(55.0, 1.5, "%"),
    "infrastructure_connection_score": FeatureReference(0.60, 0.03, "ratio"),
    "natural_hazard_score": FeatureReference(0.25, 0.02, "ratio"),
    "occupancy_safety_score": FeatureReference(0.70, 0.03, "ratio"),
    "ground_stability_score": FeatureReference(0.50, 0.03, "ratio"),
}
FEATURES = tuple(REFERENCE)

# The feature that carries the injected change, unless a scenario says
# otherwise.
TARGET_FEATURE = "structural_risk_score"
NATURALLY_CONSTANT_FEATURE = "permit_status_score"


def reference_payload() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "center": ref.center,
            "scale": ref.scale,
            "n": REFERENCE_N,
            "expected_variation": ref.expected_variation,
            "unit": ref.unit,
        }
        for name, ref in REFERENCE.items()
    }


def _noise(rng: np.random.Generator) -> np.ndarray:
    """(SEQ_LEN, n_features) matrix of reference-level noise."""
    centers = np.array([REFERENCE[name].center for name in FEATURES])
    scales = np.array([REFERENCE[name].scale for name in FEATURES])
    return centers + scales * rng.standard_normal((SEQ_LEN, len(FEATURES)))


def _column(name: str) -> int:
    return FEATURES.index(name)


def _clean(rng: np.random.Generator) -> np.ndarray:
    return _noise(rng)


def _constant(feature: str) -> Callable[[np.random.Generator], np.ndarray]:
    def build(rng: np.random.Generator) -> np.ndarray:
        matrix = _noise(rng)
        matrix[:, _column(feature)] = REFERENCE[feature].center
        return matrix

    return build


def _all_constant(rng: np.random.Generator) -> np.ndarray:
    """A frozen feed: every feature repeats its reference value."""
    matrix = _noise(rng)
    for name in FEATURES:
        matrix[:, _column(name)] = REFERENCE[name].center
    return matrix


def _drift(sigma: float) -> Callable[[np.random.Generator], np.ndarray]:
    """Linear drift that starts at the reference level and ends `sigma`
    reference scales away from it on the last step."""

    def build(rng: np.random.Generator) -> np.ndarray:
        matrix = _noise(rng)
        ramp = np.linspace(0.0, sigma, SEQ_LEN) * REFERENCE[TARGET_FEATURE].scale
        matrix[:, _column(TARGET_FEATURE)] += ramp
        return matrix

    return build


def _step(sigma: float) -> Callable[[np.random.Generator], np.ndarray]:
    """Level change of `sigma` reference scales over the second half."""

    def build(rng: np.random.Generator) -> np.ndarray:
        matrix = _noise(rng)
        matrix[SEQ_LEN // 2 :, _column(TARGET_FEATURE)] += sigma * REFERENCE[TARGET_FEATURE].scale
        return matrix

    return build


def _spike(sigma: float) -> Callable[[np.random.Generator], np.ndarray]:
    """One sample, at a random step, `sigma` reference scales high."""

    def build(rng: np.random.Generator) -> np.ndarray:
        matrix = _noise(rng)
        row = int(rng.integers(0, SEQ_LEN))
        matrix[row, _column(TARGET_FEATURE)] += sigma * REFERENCE[TARGET_FEATURE].scale
        return matrix

    return build


# name -> (window builder, whether an alarm is the desired outcome)
SCENARIOS: dict[str, tuple[Callable[[np.random.Generator], np.ndarray], bool]] = {
    "clean_noise": (_clean, False),
    "constant_expected_variation": (_constant(TARGET_FEATURE), True),
    "constant_naturally_constant": (_constant(NATURALLY_CONSTANT_FEATURE), False),
    "constant_all_features": (_all_constant, True),
    "drift_3sigma": (_drift(3.0), True),
    "step_3sigma": (_step(3.0), True),
    "spike_6sigma": (_spike(6.0), True),
    "drift_1sigma": (_drift(1.0), True),
    "step_1sigma": (_step(1.0), True),
    "spike_1sigma": (_spike(1.0), True),
}


def build_payload(matrix: np.ndarray, *, include_reference: bool = True) -> dict[str, Any]:
    sensors = [
        {
            "sensor_id": f"{name}-{step}",
            "name": name,
            "unit": REFERENCE[name].unit,
            "value": float(matrix[step, col]),
            "timestamp": float(step),
        }
        for step in range(matrix.shape[0])
        for col, name in enumerate(FEATURES)
    ]
    payload: dict[str, Any] = {"agent_id": AGENT_ID, "seq_len": SEQ_LEN, "sensors": sensors}
    if include_reference:
        payload["baseline"] = reference_payload()
    return payload


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def run_scenario(
    name: str,
    *,
    trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
    include_reference: bool = True,
    pipeline: CaurenPipeline | None = None,
) -> dict[str, Any]:
    builder, alarm_expected = SCENARIOS[name]
    pipeline = pipeline or CaurenPipeline.from_default_registry()
    threshold = sector_threshold(AGENT_ID)
    # Each scenario gets its own stream derived from the run seed, so adding
    # or reordering scenarios never changes another scenario's draws.
    rng = np.random.default_rng([seed, zlib.crc32(name.encode("utf-8"))])
    alarms = 0
    anomaly_types: dict[str, int] = {}
    for _ in range(trials):
        diagnosis = pipeline.diagnose(build_payload(builder(rng), include_reference=include_reference))
        if diagnosis.risk_score >= threshold:
            alarms += 1
        anomaly_types[diagnosis.anomaly_type] = anomaly_types.get(diagnosis.anomaly_type, 0) + 1
    low, high = wilson_interval(alarms, trials)
    return {
        "scenario": name,
        "alarm_expected": alarm_expected,
        "trials": trials,
        "alarms": alarms,
        "alarm_rate": alarms / trials if trials else 0.0,
        "wilson_low": low,
        "wilson_high": high,
        "anomaly_types": dict(sorted(anomaly_types.items(), key=lambda item: -item[1])),
    }


def run_benchmark(
    *,
    trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
    scenarios: tuple[str, ...] | None = None,
    include_reference: bool = True,
) -> list[dict[str, Any]]:
    pipeline = CaurenPipeline.from_default_registry()
    return [
        run_scenario(name, trials=trials, seed=seed, include_reference=include_reference, pipeline=pipeline)
        for name in (scenarios or tuple(SCENARIOS))
    ]


def format_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| Scenario | Alarm expected | Alarm rate | Wilson 95% | Most common anomaly_type |",
        "|---|---|---|---|---|",
    ]
    for row in results:
        top_type = next(iter(row["anomaly_types"]), "")
        lines.append(
            f"| {row['scenario']} | {'yes' if row['alarm_expected'] else 'no'} "
            f"| {100.0 * row['alarm_rate']:.1f}% "
            f"| {100.0 * row['wilson_low']:.1f}% to {100.0 * row['wilson_high']:.1f}% "
            f"| {top_type} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--scenario", action="append", choices=sorted(SCENARIOS), help="Repeatable; default runs all.")
    parser.add_argument("--no-reference", action="store_true", help="Omit the per-asset baseline from every payload.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a Markdown table.")
    args = parser.parse_args(argv)
    results = run_benchmark(
        trials=args.trials,
        seed=args.seed,
        scenarios=tuple(args.scenario) if args.scenario else None,
        include_reference=not args.no_reference,
    )
    print(json.dumps(results, indent=2) if args.json else format_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
