from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class SectorRiskProfile:
    agent_id: str
    decision_threshold: float
    core_weight: float
    physics_weight: float
    compression_start: float
    compression_factor: float
    pattern_weights: Mapping[str, float]


_DEFAULT_PATTERN_WEIGHTS = {
    "drift": 1.0,
    "spike": 0.84,
    "dip": 0.78,
    "oscillation": 0.9,
    "level_shift": 0.96,
    "variance_change": 0.92,
    "flatline": 0.96,
    "sensor_dropout": 0.97,
    "correlation_break": 1.0,
    "repeating_burst": 0.9,
    "collective_anomaly": 1.0,
    "multisensor_residual": 0.95,
    "calibration_bias": 0.88,
    "slow_degradation": 1.0,
    "contextual_anomaly": 0.97,
    "seasonal_deviation": 0.94,
}


SECTOR_RISK_PROFILES: dict[str, SectorRiskProfile] = {
    "cauren-civil": SectorRiskProfile(
        agent_id="cauren-civil",
        decision_threshold=0.34,
        core_weight=0.52,
        physics_weight=0.48,
        compression_start=0.64,
        compression_factor=0.84,
        pattern_weights={
            **_DEFAULT_PATTERN_WEIGHTS,
            "drift": 1.08,
            "spike": 0.9,
            "correlation_break": 1.04,
            "multisensor_residual": 1.08,
            "slow_degradation": 1.08,
            "contextual_anomaly": 1.04,
        },
    ),
}


def risk_profile(agent_id: str | None) -> SectorRiskProfile:
    if agent_id and agent_id in SECTOR_RISK_PROFILES:
        return SECTOR_RISK_PROFILES[agent_id]
    return SECTOR_RISK_PROFILES["cauren-civil"]


def sector_threshold(agent_id: str | None) -> float:
    return float(risk_profile(agent_id).decision_threshold)


def compress_risk(value: float, *, start: float, factor: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    if value <= start:
        return value
    return min(1.0, start + (value - start) * factor)


def calibrated_combined_risk(*, agent_id: str | None, core_risk: float, physics_risk: float, anomaly_type: str, pattern_scores: Mapping[str, float]) -> tuple[float, float]:
    """Combine core and physics risk for the final diagnosis.

    The core risk is already a calibrated statement about significance
    (see p_value_to_risk), so absent or zero physics evidence must not
    dilute it below the threshold it was mapped to: the weighted blend can
    only add to the core risk, never subtract from it. Physics stays
    fail-closed upstream (cauren_physics.common) until a rule is validated.
    """
    profile = risk_profile(agent_id)
    core_risk = max(0.0, min(1.0, float(core_risk)))
    blended = profile.core_weight * core_risk + profile.physics_weight * float(physics_risk)
    raw_combined = max(core_risk, blended)
    calibrated = compress_risk(raw_combined, start=profile.compression_start, factor=profile.compression_factor)
    return max(0.0, min(1.0, raw_combined)), max(0.0, min(1.0, calibrated))


# ---------------------------------------------------------------------------
# Reference-based evidence: tail p-values, Holm correction, risk mapping
# ---------------------------------------------------------------------------

# Family-wise error rate for one window: all per-feature, per-pattern tests
# of a diagnosis are corrected together with Holm's step-down procedure.
HOLM_ALPHA = 0.05

_SQRT2 = math.sqrt(2.0)


def normal_two_sided_p(z: float) -> float:
    return float(math.erfc(abs(float(z)) / _SQRT2))


def _incomplete_beta_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return float(math.exp(log_front) * _incomplete_beta_fraction(a, b, x) / a)
    return float(1.0 - math.exp(log_front) * _incomplete_beta_fraction(b, a, 1.0 - x) / b)


def student_t_two_sided_p(t: float, dof: float) -> float:
    """Two-sided tail of Student's t; the normal tail when dof is infinite."""
    if not math.isfinite(dof):
        return normal_two_sided_p(t)
    t = float(t)
    return float(min(1.0, _regularized_incomplete_beta(0.5 * dof, 0.5, dof / (dof + t * t))))


def max_abs_t_p(t: float, n: int, dof: float = math.inf) -> float:
    """P(max of n independent |T| >= t), i.e. Sidak over n samples."""
    single = student_t_two_sided_p(t, dof)
    if single >= 1.0:
        return 1.0
    return float(min(1.0, -math.expm1(max(1, int(n)) * math.log1p(-single))))


def max_abs_normal_p(z: float, n: int) -> float:
    """P(max of n independent |N(0,1)| >= z), i.e. Sidak over n samples."""
    single = normal_two_sided_p(z)
    if single >= 1.0:
        return 1.0
    return float(min(1.0, -math.expm1(max(1, int(n)) * math.log1p(-single))))


def chi2_lower_p(x: float, dof: int) -> float:
    """P(chi-square with `dof` degrees of freedom <= x).

    Regularized lower incomplete gamma P(dof/2, x/2) by its power series,
    which converges quickly in the small-x lower tail this is used for.
    """
    if x <= 0.0:
        return 0.0
    a = 0.5 * float(dof)
    half_x = 0.5 * float(x)
    term = 1.0 / a
    total = term
    for k in range(1, 500):
        term *= half_x / (a + k)
        total += term
        if term < total * 1e-15:
            break
    log_p = a * math.log(half_x) - half_x - math.lgamma(a) + math.log(total)
    return float(min(1.0, math.exp(log_p)))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda idx: p_values[idx])
    adjusted = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * float(p_values[idx])))
        adjusted[idx] = running
    return adjusted


def p_value_to_risk(p_value: float, *, agent_id: str | None = None, alpha: float = HOLM_ALPHA) -> float:
    """Map an adjusted p-value to [0, 1] so the sector threshold keeps its meaning.

    risk = 1 - (1 - threshold) ** (ln p / ln alpha): exactly the decision
    threshold (0.34 for civil) at p = alpha, 0 at p = 1, and approaching 1
    as the evidence grows (p = alpha**2 gives about 0.56, alpha**3 about 0.71).
    """
    p_value = float(p_value)
    if p_value >= 1.0:
        return 0.0
    if p_value <= 0.0:
        return 1.0
    threshold = sector_threshold(agent_id)
    exponent = math.log(p_value) / math.log(alpha)
    return float(max(0.0, min(1.0, 1.0 - (1.0 - threshold) ** exponent)))
