from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SectorRiskProfile:
    agent_id: str
    decision_threshold: float
    core_weight: float
    physics_weight: float
    compression_start: float
    compression_factor: float
    transient_min_scale: float
    transient_support_weight: float
    physics_confirmation_floor: float
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
        transient_min_scale=0.58,
        transient_support_weight=0.58,
        physics_confirmation_floor=0.24,
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


def transitional_support(pattern_scores: Mapping[str, float], *, physics_risk: float = 0.0, include_physics: bool = True) -> float:
    support = max(
        float(pattern_scores.get("collective_anomaly", 0.0)),
        float(pattern_scores.get("multisensor_residual", 0.0)),
        float(pattern_scores.get("correlation_break", 0.0)),
        float(pattern_scores.get("repeating_burst", 0.0)),
        float(pattern_scores.get("level_shift", 0.0)),
        float(pattern_scores.get("variance_change", 0.0)),
        float(pattern_scores.get("drift", 0.0)),
        float(pattern_scores.get("slow_degradation", 0.0)),
        float(pattern_scores.get("contextual_anomaly", 0.0)),
        float(pattern_scores.get("seasonal_deviation", 0.0)),
    )
    if include_physics:
        support = max(support, float(physics_risk))
    return max(0.0, min(1.0, support))


def calibrated_core_risk(*, agent_id: str | None, raw_scores: Mapping[str, float], anomaly_type: str, reconstruction_error: float) -> tuple[float, float]:
    profile = risk_profile(agent_id)
    weighted = {
        key: max(0.0, min(1.0, float(value) * float(profile.pattern_weights.get(key, 1.0))))
        for key, value in raw_scores.items()
    }
    top_score = max(weighted.values(), default=0.0)
    support = transitional_support(weighted, include_physics=False)
    residual_score = max(0.0, min(1.0, float(reconstruction_error) / (float(reconstruction_error) + 1.0)))
    raw_risk = 0.72 * top_score + 0.18 * max(residual_score, support * 0.8) + 0.10 * float(weighted.get("collective_anomaly", 0.0))
    if anomaly_type in {"spike", "dip"}:
        transient_energy = max(float(weighted.get("spike", 0.0)), float(weighted.get("dip", 0.0)), float(weighted.get("repeating_burst", 0.0)))
        scale = profile.transient_min_scale + profile.transient_support_weight * max(support, transient_energy * 0.5)
        raw_risk *= max(0.0, min(1.0, scale))
    elif anomaly_type == "calibration_bias":
        raw_risk = min(1.0, raw_risk * 1.10 + 0.04 * float(weighted.get("calibration_bias", 0.0)))
    calibrated = compress_risk(raw_risk, start=profile.compression_start, factor=profile.compression_factor)
    return max(0.0, min(1.0, raw_risk)), max(0.0, min(1.0, calibrated))


def calibrated_combined_risk(*, agent_id: str | None, core_risk: float, physics_risk: float, anomaly_type: str, pattern_scores: Mapping[str, float]) -> tuple[float, float]:
    profile = risk_profile(agent_id)
    raw_combined = profile.core_weight * float(core_risk) + profile.physics_weight * float(physics_risk)
    if anomaly_type in {"spike", "dip"}:
        support = transitional_support(pattern_scores, physics_risk=physics_risk, include_physics=True)
        scale = profile.transient_min_scale + profile.transient_support_weight * support
        raw_combined *= max(0.0, min(1.0, scale))
    calibrated = compress_risk(raw_combined, start=profile.compression_start, factor=profile.compression_factor)
    return max(0.0, min(1.0, raw_combined)), max(0.0, min(1.0, calibrated))
