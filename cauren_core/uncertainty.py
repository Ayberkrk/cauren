from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import AgentSchema
from .quality_control import QualityControlReport

# Data quality and model self-confidence are treated as the two primary,
# roughly-equally-weighted uncertainty drivers; missing required-feature
# coverage in the physics layer is a smaller tie-breaker signal, since the
# quality report's own completeness check already captures most of that
# same gap from the input side.
_WEIGHT_DATA_QUALITY = 0.4
_WEIGHT_MODEL_CONFIDENCE = 0.4
_WEIGHT_EVIDENCE_COVERAGE = 0.2

# At uncertainty_score == 1.0 (maximally uncertain), the reported interval
# spans this much of the [0, 1] risk-score range on each side of the point
# estimate. Kept well under 1.0 so even a maximally uncertain diagnosis
# still reports a bounded, informative band rather than "could be anything".
_MAX_HALF_WIDTH = 0.35


@dataclass(frozen=True)
class UncertaintyFactor:
    name: str
    contribution: float  # share of the final uncertainty_score this factor added
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "contribution": round(float(self.contribution), 6),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class UncertaintyReport:
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    uncertainty_score: float  # 0 = fully trusted, 1 = maximally uncertain
    factors: tuple[UncertaintyFactor, ...]
    method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_estimate": round(float(self.point_estimate), 6),
            "lower_bound": round(float(self.lower_bound), 6),
            "upper_bound": round(float(self.upper_bound), 6),
            "confidence_level": round(float(self.confidence_level), 4),
            "uncertainty_score": round(float(self.uncertainty_score), 6),
            "factors": [factor.to_dict() for factor in self.factors],
            "method": self.method,
        }


def estimate_uncertainty(
    *,
    risk_score: float,
    core_confidence: float,
    quality_report: QualityControlReport,
    schema: AgentSchema,
    missing_features: tuple[str, ...] = (),
    confidence_level: float = 0.90,
) -> UncertaintyReport:
    """Wraps a point risk score with an honest confidence band.

    A single risk_score number invites treating every diagnosis as equally
    trustworthy, whether it came from a clean, complete sensor window or
    one with half its required features missing and a stuck sensor. This
    combines three independent, already-computed uncertainty signals
    (input data quality, the anomaly core's own reported confidence, and
    how much of the physics schema actually had evidence) into one band so
    a downstream reviewer can see when a score deserves less trust,
    without having to separately go dig through the quality-control and
    physics-evidence payloads themselves.
    """
    data_quality_uncertainty = max(0.0, min(1.0, 1.0 - float(quality_report.score)))
    model_uncertainty = max(0.0, min(1.0, 1.0 - float(core_confidence)))

    required = schema.required_features or ()
    missing_required = [name for name in missing_features if name in required]
    evidence_gap = len(missing_required) / float(len(required)) if required else 0.0

    uncertainty_score = max(
        0.0,
        min(
            1.0,
            _WEIGHT_DATA_QUALITY * data_quality_uncertainty
            + _WEIGHT_MODEL_CONFIDENCE * model_uncertainty
            + _WEIGHT_EVIDENCE_COVERAGE * evidence_gap,
        ),
    )

    half_width = _MAX_HALF_WIDTH * uncertainty_score
    lower = max(0.0, float(risk_score) - half_width)
    upper = min(1.0, float(risk_score) + half_width)

    factors = (
        UncertaintyFactor(
            name="data_quality",
            contribution=_WEIGHT_DATA_QUALITY * data_quality_uncertainty,
            detail=f"Quality control score was {quality_report.score:.2f} (status={quality_report.status}).",
        ),
        UncertaintyFactor(
            name="model_confidence",
            contribution=_WEIGHT_MODEL_CONFIDENCE * model_uncertainty,
            detail=f"Anomaly core reported {float(core_confidence):.2f} confidence in its own scoring.",
        ),
        UncertaintyFactor(
            name="evidence_coverage",
            contribution=_WEIGHT_EVIDENCE_COVERAGE * evidence_gap,
            detail=(
                f"{len(missing_required)} of {len(required)} required feature(s) had no physics evidence."
                if required
                else "Schema declares no required features."
            ),
        ),
    )

    return UncertaintyReport(
        point_estimate=float(risk_score),
        lower_bound=float(lower),
        upper_bound=float(upper),
        confidence_level=float(confidence_level),
        uncertainty_score=float(round(uncertainty_score, 6)),
        factors=factors,
        method="weighted_quality_confidence_evidence_band",
    )
