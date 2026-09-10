from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import AgentSchema, NormalizationTrace, SensorWindow

# Generic unit-based sanity ranges. Deliberately not a per-feature physical
# limits table (that duplication is scoped to the civil dataset pipeline, see
# tools/audit_cauren_data_quality.py and the "key invariant" note in
# CLAUDE.md) so this check stays sector-agnostic and works for any agent
# schema built on top of cauren_core.
_UNIT_RANGES: dict[str, tuple[float, float]] = {
    "ratio": (0.0, 1.0),
    "%": (0.0, 100.0),
    "pct": (0.0, 100.0),
}

# Minimum number of genuinely observed steps before a flatline (identical
# value across the whole window) is treated as meaningful signal rather than
# noise from a short or single-snapshot window.
_FLATLINE_MIN_STEPS = 4

# Above this fraction of sensors failing to normalize, flag the window for
# review rather than silently trusting whatever did map.
_UNKNOWN_SENSOR_RATIO_WARN = 0.2

# Above this fraction of raw readings rejected before reaching the window,
# flag the ingestion path for review.
_REJECTED_SAMPLE_RATIO_WARN = 0.1


@dataclass(frozen=True)
class QualityFinding:
    code: str
    severity: str  # "info" | "warn" | "fail"
    feature: str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "feature": self.feature,
            "message": self.message,
        }


@dataclass(frozen=True)
class QualityControlReport:
    status: str  # "pass" | "warn" | "fail"
    score: float  # 0..1, higher is better
    findings: tuple[QualityFinding, ...]
    checks_run: tuple[str, ...]
    feature_completeness: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "score": round(float(self.score), 4),
            "findings": [finding.to_dict() for finding in self.findings],
            "checks_run": list(self.checks_run),
            "feature_completeness": {
                name: round(float(ratio), 4) for name, ratio in self.feature_completeness.items()
            },
        }


def _check_completeness(window: SensorWindow, schema: AgentSchema) -> tuple[list[QualityFinding], dict[str, float]]:
    findings: list[QualityFinding] = []
    completeness: dict[str, float] = {}
    steps = len(window.presence_mask)
    for col, feature in enumerate(window.features):
        if steps == 0:
            ratio = 0.0
        else:
            present = sum(1 for row in window.presence_mask if row[col])
            ratio = present / steps
        completeness[feature.name] = ratio
        if feature.name in schema.required_features and ratio < 1.0:
            severity = "fail" if ratio == 0.0 else "warn"
            findings.append(
                QualityFinding(
                    code="incomplete_required_feature",
                    severity=severity,
                    feature=feature.name,
                    message=f"Required feature '{feature.name}' present in {ratio:.0%} of window steps.",
                )
            )
    return findings, completeness


def _check_range_sanity(window: SensorWindow) -> list[QualityFinding]:
    findings: list[QualityFinding] = []
    for col, feature in enumerate(window.features):
        limits = _UNIT_RANGES.get(feature.unit.strip().lower())
        if not limits:
            continue
        lower, upper = limits
        for row_idx, row in enumerate(window.matrix):
            if not window.presence_mask[row_idx][col]:
                continue
            value = row[col]
            if value < lower or value > upper:
                findings.append(
                    QualityFinding(
                        code="value_out_of_range",
                        severity="warn",
                        feature=feature.name,
                        message=(
                            f"'{feature.name}' value {value:g} is outside the expected "
                            f"{feature.unit} range [{lower:g}, {upper:g}]."
                        ),
                    )
                )
                break  # one finding per feature is enough signal, avoid noise
    return findings


def _check_flatline(window: SensorWindow) -> list[QualityFinding]:
    """Flag features whose value never moved across the real observations.

    Only the genuinely observed steps are inspected. The adapter pads a
    short window out to seq_len by repeating its earliest row (see
    AgentSchemaAdapter.build_window), and those padded rows carry
    presence=True, so counting the whole window would both inflate the
    reported step count and describe copies as if they were readings.
    Padding is inserted at the front, so the real observations are the
    window's trailing observed_step_count rows.

    A window that does not report observed_step_count at all (0 means
    "unknown", not "no data") is skipped rather than guessed at: without
    knowing which rows are real, a flatline verdict would be unfounded.
    """
    findings: list[QualityFinding] = []
    if window.observed_step_count < _FLATLINE_MIN_STEPS:
        return findings

    observed_rows = list(zip(window.matrix, window.presence_mask))[-window.observed_step_count :]
    for col, feature in enumerate(window.features):
        values = [row[col] for row, presence in observed_rows if presence[col]]
        if len(values) >= _FLATLINE_MIN_STEPS and len(set(values)) == 1:
            findings.append(
                QualityFinding(
                    code="flatline_signal",
                    severity="warn",
                    feature=feature.name,
                    message=(
                        f"'{feature.name}' reported the identical value across {len(values)} "
                        "observed steps, which can indicate a stuck sensor."
                    ),
                )
            )
    return findings


def _check_normalization_health(normalization_trace: NormalizationTrace | None) -> list[QualityFinding]:
    findings: list[QualityFinding] = []
    if normalization_trace is None:
        return findings
    total = normalization_trace.normalized_sensor_count + normalization_trace.unknown_sensor_count
    if total > 0:
        unknown_ratio = normalization_trace.unknown_sensor_count / total
        if unknown_ratio > _UNKNOWN_SENSOR_RATIO_WARN:
            findings.append(
                QualityFinding(
                    code="high_unknown_sensor_ratio",
                    severity="warn",
                    feature=None,
                    message=(
                        f"{normalization_trace.unknown_sensor_count} of {total} sensors could not "
                        f"be normalized ({unknown_ratio:.0%})."
                    ),
                )
            )
    if normalization_trace.ambiguous_sensor_count:
        findings.append(
            QualityFinding(
                code="ambiguous_sensor_mapping",
                severity="info",
                feature=None,
                message=(
                    f"{normalization_trace.ambiguous_sensor_count} sensor name(s) matched more "
                    "than one canonical feature and need review."
                ),
            )
        )
    return findings


def _check_rejected_samples(window: SensorWindow) -> list[QualityFinding]:
    findings: list[QualityFinding] = []
    if window.raw_sensor_count > 0 and window.rejected_samples:
        rejected_ratio = len(window.rejected_samples) / window.raw_sensor_count
        if rejected_ratio > _REJECTED_SAMPLE_RATIO_WARN:
            findings.append(
                QualityFinding(
                    code="high_rejected_sample_ratio",
                    severity="warn",
                    feature=None,
                    message=(
                        f"{len(window.rejected_samples)} of {window.raw_sensor_count} raw readings "
                        f"were rejected before scoring ({rejected_ratio:.0%})."
                    ),
                )
            )
    return findings


def evaluate_quality(
    window: SensorWindow,
    schema: AgentSchema,
    normalization_trace: NormalizationTrace | None = None,
) -> QualityControlReport:
    """Run a set of cheap, deterministic input-quality checks ahead of scoring.

    This runs before the anomaly core and physics layer see the data, so a
    diagnosis can carry an honest note about how trustworthy its own inputs
    were, rather than presenting every risk score with equal confidence
    regardless of missing, stuck, or unmapped sensors.
    """
    findings: list[QualityFinding] = []
    checks_run: list[str] = []

    completeness_findings, feature_completeness = _check_completeness(window, schema)
    findings.extend(completeness_findings)
    checks_run.append("completeness")

    findings.extend(_check_range_sanity(window))
    checks_run.append("range_sanity")

    findings.extend(_check_flatline(window))
    checks_run.append("flatline")

    findings.extend(_check_normalization_health(normalization_trace))
    checks_run.append("normalization_health")

    findings.extend(_check_rejected_samples(window))
    checks_run.append("rejected_samples")

    fail_count = sum(1 for finding in findings if finding.severity == "fail")
    warn_count = sum(1 for finding in findings if finding.severity == "warn")
    if fail_count:
        status = "fail"
    elif warn_count:
        status = "warn"
    else:
        status = "pass"

    score = max(0.0, round(1.0 - fail_count * 0.35 - warn_count * 0.1, 4))

    return QualityControlReport(
        status=status,
        score=score,
        findings=tuple(findings),
        checks_run=tuple(checks_run),
        feature_completeness=feature_completeness,
    )
