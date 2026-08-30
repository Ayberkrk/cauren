from __future__ import annotations

import math
import statistics

from .backbone import CaurenHybridBackbone, HybridBackboneUnavailable
from .contracts import CoreOutput, SensorWindow
from .risk_calibration import calibrated_core_risk, risk_profile


class CaurenCoreRuntime:
    """
    General anomaly-pattern core.

    This runtime intentionally has no canonical dimension list and no global hard-limit
    status taxonomy. It scores statistical pattern shape over the agent-provided feature
    matrix and leaves physical interpretation to sector agents.
    """

    def __init__(self, *, backbone: CaurenHybridBackbone | None = None):
        self.backbone = backbone or CaurenHybridBackbone.from_env()
        self._last_backbone_metadata: dict = {}

    def calibrate(
        self,
        window: SensorWindow,
        *,
        sampling_hz: float = 1.0,
        runtime_mode: str | None = None,
    ) -> tuple[tuple[float, ...], ...]:
        if self.backbone is not None:
            try:
                calibrated, _ = self.backbone.calibrate(
                    window,
                    sampling_hz=sampling_hz,
                    runtime_mode=runtime_mode,
                )
                self._last_backbone_metadata = {
                    **self.backbone.metadata,
                    "used": True,
                    "fallback": False,
                }
                return calibrated
            except HybridBackboneUnavailable:
                self._last_backbone_metadata = {
                    **self.backbone.metadata,
                    "used": False,
                    "fallback": True,
                }

        matrix = [list(row) for row in window.matrix]
        mask = [list(row) for row in window.presence_mask]
        if not matrix:
            return tuple()
        calibrated = [list(row) for row in matrix]
        width = len(matrix[0])
        for idx in range(width):
            observed = [row[idx] for row, mask_row in zip(matrix, mask) if mask_row[idx]]
            if len(observed) < 3:
                continue
            baseline = float(statistics.median(observed))
            spread = float(statistics.pstdev(observed))
            if spread <= 1e-6:
                continue
            for row_idx, row in enumerate(matrix):
                if not mask[row_idx][idx]:
                    calibrated[row_idx][idx] = 0.0
                    continue
                delta = max(-3.0 * spread, min(3.0 * spread, row[idx] - baseline))
                calibrated[row_idx][idx] = baseline + delta
        if not self._last_backbone_metadata:
            self._last_backbone_metadata = {"name": "statistical_core", "used": True, "fallback": False}
        return tuple(tuple(float(value) for value in row) for row in calibrated)

    def classify(
        self,
        window: SensorWindow,
        *,
        sampling_hz: float = 1.0,
        runtime_mode: str | None = None,
        context: dict | None = None,
    ) -> CoreOutput:
        matrix = [list(row) for row in window.matrix]
        mask = [list(row) for row in window.presence_mask]
        context = context or {}
        profile = risk_profile(str(context.get("selected_agent_id") or ""))
        self._last_backbone_metadata = {}
        calibrated = self.calibrate(window, sampling_hz=sampling_hz, runtime_mode=runtime_mode)
        if not matrix or not any(any(row) for row in mask):
            return CoreOutput(
                anomaly_type="insufficient_observation",
                anomaly_family="unknown",
                risk_score=0.0,
                confidence=0.0,
                pattern_scores={"drift": 0.0, "spike": 0.0, "oscillation": 0.0},
                reconstruction_error=0.0,
                drift_score=0.0,
                spike_score=0.0,
                oscillation_score=0.0,
                calibrated_matrix=calibrated,
                backbone_metadata=dict(self._last_backbone_metadata),
            )

        # A single real observation gets repeated by the adapter to fill
        # out the window (see AgentSchemaAdapter.build_window), which
        # otherwise reads as a perfectly flat/constant time series to
        # every pattern-shape metric below -- guaranteeing a bogus
        # "flatline" verdict on every one-shot snapshot request. With no
        # real history there is no temporal shape to detect, so say that
        # honestly instead and let the sector's physics evaluation (which
        # reasons over absolute feature values, not their trend) carry
        # the risk signal for this request.
        if 0 < window.observed_step_count <= 1 and len(matrix) > 1:
            return CoreOutput(
                anomaly_type="insufficient_temporal_history",
                anomaly_family="snapshot_only",
                risk_score=0.0,
                confidence=0.1,
                pattern_scores={"drift": 0.0, "spike": 0.0, "oscillation": 0.0},
                reconstruction_error=0.0,
                drift_score=0.0,
                spike_score=0.0,
                oscillation_score=0.0,
                calibrated_matrix=calibrated,
                backbone_metadata=dict(self._last_backbone_metadata),
            )

        residual_values = [
            abs(matrix[row_idx][col_idx] - calibrated[row_idx][col_idx])
            for row_idx in range(len(matrix))
            for col_idx in range(len(matrix[row_idx]))
            if mask[row_idx][col_idx]
        ]
        reconstruction_error = float(sum(residual_values) / float(max(1, len(residual_values))))

        if len(matrix) > 1:
            # Step-to-step diffs must be normalized per feature before
            # comparing them, not pooled raw across every feature in the
            # matrix: the agent schema mixes 0-1 ratios (e.g.
            # structural_risk_score) with 0-100 percentages (e.g.
            # construction_progress_pct), so a pct-scaled feature's
            # ordinary year-to-year swing dwarfs a ratio feature's real
            # spike in absolute terms and dominates a pooled percentile
            # regardless of how unusual it actually is for that feature.
            # Scaling each feature's diffs by its own spread (the same
            # idiom already used for drift/level_shift/variance_change
            # below) makes "spike" scale-invariant like the others.
            spike_terms: list[float] = []
            positive_spike_terms: list[float] = []
            dip_terms: list[float] = []
            for col_idx in range(len(matrix[0])):
                y = [row[col_idx] for row, mask_row in zip(matrix, mask) if mask_row[col_idx]]
                if len(y) < 2:
                    continue
                feature_diffs = [y[i] - y[i - 1] for i in range(1, len(y))]
                if not feature_diffs:
                    continue
                spread = float(statistics.pstdev(y)) + 1e-6
                baseline = max(abs(float(statistics.median(y))) * 0.03, spread, 1e-6)
                normalized = [value / baseline for value in feature_diffs]
                spike_terms.append(_percentile([abs(value) for value in normalized], 95))
                positive = [value for value in normalized if value > 0.0]
                if positive:
                    positive_spike_terms.append(_percentile(positive, 95))
                negative = [value for value in normalized if value < 0.0]
                if negative:
                    dip_terms.append(abs(_percentile(negative, 5)))
            spike_score = max(spike_terms, default=0.0)
            positive_spike_score = max(positive_spike_terms, default=0.0)
            dip_score = max(dip_terms, default=0.0)
        else:
            spike_score = 0.0
            positive_spike_score = 0.0
            dip_score = 0.0

        dual_window_spike_score = _dual_window_spike_score(matrix, mask)

        drift_terms: list[float] = []
        oscillation_terms: list[float] = []
        level_shift_terms: list[float] = []
        variance_change_terms: list[float] = []
        flatline_terms: list[float] = []
        dynamic_feature_terms: list[float] = []
        burst_terms: list[float] = []
        width = len(matrix[0])
        for idx in range(width):
            y = [row[idx] for row, mask_row in zip(matrix, mask) if mask_row[idx]]
            if len(y) < 3:
                continue
            x_obs = [float(i) for i, mask_row in enumerate(mask) if mask_row[idx]]
            slope = _linear_slope(x_obs, y)
            spread = float(statistics.pstdev(y)) + 1e-6
            drift_terms.append(abs(slope) / spread)
            baseline = max(abs(float(statistics.median(y))), spread, 1.0)
            range_ratio = (max(y) - min(y)) / baseline
            spread_ratio = spread / baseline
            flatline_terms.append(1.0 if spread_ratio < 0.002 and len(set(round(value, 8) for value in y)) <= 2 else 0.0)
            dynamic_feature_terms.append(1.0 if range_ratio >= 0.01 or spread_ratio >= 0.004 else 0.0)
            if len(y) >= 6:
                midpoint = len(y) // 2
                left = y[:midpoint]
                right = y[midpoint:]
                left_mean = sum(left) / float(len(left))
                right_mean = sum(right) / float(len(right))
                pooled = float(statistics.pstdev(y)) + 1e-6
                absolute_shift = abs(right_mean - left_mean)
                level_shift_terms.append(
                    absolute_shift / max(abs(float(statistics.median(y))) * 0.12, pooled * 3.0, 1e-6)
                )
                left_std = float(statistics.pstdev(left)) + 1e-6
                right_std = float(statistics.pstdev(right)) + 1e-6
                variance_change_terms.append(
                    abs(right_std - left_std)
                    / max(abs(float(statistics.median(y))) * 0.04, max(left_std, right_std, 1e-6))
                )
            feature_diffs = [abs(right - left) for left, right in zip(y, y[1:])]
            if feature_diffs:
                local_spread = float(statistics.pstdev(y)) + 1e-6
                burst_threshold = max(abs(float(statistics.median(y))) * 0.02, 2.5 * local_spread, 1e-6)
                burst_count = sum(1 for value in feature_diffs if value > burst_threshold)
                burst_terms.append(min(1.0, burst_count / 3.0))
            mean_y = sum(y) / float(len(y))
            centered = [value - mean_y for value in y]
            if len(centered) >= 4:
                zero_crossings = sum(
                    1
                    for left, right in zip(centered, centered[1:])
                    if (left < 0 <= right) or (left >= 0 > right)
                )
                amplitude = max(y) - min(y)
                amplitude_gate = min(1.0, amplitude / max(abs(float(statistics.median(y))) * 0.05, spread * 2.5, 1e-6))
                oscillation_terms.append(
                    (float(zero_crossings) / float(max(1, len(centered) - 1))) * amplitude_gate
                )

        drift_score = float(sum(drift_terms) / float(len(drift_terms))) if drift_terms else 0.0
        oscillation_score = (
            float(sum(oscillation_terms) / float(len(oscillation_terms)))
            if oscillation_terms
            else 0.0
        )
        level_shift_score = (
            float(sum(level_shift_terms) / float(len(level_shift_terms)))
            if level_shift_terms
            else 0.0
        )
        variance_change_score = (
            float(sum(variance_change_terms) / float(len(variance_change_terms)))
            if variance_change_terms
            else 0.0
        )
        flatline_raw_score = (
            float(sum(flatline_terms) / float(len(flatline_terms)))
            if flatline_terms
            else 0.0
        )
        dynamic_feature_share = (
            float(sum(dynamic_feature_terms) / float(len(dynamic_feature_terms)))
            if dynamic_feature_terms
            else 0.0
        )
        repeating_burst_score = (
            float(sum(burst_terms) / float(len(burst_terms)))
            if burst_terms
            else 0.0
        )
        temporal_dropout_terms: list[float] = []
        temporal_gap_terms: list[float] = []
        missing_tail_terms: list[float] = []
        for idx in range(width):
            column_presence = [row[idx] for row in mask]
            if any(column_presence):
                temporal_dropout_terms.append(
                    sum(1 for item in column_presence if not item) / float(len(column_presence))
                )
                temporal_gap_terms.append(_largest_missing_gap(column_presence) / float(max(1, len(column_presence))))
                tail = column_presence[-8:] if len(column_presence) >= 8 else column_presence
                missing_tail_terms.append(sum(1 for item in tail if not item) / float(max(1, len(tail))))
        mask_size = sum(len(row) for row in mask)
        dropout_score = (
            float(sum(temporal_dropout_terms) / float(len(temporal_dropout_terms)))
            if temporal_dropout_terms
            else 0.0
        )
        dropout_gap_score = (
            float(sum(temporal_gap_terms) / float(len(temporal_gap_terms)))
            if temporal_gap_terms
            else 0.0
        )
        dropout_tail_score = (
            float(sum(missing_tail_terms) / float(len(missing_tail_terms)))
            if missing_tail_terms
            else 0.0
        )
        dropout_score = min(
            1.0,
            max(dropout_score / 0.30, dropout_gap_score / 0.18, dropout_tail_score / 0.45),
        ) if (dropout_score > 0.0 or dropout_gap_score > 0.0 or dropout_tail_score > 0.0) else 0.0
        flatline_score = flatline_raw_score if (flatline_raw_score >= 0.5 or dynamic_feature_share >= 0.20 or dropout_score > 0.0) else 0.0
        multisensor_residual_score = reconstruction_error / (reconstruction_error + 1.0)
        correlation_break_score = _correlation_break_score(matrix, mask)
        calibration_bias_consistency = _calibration_bias_consistency(matrix, calibrated, mask)
        calibration_bias_baseline = _baseline_delta_score(window, mask, context)
        if calibration_bias_baseline >= 0.45:
            flatline_score = min(flatline_score, 0.18)
        collective_score = _collective_score(
            [
                min(1.0, drift_score),
                min(1.0, spike_score / (spike_score + 1.0)),
                min(1.0, oscillation_score),
                min(1.0, level_shift_score),
                min(1.0, variance_change_score),
                min(1.0, repeating_burst_score),
                min(1.0, multisensor_residual_score),
            ]
        )
        contextual_score = _contextual_anomaly_score(window, context)
        seasonal_score = _seasonal_deviation_score(window, context)
        calibration_bias_score = min(
            1.0,
            max(multisensor_residual_score, calibration_bias_consistency, calibration_bias_baseline),
        )
        slow_degradation_score = min(1.0, drift_score * 0.75) if drift_score < 1.6 else 0.0

        raw_scores = {
            "drift": min(1.0, drift_score),
            "spike": min(
                1.0,
                max(
                    max(spike_score, positive_spike_score) / (max(spike_score, positive_spike_score) + 1.0),
                    dual_window_spike_score,
                ),
            ),
            "dip": min(1.0, dip_score / (dip_score + 1.0)),
            "oscillation": min(1.0, oscillation_score),
            "level_shift": min(1.0, level_shift_score),
            "variance_change": min(1.0, variance_change_score),
            "flatline": min(1.0, flatline_score),
            "sensor_dropout": min(1.0, dropout_score),
            "correlation_break": min(1.0, correlation_break_score),
            "repeating_burst": min(1.0, repeating_burst_score),
            "collective_anomaly": min(1.0, collective_score),
            "multisensor_residual": min(1.0, multisensor_residual_score),
            "calibration_bias": min(1.0, calibration_bias_score),
            "slow_degradation": min(1.0, slow_degradation_score),
            "contextual_anomaly": min(1.0, contextual_score),
            "seasonal_deviation": min(1.0, seasonal_score),
        }
        raw_anomaly_type = max(raw_scores, key=raw_scores.get)
        raw_top_score = float(raw_scores[raw_anomaly_type])
        weighted_scores = {
            key: max(0.0, min(1.0, float(value) * float(profile.pattern_weights.get(key, 1.0))))
            for key, value in raw_scores.items()
        }
        if calibration_bias_baseline >= 0.45:
            weighted_scores["calibration_bias"] = max(
                weighted_scores.get("calibration_bias", 0.0),
                min(1.0, calibration_bias_baseline),
            )
        anomaly_type = max(weighted_scores, key=weighted_scores.get)
        top_score = float(weighted_scores[anomaly_type])
        if top_score < 0.08 and reconstruction_error < 0.08:
            anomaly_type = "nominal_variation"
            anomaly_family = "normal_pattern"
        elif anomaly_type in {"drift", "slow_degradation"}:
            anomaly_family = "progressive_deviation"
        elif anomaly_type in {"spike", "dip", "repeating_burst"}:
            anomaly_family = "transient_deviation"
        elif anomaly_type == "oscillation":
            anomaly_family = "cyclic_instability"
        elif anomaly_type in {"level_shift", "variance_change"}:
            anomaly_family = "change_point_deviation"
        elif anomaly_type in {"flatline", "sensor_dropout", "calibration_bias"}:
            anomaly_family = "sensor_integrity_deviation"
        elif anomaly_type in {"correlation_break", "multisensor_residual", "collective_anomaly"}:
            anomaly_family = "multisensor_relation_deviation"
        elif anomaly_type in {"contextual_anomaly", "seasonal_deviation"}:
            anomaly_family = "contextual_deviation"
        else:
            anomaly_family = "unknown_family"

        raw_risk_score, risk_score = calibrated_core_risk(
            agent_id=str(context.get("selected_agent_id") or ""),
            raw_scores=raw_scores,
            anomaly_type=anomaly_type if anomaly_type != "nominal_variation" else raw_anomaly_type,
            reconstruction_error=reconstruction_error,
        )
        if anomaly_family == "normal_pattern":
            raw_risk_score = min(raw_risk_score, 0.18)
            risk_score = min(risk_score, 0.12)
        mask_true = sum(1 for row in mask for item in row if item)
        confidence = min(1.0, 0.35 + 0.45 * float(mask_true) / float(max(1, mask_size)) + 0.20 * risk_score)
        return CoreOutput(
            anomaly_type=anomaly_type,
            anomaly_family=anomaly_family,
            risk_score=float(risk_score),
            confidence=float(confidence),
            pattern_scores=raw_scores,
            reconstruction_error=float(reconstruction_error),
            drift_score=float(drift_score),
            spike_score=float(spike_score),
            oscillation_score=float(oscillation_score),
            calibrated_matrix=calibrated,
            raw_risk_score=float(raw_risk_score),
            calibrated_risk_score=float(risk_score),
            backbone_metadata=dict(self._last_backbone_metadata),
        )


def _percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not ordered:
        return 0.0
    idx = int(round((max(0, min(100, pct)) / 100.0) * (len(ordered) - 1)))
    return float(ordered[idx])


def _largest_missing_gap(column_presence: list[bool]) -> int:
    longest = 0
    current = 0
    for item in column_presence:
        if item:
            longest = max(longest, current)
            current = 0
        else:
            current += 1
    return max(longest, current)


def _dual_window_spike_score(matrix: list[list[float]], mask: list[list[bool]]) -> float:
    if len(matrix) < 4 or not matrix:
        return 0.0
    short_len = min(32, max(4, len(matrix) // 4))
    short_matrix = matrix[-short_len:]
    short_mask = mask[-short_len:]
    local_scores: list[float] = []
    width = len(short_matrix[0])
    for idx in range(width):
        series = [row[idx] for row, mask_row in zip(short_matrix, short_mask) if mask_row[idx]]
        if len(series) < 4:
            continue
        diffs = [abs(right - left) for left, right in zip(series, series[1:])]
        if not diffs:
            continue
        spread = float(statistics.pstdev(series)) + 1e-6
        local_scores.append(_percentile(diffs, 95) / max(spread * 2.0, abs(float(statistics.median(series))) * 0.03, 1e-6))
    return min(1.0, (sum(local_scores) / float(len(local_scores)))) if local_scores else 0.0


def _calibration_bias_consistency(
    matrix: list[list[float]],
    calibrated: tuple[tuple[float, ...], ...],
    mask: list[list[bool]],
) -> float:
    if not matrix or not calibrated:
        return 0.0
    per_feature_offsets: list[float] = []
    width = len(matrix[0])
    for idx in range(width):
        offsets = [
            matrix[row_idx][idx] - calibrated[row_idx][idx]
            for row_idx in range(len(matrix))
            if mask[row_idx][idx]
        ]
        if len(offsets) < 3:
            continue
        median_offset = float(statistics.median(offsets))
        spread = float(statistics.pstdev(offsets)) + 1e-6
        if abs(median_offset) > spread * 1.8:
            per_feature_offsets.append(abs(median_offset) / max(spread * 3.0, 1.0))
    if len(per_feature_offsets) < max(2, width // 4):
        return 0.0
    consistency = len(per_feature_offsets) / float(max(1, width))
    strength = sum(per_feature_offsets) / float(len(per_feature_offsets))
    return min(1.0, max(consistency, min(1.0, strength)))


def _baseline_delta_score(window: SensorWindow, mask: list[list[bool]], context: dict | None) -> float:
    if not context:
        return 0.0
    baseline_map = context.get("feature_baselines") or context.get("asset_baseline") or {}
    if not isinstance(baseline_map, dict):
        return 0.0
    hits: list[float] = []
    for idx, feature in enumerate(window.features):
        if idx >= len(mask[0]):
            break
        baseline_value = baseline_map.get(feature.name)
        try:
            baseline = float(baseline_value)
        except (TypeError, ValueError):
            continue
        observed = [
            float(window.matrix[row_idx][idx])
            for row_idx in range(len(window.matrix))
            if mask[row_idx][idx]
        ]
        if len(observed) < 3:
            continue
        median_observed = float(statistics.median(observed))
        spread = float(statistics.pstdev(observed)) + 1e-6
        delta = abs(median_observed - baseline)
        threshold = max(abs(baseline) * 0.08, spread * 3.0, 0.25)
        if delta > threshold:
            hits.append(delta / max(threshold * 2.0, 1e-6))
    if not hits:
        return 0.0
    coverage = len(hits) / float(max(1, len(baseline_map)))
    strength = sum(hits) / float(len(hits))
    return min(1.0, max(coverage, min(1.0, strength)))


def _correlation_break_score(matrix: list[list[float]], mask: list[list[bool]]) -> float:
    if len(matrix) < 8 or not matrix or len(matrix[0]) < 2:
        return 0.0
    midpoint = len(matrix) // 2
    scores: list[float] = []
    width = len(matrix[0])
    for left_idx in range(width):
        for right_idx in range(left_idx + 1, width):
            first_left: list[float] = []
            first_right: list[float] = []
            second_left: list[float] = []
            second_right: list[float] = []
            for row_idx, row in enumerate(matrix):
                if not (mask[row_idx][left_idx] and mask[row_idx][right_idx]):
                    continue
                if row_idx < midpoint:
                    first_left.append(row[left_idx])
                    first_right.append(row[right_idx])
                else:
                    second_left.append(row[left_idx])
                    second_right.append(row[right_idx])
            first_corr = _pearson(first_left, first_right)
            second_corr = _pearson(second_left, second_right)
            if first_corr is None or second_corr is None:
                continue
            scores.append(abs(first_corr - second_corr) / 2.0)
    return float(sum(scores) / float(len(scores))) if scores else 0.0


def _flatten_strings(value) -> list[str]:
    out: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            out.append(str(key))
            out.extend(_flatten_strings(item))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            out.extend(_flatten_strings(item))
    elif value is not None:
        out.append(str(value))
    return out


def _lowered_context_tokens(context: dict) -> set[str]:
    return {token.lower() for token in _flatten_strings(context) if str(token).strip()}


def _contextual_anomaly_score(window: SensorWindow, context: dict) -> float:
    tokens = _lowered_context_tokens(context)
    if not tokens:
        return 0.0
    if not window.matrix or not window.features:
        return 0.0
    feature_names = [feature.name.lower() for feature in window.features]
    latest_row = list(window.matrix[-1])
    medians: list[float] = []
    deviations: list[float] = []
    for idx in range(len(feature_names)):
        observed = [row[idx] for row, mask_row in zip(window.matrix, window.presence_mask) if mask_row[idx]]
        if len(observed) < 3:
            continue
        baseline = float(statistics.median(observed))
        medians.append(abs(baseline))
        spread = float(statistics.pstdev(observed)) + 1e-6
        deviations.append(abs(float(latest_row[idx]) - baseline) / max(abs(baseline) * 0.08, spread * 2.0, 1e-6))
    if not deviations:
        return 0.0
    evidence_strength = min(1.0, max(deviations))
    context_requested = bool(tokens & {"occupied", "unoccupied", "night", "day", "weekend", "after_hours", "holiday"})
    context_flagged = bool(tokens & {"anomalous_context", "context_alert", "occupancy_mismatch", "after_hours_activity"})
    if context_flagged:
        return max(0.0, min(1.0, 0.45 + 0.55 * evidence_strength))
    if context_requested and evidence_strength >= 0.35:
        return min(1.0, evidence_strength * 0.7)
    if "context_required" in tokens and evidence_strength >= 0.45:
        return min(1.0, evidence_strength * 0.6)
    return 0.0


def _seasonal_deviation_score(window: SensorWindow, context: dict) -> float:
    tokens = _lowered_context_tokens(context)
    if not tokens:
        return 0.0
    if not window.matrix or not window.features:
        return 0.0
    flagged = bool(tokens & {"seasonal_alert", "season_mismatch", "heating_out_of_season", "cooling_out_of_season"})
    winter = bool(tokens & {"winter", "heating_season"})
    summer = bool(tokens & {"summer", "cooling_season"})
    if not (flagged or winter or summer):
        return 0.0
    strongest = 0.0
    for idx, feature in enumerate(window.features):
        observed = [row[idx] for row, mask_row in zip(window.matrix, window.presence_mask) if mask_row[idx]]
        if len(observed) < 3:
            continue
        feature_name = feature.name.lower()
        baseline = float(statistics.median(observed))
        latest_value = float(window.matrix[-1][idx])
        spread = float(statistics.pstdev(observed)) + 1e-6
        normalized = abs(latest_value - baseline) / max(abs(baseline) * 0.08, spread * 2.0, 1e-6)
        if winter and any(token in feature_name for token in ("cool", "outdoor", "compressor")):
            strongest = max(strongest, normalized)
        if summer and any(token in feature_name for token in ("heat", "indoor", "hvac")):
            strongest = max(strongest, normalized)
    if flagged:
        strongest = max(strongest, 0.75)
    return min(1.0, strongest if strongest >= 0.3 else 0.0)


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 4:
        return None
    left_mean = sum(left) / float(len(left))
    right_mean = sum(right) / float(len(right))
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    left_energy = sum(value * value for value in left_centered)
    right_energy = sum(value * value for value in right_centered)
    denom = math.sqrt(left_energy * right_energy)
    if denom <= 1e-12:
        return None
    return float(sum(a * b for a, b in zip(left_centered, right_centered)) / denom)


def _collective_score(scores: list[float]) -> float:
    moderate = [score for score in scores if score >= 0.30]
    if len(moderate) < 2:
        return 0.0
    return min(1.0, sum(moderate) / float(len(moderate)) * min(1.0, len(moderate) / 4.0))


def _linear_slope(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0
    x_mean = sum(x_values) / float(len(x_values))
    y_mean = sum(y_values) / float(len(y_values))
    denom = sum((x - x_mean) ** 2 for x in x_values)
    if denom <= 1e-12:
        return 0.0
    numer = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    return float(numer / denom)
