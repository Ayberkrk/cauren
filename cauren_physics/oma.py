"""Operational Modal Analysis (OMA) / system identification for civil structures.

This module estimates a structure's modal parameters (natural frequencies and
damping ratios) directly from ambient vibration time-series -- no controlled
excitation or known input force required, which is the defining trait of
"operational" modal analysis as opposed to classical experimental modal
analysis. It implements the classic frequency-domain Peak-Picking method:

  1. Detrend and window the acceleration/velocity/displacement series.
  2. Estimate its power spectral density (PSD).
  3. Pick prominent spectral peaks -- each is read as a candidate structural
     mode's natural frequency.
  4. Estimate each mode's damping ratio from the half-power (-3 dB)
     bandwidth around its peak.

Single-channel peak-picking works without optional packages. When Timoshenko
2.0 or newer is installed, multi-channel Frequency Domain Decomposition is
also available. A structure's natural frequencies
dropping over time relative to a baseline is a standard structural-health
early-warning sign (stiffness loss from cracking, foundation movement,
member damage, etc.) -- see `compare_to_baseline`, which is how this module's
output is meant to feed into `cauren_physics.civil.CivilPhysics`.

This produces a decision-support signal only, consistent with the rest of
Cauren: a frequency drop is evidence to route to a structural engineer for
review, not an automated damage verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from .timoshenko_adapter import (
    identify_single_channel as _identify_with_timoshenko,
    pair_modes as _pair_modes_with_timoshenko,
)

try:
    import numpy as _np
except ModuleNotFoundError:  # pragma: no cover - exercised when numpy absent
    _np = None

# Naive DFT fallback (no numpy) is O(n^2); cap the analyzed window so it
# stays cheap on constrained hardware. Only the most recent samples are kept.
_NAIVE_DFT_MAX_SAMPLES = 2048


@dataclass(frozen=True)
class ModalParameter:
    frequency_hz: float
    damping_ratio: float | None
    amplitude: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "frequency_hz": round(float(self.frequency_hz), 6),
            "damping_ratio": None if self.damping_ratio is None else round(float(self.damping_ratio), 6),
            "amplitude": round(float(self.amplitude), 6),
            "confidence": round(float(self.confidence), 6),
        }


@dataclass(frozen=True)
class OMAResult:
    modal_parameters: tuple[ModalParameter, ...]
    sampling_hz: float
    num_samples: int
    method: str
    frequency_resolution_hz: float
    mode_shapes: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = {
            "modal_parameters": [mode.to_dict() for mode in self.modal_parameters],
            "sampling_hz": round(float(self.sampling_hz), 6),
            "num_samples": int(self.num_samples),
            "method": self.method,
            "frequency_resolution_hz": round(float(self.frequency_resolution_hz), 6),
        }
        if self.mode_shapes:
            result["mode_shapes"] = list(self.mode_shapes)
        return result

    @property
    def dominant_frequency_hz(self) -> float | None:
        if not self.modal_parameters:
            return None
        return max(self.modal_parameters, key=lambda mode: mode.amplitude).frequency_hz


def _detrend(values: list[float]) -> list[float]:
    if not values:
        return values
    mean = sum(values) / float(len(values))
    return [value - mean for value in values]


def _hann_window(values: list[float]) -> list[float]:
    n = len(values)
    if n < 2:
        return list(values)
    windowed = []
    for i, value in enumerate(values):
        w = 0.5 - 0.5 * math.cos(2.0 * math.pi * i / (n - 1))
        windowed.append(value * w)
    return windowed


def _power_spectrum(values: list[float], sampling_hz: float) -> tuple[list[float], list[float]]:
    """Returns (frequencies_hz, power) for the one-sided spectrum."""
    n = len(values)
    if n == 0:
        return [], []
    if _np is not None:
        spectrum = _np.fft.rfft(_np.asarray(values, dtype=float))
        freqs = _np.fft.rfftfreq(n, d=1.0 / float(sampling_hz))
        power = (_np.abs(spectrum) ** 2) / float(n)
        # Cast back to plain floats so numpy scalars never reach the
        # dataclasses (and from there JSON reports and comparisons).
        return [float(value) for value in freqs], [float(value) for value in power]

    # Pure-Python fallback: naive DFT capped to a bounded sample count so it
    # stays affordable without numpy.
    truncated = values[-_NAIVE_DFT_MAX_SAMPLES:]
    n = len(truncated)
    half = n // 2 + 1
    freqs = [k * float(sampling_hz) / n for k in range(half)]
    power = []
    for k in range(half):
        real = 0.0
        imag = 0.0
        angle_step = -2.0 * math.pi * k / n
        for t, value in enumerate(truncated):
            angle = angle_step * t
            real += value * math.cos(angle)
            imag += value * math.sin(angle)
        power.append((real * real + imag * imag) / float(n))
    return freqs, power


def _find_peaks(power: list[float], *, min_prominence_ratio: float) -> list[int]:
    if len(power) < 3:
        return []
    peak_power = max(power)
    if peak_power <= 0.0:
        return []
    threshold = peak_power * min_prominence_ratio
    peaks: list[int] = []
    for i in range(1, len(power) - 1):
        if power[i] < threshold:
            continue
        if power[i] >= power[i - 1] and power[i] >= power[i + 1]:
            peaks.append(i)
    return peaks


def _half_power_damping(power: list[float], freqs: list[float], frequency_hz: float) -> float | None:
    if len(freqs) < 3 or frequency_hz <= 0.0:
        return None
    resolution = freqs[1] - freqs[0]
    if resolution <= 0.0:
        return None
    nearest = int(round(frequency_hz / resolution))
    low, high = max(1, nearest - 1), min(len(power) - 2, nearest + 1)
    if high < low:
        return None
    peak_idx = max(range(low, high + 1), key=power.__getitem__)
    half_power = power[peak_idx] / 2.0
    if half_power <= 0.0:
        return None

    f_left = None
    for index in range(peak_idx, 0, -1):
        p0, p1 = power[index - 1], power[index]
        if p0 <= half_power < p1:
            f_left = freqs[index - 1] + (half_power - p0) * resolution / (p1 - p0)
            break

    f_right = None
    for index in range(peak_idx, len(power) - 1):
        p0, p1 = power[index], power[index + 1]
        if p0 > half_power >= p1:
            f_right = freqs[index] + (p0 - half_power) * resolution / (p0 - p1)
            break

    f_peak = freqs[peak_idx]
    if f_left is None or f_right is None or f_right - f_left < 4.0 * resolution:
        return None
    return min(1.0, (f_right - f_left) / (2.0 * f_peak))


def _averaged_psd(values: Sequence[float], sampling_hz: float) -> tuple[list[float], list[float]] | None:
    samples = list(values)
    if _np is None:
        samples = samples[-_NAIVE_DFT_MAX_SAMPLES:]
    longest = len(samples) / 4.5
    if longest < 64:
        return None
    segment_length = 2 ** int(math.floor(math.log2(longest)))
    hop = segment_length // 2
    if _np is not None:
        array = _np.asarray(samples, dtype=float)
        window = _np.hanning(segment_length)
        power = _np.zeros(segment_length // 2 + 1, dtype=float)
        count = 0
        for start in range(0, len(array) - segment_length + 1, hop):
            segment = array[start : start + segment_length]
            spectrum = _np.fft.rfft((segment - _np.mean(segment)) * window)
            power += _np.abs(spectrum) ** 2
            count += 1
        if count == 0:
            return None
        power /= count
        freqs = _np.fft.rfftfreq(segment_length, d=1.0 / float(sampling_hz))
        return [float(value) for value in freqs], [float(value) for value in power]

    averaged_power = [0.0] * (segment_length // 2 + 1)
    for start in range(0, len(samples) - segment_length + 1, hop):
        segment = _hann_window(_detrend(samples[start : start + segment_length]))
        freqs, segment_power = _power_spectrum(segment, sampling_hz)
        for index, value in enumerate(segment_power):
            averaged_power[index] += value
    count = 1 + (len(samples) - segment_length) // hop
    return freqs, [value / count for value in averaged_power]


def _estimate_damping(values: Sequence[float], sampling_hz: float, frequency_hz: float) -> float | None:
    averaged = _averaged_psd(values, sampling_hz)
    if averaged is None:
        return None
    freqs, power = averaged
    return _half_power_damping(power, freqs, frequency_hz)


def _identify_modal_parameters_fallback(
    series: Sequence[float],
    sampling_hz: float,
    *,
    max_modes: int = 3,
    min_prominence_ratio: float = 0.15,
) -> OMAResult:
    """Estimate natural frequencies and damping ratios from an ambient vibration series.

    `series` is a single-channel time-domain signal (e.g. acceleration in
    m/s^2, or any consistent unit -- amplitude is only meaningful relatively,
    for ranking modes by prominence). `sampling_hz` must match the actual
    sample rate the series was recorded at, since it fixes both the frequency
    axis and the Nyquist limit.
    """
    if sampling_hz <= 0.0:
        raise ValueError("sampling_hz must be positive")

    values = [float(v) for v in series]
    n = len(values)
    if n < 8:
        return OMAResult(
            modal_parameters=(),
            sampling_hz=float(sampling_hz),
            num_samples=n,
            method="peak_picking",
            frequency_resolution_hz=0.0,
        )

    prepared = _hann_window(_detrend(values))
    freqs, power = _power_spectrum(prepared, sampling_hz)
    resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 0.0

    peak_indices = _find_peaks(power, min_prominence_ratio=min_prominence_ratio)
    peak_power_max = max((power[i] for i in peak_indices), default=0.0)

    modes: list[ModalParameter] = []
    for idx in sorted(peak_indices, key=lambda i: power[i], reverse=True)[: max(0, max_modes)]:
        damping = _estimate_damping(values, sampling_hz, freqs[idx])
        confidence = 0.0 if peak_power_max <= 0.0 else min(1.0, power[idx] / peak_power_max)
        modes.append(
            ModalParameter(
                frequency_hz=freqs[idx],
                damping_ratio=damping,
                amplitude=power[idx],
                confidence=confidence,
            )
        )
    modes.sort(key=lambda mode: mode.frequency_hz)

    return OMAResult(
        modal_parameters=tuple(modes),
        sampling_hz=float(sampling_hz),
        num_samples=n,
        method="peak_picking",
        frequency_resolution_hz=float(resolution),
    )


def identify_modal_parameters(
    series: Sequence[float],
    sampling_hz: float,
    *,
    max_modes: int = 3,
    min_prominence_ratio: float = 0.15,
) -> OMAResult:
    """Identify single-channel modal peaks, using Timoshenko when installed.

    Cauren keeps its public result and baseline-risk behavior. Install the
    local Timoshenko package to share generic numerical work; otherwise the
    compatible pure-Python Cauren implementation is used.
    """
    shared = _identify_with_timoshenko(
        series,
        sampling_hz,
        max_modes=max_modes,
        min_prominence_ratio=min_prominence_ratio,
    )
    if shared is not None:
        return shared
    return _identify_modal_parameters_fallback(
        series,
        sampling_hz,
        max_modes=max_modes,
        min_prominence_ratio=min_prominence_ratio,
    )


@dataclass(frozen=True)
class FrequencyDriftFinding:
    baseline_frequency_hz: float
    current_frequency_hz: float | None
    drift_pct: float | None
    severity: str  # "none" | "watch" | "warn" | "alarm"
    resolution_limited: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_frequency_hz": round(float(self.baseline_frequency_hz), 6),
            "current_frequency_hz": None if self.current_frequency_hz is None else round(float(self.current_frequency_hz), 6),
            "drift_pct": None if self.drift_pct is None else round(float(self.drift_pct), 6),
            "severity": self.severity,
            "resolution_limited": self.resolution_limited,
        }


@dataclass(frozen=True)
class FrequencyDriftReport:
    findings: tuple[FrequencyDriftFinding, ...]
    max_drop_pct: float
    overall_severity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [finding.to_dict() for finding in self.findings],
            "max_drop_pct": round(float(self.max_drop_pct), 6),
            "overall_severity": self.overall_severity,
        }


def _severity_for_drop(drop_pct: float, *, warn_pct: float, alarm_pct: float) -> str:
    if drop_pct >= alarm_pct:
        return "alarm"
    if drop_pct >= warn_pct:
        return "warn"
    if drop_pct > 0.0:
        return "watch"
    return "none"


def compare_to_baseline(
    current: OMAResult,
    baseline_frequencies_hz: Sequence[float],
    *,
    match_tolerance_hz: float = 1.5,
    warn_pct: float = 5.0,
    alarm_pct: float = 10.0,
) -> FrequencyDriftReport:
    """Compares freshly identified modes against a structure's baseline (as-built or prior survey) frequencies.

    A natural frequency *drop* relative to baseline is the signal of
    interest (reduced stiffness); a rise is not flagged the same way since it
    is commonly just measurement/environmental variation (temperature,
    live-load mass changes), not damage.
    """
    current_freqs = [mode.frequency_hz for mode in current.modal_parameters]
    pairs = _pair_modes_with_timoshenko(baseline_frequencies_hz, current_freqs)
    if pairs is None:
        pairs = _pair_modes(baseline_frequencies_hz, current_freqs)
    observed_by_baseline = dict(pairs)
    findings: list[FrequencyDriftFinding] = []
    max_drop = 0.0

    for baseline_index, baseline_hz in enumerate(baseline_frequencies_hz):
        baseline_hz = float(baseline_hz)
        observed_index = observed_by_baseline.get(baseline_index)
        nearest = None if observed_index is None else current_freqs[observed_index]
        if (
            not math.isfinite(baseline_hz)
            or baseline_hz <= 0.0
            or nearest is None
            or abs(nearest - baseline_hz) > match_tolerance_hz
        ):
            findings.append(
                FrequencyDriftFinding(
                    baseline_frequency_hz=baseline_hz,
                    current_frequency_hz=None,
                    drift_pct=None,
                    severity="watch",
                )
            )
            continue
        drift_pct = 100.0 * (baseline_hz - nearest) / baseline_hz
        resolution_limited = abs(nearest - baseline_hz) <= current.frequency_resolution_hz
        severity = "none" if resolution_limited else _severity_for_drop(drift_pct, warn_pct=warn_pct, alarm_pct=alarm_pct)
        if not resolution_limited:
            max_drop = max(max_drop, drift_pct)
        findings.append(
            FrequencyDriftFinding(
                baseline_frequency_hz=baseline_hz,
                current_frequency_hz=nearest,
                drift_pct=drift_pct,
                severity=severity,
                resolution_limited=resolution_limited,
            )
        )

    severities = [finding.severity for finding in findings]
    if "alarm" in severities:
        overall = "alarm"
    elif "warn" in severities:
        overall = "warn"
    elif "watch" in severities:
        overall = "watch"
    else:
        overall = "none"

    return FrequencyDriftReport(findings=tuple(findings), max_drop_pct=max_drop, overall_severity=overall)


def _pair_modes(reference_hz: Sequence[float], observed_hz: Sequence[float]) -> tuple[tuple[int, int], ...]:
    """Pair observations to references by nearest log frequency, once per reference."""
    valid_references = [
        (index, float(value))
        for index, value in enumerate(reference_hz)
        if math.isfinite(float(value)) and float(value) > 0.0
    ]
    best: dict[int, tuple[float, int]] = {}
    for observed_index, raw_value in enumerate(observed_hz):
        frequency = float(raw_value)
        if not valid_references or not math.isfinite(frequency) or frequency <= 0.0:
            continue
        distances = [abs(math.log(frequency / reference)) for _, reference in valid_references]
        valid_reference_index = min(range(len(valid_references)), key=distances.__getitem__)
        reference_index = valid_references[valid_reference_index][0]
        distance = distances[valid_reference_index]
        if reference_index not in best or distance < best[reference_index][0]:
            best[reference_index] = (distance, observed_index)
    return tuple((index, best[index][1]) for index in sorted(best))
