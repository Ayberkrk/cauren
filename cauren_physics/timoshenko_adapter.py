"""Optional, compatibility-preserving bridge to the Timoshenko engine.

Cauren remains installable without Timoshenko. When the separate local
``Timoshenko`` package is installed, shared generic modal and shear-building
calculations can delegate to it; Cauren-specific risk and interpretation stay
in Cauren. The legacy pure-Python implementations remain the dependency-free
fallback.
"""

from __future__ import annotations

import math
from typing import Sequence


def _engine():
    try:
        import timoshenko as tm
    except ModuleNotFoundError as error:
        if error.name in {"timoshenko", "numpy"}:
            return None
        raise
    version = str(getattr(tm, "__version__", "0"))
    try:
        major = int(version.split(".", 1)[0])
    except ValueError:
        return None
    # Timoshenko 1.x pairs modes by index and treats Hann leakage as damping.
    return tm if major >= 2 else None


def pair_modes(reference_hz: Sequence[float], observed_hz: Sequence[float]):
    """Delegate mode pairing to Timoshenko 2.x, or return ``None`` if absent."""
    tm = _engine()
    if tm is None:
        return None
    valid_references = [
        (index, float(value))
        for index, value in enumerate(reference_hz)
        if math.isfinite(float(value)) and float(value) > 0.0
    ]
    valid_observed = [
        (index, float(value))
        for index, value in enumerate(observed_hz)
        if math.isfinite(float(value)) and float(value) > 0.0
    ]
    pairs = tm.modal.pair_modes(
        [value for _, value in valid_references],
        [value for _, value in valid_observed],
    )
    return tuple((valid_references[ref][0], valid_observed[obs][0]) for ref, obs in pairs)


def identify_single_channel(
    series: Sequence[float],
    sampling_hz: float,
    *,
    max_modes: int,
    min_prominence_ratio: float,
):
    """Return a Cauren-shaped result via Timoshenko, or ``None`` if unavailable.

    Peak threshold is mapped from power ratio to amplitude ratio by square
    root. Cauren's public ``peak_picking`` method and power-scaled amplitude
    semantics are retained for compatibility.
    """
    tm = _engine()
    values = [float(value) for value in series]
    if tm is None or len(values) < 8 or max_modes < 1 or not 0.0 <= min_prominence_ratio < 1.0:
        return None
    if not math.isfinite(float(sampling_hz)) or float(sampling_hz) <= 0.0:
        return None
    from .oma import ModalParameter, OMAResult

    sensor = tm.SensorData(
        samples=values,
        sampling_hz=float(sampling_hz),
        unit="unknown",
        channel="cauren_sensor",
    )
    modal = tm.modal.identify(
        sensor,
        max_modes=max_modes,
        min_peak_ratio=math.sqrt(min_prominence_ratio),
    )
    max_amplitude = max((mode.amplitude for mode in modal.modes), default=0.0)
    modes = tuple(
        ModalParameter(
            frequency_hz=mode.frequency_hz,
            damping_ratio=mode.damping_ratio,
            amplitude=mode.amplitude**2 / len(values),
            confidence=(mode.amplitude / max_amplitude) ** 2 if max_amplitude > 0.0 else 0.0,
        )
        for mode in modal.modes
    )
    return OMAResult(
        modal_parameters=modes,
        sampling_hz=float(sampling_hz),
        num_samples=len(values),
        method="peak_picking",
        frequency_resolution_hz=modal.resolution_hz,
    )


def identify_multichannel(
    columns: Sequence[Sequence[float]],
    sampling_hz: float,
    *,
    channel_ids: Sequence[str],
    max_modes: int,
    min_prominence_ratio: float,
    unit: str = "unknown",
):
    """Identify multi-channel modes through Timoshenko FDD, or return ``None``."""
    tm = _engine()
    if tm is None:
        return None
    if len(columns) < 2 or len(channel_ids) != len(columns):
        raise ValueError("multi-channel FDD requires at least two channels and one channel id per series")
    sample_count = len(columns[0])
    if any(len(column) != sample_count for column in columns):
        raise ValueError("all vibration channels must have the same number of samples")
    if not math.isfinite(float(sampling_hz)) or float(sampling_hz) <= 0.0:
        raise ValueError("sampling_hz must be positive")

    samples = [
        [float(columns[channel][sample]) for channel in range(len(columns))]
        for sample in range(sample_count)
    ]
    observations = tm.MultiChannelData(
        samples=samples,
        sampling_hz=float(sampling_hz),
        channel_ids=list(channel_ids),
        units=[unit] * len(columns),
    )
    fdd = tm.identify_fdd(
        observations,
        max_modes=max_modes,
        min_peak_ratio=min_prominence_ratio,
    )
    from .oma import ModalParameter, OMAResult

    max_singular_value = max((mode.singular_value for mode in fdd.modes), default=0.0)
    modes = tuple(
        ModalParameter(
            frequency_hz=mode.frequency_hz,
            damping_ratio=None,
            amplitude=mode.singular_value,
            confidence=mode.singular_value / max_singular_value if max_singular_value > 0.0 else 0.0,
        )
        for mode in fdd.modes
    )
    mode_shapes = tuple(
        {
            "frequency_hz": mode.frequency_hz,
            "shape": {
                channel_id: {"real": real, "imag": imag}
                for channel_id, real, imag in zip(mode.channel_ids, mode.shape_real, mode.shape_imag)
            },
        }
        for mode in fdd.modes
    )
    return OMAResult(
        modal_parameters=modes,
        sampling_hz=float(sampling_hz),
        num_samples=sample_count,
        method="fdd",
        frequency_resolution_hz=fdd.resolution_hz,
        mode_shapes=mode_shapes,
    )


def shear_building_frequencies(story_masses_kg: Sequence[float], story_stiffness_n_per_m: Sequence[float]):
    """Return shared-engine shear-building frequencies, or ``None`` if absent."""
    tm = _engine()
    if tm is None:
        return None
    model = tm.Structure(
        story_masses_kg=story_masses_kg,
        story_stiffness_n_m=story_stiffness_n_per_m,
    )
    return tuple(model.natural_frequencies_hz)


def update_scale_summary(
    story_masses_kg: Sequence[float],
    story_stiffness_n_per_m: Sequence[float],
    observed_frequencies_hz: Sequence[float],
):
    """Return Timoshenko's global stiffness scale and spread, or ``None`` if unavailable."""
    tm = _engine()
    if tm is None or not observed_frequencies_hz:
        return None
    modes = tuple(
        tm.modal.Mode(frequency_hz=float(frequency), amplitude=1.0)
        for frequency in observed_frequencies_hz
    )
    modal = tm.modal.ModalResult(
        modes=modes,
        sampling_hz=1.0,
        sample_count=len(modes),
        resolution_hz=1.0,
        channel="cauren_reference_model",
        method="cauren_frequency_list",
        status="ok",
    )
    updated = tm.update(
        tm.Structure(
            story_masses_kg=story_masses_kg,
            story_stiffness_n_m=story_stiffness_n_per_m,
        ),
        modal,
    )
    if updated.update_scale_factor is None or updated.update_mode_scale_spread_pct is None:
        return None
    return float(updated.update_scale_factor), float(updated.update_mode_scale_spread_pct)
