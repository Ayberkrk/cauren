"""Synthetic checks for the Hell Bridge Test Arena feature extraction.

The real study reads the 2.5 GB Zenodo record, which is not bundled; these
tests drive the helpers with a simulated two-mode structure whose natural
frequencies are known.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy", reason="tools/hbta_study.py is numpy based")
signal = pytest.importorskip("scipy.signal", reason="the H1 estimate uses scipy.signal")

from tools import hbta_study as study

MODES_HZ = (4.0, 9.5)


def _simulated_test(duration_s: float = 300.0, seed: int = 0):
    """White-noise shaker input and three channels responding through two lightly damped modes."""
    rng = np.random.default_rng(seed)
    fs = study.SAMPLE_RATE_HZ
    excitation = rng.normal(size=int(duration_s * fs))
    gains = np.array([[1.0, 0.4], [0.6, 1.0], [0.3, 0.8]])
    modal = []
    for frequency in MODES_HZ:
        # Prewarped so the bilinear transform puts the digital resonance at ``frequency``.
        omega = 2.0 * fs * np.tan(np.pi * frequency / fs)
        numerator, denominator = signal.bilinear([omega**2], [1.0, 2.0 * 0.02 * omega, omega**2], fs=fs)
        modal.append(signal.lfilter(numerator, denominator, excitation))
    responses = np.column_stack(modal) @ gains.T + rng.normal(0.0, 0.01, (len(excitation), 3))
    return excitation, responses


def test_windows_skip_the_edges_and_do_not_overlap():
    spans = study.windows(int(200 * study.SAMPLE_RATE_HZ))
    assert spans[0].start == int(study.EDGE_TRIM_S * study.SAMPLE_RATE_HZ)
    assert all(left.stop == right.start for left, right in zip(spans, spans[1:]))
    assert spans[-1].stop <= int((200 - study.EDGE_TRIM_S) * study.SAMPLE_RATE_HZ)
    assert len(spans) == 3


def test_steady_windows_drop_the_shaker_ramp_up():
    excitation, _ = _simulated_test(duration_s=400.0)
    ramp = np.minimum(np.arange(len(excitation)) / (100.0 * study.SAMPLE_RATE_HZ), 1.0)
    spans = study.steady_windows(excitation * ramp)
    all_spans = study.windows(len(excitation))
    assert spans[0].start > all_spans[0].start
    assert spans[-1] == all_spans[-1]


def test_reference_modes_and_peak_frequencies_recover_the_simulated_modes():
    excitation, responses = _simulated_test()
    frequencies, indicator = study.frf_indicator(excitation, responses)
    bands = study.reference_modes(frequencies, indicator)
    found = [study.peak_frequency(frequencies, indicator, band) for band in bands]
    for mode in MODES_HZ:
        assert min(abs(value - mode) for value in found) < 0.03

    per_window = study.window_frequencies(excitation, responses, bands)
    assert per_window.shape == (len(study.steady_windows(excitation)), len(bands))
    closest = [int(np.argmin([abs(value - mode) for value in found])) for mode in MODES_HZ]
    np.testing.assert_allclose(np.median(per_window[:, closest], axis=0), MODES_HZ, atol=0.05)


def test_peak_frequency_interpolates_between_bins():
    frequencies = np.arange(0.0, 10.0, 0.1)
    true_peak = 5.03
    indicator = np.exp(-((frequencies - true_peak) ** 2) / 0.02)
    assert study.peak_frequency(frequencies, indicator, (4.0, 6.0)) == pytest.approx(true_peak, abs=1e-6)


def test_load_recording_checks_every_cached_array(tmp_path):
    import hashlib
    import json

    rng = np.random.default_rng(1)
    arrays = {
        f"{study.REFERENCE}/{sensor}/{direction}": rng.normal(size=50)
        for sensor, direction in (study.INPUT_CHANNEL, *study.OUTPUT_CHANNELS)
    }
    manifest = {
        "attributes": {study.REFERENCE: {"date": "2020-09-22"}},
        "sha256": {name: hashlib.sha256(values.tobytes()).hexdigest() for name, values in arrays.items()},
    }
    cache = tmp_path / "cache.npz"
    np.savez(cache, manifest=np.array(json.dumps(manifest)), **arrays)
    with np.load(cache) as loaded:
        excitation, responses, attributes = study.load_recording(loaded, study.REFERENCE)
    assert responses.shape == (50, len(study.OUTPUT_CHANNELS))
    assert attributes == {"date": "2020-09-22"}

    arrays[f"{study.REFERENCE}/AG05/z"][3] += 1.0
    np.savez(cache, manifest=np.array(json.dumps(manifest)), **arrays)
    with np.load(cache) as loaded, pytest.raises(ValueError, match="AG05/z does not match"):
        study.load_recording(loaded, study.REFERENCE)
