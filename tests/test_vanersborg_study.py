"""Synthetic checks for the Vänersborg opening features.

The real study streams 64 openings from Zenodo; these tests drive the
feature extraction with a simulated opening whose strains are known.
"""

from __future__ import annotations

import datetime as dt

import pytest

np = pytest.importorskip("numpy", reason="tools/vanersborg_study.py is numpy based")

from tools import vanersborg_study as study


def _opening(peak_deg: float = -74.0):
    """Closed at -1.1 deg for 20 s, opening to ``peak_deg`` and closing again, at 200 Hz."""
    rate = int(study.SAMPLE_RATE_HZ)
    closed = np.full(20 * rate, -1.1)
    opening = np.linspace(-1.1, peak_deg, 60 * rate)
    inclination = np.concatenate((closed, opening, opening[::-1], closed))
    # Gauge 1 follows the angle linearly, gauge 2 not at all, gauge 3 with an offset.
    strain = np.column_stack((10.0 * inclination, np.full_like(inclination, 5.0), 400.0 - 2.0 * inclination))
    return strain, inclination


def test_opening_features_are_strain_changes_from_the_closed_position():
    strain, inclination = _opening()
    features = study.opening_features(strain, inclination)
    expected = []
    for angle in study.ANGLES_DEG:
        change = angle + 1.1
        expected.extend([10.0 * change, 0.0, -2.0 * change])
    np.testing.assert_allclose(features, expected, atol=0.2)


def test_opening_features_reject_a_partial_opening():
    strain, inclination = _opening(peak_deg=-45.0)
    assert study.opening_features(strain, inclination) is None


def test_event_time_reads_the_file_name():
    assert study.event_time("2023_03_09T23_45_25.csv") == dt.datetime(2023, 3, 9, 23, 45, 25)
    assert study.event_time("2023_03_09T23_45_25.csv") == study.FRACTURE_EVENT


def test_erratic_gauges_flags_a_heavy_tailed_channel_only():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(40, 3))
    reference[::8, 1] += 200.0
    names = [("SG1", -30.0), ("SG11", -30.0), ("SG13", -30.0)]
    assert study.erratic_gauges(reference, names) == ["SG11"]
