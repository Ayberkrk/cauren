"""Synthetic checks for the KW51-specific parts of the case study.

The real study reads the Zenodo KW51 record, which is not bundled. The shared
detectors and protocol are tested in tests/test_novelty.py.
"""

from __future__ import annotations

import datetime as dt

import pytest

np = pytest.importorskip("numpy", reason="tools/kw51_study.py is numpy based")

from tools import kw51_study as study


def test_datenum_conversion_matches_matlab_epoch():
    # datenum(2018, 10, 1) is 737334 in MATLAB; 0.5 is noon.
    times = study.datenum_to_datetime([737334.0, 737334.5, 737334.0 + 23 / 24 - 1e-9])
    assert times == [dt.datetime(2018, 10, 1), dt.datetime(2018, 10, 1, 12), dt.datetime(2018, 10, 1, 23)]


def test_kw51_protocol_segments_are_ordered_and_skip_the_retrofit_works():
    protocol = study.PROTOCOL
    assert protocol.reference[1] == protocol.holdout[0]
    assert protocol.holdout[1] == dt.datetime(2019, 5, 15)
    assert protocol.changed[0] == dt.datetime(2019, 9, 28)
