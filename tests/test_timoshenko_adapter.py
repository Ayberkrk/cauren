"""Tests for the optional Timoshenko engine gating in timoshenko_adapter.py.

Cover the branches that decide whether the optional engine is used so a
regression cannot silently fall back without being noticed.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cauren_physics import timoshenko_adapter


def test_engine_missing_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "timoshenko", None)
    assert timoshenko_adapter._engine() is None


def test_engine_unparseable_version(monkeypatch):
    stub = SimpleNamespace(__version__="dev")
    monkeypatch.setitem(sys.modules, "timoshenko", stub)
    assert timoshenko_adapter._engine() is None


def test_engine_version_1x(monkeypatch):
    stub = SimpleNamespace(__version__="1.4.0")
    monkeypatch.setitem(sys.modules, "timoshenko", stub)
    assert timoshenko_adapter._engine() is None


def test_engine_version_2x(monkeypatch):
    stub = SimpleNamespace(__version__="2.3.0")
    monkeypatch.setitem(sys.modules, "timoshenko", stub)
    assert timoshenko_adapter._engine() is stub


def test_identify_multichannel_too_few_channels(monkeypatch):
    monkeypatch.setattr(timoshenko_adapter, "_engine", lambda: SimpleNamespace())
    with pytest.raises(ValueError, match="at least two channels"):
        timoshenko_adapter.identify_multichannel(
            [[1.0, 2.0]], 100.0, channel_ids=["ch1"], max_modes=3, min_prominence_ratio=0.1
        )


def test_identify_multichannel_channel_id_mismatch(monkeypatch):
    monkeypatch.setattr(timoshenko_adapter, "_engine", lambda: SimpleNamespace())
    with pytest.raises(ValueError, match="one channel id per series"):
        timoshenko_adapter.identify_multichannel(
            [[1.0], [2.0]], 100.0, channel_ids=["only"], max_modes=3, min_prominence_ratio=0.1
        )


def test_identify_multichannel_unequal_lengths(monkeypatch):
    monkeypatch.setattr(timoshenko_adapter, "_engine", lambda: SimpleNamespace())
    with pytest.raises(ValueError, match="same number of samples"):
        timoshenko_adapter.identify_multichannel(
            [[1.0, 2.0], [3.0]], 100.0, channel_ids=["a", "b"], max_modes=3, min_prominence_ratio=0.1
        )


def test_identify_multichannel_nonpositive_sampling(monkeypatch):
    monkeypatch.setattr(timoshenko_adapter, "_engine", lambda: SimpleNamespace())
    with pytest.raises(ValueError, match="sampling_hz must be positive"):
        timoshenko_adapter.identify_multichannel(
            [[1.0, 2.0], [3.0, 4.0]], 0.0, channel_ids=["a", "b"], max_modes=3, min_prominence_ratio=0.1
        )


def test_pair_modes_returns_none_without_engine(monkeypatch):
    monkeypatch.setattr(timoshenko_adapter, "_engine", lambda: None)
    assert timoshenko_adapter.pair_modes([1.0, 2.0], [1.1, 2.1]) is None


def test_shear_building_returns_none_without_engine(monkeypatch):
    monkeypatch.setattr(timoshenko_adapter, "_engine", lambda: None)
    assert timoshenko_adapter.shear_building_frequencies([50.0], [1e5]) is None


def test_update_scale_returns_none_without_frequencies(monkeypatch):
    monkeypatch.setattr(timoshenko_adapter, "_engine", lambda: None)
    assert timoshenko_adapter.update_scale_summary([50.0], [1e5], []) is None


def test_pair_modes_success(monkeypatch):
    mock_tm = MagicMock()
    mock_tm.modal.pair_modes.return_value = [(0, 0)]
    monkeypatch.setattr(timoshenko_adapter, "_engine", lambda: mock_tm)
    result = timoshenko_adapter.pair_modes([1.0], [1.05])
    assert result == ((0, 0),)
    mock_tm.modal.pair_modes.assert_called_once()


def test_shear_building_success(monkeypatch):
    mock_tm = MagicMock()
    mock_tm.Structure.return_value.natural_frequencies_hz = (1.2, 3.4)
    monkeypatch.setattr(timoshenko_adapter, "_engine", lambda: mock_tm)
    result = timoshenko_adapter.shear_building_frequencies([100.0], [1e6])
    assert result == (1.2, 3.4)


def test_update_scale_success(monkeypatch):
    mock_tm = MagicMock()
    mock_updated = MagicMock()
    mock_updated.update_scale_factor = 1.05
    mock_updated.update_mode_scale_spread_pct = 2.0
    mock_tm.update.return_value = mock_updated
    monkeypatch.setattr(timoshenko_adapter, "_engine", lambda: mock_tm)
    result = timoshenko_adapter.update_scale_summary([100.0], [1e6], [1.2])
    assert result == (1.05, 2.0)
