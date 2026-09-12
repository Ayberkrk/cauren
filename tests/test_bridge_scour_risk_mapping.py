"""Item 113 (scour-criticality) -> ground_stability_score direction.

Requires pandas (only used by tools/build_cauren_bridge_dataset.py, not the
core runtime), so this file skips itself when pandas isn't installed rather
than adding a hard pandas dependency to the main test suite.
"""

import importlib
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
build_cauren_bridge_dataset = importlib.import_module("tools.build_cauren_bridge_dataset")

scour_risk_from_item_113 = build_cauren_bridge_dataset.scour_risk_from_item_113
ITEM_113_TIDAL_UNEVALUATED_RISK = build_cauren_bridge_dataset.ITEM_113_TIDAL_UNEVALUATED_RISK


def _risk_for(code) -> float | None:
    result = scour_risk_from_item_113(pd.Series([code]))
    value = result.iloc[0]
    return None if pd.isna(value) else float(value)


def test_stable_code_scores_lower_than_scour_critical_codes():
    # Code 8 (stable) must not outrank code 3 (scour critical) -- this is
    # exactly the direction bug the linear code/9.0 conversion had.
    assert _risk_for(8) < _risk_for(3)
    assert _risk_for(9) < _risk_for(8)


def test_scour_critical_codes_are_monotonically_worse_toward_failure():
    assert _risk_for(3) < _risk_for(2) < _risk_for(1) < _risk_for(0)
    assert _risk_for(0) == 1.0


def test_stable_and_remediated_codes_score_low_risk():
    for code in (4, 5, 7, 8, 9):
        assert _risk_for(code) is not None
        assert _risk_for(code) <= 0.10


def test_not_over_waterway_and_unknown_are_left_missing():
    assert _risk_for("N") is None
    assert _risk_for("U") is None


def test_unusual_code_six_is_left_missing_not_guessed():
    assert _risk_for(6) is None


def test_tidal_unevaluated_gets_explicit_low_risk_not_missing():
    assert _risk_for("T") == ITEM_113_TIDAL_UNEVALUATED_RISK


def test_out_of_range_and_blank_codes_are_missing():
    assert _risk_for(15) is None
    assert _risk_for("") is None
