import numpy as np
import pandas as pd
import pytest

from tools.bridge_hazard import (
    MONOTONE_INCREASING,
    cumulative_risk,
    feature_columns,
    monotone_vector,
    risk_by_horizon,
    roll_forward,
)
from tools.build_bridge_years_v4 import CATEGORICAL_FEATURES, MODEL_FEATURES


def _anchor(**overrides):
    row = {column: np.float32(1.0) for column in MODEL_FEATURES}
    row.update({column: "missing" for column in CATEGORICAL_FEATURES})
    row.update({
        "deck_cond": 7.0, "deck_prev": 8.0, "deck_delta": -1.0, "age_years": 40.0,
        "years_since_reconstruction": 40.0, "years_at_deck_rating": 1.0,
        "years_since_inspection": 0.5, "inspect_freq_months": 24.0, "inspected_this_year": 1.0,
    })
    row.update(overrides)
    return pd.DataFrame([row])


class _InspectionHazard:
    """Hazard that is non-zero only in years with an inspection."""

    def predict(self, frame):
        return np.where(frame["inspected_this_year"].to_numpy() > 0, 0.2, 0.0)


def test_roll_forward_advances_clock_without_changing_rating():
    rolled = roll_forward(_anchor(), 3)
    assert rolled.loc[0, "age_years"] == 43
    assert rolled.loc[0, "years_at_deck_rating"] == 4
    assert rolled.loc[0, "deck_prev"] == 7.0
    assert rolled.loc[0, "deck_delta"] == 0.0
    assert rolled.loc[0, "deck_cond"] == 7.0


def test_roll_forward_follows_the_inspection_schedule():
    flags = [roll_forward(_anchor(), step).loc[0, "inspected_this_year"] for step in range(1, 6)]
    # Inspected half a year ago on a 24-month cycle: next inspections fall in
    # roll-forward years 2 and 4.
    assert flags == [0.0, 1.0, 0.0, 1.0, 0.0]
    assert roll_forward(_anchor(), 2).loc[0, "years_since_inspection"] == pytest.approx(0.5)


def test_cumulative_risk_is_monotone_in_horizon():
    anchors = pd.concat([_anchor(), _anchor(years_since_inspection=1.5)], ignore_index=True)
    curves = risk_by_horizon(_InspectionHazard(), anchors)
    assert np.all(np.diff(curves, axis=1) >= 0)
    assert cumulative_risk(_InspectionHazard(), anchors) == pytest.approx(curves[:, -1])
    # The anchor row counts as step 0, so its own fresh inspection is one of
    # the three inspected years for the first bridge.
    assert curves[0, -1] == pytest.approx(1 - 0.8 ** 3)


def test_monotone_constraints_cover_only_declared_covariates():
    columns = feature_columns()
    vector = monotone_vector(columns)
    assert {column for column, sign in zip(columns, vector) if sign} == set(MONOTONE_INCREASING)
    assert "deck_cond" not in MONOTONE_INCREASING
