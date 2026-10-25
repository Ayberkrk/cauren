from cauren_core.contracts import FeatureMetadata, SensorWindow
from cauren_core.runtime import _seasonal_deviation_score


def _window(feature_name: str) -> SensorWindow:
    return SensorWindow(
        matrix=((10.0,), (20.0,), (40.0,)),
        features=(FeatureMetadata(name=feature_name, unit="ratio"),),
        presence_mask=((True,), (True,), (True,)),
    )


def test_season_name_does_not_apply_hvac_feature_name_heuristic():
    assert _seasonal_deviation_score(_window("outdoor_temperature"), {"season": "winter"}) == 0.0
    assert _seasonal_deviation_score(_window("indoor_temperature"), {"season": "summer"}) == 0.0


def test_unverified_season_alert_does_not_create_fixed_anomaly_score():
    assert _seasonal_deviation_score(_window("structural_risk_score"), {"seasonal_alert": True}) == 0.0
