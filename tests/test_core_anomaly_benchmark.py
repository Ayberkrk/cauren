import pytest

np = pytest.importorskip("numpy")

from tools import benchmark_core_anomaly as bench


def _injected(name: str, seed: int = 11) -> np.ndarray:
    """What a scenario added on top of the noise it drew, in sigma units."""
    builder, _ = bench.SCENARIOS[name]
    changed = builder(np.random.default_rng(seed))
    clean = bench._noise(np.random.default_rng(seed))
    scales = np.array([bench.REFERENCE[feature].scale for feature in bench.FEATURES])
    return (changed - clean) / scales


def _changed_columns(delta: np.ndarray) -> list[str]:
    return [bench.FEATURES[col] for col in range(delta.shape[1]) if np.any(np.abs(delta[:, col]) > 1e-9)]


def test_schema_matches_the_civil_agent_required_features():
    from cauren_agents.civil.agent import build_civil_agent

    assert bench.FEATURES == tuple(build_civil_agent().schema.required_features)
    assert bench.SEQ_LEN == 16


def test_each_change_scenario_touches_exactly_one_feature_in_sigma_units():
    step = _injected("step_3sigma")
    assert _changed_columns(step) == [bench.TARGET_FEATURE]
    column = step[:, bench.FEATURES.index(bench.TARGET_FEATURE)]
    assert np.allclose(column[: bench.SEQ_LEN // 2], 0.0)
    assert np.allclose(column[bench.SEQ_LEN // 2 :], 3.0)

    drift = _injected("drift_3sigma")[:, bench.FEATURES.index(bench.TARGET_FEATURE)]
    assert drift[0] == pytest.approx(0.0) and drift[-1] == pytest.approx(3.0)
    assert np.allclose(np.diff(drift), 3.0 / (bench.SEQ_LEN - 1))

    spike = _injected("spike_6sigma")
    assert _changed_columns(spike) == [bench.TARGET_FEATURE]
    assert sorted(np.round(spike[:, bench.FEATURES.index(bench.TARGET_FEATURE)], 9))[-2:] == [0.0, 6.0]

    assert not _changed_columns(_injected("clean_noise"))


def test_constant_scenarios_freeze_the_intended_features_at_their_reference():
    builder, _ = bench.SCENARIOS["constant_naturally_constant"]
    matrix = builder(np.random.default_rng(3))
    column = matrix[:, bench.FEATURES.index(bench.NATURALLY_CONSTANT_FEATURE)]
    assert np.all(column == bench.REFERENCE[bench.NATURALLY_CONSTANT_FEATURE].center)
    assert bench.REFERENCE[bench.NATURALLY_CONSTANT_FEATURE].expected_variation is False
    assert bench.REFERENCE[bench.TARGET_FEATURE].expected_variation is True

    frozen, _ = bench.SCENARIOS["constant_all_features"]
    assert np.all(np.ptp(frozen(np.random.default_rng(3)), axis=0) == 0.0)


def test_payload_carries_every_reading_and_the_reference():
    matrix = bench._clean(np.random.default_rng(0))
    payload = bench.build_payload(matrix)
    assert len(payload["sensors"]) == bench.SEQ_LEN * len(bench.FEATURES)
    assert set(payload["baseline"]) == set(bench.FEATURES)
    assert payload["baseline"][bench.TARGET_FEATURE]["scale"] > 0.0
    assert "baseline" not in bench.build_payload(matrix, include_reference=False)


def test_wilson_interval_matches_reference_values():
    low, high = bench.wilson_interval(10, 100)
    assert low == pytest.approx(0.0552, abs=1e-4)
    assert high == pytest.approx(0.1744, abs=1e-4)
    low, high = bench.wilson_interval(0, 500)
    assert low == pytest.approx(0.0, abs=1e-12)
    assert high == pytest.approx(0.00762, abs=1e-4)


def test_runs_are_reproducible_for_a_fixed_seed():
    first = bench.run_scenario("step_3sigma", trials=20, seed=5)
    second = bench.run_scenario("step_3sigma", trials=20, seed=5)
    assert first == second
    assert first["trials"] == 20 and 0 <= first["alarms"] <= 20
