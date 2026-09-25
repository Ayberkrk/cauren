import math
import json
import sys
from pathlib import Path

import pytest

from cauren_physics import oma
from cauren_physics.oma import OMAResult, ModalParameter, compare_to_baseline, identify_modal_parameters
from tools import run_oma_identification


def _synthetic_signal(frequencies_hz, sampling_hz, duration_s, *, damping=0.0, noise_amplitude=0.0):
    n = int(sampling_hz * duration_s)
    dt = 1.0 / sampling_hz
    values = []
    # Deterministic pseudo-noise so the test has no flaky randomness.
    seed = 12345
    for i in range(n):
        t = i * dt
        sample = 0.0
        for freq in frequencies_hz:
            sample += math.exp(-damping * 2.0 * math.pi * freq * t) * math.sin(2.0 * math.pi * freq * t)
        if noise_amplitude:
            seed = (1103515245 * seed + 12345) % (2**31)
            noise = (seed / float(2**31) - 0.5) * 2.0 * noise_amplitude
            sample += noise
        values.append(sample)
    return values


def test_identify_modal_parameters_recovers_dominant_frequency():
    sampling_hz = 100.0
    signal = _synthetic_signal([3.0], sampling_hz, duration_s=20.0)

    result = identify_modal_parameters(signal, sampling_hz)

    assert result.modal_parameters
    dominant = result.dominant_frequency_hz
    assert dominant is not None
    assert abs(dominant - 3.0) <= result.frequency_resolution_hz + 1e-6


def test_identify_modal_parameters_recovers_multiple_modes():
    sampling_hz = 100.0
    signal = _synthetic_signal([2.0, 7.5], sampling_hz, duration_s=20.0)

    result = identify_modal_parameters(signal, sampling_hz, max_modes=3)

    frequencies = sorted(mode.frequency_hz for mode in result.modal_parameters)
    assert len(frequencies) >= 2
    assert abs(frequencies[0] - 2.0) <= result.frequency_resolution_hz + 0.5
    assert abs(frequencies[-1] - 7.5) <= result.frequency_resolution_hz + 0.5


def test_identify_modal_parameters_returns_unresolved_damping_for_short_decaying_record():
    sampling_hz = 100.0
    signal = _synthetic_signal([5.0], sampling_hz, duration_s=20.0, damping=0.02)

    result = identify_modal_parameters(signal, sampling_hz)

    assert result.modal_parameters
    dominant_mode = max(result.modal_parameters, key=lambda mode: mode.amplitude)
    assert dominant_mode.damping_ratio is None


@pytest.mark.parametrize("sample_count", [1000, 100000])
def test_undamped_sinusoid_does_not_report_window_width_as_damping(sample_count, monkeypatch):
    monkeypatch.setattr(oma, "_np", None)
    sampling_hz = 100.0
    signal = [math.sin(2.0 * math.pi * 5.0 * index / sampling_hz) for index in range(sample_count)]

    result = oma._identify_modal_parameters_fallback(signal, sampling_hz)

    assert result.modal_parameters
    assert max(result.modal_parameters, key=lambda mode: mode.amplitude).damping_ratio is None


def test_compare_to_baseline_does_not_reuse_an_observed_mode_for_two_references():
    signal = _synthetic_signal([3.0], 100.0, duration_s=60.0)
    current = identify_modal_parameters(signal, 100.0)

    report = compare_to_baseline(current, [3.0, 4.2])

    assert report.overall_severity == "watch"
    assert report.findings[0].severity == "none"
    assert report.findings[1].current_frequency_hz is None
    assert report.max_drop_pct == 0.0


def test_compare_to_baseline_ignores_drift_within_frequency_resolution():
    current = identify_modal_parameters(_synthetic_signal([3.0], 100.0, duration_s=20.03), 100.0)

    report = compare_to_baseline(current, [3.0])

    assert report.overall_severity == "none"
    assert report.findings[0].resolution_limited is True
    assert report.findings[0].drift_pct is not None
    assert report.max_drop_pct == 0.0


def test_compare_to_baseline_keeps_only_the_closest_observation_per_reference():
    current = OMAResult(
        modal_parameters=(
            ModalParameter(3.1, None, 2.0, 1.0),
            ModalParameter(3.01, None, 1.0, 0.5),
        ),
        sampling_hz=100.0,
        num_samples=1000,
        method="peak_picking",
        frequency_resolution_hz=0.001,
    )

    report = compare_to_baseline(current, [3.0])

    assert report.findings[0].current_frequency_hz == 3.01


def test_pair_modes_matches_timoshenko_for_valid_and_invalid_frequencies():
    tm = pytest.importorskip("timoshenko")

    cases = [
        ([3.0, 4.2], [3.0]),
        ([1.0, 2.0, 8.0], [0.0, 1.1, 2.1, 7.8]),
        ([2.0, 5.0], [2.01, 2.2, 5.1]),
    ]
    for references, observed in cases:
        assert oma._pair_modes(references, observed) == tm.modal.pair_modes(references, observed)


def test_fallback_damping_matches_timoshenko_for_same_record():
    tm = pytest.importorskip("timoshenko")
    signal = _synthetic_signal([2.0], 50.0, duration_s=120.0, damping=0.02, noise_amplitude=0.1)

    fallback = oma._identify_modal_parameters_fallback(signal, 50.0)
    shared = oma._identify_with_timoshenko(signal, 50.0, max_modes=3, min_prominence_ratio=0.15)

    assert shared is not None
    fallback_mode = max(fallback.modal_parameters, key=lambda mode: mode.amplitude)
    shared_mode = max(shared.modal_parameters, key=lambda mode: mode.amplitude)
    assert fallback_mode.damping_ratio == pytest.approx(shared_mode.damping_ratio)


def test_welch_damping_estimates_noise_driven_single_degree_system():
    np = pytest.importorskip("numpy")
    sampling_hz = 50.0
    natural_hz = 2.0
    damping_ratio = 0.02
    duration_s = 600.0
    dt = 1.0 / sampling_hz
    substeps = 10
    sub_dt = dt / substeps
    omega = 2.0 * math.pi * natural_hz
    stiffness = omega * omega
    damping = 2.0 * damping_ratio * omega
    rng = np.random.default_rng(12345)
    force = rng.standard_normal(int(sampling_hz * duration_s))
    displacement = 0.0
    velocity = 0.0
    values = []
    for sample_force in force:
        for _ in range(substeps):
            acceleration = sample_force - damping * velocity - stiffness * displacement
            velocity += sub_dt * acceleration
            displacement += sub_dt * velocity
        values.append(displacement)

    result = oma._identify_modal_parameters_fallback(values, sampling_hz)
    identified = max(result.modal_parameters, key=lambda mode: mode.amplitude)

    assert identified.damping_ratio is not None
    assert 0.01 <= identified.damping_ratio <= 0.04


def test_identify_modal_parameters_returns_empty_for_too_short_series():
    result = identify_modal_parameters([1.0, 2.0, 3.0], sampling_hz=10.0)
    assert result.modal_parameters == ()
    assert result.num_samples == 3


def test_identify_modal_parameters_rejects_non_positive_sampling_rate():
    try:
        identify_modal_parameters([0.0] * 32, sampling_hz=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-positive sampling_hz")


def test_compare_to_baseline_flags_frequency_drop_as_alarm():
    sampling_hz = 100.0
    # Baseline was 5.0 Hz; the structure now reads meaningfully lower,
    # simulating stiffness loss.
    signal = _synthetic_signal([4.4], sampling_hz, duration_s=20.0)
    current = identify_modal_parameters(signal, sampling_hz)

    report = compare_to_baseline(current, [5.0], warn_pct=5.0, alarm_pct=10.0)

    assert report.overall_severity == "alarm"
    assert report.findings[0].drift_pct > 10.0


def test_compare_to_baseline_reports_none_when_frequency_is_stable():
    sampling_hz = 100.0
    signal = _synthetic_signal([5.0], sampling_hz, duration_s=20.0)
    current = identify_modal_parameters(signal, sampling_hz)

    report = compare_to_baseline(current, [5.0], warn_pct=5.0, alarm_pct=10.0)

    assert report.overall_severity == "none"


def test_compare_to_baseline_flags_missing_mode_as_watch():
    sampling_hz = 100.0
    signal = _synthetic_signal([3.0], sampling_hz, duration_s=20.0)
    current = identify_modal_parameters(signal, sampling_hz)

    report = compare_to_baseline(current, [50.0])

    assert report.findings[0].severity == "watch"
    assert report.findings[0].current_frequency_hz is None


def test_oma_result_to_dict_roundtrip():
    sampling_hz = 100.0
    signal = _synthetic_signal([3.0], sampling_hz, duration_s=20.0)
    result = identify_modal_parameters(signal, sampling_hz)
    payload = result.to_dict()

    assert payload["method"] == "peak_picking"
    assert payload["sampling_hz"] == 100.0
    assert isinstance(payload["modal_parameters"], list)
    assert "mode_shapes" not in payload


def test_multichannel_fdd_recovers_two_mode_shapes():
    np = pytest.importorskip("numpy")
    pytest.importorskip("timoshenko")
    from cauren_physics.timoshenko_adapter import identify_multichannel

    sampling_hz = 100.0
    sample_count = 8192
    time = np.arange(sample_count, dtype=float) / sampling_hz
    shape_a = np.asarray([1.0, 0.8, 0.3])
    shape_b = np.asarray([0.5, -0.4, 1.0])
    first = np.sin(2.0 * math.pi * 3.1 * time)
    second = np.sin(2.0 * math.pi * 9.7 * time)
    data = shape_a[:, None] * first[None, :] + shape_b[:, None] * second[None, :]

    result = identify_multichannel(
        data.tolist(),
        sampling_hz,
        channel_ids=["a", "b", "c"],
        max_modes=4,
        min_prominence_ratio=0.05,
    )

    assert result.method == "fdd"
    assert result.mode_shapes
    assert len(result.mode_shapes) >= 2
    identified = sorted(result.mode_shapes, key=lambda mode: mode["frequency_hz"])
    assert abs(identified[0]["frequency_hz"] - 3.1) <= result.frequency_resolution_hz
    assert abs(identified[1]["frequency_hz"] - 9.7) <= result.frequency_resolution_hz
    expected_shapes = (shape_a, shape_b)
    for mode, expected in zip(identified[:2], expected_shapes):
        actual = np.asarray(
            [complex(mode["shape"][channel]["real"], mode["shape"][channel]["imag"]) for channel in ("a", "b", "c")]
        )
        mac = abs(np.vdot(expected, actual)) ** 2 / (np.vdot(expected, expected).real * np.vdot(actual, actual).real)
        assert mac > 0.99


def test_timoshenko_adapter_rejects_legacy_major_version(monkeypatch):
    import sys
    from types import SimpleNamespace

    from cauren_physics import timoshenko_adapter

    monkeypatch.setitem(sys.modules, "timoshenko", SimpleNamespace(__version__="1.9.9"))

    assert timoshenko_adapter._engine() is None


def test_oma_tool_reports_fe_consistency_separately_from_measured_baseline(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "signal.csv"
    signal = _synthetic_signal([1.8], 100.0, duration_s=60.0)
    input_path.write_text("value\n" + "\n".join(map(str, signal)) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_oma_identification.py",
            "--input-csv",
            str(input_path),
            "--sampling-hz",
            "100",
            "--baseline-frequencies-hz",
            "2.0",
            "--fe-story-masses-kg",
            "50000",
            "--fe-story-stiffness-n-per-m",
            "8000000",
        ],
    )

    run_oma_identification.main()
    report = json.loads(capsys.readouterr().out)

    assert "oma_frequency_drift" in report
    assert "fe_model_consistency" in report
    assert report["fe_model_consistency"]["implied_uniform_stiffness_ratio"] is not None


def test_oma_tool_reports_clear_error_when_fdd_engine_is_missing(tmp_path, monkeypatch):
    input_path = tmp_path / "channels.csv"
    input_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    monkeypatch.setattr(run_oma_identification, "identify_multichannel", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_oma_identification.py", "--input-csv", str(input_path), "--sampling-hz", "100", "--columns", "a", "b"],
    )

    with pytest.raises(SystemExit, match="multi-channel FDD requires timoshenko-engine 2.0 or newer"):
        run_oma_identification.main()
