import math

from cauren_physics.oma import compare_to_baseline, identify_modal_parameters


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


def test_identify_modal_parameters_estimates_positive_damping_for_decaying_mode():
    sampling_hz = 100.0
    signal = _synthetic_signal([5.0], sampling_hz, duration_s=20.0, damping=0.02)

    result = identify_modal_parameters(signal, sampling_hz)

    assert result.modal_parameters
    dominant_mode = max(result.modal_parameters, key=lambda mode: mode.amplitude)
    assert dominant_mode.damping_ratio is not None
    assert dominant_mode.damping_ratio > 0.0


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
