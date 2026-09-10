import math

from cauren_physics.fe_reference_model import ShearBuildingModel, natural_frequencies_hz
from cauren_physics.oma import compare_to_baseline, identify_modal_parameters


def _synthetic_signal(frequencies_hz, sampling_hz, duration_s):
    n = int(sampling_hz * duration_s)
    dt = 1.0 / sampling_hz
    values = []
    for i in range(n):
        t = i * dt
        values.append(sum(math.sin(2.0 * math.pi * freq * t) for freq in frequencies_hz))
    return values


def test_single_story_model_matches_closed_form_natural_frequency():
    # f = (1/2pi) * sqrt(k/m), the textbook single-DOF result -- checks the
    # Jacobi eigensolver against a formula with no numerics involved.
    mass_kg = 50_000.0
    stiffness_n_per_m = 8.0e6
    model = ShearBuildingModel(story_masses_kg=(mass_kg,), story_stiffness_n_per_m=(stiffness_n_per_m,))

    expected_hz = (1.0 / (2.0 * math.pi)) * math.sqrt(stiffness_n_per_m / mass_kg)
    (identified_hz,) = natural_frequencies_hz(model)

    assert math.isclose(identified_hz, expected_hz, rel_tol=1e-6)


def test_two_story_uniform_model_matches_closed_form_frequencies():
    # For a uniform 2-story shear building (equal mass m, equal story
    # stiffness k), the eigenvalues of K/m have the closed form
    # x = (3 +/- sqrt(5)) / 2, omega^2 = (k/m) * x. See Chopra Ch. 12.
    mass_kg = 10_000.0
    stiffness_n_per_m = 4.0e6
    model = ShearBuildingModel(
        story_masses_kg=(mass_kg, mass_kg),
        story_stiffness_n_per_m=(stiffness_n_per_m, stiffness_n_per_m),
    )

    ratio = stiffness_n_per_m / mass_kg
    x1 = (3.0 - math.sqrt(5.0)) / 2.0
    x2 = (3.0 + math.sqrt(5.0)) / 2.0
    expected = sorted(
        math.sqrt(ratio * x) / (2.0 * math.pi) for x in (x1, x2)
    )

    identified = natural_frequencies_hz(model)

    assert len(identified) == 2
    for actual, expected_hz in zip(identified, expected):
        assert math.isclose(actual, expected_hz, rel_tol=1e-6)


def test_natural_frequencies_are_returned_ascending():
    model = ShearBuildingModel(
        story_masses_kg=(30_000.0, 25_000.0, 20_000.0),
        story_stiffness_n_per_m=(6.0e6, 5.0e6, 4.0e6),
    )

    frequencies = natural_frequencies_hz(model)

    assert len(frequencies) == 3
    assert list(frequencies) == sorted(frequencies)
    assert all(f > 0.0 for f in frequencies)


def test_mass_and_stiffness_matrices_match_the_textbook_assembly():
    """The two matrices the eigenproblem is built from, checked directly:
    M is diagonal with the story masses, K is the tridiagonal shear-building
    assembly where each floor carries its own story stiffness plus the one
    above, coupled by -k to its neighbours.
    """
    model = ShearBuildingModel(
        story_masses_kg=(100.0, 200.0),
        story_stiffness_n_per_m=(10.0, 20.0),
    )

    assert model.num_stories == 2
    assert model.mass_matrix() == [[100.0, 0.0], [0.0, 200.0]]
    assert model.stiffness_matrix() == [[30.0, -20.0], [-20.0, 20.0]]


def test_stiffness_matrix_is_symmetric_for_a_taller_model():
    model = ShearBuildingModel(
        story_masses_kg=(30_000.0, 25_000.0, 20_000.0),
        story_stiffness_n_per_m=(6.0e6, 5.0e6, 4.0e6),
    )
    stiffness = model.stiffness_matrix()

    for i in range(model.num_stories):
        for j in range(model.num_stories):
            assert stiffness[i][j] == stiffness[j][i]


def test_shear_building_model_rejects_non_positive_values():
    for kwargs in (
        {"story_masses_kg": (0.0,), "story_stiffness_n_per_m": (1.0,)},
        {"story_masses_kg": (1.0,), "story_stiffness_n_per_m": (-1.0,)},
        {"story_masses_kg": (), "story_stiffness_n_per_m": ()},
    ):
        try:
            ShearBuildingModel(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


def test_shear_building_model_rejects_mismatched_lengths():
    try:
        ShearBuildingModel(story_masses_kg=(1.0, 2.0), story_stiffness_n_per_m=(1.0,))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched story arrays")


def test_oma_identification_agrees_with_fe_reference_model():
    """The consistency test proper: a structure's data-driven (OMA) natural
    frequency should agree with its physics/FE reference model, when the
    vibration data actually comes from that structure and nothing has
    changed (no damage, no stiffness loss).
    """
    mass_kg = 50_000.0
    stiffness_n_per_m = 8.0e6
    model = ShearBuildingModel(story_masses_kg=(mass_kg,), story_stiffness_n_per_m=(stiffness_n_per_m,))
    (theoretical_hz,) = natural_frequencies_hz(model)

    sampling_hz = 100.0
    signal = _synthetic_signal([theoretical_hz], sampling_hz, duration_s=20.0)
    identified = identify_modal_parameters(signal, sampling_hz)

    consistency = compare_to_baseline(identified, [theoretical_hz], warn_pct=5.0, alarm_pct=10.0)

    # A tiny gap is expected: OMA can only resolve frequency to the
    # spectrum's bin width (identified.frequency_resolution_hz), so it
    # won't land exactly on the FE-theoretical value. What matters for
    # consistency is that the gap stays well below the warn threshold.
    assert consistency.overall_severity in {"none", "watch"}
    assert consistency.findings[0].current_frequency_hz is not None
    assert abs(consistency.findings[0].drift_pct) < 5.0


def test_oma_identification_flags_inconsistency_when_stiffness_drops():
    """If the structure has actually lost stiffness (damage) since the FE
    reference model was built, the theory (FE) and the data (OMA) should
    now visibly disagree -- that disagreement is the whole point of running
    this consistency check periodically.
    """
    mass_kg = 50_000.0
    stiffness_n_per_m = 8.0e6
    baseline_model = ShearBuildingModel(story_masses_kg=(mass_kg,), story_stiffness_n_per_m=(stiffness_n_per_m,))
    (baseline_hz,) = natural_frequencies_hz(baseline_model)

    degraded_model = ShearBuildingModel(
        story_masses_kg=(mass_kg,),
        story_stiffness_n_per_m=(stiffness_n_per_m * 0.7,),  # 30% stiffness loss
    )
    (degraded_hz,) = natural_frequencies_hz(degraded_model)

    sampling_hz = 100.0
    signal = _synthetic_signal([degraded_hz], sampling_hz, duration_s=20.0)
    identified = identify_modal_parameters(signal, sampling_hz)

    consistency = compare_to_baseline(identified, [baseline_hz], warn_pct=5.0, alarm_pct=10.0)

    assert consistency.overall_severity in {"warn", "alarm"}
    assert consistency.findings[0].drift_pct > 5.0
