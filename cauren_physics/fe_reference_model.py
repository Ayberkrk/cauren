"""A minimal FE-equivalent structural reference model, used to sanity-check
operational modal analysis (see `cauren_physics.oma`) against theory.

Full finite-element analysis (meshing, shape functions, assembly of a real
3D model) is out of scope for this repo. What is implemented here is the
textbook idealization every FE lateral-dynamics model reduces to for a
regular multi-story building: a "shear building" -- each story is a single
lumped mass, connected to the story above/below by a single lateral
stiffness (Chopra, *Dynamics of Structures*, ch. 9). Solving its eigenvalue
problem gives the same natural frequencies a full FE modal analysis would
converge to for that idealization, at a small fraction of the cost.

The point of having this alongside `cauren_physics.oma` is a consistency
check: a structure's data-driven, sensor-identified natural frequencies
(from ambient vibration, via OMA) should agree with what its mass/stiffness
model predicts. It's a theory-vs-data cross-check, exactly the kind of
model-consistency test a physics-informed system should be able to run on
itself rather than trusting either result blindly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

from .timoshenko_adapter import (
    pair_modes as _pair_modes_with_timoshenko,
    shear_building_frequencies as _shared_shear_building_frequencies,
    update_scale_summary as _shared_update_summary,
)


@dataclass(frozen=True)
class ShearBuildingModel:
    """Lumped-mass, lumped-stiffness idealization of a multi-story building.

    `story_masses_kg[i]` is the lumped mass of story i (0 = ground-level
    story). `story_stiffness_n_per_m[i]` is the lateral stiffness of the
    columns connecting story i to the story below it (story 0 connects to
    a fixed base).
    """

    story_masses_kg: tuple[float, ...]
    story_stiffness_n_per_m: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.story_masses_kg) != len(self.story_stiffness_n_per_m):
            raise ValueError("story_masses_kg and story_stiffness_n_per_m must have the same length")
        if not self.story_masses_kg:
            raise ValueError("a shear-building model needs at least one story")
        if any(mass <= 0.0 for mass in self.story_masses_kg):
            raise ValueError("story masses must be positive")
        if any(stiffness <= 0.0 for stiffness in self.story_stiffness_n_per_m):
            raise ValueError("story stiffnesses must be positive")

    @property
    def num_stories(self) -> int:
        return len(self.story_masses_kg)

    def mass_matrix(self) -> list[list[float]]:
        n = self.num_stories
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            matrix[i][i] = self.story_masses_kg[i]
        return matrix

    def stiffness_matrix(self) -> list[list[float]]:
        """Standard shear-building tridiagonal assembly.

        Story i's stiffness connects it to the story below (i-1, or the
        fixed base for i=0). Each floor sees the stiffness of its own
        story plus the story above pulling back on it.
        """
        n = self.num_stories
        k = self.story_stiffness_n_per_m
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            k_below = k[i]
            k_above = k[i + 1] if i + 1 < n else 0.0
            matrix[i][i] = k_below + k_above
            if i + 1 < n:
                matrix[i][i + 1] = -k_above
                matrix[i + 1][i] = -k_above
        return matrix


def _jacobi_eigenvalues_symmetric(matrix: list[list[float]], *, max_sweeps: int = 200, tol: float = 1e-12) -> list[float]:
    """Classical cyclic-Jacobi eigenvalue algorithm for a symmetric matrix.

    Chosen over a library eigensolver so this module has no numpy
    dependency: the matrices here are one row/column per building story,
    so even a handful of stories keeps this cheap, and Jacobi is simple to
    verify against a closed-form result (see tests/test_fe_physics_consistency.py).
    """
    n = len(matrix)
    a = [row[:] for row in matrix]
    for _ in range(max_sweeps):
        off_diag_max = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > off_diag_max:
                    off_diag_max = abs(a[i][j])
                    p, q = i, j
        if off_diag_max < tol:
            break

        # p, q index the largest off-diagonal element, so a[p][q] is
        # off_diag_max and is known to be >= tol at this point.
        tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
        if tau >= 0:
            t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
        else:
            t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c

        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        a[p][q] = 0.0
        a[q][p] = 0.0
        for i in range(n):
            if i == p or i == q:
                continue
            aip, aiq = a[i][p], a[i][q]
            a[i][p] = c * aip - s * aiq
            a[p][i] = a[i][p]
            a[i][q] = s * aip + c * aiq
            a[q][i] = a[i][q]

    return [a[i][i] for i in range(n)]


def natural_frequencies_hz(model: ShearBuildingModel) -> tuple[float, ...]:
    """Solves the generalized eigenproblem K*phi = omega^2 * M*phi and returns natural frequencies in Hz, ascending.

    Since M is diagonal (lumped masses), the generalized problem reduces
    to a standard symmetric eigenproblem A = M^-1/2 * K * M^-1/2 without
    needing a full matrix inversion or decomposition routine.
    """
    shared = _shared_shear_building_frequencies(model.story_masses_kg, model.story_stiffness_n_per_m)
    if shared is not None:
        return shared

    # Dependency-free fallback for installations where the separate
    # timoshenko-engine package is not installed.
    stiffness = model.stiffness_matrix()
    inv_sqrt_mass = [1.0 / math.sqrt(mass) for mass in model.story_masses_kg]
    n = model.num_stories
    normalized = [
        [stiffness[i][j] * inv_sqrt_mass[i] * inv_sqrt_mass[j] for j in range(n)]
        for i in range(n)
    ]
    eigenvalues = _jacobi_eigenvalues_symmetric(normalized)
    # Eigenvalues are omega^2 (rad/s)^2; numerical noise can leave a
    # near-zero eigenvalue slightly negative, clamp before sqrt.
    omegas = [math.sqrt(max(0.0, value)) for value in eigenvalues]
    frequencies = sorted(omega / (2.0 * math.pi) for omega in omegas)
    return tuple(frequencies)


def model_consistency(model: ShearBuildingModel, current) -> dict:
    """Compare measured OMA modes with FE frequencies for decision support.

    The returned uniform stiffness ratio follows Timoshenko's update scale.
    This is a model consistency measure, not a damage diagnosis or a drift
    baseline.
    """
    model_frequencies = natural_frequencies_hz(model)
    measured_frequencies = [mode.frequency_hz for mode in current.modal_parameters]
    pairs = _pair_modes_with_timoshenko(model_frequencies, measured_frequencies)
    if pairs is None:
        from .oma import _pair_modes

        pairs = _pair_modes(model_frequencies, measured_frequencies)

    paired_modes = [
        {
            "model_mode": model_index + 1,
            "model_frequency_hz": float(model_frequencies[model_index]),
            "measured_frequency_hz": float(measured_frequencies[measured_index]),
        }
        for model_index, measured_index in pairs
    ]
    shared_summary = _shared_update_summary(
        model.story_masses_kg,
        model.story_stiffness_n_per_m,
        measured_frequencies,
    ) if pairs else None
    if shared_summary is not None:
        ratio, spread = shared_summary
    elif paired_modes:
        ratios = [
            (pair["measured_frequency_hz"] / pair["model_frequency_hz"]) ** 2
            for pair in paired_modes
        ]
        ratio = median(ratios)
        spread = median(abs(value - ratio) for value in ratios) / ratio * 100.0 if ratio > 0.0 else None
    else:
        ratio = None
        spread = None
    return {
        "model_frequencies_hz": [float(value) for value in model_frequencies],
        "paired_modes": paired_modes,
        "implied_uniform_stiffness_ratio": ratio,
        "mode_ratio_spread_pct": spread,
    }
