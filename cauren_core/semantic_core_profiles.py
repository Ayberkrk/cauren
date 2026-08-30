from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv

from cauren_agents.registry import build_default_registry
from core_dimensions import CORE_DIMENSIONS, canonical_dimension_name, dimension_class


SEMANTIC_TARGET_SIZES: tuple[int, ...] = (16, 32)


AGENT_DIMENSION_PRIORS: dict[str, tuple[str, ...]] = {
    "cauren-civil": (
        "strain",
        "structural_fatigue_state",
        "geometry_state",
        "safety_integrity_state",
        "process_cycle_state",
        "occupancy_load",
        "external_pressure",
        "environmental_humidity",
    ),
}


@dataclass(frozen=True)
class SemanticProfileScore:
    profile_name: str
    target_size: int
    actual_size: int
    semantic_dimensions: tuple[str, ...]
    required_coverage: float
    optional_coverage: float
    dataset_coverage: float
    class_diversity: float
    collision_penalty: float
    score: float


def _token_dimension(feature_name: str) -> str | None:
    canonical = canonical_dimension_name(feature_name)
    if canonical in CORE_DIMENSIONS:
        return canonical

    text = str(feature_name or "").strip().lower()
    if not text:
        return None
    if "temp" in text or "thermal" in text:
        if "gradient" in text:
            return "thermal_gradient"
        return "material_temp"
    if "pressure" in text:
        if "delta" in text:
            return "pressure_delta_state"
        return "external_pressure"
    if "strain" in text or "deformation" in text:
        return "strain"
    if "vibration" in text or "accel" in text:
        return "vibration_rms"
    if "acoustic" in text or "piezo" in text:
        return "acoustic_emission"
    if "building" in text or "height" in text or "footprint" in text or "geometry" in text:
        return "geometry_state"
    if "construction" in text or "progress" in text or "cycle" in text or "cure_time" in text:
        return "process_cycle_state"
    if "permit" in text or "inspection" in text or "safety" in text or "hazard" in text or "risk" in text:
        return "safety_integrity_state"
    if "ground" in text or "soil" in text or "stability" in text or "settlement" in text or "foundation" in text:
        return "structural_fatigue_state"
    if "occupancy" in text or "life_safety" in text:
        return "occupancy_load"
    if "humidity" in text or "moisture" in text:
        return "environmental_humidity"
    if "load" in text or "weight" in text:
        return "torque_load_state"
    return None


def _dataset_feature_counter(dataset_dir: Path, agent_id: str) -> Counter[str]:
    path = dataset_dir / "agents" / agent_id / "raw_sensor_readings.csv"
    counter: Counter[str] = Counter()
    if not path.exists():
        return counter
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = str(row.get("sensor_name") or "").strip()
            if name:
                counter[name] += 1
    return counter


def derive_agent_semantic_dimensions(
    *,
    agent_id: str,
    feature_names: Iterable[str],
    required_features: Iterable[str],
    dataset_feature_counts: Counter[str] | None = None,
    target_size: int,
) -> tuple[str, ...]:
    dataset_feature_counts = dataset_feature_counts or Counter()
    priors = list(AGENT_DIMENSION_PRIORS.get(agent_id, ()))
    dim_score: dict[str, float] = defaultdict(float)

    for dim in priors:
        dim_score[dim] += 4.0

    required = {str(name) for name in required_features}
    for name in feature_names:
        dim = _token_dimension(str(name))
        if not dim:
            continue
        weight = 3.0 if name in required else 1.0
        weight += min(3.0, dataset_feature_counts.get(str(name), 0) / 5000.0)
        dim_score[dim] += weight

    ordered = sorted(
        dim_score.items(),
        key=lambda item: (-item[1], CORE_DIMENSIONS.index(item[0]) if item[0] in CORE_DIMENSIONS else 10_000, item[0]),
    )
    selected = [name for name, _ in ordered[:target_size]]

    for prior in priors:
        if prior not in selected and len(selected) < target_size:
            selected.append(prior)

    for dim in CORE_DIMENSIONS:
        if len(selected) >= target_size:
            break
        if dim not in selected and dim_score.get(dim, 0.0) > 0.0:
            selected.append(dim)
    return tuple(selected[:target_size])


def score_agent_semantic_profile(
    *,
    agent_id: str,
    feature_names: Iterable[str],
    required_features: Iterable[str],
    optional_features: Iterable[str],
    dataset_feature_counts: Counter[str] | None = None,
    target_size: int,
) -> SemanticProfileScore:
    dataset_feature_counts = dataset_feature_counts or Counter()
    semantic_dimensions = derive_agent_semantic_dimensions(
        agent_id=agent_id,
        feature_names=feature_names,
        required_features=required_features,
        dataset_feature_counts=dataset_feature_counts,
        target_size=target_size,
    )
    selected = set(semantic_dimensions)
    mapped_all = {name: _token_dimension(name) for name in feature_names}
    required_list = [name for name in required_features]
    optional_list = [name for name in optional_features]

    req_hits = sum(1 for name in required_list if mapped_all.get(name) in selected)
    opt_hits = sum(1 for name in optional_list if mapped_all.get(name) in selected)
    dataset_hits = sum(count for name, count in dataset_feature_counts.items() if mapped_all.get(name) in selected)
    dataset_total = sum(dataset_feature_counts.values())
    class_names = {dimension_class(dim) or "unknown" for dim in semantic_dimensions}
    collisions: Counter[str] = Counter()
    for name, dim in mapped_all.items():
        if dim in selected:
            collisions[dim] += 1
    overload = sum(max(0, count - 4) for count in collisions.values())

    required_coverage = req_hits / max(1, len(required_list))
    optional_coverage = opt_hits / max(1, len(optional_list))
    dataset_coverage = dataset_hits / max(1, dataset_total)
    class_diversity = len(class_names) / max(1, min(target_size, 8))
    collision_penalty = min(1.0, overload / max(1, len(semantic_dimensions) * 2))
    score = (
        (0.42 * required_coverage)
        + (0.18 * optional_coverage)
        + (0.20 * dataset_coverage)
        + (0.20 * class_diversity)
        - (0.18 * collision_penalty)
    )
    return SemanticProfileScore(
        profile_name=f"{agent_id}__semantic{target_size}",
        target_size=target_size,
        actual_size=len(semantic_dimensions),
        semantic_dimensions=semantic_dimensions,
        required_coverage=required_coverage,
        optional_coverage=optional_coverage,
        dataset_coverage=dataset_coverage,
        class_diversity=class_diversity,
        collision_penalty=collision_penalty,
        score=score,
    )


def benchmark_agent_semantic_profiles(dataset_roots: Iterable[Path]) -> dict[str, list[SemanticProfileScore]]:
    registry = build_default_registry()
    dataset_roots = list(dataset_roots)
    by_agent: dict[str, list[SemanticProfileScore]] = {}
    for agent in registry.all():
        dataset_counter = Counter()
        for root in dataset_roots:
            dataset_counter.update(_dataset_feature_counter(root, agent.schema.agent_id))
        rows: list[SemanticProfileScore] = []
        for target_size in SEMANTIC_TARGET_SIZES:
            rows.append(
                score_agent_semantic_profile(
                    agent_id=agent.schema.agent_id,
                    feature_names=agent.schema.feature_order,
                    required_features=agent.schema.required_features,
                    optional_features=agent.schema.optional_features,
                    dataset_feature_counts=dataset_counter,
                    target_size=target_size,
                )
            )
        by_agent[agent.schema.agent_id] = rows
    return by_agent
