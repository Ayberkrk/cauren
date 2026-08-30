from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import re


@dataclass(frozen=True)
class DimensionSpec:
    name: str
    unit: str
    accepted_range: Tuple[float, float]
    aliases: Tuple[str, ...]
    dimension_class: str
    upstream_category: str
    default_missing_policy: str = "missing_ok"
    xai_support_level: str = "generic"


LEGACY_CORE_DIMENSIONS: List[str] = [
    "electronic_temp",
    "material_temp",
    "external_pressure",
    "internal_pressure",
    "magnetic",
    "strain",
    "radiation",
]

CORE_DIMENSIONS: List[str] = list(LEGACY_CORE_DIMENSIONS) + [
    "thermal_gradient",
    "coolant_thermal_delta",
    "electrical_load_kw",
    "electrochemical_health",
    "vibration_rms",
    "acoustic_emission",
    "charge_interface_state",
    "route_duty_index",
    "lifecycle_traceability",
    "electrical_stability",
    "voltage_state",
    "current_state",
    "flow_rate_state",
    "power_quality_state",
    "pressure_delta_state",
    "torque_load_state",
    "rotational_state",
    "geometry_state",
    "occupancy_load",
    "environmental_humidity",
    "process_cycle_state",
    "traction_friction_state",
    "safety_integrity_state",
    "structural_fatigue_state",
    "refrigeration_state",
]

AUX_DIMENSIONS: List[str] = [
    "particle_count_ge_0p3um_m3",
    "particle_size_p50_um",
    "particle_size_p90_um",
    "air_change_rate_ach",
    "hepa_filter_delta_p_pa",
    "molecular_voc_ppb",
    "esd_ion_balance_v_offset",
    "microbe_count_cfu_m3",
    "humidity_rh_pct",
    "noise_db_a",
    "mains_voltage_l1_v",
    "mains_voltage_l2_v",
    "mains_voltage_l3_v",
    "line_current_l1_a",
    "line_current_l2_a",
    "line_current_l3_a",
    "ups_battery_soc_pct",
    "generator_hz",
    "generator_rpm",
    "ground_line_resistance_ohm",
    "fire_smoke_optical_index",
    "fire_smoke_thermal_index",
    "gas_leak_nh3_ppm",
    "gas_leak_ar_ppm",
    "gas_leak_h2_ppm",
    "esd_floor_resistance_ohm",
    "suppression_pressure_bar",
    "chiller_supply_temp_c",
    "chiller_return_temp_c",
    "cooling_water_flow_l_min",
    "co2_ppm",
    "o2_pct",
    "cabin_pressure_total",
    "airflow_velocity_m_s",
    "potable_water_conductivity_us_cm",
    "potable_water_toc_ppb",
    "potable_water_microbe_cfu_ml",
    "surface_microbe_cfu_cm2",
    "trace_contaminant_index",
    "water_storage_level_pct",
    "waste_storage_fill_pct",
    "combustion_co_ppm",
]

ANALOG_HABITAT_DIMENSIONS: List[str] = [
    "co2_ppm",
    "o2_pct",
    "cabin_pressure_total",
    "airflow_velocity_m_s",
    "potable_water_conductivity_us_cm",
    "potable_water_toc_ppb",
    "potable_water_microbe_cfu_ml",
    "surface_microbe_cfu_cm2",
    "trace_contaminant_index",
    "water_storage_level_pct",
    "waste_storage_fill_pct",
    "combustion_co_ppm",
]

LEGACY_38_DIMENSIONS: List[str] = list(LEGACY_CORE_DIMENSIONS) + [
    dimension for dimension in AUX_DIMENSIONS if dimension not in ANALOG_HABITAT_DIMENSIONS
]

CANONICAL_DIMENSIONS: List[str] = list(CORE_DIMENSIONS) + list(AUX_DIMENSIONS)
MODEL_INPUT_DIMENSIONS: List[str] = list(CANONICAL_DIMENSIONS)
PHYSICS_CORE_DIMENSIONS = set(CORE_DIMENSIONS)


_DIMENSION_SPECS: List[DimensionSpec] = [
    DimensionSpec(
        "electronic_temp",
        "C",
        (-120.0, 850.0),
        ("temperature", "elec_temp", "electronics_temp", "electronic_temperature"),
        "core",
        "thermal",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "material_temp",
        "C",
        (-200.0, 1500.0),
        ("mat_temp", "material_temperature"),
        "core",
        "thermal",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "external_pressure",
        "bar",
        (0.0, 500.0),
        ("pressure", "ext_pressure", "external_press", "pressure_external"),
        "core",
        "pressure",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "internal_pressure",
        "bar",
        (0.0, 500.0),
        ("int_pressure", "internal_press", "pressure_internal"),
        "core",
        "pressure",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "magnetic",
        "T",
        (0.0, 25.0),
        ("mag", "magnetic_field"),
        "core",
        "field",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "strain",
        "microstrain",
        (-1_500_000.0, 1_500_000.0),
        ("structural_strain", "deformation"),
        "core",
        "mechanical",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "radiation",
        "Gy/h",
        (0.0, 100_000.0),
        ("rad", "dose_rate", "radiation_dose"),
        "core",
        "safety",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "thermal_gradient",
        "C",
        (0.0, 500.0),
        ("temp_gradient", "temperature_gradient", "surface_temp_gradient"),
        "core",
        "thermal",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "coolant_thermal_delta",
        "C",
        (0.0, 500.0),
        ("coolant_delta_t", "delta_t_coolant", "cooling_delta_t"),
        "core",
        "thermal",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "electrical_load_kw",
        "kW",
        (0.0, 2_000.0),
        ("power_kw", "electrical_load", "load_kw"),
        "core",
        "power",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "electrochemical_health",
        "ratio",
        (0.0, 1.0),
        ("battery_health", "soh_ratio", "electrochem_health"),
        "core",
        "electrochemical",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "vibration_rms",
        "g_rms",
        (0.0, 100.0),
        ("vibration", "vibration_g_rms", "vibration_rms_g"),
        "core",
        "mechanical",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "acoustic_emission",
        "index",
        (0.0, 100_000.0),
        ("acoustic", "acoustic_emission_rate", "ae_rate"),
        "core",
        "mechanical",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "charge_interface_state",
        "index",
        (0.0, 100.0),
        ("charge_interface", "pantograph_state", "charge_contact_state"),
        "core",
        "charging",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "route_duty_index",
        "index",
        (0.0, 100.0),
        ("route_duty", "duty_index", "route_severity_index"),
        "core",
        "operations",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "lifecycle_traceability",
        "ratio",
        (0.0, 1.0),
        ("traceability", "passport_traceability", "lifecycle_evidence"),
        "core",
        "compliance",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "electrical_stability",
        "index",
        (0.0, 100.0),
        ("electrical_stability_index", "stability_state"),
        "core",
        "power",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "voltage_state",
        "V",
        (0.0, 100_000.0),
        ("voltage", "bus_voltage_state", "line_voltage_state"),
        "core",
        "power",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "current_state",
        "A",
        (0.0, 100_000.0),
        ("current", "bus_current_state", "line_current_state"),
        "core",
        "power",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "flow_rate_state",
        "L/min",
        (0.0, 100_000.0),
        ("flow_state", "coolant_flow_state", "refrigerant_flow_state"),
        "core",
        "fluid",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "power_quality_state",
        "index",
        (0.0, 100.0),
        ("power_quality", "harmonic_quality", "thd_state"),
        "core",
        "power",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "pressure_delta_state",
        "bar",
        (-1_000.0, 1_000.0),
        ("pressure_delta", "delta_pressure_state", "dp_state"),
        "core",
        "pressure",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "torque_load_state",
        "index",
        (0.0, 100.0),
        ("torque_state", "load_torque_state", "mechanical_load_state"),
        "core",
        "mechanical",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "rotational_state",
        "rpm",
        (0.0, 1_000_000.0),
        ("rpm_state", "speed_rotation_state", "rotary_speed_state"),
        "core",
        "mechanical",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "geometry_state",
        "index",
        (0.0, 100_000.0),
        ("geometry", "dimensional_state", "shape_state"),
        "core",
        "geometry",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "occupancy_load",
        "count",
        (0.0, 1_000_000.0),
        ("occupancy", "visitor_load", "people_load"),
        "core",
        "operations",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "environmental_humidity",
        "%RH",
        (0.0, 100.0),
        ("humidity_state", "humidity", "relative_humidity_state"),
        "core",
        "thermal",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "process_cycle_state",
        "s",
        (0.0, 1_000_000.0),
        ("cycle_state", "process_timing_state", "cure_cycle_state"),
        "core",
        "process",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "traction_friction_state",
        "index",
        (0.0, 100.0),
        ("friction_state", "traction_state", "grip_state"),
        "core",
        "mechanical",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "safety_integrity_state",
        "index",
        (0.0, 100.0),
        ("safety_state", "integrity_state", "isolation_state"),
        "core",
        "safety",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "structural_fatigue_state",
        "index",
        (0.0, 100.0),
        ("fatigue_state", "structure_fatigue", "wear_state"),
        "core",
        "mechanical",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "refrigeration_state",
        "index",
        (0.0, 100.0),
        ("refrigeration", "cooling_loop_state", "compressor_state"),
        "core",
        "thermal",
        "required_for_model",
        "physics",
    ),
    DimensionSpec(
        "particle_count_ge_0p3um_m3",
        "count/m3",
        (0.0, 5_000_000.0),
        ("particle_count", "particles_0p3um", "particle_count_0p3um"),
        "aux",
        "cleanroom",
        "context_only",
    ),
    DimensionSpec(
        "particle_size_p50_um",
        "um",
        (0.0, 100.0),
        ("particle_p50_um", "particle_size_median_um"),
        "aux",
        "cleanroom",
        "context_only",
    ),
    DimensionSpec(
        "particle_size_p90_um",
        "um",
        (0.0, 200.0),
        ("particle_p90_um", "particle_size_large_um"),
        "aux",
        "cleanroom",
        "context_only",
    ),
    DimensionSpec(
        "air_change_rate_ach",
        "ACH",
        (0.0, 80.0),
        ("air_changes_per_hour", "ach"),
        "aux",
        "hvac",
        "context_only",
    ),
    DimensionSpec(
        "hepa_filter_delta_p_pa",
        "Pa",
        (0.0, 5_000.0),
        ("hepa_delta_p", "hepa_filter_dp_pa"),
        "aux",
        "hvac",
        "context_only",
    ),
    DimensionSpec(
        "molecular_voc_ppb",
        "ppb",
        (0.0, 50_000.0),
        ("voc_ppb", "molecular_voc"),
        "aux",
        "air_quality",
        "context_only",
    ),
    DimensionSpec(
        "esd_ion_balance_v_offset",
        "V",
        (-1_000.0, 1_000.0),
        ("ion_balance_offset_v", "esd_ion_balance"),
        "aux",
        "esd",
        "context_only",
    ),
    DimensionSpec(
        "microbe_count_cfu_m3",
        "CFU/m3",
        (0.0, 100_000.0),
        ("microbial_count_cfu_m3", "microbe_count"),
        "aux",
        "cleanroom",
        "context_only",
    ),
    DimensionSpec(
        "humidity_rh_pct",
        "%RH",
        (0.0, 100.0),
        ("humidity", "relative_humidity", "humidity_pct"),
        "aux",
        "thermal",
        "context_only",
    ),
    DimensionSpec(
        "noise_db_a",
        "dB(A)",
        (0.0, 180.0),
        ("noise", "sound_db_a"),
        "aux",
        "acoustic",
        "context_only",
    ),
    DimensionSpec(
        "mains_voltage_l1_v",
        "V",
        (0.0, 1_000.0),
        ("mains_l1_v", "voltage_l1"),
        "aux",
        "power",
        "context_only",
    ),
    DimensionSpec(
        "mains_voltage_l2_v",
        "V",
        (0.0, 1_000.0),
        ("mains_l2_v", "voltage_l2"),
        "aux",
        "power",
        "context_only",
    ),
    DimensionSpec(
        "mains_voltage_l3_v",
        "V",
        (0.0, 1_000.0),
        ("mains_l3_v", "voltage_l3"),
        "aux",
        "power",
        "context_only",
    ),
    DimensionSpec(
        "line_current_l1_a",
        "A",
        (0.0, 5_000.0),
        ("current_l1", "line_l1_a"),
        "aux",
        "power",
        "context_only",
    ),
    DimensionSpec(
        "line_current_l2_a",
        "A",
        (0.0, 5_000.0),
        ("current_l2", "line_l2_a"),
        "aux",
        "power",
        "context_only",
    ),
    DimensionSpec(
        "line_current_l3_a",
        "A",
        (0.0, 5_000.0),
        ("current_l3", "line_l3_a"),
        "aux",
        "power",
        "context_only",
    ),
    DimensionSpec(
        "ups_battery_soc_pct",
        "%",
        (0.0, 100.0),
        ("ups_soc", "battery_soc_pct"),
        "aux",
        "power",
        "context_only",
    ),
    DimensionSpec(
        "generator_hz",
        "Hz",
        (0.0, 1_000.0),
        ("generator_freq", "generator_frequency_hz"),
        "aux",
        "power",
        "context_only",
    ),
    DimensionSpec(
        "generator_rpm",
        "rpm",
        (0.0, 100_000.0),
        ("generator_speed", "generator_speed_rpm"),
        "aux",
        "power",
        "context_only",
    ),
    DimensionSpec(
        "ground_line_resistance_ohm",
        "ohm",
        (0.0, 1_000_000.0),
        ("ground_resistance_ohm", "ground_line_resistance"),
        "aux",
        "safety",
        "context_only",
    ),
    DimensionSpec(
        "fire_smoke_optical_index",
        "index",
        (0.0, 100.0),
        ("smoke_optical_index", "fire_smoke_optical"),
        "aux",
        "safety",
        "context_only",
    ),
    DimensionSpec(
        "fire_smoke_thermal_index",
        "index",
        (0.0, 100.0),
        ("smoke_thermal_index", "fire_smoke_thermal"),
        "aux",
        "safety",
        "context_only",
    ),
    DimensionSpec(
        "gas_leak_nh3_ppm",
        "ppm",
        (0.0, 10_000.0),
        ("gas_nh3", "nh3_ppm", "ammonia_ppm"),
        "aux",
        "safety",
        "context_only",
    ),
    DimensionSpec(
        "gas_leak_ar_ppm",
        "ppm",
        (0.0, 10_000.0),
        ("gas_ar", "argon_ppm"),
        "aux",
        "safety",
        "context_only",
    ),
    DimensionSpec(
        "gas_leak_h2_ppm",
        "ppm",
        (0.0, 10_000.0),
        ("gas_h2", "hydrogen_ppm"),
        "aux",
        "safety",
        "context_only",
    ),
    DimensionSpec(
        "esd_floor_resistance_ohm",
        "ohm",
        (0.0, 1_000_000_000.0),
        ("floor_resistance_ohm", "esd_floor_resistance"),
        "aux",
        "esd",
        "context_only",
    ),
    DimensionSpec(
        "suppression_pressure_bar",
        "bar",
        (0.0, 1_000.0),
        ("suppression_pressure", "fire_suppression_pressure_bar"),
        "aux",
        "safety",
        "context_only",
    ),
    DimensionSpec(
        "chiller_supply_temp_c",
        "C",
        (-100.0, 500.0),
        ("chiller_supply_temp", "supply_temp_c"),
        "aux",
        "hvac",
        "context_only",
    ),
    DimensionSpec(
        "chiller_return_temp_c",
        "C",
        (-100.0, 500.0),
        ("chiller_return_temp", "return_temp_c"),
        "aux",
        "hvac",
        "context_only",
    ),
    DimensionSpec(
        "cooling_water_flow_l_min",
        "L/min",
        (0.0, 100_000.0),
        ("cooling_water_flow", "water_flow_l_min"),
        "aux",
        "hvac",
        "context_only",
    ),
    DimensionSpec(
        "co2_ppm",
        "ppm",
        (0.0, 100_000.0),
        ("carbon_dioxide_ppm", "cabin_co2_ppm", "co_2_ppm"),
        "aux",
        "life_support",
        "context_only",
    ),
    DimensionSpec(
        "o2_pct",
        "%",
        (0.0, 100.0),
        ("oxygen_pct", "cabin_o2_pct", "o2_percent"),
        "aux",
        "life_support",
        "context_only",
    ),
    DimensionSpec(
        "cabin_pressure_total",
        "bar",
        (0.0, 5.0),
        ("habitat_pressure_bar", "cabin_pressure", "atmosphere_pressure_bar"),
        "aux",
        "life_support",
        "context_only",
    ),
    DimensionSpec(
        "airflow_velocity_m_s",
        "m/s",
        (0.0, 50.0),
        ("air_velocity_m_s", "ventilation_flow_velocity", "cabin_airflow_velocity"),
        "aux",
        "life_support",
        "context_only",
    ),
    DimensionSpec(
        "potable_water_conductivity_us_cm",
        "uS/cm",
        (0.0, 100_000.0),
        ("water_conductivity_us_cm", "potable_conductivity", "conductivity_us_cm"),
        "aux",
        "water_quality",
        "context_only",
    ),
    DimensionSpec(
        "potable_water_toc_ppb",
        "ppb",
        (0.0, 1_000_000.0),
        ("water_toc_ppb", "potable_toc_ppb", "total_organic_carbon_ppb"),
        "aux",
        "water_quality",
        "context_only",
    ),
    DimensionSpec(
        "potable_water_microbe_cfu_ml",
        "CFU/mL",
        (0.0, 100_000.0),
        ("water_microbe_cfu_ml", "potable_microbe_cfu_ml", "water_microbiology_cfu_ml"),
        "aux",
        "water_quality",
        "context_only",
    ),
    DimensionSpec(
        "surface_microbe_cfu_cm2",
        "CFU/cm2",
        (0.0, 100_000.0),
        ("surface_bioburden_cfu_cm2", "surface_microbe_load", "surface_microbiology_cfu_cm2"),
        "aux",
        "water_quality",
        "context_only",
    ),
    DimensionSpec(
        "trace_contaminant_index",
        "index",
        (0.0, 100.0),
        ("trace_contaminants", "trace_contaminant_score", "air_trace_contaminant_index"),
        "aux",
        "life_support",
        "context_only",
    ),
    DimensionSpec(
        "water_storage_level_pct",
        "%",
        (0.0, 100.0),
        ("potable_water_storage_level_pct", "water_tank_level_pct", "water_storage_pct"),
        "aux",
        "water_quality",
        "context_only",
    ),
    DimensionSpec(
        "waste_storage_fill_pct",
        "%",
        (0.0, 100.0),
        ("waste_fill_pct", "waste_tank_fill_pct", "waste_storage_pct"),
        "aux",
        "water_quality",
        "context_only",
    ),
    DimensionSpec(
        "combustion_co_ppm",
        "ppm",
        (0.0, 10_000.0),
        ("co_ppm", "carbon_monoxide_ppm", "combustion_carbon_monoxide_ppm"),
        "aux",
        "safety",
        "context_only",
    ),
]


DIMENSION_SPECS: Dict[str, DimensionSpec] = {spec.name: spec for spec in _DIMENSION_SPECS}
CORE_DIMENSION_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(CORE_DIMENSIONS)}
CANONICAL_DIMENSION_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(CANONICAL_DIMENSIONS)}

CORE_DIMENSION_UNITS: Dict[str, str] = {
    name: DIMENSION_SPECS[name].unit
    for name in CORE_DIMENSIONS
}
CANONICAL_DIMENSION_UNITS: Dict[str, str] = {
    name: spec.unit
    for name, spec in DIMENSION_SPECS.items()
}

CORE_DIMENSION_RANGES: Dict[str, Tuple[float, float]] = {
    name: DIMENSION_SPECS[name].accepted_range
    for name in CORE_DIMENSIONS
}
CANONICAL_DIMENSION_RANGES: Dict[str, Tuple[float, float]] = {
    name: spec.accepted_range
    for name, spec in DIMENSION_SPECS.items()
}


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower()).strip("_")


CANONICAL_DIMENSION_ALIASES: Dict[str, str] = {}
for _spec in _DIMENSION_SPECS:
    CANONICAL_DIMENSION_ALIASES[_normalize_key(_spec.name)] = _spec.name
    for _alias in _spec.aliases:
        CANONICAL_DIMENSION_ALIASES[_normalize_key(_alias)] = _spec.name

CORE_DIMENSION_ALIASES: Dict[str, str] = {
    alias: target
    for alias, target in CANONICAL_DIMENSION_ALIASES.items()
    if target in CORE_DIMENSIONS
}


def canonical_dimension_name(name: str) -> Optional[str]:
    key = _normalize_key(name)
    if not key:
        return None
    if key in CANONICAL_DIMENSION_ALIASES:
        return CANONICAL_DIMENSION_ALIASES[key]
    if "internal" in key and "pressure" in key:
        return "internal_pressure"
    if "external" in key and "pressure" in key:
        return "external_pressure"
    if key.endswith("_pressure"):
        return "external_pressure"
    if "material" in key and "temp" in key:
        return "material_temp"
    if "electronic" in key and "temp" in key:
        return "electronic_temp"
    if key.startswith("mag"):
        return "magnetic"
    if key.startswith("rad"):
        return "radiation"
    if "gradient" in key and "temp" in key:
        return "thermal_gradient"
    if "coolant" in key and ("delta" in key or "dt" in key):
        return "coolant_thermal_delta"
    if ("power" in key or "load" in key) and key.endswith("kw"):
        return "electrical_load_kw"
    if "electrochem" in key or "battery_health" in key or "soh_ratio" in key:
        return "electrochemical_health"
    if "vibration" in key:
        return "vibration_rms"
    if "acoustic" in key or "ae_" in key or key.startswith("ae"):
        return "acoustic_emission"
    if "charge" in key and ("interface" in key or "contact" in key or "pantograph" in key):
        return "charge_interface_state"
    if "route" in key and ("duty" in key or "severity" in key):
        return "route_duty_index"
    if "traceability" in key or ("passport" in key and "trace" in key):
        return "lifecycle_traceability"
    if "stability" in key and ("electrical" in key or "power" in key):
        return "electrical_stability"
    if "voltage" in key:
        return "voltage_state"
    if "current" in key:
        return "current_state"
    if "flow" in key:
        return "flow_rate_state"
    if "harmonic" in key or "thd" in key or ("power" in key and "quality" in key):
        return "power_quality_state"
    if "pressure" in key and "delta" in key:
        return "pressure_delta_state"
    if "torque" in key:
        return "torque_load_state"
    if "rpm" in key or "rotation" in key or ("speed" in key and "symbol" not in key and "wind" not in key):
        return "rotational_state"
    if "diameter" in key or "radius" in key or "circumference" in key or "width" in key or "aspect_ratio" in key:
        return "geometry_state"
    if "occupancy" in key or "visitor" in key or ("door_cycle" in key):
        return "occupancy_load"
    if "humidity" in key:
        return "environmental_humidity"
    if "cycle" in key or "cure_time" in key or "open_hours" in key:
        return "process_cycle_state"
    if "grip" in key or "friction" in key or "rolling_resistance" in key or "noise_db" in key:
        return "traction_friction_state"
    if "safety" in key or "permit" in key or "inspection" in key or "hazard" in key or "isolation" in key:
        return "safety_integrity_state"
    if "fatigue" in key or "wear" in key or "tread_depth" in key:
        return "structural_fatigue_state"
    if "compressor" in key or "refrigerant" in key or "refrigeration" in key:
        return "refrigeration_state"
    if "humidity" in key:
        return "humidity_rh_pct"
    if key == "co2" or "carbon_dioxide" in key or "cabin_co2" in key:
        return "co2_ppm"
    if key == "o2" or "oxygen" in key:
        return "o2_pct"
    if "cabin" in key and "pressure" in key:
        return "cabin_pressure_total"
    if "airflow" in key and ("velocity" in key or key.endswith("_m_s")):
        return "airflow_velocity_m_s"
    if "conductivity" in key and "water" in key:
        return "potable_water_conductivity_us_cm"
    if key.endswith("toc_ppb") or "organic_carbon" in key:
        return "potable_water_toc_ppb"
    if "water" in key and "microbe" in key:
        return "potable_water_microbe_cfu_ml"
    if "surface" in key and "microbe" in key:
        return "surface_microbe_cfu_cm2"
    if "trace" in key and "contaminant" in key:
        return "trace_contaminant_index"
    if "water" in key and "storage" in key and ("level" in key or "pct" in key):
        return "water_storage_level_pct"
    if "waste" in key and ("fill" in key or "storage" in key):
        return "waste_storage_fill_pct"
    if key == "co" or "carbon_monoxide" in key or ("combustion" in key and "co" in key):
        return "combustion_co_ppm"
    if "voc" in key:
        return "molecular_voc_ppb"
    if "generator" in key and "hz" in key:
        return "generator_hz"
    if "generator" in key and ("rpm" in key or "speed" in key):
        return "generator_rpm"
    return None


def dimension_spec(name: str) -> Optional[DimensionSpec]:
    canonical = canonical_dimension_name(name)
    if canonical is None:
        return None
    return DIMENSION_SPECS.get(canonical)


def dimension_class(name: str) -> Optional[str]:
    spec = dimension_spec(name)
    return None if spec is None else spec.dimension_class


def dimension_range(name: str) -> Optional[Tuple[float, float]]:
    spec = dimension_spec(name)
    return None if spec is None else spec.accepted_range


def dimension_unit(name: str) -> Optional[str]:
    spec = dimension_spec(name)
    return None if spec is None else spec.unit


def is_core_dimension(name: str) -> bool:
    canonical = canonical_dimension_name(name)
    return canonical in PHYSICS_CORE_DIMENSIONS


def dimension_names_by_class(target_class: str) -> List[str]:
    target = str(target_class or "").strip().lower()
    return [name for name, spec in DIMENSION_SPECS.items() if spec.dimension_class == target]


def supported_dimension_names() -> List[str]:
    return list(CANONICAL_DIMENSIONS)


def normalize_dimension_names(names: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for name in names:
        canonical = canonical_dimension_name(str(name))
        if canonical:
            normalized.append(canonical)
    return normalized
