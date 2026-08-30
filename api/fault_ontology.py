from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


KNOWN_STATUS_VALUES = {
    "known",
    "known_family_unknown_subtype",
    "unknown_with_similarity",
    "unknown",
}


FAMILY_LABELS: dict[str, str] = {
    "nominal": "Nominal Operation",
    "review_required": "Needs Review",
    "unknown_anomaly": "Unknown Anomaly",
    "pressure_instability": "Pressure Instability",
    "thermal_anomaly": "Thermal Anomaly",
    "radiation_anomaly": "Radiation Anomaly",
    "structural_anomaly": "Structural Anomaly",
    "sensor_drift": "Sensor Drift",
    "vibration_mechanical": "Mechanical Vibration",
    "electrical_anomaly": "Electrical Anomaly",
    "flow_instability": "Flow Instability",
    "process_disturbance": "Process Disturbance",
    "signal_noise": "Signal Noise",
    "multi_fault": "Multi-Fault Interaction",
}


FAMILY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("nominal", (" OK ", " NORMAL ", " NOMINAL ", " STABLE ")),
    ("review_required", (" REVIEW ", " MANUAL TRIAGE ", " HUMAN REVIEW ")),
    ("multi_fault", (" MULTI ", " COMBINED ", " CASCADE ")),
    ("pressure_instability", (" PRESSURE ", " PRESSURIZATION ", " REGULATOR ", " VALVE ", " CAVITATION ", " PROP-")),
    ("thermal_anomaly", (" THERMAL ", " HEAT ", " COOLANT ", " OIL_OVERHEAT ", " CHILLER ")),
    ("radiation_anomaly", (" RADIATION ", " RAD ", " SOLAR ")),
    ("structural_anomaly", (" STRAIN ", " STRUCT", " AXIAL ", " DISPLACEMENT ", " LEAK ")),
    ("sensor_drift", (" DRIFT ", " RTD ")),
    ("vibration_mechanical", (" VIBRATION ", " BEARING ")),
    ("electrical_anomaly", (" STATOR ", " FLUX ", " VOLTAGE ", " CURRENT ", " POWER ", " MAGNETIC ", " GUIDANCE ")),
    ("flow_instability", (" FLOW ", " LEAK ", " SUPPRESSION ", " NH3 ", " H2 ")),
    ("process_disturbance", (" IDV", " PROCESS ", " DISTURBANCE ", " UPSET ")),
    ("signal_noise", (" NOISE ", " OSCILLATION ", " SATURATION ")),
]


GENERIC_SUBTYPES = {
    "ANOMALY_CAUTION",
    "ANOMALY_WARNING",
    "ANOMALY_CRITICAL",
    "UNKNOWN",
    "UNKNOWN_FALLBACK",
    "OK",
    "NORMAL",
    "REVIEW",
}


def _load_registry() -> dict[str, Any]:
    path = Path(__file__).with_name("canonical_fault_registry.json")
    if not path.exists():
        return {"families": [], "subtypes": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"families": [], "subtypes": []}
    if not isinstance(payload, dict):
        return {"families": [], "subtypes": []}
    return payload


_REGISTRY = _load_registry()
_REGISTRY_FAMILY_LABELS = {
    str(item.get("fault_family_id") or "").strip(): str(item.get("fault_family_label") or "").strip()
    for item in _REGISTRY.get("families", [])
    if isinstance(item, dict) and str(item.get("fault_family_id") or "").strip()
}
_REGISTRY_ALIAS_RULES: list[tuple[str, tuple[str, ...]]] = []
for item in _REGISTRY.get("families", []):
    if not isinstance(item, dict):
        continue
    family_id = str(item.get("fault_family_id") or "").strip()
    aliases = tuple(
        f" {str(alias).strip().upper()} "
        for alias in (item.get("aliases") or [])
        if str(alias).strip()
    )
    if family_id and aliases:
        _REGISTRY_ALIAS_RULES.append((family_id, aliases))
FAMILY_LABELS.update({key: value for key, value in _REGISTRY_FAMILY_LABELS.items() if value})
for family_id, aliases in _REGISTRY_ALIAS_RULES:
    if not any(existing_id == family_id for existing_id, _ in FAMILY_RULES):
        FAMILY_RULES.append((family_id, aliases))

_REGISTRY_SUBTYPE_BY_ALIAS: dict[str, tuple[str, str, str]] = {}
for item in _REGISTRY.get("subtypes", []):
    if not isinstance(item, dict):
        continue
    subtype_id = str(item.get("fault_subtype_id") or "").strip()
    subtype_label = str(item.get("fault_subtype_label") or "").strip()
    family_id = str(item.get("fault_family_id") or "").strip()
    if not subtype_id or not family_id:
        continue
    for alias in item.get("aliases") or []:
        alias_key = re.sub(r"\s+", " ", str(alias or "").strip()).upper()
        if alias_key:
            _REGISTRY_SUBTYPE_BY_ALIAS[alias_key] = (family_id, subtype_id, subtype_label)


@dataclass(frozen=True)
class CanonicalFault:
    fault_family_id: str
    fault_family_label: str
    fault_subtype_id: str
    fault_subtype_label: str
    fault_signature_id: str
    fault_signature_text: str
    known_status: str
    similar_family_candidates: list[dict[str, Any]]
    review_required: bool


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_label(value: Any) -> str:
    return re.sub(r"\s+", " ", _normalize_text(value))


def _normalize_upper(value: Any) -> str:
    return _normalize_label(value).upper()


def _slug(value: str, *, default: str = "") -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return text or default


def _titleize(value: str) -> str:
    text = _normalize_label(value)
    if not text:
        return ""
    text = text.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip().title()


def _humanize_subtype(label: str) -> str:
    text = _normalize_upper(label)
    if not text:
        return ""
    if text.startswith("FAULT_"):
        text = text[len("FAULT_") :]
    elif text.startswith("ANOMALY_"):
        text = text[len("ANOMALY_") :]
    return _titleize(text)


def _signature_id(signature_text: str) -> str:
    text = _normalize_label(signature_text)
    if not text:
        return ""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"sig-{digest}"


def _family_from_keywords(*values: Any) -> str:
    probe = " " + " ".join(_normalize_upper(value) for value in values if _normalize_text(value)) + " "
    if not probe.strip():
        return "unknown_anomaly"
    for family_id, keywords in FAMILY_RULES:
        if any(keyword in probe for keyword in keywords):
            return family_id
    return "unknown_anomaly"


def _family_from_dimension(primary_dimension: str) -> str:
    dim = _slug(primary_dimension)
    return {
        "external_pressure": "pressure_instability",
        "internal_pressure": "pressure_instability",
        "material_temp": "thermal_anomaly",
        "electronic_temp": "thermal_anomaly",
        "radiation": "radiation_anomaly",
        "strain": "structural_anomaly",
        "magnetic": "electrical_anomaly",
    }.get(dim, "unknown_anomaly")


def _candidate_family_candidates(raw_candidates: Any) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    if not isinstance(raw_candidates, list):
        return []
    for item in raw_candidates:
        if not isinstance(item, dict):
            continue
        raw_label = _normalize_label(item.get("label") or item.get("description"))
        if not raw_label:
            continue
        family_id = _family_from_keywords(raw_label)
        confidence = float(item.get("probability", item.get("confidence", 0.0)) or 0.0)
        if family_id not in merged or confidence > float(merged[family_id]["confidence"]):
            merged[family_id] = {
                "family_id": family_id,
                "family_label": FAMILY_LABELS.get(family_id, _titleize(family_id)),
                "confidence": max(0.0, min(1.0, confidence)),
                "source_label": raw_label,
            }
    ordered = sorted(merged.values(), key=lambda row: float(row["confidence"]), reverse=True)
    return ordered[:3]


def _match_registered_subtype(*values: Any) -> tuple[str, str, str]:
    for value in values:
        alias_key = _normalize_upper(value)
        if alias_key and alias_key in _REGISTRY_SUBTYPE_BY_ALIAS:
            return _REGISTRY_SUBTYPE_BY_ALIAS[alias_key]
    return "", "", ""


def canonicalize_fault(
    *,
    root_cause_label: Any,
    failure_type: Any,
    primary_dimension: Any,
    fault_name: Any,
    fault_descriptor: Any,
    top3_candidates: Any,
    is_unknown: Any = False,
) -> CanonicalFault:
    root_label = _normalize_upper(root_cause_label)
    signature_text = _normalize_label(fault_descriptor or fault_name)
    registered_family_id, registered_subtype_id, registered_subtype_label = _match_registered_subtype(
        root_label,
        fault_name,
        fault_descriptor,
    )
    family_id = registered_family_id or _family_from_keywords(root_label, failure_type, signature_text)
    if family_id == "unknown_anomaly":
        family_id = _family_from_dimension(_normalize_text(primary_dimension))
    if root_label in {"UNKNOWN", "UNKNOWN_FALLBACK", "UNKNOWN ANOMALY"} and family_id != "nominal":
        family_id = "unknown_anomaly"
    if root_label in {"REVIEW"}:
        family_id = "review_required"
    if root_label in {"OK", "NORMAL"}:
        family_id = "nominal"

    subtype_id = ""
    subtype_label = ""
    if registered_subtype_id:
        subtype_id = registered_subtype_id
        subtype_label = registered_subtype_label
    elif root_label and root_label not in GENERIC_SUBTYPES and root_label != family_id.upper():
        subtype_id = _slug(root_label, default="")
        subtype_label = _humanize_subtype(root_label)

    candidates = _candidate_family_candidates(top3_candidates)
    known_status = "known"
    if family_id == "unknown_anomaly" or bool(is_unknown):
        known_status = "unknown_with_similarity" if candidates else "unknown"
    elif not subtype_id:
        known_status = "known_family_unknown_subtype"

    if known_status not in KNOWN_STATUS_VALUES:
        known_status = "unknown"

    review_required = known_status != "known" or family_id == "review_required"
    return CanonicalFault(
        fault_family_id=family_id,
        fault_family_label=FAMILY_LABELS.get(family_id, _titleize(family_id)),
        fault_subtype_id=subtype_id,
        fault_subtype_label=subtype_label,
        fault_signature_id=_signature_id(signature_text),
        fault_signature_text=signature_text,
        known_status=known_status,
        similar_family_candidates=candidates,
        review_required=review_required,
    )
