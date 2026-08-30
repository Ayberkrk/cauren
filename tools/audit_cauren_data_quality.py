from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


PHYSICAL_LIMITS: dict[str, tuple[float, float]] = {
    "structural_risk_score": (0.0, 1.0),
    "inspection_finding_score": (0.0, 1.0),
    "permit_status_score": (0.0, 1.0),
    "construction_progress_pct": (0.0, 100.0),
    "infrastructure_connection_score": (0.0, 1.0),
    "natural_hazard_score": (0.0, 1.0),
    "occupancy_safety_score": (0.0, 1.0),
    "ground_stability_score": (0.0, 1.0),
    "building_height_m": (0.0, 2000.0),
    "footprint_area_m2": (0.0, 10_000_000.0),
    "soil_settlement_mm": (0.0, 10_000.0),
    "utility_disruption_score": (0.0, 1.0),
}


def violates_physical_limit(feature_name: str, value: float) -> bool:
    limits = PHYSICAL_LIMITS.get(str(feature_name).strip())
    if not limits:
        return False
    lower, upper = limits
    return float(value) < lower or float(value) > upper


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _audit_dataset_surface(dataset_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    issues: list[dict[str, object]] = []
    gates: list[dict[str, object]] = []
    agent_dir = dataset_root / "agents" / "cauren-civil"
    windows_path = agent_dir / "windows.csv"
    labels_path = agent_dir / "labels.csv"
    readings_path = agent_dir / "raw_sensor_readings.csv"
    metadata_path = agent_dir / "asset_metadata.csv"

    required = [windows_path, labels_path, readings_path]
    missing = [path.name for path in required if not path.exists()]
    gates.append(
        {
            "dataset_surface": dataset_root.name,
            "status": "pass" if not missing else "fail",
            "missing_files": missing,
        }
    )
    for missing_name in missing:
        issues.append(
            {
                "issue_id": f"DQ-MISSING-{missing_name}",
                "dataset_surface": dataset_root.name,
                "severity": "high",
                "message": f"Missing required dataset file: {missing_name}",
            }
        )

    invalid_readings = 0
    if readings_path.exists():
        with readings_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    value = float(row.get("value", 0.0) or 0.0)
                except ValueError:
                    invalid_readings += 1
                    continue
                if violates_physical_limit(str(row.get("sensor_name") or ""), value):
                    invalid_readings += 1

    if invalid_readings:
        issues.append(
            {
                "issue_id": "DQ-PHYSICAL-LIMITS",
                "dataset_surface": dataset_root.name,
                "severity": "medium",
                "message": f"{invalid_readings} reading(s) violate configured civil physical limits.",
            }
        )

    summary = {
        "dataset_surface": dataset_root.name,
        "window_rows": _count_rows(windows_path),
        "label_rows": _count_rows(labels_path),
        "reading_rows": _count_rows(readings_path),
        "metadata_rows": _count_rows(metadata_path),
        "invalid_readings": invalid_readings,
    }
    return gates, issues, summary


def run_audit(dataset_roots: Iterable[Path], llm_dir: Path | None, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gates: list[dict[str, object]] = []
    remediation_backlog: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for dataset_root in dataset_roots:
        dataset_gates, issues, summary = _audit_dataset_surface(Path(dataset_root))
        gates.extend(dataset_gates)
        remediation_backlog.extend(issues)
        summaries.append(summary)

    llm_ready = 0
    if llm_dir is not None:
        llm_dir = Path(llm_dir)
        llm_ready = int((llm_dir / "cauren_agent_sft_chat.jsonl").exists())

    pass_ratio = 0.0
    if gates:
        pass_ratio = sum(1 for gate in gates if gate["status"] == "pass") / len(gates)
    issue_penalty = min(0.6, len(remediation_backlog) * 0.1)
    overall_readiness = max(0.0, round(pass_ratio - issue_penalty + (0.1 if llm_ready else 0.0), 4))

    report = {
        "gates": gates,
        "dataset_summaries": summaries,
        "scorecard": {
            "overall_fine_tune_readiness": overall_readiness,
            "dataset_surface_count": len(summaries),
            "llm_ready": bool(llm_ready),
        },
        "remediation_backlog": remediation_backlog,
    }
    (output_dir / "civil_data_quality_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
