from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


REQUIRED_FEATURES: tuple[str, ...] = (
    "structural_risk_score",
    "inspection_finding_score",
    "permit_status_score",
    "construction_progress_pct",
    "infrastructure_connection_score",
    "natural_hazard_score",
    "occupancy_safety_score",
    "ground_stability_score",
)

OPTIONAL_FEATURES: tuple[str, ...] = (
    "building_height_m",
    "footprint_area_m2",
    "soil_settlement_mm",
    "utility_disruption_score",
)

ALL_FEATURES: tuple[str, ...] = REQUIRED_FEATURES + OPTIONAL_FEATURES

DEFAULT_SPLIT_SEQUENCE: tuple[str, ...] = ("train", "train", "train", "validation", "test")
DEFAULT_SOURCE_CSV = Path("data/public_sources/normalized/civil_public_core_assets.csv")
DEFAULT_OUTPUT_DIR = Path("data/cauren_civil")
DEFAULT_DATASET_ID = "cauren_civil_us_public_core_v1"
ANOMALY_THRESHOLD = 0.70


@dataclass(frozen=True)
class CivilAssetRecord:
    asset_id: str
    site_id: str
    split: str
    seq_len: int
    sampling_hz: float
    features: dict[str, float]
    metadata: dict[str, str]


def _load_dataset_id(dataset_dir: Path) -> str:
    summary_path = dataset_dir / "dataset_summary.json"
    if not summary_path.exists():
        return ""
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return str(payload.get("dataset_id") or "")


def _clamp(value: float, *, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _stable_index(text: str, modulo: int) -> int:
    total = 0
    for idx, char in enumerate(text):
        total += (idx + 1) * ord(char)
    return total % max(1, modulo)


def _parse_float(row: dict[str, str], field: str, *, default: float = 0.0) -> float:
    raw = str(row.get(field, "")).strip()
    if not raw:
        return default
    return float(raw)


def _normalize_record(row: dict[str, str]) -> CivilAssetRecord:
    asset_id = str(row.get("asset_id") or "").strip()
    if not asset_id:
        raise ValueError("asset_id is required")
    site_id = str(row.get("site_id") or asset_id).strip() or asset_id
    split = str(row.get("split") or "").strip().lower()
    if split not in {"train", "validation", "test"}:
        split = DEFAULT_SPLIT_SEQUENCE[_stable_index(asset_id, len(DEFAULT_SPLIT_SEQUENCE))]
    seq_len = max(4, int(float(row.get("seq_len") or 8)))
    sampling_hz = max(1.0, float(row.get("sampling_hz") or 1.0))

    features: dict[str, float] = {}
    for feature in REQUIRED_FEATURES:
        value = _parse_float(row, feature)
        if feature == "construction_progress_pct":
            value = _clamp(value, lower=0.0, upper=100.0)
        else:
            value = _clamp(value, lower=0.0, upper=1.0)
        features[feature] = value

    for feature in OPTIONAL_FEATURES:
        raw = str(row.get(feature, "")).strip()
        if not raw:
            continue
        value = float(raw)
        if feature in {"building_height_m", "footprint_area_m2", "soil_settlement_mm"}:
            value = max(0.0, value)
        else:
            value = _clamp(value, lower=0.0, upper=1.0)
        features[feature] = value

    metadata = {
        "asset_type": str(row.get("asset_type") or "civil_asset").strip() or "civil_asset",
        "source_bundle": str(row.get("source_bundle") or "us_public_core").strip() or "us_public_core",
        "jurisdiction": str(row.get("jurisdiction") or "unknown").strip() or "unknown",
    }
    return CivilAssetRecord(
        asset_id=asset_id,
        site_id=site_id,
        split=split,
        seq_len=seq_len,
        sampling_hz=sampling_hz,
        features=features,
        metadata=metadata,
    )


def _feature_unit(feature_name: str) -> str:
    if feature_name == "construction_progress_pct":
        return "%"
    if feature_name == "building_height_m":
        return "m"
    if feature_name == "footprint_area_m2":
        return "m2"
    if feature_name == "soil_settlement_mm":
        return "mm"
    return "ratio"


def _composite_risk(features: dict[str, float]) -> float:
    progress = _clamp(features.get("construction_progress_pct", 0.0) / 100.0, lower=0.0, upper=1.0)
    risk_terms = [
        _clamp(features.get("structural_risk_score", 0.0), lower=0.0, upper=1.0),
        _clamp(features.get("inspection_finding_score", 0.0), lower=0.0, upper=1.0),
        _clamp(features.get("natural_hazard_score", 0.0), lower=0.0, upper=1.0),
        1.0 - _clamp(features.get("permit_status_score", 0.0), lower=0.0, upper=1.0),
        1.0 - _clamp(features.get("infrastructure_connection_score", 0.0), lower=0.0, upper=1.0),
        1.0 - _clamp(features.get("occupancy_safety_score", 0.0), lower=0.0, upper=1.0),
        1.0 - _clamp(features.get("ground_stability_score", 0.0), lower=0.0, upper=1.0),
        1.0 - progress,
    ]
    return round(sum(risk_terms) / len(risk_terms), 6)


def _anomaly_family(features: dict[str, float]) -> tuple[str, str, str]:
    structural = features.get("structural_risk_score", 0.0)
    inspection = features.get("inspection_finding_score", 0.0)
    hazard = features.get("natural_hazard_score", 0.0)
    ground_gap = 1.0 - features.get("ground_stability_score", 1.0)
    permit_gap = 1.0 - features.get("permit_status_score", 1.0)
    progress_gap = 1.0 - _clamp(features.get("construction_progress_pct", 0.0) / 100.0, lower=0.0, upper=1.0)

    ranked = sorted(
        [
            ("structural_risk", structural, "structural_risk", "structural_ground_risk"),
            ("inspection_findings", inspection, "inspection_compliance", "inspection_findings"),
            ("natural_hazard", hazard, "natural_hazard", "hazard_exposure"),
            ("ground_instability", ground_gap, "ground_stability", "geotechnical_instability"),
            ("permit_gap", permit_gap, "permit_status", "permit_readiness_gap"),
            ("construction_delay", progress_gap, "construction_progress", "construction_delay"),
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    anomaly_type, _, root_cause_hint, anomaly_family = ranked[0]
    return anomaly_type, anomaly_family, root_cause_hint


def _window_rows(record: CivilAssetRecord, *, window_id: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for seq_index in range(record.seq_len):
        for feature_name in ALL_FEATURES:
            if feature_name not in record.features:
                continue
            rows.append(
                {
                    "window_id": window_id,
                    "agent_id": "cauren-civil",
                    "sector": "civil",
                    "asset_id": record.asset_id,
                    "client_id": "public_research",
                    "site_id": record.site_id,
                    "timestamp": f"2026-01-01T00:00:{seq_index:02d}Z",
                    "seq_index": str(seq_index),
                    "sensor_id": f"{record.asset_id}:{feature_name}",
                    "sensor_name": feature_name,
                    "unit": _feature_unit(feature_name),
                    "value": str(record.features[feature_name]),
                    "quality": "true",
                    "metadata_json": json.dumps(record.metadata, sort_keys=True),
                }
            )
    return rows


def build_cauren_civil_dataset(
    *,
    source_csv: Path,
    output_dir: Path,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> dict[str, object]:
    records: list[CivilAssetRecord] = []
    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(_normalize_record(row))

    if not records:
        raise ValueError("No civil asset records were found in source_csv.")

    agent_dir = output_dir / "agents" / "cauren-civil"
    splits_dir = output_dir / "splits"
    agent_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    windows_path = agent_dir / "windows.csv"
    labels_path = agent_dir / "labels.csv"
    readings_path = agent_dir / "raw_sensor_readings.csv"
    metadata_path = agent_dir / "asset_metadata.csv"

    window_ids_by_split: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    split_ids_by_role: dict[str, list[str]] = {
        "mae_reconstruction_train_windows.txt": [],
        "validation_windows.txt": [],
        "test_windows.txt": [],
        "train_windows.txt": [],
    }

    with windows_path.open("w", encoding="utf-8", newline="") as windows_handle, labels_path.open(
        "w", encoding="utf-8", newline=""
    ) as labels_handle, readings_path.open("w", encoding="utf-8", newline="") as readings_handle, metadata_path.open(
        "w", encoding="utf-8", newline=""
    ) as metadata_handle:
        windows_writer = csv.DictWriter(
            windows_handle,
            fieldnames=[
                "window_id",
                "agent_id",
                "sector",
                "asset_id",
                "client_id",
                "site_id",
                "window_start",
                "window_end",
                "sampling_hz",
                "seq_len",
                "sensor_count",
                "missing_ratio",
                "quality_score",
                "split",
                "training_role",
            ],
        )
        labels_writer = csv.DictWriter(
            labels_handle,
            fieldnames=[
                "window_id",
                "agent_id",
                "sector",
                "anomaly_type",
                "anomaly_family",
                "risk_score",
                "confidence",
                "label_source",
                "event_start",
                "event_end",
                "root_cause_hint",
                "notes",
                "is_anomaly",
            ],
        )
        readings_writer = csv.DictWriter(
            readings_handle,
            fieldnames=[
                "window_id",
                "agent_id",
                "sector",
                "asset_id",
                "client_id",
                "site_id",
                "timestamp",
                "seq_index",
                "sensor_id",
                "sensor_name",
                "unit",
                "value",
                "quality",
                "metadata_json",
            ],
        )
        metadata_writer = csv.DictWriter(
            metadata_handle,
            fieldnames=["asset_id", "agent_id", "sector", "metadata_json"],
        )
        windows_writer.writeheader()
        labels_writer.writeheader()
        readings_writer.writeheader()
        metadata_writer.writeheader()

        for idx, record in enumerate(records, start=1):
            window_id = f"civil-{idx:05d}"
            window_ids_by_split[record.split].append(window_id)
            if record.split == "train":
                split_ids_by_role["train_windows.txt"].append(window_id)
                split_ids_by_role["mae_reconstruction_train_windows.txt"].append(window_id)
            elif record.split == "validation":
                split_ids_by_role["validation_windows.txt"].append(window_id)
            else:
                split_ids_by_role["test_windows.txt"].append(window_id)

            windows_writer.writerow(
                {
                    "window_id": window_id,
                    "agent_id": "cauren-civil",
                    "sector": "civil",
                    "asset_id": record.asset_id,
                    "client_id": "public_research",
                    "site_id": record.site_id,
                    "window_start": "2026-01-01T00:00:00Z",
                    "window_end": f"2026-01-01T00:00:{record.seq_len - 1:02d}Z",
                    "sampling_hz": record.sampling_hz,
                    "seq_len": record.seq_len,
                    "sensor_count": len(record.features),
                    "missing_ratio": 0.0,
                    "quality_score": 1.0,
                    "split": record.split,
                    "training_role": "mae_reconstruction",
                }
            )
            risk_score = _composite_risk(record.features)
            anomaly_type, anomaly_family, root_cause_hint = _anomaly_family(record.features)
            is_anomaly = risk_score >= ANOMALY_THRESHOLD
            labels_writer.writerow(
                {
                    "window_id": window_id,
                    "agent_id": "cauren-civil",
                    "sector": "civil",
                    "anomaly_type": anomaly_type if is_anomaly else "nominal_variation",
                    "anomaly_family": anomaly_family if is_anomaly else "nominal",
                    "risk_score": risk_score,
                    "confidence": 0.85,
                    "label_source": "public_core_fusion_v1",
                    "event_start": "2026-01-01T00:00:01Z",
                    "event_end": f"2026-01-01T00:00:{max(1, record.seq_len - 2):02d}Z",
                    "root_cause_hint": root_cause_hint,
                    "notes": f"source_bundle={record.metadata['source_bundle']}",
                    "is_anomaly": str(is_anomaly).lower(),
                }
            )
            for reading_row in _window_rows(record, window_id=window_id):
                readings_writer.writerow(reading_row)
            metadata_writer.writerow(
                {
                    "asset_id": record.asset_id,
                    "agent_id": "cauren-civil",
                    "sector": "civil",
                    "metadata_json": json.dumps(record.metadata, sort_keys=True),
                }
            )

    for filename, window_ids in split_ids_by_role.items():
        (splits_dir / filename).write_text("".join(f"{window_id}\n" for window_id in window_ids), encoding="utf-8")

    summary = {
        "dataset_id": dataset_id,
        "agent_id": "cauren-civil",
        "sector": "civil",
        "source_csv": str(source_csv),
        "sampling_hz": 1.0,
        "asset_count": len(records),
        "window_count": len(records),
        "features": list(ALL_FEATURES),
        "required_features": list(REQUIRED_FEATURES),
        "optional_features": list(OPTIONAL_FEATURES),
        "splits": {name: len(ids) for name, ids in window_ids_by_split.items()},
        "license_posture": "public_research_reproducible",
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the canonical civil dataset for Cauren from normalized public-core records.")
    parser.add_argument("--source-csv", default=str(DEFAULT_SOURCE_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    args = parser.parse_args()

    summary = build_cauren_civil_dataset(
        source_csv=Path(args.source_csv),
        output_dir=Path(args.output_dir),
        dataset_id=args.dataset_id,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
