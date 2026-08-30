import csv
import subprocess
from pathlib import Path

from cauren_core.training import CaurenCoreTrainConfig, train_cauren_core
from tools.build_cauren_civil_dataset import _load_dataset_id, build_cauren_civil_dataset


def _write_public_core_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "asset_id",
                "site_id",
                "split",
                "seq_len",
                "sampling_hz",
                "structural_risk_score",
                "inspection_finding_score",
                "permit_status_score",
                "construction_progress_pct",
                "infrastructure_connection_score",
                "natural_hazard_score",
                "occupancy_safety_score",
                "ground_stability_score",
                "building_height_m",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "asset_id": "asset-001",
                    "site_id": "site-01",
                    "split": "train",
                    "seq_len": "8",
                    "sampling_hz": "1",
                    "structural_risk_score": "0.82",
                    "inspection_finding_score": "0.76",
                    "permit_status_score": "0.35",
                    "construction_progress_pct": "41",
                    "infrastructure_connection_score": "0.52",
                    "natural_hazard_score": "0.66",
                    "occupancy_safety_score": "0.48",
                    "ground_stability_score": "0.39",
                    "building_height_m": "24.5",
                },
                {
                    "asset_id": "asset-002",
                    "site_id": "site-01",
                    "split": "validation",
                    "seq_len": "8",
                    "sampling_hz": "1",
                    "structural_risk_score": "0.25",
                    "inspection_finding_score": "0.21",
                    "permit_status_score": "0.92",
                    "construction_progress_pct": "88",
                    "infrastructure_connection_score": "0.90",
                    "natural_hazard_score": "0.31",
                    "occupancy_safety_score": "0.89",
                    "ground_stability_score": "0.84",
                    "building_height_m": "12.0",
                },
                {
                    "asset_id": "asset-003",
                    "site_id": "site-02",
                    "split": "test",
                    "seq_len": "8",
                    "sampling_hz": "1",
                    "structural_risk_score": "0.61",
                    "inspection_finding_score": "0.58",
                    "permit_status_score": "0.64",
                    "construction_progress_pct": "65",
                    "infrastructure_connection_score": "0.70",
                    "natural_hazard_score": "0.53",
                    "occupancy_safety_score": "0.68",
                    "ground_stability_score": "0.57",
                    "building_height_m": "30.0",
                },
            ]
        )


def test_build_dataset_uses_summary_dataset_id(tmp_path: Path):
    source_csv = tmp_path / "normalized" / "civil_public_core_assets.csv"
    _write_public_core_csv(source_csv)
    dataset_dir = tmp_path / "dataset"
    build_cauren_civil_dataset(source_csv=source_csv, output_dir=dataset_dir, dataset_id="cauren_civil_us_public_core_v1")
    assert _load_dataset_id(dataset_dir) == "cauren_civil_us_public_core_v1"


def test_dataset_build_outputs_training_contract_and_dry_run(tmp_path: Path):
    source_csv = tmp_path / "normalized" / "civil_public_core_assets.csv"
    _write_public_core_csv(source_csv)
    dataset_dir = tmp_path / "dataset"
    summary = build_cauren_civil_dataset(source_csv=source_csv, output_dir=dataset_dir, dataset_id="cauren_civil_us_public_core_v1")

    assert summary["asset_count"] == 3
    assert (dataset_dir / "agents" / "cauren-civil" / "windows.csv").exists()
    assert (dataset_dir / "agents" / "cauren-civil" / "labels.csv").exists()
    assert (dataset_dir / "agents" / "cauren-civil" / "raw_sensor_readings.csv").exists()

    training_summary = train_cauren_core(
        CaurenCoreTrainConfig(
            dataset_dir=dataset_dir,
            output_path=tmp_path / "out.pt",
            agents=("cauren-civil",),
            training_roles=("mae_reconstruction",),
            splits=("train",),
            validation_splits=("validation",),
            batch_size=2,
            epochs=1,
            dry_run=True,
        )
    )
    assert training_summary["dry_run"] is True
    assert training_summary["summaries"]["cauren-civil"]["input_dim"] == 12


def test_event_remediation_script_updates_labels(tmp_path: Path):
    source_csv = tmp_path / "normalized" / "civil_public_core_assets.csv"
    _write_public_core_csv(source_csv)
    dataset_dir = tmp_path / "dataset"
    build_cauren_civil_dataset(source_csv=source_csv, output_dir=dataset_dir, dataset_id="cauren_civil_us_public_core_v1")

    labels_path = dataset_dir / "agents" / "cauren-civil" / "labels.csv"
    rows = list(csv.DictReader(labels_path.open("r", encoding="utf-8", newline="")))
    rows[0]["event_start"] = "2026-01-01T00:00:00Z"
    rows[0]["event_end"] = "2026-01-01T00:00:07Z"
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    out_dir = tmp_path / "out"
    subprocess.run(
        [
            "python3",
            "tools/remediate_cauren_event_windows.py",
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(out_dir),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    output_rows = list(csv.DictReader((out_dir / "agents" / "cauren-civil" / "labels.csv").open("r", encoding="utf-8", newline="")))
    assert output_rows[0]["event_start"] != "2026-01-01T00:00:00Z"
    assert "event_window_auto" in output_rows[0]["label_source"]
