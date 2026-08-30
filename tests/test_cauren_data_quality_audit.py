import csv
import json
from pathlib import Path

from tools.audit_cauren_data_quality import run_audit, violates_physical_limit
from tools.build_cauren_civil_dataset import build_cauren_civil_dataset


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
        writer.writerow(
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
            }
        )


def test_physical_limit_rule_flags_invalid_civil_ratio():
    assert violates_physical_limit("permit_status_score", 1.05) is True
    assert violates_physical_limit("permit_status_score", 0.95) is False


def test_run_audit_emits_gate_and_scorecard_on_civil_fixture(tmp_path: Path):
    source_csv = tmp_path / "normalized" / "civil_public_core_assets.csv"
    _write_public_core_csv(source_csv)
    dataset_root = tmp_path / "dataset"
    build_cauren_civil_dataset(source_csv=source_csv, output_dir=dataset_root, dataset_id="civil_fixture_v1")

    llm_dir = tmp_path / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)
    (llm_dir / "cauren_agent_sft_chat.jsonl").write_text(
        json.dumps({"record_id": "r1", "metadata": {"agent_id": "cauren-civil"}}) + "\n",
        encoding="utf-8",
    )

    report = run_audit([dataset_root], llm_dir, tmp_path / "out")

    assert "gates" in report
    assert "scorecard" in report
    assert report["scorecard"]["overall_fine_tune_readiness"] >= 0.0
    assert any(gate["dataset_surface"] == "dataset" for gate in report["gates"])
    assert report["remediation_backlog"] == []
