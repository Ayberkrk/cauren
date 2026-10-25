import csv
import json
from pathlib import Path

import pytest

from cauren_core.training import CaurenCoreTrainConfig, train_cauren_core
from tools.audit_nyc_civil_dataset import audit_dataset
from cauren_core.training import require_nyc_training_eligibility


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_dataset(path: Path, *, temporal_status: str, hazard_vintage: str, review_count: int) -> None:
    path.mkdir(parents=True)
    (path / "dataset_summary.json").write_text(
        json.dumps(
            {
                "dataset_id": "cauren_civil_nyc_fixture",
                "complete_extraction": True,
                "cohort_scope_matches_source_manifest": True,
                "target_is_engineering_safety_truth": False,
                "finding_counts": {},
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        path / "label_ledger.csv",
        [
            "building_id", "asset_id", "co_issued_within_horizon", "label_status", "followup_mature",
            "permit_issue_date", "as_of_date", "outcome_horizon_end", "outcome_horizon_years",
            "negative_label_interpretation", "match_level", "matched_certificate_ids",
        ],
        [
            {
                "building_id": "building-1",
                "asset_id": "1234567",
                "co_issued_within_horizon": "0",
                "label_status": "no_co_issued_recorded_by_horizon",
                "followup_mature": "true",
                "permit_issue_date": "2020-01-01",
                "as_of_date": "2026-01-01",
                "outcome_horizon_end": "2025-01-01",
                "outcome_horizon_years": "5",
                "negative_label_interpretation": "mature_followup_no_qualifying_co",
                "match_level": "",
                "matched_certificate_ids": "[]",
            },
            {
                "building_id": "building-2",
                "asset_id": "1234568",
                "co_issued_within_horizon": "1",
                "label_status": "positive_co_issued_within_horizon",
                "followup_mature": "true",
                "permit_issue_date": "2020-01-01",
                "as_of_date": "2026-01-01",
                "outcome_horizon_end": "2025-01-01",
                "outcome_horizon_years": "5",
                "negative_label_interpretation": "",
                "match_level": "bin_and_job",
                "matched_certificate_ids": '["cert-1"]',
            },
        ],
    )
    _write_csv(
        path / "features.csv",
        [
            "building_id", "asset_id", "borough", "split", "permit_issue_date", "feature_temporal_status",
            "feature_time_leakage_fields_json", "natural_hazard_score", "natural_hazard_source_vintage",
            "permit_status_score",
        ],
        [
            {
                "building_id": "building-1",
                "asset_id": "1234567",
                "borough": "1",
                "split": "train",
                "permit_issue_date": "2020-01-01",
                "feature_temporal_status": temporal_status,
                "feature_time_leakage_fields_json": "[]",
                "natural_hazard_score": "0.25",
                "natural_hazard_source_vintage": hazard_vintage,
                "permit_status_score": "0.5",
            },
            {
                "building_id": "building-2",
                "asset_id": "1234568",
                "borough": "1",
                "split": "train",
                "permit_issue_date": "2020-01-01",
                "feature_temporal_status": temporal_status,
                "feature_time_leakage_fields_json": "[]",
                "natural_hazard_score": "0.3",
                "natural_hazard_source_vintage": hazard_vintage,
                "permit_status_score": "0.6",
            },
        ],
    )
    _write_csv(
        path / "certificates.csv",
        ["certificate_id", "asset_id", "is_qualifying_co", "issue_date"],
        [{"certificate_id": "cert-1", "asset_id": "1234568", "is_qualifying_co": "true", "issue_date": "2022-05-01"}],
    )
    _write_csv(
        path / "linkage_review_queue.csv",
        ["human_match_decision"],
        [{"human_match_decision": "confirmed_match"} for _ in range(review_count)],
    )


def test_nyc_training_is_blocked_until_review_and_point_in_time_gates_pass(tmp_path: Path):
    dataset = tmp_path / "dataset"
    _make_dataset(
        dataset,
        temporal_status="dated_events_filtered_to_anchor_static_source_fields_unversioned",
        hazard_vintage="2021-10-13",
        review_count=0,
    )

    report = audit_dataset(dataset)

    eligibility = report["training_eligibility"]
    assert eligibility["eligible"] is False
    assert eligibility["status"] == "blocked"
    assert "manual_linkage_review" in eligibility["blockers"]
    assert "source_fields_point_in_time" in eligibility["blockers"]
    assert "hazard_vintage_point_in_time" in eligibility["blockers"]
    assert report["dataset_ready_for_modeling"] is False


def test_nyc_training_can_only_pass_with_verified_source_and_review_evidence(tmp_path: Path):
    dataset = tmp_path / "dataset"
    _make_dataset(
        dataset,
        temporal_status="all_fields_proven_available_as_of_anchor",
        hazard_vintage="2019-01-01",
        review_count=57,
    )

    report = audit_dataset(dataset)

    assert report["training_eligibility"]["eligible"] is True
    assert report["training_eligibility"]["blockers"] == []
    assert report["dataset_ready_for_modeling"] is True


def test_cauren_trainer_refuses_nyc_dataset_without_passing_audit(tmp_path: Path):
    (tmp_path / "dataset_summary.json").write_text(
        json.dumps({"dataset_id": "cauren_civil_nyc_new_building_bin_co_v2"}),
        encoding="utf-8",
    )

    config = CaurenCoreTrainConfig(dataset_dir=tmp_path, output_path=tmp_path / "checkpoint.pt", dry_run=True)
    with pytest.raises(ValueError, match="run tools/audit_nyc_civil_dataset.py"):
        require_nyc_training_eligibility(tmp_path)
    with pytest.raises(ValueError, match="run tools/audit_nyc_civil_dataset.py"):
        train_cauren_core(config)

    (tmp_path / "modeling_eligibility.json").write_text(
        json.dumps({"training_eligibility": {"eligible": False, "blockers": ["manual_linkage_review"]}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manual_linkage_review"):
        require_nyc_training_eligibility(tmp_path)

    (tmp_path / "modeling_eligibility.json").write_text(
        json.dumps({"training_eligibility": {"eligible": True, "blockers": []}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stale"):
        require_nyc_training_eligibility(tmp_path)


def test_direct_trainer_accepts_only_a_current_passing_nyc_audit(tmp_path: Path):
    dataset = tmp_path / "nyc_dataset"
    _make_dataset(
        dataset,
        temporal_status="all_fields_proven_available_as_of_anchor",
        hazard_vintage="2019-01-01",
        review_count=57,
    )
    report = audit_dataset(dataset)
    (dataset / "modeling_eligibility.json").write_text(json.dumps(report), encoding="utf-8")

    agent_dir = dataset / "agents" / "cauren-civil"
    agent_dir.mkdir(parents=True)
    _write_csv(agent_dir / "windows.csv", ["window_id", "split", "training_role"], [
        {"window_id": "window-1", "split": "train", "training_role": "mae_reconstruction"},
    ])
    _write_csv(agent_dir / "labels.csv", ["window_id", "is_anomaly"], [
        {"window_id": "window-1", "is_anomaly": "false"},
    ])
    required_features = (
        "structural_risk_score", "inspection_finding_score", "permit_status_score",
        "construction_progress_pct", "infrastructure_connection_score", "natural_hazard_score",
        "occupancy_safety_score", "ground_stability_score",
    )
    _write_csv(agent_dir / "raw_sensor_readings.csv", ["window_id", "seq_index", "sensor_name", "value", "quality"], [
        {"window_id": "window-1", "seq_index": "0", "sensor_name": feature, "value": "0.5", "quality": "true"}
        for feature in required_features
    ])

    result = train_cauren_core(
        CaurenCoreTrainConfig(
            dataset_dir=dataset,
            output_path=tmp_path / "unused.pt",
            dry_run=True,
        )
    )

    assert report["training_eligibility"]["eligible"] is True
    assert result["summaries"]["cauren-civil"]["status"] == "dry_run"

    feature_file = dataset / "features.csv"
    feature_file.write_text(feature_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="audited input changed"):
        train_cauren_core(
            CaurenCoreTrainConfig(
                dataset_dir=dataset,
                output_path=tmp_path / "stale.pt",
                dry_run=True,
            )
        )
