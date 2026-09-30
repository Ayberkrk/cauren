"""Audit NYC DOB label coverage, manual linkage review, and temporal leakage."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _wilson(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total == 0:
        return None
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = z * ((rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) ** 0.5) / denominator
    return [round(max(0.0, center - margin), 6), round(min(1.0, center + margin), 6)]


def _add_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(month=2, day=28, year=value.year + years)


def audit_dataset(
    dataset_dir: Path, *, minimum_reviewed: int = 57, minimum_decisive_matches: int = 20,
    minimum_precision: float = 0.98, minimum_precision_lower_bound: float = 0.90,
) -> dict[str, Any]:
    summary = json.loads((dataset_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    labels = _rows(dataset_dir / "label_ledger.csv")
    features = _rows(dataset_dir / "features.csv")
    certificates = _rows(dataset_dir / "certificates.csv")
    review = _rows(dataset_dir / "linkage_review_queue.csv")

    errors: list[str] = []
    warnings: list[str] = []
    label_status = Counter(row.get("label_status", "") for row in labels)
    valid_label_statuses = {
        "positive_co_issued_within_horizon",
        "no_co_issued_recorded_by_horizon",
        "right_censored_followup",
        "ambiguous_certificate_candidate",
        "certificate_type_unresolved",
        "source_snapshot_incomplete",
    }
    unresolved_statuses = {
        "right_censored_followup",
        "ambiguous_certificate_candidate",
        "certificate_type_unresolved",
        "source_snapshot_incomplete",
    }
    per_borough: dict[str, Counter] = defaultdict(Counter)
    asset_split: dict[str, str] = {}
    late_feature_rows = 0
    unversioned_feature_rows = 0
    feature_by_project = {row["building_id"]: row for row in features}
    certificate_by_id = {row["certificate_id"]: row for row in certificates}
    project_ids = {row["building_id"] for row in labels}
    feature_ids = set(feature_by_project)
    if project_ids != feature_ids:
        errors.append(f"feature/label project mismatch: {len(project_ids ^ feature_ids)} ids differ")
    if not summary.get("cohort_scope_matches_source_manifest", True):
        errors.append("findings source scope hash does not match the normalized new-building BIN cohort")
    if not summary.get("complete_extraction", False):
        warnings.append("one or more required source slices are incomplete for the new-building BIN cohort")

    for row in labels:
        project_id = row["building_id"]
        feature = feature_by_project.get(project_id, {})
        split = feature.get("split", "")
        asset = feature.get("asset_id", row.get("asset_id", ""))
        if asset in asset_split and asset_split[asset] != split:
            errors.append(f"BIN {asset} appears in both {asset_split[asset]} and {split}")
        asset_split[asset] = split
        label = row.get("co_issued_within_horizon", "")
        if label not in {"0", "1", ""}:
            errors.append(f"invalid label value {label!r} for {project_id}")
        if row.get("label_status") not in valid_label_statuses:
            errors.append(f"invalid label status {row.get('label_status')!r} for {project_id}")
        if row.get("label_status") in unresolved_statuses and label:
            errors.append(f"unresolved label status has a binary value: {project_id}")
        if label == "0" and row.get("followup_mature", "").lower() != "true":
            errors.append(f"immature project has a negative label: {project_id}")
        if label == "0" and row.get("label_status") != "no_co_issued_recorded_by_horizon":
            errors.append(f"negative label has inconsistent status: {project_id}")
        if label == "0" and not summary.get("complete_extraction", False):
            errors.append(f"negative label was emitted from an incomplete source snapshot: {project_id}")
        if label == "1" and row.get("match_level") not in {"bin_and_job", "bin_only"}:
            errors.append(f"positive label lacks a recognized BIN-level match: {project_id}")
        if label == "1" and row.get("label_status") != "positive_co_issued_within_horizon":
            errors.append(f"positive label has inconsistent status: {project_id}")
        if label == "1":
            try:
                anchor = date.fromisoformat(row["permit_issue_date"])
                horizon_end = date.fromisoformat(row["outcome_horizon_end"])
                matched_ids = json.loads(row.get("matched_certificate_ids", "[]"))
                has_qualifying_certificate = any(
                    certificate_by_id.get(certificate_id, {}).get("is_qualifying_co", "").lower() == "true"
                    and certificate_by_id.get(certificate_id, {}).get("asset_id") == row.get("asset_id")
                    and anchor <= date.fromisoformat(certificate_by_id[certificate_id]["issue_date"]) <= horizon_end
                    for certificate_id in matched_ids
                    if certificate_id in certificate_by_id
                )
                if not has_qualifying_certificate:
                    errors.append(f"positive label lacks a matched qualifying CO within the horizon: {project_id}")
            except (KeyError, ValueError, TypeError):
                errors.append(f"positive label has invalid certificate/date provenance: {project_id}")
        try:
            anchor = date.fromisoformat(row["permit_issue_date"])
            as_of = date.fromisoformat(row["as_of_date"])
            horizon_end = date.fromisoformat(row["outcome_horizon_end"])
            expected_horizon_end = _add_years(anchor, int(row["outcome_horizon_years"]))
            if horizon_end != expected_horizon_end:
                errors.append(f"outcome horizon does not match its anchor for {project_id}")
            mature_by_date = as_of >= horizon_end
            if row.get("followup_mature", "").lower() != str(mature_by_date).lower():
                errors.append(f"follow-up maturity flag does not match source as-of date for {project_id}")
        except (KeyError, ValueError, TypeError):
            errors.append(f"invalid label anchor, as-of date, or outcome horizon for {project_id}")
        if row.get("negative_label_interpretation") and label != "0":
            errors.append(f"negative-label explanation is attached to non-negative project: {project_id}")
        per_borough[feature.get("borough", "unknown")][row.get("label_status", "")] += 1
        try:
            anchor = date.fromisoformat(feature.get("permit_issue_date", ""))
            if anchor != date.fromisoformat(row.get("permit_issue_date", "")):
                errors.append(f"feature/label anchor mismatch for {project_id}")
        except ValueError:
            errors.append(f"missing/invalid feature anchor for {project_id}")
        removed_fields = json.loads(feature.get("feature_time_leakage_fields_json", "[]") or "[]")
        if removed_fields:
            late_feature_rows += 1
            if any(item.get("field") not in {"filing_date", "approved_date"} for item in removed_fields):
                errors.append(f"unsupported date leakage field was recorded for {project_id}")
        if feature.get("feature_temporal_status") == "dated_events_filtered_to_anchor_static_source_fields_unversioned":
            unversioned_feature_rows += 1
        else:
            errors.append(f"missing source-versioning status for {project_id}")
    if late_feature_rows:
        warnings.append(f"late-dated source fields were removed from {late_feature_rows} permit rows")
    if unversioned_feature_rows:
        warnings.append(
            f"{unversioned_feature_rows} feature rows include static fields from unversioned DOB records; field history must be checked before modeling"
        )
    missing_hazard = sum(not row.get("natural_hazard_score", "").strip() for row in features)
    if missing_hazard:
        warnings.append(f"natural-hazard score is missing for {missing_hazard} BINs; attach and audit FEMA/USGS spatial layers before modeling")
    hazard_vintage_rows = 0
    for feature in features:
        vintage = feature.get("natural_hazard_source_vintage", "").strip()
        if vintage:
            try:
                if date.fromisoformat(feature["permit_issue_date"]) < date.fromisoformat(vintage):
                    hazard_vintage_rows += 1
            except ValueError:
                errors.append(f"invalid hazard vintage date for {feature.get('building_id')}")
    if hazard_vintage_rows:
        warnings.append(f"FEMA hazard map vintage follows the permit anchor for {hazard_vintage_rows} BINs; use point-in-time flood map versions before historical modeling")
    permit_status_coverage = sum(bool(row.get("permit_status_score", "").strip()) for row in features)
    if permit_status_coverage < len(features):
        warnings.append(f"permit-status score is available for {permit_status_coverage} of {len(features)} BINs; missing values must remain explicit")
    finding_counts = summary.get("finding_counts", {})
    missing_disposition_counts = {
        key: int(value) for key, value in finding_counts.items()
        if key.endswith("_missing_disposition_date") and int(value)
    }
    if missing_disposition_counts:
        warnings.append(
            "some cohort violations or complaints have no disposition date and are treated as active proxies: "
            + ", ".join(f"{key}={value}" for key, value in sorted(missing_disposition_counts.items()))
        )
    for feature in features:
        for score_name in ("permit_status_score", "inspection_finding_score", "natural_hazard_score"):
            raw_value = feature.get(score_name, "").strip()
            if raw_value:
                try:
                    if not 0.0 <= float(raw_value) <= 1.0:
                        errors.append(f"{score_name} is outside [0,1] for {feature.get('building_id')}")
                except ValueError:
                    errors.append(f"{score_name} is not numeric for {feature.get('building_id')}")

    reviewed = [row for row in review if row.get("human_match_decision", "").strip()]
    confirmed = [row for row in reviewed if row["human_match_decision"].strip().lower() == "confirmed_match"]
    false_matches = [row for row in reviewed if row["human_match_decision"].strip().lower() == "false_match"]
    ambiguous = [row for row in reviewed if row["human_match_decision"].strip().lower() == "ambiguous"]
    confirmed_no_record = [row for row in reviewed if row["human_match_decision"].strip().lower() == "confirmed_no_record"]
    invalid_decisions = [
        row for row in reviewed
        if row["human_match_decision"].strip().lower() not in {
            "confirmed_match", "false_match", "ambiguous", "confirmed_no_record", "source_issue", "unresolved"
        }
    ]
    if invalid_decisions:
        errors.append(f"{len(invalid_decisions)} manual reviews have an unsupported decision")
    decided = len(confirmed) + len(false_matches)
    match_precision = len(confirmed) / decided if decided else None
    precision_interval = _wilson(len(confirmed), decided)
    if len(reviewed) < minimum_reviewed:
        warnings.append(f"manual linkage review is incomplete: {len(reviewed)} reviewed, {minimum_reviewed} required")
    if decided < minimum_decisive_matches:
        warnings.append(f"only {decided} reviewed candidate links are decisive; {minimum_decisive_matches} required")
    if match_precision is None or match_precision < minimum_precision:
        warnings.append(f"manual match precision has not met the configured {minimum_precision:.1%} point-estimate gate")
    if precision_interval is not None and precision_interval[0] < minimum_precision_lower_bound:
        warnings.append("manual match precision's 95% Wilson lower bound is below the configured gate")

    sorted_warnings = sorted(set(warnings))
    blocking_warnings = [
        warning for warning in sorted_warnings
        if not warning.startswith("late-dated source fields were removed")
    ]
    report = {
        "dataset_id": summary.get("dataset_id"),
        "building_count": len(labels),
        "label_status_counts": dict(label_status),
        "borough_label_status_counts": {borough: dict(counts) for borough, counts in sorted(per_borough.items())},
        "manual_linkage_review": {
            "sampled": len(review),
            "reviewed": len(reviewed),
            "confirmed_match": len(confirmed),
            "false_match": len(false_matches),
            "ambiguous": len(ambiguous),
            "confirmed_no_record": len(confirmed_no_record),
            "decisive_match_precision": match_precision,
            "decisive_match_precision_wilson_95": precision_interval,
            "minimum_reviewed_gate": minimum_reviewed,
            "minimum_decisive_matches_gate": minimum_decisive_matches,
            "minimum_precision_gate": minimum_precision,
            "minimum_precision_wilson_lower_bound_gate": minimum_precision_lower_bound,
            "gate_passed": len(reviewed) >= minimum_reviewed and match_precision is not None
            and decided >= minimum_decisive_matches and match_precision >= minimum_precision
            and precision_interval is not None and precision_interval[0] >= minimum_precision_lower_bound,
        },
        "all_bin_groups_single_split": not any("appears in both" in error for error in errors),
        "source_extraction_complete": summary.get("complete_extraction", False),
        "explicit_late_date_rows_removed": late_feature_rows,
        "source_record_version_unavailable_rows": unversioned_feature_rows,
        "natural_hazard_missing_rows": missing_hazard,
        "natural_hazard_vintage_after_anchor_rows": hazard_vintage_rows,
        "permit_status_score_coverage": permit_status_coverage,
        "finding_rows_missing_disposition_date": missing_disposition_counts,
        "errors": errors,
        "warnings": sorted_warnings,
        "blocking_warnings": blocking_warnings,
        "dataset_ready_for_modeling": not errors and not blocking_warnings and summary.get("complete_extraction", False),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/cauren_civil_nyc/normalized"))
    parser.add_argument("--output-json", type=Path, help="Optional path for the audit report")
    parser.add_argument("--minimum-reviewed", type=int, default=57)
    parser.add_argument("--minimum-decisive-matches", type=int, default=20)
    parser.add_argument("--minimum-precision", type=float, default=0.98)
    parser.add_argument("--minimum-precision-lower-bound", type=float, default=0.90)
    args = parser.parse_args()
    report = audit_dataset(
        args.dataset_dir,
        minimum_reviewed=args.minimum_reviewed,
        minimum_decisive_matches=args.minimum_decisive_matches,
        minimum_precision=args.minimum_precision,
        minimum_precision_lower_bound=args.minimum_precision_lower_bound,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if report["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
