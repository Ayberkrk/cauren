"""Build a narrow, provenance-preserving NYC DOB permit-to-CO dataset.

The initial cohort contains new-building permit episodes. The target is a
final Certificate of Occupancy recorded within a fixed horizon after permit
issuance. Missing links, immature follow-up, temporary-only certificates, and
ambiguous BIN-only candidates remain explicit in the label ledger.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


EXPECTED_SOURCES = {
    "bis_permits": "ipu4-2q9a",
    "dob_now_permits": "rbx6-tga4",
    "dob_now_filings": "w9ak-ipjd",
    "bis_certificates": "bs8b-p36w",
    "dob_now_certificates": "pkdm-hqz6",
    "dob_violations": "3h2n-5cm9",
    "dob_complaints": "eabe-havv",
}
FEATURE_FIELDS = (
    "job_type", "work_type", "permit_type", "permit_subtype", "building_type",
    "residential", "estimated_job_costs", "total_construction_floor_area",
    "existing_no_of_stories", "proposed_no_of_stories", "filing_date", "approved_date",
)
DENIED_CERTIFICATE_STATUSES = ("cancel", "reject", "withdraw", "void", "denied")
FINAL_CERTIFICATE_RE = re.compile(r"\bFINAL\b", re.IGNORECASE)
TEMP_CERTIFICATE_RE = re.compile(r"\b(TEMP|TEMPORARY|INTERIM|TCO)\b", re.IGNORECASE)


def _first(row: dict[str, Any], names: Iterable[str]) -> str:
    by_lower = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        value = by_lower.get(name.lower())
        if value is not None and str(value).strip() not in {"", "None", "null"}:
            return str(value).strip()
    return ""


def _key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _job_ids(row: dict[str, Any], names: Iterable[str]) -> list[str]:
    values = []
    by_lower = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        value = by_lower.get(name.lower())
        if value is None or str(value).strip() in {"", "None", "null"}:
            continue
        normalized = _key(str(value))
        if normalized:
            values.append(normalized)
            # DOB NOW filing ids append a filing/sequence suffix such as
            # -I1 or -S4, while related CO rows can carry the root filing id.
            root = re.sub(r"^([A-Z]\d{6,10})(?:I|S)\d$", r"\1", normalized)
            if root != normalized:
                values.append(root)
    return list(dict.fromkeys(values))


def _bin(value: str) -> str:
    cleaned = value.strip()
    if cleaned.endswith(".0"):
        cleaned = cleaned[:-2]
    digits = re.sub(r"\D", "", cleaned)
    return digits.zfill(7) if 0 < len(digits) <= 7 else digits


def _parse_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _borough_code(value: str) -> str:
    value = value.strip().upper()
    aliases = {"MANHATTAN": "1", "BRONX": "2", "BROOKLYN": "3", "QUEENS": "4", "STATEN ISLAND": "5"}
    if value in aliases:
        return aliases[value]
    return value if value in {"1", "2", "3", "4", "5"} else ""


def _bbl(row: dict[str, Any]) -> str:
    full = re.sub(r"\D", "", _first(row, ("bbl", "bbl_number")))
    if len(full) == 10:
        return full
    borough = _borough_code(_first(row, ("borough", "boro")))
    block = re.sub(r"\D", "", _first(row, ("block",)))
    lot = re.sub(r"\D", "", _first(row, ("lot",)))
    if borough and block and lot:
        return f"{borough}{block.zfill(5)}{lot.zfill(4)}"
    return ""


def _parse_date(value: str) -> date | None:
    value = value.strip()
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
        return parsed.date()
    except ValueError:
        pass
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y", "%Y%m%d"):
        try:
            return datetime.strptime(value[:10], fmt).date()
        except ValueError:
            continue
    for fmt in ("%m/%d/%y %I:%M:%S %p", "%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _add_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(month=2, day=28, year=value.year + years)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_number}")
            yield row


def _is_new_building(row: dict[str, Any]) -> bool:
    values = [
        re.sub(r"[^A-Z0-9]", "", _first(row, (name,)).upper())
        for name in ("job_type", "work_type", "permit_type", "permit_subtype")
    ]
    return any(value == "NB" or value in {"NEWBUILDING", "NEWBLDG"} for value in values)


def _normalize_permit(source_key: str, row: dict[str, Any]) -> dict[str, Any] | None:
    identifiers = _job_ids(row, ("tracking_number", "job__", "job_number", "job_filing_number", "job_filing_no", "job_no"))
    job = identifiers[0] if identifiers else ""
    if source_key == "dob_now_permits":
        filing_identifiers = _job_ids(row, ("job_filing_number",))
        if filing_identifiers:
            # Multiple DOB NOW work permits can refer to one filing. Group
            # them by the root filing identifier and retain tracking numbers
            # as alternate keys for CO linkage.
            job = filing_identifiers[-1]
    bin_number = _bin(_first(row, ("bin__", "bin_number", "bin")))
    issue_date = _parse_date(
        _first(row, ("issuance_date", "issued_date", "first_permit_date", "permit_issue_date", "issue_date"))
    )
    if not job or not _valid_bin(bin_number) or not issue_date or not _is_new_building(row):
        return None
    project_id = hashlib.sha256(f"{bin_number}|{job}".encode("utf-8")).hexdigest()[:20]
    normalized_features: dict[str, str] = {}
    aliases = {
        "job_type": ("job_type",),
        "work_type": ("work_type",),
        "permit_type": ("permit_type",),
        "permit_subtype": ("permit_subtype",),
        "building_type": ("bldg_type", "building_type"),
        "residential": ("residential",),
        "estimated_job_costs": ("estimated_job_costs", "initial_cost", "total_cost", "estimated_job_cost"),
        "total_construction_floor_area": ("total_construction_floor_area", "total_construction_floor_area_sqft"),
        "existing_no_of_stories": ("existing_no_of_stories", "existing_stories"),
        "proposed_no_of_stories": ("proposed_no_of_stories", "proposed_stories"),
        "filing_date": ("filing_date", "pre_filing_date"),
        "approved_date": ("approved_date",),
    }
    for field, field_aliases in aliases.items():
        normalized_features[field] = _first(row, field_aliases)
    filing_date = _parse_date(normalized_features["filing_date"])
    approval_date = _parse_date(normalized_features["approved_date"])
    late_feature_dates: list[dict[str, str]] = []
    if filing_date and filing_date > issue_date:
        late_feature_dates.append({"field": "filing_date", "value": filing_date.isoformat()})
        normalized_features["filing_date"] = ""
    if approval_date and approval_date > issue_date:
        late_feature_dates.append({"field": "approved_date", "value": approval_date.isoformat()})
        normalized_features["approved_date"] = ""
    return {
        "project_id": project_id,
        "asset_id": bin_number,
        "job_id": job,
        "alternate_job_ids": identifiers,
        "bbl": _bbl(row),
        "borough": _borough_code(_first(row, ("borough", "boro"))) or (bin_number[0] if bin_number[:1] in "12345" else ""),
        "permit_source": source_key,
        "permit_issue_date": issue_date,
        "feature_source_fields": normalized_features,
        "feature_time_leakage_fields": late_feature_dates,
        "raw_job_type": _first(row, ("job_type", "work_type")),
        "latitude": _first(row, ("gis_latitude", "latitude")),
        "longitude": _first(row, ("gis_longitude", "longitude")),
        "raw_permit_status": _first(row, ("permit_status", "filing_status")),
    }


def _normalize_certificate(source_key: str, row: dict[str, Any]) -> dict[str, Any]:
    doc_type = _first(row, ("issue_type", "certificate_type", "co_type"))
    filing_type = _first(row, ("c_of_o_filing_type",))
    status = _first(row, ("application_status_raw", "filing_status_raw", "c_of_o_status", "status"))
    issue_date = _parse_date(
        _first(row, ("c_o_issue_date", "c_of_o_issuance_date", "co_issue_date", "issuance_date", "issue_date"))
    )
    job_raw = _first(row, ("job_number", "job__", "job_filing_number", "job_no"))
    job_field = "job_number_or_filing_number"
    if not job_raw:
        job_raw = _first(row, ("job_filing_name",))
        job_field = "job_filing_name"
    identifiers = _job_ids(
        row,
        ("job_number", "job__", "job_filing_number", "job_no", "job_filing_name"),
    )
    job = _key(job_raw)
    if not job and identifiers:
        job = identifiers[0]
    bin_number = _bin(_first(row, ("bin_number", "bin__", "bin")))
    denied = any(word in status.lower() for word in DENIED_CERTIFICATE_STATUSES)
    is_now_feed = source_key == "dob_now_certificates"
    now_filing_type = filing_type.strip().lower()
    is_qualifying_co = (
        bool(re.search(r"\bCO\s+ISSUED\b", status, re.IGNORECASE))
        and not re.search(r"\bTCO\b|TEMPORARY|INTERIM", status, re.IGNORECASE)
        and now_filing_type in {"initial", "final", "amendment", "amended"}
        if is_now_feed
        else bool(FINAL_CERTIFICATE_RE.search(doc_type)) and not TEMP_CERTIFICATE_RE.search(doc_type)
    )
    is_temporary = (
        bool(re.search(r"\bTCO\b|TEMPORARY|INTERIM", status, re.IGNORECASE))
        if is_now_feed
        else bool(TEMP_CERTIFICATE_RE.search(doc_type))
    )
    return {
        "certificate_id": hashlib.sha256(
            f"{source_key}|{job}|{bin_number}|{issue_date}|{doc_type}|{status}".encode("utf-8")
        ).hexdigest()[:20],
        "certificate_source": source_key,
        "job_id": job,
        "alternate_job_ids": identifiers,
        "job_id_key_field": job_field if job else "",
        "asset_id": bin_number,
        "bbl": _bbl(row),
        "issue_date": issue_date,
        "document_type": doc_type,
        "filing_type": filing_type,
        "status": status,
        "issued_record": issue_date is not None and not denied,
        "is_qualifying_co": issue_date is not None and not denied and is_qualifying_co,
        "is_temporary_co": issue_date is not None and not denied and is_temporary,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _valid_bin(value: str) -> bool:
    return len(value) == 7 and value[0] in "12345" and value.isdigit()


def _normalize_finding(source_key: str, row: dict[str, Any]) -> dict[str, Any] | None:
    is_violation = source_key == "dob_violations"
    asset_id = _bin(_first(row, ("bin", "bin_number", "bin__")))
    issue_date = _parse_date(_first(row, ("issue_date",) if is_violation else ("date_entered",)))
    if not _valid_bin(asset_id) or not issue_date:
        return None
    finding_id = _first(row, ("isn_dob_bis_viol", ":id", "violation_number") if is_violation else ("complaint_number", ":id"))
    disposition = _parse_date(_first(row, ("disposition_date",)))
    return {
        "finding_id": finding_id or hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:20],
        "asset_id": asset_id,
        "source": source_key,
        "issue_date": issue_date,
        "disposition_date": disposition,
        "category": _first(row, ("violation_category", "violation_type", "complaint_category")),
    }


def _status_value(status: str) -> float | None:
    normalized = re.sub(r"[^A-Z0-9 ]", " ", status.upper())
    if re.search(r"OBJECT|ON HOLD|INCOMPLETE|WITHDRAW|REJECT|DENIED|CANCEL|DISAPPROV", normalized):
        return 1.0
    if re.search(r"APPROVED|PERMIT ISSUED|LOC ISSUED|CO ISSUED|PERMIT ENTIRE|SIGNED OFF", normalized):
        return 0.0
    return None


def _filing_status_score(building: dict[str, Any], filings_by_bin: dict[str, list[dict[str, Any]]]) -> tuple[float | None, int, int]:
    anchor = building["permit_issue_date"]
    matched_jobs = set(building["alternate_job_ids"])
    latest_by_filing: dict[str, tuple[date, float | None]] = {}
    unknown = 0
    for filing in filings_by_bin.get(building["asset_id"], []):
        if not matched_jobs.intersection(filing["alternate_job_ids"]):
            continue
        if not filing["filing_date"] or filing["filing_date"] > anchor:
            continue
        if not filing["status_date"] or filing["status_date"] > anchor:
            continue
        score = _status_value(filing["status"])
        key = filing["filing_id"] or ",".join(filing["alternate_job_ids"])
        prior = latest_by_filing.get(key)
        if prior is None or filing["status_date"] >= prior[0]:
            latest_by_filing[key] = (filing["status_date"], score)
    values = [value for _, value in latest_by_filing.values() if value is not None]
    unknown = sum(value is None for _, value in latest_by_filing.values())
    return (sum(values) / len(values) if values else None, len(values), unknown)


def _review_sample(
    buildings: list[dict[str, Any]], candidates_by_building: dict[str, list[dict[str, Any]]],
    labels_by_building: dict[str, dict[str, Any]], sample_size: int,
) -> list[dict[str, Any]]:
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for building in buildings:
        label = labels_by_building[building["building_id"]]
        stratum = f"{building['borough']}:{label['match_level']}:{label['label_status']}"
        strata[stratum].append(building)
    rng = random.Random(42)
    selected: list[dict[str, Any]] = []
    names = sorted(strata)
    while len(selected) < min(sample_size, len(buildings)) and names:
        progressed = False
        for name in names:
            if strata[name]:
                row = strata[name].pop(rng.randrange(len(strata[name])))
                selected.append(row)
                progressed = True
                if len(selected) >= sample_size:
                    break
        if not progressed:
            break
    result = []
    for building in selected:
        label = labels_by_building[building["building_id"]]
        candidates = candidates_by_building[building["building_id"]]
        result.append({
            "building_id": building["building_id"],
            "asset_id": building["asset_id"],
            "job_ids_json": json.dumps(building["alternate_job_ids"]),
            "borough": building["borough"],
            "match_group": label["match_level"],
            "permit_issue_date": building["permit_issue_date"].isoformat(),
            "label_status": label["label_status"],
            "co_issued_within_horizon": label["co_issued_within_horizon"],
            "candidate_certificates_json": json.dumps([
                {key: (value.isoformat() if isinstance(value, date) else value)
                 for key, value in certificate.items()
                 if key in {"certificate_source", "job_id", "alternate_job_ids", "asset_id", "bbl", "issue_date",
                            "document_type", "filing_type", "status", "is_qualifying_co", "is_temporary_co",
                            "match_level", "bbl_conflict"}}
                for certificate in candidates
            ], ensure_ascii=False),
            "human_match_decision": "",
            "reviewer_notes": "",
        })
    return result


def build_dataset(*, raw_dir: Path, output_dir: Path, horizon_years: int, as_of: date | None, review_sample_size: int) -> dict[str, Any]:
    manifest_path = raw_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_manifest = manifest.get("sources", {})
    source_ids_ok = {
        key: source_manifest[key].get("dataset_id") == expected
        for key, expected in EXPECTED_SOURCES.items() if key in source_manifest
    }
    if not all(source_ids_ok.values()):
        raise ValueError(f"Source manifest contains unexpected dataset ids: {source_ids_ok}")
    if as_of is None:
        starts = [source_manifest[key].get("snapshot_started_at_utc", "") for key in source_ids_ok]
        # The feeds expose dates without reliable row-level snapshot times.
        # Use the calendar day before the earliest extraction began so a
        # same-day record cannot be treated as observed or mature by inference.
        as_of = (
            min(datetime.fromisoformat(value.replace("Z", "+00:00")).date() for value in starts if value) - timedelta(days=1)
            if any(starts) else date.fromisoformat(manifest["started_at_utc"][:10]) - timedelta(days=1)
        )

    now_filing_by_key: dict[str, dict[str, str]] = {}
    filings_by_bin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    permit_counts = Counter()
    for raw_row in _iter_jsonl(raw_dir / source_manifest["dob_now_filings"]["raw_file"]):
        permit_counts["dob_now_filing_rows"] += 1
        filing = {
            "job_type": _first(raw_row, ("job_type", "job_type_description")),
            "first_permit_date": _first(raw_row, ("first_permit_date",)),
            "filing_date": _first(raw_row, ("filing_date",)),
            "approved_date": _first(raw_row, ("approved_date",)),
            "building_type": _first(raw_row, ("building_type",)),
            "total_construction_floor_area": _first(raw_row, ("total_construction_floor_area",)),
        }
        ids = _job_ids(raw_row, ("job_filing_number", "tracking_number", "job__", "job_number"))
        for key in ids:
            now_filing_by_key.setdefault(key, filing)
        bin_number = _bin(_first(raw_row, ("bin", "bin_number")))
        if _valid_bin(bin_number):
            filing_row = {
                "asset_id": bin_number,
                "filing_id": _key(_first(raw_row, ("job_filing_number",))),
                "alternate_job_ids": ids,
                "filing_date": _parse_date(_first(raw_row, ("filing_date",))),
                "status_date": _parse_date(_first(raw_row, ("current_status_date",))),
                "status": _first(raw_row, ("filing_status",)),
            }
            filings_by_bin[bin_number].append(filing_row)

    permits_by_project: dict[str, dict[str, Any]] = {}
    for source_key in ("bis_permits", "dob_now_permits"):
        for raw_row in _iter_jsonl(raw_dir / source_manifest[source_key]["raw_file"]):
            permit_counts[f"{source_key}_rows"] += 1
            normalized_row = raw_row
            if source_key == "dob_now_permits":
                filing = None
                for identifier in ("job_filing_number", "tracking_number", "job_number"):
                    job_keys = _job_ids(raw_row, (identifier,))
                    matched_key = next((key for key in job_keys if key in now_filing_by_key), None)
                    if matched_key:
                        filing = now_filing_by_key[matched_key]
                        break
                if filing:
                    normalized_row = dict(raw_row)
                    normalized_row["job_type"] = filing["job_type"]
                    for field, value in filing.items():
                        if value and not _first(normalized_row, (field,)):
                            normalized_row[field] = value
                    permit_counts["dob_now_permits_classified_from_job_filing"] += 1
            permit = _normalize_permit(source_key, normalized_row)
            if permit is None:
                permit_counts[f"{source_key}_excluded"] += 1
                continue
            permit_counts[f"{source_key}_new_building_rows"] += 1
            old = permits_by_project.get(permit["project_id"])
            if old is None:
                permits_by_project[permit["project_id"]] = permit
            else:
                permit_counts["duplicate_permit_rows"] += 1
                merged_sources = ",".join(sorted(set(old["permit_source"].split(",")) | {source_key}))
                if permit["permit_issue_date"] < old["permit_issue_date"]:
                    permit["permit_source"] = merged_sources
                    permits_by_project[permit["project_id"]] = permit
                else:
                    old["permit_source"] = merged_sources

    permits = list(permits_by_project.values())
    permits_by_bin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for permit in permits:
        permits_by_bin[permit["asset_id"]].append(permit)
    buildings: list[dict[str, Any]] = []
    for asset_id, bin_permits in permits_by_bin.items():
        first_permit = min(bin_permits, key=lambda row: (row["permit_issue_date"], row["project_id"]))
        job_ids = sorted({job for row in bin_permits for job in row["alternate_job_ids"]})
        source_names = sorted({source for row in bin_permits for source in row["permit_source"].split(",")})
        buildings.append({
            "building_id": hashlib.sha256(asset_id.encode("utf-8")).hexdigest()[:20],
            "asset_id": asset_id,
            "permit_issue_date": first_permit["permit_issue_date"],
            "job_id": first_permit["job_id"],
            "alternate_job_ids": job_ids,
            "bbl": first_permit["bbl"],
            "borough": first_permit["borough"],
            "permit_source": ",".join(source_names),
            "permit_episode_count": len(bin_permits),
            "feature_source_fields": first_permit["feature_source_fields"],
            "feature_time_leakage_fields": first_permit["feature_time_leakage_fields"],
            "latitude": first_permit["latitude"],
            "longitude": first_permit["longitude"],
        })
    buildings.sort(key=lambda row: row["asset_id"])

    certificates: list[dict[str, Any]] = []
    certificate_counts = Counter()
    for source_key in ("bis_certificates", "dob_now_certificates"):
        if source_key not in source_manifest:
            continue
        for raw_row in _iter_jsonl(raw_dir / source_manifest[source_key]["raw_file"]):
            certificate_counts[f"{source_key}_rows"] += 1
            certificate = _normalize_certificate(source_key, raw_row)
            if _valid_bin(certificate["asset_id"]):
                certificates.append(certificate)
            else:
                certificate_counts["certificates_without_valid_bin"] += 1
    certificates_by_bin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for certificate in certificates:
        certificates_by_bin[certificate["asset_id"]].append(certificate)

    finding_counts = Counter()
    findings_by_bin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    finding_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    cohort_bins = set(permits_by_bin)
    cohort_bin_sha256 = hashlib.sha256(("\n".join(sorted(cohort_bins)) + "\n").encode("utf-8")).hexdigest()
    cohort_scope_matches = manifest.get("cohort_bin_sha256") in {None, cohort_bin_sha256}
    cohort_scope_matches = cohort_scope_matches and all(
        source_manifest[key].get("bin_scope_sha256") in {None, cohort_bin_sha256}
        for key in EXPECTED_SOURCES if key in source_manifest
    )
    for source_key in ("dob_violations", "dob_complaints"):
        if source_key not in source_manifest:
            finding_counts[f"{source_key}_source_missing"] += 1
            continue
        for raw_row in _iter_jsonl(raw_dir / source_manifest[source_key]["raw_file"]):
            finding_counts[f"{source_key}_rows"] += 1
            finding = _normalize_finding(source_key, raw_row)
            if finding is None:
                finding_counts[f"{source_key}_excluded_invalid_bin_or_date"] += 1
                continue
            if finding["asset_id"] not in cohort_bins:
                finding_counts[f"{source_key}_excluded_outside_new_building_cohort"] += 1
                continue
            if finding["disposition_date"] is None:
                finding_counts[f"{source_key}_missing_disposition_date"] += 1
            finding_key = (finding["asset_id"], finding["source"], finding["finding_id"])
            previous = finding_by_key.get(finding_key)
            if previous is None:
                finding_by_key[finding_key] = finding
            else:
                finding_counts["duplicate_finding_rows_coalesced"] += 1
                previous_date = previous["disposition_date"]
                current_date = finding["disposition_date"]
                if current_date and (previous_date is None or current_date > previous_date):
                    finding_by_key[finding_key] = finding
    for finding in finding_by_key.values():
        findings_by_bin[finding["asset_id"]].append(finding)

    hazard_scores_path = output_dir / "hazard_scores.csv"
    hazard_scores: dict[str, dict[str, str]] = {}
    if hazard_scores_path.exists():
        with hazard_scores_path.open("r", encoding="utf-8", newline="") as handle:
            hazard_scores = {row["asset_id"]: row for row in csv.DictReader(handle)}
    hazard_manifest_path = output_dir / "hazard_source_manifest.json"
    hazard_manifest = json.loads(hazard_manifest_path.read_text(encoding="utf-8")) if hazard_manifest_path.exists() else None

    complete_download = (
        set(source_manifest) >= set(EXPECTED_SOURCES)
        and bool(manifest.get("complete_extraction"))
        and all(bool(source_manifest[key].get("complete_extraction")) for key in EXPECTED_SOURCES)
        and cohort_scope_matches
    )
    labels: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    candidates_by_building: dict[str, list[dict[str, Any]]] = {}
    leakage_fields = Counter()
    label_counts = Counter()
    match_counts = Counter()
    split_groups: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for building in buildings:
        building_id = building["building_id"]
        anchor = building["permit_issue_date"]
        outcome_end = _add_years(anchor, horizon_years)
        job_ids = set(building["alternate_job_ids"])
        candidates = []
        for certificate in certificates_by_bin.get(building["asset_id"], []):
            match = bool(job_ids.intersection(certificate["alternate_job_ids"]))
            bbl_conflict = bool(certificate["bbl"] and building["bbl"] and certificate["bbl"] != building["bbl"])
            candidates.append({**certificate,
                               "match_level": "bin_and_job" if match else "bin_only",
                               "bbl_conflict": bbl_conflict})
        candidates_by_building[building_id] = candidates
        issued_in_horizon = [row for row in candidates if row["issued_record"] and row["issue_date"]
                             and row["issue_date"] <= as_of and anchor <= row["issue_date"] <= outcome_end]
        plausible = [row for row in issued_in_horizon if not row["bbl_conflict"]]
        qualifying_cos_in_horizon = [row for row in plausible if row["is_qualifying_co"]]
        temporary_in_horizon = [row for row in plausible if row["is_temporary_co"]]
        untyped_in_horizon = [row for row in plausible if not row["is_qualifying_co"] and not row["is_temporary_co"]]
        conflicting_in_horizon = [row for row in issued_in_horizon if row["bbl_conflict"]]
        mature = as_of >= outcome_end
        exact_qualifying = [row for row in qualifying_cos_in_horizon if row["match_level"] == "bin_and_job"]
        earliest_co_issued = min(qualifying_cos_in_horizon, key=lambda row: row["issue_date"]) if qualifying_cos_in_horizon else None
        if qualifying_cos_in_horizon:
            label, status = 1, "positive_co_issued_within_horizon"
            match_level = "bin_and_job" if exact_qualifying else "bin_only"
        elif not mature:
            label, status = None, "right_censored_followup"
            match_level = "bin_only" if issued_in_horizon else "no_certificate_candidate"
        elif untyped_in_horizon:
            label, status, match_level = None, "certificate_type_unresolved", "bin_only"
        elif conflicting_in_horizon:
            label, status, match_level = None, "ambiguous_certificate_candidate", "bbl_conflict"
        elif not complete_download:
            label, status, match_level = None, "source_snapshot_incomplete", "no_certificate_candidate"
        else:
            label, status, match_level = 0, "no_co_issued_recorded_by_horizon", "no_co_record_in_either_feed"
        label_counts[status] += 1
        match_counts[f"match_level_{match_level}"] += 1
        if qualifying_cos_in_horizon:
            match_counts["buildings_with_co_issued_in_horizon"] += 1
            match_counts["bin_only_co_issued_links"] += int(not exact_qualifying)
        match_counts["buildings_with_any_issued_co_in_horizon"] += int(bool(issued_in_horizon))
        match_counts["buildings_with_temporary_co_in_horizon"] += int(bool(temporary_in_horizon))
        label_row = {
            "building_id": building_id,
            "asset_id": building["asset_id"],
            "permit_issue_date": anchor.isoformat(),
            "outcome_horizon_years": horizon_years,
            "outcome_horizon_end": outcome_end.isoformat(),
            "as_of_date": as_of.isoformat(),
            "followup_mature": mature,
            "co_issued_within_horizon": label,
            "label_status": status,
            "match_level": match_level,
            "issued_co_count_in_horizon": len(issued_in_horizon),
            "temporary_co_within_horizon": bool(temporary_in_horizon),
            "earliest_co_issued_date": earliest_co_issued["issue_date"].isoformat() if earliest_co_issued else "",
            "earliest_co_issued_source": earliest_co_issued["certificate_source"] if earliest_co_issued else "",
            "earliest_co_issued_identifier_match": earliest_co_issued["match_level"] if earliest_co_issued else "",
            "matched_certificate_ids": json.dumps([row["certificate_id"] for row in qualifying_cos_in_horizon]),
            "negative_label_interpretation": "No qualifying non-temporary CO issued by the horizon in either complete DOB feed; administrative outcome, not a safety verdict" if label == 0 else "",
        }
        labels.append(label_row)

        group_hash = hashlib.sha256(building["asset_id"].encode("utf-8")).digest()[0] % 100
        split = "train" if group_hash < 70 else "validation" if group_hash < 85 else "test"
        split_groups[split].append(building_id)
        status_score, status_count, unknown_status_count = _filing_status_score(building, filings_by_bin)
        anchor_findings = [row for row in findings_by_bin.get(building["asset_id"], []) if row["issue_date"] <= anchor]
        active_findings = [row for row in anchor_findings if row["disposition_date"] is None or row["disposition_date"] > anchor]
        active_violations = sum(row["source"] == "dob_violations" for row in active_findings)
        active_complaints = sum(row["source"] == "dob_complaints" for row in active_findings)
        active_findings_missing_disposition = sum(row["disposition_date"] is None for row in active_findings)
        total_violations = sum(row["source"] == "dob_violations" for row in anchor_findings)
        total_complaints = sum(row["source"] == "dob_complaints" for row in anchor_findings)
        finding_sources_complete = all(
            key in source_manifest and bool(source_manifest[key].get("complete_extraction"))
            and source_manifest[key].get("bin_scope_sha256") in {None, cohort_bin_sha256}
            for key in ("dob_violations", "dob_complaints")
        ) and cohort_scope_matches
        findings_score = min(1.0, (active_violations + active_complaints) / 5.0) if finding_sources_complete else None
        lat = _parse_float(building["latitude"])
        lon = _parse_float(building["longitude"])
        hazard = hazard_scores.get(building["asset_id"], {})
        late_dates = building["feature_time_leakage_fields"]
        leakage_fields.update(item["field"] for item in late_dates)
        feature_fields = dict(building["feature_source_fields"])
        features.append({
            "building_id": building_id,
            "asset_id": building["asset_id"],
            "borough": building["borough"],
            "bbl": building["bbl"],
            "permit_source": building["permit_source"],
            "permit_issue_date": anchor.isoformat(),
            "latitude": lat,
            "longitude": lon,
            "permit_status_score": status_score,
            "permit_status_records_as_of_anchor": status_count,
            "permit_status_unclassified_records": unknown_status_count,
            "inspection_finding_score": findings_score,
            "violations_before_anchor": total_violations,
            "active_violations_at_anchor": active_violations,
            "complaints_before_anchor": total_complaints,
            "active_complaints_at_anchor": active_complaints,
            "active_findings_missing_disposition_date": active_findings_missing_disposition,
            "natural_hazard_score": _parse_float(hazard.get("natural_hazard_score", "")),
            "pga_2pct_50yr_g": _parse_float(hazard.get("pga_2pct_50yr_g", "")),
            "seismic_component": _parse_float(hazard.get("seismic_component", "")),
            "fema_sfha": hazard.get("fema_sfha", ""),
            "natural_hazard_source_vintage": (hazard_manifest or {}).get("fema", {}).get("nfhl_source_vintage", ""),
            "feature_temporal_status": "dated_events_filtered_to_anchor_static_source_fields_unversioned",
            "split": split,
            "feature_source_fields_json": json.dumps(feature_fields, ensure_ascii=False, sort_keys=True),
            "feature_time_leakage_fields_json": json.dumps(late_dates),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_permits = [{
        "project_id": permit["project_id"], "asset_id": permit["asset_id"], "job_id": permit["job_id"],
        "alternate_job_ids_json": json.dumps(permit["alternate_job_ids"]), "bbl": permit["bbl"],
        "borough": permit["borough"], "permit_source": permit["permit_source"],
        "permit_issue_date": permit["permit_issue_date"].isoformat(), "latitude": permit["latitude"],
        "longitude": permit["longitude"], "raw_permit_status": permit["raw_permit_status"],
        "feature_source_fields_json": json.dumps(permit["feature_source_fields"], ensure_ascii=False, sort_keys=True),
    } for permit in permits]
    normalized_certificates = [{key: (value.isoformat() if isinstance(value, date) else value)
                                for key, value in certificate.items()} for certificate in certificates]
    _write_csv(output_dir / "permits.csv", normalized_permits, list(normalized_permits[0]) if normalized_permits else ["project_id"])
    _write_csv(output_dir / "certificates.csv", normalized_certificates, list(normalized_certificates[0]) if normalized_certificates else ["certificate_id"])
    _write_csv(output_dir / "features.csv", features, list(features[0]) if features else ["building_id"])
    _write_csv(output_dir / "label_ledger.csv", labels, list(labels[0]) if labels else ["building_id"])
    (output_dir / "cohort_bins.txt").write_text("\n".join(sorted(cohort_bins)) + "\n", encoding="utf-8")
    labels_by_building = {row["building_id"]: row for row in labels}
    review_rows = _review_sample(buildings, candidates_by_building, labels_by_building, review_sample_size)
    _write_csv(output_dir / "linkage_review_queue.csv", review_rows, [
        "building_id", "asset_id", "job_ids_json", "borough", "match_group", "permit_issue_date",
        "label_status", "co_issued_within_horizon", "candidate_certificates_json", "human_match_decision", "reviewer_notes",
    ])
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, building_ids in split_groups.items():
        (split_dir / f"{split}_building_ids.txt").write_text("\n".join(sorted(building_ids)) + ("\n" if building_ids else ""), encoding="utf-8")

    horizon_sensitivity = {}
    for years in range(2, 8):
        counts = Counter()
        for building in buildings:
            anchor = building["permit_issue_date"]
            end = _add_years(anchor, years)
            candidates = candidates_by_building[building["building_id"]]
            issued = [row for row in candidates if row["issued_record"] and row["issue_date"]
                      and row["issue_date"] <= as_of and anchor <= row["issue_date"] <= end]
            plausible = [row for row in issued if not row["bbl_conflict"]]
            qualifying = any(row["is_qualifying_co"] for row in plausible)
            untyped = any(not row["is_qualifying_co"] and not row["is_temporary_co"] for row in plausible)
            conflicting = any(row["bbl_conflict"] for row in issued)
            if qualifying:
                counts["positive"] += 1
            elif as_of < end:
                counts["right_censored"] += 1
            elif untyped:
                counts["certificate_type_unresolved"] += 1
            elif conflicting:
                counts["ambiguous_certificate_candidate"] += 1
            elif not complete_download:
                counts["source_snapshot_incomplete"] += 1
            else:
                counts["negative"] += 1
        pos = counts["positive"]
        neg = counts["negative"]
        binary_count = pos + neg
        horizon_sensitivity[str(years)] = {
            **dict(counts),
            "binary_labeled": binary_count, "positive_rate_among_binary_labeled": pos / binary_count if binary_count else None,
        }

    eligible = [row for row in labels if row["co_issued_within_horizon"] in {0, 1}]
    positive_rate = sum(int(row["co_issued_within_horizon"]) for row in eligible) / len(eligible) if eligible else None
    coordinate_count = sum(feature["latitude"] is not None and feature["longitude"] is not None for feature in features)
    summary = {
        "dataset_id": "cauren_civil_nyc_new_building_bin_co_v2",
        "entity": "one building record per normalized NYC BIN",
        "target": "qualifying non-temporary CO issued by either DOB system within the configured horizon after the earliest new-building permit issuance for the BIN",
        "target_is_engineering_safety_truth": False,
        "outcome_horizon_years": horizon_years,
        "outcome_horizon_sensitivity": horizon_sensitivity,
        "as_of_date": as_of.isoformat(),
        "cohort": "NYC DOB new-building BINs",
        "source_dataset_ids": EXPECTED_SOURCES,
        "source_manifest_path": str(manifest_path),
        "all_required_feeds_downloaded": set(source_manifest) >= set(EXPECTED_SOURCES),
        "complete_extraction": complete_download,
        "cohort_bin_count": len(cohort_bins),
        "cohort_bin_sha256": cohort_bin_sha256,
        "cohort_scope_matches_source_manifest": cohort_scope_matches,
        "permit_counts": dict(permit_counts),
        "certificate_counts": dict(certificate_counts),
        "finding_counts": dict(finding_counts),
        "permit_episode_count": len(permits),
        "building_count": len(buildings),
        "unique_bin_count": len({row["asset_id"] for row in buildings}),
        "label_status_counts": dict(label_counts),
        "match_level_counts": dict(match_counts),
        "labeled_building_count": len(eligible),
        "positive_rate_among_binary_labeled_buildings": positive_rate,
        "leakage_date_fields_removed": dict(leakage_fields),
        "coordinate_coverage": {"with_coordinates": coordinate_count, "total": len(features), "fraction": coordinate_count / len(features) if features else None},
        "linkage_review_sample_size": len(review_rows),
        "linkage_review_required_before_modeling": True,
        "feature_contract": {
            "prediction_anchor": "earliest permit issuance date among DOB new-building permit rows for the BIN",
            "outcome_link": "BIN-level earliest qualifying non-temporary CO across BIS and DOB NOW; job ID match is recorded as confidence evidence",
            "permit_status_score": "mean 0/1 status friction across matched DOB NOW filings whose filing and current-status dates are no later than anchor; 1=objection/hold/incomplete/withdrawal, 0=approved/issued",
            "inspection_finding_score": "min(1, active dated DOB violations plus complaints at anchor divided by 5); missing disposition dates are conservatively counted as active proxies and audited",
            "natural_hazard_score": "0.6 * clamp(USGS 2%-in-50-year PGA / 0.6, 0, 1) + 0.4 * FEMA Special Flood Hazard Area flag; source vintages and missing coverage remain explicit",
            "source_record_versioning": "current Socrata snapshots do not contain full historical field versions; dated-event features are anchor-filtered, other permit attributes need temporal validation",
            "raw_fields_preserved_in": "features.csv:feature_source_fields_json and source JSONL",
            "cauren_civil_eight_score_schema_generated": False,
        },
        "split_counts": {split: len(ids) for split, ids in split_groups.items()},
        "split_grouping": "one BIN is one entity and is assigned wholly to one deterministic hash split",
        "manual_review_file": "linkage_review_queue.csv",
        "natural_hazard_source_manifest": hazard_manifest,
        "natural_hazard_scored_building_count": sum(row.get("natural_hazard_score") is not None for row in features),
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/cauren_civil_nyc/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cauren_civil_nyc/normalized"))
    parser.add_argument("--horizon-years", type=int, default=5)
    parser.add_argument("--as-of-date", type=date.fromisoformat, help="Override the conservative date of the source snapshot")
    parser.add_argument("--review-sample-size", type=int, default=57)
    args = parser.parse_args()
    if args.horizon_years <= 0:
        parser.error("--horizon-years must be positive")
    if args.review_sample_size < 0:
        parser.error("--review-sample-size cannot be negative")
    result = build_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        horizon_years=args.horizon_years,
        as_of=args.as_of_date,
        review_sample_size=args.review_sample_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
