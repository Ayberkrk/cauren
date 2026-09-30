"""Download the NYC DOB feeds used by the narrow Cauren civil cohort.

The source responses are stored as JSON Lines so the original field names and
source-specific columns remain intact. Re-run into a new directory for a new
point-in-time snapshot; the tool never overwrites an existing raw snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SOURCES = {
    "bis_permits": {
        "dataset_id": "ipu4-2q9a",
        "title": "DOB Permit Issuance (BIS)",
        "url": "https://data.cityofnewyork.us/resource/ipu4-2q9a.json",
        "default_where": "job_type = 'NB'",
    },
    "dob_now_permits": {
        "dataset_id": "rbx6-tga4",
        "title": "DOB NOW Build Approved Permits",
        "url": "https://data.cityofnewyork.us/resource/rbx6-tga4.json",
        "default_where": "issued_date IS NOT NULL",
    },
    "dob_now_filings": {
        "dataset_id": "w9ak-ipjd",
        "title": "DOB NOW Build Job Application Filings",
        "url": "https://data.cityofnewyork.us/resource/w9ak-ipjd.json",
        "default_where": "upper(job_type) = 'NEW BUILDING'",
    },
    "bis_certificates": {
        "dataset_id": "bs8b-p36w",
        "title": "DOB Certificate of Occupancy (BIS)",
        "url": "https://data.cityofnewyork.us/resource/bs8b-p36w.json",
        "default_where": "job_type = 'NB'",
    },
    "dob_now_certificates": {
        "dataset_id": "pkdm-hqz6",
        "title": "DOB NOW Certificate of Occupancy",
        "url": "https://data.cityofnewyork.us/resource/pkdm-hqz6.json",
        "default_where": "upper(job_type) = 'NEW BUILDING'",
    },
    "dob_violations": {
        "dataset_id": "3h2n-5cm9",
        "title": "DOB Violations (BIS)",
        "url": "https://data.cityofnewyork.us/resource/3h2n-5cm9.json",
        "default_where": "bin IS NOT NULL AND issue_date IS NOT NULL",
    },
    "dob_complaints": {
        "dataset_id": "eabe-havv",
        "title": "DOB Complaints Received",
        "url": "https://data.cityofnewyork.us/resource/eabe-havv.json",
        "default_where": "bin IS NOT NULL AND date_entered IS NOT NULL",
    },
}
CORE_SOURCES = (
    "bis_permits", "dob_now_permits", "dob_now_filings",
    "bis_certificates", "dob_now_certificates",
)
PAGE_SIZE = 50_000
REQUEST_TIMEOUT_SECONDS = 240


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fetch_json(url: str, *, attempts: int = 10) -> object:
    for attempt in range(attempts):
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "Cauren-NYC-Civil-Research/1.0"})
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt + 1 == attempts:
                raise
            time.sleep(min(60, 2**attempt))
        except (URLError, TimeoutError, socket.timeout, ConnectionError, http.client.IncompleteRead):
            if attempt + 1 == attempts:
                raise
            print(f"retrying Socrata request ({attempt + 1}/{attempts - 1}) after connection interruption", flush=True)
            time.sleep(min(60, 2**attempt))
    raise RuntimeError("unreachable retry state")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_checkpoint(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _soql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _download_source(
    source_key: str,
    source: dict[str, str],
    output_dir: Path,
    *,
    max_rows: int | None,
    where: str | None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{source_key}.jsonl"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing raw snapshot: {target}")
    temporary = target.with_suffix(".jsonl.partial")
    checkpoint_path = output_dir / f".{source_key}.download_checkpoint.json"
    query_config = {
        "dataset_id": source["dataset_id"],
        "soql_where": where,
        "max_rows_limit": max_rows,
        "page_size": PAGE_SIZE,
    }
    checkpoint: dict[str, object]
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if any(checkpoint.get(key) != value for key, value in query_config.items()):
            raise ValueError(f"Existing download checkpoint uses a different query: {checkpoint_path}")
        if not temporary.exists():
            raise ValueError(f"Download checkpoint exists but its partial source file is missing: {temporary}")
        checkpoint_bytes = int(checkpoint["bytes_written"])
        actual_bytes = temporary.stat().st_size
        if actual_bytes < checkpoint_bytes:
            raise ValueError(f"Partial source is shorter than its checkpoint: {temporary}")
        with temporary.open("r+b") as handle:
            handle.truncate(checkpoint_bytes)
        if _sha256_file(temporary) != checkpoint["sha256"]:
            raise ValueError(f"Partial source checksum does not match its checkpoint: {temporary}")
        row_count = int(checkpoint["row_count"])
        last_id = str(checkpoint["last_id"]) if checkpoint.get("last_id") is not None else None
        started_at = str(checkpoint["snapshot_started_at_utc"])
        print(f"{source_key}: resuming from {row_count:,} verified rows", flush=True)
    else:
        if temporary.exists() and temporary.stat().st_size:
            raise ValueError(f"Partial source has no checkpoint; preserving it for inspection: {temporary}")
        temporary.touch(exist_ok=True)
        row_count = 0
        last_id = None
        started_at = _utc_now()
        checkpoint = {
            **query_config,
            "snapshot_started_at_utc": started_at,
            "row_count": 0,
            "last_id": None,
            "bytes_written": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        }
        _write_checkpoint(checkpoint_path, checkpoint)

    while True:
        page_limit = min(PAGE_SIZE, max_rows - row_count) if max_rows is not None else PAGE_SIZE
        if page_limit <= 0:
            break
        predicates = []
        if where:
            predicates.append(f"({where})")
        if last_id is not None:
            predicates.append(f":id > {_soql_literal(last_id)}")
        params = {
            "$limit": page_limit,
            "$order": ":id",
            # Socrata's system id gives us a stable keyset cursor. Offset
            # paging degrades sharply on large resources and may time out.
            "$select": "*, :id",
        }
        if predicates:
            params["$where"] = " AND ".join(predicates)
        query = urlencode(params)
        rows = _fetch_json(f"{source['url']}?{query}")
        if not isinstance(rows, list):
            raise ValueError(f"Unexpected response for {source_key}: expected a JSON array")
        if any(not isinstance(row, dict) or not row.get(":id") for row in rows):
            raise ValueError(f"Socrata response for {source_key} is missing the requested :id cursor")
        page_ids = [str(row[":id"]) for row in rows]
        if len(page_ids) != len(set(page_ids)) or (last_id is not None and last_id in page_ids):
            raise ValueError(f"Socrata returned a duplicate or non-advancing :id page for {source_key}")

        with temporary.open("a", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        row_count += len(rows)
        if page_ids:
            last_id = page_ids[-1]
        checkpoint = {
            **query_config,
            "snapshot_started_at_utc": started_at,
            "row_count": row_count,
            "last_id": last_id,
            "bytes_written": temporary.stat().st_size,
            "sha256": _sha256_file(temporary),
        }
        _write_checkpoint(checkpoint_path, checkpoint)
        print(f"{source_key}: {row_count:,} rows", flush=True)
        if len(rows) < page_limit or (max_rows is not None and row_count >= max_rows):
            break

    temporary.replace(target)
    checkpoint_path.unlink(missing_ok=True)

    return {
        **source,
        "source_key": source_key,
        "snapshot_started_at_utc": started_at,
        "snapshot_finished_at_utc": _utc_now(),
        "row_count": row_count,
        "complete_extraction": max_rows is None,
        "max_rows_limit": max_rows,
        "page_size": PAGE_SIZE,
        "pagination": "Socrata :id keyset",
        "raw_file": target.name,
            "sha256": _sha256_file(target),
            "soql_where": where,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/cauren_civil_nyc/raw"))
    parser.add_argument("--max-rows-per-source", type=int, help="Pilot-only row cap; capped snapshots cannot support negative labels")
    parser.add_argument("--include-all", action="store_true", help="Download all rows rather than the new-building cohort filters")
    parser.add_argument(
        "--source", choices=CORE_SOURCES, action="append",
        help="Download selected permit/CO feeds only; findings feeds are fetched for the cohort via download_nyc_cohort_findings.py",
    )
    args = parser.parse_args()
    if args.max_rows_per_source is not None and args.max_rows_per_source <= 0:
        parser.error("--max-rows-per-source must be positive")

    selected = args.source or list(CORE_SOURCES)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "source_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_id") != "cauren_civil_nyc_dob_sources_v1":
            parser.error(f"Unexpected existing manifest: {manifest_path}")
    else:
        manifest = {
            "dataset_id": "cauren_civil_nyc_dob_sources_v1",
            "started_at_utc": _utc_now(),
            "sources": {},
        }
    manifest.setdefault("sources", {})
    for source_key in selected:
        existing = manifest["sources"].get(source_key)  # type: ignore[index]
        if existing:
            existing_file = args.output_dir / existing["raw_file"]
            requested_where = None if args.include_all else SOURCES[source_key].get("default_where")
            if existing_file.exists():
                digest = _sha256_file(existing_file)
                if (
                    digest == existing.get("sha256")
                    and existing.get("complete_extraction") == (args.max_rows_per_source is None)
                    and existing.get("max_rows_limit") == args.max_rows_per_source
                    and existing.get("soql_where") == requested_where
                ):
                    print(f"{source_key}: keeping verified snapshot ({existing['row_count']:,} rows)", flush=True)
                    continue
            parser.error(f"Existing source snapshot is incomplete or changed: {source_key}; use a new output directory")
        source_manifest = _download_source(
            source_key,
            SOURCES[source_key],
            args.output_dir,
            max_rows=args.max_rows_per_source,
            where=None if args.include_all else SOURCES[source_key].get("default_where"),
        )
        manifest["sources"][source_key] = source_manifest  # type: ignore[index]
        manifest["finished_at_utc"] = _utc_now()
        manifest["all_required_feeds_downloaded"] = set(manifest["sources"]) == set(SOURCES)
        manifest["complete_extraction"] = manifest["all_required_feeds_downloaded"] and all(
            bool(item.get("complete_extraction")) for item in manifest["sources"].values()
        )
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary_manifest.replace(manifest_path)

    manifest["finished_at_utc"] = _utc_now()
    manifest["all_required_feeds_downloaded"] = set(manifest["sources"]) == set(SOURCES)
    manifest["complete_extraction"] = manifest["all_required_feeds_downloaded"] and all(
        bool(item.get("complete_extraction")) for item in manifest["sources"].values()
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
