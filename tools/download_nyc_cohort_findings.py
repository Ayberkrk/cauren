"""Download DOB violations and complaints only for the new-building BIN cohort.

The citywide DOB findings feeds are intentionally not copied in full.  Each
request is restricted to a chunk of cohort BINs; the sorted BIN list and its
hash are recorded so the builder can reject a stale or incomplete slice.
"""

from __future__ import annotations

import argparse
import concurrent.futures
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
    "dob_violations": {
        "dataset_id": "3h2n-5cm9",
        "title": "DOB Violations (BIS)",
        "url": "https://data.cityofnewyork.us/resource/3h2n-5cm9.json",
        "where": "issue_date IS NOT NULL",
    },
    "dob_complaints": {
        "dataset_id": "eabe-havv",
        "title": "DOB Complaints Received",
        "url": "https://data.cityofnewyork.us/resource/eabe-havv.json",
        "where": "date_entered IS NOT NULL",
    },
}
DEFAULT_PAGE_SIZE = 10_000
DEFAULT_BIN_CHUNK_SIZE = 250
REQUEST_TIMEOUT_SECONDS = 90


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bins(bins: list[str]) -> str:
    return hashlib.sha256(("\n".join(bins) + "\n").encode("utf-8")).hexdigest()


def _write_checkpoint(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _fetch_json(url: str, *, attempts: int = 6) -> object:
    for attempt in range(attempts):
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "Cauren-NYC-Civil-Research/1.0"})
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt + 1 == attempts:
                raise
        except (URLError, TimeoutError, socket.timeout, ConnectionError, http.client.IncompleteRead):
            if attempt + 1 == attempts:
                raise
        print(f"retrying Socrata request ({attempt + 1}/{attempts - 1}) after connection interruption", flush=True)
        time.sleep(min(20, 2**attempt))
    raise RuntimeError("unreachable retry state")


def _query_chunk(source: dict[str, str], bins: list[str], page_size: int) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    last_id: str | None = None
    bin_predicate = "bin in (" + ",".join("'" + value + "'" for value in bins) + ")"
    while True:
        predicates = [bin_predicate, source["where"]]
        if last_id is not None:
            predicates.append(":id > '" + last_id.replace("'", "''") + "'")
        params = {
            "$limit": page_size,
            "$order": ":id",
            "$select": "*, :id",
            "$where": " AND ".join(f"({value})" for value in predicates),
        }
        rows = _fetch_json(f"{source['url']}?{urlencode(params)}")
        if not isinstance(rows, list):
            raise ValueError(f"Unexpected Socrata response: expected JSON array, got {rows!r}")
        if any(not isinstance(row, dict) or not row.get(":id") for row in rows):
            raise ValueError("Socrata response is missing the requested :id cursor")
        page_ids = [str(row[":id"]) for row in rows]
        if len(page_ids) != len(set(page_ids)) or (last_id is not None and last_id in page_ids):
            raise ValueError("Socrata returned a duplicate or non-advancing :id page")
        result.extend(rows)
        if page_ids:
            last_id = page_ids[-1]
        if len(rows) < page_size:
            return result


def _download_source(
    source_key: str,
    source: dict[str, str],
    output_dir: Path,
    bins: list[str],
    bin_hash: str,
    *,
    page_size: int,
    bin_chunk_size: int,
    workers: int,
) -> dict[str, object]:
    target = output_dir / f"{source_key}_cohort.jsonl"
    partial = target.with_suffix(target.suffix + ".partial")
    checkpoint_path = output_dir / f".{source_key}_cohort.download_checkpoint.json"
    chunks = [bins[start:start + bin_chunk_size] for start in range(0, len(bins), bin_chunk_size)]
    config = {
        "dataset_id": source["dataset_id"],
        "cohort_bin_sha256": bin_hash,
        "cohort_bin_count": len(bins),
        "page_size": page_size,
        "bin_chunk_size": bin_chunk_size,
    }

    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing cohort source: {target}")
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if any(checkpoint.get(key) != value for key, value in config.items()):
            raise ValueError(f"Existing checkpoint uses a different cohort or page configuration: {checkpoint_path}")
        if not partial.exists():
            raise ValueError(f"Cohort checkpoint exists but its partial file is missing: {partial}")
        checkpoint_bytes = int(checkpoint["bytes_written"])
        if partial.stat().st_size < checkpoint_bytes:
            raise ValueError(f"Partial source is shorter than checkpoint: {partial}")
        with partial.open("r+b") as handle:
            handle.truncate(checkpoint_bytes)
        if _sha256_file(partial) != checkpoint["sha256"]:
            raise ValueError(f"Partial source checksum does not match checkpoint: {partial}")
        completed = int(checkpoint["completed_chunks"])
        row_count = int(checkpoint["row_count"])
        started_at = str(checkpoint["snapshot_started_at_utc"])
        print(f"{source_key}: resuming at cohort BIN chunk {completed + 1:,}/{len(chunks):,} after {row_count:,} rows", flush=True)
    else:
        partial.touch(exist_ok=True)
        completed = 0
        row_count = 0
        started_at = _utc_now()
        checkpoint = {
            **config,
            "snapshot_started_at_utc": started_at,
            "completed_chunks": 0,
            "row_count": 0,
            "bytes_written": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        }
        _write_checkpoint(checkpoint_path, checkpoint)

    next_chunk = completed
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        while next_chunk < len(chunks):
            end_chunk = min(next_chunk + workers, len(chunks))
            futures = {
                index: executor.submit(_query_chunk, source, chunks[index], page_size)
                for index in range(next_chunk, end_chunk)
            }
            # Commit in chunk order. A restart repeats at most the unfinished
            # window and never appends an uncheckpointed response twice.
            for index in range(next_chunk, end_chunk):
                rows = futures[index].result()
                with partial.open("a", encoding="utf-8", newline="\n") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                row_count += len(rows)
                completed = index + 1
                checkpoint = {
                    **config,
                    "snapshot_started_at_utc": started_at,
                    "completed_chunks": completed,
                    "row_count": row_count,
                    "bytes_written": partial.stat().st_size,
                    "sha256": _sha256_file(partial),
                }
                _write_checkpoint(checkpoint_path, checkpoint)
                print(f"{source_key}: BIN chunks {completed:,}/{len(chunks):,}; {row_count:,} rows", flush=True)
            next_chunk = end_chunk

    partial.replace(target)
    checkpoint_path.unlink(missing_ok=True)
    return {
        **source,
        "source_key": source_key,
        "snapshot_started_at_utc": started_at,
        "snapshot_finished_at_utc": _utc_now(),
        "row_count": row_count,
        "complete_extraction": True,
        "extraction_scope": "all_records_for_cohort_bins",
        "bin_scope_sha256": bin_hash,
        "bin_scope_count": len(bins),
        "bin_chunk_size": bin_chunk_size,
        "page_size": page_size,
        "pagination": "Socrata :id keyset within complete cohort BIN chunks",
        "raw_file": target.name,
        "sha256": _sha256_file(target),
        "soql_where": f"bin IN cohort BIN list AND ({source['where']})",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-file", type=Path, default=Path("data/cauren_civil_nyc/normalized/cohort_bins.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cauren_civil_nyc/raw"))
    parser.add_argument("--source", choices=sorted(SOURCES), action="append")
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--bin-chunk-size", type=int, default=DEFAULT_BIN_CHUNK_SIZE)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if min(args.page_size, args.bin_chunk_size, args.workers) <= 0:
        parser.error("page size, BIN chunk size, and worker count must be positive")
    bins = sorted({line.strip() for line in args.bin_file.read_text(encoding="utf-8").splitlines() if line.strip()})
    if not bins or any(len(value) != 7 or not value.isdigit() for value in bins):
        parser.error("BIN file must contain at least one seven-digit BIN per line")
    bin_hash = _sha256_bins(bins)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "source_manifest.json"
    if not manifest_path.exists():
        parser.error(f"Expected existing permit/CO source manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("cohort_bin_sha256") not in {None, bin_hash}:
        parser.error("BIN file differs from the cohort hash already recorded in the source manifest")
    sources = manifest.setdefault("sources", {})
    selected = args.source or list(SOURCES)
    for source_key in selected:
        existing = sources.get(source_key)
        if existing:
            existing_file = args.output_dir / existing.get("raw_file", "")
            if (
                existing_file.is_file()
                and existing.get("complete_extraction") is True
                and existing.get("bin_scope_sha256") == bin_hash
                and existing.get("bin_scope_count") == len(bins)
                and _sha256_file(existing_file) == existing.get("sha256")
            ):
                print(f"{source_key}: keeping verified cohort slice ({existing['row_count']:,} rows)", flush=True)
                continue
            parser.error(f"Existing cohort source is incomplete or uses a different BIN set: {source_key}")
        item = _download_source(
            source_key, SOURCES[source_key], args.output_dir, bins, bin_hash,
            page_size=args.page_size, bin_chunk_size=args.bin_chunk_size, workers=args.workers,
        )
        sources[source_key] = item
        manifest["cohort_bin_sha256"] = bin_hash
        manifest["cohort_bin_count"] = len(bins)
        manifest["cohort_scope"] = "NYC new-building BIN cohort; findings feeds queried only for these BINs"
        manifest["finished_at_utc"] = _utc_now()
        manifest["all_required_feeds_downloaded"] = set(sources) >= set(SOURCES) | {
            "bis_permits", "dob_now_permits", "dob_now_filings", "bis_certificates", "dob_now_certificates"
        }
        manifest["complete_extraction"] = manifest["all_required_feeds_downloaded"] and all(
            bool(value.get("complete_extraction")) for value in sources.values()
        )
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
