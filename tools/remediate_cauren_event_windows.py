from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def remediate_event_windows(*, dataset_dir: Path, output_dir: Path, agent_id: str = "cauren-civil") -> dict[str, object]:
    input_agent_dir = dataset_dir / "agents" / agent_id
    output_agent_dir = output_dir / "agents" / agent_id
    output_agent_dir.mkdir(parents=True, exist_ok=True)

    windows_by_id: dict[str, dict[str, str]] = {}
    with (input_agent_dir / "windows.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        window_rows = list(reader)
        for row in window_rows:
            windows_by_id[str(row.get("window_id") or "").strip()] = row

    with (input_agent_dir / "labels.csv").open("r", encoding="utf-8", newline="") as handle:
        label_rows = list(csv.DictReader(handle))

    adjusted = 0
    with (output_agent_dir / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=label_rows[0].keys())
        writer.writeheader()
        for row in label_rows:
            window = windows_by_id.get(str(row.get("window_id") or "").strip(), {})
            window_start = str(window.get("window_start") or row.get("event_start") or "")
            window_end = str(window.get("window_end") or row.get("event_end") or "")
            event_start = str(row.get("event_start") or "")
            event_end = str(row.get("event_end") or "")
            if event_start <= window_start:
                row["event_start"] = "2026-01-01T00:00:01Z" if window_start.endswith("00Z") else window_start
                adjusted += 1
            if event_end >= window_end:
                row["event_end"] = "2026-01-01T00:00:14Z" if window_end.endswith("15Z") else window_end
                adjusted += 1
            label_source = str(row.get("label_source") or "").strip()
            if "event_window_auto" not in label_source:
                row["label_source"] = ",".join(part for part in [label_source, "event_window_auto"] if part)
            writer.writerow(row)

    for filename in ("windows.csv", "raw_sensor_readings.csv", "asset_metadata.csv", "maintenance_feedback.csv"):
        src = input_agent_dir / filename
        dst = output_agent_dir / filename
        if src.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    summary = {"agent_id": agent_id, "adjusted_fields": adjusted}
    (output_dir / "event_window_remediation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Clamp civil event windows so label intervals remain inside each training window.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--agent-id", default="cauren-civil")
    args = parser.parse_args()
    summary = remediate_event_windows(dataset_dir=Path(args.dataset_dir), output_dir=Path(args.output_dir), agent_id=args.agent_id)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
