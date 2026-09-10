from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cauren_physics.oma import compare_to_baseline, identify_modal_parameters


def _read_series_column(path: Path, column: str) -> list[float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise SystemExit(
                f"Column '{column}' not found in {path}. Available columns: {reader.fieldnames}"
            )
        values: list[float] = []
        for row in reader:
            raw = row.get(column)
            if raw in (None, ""):
                continue
            values.append(float(raw))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run operational modal analysis (peak-picking) on a single-channel vibration "
            "time series and, optionally, compare the result against baseline frequencies."
        )
    )
    parser.add_argument("--input-csv", required=True, help="CSV file with a time-series column.")
    parser.add_argument("--column", default="value", help="Column name holding the signal values.")
    parser.add_argument("--sampling-hz", type=float, required=True, help="Sample rate the series was recorded at.")
    parser.add_argument("--max-modes", type=int, default=3, help="Maximum number of modes to report.")
    parser.add_argument(
        "--min-prominence-ratio",
        type=float,
        default=0.15,
        help="Minimum peak power relative to the spectrum max to count as a mode.",
    )
    parser.add_argument(
        "--baseline-frequencies-hz",
        nargs="*",
        type=float,
        default=None,
        help="Known baseline natural frequencies (Hz) to compare the current identification against.",
    )
    parser.add_argument("--warn-pct", type=float, default=5.0)
    parser.add_argument("--alarm-pct", type=float, default=10.0)
    parser.add_argument("--output-path", default=None, help="Optional path to write the JSON report to.")
    args = parser.parse_args()

    series = _read_series_column(Path(args.input_csv), args.column)
    result = identify_modal_parameters(
        series,
        args.sampling_hz,
        max_modes=args.max_modes,
        min_prominence_ratio=args.min_prominence_ratio,
    )

    report: dict = {"oma_result": result.to_dict()}
    if args.baseline_frequencies_hz:
        drift = compare_to_baseline(
            result,
            args.baseline_frequencies_hz,
            warn_pct=args.warn_pct,
            alarm_pct=args.alarm_pct,
        )
        report["oma_frequency_drift"] = drift.to_dict()

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output_path:
        Path(args.output_path).write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
