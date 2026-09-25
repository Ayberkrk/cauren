from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cauren_physics.fe_reference_model import ShearBuildingModel, model_consistency
from cauren_physics.oma import compare_to_baseline, identify_modal_parameters
from cauren_physics.timoshenko_adapter import identify_multichannel
from tools.run_explainable_review import load_vibration_columns, load_vibration_series


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run single-channel peak-picking OMA or multi-channel FDD. FE frequencies are "
            "reported as model consistency only, not used as a drift baseline or damage verdict."
        )
    )
    parser.add_argument("--input-csv", required=True, help="CSV file with a time-series column.")
    input_columns = parser.add_mutually_exclusive_group()
    input_columns.add_argument("--column", default=None, help="Single signal column, default: value.")
    input_columns.add_argument("--columns", nargs="+", default=None, help="Two or more channel columns for Timoshenko FDD.")
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
        help="Measured baseline frequencies (Hz) for drift comparison. FE model frequencies never replace this baseline.",
    )
    parser.add_argument("--fe-story-masses-kg", nargs="+", type=float, default=None)
    parser.add_argument("--fe-story-stiffness-n-per-m", nargs="+", type=float, default=None)
    parser.add_argument("--warn-pct", type=float, default=5.0)
    parser.add_argument("--alarm-pct", type=float, default=10.0)
    parser.add_argument("--output-path", default=None, help="Optional path to write the JSON report to.")
    args = parser.parse_args()

    if args.columns is not None and len(args.columns) < 2:
        parser.error("--columns requires at least two columns")
    if (args.fe_story_masses_kg is None) != (args.fe_story_stiffness_n_per_m is None):
        parser.error("--fe-story-masses-kg and --fe-story-stiffness-n-per-m must be supplied together")
    if (
        args.fe_story_masses_kg is not None
        and len(args.fe_story_masses_kg) != len(args.fe_story_stiffness_n_per_m)
    ):
        parser.error("FE story masses and stiffnesses must have equal lengths")

    if args.columns is not None:
        channel_columns = list(args.columns)
        channel_data = load_vibration_columns(Path(args.input_csv), channel_columns)
        result = identify_multichannel(
            channel_data,
            args.sampling_hz,
            channel_ids=channel_columns,
            max_modes=args.max_modes,
            min_prominence_ratio=args.min_prominence_ratio,
        )
        if result is None:
            raise SystemExit("multi-channel FDD requires timoshenko-engine 2.0 or newer")
    else:
        series = load_vibration_series(Path(args.input_csv), args.column or "value")
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
    if args.fe_story_masses_kg is not None:
        model = ShearBuildingModel(
            story_masses_kg=tuple(args.fe_story_masses_kg),
            story_stiffness_n_per_m=tuple(args.fe_story_stiffness_n_per_m),
        )
        report["fe_model_consistency"] = model_consistency(model, result)

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output_path:
        Path(args.output_path).write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
