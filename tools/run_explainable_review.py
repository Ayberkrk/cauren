"""General-purpose explainable review pipeline for real Cauren civil data.

Runs whatever civil-asset data a caller provides through the full stack --
normalization, the anomaly core, CivilPhysics, quality control, an optional
OMA/FE modal-frequency consistency check, and an uncertainty band -- and
renders the result as a human-reviewable explanation
(cauren_core.explanations.render_diagnosis_explanation).

This is not a fixed demo scenario: it takes real input files and reacts to
whatever anomaly pattern is actually in that data (elevated risk, drift,
flatline, etc. -- whatever cauren_core.runtime detects), which is the whole
point of routing everything through CaurenPipeline.diagnose() rather than
a canned payload.

Two input shapes are supported:

  --wide-csv   One row per asset, one column per canonical civil feature
               (the shape data/public_sources/normalized/civil_public_core_assets.csv
               and tools/build_cauren_civil_dataset.py already use). Good for
               "I have a spreadsheet/export of asset scores".

  --sensors-json  A JSON file holding a list of raw sensor readings
               (sensor_id/name/unit/value/timestamp), i.e. exactly the shape
               CaurenPipeline.diagnose()'s "sensors" payload key expects. Good
               for "I have a live feed or a custom sensor layout" -- lets the
               normalizer's alias matching and the quality-control layer do
               real work on names it has to resolve itself.

Vibration data for modal identification is optional and separate, since most
asset exports won't include raw accelerometer channels. Pass --vibration-csv
and --sampling-hz; use --vibration-column for single-channel peak picking or
--vibration-columns for Timoshenko FDD. A measured baseline is needed for drift
comparison. FE model consistency is reported separately by the OMA tool.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cauren_agents.civil.agent import build_civil_agent
from cauren_core import CaurenPipeline
from cauren_core.contracts import AgentSchema
from cauren_core.explanations import render_diagnosis_explanation
from cauren_physics.oma import compare_to_baseline, identify_modal_parameters
from cauren_physics.timoshenko_adapter import identify_multichannel


def sensors_from_wide_row(row: dict[str, str], schema: AgentSchema, *, timestamp: float | None = None) -> list[dict[str, Any]]:
    """Turns one wide-format row (one column per canonical feature) into a
    sensor-reading list the pipeline can ingest.

    Reads the feature list straight from the agent schema
    (cauren_agents.civil.agent.build_civil_agent) rather than hardcoding
    the civil feature names here a second time, so this stays in sync with
    the schema by construction instead of by the "update all four places"
    discipline the rest of the civil dataset tooling relies on.
    """
    ts = time.time() if timestamp is None else timestamp
    asset_id = str(row.get("asset_id") or "asset")
    sensors: list[dict[str, Any]] = []
    for feature in schema.feature_order:
        raw = row.get(feature)
        if raw in (None, ""):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        sensors.append(
            {
                "sensor_id": f"{asset_id}-{feature}",
                "name": feature,
                "unit": schema.units.get(feature, ""),
                "value": value,
                "timestamp": ts,
            }
        )
    return sensors


def load_wide_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_sensors_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("sensors"), list):
        return payload["sensors"]
    if isinstance(payload, list):
        return payload
    raise SystemExit(
        f"{path}: expected a JSON list of sensor readings, or an object with a 'sensors' list."
    )


def load_vibration_columns(path: Path, columns: Sequence[str]) -> list[list[float]]:
    """Load selected vibration columns in the requested channel-major order."""
    requested = list(columns)
    if not requested:
        raise SystemExit(f"{path}: at least one vibration column is required.")
    if len(set(requested)) != len(requested):
        raise SystemExit(f"{path}: vibration columns must be unique.")
    values = [[] for _ in requested]
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            header = []
        missing = [column for column in requested if column not in header]
        if missing:
            raise SystemExit(f"{path}: column(s) {missing!r} not found. Available columns: {header}")
        indices = [header.index(column) for column in requested]
        for row in reader:
            if not row:
                raise SystemExit(f"{path}:{reader.line_num}: empty vibration sample row.")
            for output_index, (column, source_index) in enumerate(zip(requested, indices)):
                raw = row[source_index].strip() if source_index < len(row) else ""
                if not raw:
                    raise SystemExit(f"{path}:{reader.line_num}: column '{column}' has an empty sample.")
                try:
                    value = float(raw)
                except ValueError:
                    raise SystemExit(
                        f"{path}:{reader.line_num}: column '{column}' has non-numeric value {raw!r}. "
                        "A vibration series must be numeric throughout."
                    ) from None
                if not math.isfinite(value):
                    raise SystemExit(
                        f"{path}:{reader.line_num}: column '{column}' has non-finite value {raw!r}."
                    )
                values[output_index].append(value)
    if not values[0]:
        raise SystemExit(f"{path}: column(s) {requested!r} contained no numeric samples.")
    return values


def load_vibration_series(path: Path, column: str) -> list[float]:
    return load_vibration_columns(path, [column])[0]


def build_oma_frequency_drift(
    vibration_csv: Path,
    *,
    column: str | None = None,
    columns: Sequence[str] | None = None,
    sampling_hz: float,
    baseline_frequencies_hz: list[float] | None,
) -> dict[str, Any]:
    if columns:
        channel_columns = list(columns)
        if len(channel_columns) < 2:
            raise SystemExit("multi-channel FDD requires at least two vibration columns.")
        channel_data = load_vibration_columns(vibration_csv, channel_columns)
        identified = identify_multichannel(
            channel_data,
            sampling_hz,
            channel_ids=channel_columns,
            max_modes=8,
            min_prominence_ratio=0.05,
        )
        if identified is None:
            raise SystemExit("multi-channel FDD requires timoshenko-engine 2.0 or newer")
    else:
        series = load_vibration_series(vibration_csv, column or "value")
        identified = identify_modal_parameters(series, sampling_hz)
    if not baseline_frequencies_hz:
        # No baseline to diff against: report the raw identification only,
        # CivilPhysics' modal_frequency_drift relation stays inactive
        # (it needs max_drop_pct, which only exists once there's a baseline).
        return {"oma_result": identified.to_dict()}
    drift = compare_to_baseline(identified, baseline_frequencies_hz)
    return {"oma_result": identified.to_dict(), **drift.to_dict()}


def run_review(
    pipeline: CaurenPipeline,
    sensors: list[dict[str, Any]],
    *,
    agent_id: str = "cauren-civil",
    asset_id: str | None = None,
    site_id: str | None = None,
    oma_frequency_drift: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"agent_id": agent_id, "sensors": sensors}
    if asset_id:
        payload["asset_id"] = asset_id
    if site_id:
        payload["site_id"] = site_id
    if oma_frequency_drift:
        payload["oma_frequency_drift"] = oma_frequency_drift

    diagnosis = pipeline.diagnose(payload)
    explanation = render_diagnosis_explanation(diagnosis)
    return {
        "asset_id": asset_id,
        "sensor_reading_count": len(sensors),
        "diagnosis": diagnosis.to_dict(),
        "explainable_review": explanation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real Cauren civil diagnosis and render its explainable review.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wide-csv", help="CSV with one row per asset, one column per canonical civil feature.")
    source.add_argument("--sensors-json", help="JSON file with a list of raw sensor readings (or {'sensors': [...]}).")

    parser.add_argument("--asset-id", default=None, help="With --wide-csv, only process this asset_id. Also used to label a --sensors-json run.")
    parser.add_argument("--agent-id", default="cauren-civil")
    parser.add_argument("--site-id", default=None)

    parser.add_argument("--vibration-csv", default=None, help="Optional single-channel vibration time series for OMA.")
    vibration_columns = parser.add_mutually_exclusive_group()
    vibration_columns.add_argument("--vibration-column", default=None)
    vibration_columns.add_argument("--vibration-columns", nargs="+", default=None)
    parser.add_argument("--sampling-hz", type=float, default=None, help="Required if --vibration-csv is given.")
    parser.add_argument(
        "--baseline-frequencies-hz",
        nargs="*",
        type=float,
        default=None,
        help="Known baseline natural frequencies (Hz) to diff the identified frequency against.",
    )

    parser.add_argument("--output-path", default=None, help="Optional path to write the JSON report(s) to.")
    args = parser.parse_args()

    if args.vibration_csv and args.sampling_hz is None:
        parser.error("--vibration-csv requires --sampling-hz")
    if (args.vibration_column is not None or args.vibration_columns is not None) and not args.vibration_csv:
        parser.error("vibration column options require --vibration-csv")
    if args.vibration_columns is not None and len(args.vibration_columns) < 2:
        parser.error("--vibration-columns requires at least two columns")
    if args.vibration_columns is not None and not args.vibration_csv:
        parser.error("--vibration-columns requires --vibration-csv")

    oma_frequency_drift = None
    if args.vibration_csv:
        oma_frequency_drift = build_oma_frequency_drift(
            Path(args.vibration_csv),
            column=args.vibration_column,
            columns=args.vibration_columns,
            sampling_hz=args.sampling_hz,
            baseline_frequencies_hz=args.baseline_frequencies_hz,
        )

    pipeline = CaurenPipeline.from_default_registry()
    schema = build_civil_agent().schema

    reports: list[dict[str, Any]]
    if args.wide_csv:
        rows = load_wide_csv_rows(Path(args.wide_csv))
        if args.asset_id:
            rows = [row for row in rows if str(row.get("asset_id")) == args.asset_id]
            if not rows:
                raise SystemExit(f"No row with asset_id={args.asset_id!r} found in {args.wide_csv}")
        reports = []
        for row in rows:
            sensors = sensors_from_wide_row(row, schema)
            reports.append(
                run_review(
                    pipeline,
                    sensors,
                    agent_id=args.agent_id,
                    asset_id=str(row.get("asset_id") or args.asset_id or ""),
                    site_id=str(row.get("site_id") or args.site_id or "") or None,
                    oma_frequency_drift=oma_frequency_drift,
                )
            )
    else:
        sensors = load_sensors_json(Path(args.sensors_json))
        reports = [
            run_review(
                pipeline,
                sensors,
                agent_id=args.agent_id,
                asset_id=args.asset_id,
                site_id=args.site_id,
                oma_frequency_drift=oma_frequency_drift,
            )
        ]

    text = json.dumps(reports, indent=2, sort_keys=True, ensure_ascii=False)
    if args.output_path:
        Path(args.output_path).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
