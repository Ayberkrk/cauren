"""Build the v4 bridge-year panel for deterioration hazard modelling.

The v3 builder (tools/build_cauren_bridge_dataset.py) keeps one window of
three derived scores per bridge. v4 keeps every bridge-year row with the
NBI items the deterioration literature relies on (traffic, truck share,
age, reconstruction, structure and deck type, deck protection, geometry,
inspection cadence) plus causal condition-history features, so a
discrete-time hazard model can learn from every observed inspection
interval rather than from a single snapshot.

Source data is the same public bundle as v3 (CC BY 4.0, Hugging Face
sweetapricity/bridgedeck-{nbi,nshm,nfhl}); see that module's docstring.

    python3 tools/build_bridge_years_v4.py \\
        --raw-dir /path/to/nbi_raw --nshm-csv nshm.csv --nfhl-csv nfhl.csv \\
        --states CA IA PA --v3-windows data/cauren_bridge/agents/cauren-bridge/windows.csv \\
        --output-dir data/cauren_bridge_v4

Design notes:

- Every feature on row (bridge, t) is computed from NBI records of year t or
  earlier. Columns that look at later years are prefixed ``label_`` and must
  never be used as model inputs (tests/test_bridge_v4_dataset.py checks this
  by poisoning later years).
- NBI annual files repeat the last inspection's ratings in years without a
  new inspection (most bridges are inspected every 24 months). Item 090
  (date of inspection) is parsed so a model can tell a fresh inspection from
  a carried-forward rating.
- ``label_deck_drop_next`` is the one-year hazard event: 1 if the next
  year's deck rating is lower, 0 if equal, missing if the next year is
  absent or the rating improved (a repair ends the spell; the next spell's
  history features record the improvement).
- ``label_deck_drop_5yr`` reproduces the v3 target exactly (rating five
  years later lower than at t; equal is 0; improvements and censored years
  are missing) so v4 models are scored against the same outcome.
- Splits reuse the v3 assignment by bridge when ``--v3-windows`` is given,
  so v3 and v4 models are compared on the same test bridges. Bridges absent
  from v3 get a deterministic hash split with the same 70/15/15 shares.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tools.build_cauren_bridge_dataset import (
    _dms_to_decimal,
    _year_from_filename,
    join_hazard,
    scour_risk_from_item_113,
)
from tools.provenance import file_record, package_versions

DATASET_ID = "cauren_bridge_fhwa_nbi_v4_bridge_years"

CONDITION_ITEMS = {
    "DECK_COND_058": "deck_cond",
    "SUPERSTRUCTURE_COND_059": "superstructure_cond",
    "SUBSTRUCTURE_COND_060": "substructure_cond",
    "CULVERT_COND_062": "culvert_cond",
    "CHANNEL_COND_061": "channel_cond",
}
NUMERIC_ITEMS = {
    "YEAR_BUILT_027": "year_built",
    "YEAR_RECONSTRUCTED_106": "year_reconstructed",
    "ADT_029": "adt",
    "PERCENT_ADT_TRUCK_109": "truck_pct",
    "TRAFFIC_LANES_ON_028A": "lanes_on",
    "DEGREES_SKEW_034": "skew_deg",
    "MAX_SPAN_LEN_MT_048": "max_span_m",
    "STRUCTURE_LEN_MT_049": "structure_length_m",
    "DECK_WIDTH_MT_052": "deck_width_m",
    "MAIN_UNIT_SPANS_045": "main_spans",
    "INSPECT_FREQ_MONTHS_091": "inspect_freq_months",
}
CATEGORICAL_ITEMS = {
    "OWNER_022": "owner",
    "FUNCTIONAL_CLASS_026": "functional_class",
    "DESIGN_LOAD_031": "design_load",
    "STRUCTURE_KIND_043A": "structure_kind",
    "STRUCTURE_TYPE_043B": "structure_type",
    "WATERWAY_EVAL_071": "waterway_eval",
    "DECK_STRUCTURE_TYPE_107": "deck_structure_type",
    "SURFACE_TYPE_108A": "surface_type",
    "MEMBRANE_TYPE_108B": "membrane_type",
    "DECK_PROTECTION_108C": "deck_protection",
}
ID_ITEMS = [
    "STATE_CODE_001", "STRUCTURE_NUMBER_008", "COUNTY_CODE_003",
    "LAT_016", "LONG_017", "DATE_OF_INSPECT_090", "SCOUR_CRITICAL_113",
]
RAW_COLS = [*ID_ITEMS, *CONDITION_ITEMS, *NUMERIC_ITEMS, *CATEGORICAL_ITEMS]

HISTORY_FEATURES = [
    "deck_prev", "deck_delta", "years_at_deck_rating", "deck_drops_past",
    "deck_improvements_past", "years_since_inspection", "inspected_this_year",
    "age_years", "years_since_reconstruction", "log_adt", "truck_adt",
]
MODEL_FEATURES = [
    *CONDITION_ITEMS.values(),
    *[name for name in NUMERIC_ITEMS.values() if name not in {"year_built", "year_reconstructed"}],
    *HISTORY_FEATURES,
    "structural_risk_score", "ground_stability_score", "natural_hazard_score",
]
CATEGORICAL_FEATURES = list(CATEGORICAL_ITEMS.values())
LABEL_COLUMNS = ["label_deck_drop_next", "label_deck_drop_5yr"]
SPLIT_SHARES = (0.70, 0.15)


def parse_inspection_date(raw: pd.Series) -> pd.Series:
    """Item 090 is MMYY with the leading zero often stripped (e.g. ``399``)."""
    digits = raw.astype(str).str.strip().str.zfill(4)
    month = pd.to_numeric(digits.str[:2], errors="coerce")
    year2 = pd.to_numeric(digits.str[2:], errors="coerce")
    year = np.where(year2 > 50, 1900 + year2, 2000 + year2)
    valid = month.between(1, 12) & year2.notna()
    return pd.Series(np.where(valid, year + (month - 0.5) / 12.0, np.nan), index=raw.index)


def _read_year_file(path: str) -> pd.DataFrame:
    kwargs = dict(usecols=lambda c: c in RAW_COLS, dtype=str, low_memory=False)
    try:
        frame = pd.read_csv(path, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        frame = pd.read_csv(path, encoding="latin-1", **kwargs)
    for column in RAW_COLS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame


def load_nbi_years(raw_dir: Path, states: set[str]) -> pd.DataFrame:
    """Read NBI year files for the given FIPS state codes into typed columns."""
    files = sorted(glob.glob(os.path.join(raw_dir, "??[0-9][0-9].txt")))
    if not files:
        raise SystemExit(f"No NBI *.txt files found under {raw_dir}")
    frames = []
    for path in files:
        year = _year_from_filename(path)
        if year is None:
            continue
        frame = _read_year_file(path)
        state = frame["STATE_CODE_001"].astype(str).str.strip().str.zfill(2)
        if states:
            frame = frame[state.isin(states)]
        if frame.empty:
            continue
        frame = frame.assign(year=year)
        frames.append(frame)
    if not frames:
        raise SystemExit("No rows matched the requested states.")
    raw = pd.concat(frames, ignore_index=True)

    nbi = pd.DataFrame(index=raw.index)
    nbi["year"] = raw["year"].astype(np.int16)
    nbi["state_code"] = raw["STATE_CODE_001"].astype(str).str.strip().str.zfill(2)
    county = raw["COUNTY_CODE_003"].astype(str).str.strip().str.zfill(3)
    structure = raw["STRUCTURE_NUMBER_008"].astype(str).str.strip()
    nbi["structure_number"] = structure
    nbi["bridge_id"] = nbi["state_code"] + "_" + county + "_" + structure
    for source, name in CONDITION_ITEMS.items():
        values = pd.to_numeric(raw[source], errors="coerce")
        nbi[name] = values.where(values.between(0, 9)).astype(np.float32)
    for source, name in NUMERIC_ITEMS.items():
        nbi[name] = pd.to_numeric(raw[source], errors="coerce").astype(np.float32)
    for source, name in CATEGORICAL_ITEMS.items():
        codes = raw[source].astype(str).str.strip().str.upper()
        nbi[name] = codes.where(~codes.isin({"", "NAN"}), "missing")
    nbi["inspection_date"] = parse_inspection_date(raw["DATE_OF_INSPECT_090"]).astype(np.float32)
    nbi["ground_stability_score"] = scour_risk_from_item_113(raw["SCOUR_CRITICAL_113"]).astype(np.float32)
    nbi["lat_dd"] = _dms_to_decimal(pd.to_numeric(raw["LAT_016"], errors="coerce"), is_lon=False)
    nbi["lon_dd"] = _dms_to_decimal(pd.to_numeric(raw["LONG_017"], errors="coerce"), is_lon=True)
    valid_coord = nbi["lat_dd"].between(15, 72) & nbi["lon_dd"].between(-180, -60)
    nbi.loc[~valid_coord, ["lat_dd", "lon_dd"]] = np.nan
    return nbi


def add_history_features(nbi: pd.DataFrame) -> pd.DataFrame:
    """Add causal per-bridge history features and the forward-looking labels."""
    nbi = nbi.drop_duplicates(subset=["bridge_id", "year"], keep="first")
    nbi = nbi.sort_values(["bridge_id", "year"]).reset_index(drop=True)
    by_bridge = nbi.groupby("bridge_id", sort=False)

    deck = nbi["deck_cond"]
    prev_year = by_bridge["year"].shift(1)
    contiguous = (nbi["year"] - prev_year) == 1
    deck_prev = by_bridge["deck_cond"].shift(1).where(contiguous)
    nbi["deck_prev"] = deck_prev.astype(np.float32)
    nbi["deck_delta"] = (deck - deck_prev).astype(np.float32)

    dropped = (deck < deck_prev).astype(np.int16)
    improved = (deck > deck_prev).astype(np.int16)
    nbi["deck_drops_past"] = dropped.groupby(nbi["bridge_id"], sort=False).cumsum().astype(np.float32)
    nbi["deck_improvements_past"] = improved.groupby(nbi["bridge_id"], sort=False).cumsum().astype(np.float32)

    # Run length of the current deck rating, counted in years. A missing
    # rating, a gap in the yearly records, or a changed rating starts a new run.
    new_run = ~contiguous | deck.isna() | deck_prev.isna() | (deck != deck_prev)
    run_id = new_run.astype(np.int64).groupby(nbi["bridge_id"], sort=False).cumsum()
    nbi["years_at_deck_rating"] = (
        nbi.groupby([nbi["bridge_id"], run_id], sort=False).cumcount() + 1
    ).where(deck.notna()).astype(np.float32)

    inspection = nbi["inspection_date"]
    previous_inspection = by_bridge["inspection_date"].shift(1)
    nbi["inspected_this_year"] = (
        (inspection != previous_inspection) & inspection.notna()
    ).astype(np.float32)
    nbi["years_since_inspection"] = (nbi["year"] + 0.5 - inspection).clip(lower=0).astype(np.float32)

    year = nbi["year"].astype(np.float32)
    built = nbi["year_built"].where(nbi["year_built"].between(1800, 2030))
    nbi["age_years"] = (year - built).where(lambda age: age >= 0).astype(np.float32)
    reconstructed = nbi["year_reconstructed"].where(nbi["year_reconstructed"].between(1800, 2030))
    # A reconstruction year later than t is future information and is dropped.
    reconstructed = reconstructed.where(reconstructed <= year)
    nbi["years_since_reconstruction"] = (year - reconstructed.fillna(built)).astype(np.float32)
    nbi["log_adt"] = np.log1p(nbi["adt"].clip(lower=0)).astype(np.float32)
    nbi["truck_adt"] = (nbi["adt"].clip(lower=0) * nbi["truck_pct"].clip(0, 100) / 100.0).astype(np.float32)

    ratings = nbi[["deck_cond", "superstructure_cond", "substructure_cond"]]
    worst = ratings.min(axis=1, skipna=True)
    nbi["structural_risk_score"] = np.where(ratings.notna().any(axis=1), 1.0 - worst / 9.0, np.nan).astype(np.float32)

    next_year = by_bridge["year"].shift(-1)
    next_deck = by_bridge["deck_cond"].shift(-1).where((next_year - nbi["year"]) == 1)
    nbi["label_deck_drop_next"] = np.select(
        [next_deck < deck, next_deck == deck], [1.0, 0.0], default=np.nan
    ).astype(np.float32)

    future = nbi[["bridge_id", "year", "deck_cond"]].rename(columns={"deck_cond": "deck_cond_5yr"})
    future = future.assign(year=future["year"] - 5)
    nbi = nbi.merge(future, on=["bridge_id", "year"], how="left")
    censor_cutoff = int(nbi["year"].max()) - 5
    nbi.loc[nbi["year"] > censor_cutoff, "deck_cond_5yr"] = np.nan
    nbi["label_deck_drop_5yr"] = np.select(
        [nbi["deck_cond_5yr"] < nbi["deck_cond"], nbi["deck_cond_5yr"] == nbi["deck_cond"]],
        [1.0, 0.0],
        default=np.nan,
    ).astype(np.float32)
    return nbi.drop(columns=["deck_cond_5yr"])


def mark_v3_anchors(nbi: pd.DataFrame, min_real_years: int = 8, max_seq_len: int = 16) -> pd.Series:
    """Flag the row v3 would use as each bridge's prediction anchor.

    v3 takes the last year with a 5-year label and keeps the bridge only if at
    least ``min_real_years`` records exist at or before it (within the last
    ``max_seq_len`` years).
    """
    labeled = nbi["label_deck_drop_5yr"].notna()
    last_labeled_year = nbi["year"].where(labeled).groupby(nbi["bridge_id"]).transform("max")
    is_anchor = labeled & (nbi["year"] == last_labeled_year)
    in_window = (nbi["year"] <= last_labeled_year) & (nbi["year"] > last_labeled_year - max_seq_len)
    history_count = in_window.groupby(nbi["bridge_id"]).transform("sum")
    return is_anchor & (history_count >= min_real_years)


def assign_splits(bridge_ids: pd.Series, v3_windows: Path | None) -> pd.Series:
    """Reuse v3 bridge splits; hash-assign bridges unseen by v3."""
    known: dict[str, str] = {}
    if v3_windows is not None:
        with open(v3_windows, encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                known[row["asset_id"]] = row["split"]

    def fallback(bridge_id: str) -> str:
        bucket = int(hashlib.sha256(bridge_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        if bucket < SPLIT_SHARES[0]:
            return "train"
        return "validation" if bucket < SPLIT_SHARES[0] + SPLIT_SHARES[1] else "test"

    unique = pd.Series(bridge_ids.unique())
    mapping = {bid: known.get(bid) or fallback(bid) for bid in unique}
    return bridge_ids.map(mapping)


def build(
    nbi: pd.DataFrame,
    output_dir: Path,
    *,
    v3_windows: Path | None = None,
    source_inputs: dict | None = None,
) -> dict:
    panel = add_history_features(nbi)
    panel["is_v3_anchor"] = mark_v3_anchors(panel)
    panel["split"] = assign_splits(panel["bridge_id"], v3_windows)
    for column in CATEGORICAL_FEATURES:
        panel[column] = panel[column].astype("category")

    output_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_dir / "bridge_years.parquet", index=False)

    anchors = panel[panel["is_v3_anchor"]]
    summary = {
        "dataset_id": DATASET_ID,
        "unit": "bridge_year",
        "row_count": int(len(panel)),
        "bridge_count": int(panel["bridge_id"].nunique()),
        "years": [int(panel["year"].min()), int(panel["year"].max())],
        "states_included": sorted(panel["state_code"].unique().tolist()),
        "model_features": MODEL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "label_columns": LABEL_COLUMNS,
        "label_rule": (
            "label_* columns look at later years and must never be model inputs. "
            "label_deck_drop_5yr reproduces the v3 deck_drop_5yr target."
        ),
        "hazard_event_count": int(panel["label_deck_drop_next"].eq(1).sum()),
        "hazard_interval_count": int(panel["label_deck_drop_next"].notna().sum()),
        "v3_anchor_count": int(len(anchors)),
        "v3_anchor_positive_rate": round(float(anchors["label_deck_drop_5yr"].mean()), 6) if len(anchors) else None,
        "splits_by_bridge": panel.drop_duplicates("bridge_id")["split"].value_counts().to_dict(),
        "split_source": "v3 windows.csv by asset_id, hash fallback" if v3_windows else "hash by bridge_id",
        "license_posture": "public_research_reproducible_cc_by_4.0",
        "build_provenance": {
            "python_version": sys.version.split()[0],
            "package_versions": package_versions(("numpy", "pandas", "scipy", "pyarrow")),
            "source_inputs": source_inputs or {},
        },
    }
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--nshm-csv", required=True, type=Path)
    parser.add_argument("--nfhl-csv", required=True, type=Path)
    parser.add_argument("--states", nargs="+", default=["CA", "IA", "PA"])
    parser.add_argument("--v3-windows", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/cauren_bridge_v4"))
    args = parser.parse_args()

    fips_by_abbr = {"CA": "06", "IA": "19", "PA": "42", "DE": "10", "RI": "44"}
    states = {fips_by_abbr[s.upper()] for s in args.states if s.upper() in fips_by_abbr}
    if not states:
        raise SystemExit(f"No recognized state codes in {args.states}")

    nbi = load_nbi_years(args.raw_dir, states)
    nbi = join_hazard(nbi, args.nshm_csv, args.nfhl_csv)
    nbi = nbi.drop(columns=["seismic_component"])
    nbi["natural_hazard_score"] = nbi["natural_hazard_score"].astype(np.float32)
    source_inputs = {
        "nbi": [file_record(path, source_id="fhwa_nbi") for path in sorted(args.raw_dir.glob("*.txt"))],
        "usgs_nshm": file_record(args.nshm_csv, source_id="usgs_nshm"),
        "fema_nfhl": file_record(args.nfhl_csv, source_id="fema_nfhl"),
    }
    summary = build(nbi, args.output_dir, v3_windows=args.v3_windows, source_inputs=source_inputs)
    print(json.dumps({key: summary[key] for key in summary if key != "build_provenance"}, indent=2))


if __name__ == "__main__":
    main()
