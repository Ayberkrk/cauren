"""Build the cauren-bridge training dataset from public FHWA/USGS/FEMA data.

Source data (download manually, see the README table below, then point
--raw-dir at the folder holding the extracted files):

    | Source                         | Hugging Face dataset             | File(s) needed                                  |
    |---------------------------------|-----------------------------------|--------------------------------------------------|
    | FHWA National Bridge Inventory  | sweetapricity/bridgedeck-nbi      | data/<STATE>.zip per state, unzipped to *.txt    |
    | USGS seismic hazard             | sweetapricity/bridgedeck-nshm     | data/raw/nshm_2014_conus_0p05deg.csv             |
    | FEMA flood zone                 | sweetapricity/bridgedeck-nfhl     | data/raw/nfhl_bridge_zones.csv                   |

Example:

    python3 tools/build_cauren_bridge_dataset.py \\
        --raw-dir /path/to/nbi_raw \\
        --nshm-csv /path/to/nshm_2014_conus_0p05deg.csv \\
        --nfhl-csv /path/to/nfhl_bridge_zones.csv \\
        --states CA IA PA \\
        --output-dir data/cauren_bridge

Requires pandas and scipy in addition to the project's normal runtime
dependencies (matches the `dataset` extra in the root pyproject.toml).

Design notes (why the logic looks the way it does):

- NBI's LAT_016/LONG_017 fields are fixed-width DD(D)MMSS.ss codes, not
  decimal degrees: the trailing 6 digits are always MMSSss regardless of
  whether the degree part is 2 digits (latitude) or 3 (longitude), so
  the same divisor (10**6) peels off the degree part either way.
- structural_risk_score is 1 minus the worst (lowest) of deck,
  superstructure, and substructure condition, divided by 9 (the NBI
  0-9 scale, 9 = excellent). ground_stability_score comes from the
  scour-criticality rating (Item 113) via ITEM_113_RISK_BY_CODE, an
  explicit code -> risk mapping rather than a linear code/9.0 scale --
  Item 113 is categorical, not a linear condition rating, so a stable
  code (8) and a scour-critical code (3) are not "8/9 of the way" and
  "3/9 of the way" to the same failure mode. "N"/"U" (not over water /
  unevaluated) are left missing rather than guessed; "T" (tidal, not
  evaluated but documented by FHWA as low risk) gets an explicit small
  risk value instead of being lumped in with the true unknowns.
- A window's input rows are truncated to years at or before its own
  label's reference year (the anchor year). An earlier version of this
  script used each bridge's overall latest inspection as the window
  end regardless of the label's reference year, which let post-anchor
  rows leak the future condition rating the label was trying to
  predict, producing meaningless 100% validation accuracy. Truncating
  at the anchor year is what makes this a genuine forecasting setup.
- Splits are assigned by bridge_id (each bridge contributes at most one
  window here, so this also means by window) so no bridge appears in
  more than one split.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

MAX_SEQ_LEN = 16
MIN_REAL_YEARS = 8
FEATURES = ["structural_risk_score", "ground_stability_score", "natural_hazard_score"]

RAW_COLS = [
    "STATE_CODE_001", "STRUCTURE_NUMBER_008", "COUNTY_CODE_003",
    "LAT_016", "LONG_017", "YEAR_BUILT_027",
    "DECK_COND_058", "SUPERSTRUCTURE_COND_059", "SUBSTRUCTURE_COND_060",
    "SCOUR_CRITICAL_113",
]

# FHWA Item 113 ("Scour Critical Bridges") code -> scour risk/vulnerability
# score, where 1.0 means poor stability / high vulnerability. This matches
# how ground_stability_score is interpreted everywhere else in this project
# (see cauren_physics/bridge.py, cauren_physics/civil.py: it is combined
# positively with structural risk, so higher must mean higher risk).
#
# Item 113 is a categorical rating, not a linear 0-9 scale: codes 0-3
# describe worsening scour-critical conditions (0 = failed/closed, 3 =
# foundations unstable for the calculated/observed scour condition), while
# 4/5/7/8/9 describe stable or already-remediated conditions. A naive
# `code / 9.0` conversion scores a stable bridge (8 -> 0.89) as *more* at
# risk than a scour-critical one (3 -> 0.33), which is backwards.
#
# Source: FHWA Recording and Coding Guide for the NBI, Item 113 definitions.
ITEM_113_RISK_BY_CODE: dict[int, float] = {
    0: 1.00,  # failed and closed to traffic
    1: 0.90,  # scour critical, immediate action required to provide countermeasures
    2: 0.75,  # scour critical, field review indicates action required to protect foundations
    3: 0.60,  # scour critical, foundations unstable for calculated/observed scour
    4: 0.10,  # stable for calculated scour condition (tidal)
    5: 0.05,  # stable for calculated scour condition, within limits of footing/piles
    7: 0.05,  # previously scour-critical, corrected with countermeasures
    8: 0.05,  # stable for calculated scour condition, above bottom of footing
    9: 0.00,  # foundations not exposed to scour (above flood elevation / dry channel)
}
# Code 6 ("scour evaluation not yet complete", rarely used outside the
# original coding pass) and "U" (unknown foundation, not evaluated) carry no
# defensible risk direction and are left missing rather than guessed (they
# fall out of ITEM_113_RISK_BY_CODE naturally, since neither is a key in
# it). "N" (bridge not over a waterway) is not applicable and also left
# missing for the same reason.
# "T" (bridge over tidal waters, not evaluated) is explicitly documented by
# FHWA as low risk despite being unevaluated, so unlike the true unknowns
# above it gets a small, explicit non-missing risk value.
ITEM_113_TIDAL_UNEVALUATED_RISK = 0.10


def scour_risk_from_item_113(raw: "pd.Series") -> "pd.Series":
    """Map raw Item 113 codes to an explicit, documented risk score.

    See ITEM_113_RISK_BY_CODE's module-level comment for why this can't be
    a linear function of the code.
    """
    codes = raw.astype(str).str.strip().str.upper()
    numeric = pd.to_numeric(codes, errors="coerce")
    risk = numeric.map(ITEM_113_RISK_BY_CODE)
    return risk.where(~codes.eq("T"), ITEM_113_TIDAL_UNEVALUATED_RISK)


def _year_from_filename(path: str) -> int | None:
    match = re.match(r"^[A-Za-z]{2}(\d{2})\.txt$", os.path.basename(path))
    return 2000 + int(match.group(1)) if match else None


def _dms_to_decimal(raw: pd.Series, *, is_lon: bool) -> pd.Series:
    raw = raw.fillna(0).astype(np.int64)
    deg = raw // 1_000_000
    rem = raw % 1_000_000
    minutes = rem // 10000
    seconds = (rem % 10000) / 100.0
    dec = deg + minutes / 60.0 + seconds / 3600.0
    return -dec if is_lon else dec


def load_nbi(raw_dir: Path, states: set[str]) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(raw_dir, "??[0-9][0-9].txt")))
    if not files:
        raise SystemExit(f"No NBI *.txt files found under {raw_dir}")
    frames = []
    for path in files:
        year = _year_from_filename(path)
        if year is None:
            continue
        try:
            df = pd.read_csv(path, usecols=lambda c: c in RAW_COLS, dtype=str, low_memory=False, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, usecols=lambda c: c in RAW_COLS, dtype=str, low_memory=False, encoding="latin-1")
        state = df["STATE_CODE_001"].str.strip().str.zfill(2)
        if states and not state.isin(states).any():
            continue
        df = df[state.isin(states)] if states else df
        df["year"] = year
        frames.append(df)
    if not frames:
        raise SystemExit("No rows matched the requested states.")
    nbi = pd.concat(frames, ignore_index=True)

    def to_num(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    nbi["state_code"] = nbi["STATE_CODE_001"].str.strip().str.zfill(2)
    nbi["county_code"] = nbi["COUNTY_CODE_003"].str.strip().str.zfill(3)
    nbi["structure_number"] = nbi["STRUCTURE_NUMBER_008"].str.strip()
    nbi["bridge_id"] = nbi["state_code"] + "_" + nbi["county_code"] + "_" + nbi["structure_number"]
    nbi["deck_cond"] = to_num(nbi["DECK_COND_058"])
    nbi["superstructure_cond"] = to_num(nbi["SUPERSTRUCTURE_COND_059"])
    nbi["substructure_cond"] = to_num(nbi["SUBSTRUCTURE_COND_060"])
    nbi["scour_code"] = to_num(nbi["SCOUR_CRITICAL_113"])  # kept for debugging/inspection
    nbi["year_built"] = to_num(nbi["YEAR_BUILT_027"])
    nbi["lat_dd"] = _dms_to_decimal(to_num(nbi["LAT_016"]), is_lon=False)
    nbi["lon_dd"] = _dms_to_decimal(to_num(nbi["LONG_017"]), is_lon=True)
    valid_coord = nbi["lat_dd"].between(15, 72) & nbi["lon_dd"].between(-180, -60)
    nbi.loc[~valid_coord, ["lat_dd", "lon_dd"]] = np.nan

    cond_min = nbi[["deck_cond", "superstructure_cond", "substructure_cond"]].min(axis=1, skipna=True)
    has_structural = nbi[["deck_cond", "superstructure_cond", "substructure_cond"]].notna().any(axis=1)
    nbi["structural_risk_score"] = np.where(has_structural, 1.0 - (cond_min.clip(0, 9) / 9.0), np.nan)
    nbi["ground_stability_score"] = scour_risk_from_item_113(nbi["SCOUR_CRITICAL_113"]).to_numpy()
    return nbi


def join_hazard(nbi: pd.DataFrame, nshm_csv: Path, nfhl_csv: Path) -> pd.DataFrame:
    from scipy.spatial import cKDTree

    nshm = pd.read_csv(nshm_csv)
    nfhl = pd.read_csv(nfhl_csv)

    tree = cKDTree(nshm[["lon", "lat"]].values)
    coords = nbi[["lon_dd", "lat_dd"]].values
    valid = ~np.isnan(coords).any(axis=1)
    pga = np.full(len(nbi), np.nan)
    _dist, idx = tree.query(np.nan_to_num(coords[valid]), k=1)
    pga[valid] = nshm["pga_2pct_50yr"].values[idx]
    nbi["seismic_component"] = (pga / 0.6).clip(0, 1)

    nfhl_flag = nfhl.set_index("bridge_id")["sfha_tf"]
    flood_flag = nbi["bridge_id"].map(nfhl_flag)
    flood_component = flood_flag.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})

    has_seismic = nbi["seismic_component"].notna()
    has_flood = flood_component.notna()
    natural_hazard = pd.Series(np.nan, index=nbi.index)
    both = has_seismic & has_flood
    natural_hazard[both] = 0.6 * nbi.loc[both, "seismic_component"] + 0.4 * flood_component[both]
    natural_hazard[has_seismic & ~has_flood] = nbi.loc[has_seismic & ~has_flood, "seismic_component"]
    natural_hazard[has_flood & ~has_seismic] = flood_component[has_flood & ~has_seismic]
    nbi["natural_hazard_score"] = natural_hazard
    return nbi


def build_windows(nbi: pd.DataFrame, output_dir: Path, *, seed: int = 42) -> dict:
    nbi = nbi.drop_duplicates(subset=["bridge_id", "year"], keep="first")
    nbi = nbi.sort_values(["bridge_id", "year"]).reset_index(drop=True)

    censor_cutoff_year = int(nbi["year"].max()) - 5
    future = nbi[["bridge_id", "year", "deck_cond"]].copy()
    future["year"] = future["year"] - 5
    future = future.rename(columns={"deck_cond": "deck_cond_future"})
    nbi = nbi.merge(future, on=["bridge_id", "year"], how="left")
    nbi.loc[nbi["year"] > censor_cutoff_year, "deck_cond_future"] = np.nan
    drop = nbi["deck_cond_future"] < nbi["deck_cond"]
    same = nbi["deck_cond_future"] == nbi["deck_cond"]
    nbi["deck_drop_5yr"] = np.select([drop, same], [1.0, 0.0], default=np.nan)

    agent_dir = output_dir / "agents" / "cauren-bridge"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "splits").mkdir(parents=True, exist_ok=True)

    windows_fields = ["window_id", "agent_id", "sector", "asset_id", "client_id", "site_id",
                       "window_start", "window_end", "sampling_hz", "seq_len", "sensor_count",
                       "missing_ratio", "quality_score", "split", "training_role", "real_step_count"]
    labels_fields = ["window_id", "agent_id", "sector", "anomaly_type", "anomaly_family", "risk_score",
                      "confidence", "label_source", "event_start", "event_end", "root_cause_hint",
                      "notes", "is_anomaly", "deck_drop_5yr"]
    metadata_fields = ["asset_id", "agent_id", "sector", "metadata_json"]
    readings_fields = ["window_id", "agent_id", "sector", "asset_id", "client_id", "site_id",
                        "timestamp", "seq_index", "sensor_id", "sensor_name", "unit", "value",
                        "quality", "metadata_json"]

    windows_fp = open(agent_dir / "windows.csv", "w", newline="", encoding="utf-8")
    labels_fp = open(agent_dir / "labels.csv", "w", newline="", encoding="utf-8")
    metadata_fp = open(agent_dir / "asset_metadata.csv", "w", newline="", encoding="utf-8")
    readings_fp = open(agent_dir / "raw_sensor_readings.csv", "w", newline="", encoding="utf-8")
    windows_w = csv.DictWriter(windows_fp, fieldnames=windows_fields); windows_w.writeheader()
    labels_w = csv.DictWriter(labels_fp, fieldnames=labels_fields); labels_w.writeheader()
    metadata_w = csv.DictWriter(metadata_fp, fieldnames=metadata_fields); metadata_w.writeheader()
    readings_w = csv.DictWriter(readings_fp, fieldnames=readings_fields); readings_w.writeheader()

    window_idx = 0
    labeled_count = 0
    positive_count = 0
    skipped_too_short = 0
    window_ids_by_bridge: list[tuple[str, str]] = []

    for bridge_id, group in nbi.groupby("bridge_id", sort=False):
        group = group.sort_values("year")
        labeled_rows = group[group["deck_drop_5yr"].notna()]
        if len(labeled_rows) == 0:
            continue
        anchor_row = labeled_rows.iloc[-1]
        anchor_year = int(anchor_row["year"])

        window = group[group["year"] <= anchor_year].tail(MAX_SEQ_LEN)
        real_step_count = len(window)
        if real_step_count < MIN_REAL_YEARS:
            skipped_too_short += 1
            continue

        window_idx += 1
        window_id = f"bridge-{window_idx:06d}"
        is_anomaly = bool(anchor_row["deck_drop_5yr"] == 1.0)
        labeled_count += 1
        positive_count += int(is_anomaly)

        first_row = window.iloc[0]
        years = window["year"].tolist()
        metadata = {
            "asset_type": "bridge",
            "source_bundle": "fhwa_nbi_2000_2023",
            "jurisdiction": "US",
            "state_code": str(first_row["state_code"]),
            "structure_number": str(first_row["structure_number"]),
            "year_built": None if pd.isna(first_row["year_built"]) else float(first_row["year_built"]),
            "anchor_year": anchor_year,
        }
        metadata_json = json.dumps(metadata, sort_keys=True)

        windows_w.writerow({
            "window_id": window_id, "agent_id": "cauren-bridge", "sector": "civil",
            "asset_id": bridge_id, "client_id": "fhwa_nbi", "site_id": f"state-{first_row['state_code']}",
            "window_start": f"{years[0]}-07-01T00:00:00Z", "window_end": f"{years[-1]}-07-01T00:00:00Z",
            "sampling_hz": "3.168808781402895e-08",
            "seq_len": str(real_step_count), "sensor_count": str(len(FEATURES)),
            "missing_ratio": str(round(1.0 - window[FEATURES].notna().mean().mean(), 6)),
            "quality_score": "1.0", "split": "PLACEHOLDER", "training_role": "mae_reconstruction",
            "real_step_count": str(real_step_count),
        })
        labels_w.writerow({
            "window_id": window_id, "agent_id": "cauren-bridge", "sector": "civil",
            "anomaly_type": "deck_condition_drop" if is_anomaly else "nominal_variation",
            "anomaly_family": "structural_deterioration" if is_anomaly else "nominal",
            "risk_score": str(round(1.0 - float(anchor_row["deck_cond"]) / 9.0, 6)),
            "confidence": "1.0",
            "label_source": "fhwa_nbi_future_inspection_5yr",
            "event_start": f"{anchor_year}-07-01T00:00:00Z",
            "event_end": f"{anchor_year + 5}-07-01T00:00:00Z",
            "root_cause_hint": "structural_risk" if is_anomaly else "",
            "notes": "real_future_inspection_outcome_not_derived_from_input_features; input_window_truncated_at_anchor_year_to_avoid_leakage",
            "is_anomaly": "true" if is_anomaly else "false",
            "deck_drop_5yr": str(float(anchor_row["deck_drop_5yr"])),
        })
        metadata_w.writerow({"asset_id": bridge_id, "agent_id": "cauren-bridge", "sector": "civil", "metadata_json": metadata_json})

        for seq_index, (_, row) in enumerate(window.iterrows()):
            for feature in FEATURES:
                value = row[feature]
                readings_w.writerow({
                    "window_id": window_id, "agent_id": "cauren-bridge", "sector": "civil",
                    "asset_id": bridge_id, "client_id": "fhwa_nbi", "site_id": f"state-{first_row['state_code']}",
                    "timestamp": f"{int(row['year'])}-07-01T00:00:00Z", "seq_index": str(seq_index),
                    "sensor_id": f"{bridge_id}:{feature}", "sensor_name": feature, "unit": "ratio",
                    "value": "" if pd.isna(value) else str(round(float(value), 6)),
                    "quality": "false" if pd.isna(value) else "true",
                    "metadata_json": metadata_json,
                })
        window_ids_by_bridge.append((window_id, bridge_id))

    windows_fp.close(); labels_fp.close(); metadata_fp.close(); readings_fp.close()

    bridge_ids_unique = sorted({bid for _wid, bid in window_ids_by_bridge})
    rng = np.random.default_rng(seed)
    rng.shuffle(bridge_ids_unique)
    n = len(bridge_ids_unique)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    split_of = {}
    for i, bid in enumerate(bridge_ids_unique):
        split_of[bid] = "train" if i < n_train else ("validation" if i < n_train + n_val else "test")

    windows_path = agent_dir / "windows.csv"
    rows = list(csv.DictReader(open(windows_path, encoding="utf-8")))
    wid_to_bid = dict(window_ids_by_bridge)
    split_lists: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for row in rows:
        bid = wid_to_bid[row["window_id"]]
        split = split_of[bid]
        row["split"] = split
        split_lists[split].append(row["window_id"])
    with open(windows_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=windows_fields)
        writer.writeheader()
        writer.writerows(rows)

    for split, ids in split_lists.items():
        filename = {"train": "train_windows.txt", "validation": "validation_windows.txt", "test": "test_windows.txt"}[split]
        with open(output_dir / "splits" / filename, "w") as fp:
            fp.write("\n".join(ids) + "\n")

    summary = {
        "dataset_id": "cauren_bridge_fhwa_nbi_v3_scour_risk_direction_fixed",
        "agent_id": "cauren-bridge",
        "sector": "civil",
        "asset_count": window_idx,
        "features": FEATURES,
        "required_features": FEATURES,
        "optional_features": [],
        "feature_definitions": {
            "ground_stability_score": (
                "Scour risk/vulnerability derived from FHWA NBI Item 113 via an explicit "
                "code -> risk mapping (see ITEM_113_RISK_BY_CODE in "
                "tools/build_cauren_bridge_dataset.py), not a linear code/9.0 scale. 1.0 means "
                "poor stability / high vulnerability (e.g. code 0, failed and closed); 0.0 means "
                "foundations not exposed to scour (code 9). Codes 'N' (not over a waterway) and "
                "'U' (unknown foundation, unevaluated) are left missing; code 'T' (tidal, "
                "unevaluated but documented by FHWA as low risk) is scored "
                f"{ITEM_113_TIDAL_UNEVALUATED_RISK}."
            ),
        },
        "license_posture": "public_research_reproducible_cc_by_4.0",
        "source": "FHWA National Bridge Inventory 2000-2023 (via sweetapricity/bridgedeck-nbi, sweetapricity/bridgedeck-nshm, sweetapricity/bridgedeck-nfhl on Hugging Face)",
        "states_included": sorted(nbi["state_code"].unique().tolist()),
        "sampling_hz": 3.168808781402895e-08,
        "splits": {k: len(v) for k, v in split_lists.items()},
        "window_count": window_idx,
        "supervised_target": "deck_drop_5yr",
        "supervised_target_description": (
            "1 if NBI deck condition rating dropped by >=1 point within a real, independently-observed "
            "5-year future inspection; 0 if stable. Every window excludes any input row from after its "
            "own anchor_year to prevent the future outcome from leaking into the input features."
        ),
        "labeled_window_count": labeled_count,
        "positive_rate": round(positive_count / max(1, labeled_count), 6),
        "skipped_too_short_after_truncation": skipped_too_short,
    }
    with open(output_dir / "dataset_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", required=True, type=Path, help="Directory with unzipped NBI *.txt files")
    parser.add_argument("--nshm-csv", required=True, type=Path)
    parser.add_argument("--nfhl-csv", required=True, type=Path)
    parser.add_argument("--states", nargs="+", default=["CA", "IA", "PA"], help="Two-letter state codes")
    parser.add_argument("--output-dir", type=Path, default=Path("data/cauren_bridge"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fips_by_abbr = {
        "CA": "06", "IA": "19", "PA": "42", "TX": "48", "FL": "12", "NY": "36",
        "CO": "08", "WA": "53", "OH": "39", "IL": "17", "MO": "29", "OK": "40",
        "KS": "20",
    }
    states = {fips_by_abbr[s.upper()] for s in args.states if s.upper() in fips_by_abbr}
    if not states:
        raise SystemExit(f"No recognized state codes in {args.states}. Add missing FIPS codes to fips_by_abbr.")

    nbi = load_nbi(args.raw_dir, states)
    nbi = join_hazard(nbi, args.nshm_csv, args.nfhl_csv)
    summary = build_windows(nbi, args.output_dir, seed=args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
