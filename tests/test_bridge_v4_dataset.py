import csv

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

pytest.importorskip("pyarrow")

from tools.build_bridge_years_v4 import (
    CATEGORICAL_FEATURES,
    LABEL_COLUMNS,
    MODEL_FEATURES,
    RAW_COLS,
    add_history_features,
    assign_splits,
    build,
    load_nbi_years,
    parse_inspection_date,
)

DECKS = {
    "B1": [7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 5],
    "B2": [8, 8, 7, 7, 8, 8, 8, 8, 7, 7, 7, 7],
}


def _write_year(raw_dir, year, decks, adt):
    path = raw_dir / f"DE{year - 2000:02d}.txt"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLS)
        writer.writeheader()
        for structure, deck in decks.items():
            row = {column: "" for column in RAW_COLS}
            row.update({
                "STATE_CODE_001": "10", "STRUCTURE_NUMBER_008": structure, "COUNTY_CODE_003": "1",
                "LAT_016": "39000000", "LONG_017": "75300000",
                "DATE_OF_INSPECT_090": f"6{(year - 1 - 2000 if year % 2 else year - 2000):02d}",
                "SCOUR_CRITICAL_113": "8", "DECK_COND_058": str(deck),
                "SUPERSTRUCTURE_COND_059": "7", "SUBSTRUCTURE_COND_060": "7",
                "YEAR_BUILT_027": "1970", "YEAR_RECONSTRUCTED_106": "0", "ADT_029": str(adt),
                "PERCENT_ADT_TRUCK_109": "10", "INSPECT_FREQ_MONTHS_091": "24",
                "STRUCTURE_KIND_043A": "1", "DECK_PROTECTION_108C": "0",
            })
            writer.writerow(row)


def _panel(tmp_path, poison_after=None):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    years = range(2000, 2012)
    for index, year in enumerate(years):
        decks = {bridge: values[index] for bridge, values in DECKS.items()}
        adt = 1000
        if poison_after is not None and year > poison_after:
            decks = {bridge: 0 for bridge in decks}
            adt = 999999
        _write_year(raw_dir, year, decks, adt)
    nbi = load_nbi_years(raw_dir, {"10"})
    nbi["natural_hazard_score"] = np.float32(0.2)
    return add_history_features(nbi)


def test_inspection_date_parses_mmyy_without_leading_zero():
    parsed = parse_inspection_date(pd.Series(["399", "1299", "0622", "1308", ""]))
    assert parsed.iloc[0] == pytest.approx(1999 + 2.5 / 12)
    assert parsed.iloc[1] == pytest.approx(1999 + 11.5 / 12)
    assert parsed.iloc[2] == pytest.approx(2022 + 5.5 / 12)
    assert parsed.iloc[3:].isna().all()


def test_features_ignore_later_years(tmp_path):
    clean = _panel(tmp_path / "clean")
    poisoned = _panel(tmp_path / "poisoned", poison_after=2005)
    columns = [*MODEL_FEATURES, *CATEGORICAL_FEATURES]
    keep_clean = clean[clean["year"] <= 2005].reset_index(drop=True)
    keep_poisoned = poisoned[poisoned["year"] <= 2005].reset_index(drop=True)
    pd.testing.assert_frame_equal(keep_clean[columns], keep_poisoned[columns])
    assert not set(LABEL_COLUMNS) & set(MODEL_FEATURES)


def test_hazard_label_marks_drops_and_censors_repairs(tmp_path):
    panel = _panel(tmp_path).set_index(["bridge_id", "year"])
    b1 = panel.loc["10_001_B1", "label_deck_drop_next"]
    assert b1.loc[2002] == 1.0
    assert b1.loc[2003] == 0.0
    b2 = panel.loc["10_001_B2", "label_deck_drop_next"]
    assert np.isnan(b2.loc[2003])
    assert np.isnan(b1.loc[2011])


def test_history_features_count_runs_drops_and_repairs(tmp_path):
    panel = _panel(tmp_path).set_index(["bridge_id", "year"])
    b1 = panel.loc["10_001_B1"]
    assert b1.loc[2002, "years_at_deck_rating"] == 3
    assert b1.loc[2003, "years_at_deck_rating"] == 1
    assert b1.loc[2011, "deck_drops_past"] == 2
    b2 = panel.loc["10_001_B2"]
    assert b2.loc[2005, "deck_improvements_past"] == 1
    assert b1.loc[2001, "age_years"] == 31
    assert b1.loc[2001, "truck_adt"] == pytest.approx(100.0)


def test_five_year_label_matches_v3_definition(tmp_path):
    panel = _panel(tmp_path).set_index(["bridge_id", "year"])
    assert panel.loc[("10_001_B1", 2000), "label_deck_drop_5yr"] == 1.0
    assert panel.loc[("10_001_B2", 2004), "label_deck_drop_5yr"] == 1.0
    assert panel.loc[("10_001_B2", 2003), "label_deck_drop_5yr"] == 0.0
    assert np.isnan(panel.loc[("10_001_B1", 2007), "label_deck_drop_5yr"])


def test_splits_reuse_v3_assignment_and_hash_the_rest(tmp_path):
    windows = tmp_path / "windows.csv"
    windows.write_text("window_id,asset_id,split\nw1,known,test\n", encoding="utf-8")
    splits = assign_splits(pd.Series(["known", "unknown", "unknown"]), windows)
    assert splits.iloc[0] == "test"
    assert splits.iloc[1] == splits.iloc[2]
    assert splits.iloc[1] in {"train", "validation", "test"}


def test_build_writes_panel_and_summary(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for index, year in enumerate(range(2000, 2012)):
        _write_year(raw_dir, year, {bridge: values[index] for bridge, values in DECKS.items()}, 1000)
    nbi = load_nbi_years(raw_dir, {"10"})
    nbi["natural_hazard_score"] = np.float32(0.2)
    summary = build(nbi, tmp_path / "out")
    panel = pd.read_parquet(tmp_path / "out" / "bridge_years.parquet")
    assert summary["row_count"] == len(panel) == 24
    assert summary["bridge_count"] == 2
    assert set(MODEL_FEATURES) <= set(panel.columns)
