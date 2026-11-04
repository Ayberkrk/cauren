"""Vänersborg bridge study: novelty detection across a real, verified fracture.

Uses the monitoring data of the Vänersborg railway bridge in Sweden, a
single-leaf bascule bridge of riveted steel trusses from 1916 (Zenodo record
8300495, CC BY 4.0; data article in Data in Brief, 2023, PMC10641142). The
record holds 64 bridge openings between 2023-02-20 and 2023-03-11. During the
opening at 2023-03-09 23:45 a counterweight truss member cracked near strain
gauges SG9 and SG10, which an inspection then confirmed.

The records are openings, not ambient vibration, so the features are not
natural frequencies. As the leaf rotates, the dead load in the counterweight
truss redistributes; each strain gauge's change from the closed position at
fixed leaf angles is a signature of how the truss carries that load, and a
fracture changes it. Differences from the closed position remove most of the
thermal offset in the strains.

Segments (half-open, fixed before any feature was computed):

- reference: openings from 2023-02-20 to 2023-03-02, 40 events;
- holdout: openings from 2023-03-03 to the last one before the fracture;
- fracture: the opening during which the member cracked;
- changed: openings after the fracture.

Every opening is scored with the shared detectors and conformal p-values in
``cauren_physics.novelty.evaluate_states`` at alpha 0.05 (40 reference events
allow p down to 1/41).

The data is not bundled. ``--fetch`` downloads the 550 MB zip from Zenodo,
checks its published MD5, computes the features of every opening and keeps
only those, in ``data/vanersborg/features.json`` (gitignored); the zip is
deleted afterwards. Needs numpy and pandas (``pip install -e '.[study]'``).

Usage:
    python3 tools/vanersborg_study.py --fetch
    python3 tools/vanersborg_study.py --json-out runtime_outputs/vanersborg_study.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cauren_physics.novelty import DETECTORS, MaxDeviationDetector, evaluate_states

DEFAULT_FEATURES = PROJECT_ROOT / "data" / "vanersborg" / "features.json"

ZENODO_RECORD = "8300495"
ZENODO_DOI = "10.5281/zenodo.8300495"
DATA_LICENSE = "CC BY 4.0"
DATA_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/DiB.zip/content"
DATA_MD5 = "4235a9755b301b193f61a1183ef1b21b"

SAMPLE_RATE_HZ = 200.0
STRAIN_CHANNELS = tuple(f"ch_{index}" for index in range(1, 17))  # SG1 to SG16
INCLINATION_CHANNEL = "ch_25"
AIR_TEMPERATURE_CHANNEL = "ch_30"
DEAD_VALUE = -1000000.0

CLOSED_SECONDS = 10.0
OPENING_MARGIN_DEG = 1.0
ANGLES_DEG = (-30.0, -60.0)
ANGLE_TOLERANCE_DEG = 1.0

REFERENCE = (dt.datetime(2023, 2, 20), dt.datetime(2023, 3, 3))
HOLDOUT_START = dt.datetime(2023, 3, 3)
FRACTURE_EVENT = dt.datetime(2023, 3, 9, 23, 45, 25)
ALPHA = 0.05
# A gauge whose reference features have a standard deviation this many
# times their robust (MAD) scale has outliers no healthy load path explains.
# SG11 reaches 30 to 53; every other gauge stays below 6.
ERRATIC_RATIO = 10.0


def event_time(member: str) -> dt.datetime:
    """Start time of an opening from its file name, for example ``2023_03_09T23_45_25.csv``."""
    return dt.datetime.strptime(Path(member).stem, "%Y_%m_%dT%H_%M_%S")


def opening_features(strain: np.ndarray, inclination: np.ndarray) -> np.ndarray | None:
    """Strain change from the closed position at each of ``ANGLES_DEG`` while the leaf opens.

    The closed level is the median of the first ``CLOSED_SECONDS``; the
    opening runs from the first sample more than ``OPENING_MARGIN_DEG`` below
    it to the most open sample. At each angle the strain is the median over
    opening samples within ``ANGLE_TOLERANCE_DEG``. Returns ``None`` when the
    leaf never reaches every angle, so no event is scored on a partial
    signature.
    """
    closed_samples = int(CLOSED_SECONDS * SAMPLE_RATE_HZ)
    closed_angle = float(np.median(inclination[:closed_samples]))
    closed_strain = np.median(strain[:closed_samples], axis=0)
    below = np.flatnonzero(inclination < closed_angle - OPENING_MARGIN_DEG)
    if len(below) == 0:
        return None
    start = below[0]
    stop = start + int(np.argmin(inclination[start:])) + 1
    phase = inclination[start:stop]
    rows = []
    for angle in ANGLES_DEG:
        near = np.abs(phase - angle) <= ANGLE_TOLERANCE_DEG
        if not near.any():
            return None
        rows.append(np.median(strain[start:stop][near], axis=0) - closed_strain)
    return np.concatenate(rows)


def fetch_features(source: str | Path = DATA_URL, target: Path = DEFAULT_FEATURES) -> Path:
    """Compute the features of every opening and keep only those.

    ``source`` is the Zenodo URL, downloaded to a temporary file next to
    ``target`` and deleted afterwards, or a local copy of ``DiB.zip``. The
    zip's MD5 is checked against the published one, and ``zipfile`` checks
    each member's CRC-32 as it is read.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    remote = str(source).startswith(("http://", "https://"))
    archive = target.parent / "DiB.zip.part" if remote else Path(source)
    try:
        if remote:
            urllib.request.urlretrieve(str(source), archive)
        digest = hashlib.md5()
        with archive.open("rb") as handle:
            for block in iter(lambda: handle.read(2**24), b""):
                digest.update(block)
        if digest.hexdigest() != DATA_MD5:
            raise ValueError(f"{archive} has MD5 {digest.hexdigest()}, expected {DATA_MD5}")
        events = _events_from_zip(archive)
    finally:
        if remote:
            archive.unlink(missing_ok=True)
    payload = {"zenodo_doi": ZENODO_DOI, "license": DATA_LICENSE, "angles_deg": list(ANGLES_DEG), "events": events}
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def _events_from_zip(archive: Path) -> list[dict]:
    import pandas as pd

    columns = ["ts", *STRAIN_CHANNELS, INCLINATION_CHANNEL, AIR_TEMPERATURE_CHANNEL]
    events = []
    with zipfile.ZipFile(archive) as bundle:
        for info in sorted(bundle.infolist(), key=lambda item: item.filename):
            if not info.filename.endswith(".csv"):
                continue
            with bundle.open(info) as member:
                frame = pd.read_csv(member, usecols=columns)
            strain = frame[list(STRAIN_CHANNELS)].to_numpy(dtype=float)
            inclination = frame[INCLINATION_CHANNEL].to_numpy(dtype=float)
            dead = [name for name, column in zip(STRAIN_CHANNELS, strain.T) if np.any(column == DEAD_VALUE)]
            features = opening_features(strain, inclination)
            events.append(
                {
                    "file": info.filename,
                    "start": event_time(info.filename).isoformat(),
                    "duration_s": len(frame) / SAMPLE_RATE_HZ,
                    "min_inclination_deg": float(inclination.min()),
                    "air_temperature_c": float(np.median(frame[AIR_TEMPERATURE_CHANNEL])),
                    "dead_channels": dead,
                    "features": None if features is None else [float(value) for value in features],
                }
            )
            print(f"{info.filename}: {'ok' if features is not None else 'no full opening'}", flush=True)
    return events


def feature_names(dead: set[str]) -> list[tuple[str, float]]:
    return [
        (f"SG{STRAIN_CHANNELS.index(channel) + 1}", angle)
        for angle in ANGLES_DEG
        for channel in STRAIN_CHANNELS
        if channel not in dead
    ]


def erratic_gauges(reference: np.ndarray, names: list[tuple[str, float]]) -> list[str]:
    """Gauges with at least one reference feature whose std exceeds ``ERRATIC_RATIO`` robust scales."""
    centre = np.median(reference, axis=0)
    robust = 1.4826 * np.median(np.abs(reference - centre), axis=0)
    ratio = np.std(reference, axis=0, ddof=1) / robust
    return sorted({names[index][0] for index in np.flatnonzero(ratio > ERRATIC_RATIO)})


def run_study(
    path: Path = DEFAULT_FEATURES, alpha: float = ALPHA, max_deviation: bool = False, exclude_erratic: bool = False
) -> dict:
    """Score the holdout, the fracture opening and the later openings.

    ``max_deviation`` adds the largest robust z-score over the features as a
    fourth arm, and ``exclude_erratic`` drops gauges whose reference features
    are heavy-tailed. Both were added after the default arms had been seen
    to miss the fracture, so their results are post hoc.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload["events"]
    dead = {channel for event in events for channel in event["dead_channels"]}
    keep = np.array([channel not in dead for channel in STRAIN_CHANNELS] * len(ANGLES_DEG))
    usable = [event for event in events if event["features"] is not None]
    groups: dict[str, list[dict]] = {"reference": [], "holdout": [], "fracture": [], "changed": []}
    for event in usable:
        start = dt.datetime.fromisoformat(event["start"])
        if REFERENCE[0] <= start < REFERENCE[1]:
            groups["reference"].append(event)
        elif HOLDOUT_START <= start < FRACTURE_EVENT:
            groups["holdout"].append(event)
        elif start == FRACTURE_EVENT:
            groups["fracture"].append(event)
        else:
            groups["changed"].append(event)
    names = feature_names(dead)
    matrices = {
        label: np.array([np.asarray(event["features"])[keep] for event in members])
        for label, members in groups.items()
    }
    excluded = erratic_gauges(matrices["reference"], names) if exclude_erratic else []
    if excluded:
        retained = np.array([gauge not in excluded for gauge, _ in names])
        matrices = {label: values[:, retained] for label, values in matrices.items()}
        names = [name for name, used in zip(names, retained) if used]
    result = evaluate_states(
        matrices["reference"],
        matrices["holdout"],
        {"fracture": matrices["fracture"], "changed": matrices["changed"]},
        alpha=alpha,
        detectors={**DETECTORS, "max_deviation": MaxDeviationDetector} if max_deviation else None,
    )
    centre = np.median(matrices["reference"], axis=0)
    spread = np.std(matrices["reference"], axis=0, ddof=1)
    shifts = {
        label: (np.median(values, axis=0) - centre) / spread
        for label, values in matrices.items()
        if label != "reference" and len(values)
    }
    largest = {
        label: [
            {"gauge": names[index][0], "angle_deg": names[index][1], "shift_over_sigma": float(values[index])}
            for index in np.argsort(-np.abs(values))[:5]
        ]
        for label, values in shifts.items()
    }
    temperatures = {
        label: [min(event["air_temperature_c"] for event in members), max(event["air_temperature_c"] for event in members)]
        for label, members in groups.items()
        if members
    }
    return {
        "source": {"zenodo_doi": ZENODO_DOI, "license": DATA_LICENSE},
        "events": {label: len(members) for label, members in groups.items()},
        "events_without_full_opening": [event["file"] for event in events if event["features"] is None],
        "dead_channels": sorted(dead),
        "excluded_erratic_gauges": excluded,
        "features": len(names),
        "air_temperature_range_c": temperatures,
        "largest_shifts": largest,
        **result,
    }


def markdown_table(result: dict) -> str:
    arms = result["arms"]
    lines = [
        "| Openings | Events | " + " | ".join(f"{name} alarmed (median p)" for name in arms) + " |",
        "| --- | --- | " + " | ".join("---" for _ in arms) + " |",
    ]
    for label in ("holdout", "fracture", "changed"):
        cells = [
            f"{arm[label]['alarmed']} ({100 * arm[label]['alarm_rate']:.0f}%, p {arm[label]['median_p_value']:.3f})"
            for arm in arms.values()
        ]
        lines.append(f"| {label} | {arms['knn'][label]['samples']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument(
        "--fetch", action="store_true", help="download the 550 MB zip, keep only the features, delete the zip"
    )
    parser.add_argument("--source", default=DATA_URL, help="Zenodo URL (default) or a local copy of DiB.zip")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument(
        "--max-deviation", action="store_true", help="post hoc: add the largest robust z-score as a fourth arm"
    )
    parser.add_argument(
        "--exclude-erratic", action="store_true", help="post hoc: drop gauges with heavy-tailed reference features"
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    if args.fetch:
        fetch_features(args.source, args.features)
    if not args.features.exists():
        print(f"{args.features} not found; rerun with --fetch", file=sys.stderr)
        return 2

    result = run_study(
        args.features, alpha=args.alpha, max_deviation=args.max_deviation, exclude_erratic=args.exclude_erratic
    )
    print(markdown_table(result))
    summary_keys = (
        "events",
        "events_without_full_opening",
        "dead_channels",
        "excluded_erratic_gauges",
        "features",
        "largest_shifts",
    )
    print(json.dumps({key: result[key] for key in summary_keys}, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
