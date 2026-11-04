"""KW51 railway bridge case study: environmental normalization before novelty detection.

Uses the hourly identified natural frequencies and deck temperature published
by Maes and Lombaert (KU Leuven) for steel bowstring railway bridge KW51 in
Leuven, Belgium (Zenodo record 3745914, CC BY-NC-SA 4.0). The diagonal
connections were strengthened between 2019-05-15 and 2019-09-27, so the
structure after the retrofit is a known, real change of state.

The detectors and the protocol live in ``cauren_physics.novelty`` and are
shared with the other bridge studies: fit on the reference period, set the
control limit on the reference daily scores, then count alarmed days in a
later healthy holdout (false alarms) and after the retrofit (detections). This is a single-bridge case study, not a validation
of these detectors in general.

The data is not bundled. ``--download`` fetches ``trackedmodes.zip`` (about
13 MB) into ``data/kw51/raw/``, which is gitignored. Needs numpy and scipy
(``pip install -e '.[study]'``). Timoshenko's ``fit_environmental_baseline``
is used for the temperature fit when it is installed; otherwise an identical
ordinary least-squares fit runs here.

Usage:
    python3 tools/kw51_study.py --download
    python3 tools/kw51_study.py --json-out runtime_outputs/kw51_study.json
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

from cauren_physics.novelty import StudyProtocol, markdown_table, run_protocol

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "kw51" / "raw"

ZENODO_RECORD = "3745914"
ZENODO_DOI = "10.5281/zenodo.3745914"
DATA_LICENSE = "CC BY-NC-SA 4.0"
TRACKED_MODES_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/trackedmodes.zip/content"
TRACKED_MODES_MD5 = "5cdb446519d1150228fab51b3a002239"

# Segment boundaries are half-open [start, end) in local hours as published.
# The retrofit works (2019-05-15 to 2019-09-27) fall between holdout and
# changed and are not scored.
PROTOCOL = StudyProtocol(
    reference=(dt.datetime(2018, 10, 1), dt.datetime(2019, 4, 1)),
    holdout=(dt.datetime(2019, 4, 1), dt.datetime(2019, 5, 15)),
    changed=(dt.datetime(2019, 9, 28), dt.datetime(2020, 1, 16)),
)

# Structural steel loses roughly 0.03 percent of its Young's modulus per
# kelvin near room temperature. With f proportional to sqrt(E), the
# material-only frequency sensitivity is about half of that.
STEEL_DE_OVER_E_PER_K = -3.0e-4
EXPECTED_DF_OVER_F_PER_K = 0.5 * STEEL_DE_OVER_E_PER_K


def download_tracked_modes(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Download and unpack ``trackedmodes.mat``, verifying the published MD5."""
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "trackedmodes.zip"
    if not archive.exists():
        urllib.request.urlretrieve(TRACKED_MODES_URL, archive)
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    if digest != TRACKED_MODES_MD5:
        raise ValueError(f"{archive} has MD5 {digest}, expected {TRACKED_MODES_MD5}")
    with zipfile.ZipFile(archive) as bundle:
        member = next(name for name in bundle.namelist() if name.endswith("trackedmodes.mat"))
        target = data_dir / "trackedmodes.mat"
        target.write_bytes(bundle.read(member))
    return target


def datenum_to_datetime(datenums) -> list[dt.datetime]:
    """Convert MATLAB datenums (day 1 is 0000-01-01) to rounded hourly datetimes."""
    out = []
    for value in np.asarray(datenums, dtype=float):
        stamp = dt.datetime.fromordinal(int(value)) + dt.timedelta(days=float(value) % 1.0)
        stamp -= dt.timedelta(days=366)
        out.append((stamp + dt.timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0))
    return out


def load_tracked_modes(path: Path) -> dict:
    """Read hourly frequencies (Hz) and deck steel temperature (C) from the MAT file."""
    from scipy.io import loadmat

    modes = loadmat(path, squeeze_me=True, struct_as_record=False)["modes"]
    env_labels = [str(label) for label in modes.labels_env]
    return {
        "times": datenum_to_datetime(modes.sdn),
        "frequencies_hz": np.asarray(modes.f, dtype=float),
        "deck_temperature_c": np.asarray(modes.env[:, env_labels.index("tBD31A")], dtype=float),
    }


def run_study(data: dict, modes: list[int] | None = None) -> dict:
    """Run every detector arm. ``modes`` overrides the reference-only mode selection."""
    result = run_protocol(
        data["times"],
        data["frequencies_hz"],
        data["deck_temperature_c"],
        PROTOCOL,
        expected_df_over_f=EXPECTED_DF_OVER_F_PER_K,
        modes=modes,
    )
    result["deck_temperature_range_c"] = result.pop("temperature_range_c")
    return {"source": {"zenodo_doi": ZENODO_DOI, "license": DATA_LICENSE}, **result}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--download", action="store_true", help="fetch trackedmodes.zip from Zenodo first")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--modes",
        type=int,
        nargs="+",
        help="0-based mode indices to monitor instead of the reference-only selection",
    )
    args = parser.parse_args(argv)

    mat_path = args.data_dir / "trackedmodes.mat"
    if args.download:
        mat_path = download_tracked_modes(args.data_dir)
    if not mat_path.exists():
        print(f"{mat_path} not found; rerun with --download", file=sys.stderr)
        return 2

    result = run_study(load_tracked_modes(mat_path), modes=args.modes)
    print(markdown_table(result, changed_label="Post-retrofit"))
    summary_keys = ("modes", "baseline_source", "complete_hours", "deck_temperature_range_c", "frequency_ratio")
    print(json.dumps({key: result[key] for key in summary_keys}, indent=2))
    for row in result["temperature_sensitivity"]:
        print(
            f"{row['reference_hz']:.3f} Hz: df/f per K {row['df_over_f_per_k_above']:.2e} above hinge, "
            f"{row['df_over_f_per_k_below']:.2e} below, {row['ratio_to_material_only']:.1f}x material-only"
        )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
