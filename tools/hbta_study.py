"""Hell Bridge Test Arena study: modal novelty detection against known imposed damage.

Uses the benchmark data set of the Hell Bridge Test Arena (HBTA), a riveted
steel truss railway bridge near Trondheim, Norway (Svendsen et al.; Zenodo
record 14632942, CC BY 4.0). In September 2020 eight damage states were
imposed by temporarily removing bolts that had replaced rivets in deck and
bracing connections, and each was repaired afterwards. A modal vibration
shaker excited the bridge while 18 global and 40 local accelerometers
recorded the response.

Unlike KW51 this is not a continuous record: each structural state has one
recording per shaker setup, and the undamaged state has two, measured on
2020-09-22 and 2020-09-27. The study uses the vertical white-noise
recordings with the shaker at position P2 (``MVS_P2_*_NM_Z_*``), because
their excitation is stationary and they can be cut into windows. In each
window the H1 frequency response from the shaker acceleration to the
vertical global accelerometers gives the natural frequencies of the modes
picked on the reference. The first undamaged recording is the reference,
the second one, five days later, is the healthy holdout, and each damage
state is scored against the reference with the shared detectors and
conformal p-values in ``cauren_physics.novelty``.

The data is not bundled. The full record is a single 2.5 GB HDF5 file, but
the study needs only 10 recordings and 19 channels of it, about 145 MB.
``--fetch`` reads just those datasets from Zenodo with HTTP range requests
and stores them in ``data/hbta/hbta_p2_noise_z.npz`` (gitignored) with a
SHA-256 of every array; ``--fetch --source data_100Hz.h5`` builds the same
cache from a downloaded copy of the full file instead. Needs numpy, scipy
and h5py, plus fsspec and aiohttp for the remote read
(``pip install -e '.[study]'``).

Usage:
    python3 tools/hbta_study.py --fetch
    python3 tools/hbta_study.py --json-out runtime_outputs/hbta_study.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cauren_physics.novelty import evaluate_states

DEFAULT_CACHE = PROJECT_ROOT / "data" / "hbta" / "hbta_p2_noise_z.npz"

ZENODO_RECORD = "14632942"
ZENODO_DOI = "10.5281/zenodo.14632942"
DATA_LICENSE = "CC BY 4.0"
DATA_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/data_100Hz.h5/content"
DATA_MD5 = "43406591cd7f8b8bd1b0b868d3312e38"  # of the full file, for a local --source copy
# Small reads: each channel is one contiguous dataset of about 760 kB, and a
# large read-ahead block would fetch several megabytes of unrelated data.
REMOTE_BLOCK_BYTES = 2**16

SAMPLE_RATE_HZ = 100.0
INPUT_CHANNEL = ("AS", "z")
# AG09 is the only global sensor without a lateral channel; all 18 have z.
OUTPUT_CHANNELS = tuple((f"AG{index:02d}", "z") for index in range(1, 19))

REFERENCE = "MVS_P2_UDS_NM_Z_02"  # 2020-09-22, before any damage
HOLDOUT = "MVS_P2_UDS_NM_Z_01"  # 2020-09-27, undamaged again after repairs
DAMAGE_STATES = tuple(f"MVS_P2_DS{index}_NM_Z_01" for index in range(1, 9))

# Damage state descriptions from Svendsen et al. (2023), Table 1.
DAMAGE_DESCRIPTIONS = {
    "DS1": "local: one stringer-to-floor-beam connection",
    "DS2": "local: several stringer-to-floor-beam connections",
    "DS3": "local: main part of one stringer cross beam removed",
    "DS4": "local: main parts of several stringer cross beams removed",
    "DS5": "global: one lateral bracing connection",
    "DS6": "global: one lateral bracing connection",
    "DS7": "global: several lateral bracing connections",
    "DS8": "global: one floor-beam to main load-carrying member connection",
}

WINDOW_S = 60.0
EDGE_TRIM_S = 5.0
# The shaker takes one to two minutes to reach its level. A window whose
# input RMS is below this share of the recording's median is not yet at the
# steady excitation and is dropped; the rule looks only at the input signal.
MIN_INPUT_RMS_SHARE = 0.9
SEGMENT_SAMPLES = 1024
BAND_HZ = (2.0, 25.0)
MAX_MODES = 8
MIN_PEAK_PROMINENCE = 1.0  # natural-log units of the reference indicator
# Each band spans this share of the distance to the nearest kept peak on
# either side, capped, so a strong neighbour's flank cannot win a band.
BAND_SHARE = 0.4
MAX_HALF_WIDTH_HZ = 0.5
ALPHA = 0.1


def fetch_subset(source: str | Path = DATA_URL, cache: Path = DEFAULT_CACHE) -> Path:
    """Copy the channels this study reads into a small local cache.

    ``source`` is the Zenodo URL (read remotely with range requests) or a
    local copy of ``data_100Hz.h5``, whose MD5 is checked first. Each array
    is stored with its SHA-256 so a later run can tell the cache is intact.
    """
    import h5py

    arrays: dict[str, np.ndarray] = {}
    attributes: dict[str, dict] = {}

    def read(bundle):
        for recording in (REFERENCE, HOLDOUT, *DAMAGE_STATES):
            group = bundle[recording]
            for sensor, direction in (INPUT_CHANNEL, *OUTPUT_CHANNELS):
                arrays[f"{recording}/{sensor}/{direction}"] = np.asarray(
                    group["acceleration"][sensor][direction][:], dtype=float
                )
            attributes[recording] = {key: _plain(value) for key, value in group.attrs.items()}

    if str(source).startswith(("http://", "https://")):
        import fsspec

        with fsspec.open(str(source), "rb", block_size=REMOTE_BLOCK_BYTES) as handle:
            with h5py.File(handle, "r") as bundle:
                read(bundle)
    else:
        if _md5(Path(source)) != DATA_MD5:
            raise ValueError(f"{source} does not match the published MD5 {DATA_MD5}")
        with h5py.File(source, "r") as bundle:
            read(bundle)

    manifest = {
        "zenodo_doi": ZENODO_DOI,
        "license": DATA_LICENSE,
        "attributes": attributes,
        "sha256": {name: hashlib.sha256(values.tobytes()).hexdigest() for name, values in arrays.items()},
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, manifest=np.array(json.dumps(manifest)), **arrays)
    return cache


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**24), b""):
            digest.update(block)
    return digest.hexdigest()


def load_recording(cache, recording: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Shaker acceleration, the vertical global accelerations, and the recording's attributes.

    ``cache`` is an open ``np.load`` of the file written by ``fetch_subset``;
    every array is checked against its stored SHA-256.
    """
    manifest = json.loads(str(cache["manifest"]))

    def channel(sensor: str, direction: str) -> np.ndarray:
        name = f"{recording}/{sensor}/{direction}"
        values = cache[name]
        if hashlib.sha256(values.tobytes()).hexdigest() != manifest["sha256"][name]:
            raise ValueError(f"cached array {name} does not match its SHA-256; rerun with --fetch")
        return values

    excitation = channel(*INPUT_CHANNEL)
    responses = np.column_stack([channel(sensor, direction) for sensor, direction in OUTPUT_CHANNELS])
    return excitation, responses, manifest["attributes"][recording]


def _plain(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def windows(n_samples: int, sample_rate_hz: float = SAMPLE_RATE_HZ) -> list[slice]:
    """Non-overlapping windows, skipping the shaker's start and stop transients."""
    length = int(round(WINDOW_S * sample_rate_hz))
    trim = int(round(EDGE_TRIM_S * sample_rate_hz))
    starts = range(trim, n_samples - trim - length + 1, length)
    return [slice(start, start + length) for start in starts]


def frf_indicator(excitation: np.ndarray, responses: np.ndarray, sample_rate_hz: float = SAMPLE_RATE_HZ):
    """Mean squared H1 magnitude over the output channels.

    H1 = S_xy / S_xx divides out the shaker's own spectrum, so a peak is a
    resonance of the bridge rather than of the excitation. Averaging |H1|^2
    over channels keeps every mode that some channel sees.
    """
    from scipy.signal import csd, welch

    frequencies, input_psd = welch(excitation, fs=sample_rate_hz, nperseg=SEGMENT_SAMPLES)
    power = np.zeros_like(frequencies)
    for channel in range(responses.shape[1]):
        _, cross = csd(excitation, responses[:, channel], fs=sample_rate_hz, nperseg=SEGMENT_SAMPLES)
        power += np.abs(cross / input_psd) ** 2
    return frequencies, power / responses.shape[1]


def reference_modes(frequencies: np.ndarray, indicator: np.ndarray) -> list[tuple[float, float]]:
    """Frequency bands around the clearest resonances of the reference indicator.

    Peaks inside ``BAND_HZ`` are ranked by prominence on a log scale and the
    strongest ``MAX_MODES`` are kept. Each band is centred on its peak and
    reaches ``BAND_SHARE`` of the way to the nearest kept neighbour, at most
    ``MAX_HALF_WIDTH_HZ``: wide enough for a mode to drift, narrow enough
    that the flank of a stronger neighbouring mode is not picked instead.
    """
    from scipy.signal import find_peaks

    inside = (frequencies >= BAND_HZ[0]) & (frequencies <= BAND_HZ[1])
    log_indicator = np.log(indicator[inside])
    peaks, properties = find_peaks(log_indicator, prominence=MIN_PEAK_PROMINENCE)
    strongest = peaks[np.argsort(properties["prominences"])[::-1][:MAX_MODES]]
    centres = np.sort(frequencies[inside][strongest])
    spacing = np.diff(np.concatenate(([BAND_HZ[0]], centres, [BAND_HZ[1]])))
    half_widths = np.minimum(BAND_SHARE * np.minimum(spacing[:-1], spacing[1:]), MAX_HALF_WIDTH_HZ)
    return [(float(centre - half), float(centre + half)) for centre, half in zip(centres, half_widths)]


def peak_frequency(frequencies: np.ndarray, indicator: np.ndarray, band: tuple[float, float]) -> float:
    """Frequency of the indicator maximum in a band, refined by a parabola through log values."""
    inside = np.flatnonzero((frequencies >= band[0]) & (frequencies < band[1]))
    best = inside[np.argmax(indicator[inside])]
    if best in (0, len(frequencies) - 1):
        return float(frequencies[best])
    left, centre, right = np.log(indicator[best - 1 : best + 2])
    curvature = left - 2.0 * centre + right
    offset = 0.0 if curvature >= 0.0 else 0.5 * (left - right) / curvature
    return float(frequencies[best] + offset * (frequencies[1] - frequencies[0]))


def steady_windows(excitation: np.ndarray) -> list[slice]:
    """Windows in which the shaker runs at its steady level."""
    spans = windows(len(excitation))
    rms = np.array([np.std(excitation[span]) for span in spans])
    return [span for span, value in zip(spans, rms) if value >= MIN_INPUT_RMS_SHARE * np.median(rms)]


def window_frequencies(excitation, responses, bands) -> np.ndarray:
    """Natural frequencies per steady window and band, one row per window."""
    rows = []
    for window in steady_windows(excitation):
        frequencies, indicator = frf_indicator(excitation[window], responses[window])
        rows.append([peak_frequency(frequencies, indicator, band) for band in bands])
    return np.asarray(rows)


def run_study(
    cache_path: Path = DEFAULT_CACHE, alpha: float = ALPHA, reference: str = REFERENCE, holdout: str = HOLDOUT
) -> dict:
    """Score the holdout and every damage state against the reference recording.

    The default pairing is chronological: the first undamaged recording is
    the reference. Swapping the two undamaged recordings puts the reference
    in the same measurement session as most damage tests, which is a post
    hoc analysis and must be reported as one.
    """
    with np.load(cache_path) as cache:
        return _run_study(cache, alpha, reference, holdout)


def _run_study(path, alpha: float, reference_name: str, holdout_name: str) -> dict:
    excitation, responses, reference_attributes = load_recording(path, reference_name)
    frequencies, indicator = frf_indicator(excitation, responses)
    bands = reference_modes(frequencies, indicator)
    reference = window_frequencies(excitation, responses, bands)

    samples = {}
    attributes = {reference_name: reference_attributes}
    for recording in (holdout_name, *DAMAGE_STATES):
        excitation, responses, attributes[recording] = load_recording(path, recording)
        samples[recording] = window_frequencies(excitation, responses, bands)

    holdout = samples.pop(holdout_name)
    states = {name.split("_")[2]: values for name, values in samples.items()}
    result = evaluate_states(reference, holdout, states, alpha=alpha)
    reference_median = np.median(reference, axis=0)
    shifts = {
        label: [float(value) for value in (np.median(values, axis=0) - reference_median) / reference_median]
        for label, values in {"holdout": holdout, **states}.items()
    }
    # Within one recording the structure and the weather do not change, so
    # the window-to-window scatter of the reference is the identification
    # uncertainty of a single window.
    window_sigma = np.std(reference, axis=0, ddof=1)
    window_spread = [float(value) for value in window_sigma / reference_median]
    shift_over_sigma = {
        label: [float(value) for value in (np.median(values, axis=0) - reference_median) / window_sigma]
        for label, values in {"holdout": holdout, **states}.items()
    }
    return {
        "source": {"zenodo_doi": ZENODO_DOI, "license": DATA_LICENSE},
        "reference_recording": reference_name,
        "holdout_recording": holdout_name,
        "mode_bands_hz": bands,
        "reference_hz": [float(value) for value in reference_median],
        "reference_window_spread": window_spread,
        "median_relative_shift": shifts,
        "median_shift_over_window_sigma": shift_over_sigma,
        "recordings": {
            name: {key: attributes[name].get(key) for key in ("date", "temperature", "train_passing", "wind_range")}
            for name in attributes
        },
        **result,
    }


def markdown_table(result: dict) -> str:
    lines = [
        "| Recording | Description | Windows | Mahalanobis alarmed | PCA residual alarmed | k-NN alarmed "
        "| Median p (Mahalanobis) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    raw, pca, knn = (result["arms"][name] for name in ("raw_frequencies", "pca_residual", "knn"))
    for label in ("holdout", *(f"DS{index}" for index in range(1, 9))):
        description = "the other undamaged recording" if label == "holdout" else DAMAGE_DESCRIPTIONS[label]
        lines.append(
            f"| {label} | {description} | {raw[label]['samples']} "
            f"| {raw[label]['alarmed']} ({100 * raw[label]['alarm_rate']:.0f}%) "
            f"| {pca[label]['alarmed']} ({100 * pca[label]['alarm_rate']:.0f}%) "
            f"| {knn[label]['alarmed']} ({100 * knn[label]['alarm_rate']:.0f}%) "
            f"| {raw[label]['median_p_value']:.3f} |"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--fetch", action="store_true", help="read the needed channels (about 145 MB) first")
    parser.add_argument("--source", default=DATA_URL, help="Zenodo URL (default) or a local data_100Hz.h5")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument(
        "--same-session-reference",
        action="store_true",
        help="post hoc: use the 2020-09-27 undamaged recording as reference and 2020-09-22 as holdout",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    if args.fetch:
        fetch_subset(args.source, args.cache)
    if not args.cache.exists():
        print(f"{args.cache} not found; rerun with --fetch", file=sys.stderr)
        return 2

    pairing = (HOLDOUT, REFERENCE) if args.same_session_reference else (REFERENCE, HOLDOUT)
    result = run_study(args.cache, alpha=args.alpha, reference=pairing[0], holdout=pairing[1])
    print(markdown_table(result))
    summary_keys = ("reference_hz", "reference_window_spread", "reference_samples", "p_value_floor")
    print(json.dumps({key: result[key] for key in summary_keys}, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
