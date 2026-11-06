"""Z24 bridge study: a year of healthy monitoring, then progressive real damage.

Uses the Z24 benchmark of KU Leuven: a post-tensioned concrete box-girder road
bridge between Koppigen and Utzenstorf, Switzerland, monitored hourly from
November 1997 and progressively damaged in August and September 1998 before
its demolition (Maeck and De Roeck, Mechanical Systems and Signal Processing
17(1), 2003; Peeters and De Roeck, Earthquake Engineering and Structural
Dynamics 30(2), 2001). The data is available for non-commercial research on
request from the KU Leuven Structural Mechanics Section and may not be passed
on; it is never committed here.

Every hour the environmental monitoring system recorded 65536 samples at 100 Hz
from eight accelerometers and, before and after them, about fifty temperature,
weather and soil sensors. This study identifies the natural frequencies of each
hour with Frequency Domain Decomposition of the eight channels, tracks each mode
by the similarity of its shape (MAC) to the reference shape rather than by a
fixed frequency band, because frozen asphalt raises Z24's frequencies by more
than its damage lowers them, and scores the hours with the shared protocol in
``cauren_physics.novelty``.

Segments, fixed before any frequency was identified:

- reference: 1997-11-10 to 1998-06-30, including the winter freezing periods;
- holdout: 1998-07-01 to 1998-08-03, healthy summer days;
- excluded: 1998-08-04 to 1998-08-11, two reference measurements with test
  equipment on the bridge and the first works;
- changed: from 1998-08-12, after the first pier settlement.

The archive as distributed can be damaged in transfer. The reader here does not
trust the zip's own offsets: it locates every member by its local header,
checks every CRC and skips, and reports, any hour whose data is damaged.

The archive's own description of channels, sensor labels and voltage
conversions is confidential and is not reproduced in this repository. The
study reads them from ``data/z24/z24_config.json`` (gitignored), which a
user with access fills in from the documentation in their own copy;
``tools/z24_config.template.json`` lists the fields.

Usage:
    python3 tools/z24_study.py --extract     # once, about twelve minutes
    python3 tools/z24_study.py --json-out runtime_outputs/z24_study.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import struct
import sys
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cauren_physics.novelty import StudyProtocol, markdown_table, run_protocol

DEFAULT_ARCHIVE = PROJECT_ROOT / "data" / "z24" / "data-z24.zip"
DEFAULT_FDD = PROJECT_ROOT / "data" / "z24" / "fdd.npz"
DEFAULT_CONFIG = PROJECT_ROOT / "data" / "z24" / "z24_config.json"
CONFIG_TEMPLATE = PROJECT_ROOT / "tools" / "z24_config.template.json"
MONITORING_MEMBERS = ("Z24ems1.zip", "Z24ems2.zip", "Z24ems3.zip")

SAMPLE_RATE_HZ = 100.0

# A channel whose median RMS differs by more than this factor between two
# reference months changed its gain or failed; one channel is silent from
# December to April and twenty times louder than the others in November and
# from May. The rule looks at the reference months only.
CHANNEL_RMS_RATIO = 10.0

SEGMENT_SAMPLES = 2048
BAND_HZ = (2.0, 14.0)
MAX_MODES = 5
MIN_PEAK_PROMINENCE = 0.5  # natural-log units of the first singular value
UNFROZEN_C = 5.0
# The reference hours' best-candidate MACs fall into two groups, noise peaks
# below 0.5 and the mode itself above 0.7; 0.6 lies in the valley between
# them. It was lowered from 0.8 after looking at that reference distribution,
# before any holdout or damaged hour was scored.
MAC_THRESHOLD = 0.6
SEARCH_SHARE = 0.2

PROTOCOL = StudyProtocol(
    reference=(dt.datetime(1997, 11, 10), dt.datetime(1998, 7, 1)),
    holdout=(dt.datetime(1998, 7, 1), dt.datetime(1998, 8, 4)),
    changed=(dt.datetime(1998, 8, 12), dt.datetime(1998, 9, 11)),
)
DAMAGE_START = dt.datetime(1998, 8, 12)

# Temperature readings outside this range are sensor failures.
PLAUSIBLE_C = (-40.0, 60.0)

# There is no single material-only temperature sensitivity for a concrete deck
# under asphalt, so the slopes are reported without that comparison.
EXPECTED_DF_OVER_F_PER_K = None


@dataclass(frozen=True)
class Z24Config:
    """What the archive's confidential documentation says about its records.

    ``channels`` are the accelerometer suffixes of the hourly files,
    ``name_origin`` the date of week 01, day A in the record names (``WWDHH``),
    the label tuples name environmental columns, and
    ``voltage_conversions`` maps a label to the ``(scale, offset)`` that turns
    its recorded voltage into physical units.
    """

    channels: tuple[str, ...]
    name_origin: dt.datetime
    deck_temperatures: tuple[str, ...]
    pavement_temperatures: tuple[str, ...]
    air_temperature: str
    humidity: str
    wind_speed: str
    voltage_conversions: dict[str, tuple[float, float]]


def load_config(path: Path = DEFAULT_CONFIG) -> Z24Config:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found: copy {CONFIG_TEMPLATE.name} there and fill it in from the "
            "documentation of your own copy of the Z24 archive"
        )
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return Z24Config(
        channels=tuple(raw["channels"]),
        name_origin=dt.datetime.fromisoformat(raw["name_origin"]),
        deck_temperatures=tuple(raw["deck_temperatures"]),
        pavement_temperatures=tuple(raw["pavement_temperatures"]),
        air_temperature=raw["air_temperature"],
        humidity=raw["humidity"],
        wind_speed=raw["wind_speed"],
        voltage_conversions={label: tuple(pair) for label, pair in raw["voltage_conversions"].items()},
    )


@dataclass
class _Entry:
    name: str
    method: int
    crc: int
    compressed: int
    size: int
    offset: int


def _central_directory(handle, start: int, end: int) -> tuple[list[_Entry], int]:
    """Entries of the zip stored in ``[start, end)`` and how far its offsets overstate positions.

    The end-of-central-directory records give the directory's size and its
    declared offset; the directory really ends where the record begins. If
    bytes were lost earlier in the file, every offset after the loss is too
    large by the same amount, which is returned as the shift.
    """
    handle.seek(max(start, end - 65558))
    tail = handle.read(end - max(start, end - 65558))
    base = end - len(tail)
    record = tail.rfind(b"PK\x05\x06")
    if record < 0:
        raise ValueError("no end of central directory record")
    count, size, offset = struct.unpack("<HII", tail[record + 10 : record + 20])
    directory_end = base + record
    zip64 = tail.rfind(b"PK\x06\x06", 0, record)
    if zip64 >= 0 and (offset == 0xFFFFFFFF or count == 0xFFFF or size == 0xFFFFFFFF):
        values = struct.unpack("<4sQHHIIQQQQ", tail[zip64 : zip64 + 56])
        count, size, offset = values[7], values[8], values[9]
        directory_end = base + zip64
    actual = directory_end - size
    handle.seek(actual)
    data = handle.read(size)
    entries = []
    index = 0
    while data[index : index + 4] == b"PK\x01\x02":
        fields = struct.unpack("<4sHHHHHHIIIHHHHHII", data[index : index + 46])
        name_length, extra_length, comment_length = fields[10], fields[11], fields[12]
        name = data[index + 46 : index + 46 + name_length].decode("latin-1")
        extra = data[index + 46 + name_length : index + 46 + name_length + extra_length]
        compressed, size_value, local = fields[8], fields[9], fields[16]
        cursor = 0
        while cursor + 4 <= len(extra):
            header, length = struct.unpack("<HH", extra[cursor : cursor + 4])
            if header == 1:
                values = list(struct.unpack("<" + "Q" * (length // 8), extra[cursor + 4 : cursor + 4 + length]))
                if size_value == 0xFFFFFFFF:
                    size_value = values.pop(0)
                if compressed == 0xFFFFFFFF:
                    compressed = values.pop(0)
                if local == 0xFFFFFFFF:
                    local = values.pop(0)
            cursor += 4 + length
        entries.append(_Entry(name, fields[4], fields[7], compressed, size_value, local))
        index += 46 + name_length + extra_length + comment_length
    return entries, offset - (actual - start)


def _data_start(handle, start: int, entry: _Entry, shift: int) -> int:
    """Position of an entry's data, trusting its local header over the stored offset."""
    for position in (start + entry.offset, start + entry.offset - shift):
        handle.seek(position)
        header = handle.read(30)
        if header[:4] == b"PK\x03\x04":
            name_length, extra_length = struct.unpack("<HH", header[26:30])
            return position + 30 + name_length + extra_length
    raise ValueError(f"no local header for {entry.name}")


class Z24Archive:
    """Hourly monitoring records of the Z24 archive, read without extracting it."""

    def __init__(self, path: Path = DEFAULT_ARCHIVE):
        self.handle = open(path, "rb")
        size = Path(path).stat().st_size
        self.outer, self.outer_shift = _central_directory(self.handle, 0, size)

    def close(self) -> None:
        self.handle.close()

    def records(self):
        """Yield ``(name, zipfile or None)`` for every hourly record; ``None`` when its CRC fails."""
        for member in MONITORING_MEMBERS:
            entry = next(item for item in self.outer if item.name == member)
            start = _data_start(self.handle, 0, entry, self.outer_shift)
            end = start + entry.compressed
            try:
                inner, shift = _central_directory(self.handle, start, end)
            except ValueError:
                inner, shift = _central_directory(self.handle, start, end - self.outer_shift)
            for record in inner:
                position = _data_start(self.handle, start, record, shift)
                self.handle.seek(position)
                raw = self.handle.read(record.compressed)
                try:
                    data = zlib.decompress(raw, -15) if record.method == 8 else raw
                except zlib.error:
                    data = b""
                if zlib.crc32(data) != record.crc:
                    yield record.name, None
                    continue
                yield record.name, zipfile.ZipFile(io.BytesIO(data))


def record_time(name: str, origin: dt.datetime) -> dt.datetime:
    """Start of the hour a record name ``WWDHH`` (week, weekday letter, hour) stands for."""
    return origin + dt.timedelta(days=7 * (int(name[:2]) - 1) + ord(name[2].upper()) - ord("A"), hours=int(name[3:5]))


def read_acceleration(bundle: zipfile.ZipFile, prefix: str, channels: tuple[str, ...]) -> np.ndarray | None:
    """Samples of every channel in ``channels`` as columns, or ``None`` when one is missing."""
    names = {name.lower(): name for name in bundle.namelist()}
    columns = []
    for channel in channels:
        name = names.get(f"{prefix}{channel}.aaa".lower())
        if name is None:
            return None
        lines = bundle.read(name).decode("latin-1").split("\n")
        count = int(lines[1])
        columns.append(np.asarray(lines[3 : 3 + count], dtype=float))
    length = min(len(column) for column in columns)
    return np.column_stack([column[:length] for column in columns])


def read_environment(
    bundle: zipfile.ZipFile, prefix: str, conversions: dict[str, tuple[float, float]]
) -> dict[str, float]:
    """Mean of the ten scans of each environmental channel taken before the acceleration record."""
    names = {name.lower(): name for name in bundle.namelist()}
    name = names.get(f"{prefix}pre.env".lower())
    if name is None:
        return {}
    lines = [line for line in bundle.read(name).decode("latin-1").splitlines() if line.strip()]
    header = lines[0].split()
    labels = [token for token in header if not token.startswith("[")]
    scans = []
    # The scans are followed by free-text lines (scan start and stop time, status).
    for line in lines[1:]:
        try:
            scans.append([float(token) for token in line.split()])
        except ValueError:
            break
    if not scans:
        return {}
    values = np.array([row[: min(map(len, scans))] for row in scans])
    readings: dict[str, float] = {}
    for column, label in enumerate(labels):
        if label in readings or column >= values.shape[1]:
            continue
        scale, offset = conversions.get(label, (1.0, 0.0))
        readings[label] = float(np.mean(values[:, column])) * scale + offset
    return readings


def _plausible_median(readings: dict[str, float], labels) -> float:
    values = [readings[label] for label in labels if label in readings and PLAUSIBLE_C[0] < readings[label] < PLAUSIBLE_C[1]]
    return float(np.median(values)) if values else float("nan")


def fdd_spectrum(samples: np.ndarray, sample_rate_hz: float = SAMPLE_RATE_HZ):
    """First singular value and vector of the Welch cross-spectral matrix inside ``BAND_HZ``."""
    step = SEGMENT_SAMPLES // 2
    window = np.hanning(SEGMENT_SAMPLES)
    centred = samples - samples.mean(axis=0)
    starts = range(0, len(centred) - SEGMENT_SAMPLES + 1, step)
    frequencies = np.fft.rfftfreq(SEGMENT_SAMPLES, 1.0 / sample_rate_hz)
    inside = (frequencies >= BAND_HZ[0]) & (frequencies <= BAND_HZ[1])
    spectra = np.array([np.fft.rfft(centred[s : s + SEGMENT_SAMPLES] * window[:, None], axis=0)[inside] for s in starts])
    cross = np.einsum("sfi,sfj->fij", spectra, spectra.conj()) / len(spectra)
    left, singular, _ = np.linalg.svd(cross)
    return frequencies[inside], singular[:, 0], left[:, :, 0]


def channel_rms(config: Z24Config, archive_path: Path = DEFAULT_ARCHIVE) -> tuple[list[str], np.ndarray]:
    """RMS of every channel in every readable hour (first pass of the extraction)."""
    archive = Z24Archive(archive_path)
    names, rms = [], []
    try:
        for name, bundle in archive.records():
            if bundle is None:
                continue
            samples = read_acceleration(bundle, Path(name).stem, config.channels)
            if samples is not None:
                names.append(Path(name).stem)
                rms.append(samples.std(axis=0))
    finally:
        archive.close()
    return names, np.array(rms)


def stable_channels(names: list[str], rms: np.ndarray, origin: dt.datetime) -> list[int]:
    """Columns whose monthly median RMS stays within ``CHANNEL_RMS_RATIO`` over the reference months."""
    months = np.array([record_time(name, origin).strftime("%Y-%m") for name in names])
    stamps = np.array([record_time(name, origin) for name in names], dtype="datetime64[s]")
    in_reference = (stamps >= np.datetime64(PROTOCOL.reference[0])) & (stamps < np.datetime64(PROTOCOL.reference[1]))
    medians = np.array([np.median(rms[in_reference & (months == month)], axis=0) for month in sorted(set(months[in_reference]))])
    ratio = medians.max(axis=0) / np.maximum(medians.min(axis=0), 1e-12)
    return [column for column, value in enumerate(ratio) if value <= CHANNEL_RMS_RATIO]


def extract(config: Z24Config, archive_path: Path = DEFAULT_ARCHIVE, target: Path = DEFAULT_FDD) -> Path:
    """Identify every hour once and keep only the FDD results and the environment.

    A first pass reads every channel's RMS and drops the channels that are
    not stable over the reference months; the second pass runs FDD on the
    rest.
    """
    rms_names, rms = channel_rms(config, archive_path)
    kept = stable_channels(rms_names, rms, config.name_origin)
    print("kept channels", [config.channels[column] for column in kept], flush=True)
    archive = Z24Archive(archive_path)
    names, damaged, incomplete = [], [], []
    singular_values, vectors, deck, pavement, air, humidity, wind = [], [], [], [], [], [], []
    frequencies = None
    try:
        for count, (name, bundle) in enumerate(archive.records()):
            if bundle is None:
                damaged.append(name)
                continue
            prefix = Path(name).stem
            samples = read_acceleration(bundle, prefix, config.channels)
            if samples is None or len(samples) < 8 * SEGMENT_SAMPLES:
                incomplete.append(name)
                continue
            frequencies, singular, vector = fdd_spectrum(samples[:, kept])
            readings = read_environment(bundle, prefix, config.voltage_conversions)
            names.append(prefix)
            singular_values.append(singular.astype(np.float32))
            vectors.append(vector.astype(np.complex64))
            deck.append(_plausible_median(readings, config.deck_temperatures))
            pavement.append(_plausible_median(readings, config.pavement_temperatures))
            air.append(_plausible_median(readings, (config.air_temperature,)))
            humidity.append(readings.get(config.humidity, float("nan")))
            wind.append(readings.get(config.wind_speed, float("nan")))
            if count % 250 == 0:
                print(f"{count} records, last {prefix}", flush=True)
    finally:
        archive.close()
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        names=np.array(names),
        frequencies_hz=frequencies,
        singular_values=np.array(singular_values),
        vectors=np.array(vectors),
        deck_temperature_c=np.array(deck),
        pavement_temperature_c=np.array(pavement),
        air_temperature_c=np.array(air),
        humidity_pct=np.array(humidity),
        wind_speed_ms=np.array(wind),
        damaged=np.array(damaged),
        incomplete=np.array(incomplete),
        channels=np.array([config.channels[column] for column in kept]),
        rms_names=np.array(rms_names),
        channel_rms=rms,
    )
    print(f"{len(names)} hours identified, {len(damaged)} damaged, {len(incomplete)} incomplete")
    return target


def mac(first: np.ndarray, second: np.ndarray) -> float:
    """Modal assurance criterion of two complex shape vectors."""
    return float(abs(np.vdot(first, second)) ** 2 / (np.vdot(first, first).real * np.vdot(second, second).real))


def _peak_offset(values: np.ndarray, index: int) -> float:
    if index == 0 or index == len(values) - 1:
        return 0.0
    left, centre, right = np.log(values[index - 1 : index + 2])
    curvature = left - 2.0 * centre + right
    return 0.0 if curvature >= 0.0 else 0.5 * (left - right) / curvature


def reference_modes(frequencies, singular_values, vectors, usable) -> list[tuple[float, np.ndarray]]:
    """Frequencies and shapes of the clearest peaks of the reference's mean first singular value.

    ``usable`` selects the reference hours above ``UNFROZEN_C``. Each shape is
    the dominant direction of the reference hours' singular vectors at the
    peak bin, which removes their arbitrary complex phase.
    """
    from scipy.signal import find_peaks

    mean_log = np.mean(np.log(singular_values[usable]), axis=0)
    peaks, properties = find_peaks(mean_log, prominence=MIN_PEAK_PROMINENCE)
    strongest = np.sort(peaks[np.argsort(properties["prominences"])[::-1][:MAX_MODES]])
    modes = []
    for index in strongest:
        # Each row of the stack is one hour's vector times an arbitrary complex
        # phase; the first right singular vector is their common direction.
        stack = vectors[usable, index, :]
        _, _, rows = np.linalg.svd(stack, full_matrices=False)
        modes.append((float(frequencies[index]), rows[0]))
    return modes


def track_modes(frequencies, singular_values, vectors, modes) -> tuple[np.ndarray, np.ndarray]:
    """Per hour and mode, the frequency of the strongest peak shaped like the mode, and its MAC.

    Candidates are local maxima of the first singular value within
    ``SEARCH_SHARE`` of the reference frequency whose singular vector has a
    MAC of at least ``MAC_THRESHOLD`` with the reference shape; the strongest
    of them wins. Picking the highest MAC instead let the first mode jump to a
    weaker neighbouring peak of similar shape in 40% of reference hours.
    Without a candidate the hour has no estimate for that mode.
    """
    step = frequencies[1] - frequencies[0]
    hours = len(singular_values)
    tracked = np.full((hours, len(modes)), np.nan)
    similarity = np.full((hours, len(modes)), np.nan)
    interior = (singular_values[:, 1:-1] > singular_values[:, :-2]) & (singular_values[:, 1:-1] >= singular_values[:, 2:])
    for column, (centre, shape) in enumerate(modes):
        window = np.flatnonzero(np.abs(frequencies - centre) <= SEARCH_SHARE * centre)
        window = window[(window > 0) & (window < len(frequencies) - 1)]
        for hour in range(hours):
            candidates = window[interior[hour, window - 1]]
            if len(candidates) == 0:
                continue
            scores = np.array([mac(vectors[hour, index], shape) for index in candidates])
            similar = scores >= MAC_THRESHOLD
            if not similar.any():
                similarity[hour, column] = scores.max()
                continue
            best = np.flatnonzero(similar)[np.argmax(singular_values[hour, candidates[similar]])]
            index = candidates[best]
            similarity[hour, column] = scores[best]
            tracked[hour, column] = frequencies[index] + _peak_offset(singular_values[hour], index) * step
    return tracked, similarity


def run_study(
    config: Z24Config, fdd_path: Path = DEFAULT_FDD, modes: list[int] | None = None, environment: bool = False
) -> dict:
    data = np.load(fdd_path)
    times = [record_time(name, config.name_origin) for name in data["names"]]
    order = np.argsort(np.array(times, dtype="datetime64[s]"))
    times = [times[index] for index in order]
    singular_values = data["singular_values"][order].astype(float)
    vectors = data["vectors"][order]
    deck = data["deck_temperature_c"][order]
    frequencies = data["frequencies_hz"]
    stamps = np.array(times, dtype="datetime64[s]")
    in_reference = (stamps >= np.datetime64(PROTOCOL.reference[0])) & (stamps < np.datetime64(PROTOCOL.reference[1]))
    usable = in_reference & (deck > UNFROZEN_C)
    found = reference_modes(frequencies, singular_values, vectors, usable)
    tracked, similarity = track_modes(frequencies, singular_values, vectors, found)
    weather = (
        np.column_stack([data[key][order] for key in ("air_temperature_c", "humidity_pct", "wind_speed_ms", "pavement_temperature_c")])
        if environment
        else None
    )
    # The modes are already chosen on the reference; an hour without an
    # estimate of every mode is dropped, as in the other studies.
    result = run_protocol(
        times,
        tracked,
        deck,
        PROTOCOL,
        expected_df_over_f=EXPECTED_DF_OVER_F_PER_K,
        modes=list(range(len(found))) if modes is None else modes,
        environment=weather,
    )
    result["reference_modes_hz"] = [frequency for frequency, _ in found]
    result["tracked_share"] = [float(np.mean(np.isfinite(tracked[:, column]))) for column in range(len(found))]
    result["damaged_records"] = [str(name) for name in data["damaged"]]
    result["incomplete_records"] = [str(name) for name in data["incomplete"]]
    result["hours"] = len(times)
    result["channels"] = [str(channel) for channel in data["channels"]]
    first_alarms = {
        arm: summary["changed"]["first_alarm"] for arm, summary in result["arms"].items()
    }
    result["first_alarm_after_damage_start"] = first_alarms
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--fdd", type=Path, default=DEFAULT_FDD)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--extract", action="store_true", help="identify every hour once (about half an hour)")
    parser.add_argument("--modes", type=int, nargs="+", help="0-based reference modes to monitor instead of all")
    parser.add_argument("--environment", action="store_true", help="add the weather and pavement temperature arm")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 2
    if args.extract:
        extract(config, args.archive, args.fdd)
    if not args.fdd.exists():
        print(f"{args.fdd} not found; rerun with --extract", file=sys.stderr)
        return 2
    result = run_study(config, args.fdd, modes=args.modes, environment=args.environment)
    print(markdown_table(result, changed_label="Damaged"))
    summary_keys = ("reference_modes_hz", "tracked_share", "modes", "complete_hours", "temperature_range_c", "frequency_ratio")
    print(json.dumps({key: result[key] for key in summary_keys}, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
