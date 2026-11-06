"""Synthetic checks for the Z24 reader and mode tracking.

The real archive is available only on request from KU Leuven and is never
bundled, so these tests build small zips and spectra with known answers.
"""

from __future__ import annotations

import datetime as dt
import io
import zipfile
import zlib

import pytest

np = pytest.importorskip("numpy", reason="tools/z24_study.py is numpy based")

from tools import z24_study as study

# Made-up labels and conversions: the real ones are in the archive's
# confidential documentation and are not reproduced here.
ORIGIN = dt.datetime(2000, 1, 1)
CONVERSIONS = {"VX": (4.0, 0.0), "VY": (50.0, -10.0)}


def _stored_zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    return buffer.getvalue()


def test_central_directory_detects_bytes_lost_inside_a_member(tmp_path):
    members = {"a.bin": b"A" * 500, "b.bin": b"B" * 500, "c.bin": b"C" * 500}
    raw = _stored_zip(members)
    cut = raw.find(b"B" * 500) + 200
    damaged = raw[:cut] + raw[cut + 37 :]
    path = tmp_path / "damaged.zip"
    path.write_bytes(damaged)

    with path.open("rb") as handle:
        entries, shift = study._central_directory(handle, 0, len(damaged))
        assert shift == 37
        recovered = {}
        for entry in entries:
            start = study._data_start(handle, 0, entry, shift)
            handle.seek(start)
            data = handle.read(entry.compressed)
            recovered[entry.name] = zlib.crc32(data) == entry.crc
    assert recovered == {"a.bin": True, "b.bin": False, "c.bin": True}


def test_record_time_follows_the_week_day_hour_names():
    assert study.record_time("01A00.zip", ORIGIN) == ORIGIN
    assert study.record_time("01C14.zip", ORIGIN) == dt.datetime(2000, 1, 3, 14)
    assert study.record_time("12E00", ORIGIN) == dt.datetime(2000, 3, 22, 0)


def test_read_environment_converts_voltages_and_ignores_the_footer():
    text = "\n".join(
        [
            "VX    [V]      VY    [V]      T1    [°C]     T2    [°C]",
            "1.0e+00   5.0e-01   7.0e+00   3.276700e+03",
            "1.2e+00   5.2e-01   7.2e+00   3.276700e+03",
            "",
            "EnvScan started : Mon Nov 10 14:00:13 1997",
            "End of File",
        ]
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("01C14PRE.env", text.encode("latin-1"))
    readings = study.read_environment(zipfile.ZipFile(buffer), "01C14", CONVERSIONS)
    assert readings["VX"] == pytest.approx(4.4)
    assert readings["VY"] == pytest.approx(15.5)
    assert study._plausible_median(readings, ("T1", "T2")) == pytest.approx(7.1)


def test_fdd_and_mode_tracking_follow_a_mode_whose_frequency_moves():
    rng = np.random.default_rng(0)
    rate = study.SAMPLE_RATE_HZ
    time = np.arange(16 * study.SEGMENT_SAMPLES) / rate
    # A complex shape, as a lightly damped mode with non-proportional damping has.
    shape_a = np.array([1.0, 0.8, 0.5, 0.2, -0.2, -0.5, -0.8, -1.0]) * np.exp(1j * np.linspace(0.0, 0.6, 8))
    shape_b = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

    def hour(f_a: float, f_b: float) -> np.ndarray:
        phase = 2 * np.pi * f_a * time + rng.uniform(0, 6.3)
        a = np.abs(shape_a) * np.sin(phase[:, None] + np.angle(shape_a))
        b = np.outer(np.sin(2 * np.pi * f_b * time + rng.uniform(0, 6.3)), shape_b)
        return a + b + rng.normal(0.0, 0.05, (len(time), 8))

    spectra = [study.fdd_spectrum(hour(f_a, 9.0)) for f_a in (4.0, 4.0, 4.0, 4.4)]
    frequencies = spectra[0][0]
    singular_values = np.array([s for _, s, _ in spectra])
    vectors = np.array([v for _, _, v in spectra])
    modes = study.reference_modes(frequencies, singular_values, vectors, np.array([True, True, True, False]))
    assert [round(f, 1) for f, _ in modes] == [4.0, 9.0]
    assert study.mac(modes[0][1], shape_a) > 0.99

    tracked, similarity = study.track_modes(frequencies, singular_values, vectors, modes)
    np.testing.assert_allclose(tracked[:, 0], [4.0, 4.0, 4.0, 4.4], atol=0.03)
    np.testing.assert_allclose(tracked[:, 1], 9.0, atol=0.03)
    assert np.all(similarity > 0.99)


def test_stable_channels_drops_a_channel_that_changes_regime_within_the_reference():
    origin = study.PROTOCOL.reference[0]
    names = ["01A12", "06A12", "12A12", "20A12"]  # four different reference months
    rms = np.array([[0.5, 13.0], [0.6, 0.0], [0.5, 0.0], [0.7, 0.0]])
    assert study.stable_channels(names, rms, origin) == [0]


def test_load_config_reads_the_fields_and_points_to_the_template(tmp_path):
    path = tmp_path / "z24_config.json"
    with pytest.raises(FileNotFoundError, match="z24_config.template.json"):
        study.load_config(path)
    path.write_text(
        '{"channels": ["a", "b"], "name_origin": "2000-01-01T00:00:00", "deck_temperatures": ["T1"], '
        '"pavement_temperatures": ["T2"], "air_temperature": "VY", "humidity": "VZ", "wind_speed": "VX", '
        '"voltage_conversions": {"VX": [4.0, 0.0]}}',
        encoding="utf-8",
    )
    config = study.load_config(path)
    assert config.channels == ("a", "b")
    assert config.name_origin == ORIGIN
    assert config.voltage_conversions == {"VX": (4.0, 0.0)}


def test_config_template_has_every_field_and_no_values():
    import json

    template = json.loads(study.CONFIG_TEMPLATE.read_text(encoding="utf-8"))
    assert set(study.Z24Config.__dataclass_fields__) <= set(template)

    def leaves(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from leaves(item)
        elif isinstance(value, list):
            for item in value:
                yield from leaves(item)
        else:
            yield value

    for leaf in leaves(template):
        assert isinstance(leaf, str)
        with pytest.raises(ValueError):
            float(leaf)
        with pytest.raises(ValueError):
            dt.datetime.fromisoformat(leaf)
