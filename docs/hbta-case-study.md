# Hell Bridge Test Arena case study

`tools/hbta_study.py` asks whether modal novelty detection, fitted on one
undamaged recording, separates eight known imposed damage states of a
full-scale steel bridge from a second undamaged recording. Unlike KW51, the
damage here is real structural damage at known locations, and the
excitation is a controlled shaker rather than traffic and wind.

The answer is no, and the reason is the useful result: between two
undamaged measurement sessions five days apart, several natural frequencies
moved by 1 to 1.7%, as much as most of the damage states and four to six
times the scatter within one recording. A reference taken from a single
session cannot tell damage from a new session.

## Data

- Source: B. T. Svendsen, G. T. Frøseth, O. Øiseth and A. Rønnquist, data
  set of the Hell Bridge Test Arena experimental benchmark study, NTNU.
  Zenodo record 14632942, DOI
  [10.5281/zenodo.14632942](https://doi.org/10.5281/zenodo.14632942),
  CC BY 4.0. Damage states are described in Svendsen et al., "A hybrid
  structural health monitoring approach for damage detection in steel
  bridges under simulated environmental conditions using numerical and
  experimental data", Structural Health Monitoring 22(1), 2023, Table 1.
- The bridge: a riveted steel truss railway bridge, 35 m span, retired
  after 104 years and moved to Hell, near Trondheim, as a test structure.
  Damage was imposed in September 2020 by temporarily removing bolts that
  had replaced rivets, and repaired after each state.
- The record is one 2.5 GB HDF5 file. The study reads only what it uses,
  about 145 MB: the vertical white-noise recordings with the shaker at
  position P2 (`MVS_P2_*_NM_Z_*`, about 950 s each at 100 Hz), and in each
  the shaker acceleration `AS/z` and the 18 vertical global accelerations
  `AG01/z` to `AG18/z`. `--fetch` reads them from Zenodo with HTTP range
  requests into `data/hbta/hbta_p2_noise_z.npz` (gitignored), with a
  SHA-256 of every array.

| Recording | Date | Temperature (C) | State |
| --- | --- | --- | --- |
| `UDS_NM_Z_02` | 2020-09-22 | 10 | undamaged, reference |
| `UDS_NM_Z_01` | 2020-09-27 | 13 | undamaged, holdout |
| `DS1` | 2020-09-27 | 14 | local: one stringer-to-floor-beam connection |
| `DS2` | 2020-09-28 | 13 | local: several stringer-to-floor-beam connections |
| `DS3` | 2020-09-28 | 21 | local: main part of one stringer cross beam removed |
| `DS4` | 2020-09-28 | 22 | local: main parts of several stringer cross beams removed |
| `DS5` | 2020-09-27 | 13 | global: one lateral bracing connection |
| `DS6` | 2020-09-27 | 13 | global: one lateral bracing connection |
| `DS7` | 2020-09-27 | 13 | global: several lateral bracing connections |
| `DS8` | 2020-09-27 | 11 | global: one floor-beam to main load-carrying member connection |

## Protocol

- Each recording is cut into non-overlapping 60 s windows after a 5 s
  trim. Windows in which the shaker's input RMS is below 90% of the
  recording's median are dropped: the shaker takes one to two minutes to
  reach its level. That leaves 13 windows per recording.
- In each window, the H1 frequency response from the shaker acceleration to
  every vertical global accelerometer is estimated with Welch averaging
  (1024-sample segments, 0.098 Hz bins), and |H1|^2 is averaged over the 18
  channels. Using the measured input divides out the shaker's own spectrum.
- Modes are picked once on the full reference recording: the five most
  prominent peaks between 2 and 25 Hz, at 6.75, 9.50, 12.14, 17.07 and
  23.96 Hz. Each band is centred on its peak and spans 40% of the distance
  to the nearest neighbour, at most 0.5 Hz. In every window the band
  maximum is refined by a parabola through the log values.
- Detectors from `cauren_physics.novelty.evaluate_states`, fitted on the
  13 reference windows: Mahalanobis distance (Ledoit-Wolf covariance), PCA
  residual (4 components kept) and the mean distance to the 10 nearest
  reference windows. Calibration scores are leave-one-out reference scores.
  A window alarms when its conformal p-value is at most 0.1. With 13
  reference windows the smallest attainable p is 1/14 = 0.071, so 0.1 is
  the strictest round level the design supports; 0.01 would need 99
  windows and is refused.
- No temperature normalization: each recording lasts 16 minutes at one
  temperature, so the reference cannot show how frequencies depend on it.

The window length, segment length and alpha were set in the code before any
recording was opened. The band rule and the ramp-up rule were changed after
looking at the reference recording only; the second is described below.

## Results

Fraction of 13 windows that alarm, reference 2020-09-22:

| Recording | Mahalanobis | PCA residual | k nearest neighbours |
| --- | --- | --- | --- |
| Holdout (undamaged, 2020-09-27) | 13 (100%) | 0 (0%) | 13 (100%) |
| DS1 | 13 (100%) | 1 (8%) | 13 (100%) |
| DS2 | 13 (100%) | 13 (100%) | 13 (100%) |
| DS3 | 12 (92%) | 0 (0%) | 12 (92%) |
| DS4 | 13 (100%) | 6 (46%) | 13 (100%) |
| DS5 | 12 (92%) | 2 (15%) | 12 (92%) |
| DS6 | 13 (100%) | 4 (31%) | 13 (100%) |
| DS7 | 13 (100%) | 5 (38%) | 13 (100%) |
| DS8 | 13 (100%) | 4 (31%) | 13 (100%) |

Mahalanobis and k nearest neighbours alarm on almost every window of every
damage state, and on every window of the undamaged holdout. They detect
"not the 22nd of September", not damage. PCA is quiet on the holdout but
misses most damage states as well.

### Why: between-session variation

Median frequency change from the reference, in percent and in units of the
reference's window-to-window standard deviation (the identification
uncertainty of one window: 0.51%, 0.29%, 0.29%, 0.26% and 0.13% for the
five modes):

| Recording | 6.75 Hz | 9.50 Hz | 12.14 Hz | 17.07 Hz | 23.96 Hz |
| --- | --- | --- | --- | --- | --- |
| Holdout | +1.32% (+2.6) | -0.42% (-1.5) | +1.67% (+5.7) | +1.07% (+4.2) | +0.70% (+5.2) |
| DS1 | -0.28% (-0.5) | -0.19% (-0.7) | +0.61% (+2.1) | -0.58% (-2.2) | +1.79% (+13.3) |
| DS2 | -0.22% (-0.4) | -0.15% (-0.5) | +0.67% (+2.3) | +2.97% (+11.6) | +1.20% (+8.9) |
| DS3 | +0.62% (+1.2) | -0.35% (-1.2) | +1.40% (+4.8) | +0.86% (+3.3) | +0.12% (+0.9) |
| DS4 | +4.54% (+9.0) | -0.33% (-1.2) | +1.34% (+4.5) | +1.02% (+4.0) | +1.45% (+10.8) |
| DS5 | +0.71% (+1.4) | -0.44% (-1.5) | +0.74% (+2.5) | +0.69% (+2.7) | +1.19% (+8.8) |
| DS6 | +0.01% (+0.0) | -0.41% (-1.4) | +0.95% (+3.2) | +0.36% (+1.4) | +1.66% (+12.3) |
| DS7 | -0.05% (-0.1) | -0.37% (-1.3) | +0.52% (+1.8) | +0.06% (+0.2) | +1.56% (+11.6) |
| DS8 | +1.40% (+2.8) | -0.15% (-0.5) | -0.84% (-2.9) | -2.02% (-7.9) | +1.48% (+11.0) |

The undamaged holdout differs from the reference by 2.6 to 5.7 window
sigmas in four of five modes, more than DS3, DS5, DS6 or DS7 do in most
modes. Several things changed between the sessions: the deck temperature
rose by 3 C, which for steel alone would lower frequencies by about 0.05%
rather than raise them by 1%; the rivets in the damage locations were
replaced by bolts at some point during the campaign; and those bolts were
removed and refitted for every damage state. The data do
not say which of these dominates, but none of them is damage, and all of
them are invisible within one 16-minute recording.

### Post hoc: reference in the same session

Most damage states were recorded on 27 and 28 September, the same session as
the second undamaged recording. With that recording as reference
(`--same-session-reference`), the mode picking finds a different set of
peaks (6.84, 7.46, 9.46 and 24.13 Hz), and:

| Recording | Mahalanobis | PCA residual | k nearest neighbours |
| --- | --- | --- | --- |
| Holdout (undamaged, 2020-09-22) | 13 (100%) | 12 (92%) | 13 (100%) |
| DS1 | 13 (100%) | 13 (100%) | 13 (100%) |
| DS2 | 10 (77%) | 12 (92%) | 12 (92%) |
| DS3 | 9 (69%) | 7 (54%) | 11 (85%) |
| DS4 | 13 (100%) | 2 (15%) | 13 (100%) |
| DS5 | 3 (23%) | 4 (31%) | 3 (23%) |
| DS6 | 12 (92%) | 12 (92%) | 13 (100%) |
| DS7 | 10 (77%) | 12 (92%) | 11 (85%) |
| DS8 | 13 (100%) | 13 (100%) | 13 (100%) |

Within one session, seven of the eight damage states alarm on most windows
with Mahalanobis or k nearest neighbours; DS5, one lateral bracing
connection, mostly does not. But the other undamaged session again alarms
on every window, so these detections come without any healthy data from
the same session to check false alarms against, and they cannot be read as
detection rates. This analysis was chosen after seeing the first one.

### First run

The study was first run with only the 5 s edge trim. The first window of
every recording, where the shaker input was at 13% of its level, then gave
a leave-one-out Mahalanobis score of 16.4 against 0.7 to 2.2 for the
others. That single window set the alarm limit, and Mahalanobis alarmed on
0 of 15 holdout windows and on at most 1 of 15 windows of any damage state
(k nearest neighbours: at most 3), while PCA alarmed on 3 holdout windows
and on 87 to 100% of windows of seven damage states. The ramp-up rule, which
looks only at the input signal and treats every recording alike, was added
after this run.

## Limitations

- Two undamaged sessions are too few to learn how much healthy frequencies
  vary between sessions. A false-alarm-controlled detector for this bridge
  needs many undamaged sessions across temperatures, as KW51's 130
  reference days provide.
- One recording per damage state, all within two days, with temperatures
  from 11 to 22 C. Temperature and damage are partly confounded (DS3 and
  DS4 were recorded at 21 and 22 C).
- The 13 windows of one recording are serially correlated and share one
  temperature, so even within a session the conformal guarantee is
  approximate.
- Only the five clearest resonances of the averaged FRF are tracked. Local
  damage to stringers and cross beams mainly affects local deck modes,
  which the 40 local accelerometers (not used here) would resolve better.
- Mode tracking between sessions by fixed frequency bands can swap modes
  when two resonances are close, as the different peak sets of the two
  undamaged recordings show.

## Reproduce

```bash
pip install -e '.[study]'
python3 tools/hbta_study.py --fetch          # about 145 MB from Zenodo
python3 tools/hbta_study.py
python3 tools/hbta_study.py --same-session-reference
```

`--fetch --source data_100Hz.h5` builds the cache from a downloaded copy of
the full file, after checking its MD5. `--json-out <file>` writes the full
result, including per-state shifts. `tests/test_hbta_study.py` checks the
feature extraction on a simulated two-mode structure and needs no download.
