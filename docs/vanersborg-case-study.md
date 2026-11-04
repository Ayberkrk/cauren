# Vänersborg bridge case study

`tools/vanersborg_study.py` asks whether novelty detection, fitted on the
first two weeks of a bridge's openings, flags the openings after a real
fracture without flagging the healthy openings in between. It is the only
open data set found with real, verified damage on a bridge in service and
weather records, and the only one of the three studies here whose change is
damage.

The pre-registered analysis missed the fracture. After one strain gauge
with erratic readings was excluded, by a rule that looks only at the
reference openings, the Mahalanobis distance flagged all six openings after
the fracture and one of 17 healthy holdout openings. That rule was added
after seeing the miss, so the second result is post hoc.

## Data

- Source: the Vänersborg bridge data set, Zenodo record 8300495, DOI
  [10.5281/zenodo.8300495](https://doi.org/10.5281/zenodo.8300495),
  CC BY 4.0, described in a 2023 Data in Brief article (PMC10641142).
- The bridge: a single-leaf bascule (Strauss heel-trunnion) railway bridge
  over a canal in south-west Sweden, built 1914 to 1916 of riveted steel
  trusses with a reinforced concrete counterweight.
- The record: 64 bridge openings from 2023-02-20 to 2023-03-11, each a CSV
  at 200 Hz with 16 strain gauges (SG1 to SG16, µm/m), five
  accelerometers, an inclinometer for the leaf angle and a weather station.
  SG12 reads -1000000 throughout and is excluded as dead.
- The damage: during the opening that starts at 2023-03-09 23:45 a
  counterweight truss member cracked in its bottom flange and part of its
  web, close to SG9 and SG10; an emergency inspection confirmed it.
- `--fetch` downloads the 550 MB zip, checks its published MD5, keeps only
  the features of each opening in `data/vanersborg/features.json`
  (gitignored, 69 kB) and deletes the zip.

## Protocol

- **Features.** As the leaf rotates, the dead load in the counterweight
  truss redistributes. For each gauge, the strain change from the closed
  position (median of the first 10 s) is taken at leaf angles of -30° and
  -60° while the leaf opens (median within ±1°). Taking the difference
  removes most of the thermal offset of each gauge. That gives 30 features
  per opening (15 live gauges at two angles). Every opening reached both
  angles.
- **Segments**, fixed before any feature was computed:

  | Segment | Openings | Air temperature (C) |
  | --- | --- | --- |
  | Reference, 2023-02-20 to 2023-03-02 | 40 | -2.1 to 8.2 |
  | Holdout, 2023-03-03 to before the fracture | 17 | -8.0 to 6.9 |
  | Fracture opening, 2023-03-09 23:45 | 1 | -6.9 |
  | After the fracture, 2023-03-10 to 2023-03-11 | 6 | -2.7 to 0.0 |

- **Detectors** from `cauren_physics.novelty.evaluate_states`, fitted on the
  40 reference openings: Mahalanobis distance (Ledoit-Wolf covariance), PCA
  residual and mean distance to the 10 nearest reference openings, each
  calibrated with leave-one-out reference scores. An opening alarms when
  its conformal p-value is at most 0.05; with 40 reference openings the
  smallest attainable p is 1/41 = 0.024.

## Results

### Pre-registered

| Openings | Mahalanobis | PCA residual | k nearest neighbours |
| --- | --- | --- | --- |
| Holdout (17, healthy) | 0 | 5 (29%) | 1 (6%) |
| Fracture opening (1) | 0 | 1 | 0 |
| After the fracture (6) | 0 | 6 (100%) | 0 |

Mahalanobis and k nearest neighbours miss every opening after the fracture.
PCA flags all of them, and also five of the 17 healthy holdout openings,
far above the 5% the rule promises; the holdout includes nights down to
-8 C, colder than any reference opening.

### The signal is there

Median shift of each feature from the reference, in reference standard
deviations, for the six openings after the fracture: SG10 at -60° moves by
-5.4 and SG9 at -60° by +4.0, the two gauges next to the crack. No other
feature moves by more than 1.9; in the healthy holdout no feature moves by
more than 0.6. At -60° every opening after the fracture has SG10 between
229.7 and 232.6 µm/m, below the lowest of the 40 reference openings
(238.9), a drop of about 6%. The features see the crack, in the right
place.

### Why the detectors missed it

SG11's features are heavy-tailed in the reference: their standard deviation
is 30 to 53 times their robust (MAD) scale, against at most 5.7 for every
other gauge, and SG11 sets the largest robust z-score of almost every
reference opening (up to 263). Its outliers inflate the leave-one-out
calibration scores, so the limit lies far above what a two-gauge change in
30 features can reach: the median Mahalanobis score after the fracture is
0.18 of its limit.

### Post hoc: excluding the erratic gauge

`--exclude-erratic` drops any gauge whose reference features have a
standard deviation above 10 robust scales. The rule uses the reference
openings only and here removes SG11 alone. The threshold of 10 was chosen
after seeing the ratios, which separate SG11 (30 to 53) from all others
(below 6).

| Openings | Mahalanobis | PCA residual | k nearest neighbours |
| --- | --- | --- | --- |
| Holdout (17, healthy) | 1 (6%) | 6 (35%) | 1 (6%) |
| Fracture opening (1) | 0 | 1 | 0 |
| After the fracture (6) | 6 (100%, p = 1/41) | 6 (100%) | 0 |

The Mahalanobis distance now flags every opening after the fracture at the
smallest attainable p-value, with the median score at 1.98 times its limit,
and one healthy holdout opening (6%, close to the 5% target). The fracture
opening itself stays quiet: the crack formed during it, so most of its
strains were measured on the intact truss. k nearest neighbours still
misses: averaging distances over 28 features dilutes a change in two.

A largest-robust-z-score detector (`--max-deviation`), also tried post hoc
because a crack is local, missed the fracture with and without SG11 (median
p 0.073 after exclusion).

## Limitations

- The successful result depends on a rule added after the failure. It is
  reference-only and would be a reasonable data-quality gate in any
  monitoring system, but it was not pre-registered.
- Six openings after the fracture, over two days, and 17 holdout openings.
  One false alarm in 17 is consistent with a 5% rate and with rates several
  times higher.
- Air temperature is recorded but not modelled; the holdout's coldest
  nights lie outside the reference range, which is what PCA reacts to.
- The features use two leaf angles of the opening only. Closing, the
  accelerometers, and the dynamic response of the leaf are unused.
- Thirty features for 40 reference openings is a hard regime for any
  covariance; a detector that is told where the damage is would do better,
  but such a detector cannot be fitted without damage data.

## Reproduce

```bash
pip install -e '.[study]'
python3 tools/vanersborg_study.py --fetch          # 550 MB download, deleted after
python3 tools/vanersborg_study.py
python3 tools/vanersborg_study.py --exclude-erratic
python3 tools/vanersborg_study.py --exclude-erratic --max-deviation
```

`tests/test_vanersborg_study.py` checks the feature extraction on a
simulated opening and needs no download.
