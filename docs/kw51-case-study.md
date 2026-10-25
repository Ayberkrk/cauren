# KW51 railway bridge case study

`tools/kw51_study.py` runs three modal novelty detectors on one real bridge
with a known change of state. It answers a narrow question: on KW51, does a
detector fitted on healthy data stay quiet on later healthy data and alarm
after the retrofit, and does temperature normalization help?

This is a single-bridge case study, not a validation of Cauren's anomaly
detection. One structure, one change, and one season of post-change data
cannot establish false-alarm or detection rates for other bridges.

## Data

- Source: K. Maes and G. Lombaert, "Monitoring railway bridge KW51 before,
  during, and after retrofitting", ASCE Journal of Bridge Engineering 26(3),
  2021. Data on Zenodo, record 3745914, DOI
  [10.5281/zenodo.3745914](https://doi.org/10.5281/zenodo.3745914).
- License: CC BY-NC-SA 4.0 (attribution, non-commercial, share-alike). The
  data is not committed to this repository. `--download` fetches
  `trackedmodes.zip` (about 13 MB, MD5 checked) into `data/kw51/raw/`, which
  is gitignored.
- Content used: hourly identified natural frequencies of 14 tracked modes and
  the deck steel temperature `tBD31A`, 2018-10-01 to 2020-01-15. The
  connections of the diagonals to the deck and arches were strengthened
  between 2019-05-15 and 2019-09-27.

## Protocol

| Segment | Dates | Use |
| --- | --- | --- |
| Reference | 2018-10-01 to 2019-03-31 | fit every model and set every control limit |
| Holdout | 2019-04-01 to 2019-05-14 | healthy, unseen: false alarms |
| Retrofit | 2019-05-15 to 2019-09-27 | excluded |
| Changed | 2019-09-28 to 2020-01-15 | retrofitted structure: detections |

- Modes are chosen from the reference period only: every mode identified in
  at least 90% of reference hours. That gives modes 2, 4, 5, 8 and 12
  (0-based), at about 1.89, 2.58, 2.93, 4.11 and 6.33 Hz. Hours missing any
  of these modes or the deck temperature are dropped (2,892 reference, 969
  holdout and 1,646 changed hours remain).
- Each arm produces an hourly score. The daily score is the median of a day's
  hourly scores, using days with at least 6 scored hours. The control limit
  is the 99th percentile of the reference daily scores, so about 1% of
  reference days (2 of 130) exceed it by construction.
- Detection delay is counted in days from 2019-09-28 to the first alarmed day.

### Arms

1. **Raw frequencies:** Mahalanobis distance of the hourly frequency vector
   from the reference mean, with a Ledoit-Wolf shrinkage covariance.
2. **Temperature normalized:** each mode's frequency is regressed on deck
   temperature with `timoshenko.fit_environmental_baseline`, using the
   bilinear covariates `[T, min(T, 0 C)]` so the slope can change below
   freezing, where frozen ballast and bearings stiffen the deck. The
   Mahalanobis distance (shrinkage covariance) is then computed on the
   residuals.
3. **PCA residual:** frequencies are standardized with reference statistics;
   the leading components that explain 90% of reference variance (4 of 5
   here) are treated as the normal co-variation pattern, and the score is
   the norm of what is left outside them.

Timoshenko is optional. `fit_environmental_baseline` is on Timoshenko's main
branch and not in the released `timoshenko-engine` 2.1.0, so the script falls
back to the same ordinary least-squares fit in numpy when it is missing. The
results below were produced with Timoshenko at commit `514f696`; the numpy
fallback gives identical alarms and coefficients.

## Results

Pre-registered mode set (2, 4, 5, 8, 12):

| Arm | Holdout false-alarm days | Post-retrofit days alarmed | First alarm (delay) | Holdout max / limit | Post-retrofit min / limit |
| --- | --- | --- | --- | --- | --- |
| Raw frequencies | 0/44 (0.0%) | 103/103 (100.0%) | 2019-09-28 (0 d) | 0.60 | 2.48 |
| Temperature normalized | 0/44 (0.0%) | 103/103 (100.0%) | 2019-09-28 (0 d) | 0.58 | 2.64 |
| PCA residual | 0/44 (0.0%) | 103/103 (100.0%) | 2019-09-28 (0 d) | 0.60 | 3.13 |

All three arms meet both targets (holdout false-alarm rate at most 5%,
at least 90% of post-retrofit days alarmed) and alarm on the first scored
day. The last two columns show the margin: the highest holdout daily score
and the lowest post-retrofit daily score, each divided by the arm's control
limit. Values below 1 stay quiet, values above 1 alarm.

### Why the arms look alike here

The retrofit is a large change for this monitoring set. After temperature
normalization, the median frequencies of the changed segment differ from the
reference by -0.33%, -0.28%, +2.11%, -0.73% and +1.64% for the five modes.
The two modes that rise by about 2% alone push every detector far past its
limit.

The comparison also does not stress temperature normalization much. The
changed segment (October to January) has deck temperatures from 0.7 to
22.7 C, inside the reference range of -2.9 to 19.0 C, so the raw arm sees the
same seasonal pattern it was trained on. The warmer holdout (1.9 to 26.3 C)
is the real test of environmental robustness, and all three arms stayed
below 0.6 of their limits there.

### Post-hoc stress test

To see what happens when the change is small, the study was rerun with only
the three modes that move by less than 1% (2, 4, 8). This subset was picked
after looking at the post-retrofit shifts, so it is illustrative, not a
pre-registered result (`--modes 2 4 8`):

| Arm | Holdout false-alarm days | Post-retrofit days alarmed | First alarm (delay) | Holdout max / limit | Post-retrofit min / limit |
| --- | --- | --- | --- | --- | --- |
| Raw frequencies | 0/44 (0.0%) | 107/107 (100.0%) | 2019-09-28 (0 d) | 0.58 | 1.27 |
| Temperature normalized | 0/44 (0.0%) | 107/107 (100.0%) | 2019-09-28 (0 d) | 0.82 | 1.82 |
| PCA residual | 10/44 (22.7%) | 107/107 (100.0%) | 2019-09-28 (0 d) | 1.93 | 3.38 |

With small shifts, temperature normalization widens the post-retrofit margin
(lowest daily score 1.82 times the limit, against 1.27 for raw frequencies),
while the holdout still stays below the limit. PCA keeps 2 of 3 components,
and its single residual direction picks up the warmer spring temperatures,
which lie outside the reference range: 23% of holdout days alarm, above the
5% target.

## Physics checks

### Temperature-stiffness sign

Steel loses roughly 0.03% of its Young's modulus per kelvin near room
temperature. Since a natural frequency scales with the square root of
stiffness, df/f is about half of dE/E, so a material-only estimate is about
-1.5e-4 per K. Frequencies should fall as temperature rises.

| Mode | Reference Hz | df/f per K above 0 C | df/f per K below 0 C | Ratio to material-only |
| --- | --- | --- | --- | --- |
| 2 | 1.891 | -2.07e-4 | -7.75e-3 | 1.4 |
| 4 | 2.580 | -1.91e-4 | -3.03e-3 | 1.3 |
| 5 | 2.926 | -3.70e-4 | -4.21e-3 | 2.5 |
| 8 | 4.110 | -1.84e-4 | -1.56e-3 | 1.2 |
| 12 | 6.334 | -9.49e-4 | -1.42e-2 | 6.3 |

Every fitted slope above freezing is negative, as expected. Four modes are
within 1.2 to 2.5 times the steel-only estimate; mode 12 is about 6 times
more sensitive, which points to temperature-dependent parts other than the
steel (ballast, track, bearings) for that mode. Below 0 C the slopes are an
order of magnitude steeper, consistent with freezing stiffening, but they
rest on only 98 reference hours below 0 C and should be read as indicative.

### Frequency-ratio consistency

A uniform change of stiffness or mass scales every frequency by the same
factor, so the ratios between modes stay fixed. A local change moves modes
by different amounts. The temperature-normalized shifts after the retrofit
range from -0.73% to +2.11% and have mixed signs; mode 5's ratio to mode 2
changes by +2.4% and mode 12's by +2.0%. The study classifies this as a
**local** change, which matches what was done to the bridge: strengthening
of the diagonal connections, not a global stiffness change. A detector that
only tracked the mean frequency would have partly cancelled these opposite
moves.

## Limitations

- One bridge and one intervention. The retrofit raised some frequencies;
  damage usually lowers them, and a smaller or slower change would be harder
  to see.
- The changed segment covers autumn and winter only, and its temperature
  range sits inside the reference range.
- The holdout is 44 days, so a 0/44 false-alarm count still allows a true
  daily rate up to about 7% (95% upper bound).
- Only deck temperature is used as a covariate. Humidity, train loading and
  the identification uncertainty of each hourly estimate are not modelled.
- The 99th percentile limit uses 130 reference days and is itself uncertain.

## Reproduce

```bash
pip install -e '.[dataset]'
python3 tools/kw51_study.py --download
python3 tools/kw51_study.py --modes 2 4 8
```

Install Timoshenko from its main branch to use `fit_environmental_baseline`
directly; without it the identical numpy fit runs. `--json-out <file>`
writes the full result, including per-mode coefficients and shifts.
`tests/test_kw51_study.py` checks the helper functions on synthetic data
and needs no download.
