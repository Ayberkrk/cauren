# KW51 railway bridge case study

`tools/kw51_study.py` runs five modal novelty detectors on one real bridge
with a known change of state. It answers a narrow question: on KW51, does a
detector fitted on healthy data stay quiet on later healthy data and alarm
after the retrofit, and does temperature normalization help?

This is a single-bridge case study, not a validation of these detectors
in general. One structure, one change, and one season of post-change data
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
| Reference | 2018-10-01 to 2019-03-31 | fit every model; reference days for the p-values |
| Holdout | 2019-04-01 to 2019-05-14 | healthy, unseen: false alarms |
| Retrofit | 2019-05-15 to 2019-09-27 | excluded |
| Changed | 2019-09-28 to 2020-01-15 | retrofitted structure: detections |

- Modes are chosen from the reference period only: every mode identified in
  at least 90% of reference hours. That gives modes 2, 4, 5, 8 and 12
  (0-based), at about 1.89, 2.58, 2.93, 4.11 and 6.33 Hz. Hours missing any
  of these modes or the deck temperature are dropped (2,892 reference, 969
  holdout and 1,646 changed hours remain).
- Each arm produces an hourly score. The daily score is the median of a day's
  hourly scores, using days with at least 6 scored hours.
- Each later day gets a conformal p-value against the 130 reference days:
  p = (1 + reference days scoring at least as high) / 131. A day alarms when
  p <= 0.01. If a healthy day is exchangeable with the reference days, this
  keeps its false-alarm probability at most 1%. The smallest attainable p is
  1/131 = 0.0076, so alpha 0.01 is reachable; a smaller alpha would need a
  longer reference period, and the protocol refuses to run instead of
  silently never alarming. For 130 days and alpha 0.01 the rule is
  equivalent to alarming above the highest reference score.
- Each reference day's score comes from a model fitted on the other 129
  reference days (leave one day out). A model scores the hours it was
  fitted on as more typical than a new healthy day, so in-sample reference
  scores would set every limit too low. On KW51 the in-sample limits were
  about 8% lower; no alarm count changed.
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
3. **Gaussian process normalized:** as arm 2, but each mode's
   temperature curve is a Gaussian process (squared-exponential kernel,
   length scale chosen per mode by marginal likelihood) fitted to
   0.5 C bin means, so the curve may bend wherever the data says rather
   than only at a chosen hinge.
4. **PCA residual:** frequencies are standardized with reference statistics;
   the leading components that explain 90% of reference variance (4 of 5
   here) are treated as the normal co-variation pattern, and the score is
   the norm of what is left outside them.
5. **k nearest neighbours:** mean distance to the 10 nearest reference
   hours in standardized frequency space. It fits no ellipse and no
   temperature model, so it follows a curved healthy pattern, such as the
   steeper slope below freezing, from the reference hours themselves.

Timoshenko is optional. `fit_environmental_baseline` is on Timoshenko's main
branch and not in the released `timoshenko-engine` 2.1.0, so the script falls
back to the same ordinary least-squares fit in numpy when it is missing. The
results below were produced with the numpy fit; under the earlier
percentile limit, Timoshenko at commit `514f696` gave identical alarms and
coefficients.

## Results

Pre-registered mode set (2, 4, 5, 8, 12):

| Arm | Holdout false-alarm days | Post-retrofit days alarmed | First alarm (delay) | Holdout min p | Holdout max / limit | Post-retrofit min / limit |
| --- | --- | --- | --- | --- | --- | --- |
| Raw frequencies | 0/44 (0.0%) | 103/103 (100.0%) | 2019-09-28 (0 d) | 0.053 | 0.50 | 2.09 |
| Temperature normalized | 0/44 (0.0%) | 103/103 (100.0%) | 2019-09-28 (0 d) | 0.053 | 0.46 | 2.08 |
| Gaussian process normalized | 0/44 (0.0%) | 103/103 (100.0%) | 2019-09-28 (0 d) | 0.031 | 0.50 | 2.10 |
| PCA residual | 0/44 (0.0%) | 103/103 (100.0%) | 2019-09-28 (0 d) | 0.053 | 0.49 | 2.55 |
| k nearest neighbours | 0/44 (0.0%) | 103/103 (100.0%) | 2019-09-28 (0 d) | 0.069 | 0.56 | 4.25 |

All five arms meet both targets (holdout false-alarm rate at most 5%,
at least 90% of post-retrofit days alarmed) and alarm on the first scored
day. Every post-retrofit day reaches the smallest attainable p-value,
1/131. The holdout min p column is the strongest evidence any healthy
holdout day gave against "unchanged"; 0.053 means the most unusual holdout
day was still matched or exceeded by 6 of the 130 reference days. The last
two columns show the margin: the highest holdout daily score and the lowest
post-retrofit daily score, each divided by the highest score that does not
alarm. Values below 1 stay quiet, values above 1 alarm.

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
is the real test of environmental robustness, and all five arms stayed
below 0.6 of their limits there. The k-nearest-neighbour arm separates the
retrofit by the widest margin (the lowest post-retrofit day at 4.25 times
its limit).

### Post-hoc stress test

To see what happens when the change is small, the study was rerun with only
the three modes that move by less than 1% (2, 4, 8). This subset was picked
after looking at the post-retrofit shifts, so it is illustrative, not a
pre-registered result (`--modes 2 4 8`):

| Arm | Holdout false-alarm days | Post-retrofit days alarmed | First alarm (delay) | Holdout min p | Holdout max / limit | Post-retrofit min / limit |
| --- | --- | --- | --- | --- | --- | --- |
| Raw frequencies | 0/44 (0.0%) | 107/107 (100.0%) | 2019-09-28 (0 d) | 0.061 | 0.51 | 1.12 |
| Temperature normalized | 0/44 (0.0%) | 107/107 (100.0%) | 2019-09-28 (0 d) | 0.031 | 0.69 | 1.54 |
| Gaussian process normalized | 0/44 (0.0%) | 107/107 (100.0%) | 2019-09-28 (0 d) | 0.015 | 0.89 | 1.95 |
| PCA residual | 7/44 (15.9%) | 107/107 (100.0%) | 2019-09-28 (0 d) | 0.008 | 1.54 | 2.70 |
| k nearest neighbours | 0/44 (0.0%) | 107/107 (100.0%) | 2019-09-28 (0 d) | 0.061 | 0.66 | 3.06 |

With small shifts, temperature normalization widens the post-retrofit margin
(lowest daily score 1.54 times the limit with the bilinear model and 1.95
with the Gaussian process, against 1.12 for raw frequencies), while the
holdout stays below the limit; the Gaussian process pays for its wider
margin with a holdout day at 0.89 of its limit. The k-nearest-neighbour arm
reaches 3.06 with the holdout at 0.66. PCA keeps 2 of 3 components,
and its single residual direction picks up the warmer spring temperatures,
which lie outside the reference range: 16% of holdout days alarm, far above
the 1% the p-value rule promises for days that resemble the reference. Those
spring days do not resemble the reference, which is exactly the
exchangeability the guarantee assumes.

## Identification uncertainty

Each hourly frequency is an estimate with its own error. The true frequency
moves little within an hour, so the difference between consecutive hourly
estimates is mostly identification error; half its mean square is the
error variance. Over 2,731 consecutive reference hour pairs, after the
bilinear temperature model:

| Mode | Reference Hz | Identification sigma | Residual sigma | Share of residual variance that is identification noise | Post-retrofit shift / sigma |
| --- | --- | --- | --- | --- | --- |
| 2 | 1.891 | 5.1 mHz (0.27%) | 7.8 mHz | 0.44 | -1.2 |
| 4 | 2.580 | 12.7 mHz (0.49%) | 14.0 mHz | 0.83 | -0.6 |
| 5 | 2.926 | 4.1 mHz (0.14%) | 6.0 mHz | 0.48 | +14.9 |
| 8 | 4.110 | 3.3 mHz (0.08%) | 3.9 mHz | 0.70 | -9.1 |
| 12 | 6.334 | 12.7 mHz (0.20%) | 32.3 mHz | 0.15 | +8.2 |

For mode 4 almost all of what the temperature model leaves is
identification noise, so a better environmental model cannot help it. For
mode 12 only 15% is, and the rest is slow variation the deck temperature
does not explain, which fits its sixfold temperature sensitivity below. The
retrofit moved modes 5, 8 and 12 by 8 to 15 hourly sigmas, while the shifts
of modes 2 and 4 are within about one sigma of a single hour; a daily median
of about 22 hours narrows that sigma roughly fivefold.

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
- The identification noise estimate assumes the true frequency is constant
  within an hour; train passages and fast temperature changes make it a
  slight overestimate.
- The p-value guarantee assumes healthy days are exchangeable with the
  reference days. Consecutive days are correlated and seasons drift, so
  the 1% false-alarm bound is approximate, as the PCA arm's warm holdout
  shows.
- With 130 reference days the strongest attainable evidence is p = 1/131;
  every detection here sits at that floor, so the p-values cannot rank the
  five arms' post-retrofit evidence against each other.

The control rule changed from the 99th percentile of the reference daily
scores to the conformal p-value, and later to leave-one-day-out calibration.
With the percentile, 2 of 130 reference days exceeded the limit by
construction and the post-hoc PCA arm alarmed on 10 of 44 holdout days;
every other alarm count is unchanged across the three rules.

## Reproduce

```bash
pip install -e '.[study]'
python3 tools/kw51_study.py --download
python3 tools/kw51_study.py --modes 2 4 8
```

Install Timoshenko from its main branch to use `fit_environmental_baseline`
directly; without it the identical numpy fit runs. `--json-out <file>`
writes the full result, including per-mode coefficients and shifts.
`tests/test_kw51_study.py` checks the helper functions on synthetic data
and needs no download.
