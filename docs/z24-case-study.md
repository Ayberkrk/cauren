# Z24 bridge case study

`tools/z24_study.py` asks the research question on the one data set built
for it: after a healthy reference that spans a winter, does a detector stay
quiet through a warm summer and still flag real damage, and which
normalization makes the difference?

It does, for some arms. With temperature normalization or k nearest
neighbours, no healthy summer day alarmed, and the k-nearest-neighbour arm
flagged every scored damaged day from the first one, including the
smallest pier settlements. Raw frequencies, without normalization, saw the
damage on only 5 of 19 days.

## Data

- Source: the Z24 benchmark of the KU Leuven Structural Mechanics Section,
  from the SIMCES project (Maeck and De Roeck, "Description of Z24
  benchmark", Mechanical Systems and Signal Processing 17(1), 2003; Peeters
  and De Roeck, "One-year monitoring of the Z24-Bridge: environmental
  effects versus damage events", Earthquake Engineering and Structural
  Dynamics 30(2), 2001). Available for non-commercial research on request
  from KU Leuven; it may not be passed on and is never committed here.
- The bridge: a three-span post-tensioned concrete box-girder road bridge
  between Koppigen and Utzenstorf, Switzerland, demolished in 1998.
- The record used: the environmental monitoring system, which every hour
  stored 65536 samples at 100 Hz from the accelerometers that still worked
  and readings of about fifty temperature, weather and soil sensors, from
  1997-11-10 to 1998-09-10. The progressive damage tests themselves (forced
  and ambient vibration surveys per damage step) are in the archive but not
  used here.
- Archive integrity: the copy received lost 393,586 bytes in transfer, all
  inside one monitoring sub-archive. The reader locates every member by its
  local header instead of the zip's offsets and checks every CRC: one hourly
  record (`09F23`, 1998-01-08 23:00) is damaged and skipped, one more lacks
  a channel, and the other 5,651 hours and both damage test archives are
  intact.

### Damage timeline

As published for the benchmark (Maeck and De Roeck, 2003), with the dates of
each step's ambient survey:

| Dates (1998) | Step |
| --- | --- |
| Aug 4 to 5, Aug 9 to 10 | First and second reference measurements |
| Aug 13 to 14 | Pier settlement 40 mm (preceded by a first, smaller step) |
| Aug 17 to 18 | Pier settlement 80 mm |
| Aug 18 to 20 | Settlement of the pier, tilt of the foundation |
| Aug 20 to 21 | Third reference measurement |
| Aug 25 to 27 | Spalling of concrete, 12 and 24 m² |
| Aug 27 to 28 | Landslide at the abutment |
| Aug 31 to Sep 1 | Failure of concrete hinges at the abutment piers |
| Sep 2 to 4 | Failure of post-tensioning anchor heads, 1 and 4 |
| Sep 8 to 10 | Rupture of tendons |

## Protocol

Segments were fixed before any frequency was identified:

| Segment | Dates | Use |
| --- | --- | --- |
| Reference | 1997-11-10 to 1998-06-30 | 180 days with at least 6 scored hours; deck temperature -5.0 to 30.8 C, including freezing spells |
| Holdout | 1998-07-01 to 1998-08-03 | 26 scored healthy summer days, deck 17.0 to 32.0 C |
| Excluded | 1998-08-04 to 1998-08-11 | reference measurements with test equipment on the bridge, first works |
| Damaged | 1998-08-12 onwards | 19 scored days, to 1998-09-03 |

Identification, per hour:

- **Channels.** A first pass reads every channel's RMS. A channel whose
  monthly median RMS changes by more than a factor of 10 between reference
  months is dropped: one of the eight accelerometers is silent from
  December to April and about twenty times louder than the others in
  November and from May. The other seven are used.
- **Frequency Domain Decomposition.** Welch cross-spectral matrix of the
  seven channels (2048-sample Hann segments, 0.049 Hz bins), and its first
  singular value and vector between 2 and 14 Hz.
- **Reference modes.** The four most prominent peaks of the mean log first
  singular value over reference hours with deck temperature above 5 C:
  3.96, 5.18, 10.01 and 12.84 Hz. Each shape is the dominant direction of
  the reference hours' singular vectors at the peak.
- **Tracking by shape.** In every hour, among the local maxima within 20%
  of each reference frequency, those whose singular vector has a MAC of at
  least 0.6 with the reference shape are candidates, and the strongest wins.
  A fixed frequency band would not work: with the deck below -2 C the median
  frequencies are 5 to 9% higher than above 10 C, and up to 12% higher at
  the extremes.

Scoring uses `cauren_physics.novelty.run_protocol` unchanged: all seven arms,
the bilinear temperature model on the median of the deck-top temperature
sensors, leave-one-day-out calibration, conformal p-values at alpha 0.01
(floor 1/181) and the CUSUM of daily evidence. Hours without an estimate of
all four modes are dropped (88 to 97% of hours per mode are tracked).

## Results

| Arm | Holdout false-alarm days | Damaged days alarmed | First alarm (delay) | Holdout max / limit | Damaged min / limit | CUSUM: holdout alarm, damage delay |
| --- | --- | --- | --- | --- | --- | --- |
| Raw frequencies | 0/26 | 5/19 (26%) | 1998-08-28 (16 d) | 0.44 | 0.44 | none, 5 d |
| Temperature normalized | 0/26 | 13/19 (68%) | 1998-08-18 (6 d) | 0.51 | 0.82 | none, 2 d |
| Gaussian process normalized | 0/26 | 15/19 (79%) | 1998-08-13 (1 d) | 0.51 | 0.94 | none, 1 d |
| PCA residual | 5/26 (19%) | 1/19 (5%) | 1998-08-18 (6 d) | 2.64 | 0.22 | day 15, 6 d |
| k nearest neighbours | 0/26 | **19/19 (100%)** | **1998-08-12 (0 d)** | 0.97 | 1.32 | none, 1 d |
| Cointegration | 0/26 | 6/19 (32%) | 1998-08-19 (7 d) | 0.68 | 0.38 | none, 7 d |
| Daily mean normalized | 0/26 | 12/19 (63%) | 1998-08-18 (6 d) | 0.75 | 0.77 | day 20, 2 d |

Daily p-values through the damage period (an alarm is p ≤ 0.01; the floor is
0.0055):

| Day | Step | Raw | Bilinear | Gaussian process | k-NN |
| --- | --- | --- | --- | --- | --- |
| Aug 12 to 16 | settlement up to 40 mm | 0.09 to 0.13 | 0.017 to 0.022 | 0.006 to 0.017 | 0.0055 |
| Aug 17 | settlement 80 mm | 0.055 | 0.022 | 0.0055 | 0.0055 |
| Aug 18 to 26 | settlement, tilt, reference, spalling | 0.011 to 0.050 | 0.0055 | 0.0055 | 0.0055 |
| Aug 28 to Sep 3 | landslide, hinges, anchor heads | 0.0055 | 0.0055 | 0.0055 | 0.0055 |

### What it shows

- **Temperature is the masking effect, and normalizing it is the difference
  between seeing and missing damage.** Below 0 C the frequencies rise by
  0.85 to 2.6% per kelvin of cooling, 16 to 60 times their slope above
  freezing (-0.15 to +0.04% per kelvin), as the asphalt freezes. Raw
  frequencies reach the summer damage only once it is severe (from Aug 28);
  the bilinear model from the day after the 80 mm settlement (Aug 18); the
  Gaussian process on 2 of the first 6 days and on every day from Aug 17.
- **The k-nearest-neighbour arm is the most sensitive**, as in the synthetic
  drops on KW51: it flags the pier settlement from the first scored day,
  without any temperature model. It does so with the narrowest healthy
  margin (the hottest holdout day at 0.97 of its limit), since July reaches
  1 C above the warmest reference hour.
- **The sequential test works here and failed on KW51.** With a reference
  that spans winter to early summer, the CUSUM raised no holdout false alarm
  for five of seven arms and reached the damage 1 to 2 days in for the
  normalized arms. On KW51, whose reference ends in March, it false-alarmed
  in every arm's warm spring holdout.
- **PCA residuals fail again on a warm holdout** (5 of 26 days), as on KW51.
- The damage lowers every mode (-0.8, -6.6, -1.3 and -4.8% at the median),
  unevenly, consistent with a local change; the second mode moves by 7.8
  hourly identification sigmas.

### Identification uncertainty

Hourly identification noise is 0.8, 0.8, 1.1 and 3.5% of each frequency,
and 44, 32, 61 and 87% of what the bilinear temperature model leaves. The
fourth mode, at 12.8 Hz, is the least certain, and what is left of the
first two is mostly slow variation the deck temperature does not explain.
Mode tracking weakens after the damage starts (49% of August hours and 20%
of September hours have all four modes): the shapes change, and the RMS of
every channel drops during the tests, so the bridge is less excited. No day
after Sep 3 has enough hours to be scored, so the tendon ruptures are not
covered.

### Changes made after a first run

Two identification rules were changed after looking at the data, and one of
them after seeing results:

1. The MAC threshold was lowered from 0.8 to 0.6 after its distribution over
   reference hours showed noise peaks below 0.5 and the modes above 0.7.
   Only reference hours were inspected.
2. The first complete run picked, among candidates, the one with the
   highest MAC. That let the first mode jump to a weaker neighbouring peak
   of similar shape in 40% of reference hours (its robust hourly scatter
   was 3.0% of the frequency against 0.19% with the strongest peak). The
   rule was changed to the strongest candidate after the first run's
   results had been seen. With the first rule: raw 2/19, bilinear 13/19,
   Gaussian process 14/19, PCA 2/19, k-NN 12/19, cointegration 6/19, daily
   mean 12/19 damaged days, and 0/26 holdout false alarms for every arm.

The channel rule was added after the first identification showed channel
10's two regimes; it uses reference months only.

## Limitations

- One bridge, one damage campaign, 19 scored damaged days, 26 holdout days.
  0 of 26 still allows a true daily false-alarm rate up to about 11%.
- The damage period is also the period the bridge was under test; lower
  excitation and test equipment change the response as well as the damage
  does. The healthy holdout ends a week before the first damage
  step, so it does not cover that closure.
- The holdout is hotter than any reference hour (up to 32.0 C against
  30.8 C), which is why k-NN's healthy margin is narrow.
- Only the deck temperature enters the bilinear and Gaussian process
  models. `--environment` adds air temperature, humidity, wind and pavement
  temperature; the humidity channel's documented conversion appears to be
  a tenth of the correct one, which does not affect a linear model.
- The tendon ruptures (Sep 8 to 10) are not scored.

## Reproduce

The archive must be requested from KU Leuven and placed at
`data/z24/data-z24.zip`. Its documentation of channels, sensor labels and
voltage conversions is confidential and is not in this repository: copy
`tools/z24_config.template.json` to `data/z24/z24_config.json` and fill it
in from your own copy.

```bash
pip install -e '.[study]'
python3 tools/z24_study.py --extract     # about 12 minutes, writes data/z24/fdd.npz
python3 tools/z24_study.py
python3 tools/z24_study.py --environment
```

`tests/test_z24_study.py` checks the archive reader on a zip with bytes
missing from one member, the record names, the environmental file parser,
the channel rule, the configuration and mode tracking on simulated spectra,
with made-up labels and conversions; it needs no data.
