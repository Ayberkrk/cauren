# Cauren

[![Tests](https://github.com/Ayberkrk/cauren/actions/workflows/tests.yml/badge.svg)](https://github.com/Ayberkrk/cauren/actions/workflows/tests.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Cauren is a research study on **vibration-based damage detection for
bridges**. It identifies natural frequencies from monitoring data,
removes the part of their variation explained by the environment, and
tests novelty detectors for how early they flag a structural change and
how often they raise false alarms on a healthy structure.

It is not a product or a service. Every result here comes from public
monitoring data, under one written protocol, with the code to reproduce it.

## Research question

> When temperature and other environmental effects move a bridge's natural
> frequencies by as much as damage does, which normalization and novelty
> detection methods detect the damage earliest, with the fewest false
> alarms on healthy data?

Natural frequencies fall when a structure loses stiffness, which makes
them a standard damage indicator. But they also change with temperature:
steel softens as it warms, and frozen asphalt, ballast or bearings can
stiffen a deck by several percent in winter. A detector that ignores this
either misses small damage or alarms every cold snap.

## Datasets

| Bridge | What is known | Access | Status |
|---|---|---|---|
| **KW51**, steel railway bowstring, Leuven (Maes and Lombaert) | 15 months of hourly modal frequencies and temperatures; diagonal connections strengthened in 2019 | Open, Zenodo [3745914](https://doi.org/10.5281/zenodo.3745914), CC BY-NC-SA 4.0 | Done, see [case study](docs/kw51-case-study.md) |
| **Hell Bridge Test Arena**, riveted steel truss railway bridge (Norway) | Two undamaged recordings and eight imposed damage states, shaker excitation, 100 Hz | Open, Zenodo [14632942](https://doi.org/10.5281/zenodo.14632942), CC BY 4.0 | Done, see [case study](docs/hbta-case-study.md) |
| **Vänersborg**, riveted steel bascule railway bridge (Sweden) | 64 bridge openings with strain, acceleration and weather; a real counterweight truss fracture in March 2023 | Open, Zenodo [8300495](https://doi.org/10.5281/zenodo.8300495), CC BY 4.0 | Done, see [case study](docs/vanersborg-case-study.md) |
| **Z24**, post-tensioned concrete box girder (Switzerland) | One year of monitoring with air and soil temperatures, then progressive damage tests before demolition | On request from [KU Leuven](https://bwk.kuleuven.be/bwm/z24), non-commercial research | Done, see [case study](docs/z24-case-study.md) |

No monitoring data is committed to this repository. Each study downloads
or reads its data into a gitignored folder under `data/`. Z24's terms in
particular forbid passing the data to third parties.

## Method

1. **Modal identification.** `cauren_physics/oma.py` identifies natural
   frequencies from one vibration channel by peak picking on a Welch
   spectrum, with half-power damping reported only when the bandwidth is
   resolved. With several aligned channels and Timoshenko 2.x installed,
   Frequency Domain Decomposition adds mode shapes. Identified modes are
   paired with a baseline one to one, and differences inside one frequency
   bin are not counted as drift. See
   [OMA and Timoshenko integration](docs/oma-timoshenko-integration.md).
   For shaker tests with a measured input (HBTA), the H1 frequency response
   is used instead; for bridge openings (Vänersborg), the strain change of
   each gauge at fixed leaf angles.
2. **Environmental normalization.** Each mode's frequency is modelled
   against temperature, either bilinear (the slope can change below
   freezing), with a Gaussian process, or with weather-station covariates
   added, and detection runs on the residuals.
3. **Novelty detection.** Seven arms are fitted on a reference period only:
   Mahalanobis distance of the raw frequencies, of the bilinear and Gaussian
   process residuals and of daily-averaged residuals, a PCA residual, the
   mean distance to the nearest reference samples, and cointegration
   (combinations of the modes that cancel their common environmental
   trend). Each reference day is scored by a model fitted without it. Each later day gets a conformal p-value against
   those reference days, and alarms when p <= alpha (0.01 by default). If a
   healthy day resembles the reference days, its false-alarm probability is
   at most alpha. A reference too short for the chosen rate is refused
   before anything is scored, rather than producing a detector that can
   never alarm. A CUSUM of daily evidence is run alongside as a sequential
   test.
4. **Identification uncertainty.** The error of each identified frequency
   is estimated from consecutive hourly estimates (KW51) or from the
   window-to-window scatter of one recording (HBTA), and every shift is
   also reported in units of it.
5. **Evaluation.** False alarms are counted on a later healthy holdout and
   detections after the known change, with the delay to the first alarm.
   On KW51, frequency drops of known size are also injected into the
   healthy holdout to find the smallest change each arm detects.

Steps 2 to 5 live in `cauren_physics/novelty.py`, so every bridge is
scored by the same code; a study only supplies its data and segment
dates.

A shear-building reference model (`cauren_physics/fe_reference_model.py`)
compares measured modes with model frequencies as a separate
model-consistency check. Model frequencies never replace a measured
baseline.

## Results so far

**KW51.** Every arm had 0 of 44 false-alarm days in the spring 2019 holdout
and alarmed on every post-retrofit day from the first. The retrofit moved
two modes by about 2%, an easy case, so drops of known size were injected
into the healthy holdout: cointegration and k nearest neighbours detect a
0.5% drop of the best-observed modes, or of all modes together, on at
least 90% of days, where the Mahalanobis arms need 1%. Weather covariates
beyond the deck temperature change little, and a CUSUM raises false alarms
in the warmer holdout. Hourly identification noise is 0.08 to 0.49% of each
frequency. Details are in the [KW51 case study](docs/kw51-case-study.md).

**Hell Bridge Test Arena.** Fitted on one undamaged recording, the
detectors that respond to the eight imposed damage states also alarm on
every window of the second undamaged recording, five days later. Several
frequencies moved by 1 to 1.7% between the two healthy sessions, about five
times the scatter within one recording and as much as most damage states.
Details are in the [HBTA case study](docs/hbta-case-study.md).

**Vänersborg.** The two strain gauges next to a real fracture moved by 5.4
and 4.0 reference sigmas, but the pre-registered detectors missed it,
because one erratic gauge inflated their calibration. Excluding that gauge
by a reference-only rule, added after the miss, the Mahalanobis distance
flagged all six openings after the fracture and one of 17 healthy ones.
Details are in the [Vänersborg case study](docs/vanersborg-case-study.md).

**Z24.** The one data set with a healthy reference spanning a winter and
real damage afterwards. Hourly frequencies come from Frequency Domain
Decomposition of seven channels, tracked by mode shape because frozen
asphalt raises them by up to 12%. No arm with temperature normalization or
k nearest neighbours alarmed on any of 26 healthy summer days; k nearest
neighbours flagged all 19 scored damaged days from the first pier
settlement, the Gaussian process 15 and the bilinear model 13, while raw
frequencies saw the damage on 5. With a reference that spans the seasons,
the CUSUM that failed on KW51 raised no healthy false alarm for five of
seven arms and reached the damage within 1 to 2 days. Details are in the
[Z24 case study](docs/z24-case-study.md).

The four studies are compared in [Results across bridges](docs/results-summary.md).

## Limitations

- Only Z24 combines real damage with a healthy reference spanning the
  seasons, and its damage was imposed during a test campaign on a bridge
  about to be demolished. KW51's change is a strengthening, HBTA has two
  healthy recordings, and Vänersborg's reference covers 11 winter days.
- Two Z24 identification rules (the MAC threshold and choosing the strongest
  candidate peak) were changed after a first run; both runs are reported.
- The Vänersborg detection depends on a data-quality rule added after the
  pre-registered analysis missed the fracture.
- Short holdouts leave wide uncertainty on false-alarm rates: 0 of 44 days
  still allows a true daily rate up to about 7%.
- The false-alarm guarantee assumes healthy data resembles the reference.
  Correlated days, seasons outside the reference range, and new measurement
  sessions weaken it, as HBTA and the CUSUM on KW51 show.

## Reproduce

```bash
pip install -e '.[test]'
python3 -m pytest tests/            # 34 tests with pytest only (76 total with numpy, scipy and timoshenko-engine)

pip install -e '.[study]'           # numpy, scipy, h5py and pandas for the case studies
python3 tools/kw51_study.py --download
python3 tools/kw51_study.py
python3 tools/kw51_study.py --sensitivity
python3 tools/hbta_study.py --fetch      # about 145 MB of the 2.5 GB record
python3 tools/hbta_study.py
python3 tools/vanersborg_study.py --fetch  # 550 MB download, deleted after
python3 tools/vanersborg_study.py --exclude-erratic
python3 tools/z24_study.py --extract     # archive from KU Leuven at data/z24/data-z24.zip
python3 tools/z24_study.py
```

Single-channel identification, optionally with a baseline and an FE
model-consistency check:

```bash
python3 tools/run_oma_identification.py \
  --input-csv my_vibration.csv --column acceleration --sampling-hz 100 \
  --baseline-frequencies-hz 2.01 \
  --fe-story-masses-kg 50000 --fe-story-stiffness-n-per-m 8000000
```

Multi-channel FDD needs [Timoshenko Engine](https://github.com/Ayberkrk/timoshenko)
2.0 or newer in the same environment (`pip install timoshenko-engine`;
the import name is `timoshenko`):

```bash
python3 tools/run_oma_identification.py \
  --input-csv my_channels.csv --columns north east west --sampling-hz 100
```

## Repository layout

| Path | What it is |
|---|---|
| `cauren_physics/` | Modal identification, baseline pairing, FE reference model, optional Timoshenko adapter, and the shared normalization and novelty-detection protocol (`novelty.py`) |
| `tools/` | Bridge studies and the OMA command line tool |
| `docs/` | One page per study: data, protocol, results, limitations |
| `tests/` | Synthetic tests with known answers; no download needed |

## Earlier work

Before this study, Cauren was a decision-support prototype: an HTTP API,
a civil building risk agent, a bridge deterioration model trained on the
FHWA National Bridge Inventory, and an NYC building permit cohort. That
work is preserved at the
[`decision-support-prototype`](https://github.com/Ayberkrk/cauren/tree/decision-support-prototype)
tag.

## Contributing

See `CONTRIBUTING.md`.

## License

Apache 2.0. See `LICENSE`. Datasets keep their own licenses.
