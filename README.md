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
| **Hell Bridge Test Arena**, full-scale steel truss (Norway) | Undamaged reference and known imposed damage states, shaker excitation, 100 Hz | Open, Zenodo [14028239](https://zenodo.org/records/14028239), CC BY 4.0 | Planned |
| **Z24**, post-tensioned concrete box girder (Switzerland) | One year of monitoring with air and soil temperatures, then progressive damage tests before demolition | On request from [KU Leuven](https://bwk.kuleuven.be/bwm/z24), non-commercial research | Access requested |

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
2. **Environmental normalization.** Each mode's frequency is regressed on
   temperature (bilinear, so the slope can change below freezing), and
   detection runs on the residuals.
3. **Novelty detection.** Detectors are fitted on a reference period only:
   raw-frequency Mahalanobis distance, temperature-normalized residual
   Mahalanobis distance, and a PCA residual. The control limit is the 99th
   percentile of the reference daily scores.
4. **Evaluation.** False alarms are counted on a later healthy holdout and
   detections after the known change, with the delay to the first alarm.

Steps 2 to 4 live in `cauren_physics/novelty.py`, so every bridge is
scored by the same code; a study only supplies its data and segment
dates.

A shear-building reference model (`cauren_physics/fe_reference_model.py`)
compares measured modes with model frequencies as a separate
model-consistency check. Model frequencies never replace a measured
baseline.

## Results so far

**KW51.** All three detectors had 0 of 44 false-alarm days in the spring
2019 holdout and alarmed on every post-retrofit day from the first one.
The retrofit moved two modes by about 2%, which makes this an easy case.
In a post-hoc rerun on the three least-shifted modes, temperature
normalization widened the detection margin (lowest post-retrofit score
1.82 times the limit, against 1.27 raw), while the PCA residual alarmed
on 23% of the warmer holdout days. Fitted temperature slopes are negative
for every mode, and the uneven, mixed-sign shifts are consistent with a
local change. This is one bridge and one intervention, not a validation.
Details are in the [KW51 case study](docs/kw51-case-study.md).

## Limitations

- Only one bridge has been analysed so far, and its change was a
  strengthening that raised some frequencies. Damage usually lowers them.
- Temperature is the only environmental covariate used. Humidity, traffic
  and the uncertainty of each identified frequency are not modelled.
- Short holdouts leave wide uncertainty on false-alarm rates: 0 of 44 days
  still allows a true daily rate up to about 7%.

## Reproduce

```bash
pip install -e '.[test]'
python3 -m pytest tests/            # 34 tests with pytest only (47 total with numpy, scipy and timoshenko-engine)

pip install -e '.[study]'           # numpy and scipy for the case studies
python3 tools/kw51_study.py --download
python3 tools/kw51_study.py
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
