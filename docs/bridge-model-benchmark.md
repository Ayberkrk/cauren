# Bridge deterioration model comparison

`tools/benchmark_cauren_bridge_models.py` compares simple, well-understood
models with the saved hybrid bridge model on the same bridge-disjoint splits.
It answers one question: does a candidate model rank and calibrate
deterioration risk better than baselines, on bridges it has not seen?

## Models

- **Train prevalence:** every bridge gets the training-set deterioration rate.
  It cannot rank bridges; it shows what calibration looks like with no
  discrimination at all.
- **Last composite structural score, logistic calibration:** one feature, the
  most recent structural risk score.
- **Additive logistic model:** summaries of each score history (last, first,
  mean, spread, extremes, trend per year, missing ratio), bridge age at the
  anchor year, and inspection count, span, and gap statistics.
- **Histogram gradient boosting:** bounded tree ensemble, run when
  scikit-learn is installed (`benchmark` extra).
- **Saved hybrid model:** evaluated with the same metrics when PyTorch is
  installed. This is the previously saved checkpoint, scored once on the
  untouched test split.

## Protocol

- Models are fitted on the training split only; the decision threshold (0.5)
  is fixed and never tuned on test data.
- Bridges never appear in more than one split, and the script refuses a
  dataset in which they do.
- Every input reading must be at or before its window's anchor year. The
  script refuses readings after the anchor year, because they would leak the
  outcome being predicted.
- The primary comparison evaluates every model on the exact same 8,503 test
  windows. The primary rows use the saved checkpoint, train-prevalence
  predictor, last-score logistic model, the additive logistic model using
  only the three core score histories, and histogram boosting with those same
  histories. Expanded tabular models are reported separately.
- Strict leave-one-state-out (LOSO) evaluation removes every window from the
  held-out state from both training and validation. Tabular models are refit
  on the remaining states' train rows; the hybrid architecture is retrained
  from scratch on those rows and early-stopped on remaining-state validation
  rows. Every window in the held-out state is scored. This is a geographic
  transfer check, not the same estimand as the untouched random test split.
- The 836 MB readings file is streamed in one pass and numerical libraries are
  pinned to one thread, so the run fits in a few hundred megabytes of memory.

## Metrics

- **PR-AUC (average precision):** ranking quality for the minority class.
  Equal to the positive rate for a model that cannot rank.
- **Brier score:** mean squared error of the predicted probability. Rewards
  both ranking and calibration.
- **Expected calibration error (ECE):** gap between predicted and observed
  rates over 10 equal-frequency bins. Tied scores always share a bin, so the
  result does not depend on row order. ECE alone rewards a constant predictor
  that matches the base rate; read it together with PR-AUC and Brier.
- **Calibration intercept and slope:** logistic recalibration diagnostics on
  the test labels. Ideal values are 0 and 1. A slope below 1 means predictions
  are too spread out; a negative intercept shifts calibrated probabilities
  down around the middle of the score range. The report leaves these values
  null and records a fit status when scores are constant, labels contain one
  class, the fit is separated, or the numerical fit is unstable.
- The report retains the 10 equal-frequency calibration-bin counts, mean
  predicted probabilities, and observed rates for each model and state.
- **Recall and false alarms at 0.5**, and **inspection capacity:** recall and
  false alarms when reviewing the highest-risk 5% or 10% of bridges. When the
  cutoff falls inside a block of tied scores, expected values over random tie
  breaks are reported, so CSV row order does not decide which bridges count.

## Results on the local v3 split

Test split: 8,503 bridges, 25.97% deteriorating within five years
(training rate 25.20%).

| Model | Input tier | PR-AUC | Brier | ECE | Calibration intercept | Calibration slope |
|---|---|---:|---:|---:|---:|---:|
| Train prevalence | constant | 0.260 | 0.1923 | 0.0077 | — | — |
| Last structural score, logistic | one core score | 0.329 | 0.1887 | 0.0435 | 0.116 | 1.062 |
| Same-input additive logistic | three core score histories | 0.421 | 0.1779 | 0.0229 | 0.053 | 1.013 |
| Expanded additive logistic | score histories, bridge age, inspection cadence | 0.426 | 0.1768 | 0.0144 | 0.049 | 1.010 |
| Same-input histogram boosting | three core score histories | **0.551** | **0.1609** | 0.0152 | 0.147 | 1.117 |
| Expanded histogram boosting | score histories, bridge age, inspection cadence | **0.556** | **0.1600** | 0.0145 | 0.145 | 1.120 |
| Saved hybrid checkpoint | three core score histories | 0.532 | 0.1640 | **0.0119** | 0.082 | 1.067 |

The hybrid's test PR-AUC is 0.532 and Brier score 0.1640. It improves on the
additive baselines, while both histogram boosters score higher on PR-AUC and
lower on Brier. The hybrid has the lowest ECE among learned models.
Calibration intercept/slope are diagnostics; the calibration-bin values
remain in `model_benchmark.json` for inspection.

## Paired bridge-cluster uncertainty

The benchmark resamples the 8,503 test bridges 2,000 times with replacement
(seed 42). Every model uses the same sampled bridge IDs in each replicate.
Intervals are percentile 95% intervals. The saved checkpoint and predictions
are held fixed, so these intervals describe test-bridge sampling uncertainty;
they do not include training-seed, checkpoint-selection, or source-data
uncertainty. The current test split has one window per bridge, but the code
resamples by `asset_id` so additional windows for a bridge stay together.
Intervals are descriptive and unadjusted for multiple comparisons.

The table reports each paired difference as **hybrid minus comparator**.
Positive PR-AUC favors the hybrid. Negative Brier favors the hybrid.

| Comparator | Δ PR-AUC (95% interval) | Δ Brier (95% interval) |
|---|---:|---:|
| Train prevalence | +0.273 [+0.255, +0.291] | -0.0283 [-0.0312, -0.0256] |
| Last structural score, logistic | +0.204 [+0.188, +0.220] | -0.0247 [-0.0272, -0.0222] |
| Same-input additive logistic | +0.111 [+0.095, +0.126] | -0.0139 [-0.0160, -0.0118] |
| Expanded additive logistic | +0.106 [+0.090, +0.122] | -0.0128 [-0.0148, -0.0108] |
| Same-input histogram boosting | -0.018 [-0.026, -0.010] | +0.0032 [+0.0019, +0.0043] |
| Expanded histogram boosting | -0.024 [-0.032, -0.015] | +0.0040 [+0.0028, +0.0053] |

On this fixed split, paired intervals favor the hybrid over the logistic
baselines and favor histogram boosting over the hybrid on both PR-AUC and
Brier. This is evidence about these test bridges and frozen predictions, not
generalization to new states or new training runs. The machine-readable
intervals for each model and each paired difference are in
`model_benchmark.json`. That report also records the exact test window IDs,
input and checkpoint SHA-256 hashes, package versions, and model random seeds.

## Geographic holdouts

The strict LOSO refit removes the held-out state from all fitting and
validation. The table compares the hybrid refit with both same-input tabular
baselines. State codes are FHWA FIPS: `06` California, `19` Iowa, `42`
Pennsylvania.

| Held-out state | Test windows | Positive rate | Hybrid PR-AUC / Brier | Additive logistic PR-AUC / Brier | Histogram boosting PR-AUC / Brier |
|---|---:|---:|---:|---:|---:|
| California (`06`) | 20,852 | 33.87% | 0.356 / 0.2573 | 0.370 / 0.2940 | **0.461 / 0.2268** |
| Iowa (`19`) | 20,096 | 25.68% | 0.280 / 0.2423 | **0.286** / 0.2072 | 0.284 / **0.1963** |
| Pennsylvania (`42`) | 15,731 | 13.88% | **0.199** / 0.1431 | 0.200 / 0.1284 | 0.195 / **0.1338** |

Held-out expected calibration error (10 equal-frequency bins) is also mixed:

| Held-out state | Hybrid ECE | Additive logistic ECE | Histogram boosting ECE |
|---|---:|---:|---:|
| California (`06`) | 0.1825 | 0.2510 | **0.1102** |
| Iowa (`19`) | 0.1826 | 0.1244 | **0.0715** |
| Pennsylvania (`42`) | 0.1550 | **0.0986** | 0.0925 |

The saved checkpoint's unrefit test-window PR-AUC varies by state (0.687,
0.343, 0.244), and the strict refits show weaker and uneven geographic
transfer. The holdout rates also differ substantially, so cross-state results
are not interchangeable. These data support using the hybrid as a candidate,
not claiming robust cross-state superiority. Full state-level calibration
bins, ECE, intercept/slope, capacity metrics, and every model are retained in
the generated report. California's additive-logistic intercept/slope fit is
ill-conditioned and is marked null with a fit status; its ECE and Brier score
remain valid direct probability diagnostics.

## Component condition histories

The three-feature bridge dataset keeps a composite structural score but not
the separate deck, superstructure, and substructure ratings. The dataset
builder now also writes `condition_history.csv` with those ratings as risk
scores, truncated at each window's anchor year, without changing the bridge
agent's three-feature runtime contract. The benchmark uses the file when it
exists; it never reconstructs component scores from the composite. Datasets
built before this change lack the file and must be rebuilt from the FHWA
source bundle to compare component histories. Rebuilt datasets also treat
condition codes outside 0 to 9 (such as the sentinel 99) as missing instead of
as ratings.

## Reproduce

```bash
python3.11 -m pip install -e ".[benchmark]"
python3.11 tools/benchmark_cauren_bridge_models.py \
  --dataset-dir data/cauren_bridge \
  --checkpoint cauren_core/checkpoints/cauren_bridge_backbone_bundle.pt \
  --output-json data/cauren_bridge/model_benchmark.json \
  --output-predictions data/cauren_bridge/model_predictions.csv \
  --bootstrap-replicates 2000 \
  --bootstrap-seed 42 \
  --run-geographic-holdout
```

`--run-geographic-holdout` adds the strict LOSO refits reported above, which
retrain the neural hybrid once per state; leave it off for a quick same-split
comparison. Add `--skip-hybrid` when PyTorch is unavailable. The local dataset and generated report/predictions are not stored
in the repository; the benchmark code reproduces them from the FHWA source
bundle.
