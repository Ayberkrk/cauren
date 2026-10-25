# Bridge deterioration model v4

This page documents the v4 bridge-year panel, the models trained on it, and
how they compare with the v3 benchmark on the same test bridges. Numbers come
from `docs/bridge-model-v4-results.json`, produced by
`tools/colab/bridge_v4.ipynb` (three runs on the pinned public bundles listed
in `tools/colab/run_bridge_v4.py`).

## What changed from v3

- **Data unit.** v3 keeps one window of three derived scores per bridge. v4
  (`tools/build_bridge_years_v4.py`) keeps every bridge-year from 2000 to 2023
  for CA, IA and PA: 1,738,454 rows over 114,958 bridges.
- **Inputs.** v4 adds the NBI items the deterioration literature relies on:
  ADT and truck share, reconstruction year, structure kind and type, deck
  structure type, wearing surface, membrane and deck protection, span, length,
  width, skew, owner, functional class, design load, waterway adequacy,
  inspection date and inspection frequency. Causal history features (time at
  the current deck rating, past drops and repairs, years since inspection and
  reconstruction) are computed only from year t or earlier.
- **Leakage guard.** Every column that looks at later years is prefixed
  `label_`. `tests/test_bridge_v4_dataset.py` poisons all years after a cutoff
  and checks no feature changes.
- **Same target.** `label_deck_drop_5yr` reproduces the v3 `deck_drop_5yr`
  label; agreement on the 56,668 v3 anchors is 100%.

### Inspection timing matters

NBI annual files repeat the last inspection's ratings in years without a new
inspection. In the Delaware check the one-year deck drop rate was 13.7% in
years with a fresh inspection and 0.1% otherwise. A drop is observed at an
inspection, so the inspection clock (item 090 date, item 091 frequency) is an
input to every v4 model. Years since inspection is the fourth most important
v4 feature by gain.

## Models

All models are LightGBM boosters with the same hyperparameters
(`HAZARD_PARAMS` in `tools/bridge_hazard.py`), early stopped on the
validation bridges. They are scored on the 8,503 v3 test bridges
(25.97% positive).

| Model | Formulation | Inputs |
|---|---|---|
| `core_single_window_boosting` | one anchor per bridge | v3-equivalent: three runtime scores, component condition history, age |
| `v4_single_window_boosting` | one anchor per bridge | all v4 inputs |
| `v4_single_window_boosting_monotone` | same, with monotone constraints | risk never falls with age, years since reconstruction, ADT or truck ADT |
| `*_out_of_time` | trained only on bridge-years whose outcome was known by 2018 (anchor year at most 2013), scored on the 2018 anchors | core or v4 |
| `v4_hazard_end*` | discrete-time hazard on every observed year, rolled forward to five years | all v4 inputs |

## Results

PR-AUC is the main ranking metric; Brier and expected calibration error (ECE)
measure probability quality. Intervals are paired bridge-cluster bootstrap
percentile intervals (2,000 draws) on identical test bridges.

### Same period (anchor 2018, outcome by 2023)

| Model | PR-AUC | Brier | ECE | Top 5% precision |
|---|---:|---:|---:|---:|
| Train prevalence | 0.260 | 0.192 | | |
| v3 benchmark boosting (with condition history) | 0.655 | 0.142 | 0.019 | |
| Core (v3-equivalent) boosting | 0.646 | 0.143 | 0.010 | 0.97 |
| v4 hazard, trained through 2023 | 0.660 | 0.141 | 0.034 | 0.95 |
| **v4 single window** | **0.706** | **0.130** | 0.015 | 0.98 |
| v4 single window, monotone | 0.705 | 0.130 | 0.015 | 0.98 |

- v4 minus core: PR-AUC +0.060 [+0.049, +0.070], Brier -0.013 [-0.015, -0.011].
- v4 minus the v3 benchmark booster: PR-AUC +0.051 [+0.039, +0.062].
- The monotone constraints cost nothing measurable (PR-AUC 0.7054 vs 0.7055).

### Out of time (the deployment setting)

Trained only on outcomes known by 2018, scored on the 2018 anchors:

| Model | PR-AUC | Brier | ECE |
|---|---:|---:|---:|
| Core out of time | 0.486 | 0.171 | 0.037 |
| **v4 out of time** | **0.553** | **0.160** | 0.027 |

v4 minus core: PR-AUC +0.067 [+0.054, +0.080]. The new inputs help when
forecasting a later period, not only cross-sectionally.

**Same-period scores overstate deployment accuracy.** Both feature sets lose
about 0.15 PR-AUC when the model has not seen the outcome period (0.706 to
0.553 for v4). Report the out-of-time figure when describing expected field
performance.

### Leave one state out

The model is trained on two states and scored on every bridge of the third.

| Held-out state | Core | v4 | v4 hazard |
|---|---:|---:|---:|
| California | **0.613** | 0.462 | 0.422 |
| Iowa | 0.373 | 0.357 | 0.368 |
| Pennsylvania | 0.293 | 0.287 | 0.283 |

v4 does **not** transfer better across states and is clearly worse in
California. The added categorical items (owner, surface type, functional
class) carry state-specific coding and practice. Use v4 only for states it
was trained on; for a new state, recalibrate (below) or prefer the core
inputs until local outcomes are available.

### Calibration and uncertainty (`tools/bridge_calibration.py`)

- In distribution the raw v4 probabilities are already calibrated (ECE
  0.015). Per-state Platt scaling fitted on validation bridges gives 0.011.
- In a new state, raw probabilities are badly calibrated (ECE 0.09 to 0.12).
  Platt recalibration on k labelled bridges from that state brings mean ECE
  to 0.04 to 0.07 with k = 100, 0.03 to 0.05 with k = 300, and 0.02 to 0.04
  with k = 500 (20 random draws each). This is "leave one state out plus k
  labels", not pure transfer.
- Mondrian split-conformal outcome sets at 90% nominal coverage reach 89.2%
  to 90.8% empirical coverage per state. A set containing both outcomes means
  the model cannot separate them at that confidence: 5% of Pennsylvania test
  bridges, 20% in California and 35% in Iowa.

## Negative results

- **Discrete-time hazard.** Fitting the one-year hazard on every interval and
  composing it to five years (PR-AUC 0.660) is statistically
  indistinguishable from the v3 booster (difference +0.005, interval
  [-0.008, +0.020]) and worse than the v4 single-window booster. Its
  probabilities are also less calibrated (slope 0.82). The structure is kept
  in `tools/bridge_hazard.py` because it gives horizon-consistent curves, but
  it is not the recommended model.
- **Monotone constraints.** They neither help nor hurt accuracy; they are
  kept because they make the model's response to age and traffic physically
  plausible.
- **Neural models.** The earlier hybrid network (PR-AUC 0.532 on the v3
  split) remains below every boosted model here. No new neural model was
  adopted.

## Reproducing

Open `tools/colab/bridge_v4.ipynb` in Colab and run all cells. The runner
downloads the pinned bundles, rebuilds v3 and v4, trains every arm and
copies results to Google Drive. Pass `--hazard-end-years` without values to
skip the hazard arm and `--skip-v3-benchmark` to skip the v3 benchmark. On a
machine with enough memory:

```bash
python3 tools/build_bridge_years_v4.py --raw-dir nbi_raw --nshm-csv nshm.csv \
    --nfhl-csv nfhl.csv --states CA IA PA \
    --v3-windows data/cauren_bridge/agents/cauren-bridge/windows.csv \
    --output-dir data/cauren_bridge_v4
python3 tools/bridge_hazard.py --panel data/cauren_bridge_v4/bridge_years.parquet \
    --v3-dataset-dir data/cauren_bridge --output-dir data/cauren_bridge_v4/hazard
python3 tools/bridge_calibration.py \
    --predictions data/cauren_bridge_v4/hazard/hazard_predictions.csv \
    --model v4_single_window_boosting
```

The v4 build reads all three states into memory at once, which did not fit
comfortably on an 8 GB laptop; use Colab or a larger machine.

## Limits

- Outcomes are NBI deck rating drops, which are inspector judgements on a
  coarse 0-9 scale, not measured deterioration.
- Three states only. Results for other states are unknown until they are
  added and evaluated.
- Predictions are decision support for inspection prioritization, not
  structural safety assessments.
