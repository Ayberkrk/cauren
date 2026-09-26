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
  installed.

## Protocol

- Models are fitted on the training split only; the decision threshold (0.5)
  is fixed and never tuned on test data.
- Bridges never appear in more than one split, and the script refuses a
  dataset in which they do.
- Every input reading must be at or before its window's anchor year. The
  script refuses readings after the anchor year, because they would leak the
  outcome being predicted.
- A leave-one-state-out analysis fits the additive model on the training
  windows of the other states and scores every window of the held-out state.
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
- **Recall and false alarms at 0.5**, and **inspection capacity:** recall and
  false alarms when reviewing the highest-risk 5% or 10% of bridges. When the
  cutoff falls inside a block of tied scores, expected values over random tie
  breaks are reported, so CSV row order does not decide which bridges count.

## Results on the local v3 split

Test split: 8,503 bridges, 25.97% deteriorating within five years
(training rate 25.20%).

| Model | PR-AUC | Brier | ECE | Recall / false alarms at 0.5 | Top 5%: recall / false alarms | Top 10%: recall / false alarms |
|---|---:|---:|---:|---:|---:|---:|
| Train prevalence | 0.260 | 0.192 | 0.008 | 0.0% / 0 | 5.0% / 315 | 10.0% / 630 |
| Last composite score, logistic | 0.329 | 0.189 | 0.044 | 0.0% / 0 | 10.7% / 190 | 16.7% / 481 |
| Additive logistic | **0.426** | **0.177** | 0.014 | 11.7% / 186 | 11.5% / 173 | **20.3% / 403** |

The additive model ranks deteriorating bridges clearly better than both
baselines and has the lowest Brier score. Its ECE is low but not lower than the
constant prevalence predictor, whose near-zero ECE only reflects that the
training and test base rates are close. Leave-one-state-out PR-AUC for the
additive model was 0.443 (California), 0.280 (Iowa), and 0.198
(Pennsylvania), which is not enough evidence of geographic transfer.

The saved hybrid model was reported earlier at 19.2% recall with 144 false
alarms at the same 0.5 threshold, from a separate evaluation. Its PR-AUC,
Brier score, and capacity metrics require PyTorch and were not produced in this
run, so the table does not rank it against the others. None of these outputs
is exposed through the API; a model would first need validated ranking and
adequate probability calibration.

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
python3 -m pip install -e ".[benchmark]"
python3 tools/benchmark_cauren_bridge_models.py \
  --dataset-dir data/cauren_bridge \
  --checkpoint cauren_core/checkpoints/cauren_bridge_backbone_bundle.pt \
  --output-json data/cauren_bridge/model_benchmark.json
```

Add `--skip-hybrid` to run without PyTorch. The dataset itself is built with
`tools/build_cauren_bridge_dataset.py` and is not stored in the repository.
