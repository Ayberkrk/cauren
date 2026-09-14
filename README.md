# Cauren

[![Tests](https://github.com/Ayberkrk/cauren/actions/workflows/tests.yml/badge.svg)](https://github.com/Ayberkrk/cauren/actions/workflows/tests.yml)

Cauren is a civil-engineering diagnostics and decision-support system.
It ingests sensor/CBS ("bina/insaat" GIS) readings or bridge inspection
records, normalizes them into a canonical feature schema, runs them
through a generic anomaly-detection core, and hands the result to a
domain physics/reasoning layer to produce an explainable,
**non-authoritative** risk diagnosis.

Cauren does not issue final engineering decisions or automated
enforcement. Diagnoses are meant for field inspection, permit review,
and risk prioritization, never as an automated final engineering or
enforcement decision.

```
raw sensor/CBS/inspection payload
        |
        v
  normalization  ->  agent router  ->  sector agent  ->  explainable diagnosis
                                            |
                                statistical anomaly core
                                            +
                                  domain physics layer
```

## Two sector agents

| | `cauren-civil` | `cauren-bridge` |
|---|---|---|
| Domain | Buildings / construction | Bridges |
| Required features | 8 | 3 |
| Live API today | Yes (default) | Registered, not exposed by default |
| Trained model | No (statistical fallback only) | Yes, see [Evaluation](#evaluation) |
| Ground truth used | None available publicly at this scope | Real 5-year deterioration outcome (FHWA) |

`cauren-bridge` exists because real, public, per-asset ground truth is
available for bridges (FHWA inspection history) but not, at this scope,
for the broader building schema. It is intentionally narrower.

### `cauren-civil` feature schema (8 required)

| Feature | Unit | What it represents |
|---|---|---|
| `structural_risk_score` | ratio (0-1) | Structural condition / risk level |
| `inspection_finding_score` | ratio (0-1) | Severity of open field-inspection findings |
| `permit_status_score` | ratio (0-1) | Permit/approval completeness |
| `construction_progress_pct` | % (0-100) | Construction completion level |
| `infrastructure_connection_score` | ratio (0-1) | Utility/infrastructure connection readiness |
| `natural_hazard_score` | ratio (0-1) | Seismic/flood/other natural hazard exposure |
| `occupancy_safety_score` | ratio (0-1) | Life-safety / occupancy risk |
| `ground_stability_score` | ratio (0-1) | Geotechnical/foundation stability |

Optional: `building_height_m`, `footprint_area_m2`, `soil_settlement_mm`,
`utility_disruption_score`.

### `cauren-bridge` feature schema (3 required)

| Feature | Unit | Source |
|---|---|---|
| `structural_risk_score` | ratio (0-1) | Derived from FHWA deck/superstructure/substructure condition ratings |
| `ground_stability_score` | ratio (0-1) | Scour risk/vulnerability derived from FHWA Item 113 (scour-criticality rating) via an explicit code -> risk mapping (categorical, not linear); 1.0 = poor stability / high vulnerability, same risk direction as `cauren-civil`'s `ground_stability_score` |
| `natural_hazard_score` | ratio (0-1) | USGS seismic hazard (PGA) + FEMA flood zone |

## Evaluation

`cauren-bridge`'s backbone (a masked-autoencoder encoder, a sequential
1-D convolution branch, and an RNN-Twin temporal-consistency branch,
plus a small classification head) is trained and evaluated on a real
dataset built from public sources, not synthetic or heuristic data.

| | |
|---|---|
| Source | FHWA National Bridge Inventory (2000-2023) + USGS seismic hazard + FEMA flood zone |
| States covered | California, Iowa, Pennsylvania |
| Bridges | 56,679, each with a real, independently-observed 5-year outcome |
| Supervised target | `deck_drop_5yr`: did the deck condition rating actually drop within 5 years? |
| Train / validation / test | 39,675 / 8,501 / 8,503 windows, split by bridge (no bridge appears in two splits) |
| Validation accuracy | **77.3%** (pre-dates the checkpoint selection rule below; see note) |
| Majority-class baseline | 74.1% |
| Parameters | 176,196 |

77.3% against a 74.1% baseline is a modest, genuine improvement, not a
spectacular one, and that is the point: the input window for each
prediction is strictly limited to years at or before the outcome's
reference year, so the model cannot see the future it is predicting. A
model claiming much higher accuracy on this kind of task without a
similar leakage check should be treated with suspicion. See
`data/cauren_bridge/dataset_summary.json` (built locally, gitignored
because of size, see [Rebuilding the bridge dataset](#rebuilding-the-bridge-dataset))
and `cauren_core/checkpoints/cauren_bridge_backbone_bundle.pt` for the
trained artifact.

**Checkpoint selection.** For a dataset without a real supervised target
(`cauren-civil`), the best checkpoint is the epoch with the lowest
validation reconstruction (MAE) loss. For `cauren-bridge`, which has one
(`deck_drop_5yr`), the best checkpoint is instead the epoch with the
lowest validation supervised loss (binary cross-entropy on
`deck_drop_5yr`), not reconstruction loss -- the two can diverge (an
epoch can reconstruct the input window better while forecasting the
outcome worse), and the model is explicitly evaluated on the forecasting
task. BCE is used over raw accuracy for selection because the target is
imbalanced (~25% positive); reconstruction loss is still recorded every
epoch for diagnostics (`cauren_core/training.py`'s
`train_summary["validation_loss_history"]`), and the selected metric and
its value/epoch are recorded in `train_summary["selection_metric"]` /
`train_summary["best_selection_metric_value"]`. The 77.3% figure above
was measured before this rule existed (the checkpoint was previously
selected by reconstruction loss); it should be re-measured by retraining
(`python3 tools/train_cauren_core.py --profile civil_cpu_small --agents
cauren-bridge`) and evaluating the resulting checkpoint once on the
untouched test split.

## Status and honest limitations

This project is a working research prototype, not a finished product.

- **`cauren-civil` has no pretrained model yet.**
  `cauren_core/checkpoints/` ships without one for it. Out of the box,
  the anomaly-detection core runs on its statistical fallback
  (median/spread-based calibration and pattern-shape scoring), not a
  trained neural backbone.
- **A single-snapshot request has no time series to analyze.** If you
  send one reading per feature with no history, the core reports
  `insufficient_temporal_history` / `snapshot_only` instead of a
  fabricated pattern verdict, and the risk score is driven mainly by
  the physics layer's read of the values you sent. Send a real time
  series (`seq_len` distinct timestamps) for genuine pattern-shape
  detection (drift, spike, oscillation, flatline, and so on).
- **`LLM/` (`rocket_llm`) is pre-alpha.** Its `pyproject.toml` declares
  several CLI entry points that are not implemented yet; only
  `LLM/src/rocket_llm/pipelines/training.py` exists.
- **`cauren-civil` does not derive its 8 scores from raw sensor/CBS
  data, it consumes them.** The required features are expected to
  already exist (from a field inspection, an engineering assessment,
  another model) before they reach Cauren. This was validated end to
  end against a real public dataset (US Census building-permit records
  via Hugging Face, `thanna94/us-building-permits`): it cleanly
  produced two of the eight, `permit_status_score` and
  `construction_progress_pct`, but the other six have no off-the-shelf
  public source, and the pipeline correctly reported them as
  `missing_features` instead of inventing values. What Cauren adds on
  top of scores you already have is anomaly-pattern detection over
  time, physics-relation reasoning between scores, and an explainable
  diagnosis, not score generation from raw measurements.

## What ships with every diagnosis

Beyond the risk score itself, each diagnosis carries the checks that were
run on the way to it, so a reviewer can see how much to trust it:

| Output key | What it answers |
|---|---|
| `quality_control` | Were the inputs usable? Required-feature completeness, unit-range sanity, flatline/stuck sensors, how many sensor names failed to normalize, how many raw readings were rejected. |
| `uncertainty_report` | How much should the score be trusted? A confidence band around the risk score derived from data quality, the core's own confidence, and how much of the physics schema had evidence. |
| `physics_evidence` | Why is the score what it is? The domain relations that fired, and which required features were missing. |

Structural-health inputs are optional and separate. `cauren_physics/oma.py`
identifies a structure's natural frequencies and damping from an ambient
vibration series (peak-picking operational modal analysis), and
`cauren_physics/fe_reference_model.py` computes what a shear-building
mass/stiffness model predicts those frequencies should be. Comparing the
two is a theory-vs-data consistency check; a measured frequency dropping
below the baseline is read as possible stiffness loss and folded into the
civil physics layer's `structural_reliability` signal.

```bash
# Full explainable review over real data (repo ships a small public dataset)
python3 tools/run_explainable_review.py \
  --wide-csv data/public_sources/normalized/civil_public_core_assets.csv

# ...or from a raw sensor feed, with a modal-drift check folded in
python3 tools/run_explainable_review.py \
  --sensors-json my_sensors.json \
  --vibration-csv my_vibration.csv --sampling-hz 100 \
  --baseline-frequencies-hz 2.01
```

These are decision-support signals for routing an asset to a human
reviewer, not damage verdicts.

## Repository layout

| Path | What it is |
|---|---|
| `cauren_core/` | Sector-agnostic runtime: contracts, orchestrator, statistical anomaly core, the trainable backbone, normalization |
| `cauren_agents/` | Sector agents (`civil`, `bridge`) and the router |
| `cauren_physics/` | Domain physics/reasoning layer, one module per sector |
| `api/` | FastAPI service exposing the pipeline over HTTP |
| `tools/` | Dataset build and training scripts |
| `data/` | Dataset manifests, sources, and (locally built) training data |
| `tests/` | Test suite (62 tests) |
| `operations/` | Architecture and data-contract reference docs |
| `LLM/` | Pre-alpha advisory-LLM sub-project, not yet functional |

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health/liveness`, `/health/readiness` | Health checks |
| `GET` | `/agents` | List registered sector agents |
| `POST` | `/calibrate` | Normalize/calibrate a sensor payload without running physics |
| `POST` | `/diagnose` | Full pipeline: route, calibrate, classify, physics, compose. Accepts either a `sensors` list or a CBS-shaped building record (`building_id` + `risk_assessments`/`project_permit`/etc, converted internally) |

`api/app.py` is a thin, unauthenticated wrapper around
`cauren_core.CaurenPipeline` -- there is no built-in login, token, or
per-tenant access control. It's meant to be run locally or behind
whatever the operator puts in front of it (a reverse proxy, an auth
layer), not exposed to the open internet as-is.

## Quick start

The project is pip-installable from the repo root (`pyproject.toml`
declares `cauren_core`, `cauren_agents`, `cauren_physics`, `api`, and
`tools` as packages, with no required dependencies of its own):

```bash
pip install -e '.[test]'
python3 -m pytest tests/            # 62 tests, no network/GPU/numpy/torch required

pip install -e '.[api]'             # only needed to actually run the API
python3 api/app.py                  # or: uvicorn api.app:app --reload
```

Installing individual packages directly still works if you'd rather not
use the extras above:

```bash
pip install fastapi pydantic pytest httpx
pip install uvicorn                 # only needed to actually run the API
```

`numpy` is optional: `cauren_physics/oma.py` uses it for faster FFTs when
present, and falls back to a pure-Python DFT otherwise. Nothing else in
the pipeline touches it. Building/training against the bridge dataset
needs the `dataset` extra (`pip install -e '.[dataset]'`, i.e.
`numpy`/`pandas`/`scipy`), and training the core backbone needs the
`backbone` extra (`torch`) -- see the next section.

### Rebuilding the bridge dataset

`data/cauren_bridge/` is not checked in (it is ~1.5GB and fully
reproducible from public sources). To rebuild it you need `pandas`,
`pyarrow`, and `scipy` in addition to the packages above, and to pull
the source data yourself:

- FHWA National Bridge Inventory: `sweetapricity/bridgedeck-nbi` on
  Hugging Face
- USGS seismic hazard: `sweetapricity/bridgedeck-nshm`
- FEMA flood zone: `sweetapricity/bridgedeck-nfhl`

See `data/cauren_bridge/dataset_summary.json`'s `source` field and
`cauren_physics/bridge.py`'s module docstring for the feature
derivations, and `tools/train_cauren_core.py` for the training
entry point once the dataset exists locally.

## Contributing

See `CONTRIBUTING.md`.

## License

Apache 2.0. See `LICENSE`.
