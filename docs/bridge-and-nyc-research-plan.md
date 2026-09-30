# Cauren bridge and NYC civil research plan

**Decision purpose:** decide whether the saved bridge hybrid deserves to remain
the default research candidate, and whether a narrowly defined NYC permit
outcome can support a credible `cauren-civil` research dataset. Neither work
stream changes the production API or turns the project into an automated
engineering decision system.

## Workstream A: fair bridge model comparison

### Question and comparison contract

The question is whether the saved hybrid ranks and estimates five-year deck
deterioration risk better than simpler alternatives on the same unseen bridges.
Freeze the checkpoint, v3 data snapshot, bridge-disjoint train/validation/test
lists, feature definitions, and five-year target before scoring. A model may
use only inspection history dated on or before its prediction anchor. A test
label may be read only by the metric calculation; no test result may select a
checkpoint, feature, threshold, or calibrator.

Compare these candidates on the exact same test rows:

1. Training prevalence (no ranking ability; probability reference).
2. Last structural score with logistic calibration.
3. Additive logistic model on the same three score histories as the hybrid.
4. Histogram gradient boosting on those same three score histories.
5. Expanded additive logistic and histogram boosting using age and inspection
   cadence, shown in a separate feature tier.
6. The saved hybrid checkpoint, scored without refitting on the test set.

The same-input tier is the primary comparison. The expanded tier answers
whether simple tabular metadata adds value; it must not be used to imply that
the hybrid lost an equal-input contest if feature sets differ.

### Required reporting

For every model, publish row count and prevalence beside:

- PR-AUC / average precision as the primary ranking metric for the minority
  deterioration class.
- Brier score for overall probabilistic error.
- Ten-bin reliability values, expected calibration error, calibration
  intercept, and calibration slope. Read ECE with Brier and PR-AUC; a constant
  prevalence predictor can have low ECE without ranking any bridge.
- Precision, recall, and false alarms at the fixed 0.5 threshold, plus recall
  and false alarms at predeclared 5% and 10% inspection capacity. Resolve tied
  scores with expected random tie breaking rather than file order.
- Paired uncertainty intervals for score differences, resampling at the bridge
  level so multiple windows from one bridge cannot act as independent
  observations. Include intervals for PR-AUC and Brier differences before
  claiming a statistically meaningful winner.

Keep the test split as a final descriptive estimate. Use validation only for
early stopping and other fitted choices. Record model seeds, library versions,
checkpoint hash, data-manifest hash, and exact row IDs with the result bundle.

### Geographic transfer

Run strict leave-one-state-out folds for California, Iowa, and Pennsylvania.
Remove the held-out state's windows from both fitting and validation; refit
every learned model from scratch on the remaining states; then score all
windows in the held-out state. Report the same ranking and probability metrics
and the held-out state's prevalence. These three states are a small transfer
study, not evidence of nationwide performance. Do not average away large
state-to-state differences.

### Current result and decision rule

The saved hybrid reaches PR-AUC 0.5323 and Brier 0.1640 on the 8,503-row test
split. Same-input histogram boosting reaches 0.5506 and 0.1609; the hybrid has
the lowest ECE among learned models at 0.0119. In 2,000 paired bridge-cluster
bootstrap draws, the hybrid-minus-same-input-boosting difference is -0.0184
PR-AUC (95% interval [-0.0263, -0.0096]) and +0.00315 Brier (95% interval
[+0.00186, +0.00433]). Those intervals favor boosting on both metrics for
this frozen test split. The hybrid remains a candidate because its ECE is
lower and the geographic holdouts are mixed; neither result supports a new
API probability or a broad transportability claim.

## Workstream B: narrow NYC civic data set

### Estimand and cohort

Use one record per valid NYC BIN in the DOB new-building cohort. Set the
prediction anchor to the earliest observed new-building permit issuance date
for that BIN. The primary outcome is a genuine initial/final Certificate of
Occupancy (CO) recorded within five years after the anchor. This is an
administrative permit-completion outcome, not a structural-safety or
inspection-quality outcome.

Keep these cases separate in the label ledger:

- Positive: qualifying final CO in the horizon, with BIN, date, source, and
  job-identifier match confidence retained.
- Negative: only when every required permit, CO, violation, and complaint
  extraction is complete for the cohort; five-year follow-up has matured; and
  no plausible unresolved CO candidate remains.
- Unknown: right-censored follow-up, incomplete source slice, unresolved
  certificate type, or conflicting parcel evidence.

Use both BIS and DOB NOW CO feeds. Normalize and deduplicate by BIN, preserve
all source identifiers, and keep BIN-only links reviewable. The initial 57
building investigation found that relying on DOB NOW alone missed CO evidence;
matching coverage therefore must be measured across the two systems together.

### Source and feature plan

The reproducible source set is:

| Feed | Use |
|---|---|
| BIS permit issuance | Legacy new-building permit dates and fields |
| DOB NOW approved permits | Current permit events, linked to application filings |
| DOB NOW job application filings | New-building classification and dated status context |
| BIS Certificates of Occupancy | Legacy certificate outcomes |
| DOB NOW Certificates of Occupancy | Current certificate outcomes |
| DOB Violations | Dated pre-anchor inspection findings |
| DOB Complaints Received | Dated pre-anchor complaint findings |

Download the permit and CO feeds with source IDs, filters, row counts, and
SHA-256 hashes. Query the much larger findings feeds only for the sorted BINs
in the new-building cohort, in resumable chunks; record the cohort-list hash,
chunk/page sizes, and source snapshot times. A cohort hash mismatch invalidates
the findings slice and prevents negative labels.

Initial feature candidates are intentionally few and traceable:

- `permit_status_score`: dated DOB NOW filing statuses at or before the
  permit anchor, with an explicit status-to-score map and missing values left
  missing.
- `inspection_finding_score`: active DOB violation and complaint counts at
  the anchor, divided by a fixed five-event scale and clipped to [0,1].
- `natural_hazard_score`: USGS 2%-in-50-year PGA and FEMA flood-zone status,
  with component values, coordinate coverage, and source vintage retained.

Do not synthesize the other five required civil scores from permit records.
Keep the permit-record fields and derived administrative outcome distinct.

### Leakage and label-quality gates

Before modeling, require all of the following:

1. Source snapshots and query scopes reproduce from their manifest; the
   findings feeds cover exactly the cohort BIN hash.
2. Every positive links to a genuine issued CO within the five-year window.
   Every negative is mature and comes from complete source coverage. Unknowns
   stay out of binary model fitting.
3. Review the stratified 57-record linkage sample, including exact job matches,
   BIN-only matches, parcel conflicts, and no-record cases. Require at least
   57 reviewed rows, at least 20 decisive match decisions, at least 98% point
   precision, and a 95% Wilson lower bound of at least 90%. If the confidence
   gate fails, expand the review sample.
4. Filter every dated permit, violation, and complaint event to the anchor;
   assert no source event or feature date is later than the anchor. Preserve
   removed late-date counts in the audit.
5. Validate the historical availability of static DOB permit fields before
   putting them into a retrospective model. Current feed values do not prove
   what was known at the old permit date.
6. Validate FEMA map vintage against every anchor. The currently discovered
   New York State copy derives from a 2021 NFHL snapshot, so it cannot be used
   for older anchors as though it were point-in-time truth. Obtain historical
   NFHL versions or omit the flood component from historical-model claims.
7. Keep all records for one BIN in one split. In addition to the deterministic
   BIN-grouped split, report a borough holdout and a later-period test cohort
   after labels have matured.

The dataset is not ready for model training until the linkage review and
point-in-time field/hazard checks pass. A clean structural audit is necessary
but does not replace human link review or source-history validation.

## Delivery gates and sequence

| Gate | Deliverable | Exit condition |
|---|---|---|
| A1 | Frozen bridge benchmark and reproducible predictions | Identical test rows, no split crossing, no post-anchor inputs, all required metrics |
| A2 | Paired uncertainty analysis | **Complete:** paired bridge-cluster intervals for every test model versus the hybrid; conclusions limited to this frozen split |
| B1 | Versioned NYC raw-source manifest and BIN cohort | All seven source slices complete, hashes verified, cohort scope matches |
| B2 | Label ledger and linkage queue | Label invariants pass; 57 manual cases reviewed; precision gates pass |
| B3 | Point-in-time feature audit | Dated events pass; static fields and FEMA vintage either validated or excluded |
| B4 | Baseline model evaluation | Only after B2 and B3; BIN-grouped, borough-held-out, and temporal test results |

At the 2026-09-30 review of the 2026-09-29 data snapshot, A1, A2, and B1 are
complete. A2 uses 2,000 paired bootstrap draws, seed 42, and the 8,503 common
test bridges; its machine-readable output is part of the generated benchmark
report. B2 has a passing structural audit but has not passed its human-review
gate (0/57 reviewed). B3 has dated-event filters, but historical static-field
versions, FEMA vintage, and missing-disposition semantics remain unverified.
B4 has not started because it depends on B2 and B3.

Do the work in dependency order: A1 and B1 can proceed in parallel; A2
depends on A1; B2 and B3 depend on B1; B4 depends on both B2 and B3. Keep
generated source extracts and predictions out of git; commit code, manifests,
audit summaries, and concise methodology/results documents.

## Known limitations and follow-on investigations

- Three bridge states do not establish broad geographic transportability.
- NYC CO timing and recorded completion are administrative outcomes subject
  to source coverage and filing practice; they do not measure whether a
  building is safe.
- The permit cohort is a narrow new-building slice. Major alterations and
  different CO-obligation classes need their own cohort and target definition.
- Complaint/violation severity, duplicate handling, disposition semantics,
  and missing BIN rates need source-level sensitivity checks before treating
  the combined count as a calibrated severity scale.
- The existing civil runtime consumes eight precomputed scores. This work
  supplies only three candidate research features and does not create a
  `cauren-civil` production model.
