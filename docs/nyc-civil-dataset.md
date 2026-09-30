# NYC DOB new-building CO outcome data set

This research data set links new-building permit records to later NYC
Certificates of Occupancy (COs), with three point-in-time candidate features.
The outcome measures permit administration and recorded completion. It is not
a building-safety, structural-condition, or inspection-quality label.

## Cohort, anchor, and outcome

- **Entity:** one record per valid NYC Building Identification Number (BIN).
- **Cohort:** BINs with an observed DOB new-building permit in BIS or DOB NOW.
- **Anchor:** earliest observed new-building permit issuance date for the BIN.
- **Primary outcome:** a qualifying non-temporary CO issued within five years
  after the anchor. BIS certificate types and DOB NOW `CO Issued` records are
  interpreted by their source-specific status/type fields; DOB NOW Initial,
  Final, and amendment filings can qualify. Temporary COs alone do not.
- **Conservative as-of date:** 2026-09-28, the day before the earliest source
  extraction began. Source feeds expose calendar dates without dependable row
  snapshot times, so records from the extraction day are not assumed observed.

The project review of the initial 57 Brooklyn BINs found that certificate
coverage required both DOB NOW and BIS. The builder searches both, stores
source and identifier evidence, and retains uncertain BIN-only links for
review. NYC describes different BIS and DOB NOW request periods in its
[CO guidance](https://www.nyc.gov/site/buildings/industry/obtain-a-co.page).

## Source inventory

The permit and CO feeds are downloaded with stable Socrata `:id` keyset
pagination. The violations and complaints feeds are much larger, so those
queries are restricted to the sorted 94,910-BIN cohort and run in resumable
chunks. Source IDs, query filters, row counts, extraction times, BIN-scope
hash, and SHA-256 file hashes are recorded in
[nyc-civil-dataset-snapshot.json](nyc-civil-dataset-snapshot.json).

| Source | Socrata ID | Extracted rows | Role / scope |
|---|---|---:|---|
| DOB Permit Issuance (BIS) | `ipu4-2q9a` | 571,868 | Legacy new-building permit rows (`job_type = NB`) |
| DOB NOW Build Approved Permits | `rbx6-tga4` | 1,006,100 | Issued permit rows, classified against DOB NOW filings |
| DOB NOW Build Job Application Filings | `w9ak-ipjd` | 56,874 | New-building classification and dated status context |
| DOB Certificate of Occupancy (BIS) | `bs8b-p36w` | 53,477 | Legacy new-building COs (`job_type = NB`) |
| DOB NOW Certificate of Occupancy | `pkdm-hqz6` | 33,825 | New-building COs (`job_type = New Building`) |
| DOB Violations (BIS) | `3h2n-5cm9` | 273,958 | Dated findings for cohort BINs only |
| DOB Complaints Received | `eabe-havv` | 463,499 | Dated complaints for cohort BINs only |

The first five raw feeds total about 2.1 GB; cohort findings are stored as
separate files. Raw and normalized extracts are git-ignored. A 600,000-row
citywide violations download was interrupted during collection and is not in
the source manifest or used by the data set.

## Current extraction result

The completed snapshot contains **94,910 buildings** from 102,914 normalized
permit episodes. The cohort BIN list has SHA-256
`0b5bc3e4a9096d756651bfb628f74a0ab75227e39f003860972eda8ee852e20b`; both
findings feeds record the same BIN hash and complete chunk coverage.

At the five-year horizon:

| Label status | Buildings |
|---|---:|
| Qualifying CO issued | 14,612 |
| Mature follow-up, no qualifying CO in either source | 73,336 |
| Right-censored follow-up | 3,525 |
| Conflicting parcel/BBL certificate candidates | 3,289 |
| Certificate type unresolved | 148 |

The binary-labeled subset contains 87,948 buildings; 16.6% are positive. Of
the positive links, 14,596 have both BIN and a matched job/filing identifier;
16 are BIN-only and remain part of the human review gate. Another 3,040
buildings have a temporary CO in the window. The labels describe what these
feeds recorded; absence is not a finding that work stopped or that a building
is unsafe.

The outcome-horizon sensitivity uses the same BIN, parcel-conflict, maturity,
and unresolved-type rules at each horizon:

| Horizon | Positive | Negative | Binary labeled | Right-censored | Parcel ambiguous | Type unresolved | Positive rate among binary labels |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 years | 7,439 | 83,418 | 90,857 | 2,226 | 1,816 | 11 | 8.2% |
| 3 years | 11,182 | 78,138 | 89,320 | 2,752 | 2,801 | 37 | 12.5% |
| 4 years | 13,300 | 75,189 | 88,489 | 3,113 | 3,212 | 96 | 15.0% |
| 5 years | 14,612 | 73,336 | 87,948 | 3,525 | 3,289 | 148 | 16.6% |
| 6 years | 15,692 | 71,964 | 87,656 | 3,795 | 3,252 | 207 | 17.9% |
| 7 years | 16,499 | 70,992 | 87,491 | 4,026 | 3,135 | 258 | 18.9% |

Positive labels preserve the qualifying certificate ID, date, source, BIN, BBL
comparison, and exact-job versus BIN-only match level. Negative labels are
emitted only when all seven required source slices are complete for this
cohort, follow-up has matured, and no plausible unresolved certificate
candidate remains. Censored and ambiguous records have no binary value.

## Candidate features and time checks

The data set creates three research candidates; it does not synthesize the
other five required `cauren-civil` scores.

| Feature | Construction | Current coverage / limitation |
|---|---|---|
| `permit_status_score` | Mean mapped friction status from DOB NOW filings whose filing and status dates are at or before the anchor; 1 is an objection/hold/incomplete/rejection, 0 is approved/issued | Available for 1,447 of 94,910 BINs (1.5%); missing remains missing |
| `inspection_finding_score` | `min(1, active dated violations + complaints at anchor, divided by 5)` | Available for all 94,910 BINs after complete cohort extraction; missing disposition dates are conservatively counted as active proxies |
| `natural_hazard_score` | `0.6 × clamp(PGA / 0.6g, 0, 1) + 0.4 × SFHA flag` | Available for 94,574 BINs; 336 lack coordinates |

The event records are filtered to dates at or before the permit anchor. The
audit found three late filing-date values and records that they were removed.
The findings feeds contain 37,969 violation rows and 1,339 complaint rows
without a disposition date; the derived score treats such pre-anchor records
as active proxies, a semantic assumption that still needs review.

Permit fields come from current Socrata snapshots, not complete historical
record versions. The audit therefore marks all 94,910 rows as having
unversioned static DOB fields. It also finds that the current 2021-10-13 FEMA
NFHL-derived layer postdates the anchor for 87,851 buildings. Current FEMA
data cannot be presented as historically available at those anchors. The FEMA
copy is an SFHA reduced layer that omits low-risk X polygons, and a no-hit is
provisional until map coverage is confirmed. The joined candidate layer found
1,040 SFHA point hits among 750 queried polygons.

USGS seismic values use the 2023.R2 CONUS hazard service, site class D, and the
nearest 0.05-degree grid point. All 57 queried grid cells returned curves; the
USGS release documents the current model at its [official dataset page](https://www.usgs.gov/data/mean-seismic-hazard-curves-and-uniform-hazard-ground-motion-values-revised-2023-us-50-state).
The flood source is the [NYSDOS-hosted FEMA NFHL reduced layer](https://opdgig.dos.ny.gov/datasets/NYSDOS%3A%3Afema-flood-hazard-zones/explore?showTable=true);
the [FEMA National Flood Hazard Layer catalog entry](https://catalog.data.gov/dataset/national-flood-hazard-layer)
describes the underlying program. Historical NFHL versions or exclusion of
the flood score are required before retrospective hazard claims.

## Audit result and modeling gate

`tools/audit_nyc_civil_dataset.py` reports **zero structural errors**: every
feature has a label-ledger row, all BIN groups remain in one split, positive
labels trace to an in-window qualifying CO, negative labels are mature, and
the cohort hash matches the complete findings slices. The deterministic BIN
split has 72,574 train, 11,259 validation, and 11,077 test buildings.

The audit correctly reports
`dataset_ready_for_modeling = false`. The 57-row stratified linkage queue has
not been human-reviewed (0/57), so match precision is unknown. The gate is at
least 57 reviewed rows, 20 decisive match decisions, at least 98% point
precision, and a 95% Wilson lower bound of at least 90%; expand the sample if
the confidence bound fails. The remaining gates are historical field-version
validation, the old FEMA vintage for 87,851 anchors, low permit-status
coverage, missing disposition semantics, and 336 missing hazard scores. A
clean structural audit does not resolve any of those evidence gaps. Do not
train a model or describe these rows as validated ground truth until the
review and point-in-time gates pass.

## Reproduction

```bash
# Download five core permit and CO feeds into a fresh snapshot directory.
python3 tools/download_nyc_civil_sources.py \
  --output-dir data/cauren_civil_nyc/raw

# Build once to define the BIN cohort. Missing findings feeds make all
# otherwise-negative records unknown in this intermediate output.
python3 tools/build_nyc_civil_dataset.py \
  --raw-dir data/cauren_civil_nyc/raw \
  --output-dir data/cauren_civil_nyc/normalized \
  --horizon-years 5

# Fetch findings only for those BINs, with resumable chunks and a scope hash.
python3 tools/download_nyc_cohort_findings.py \
  --bin-file data/cauren_civil_nyc/normalized/cohort_bins.txt \
  --output-dir data/cauren_civil_nyc/raw

# Build the candidate hazard layer, then rebuild and audit the final tables.
python3 tools/build_nyc_civil_hazard_layer.py \
  --feature-file data/cauren_civil_nyc/normalized/features.csv \
  --output-dir data/cauren_civil_nyc/normalized
python3 tools/build_nyc_civil_dataset.py \
  --raw-dir data/cauren_civil_nyc/raw \
  --output-dir data/cauren_civil_nyc/normalized \
  --horizon-years 5
python3 tools/audit_nyc_civil_dataset.py \
  --dataset-dir data/cauren_civil_nyc/normalized \
  --output-json data/cauren_civil_nyc/normalized/audit_report.json
```

Start from a new raw snapshot directory for changed source filters or cohort
definitions. The downloader refuses to overwrite existing data and verifies
completed snapshots by hash. A capped pilot is not complete cohort coverage
and cannot support negative labels.
