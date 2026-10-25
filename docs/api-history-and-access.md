# API history and access controls

The API is intended for a single local operator. It is not a multi-tenant
service and does not partition history by client ID.

## Local history

Successful `/diagnose` and `/calibrate` responses are stored in SQLite. The
default database path is `~/.local/share/cauren/history.sqlite3`; set
`CAUREN_HISTORY_DB` to choose another path. The default retention is 90 days;
set `CAUREN_HISTORY_RETENTION_DAYS` to a positive integer to change it.
Expired records are removed when history is written or read. The API does not
store the raw request body, including addresses or geometry. Stored response
data still contains feature values and diagnosis results, so protect the
database file and configure its location accordingly.
On POSIX systems, a newly created Cauren database directory is owner-only
(`0700`) and the SQLite file is owner-only (`0600`). Existing parent volume
directory permissions are preserved; an existing database file is restricted
to `0600` when the operator has permission to change it.

Retrieve recent records for one asset with `GET /history/{asset_id}`. The
optional `limit` query parameter defaults to 100 and is capped at 1,000.
Records are returned newest first. History is local to the process host; it is
not encrypted or replicated by Cauren.

Docker Compose mounts a named `cauren_history` volume at `/data`, so the
container database survives ordinary container recreation. Set
`CAUREN_API_KEY` in the host environment before starting Compose to protect
API routes; health checks remain available without a key.

## Bearer authentication and request bounds

With `CAUREN_API_KEY` unset, local calls remain unauthenticated for
backward-compatible development. When configured, all routes except
`/health/liveness` and `/health/readiness` require
`Authorization: Bearer <CAUREN_API_KEY>`. Keep the key in the process
environment or a secret manager; do not commit it to the repository.

Request bodies are limited to 1 MiB and sensor arrays to 8,192 rows. A request
that exceeds either bound is rejected before diagnosis. The service continues
to be a single-operator prototype; the Bearer key does not provide user
accounts or tenant isolation.

## Per-asset baselines

The statistical core scores a window against a per-asset reference: for each
feature, a robust center, a robust scale, the number of reference samples, and
whether the feature is expected to vary at all. A reference is resolved per
request in this order:

1. `baseline` in the `/diagnose` body:
   `{"<feature>": {"center": 0.4, "scale": 0.03, "n": 120, "expected_variation": true, "unit": "ratio"}}`.
   Every value must be finite, `scale` must be positive, and `n` must be at
   least 30; otherwise the request is rejected with 422.
2. The baseline stored for the request's `asset_id` with
   `POST /baseline/{asset_id}` (only when `asset_id` is given explicitly).
3. None. The core then has only the window itself to compare against; the
   response reports `"reference": "none"` and a lower core confidence.

The response always carries `reference` (`payload`, `stored`, or `none`) and
`reference_coverage`, which lists the features with and without a reference
and any supplied features outside the agent schema.

`POST /baseline/{asset_id}` takes a reference window in the `/diagnose` sensor
row format (every row needs a `timestamp`), normalizes it with the same alias
and unit rules as diagnosis, and fits each feature's median and
MAD x 1.4826 (the mean absolute deviation when more than half the samples are
identical). A feature that never varies in the reference has no scale to
measure against: pass `min_scale` for it (for example the measurement
resolution) and it is stored with `expected_variation: false`. Features with
fewer than 30 samples are listed under `skipped`. Only these summary
statistics are stored, in the `asset_baselines` table of the history
database; the reference readings themselves are not retained. Stored
baselines are configuration and are not removed by history retention.
`GET /baseline/{asset_id}` returns what is stored.

A baseline is decision-support context, not a certification of normal
behavior: fit it from a period the operator has reviewed and believes to be
representative.
