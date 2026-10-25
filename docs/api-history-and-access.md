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
