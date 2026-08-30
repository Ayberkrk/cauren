from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ASSET_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slug_part(value: str) -> str:
    lowered = str(value or "").strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    lowered = re.sub(r"_+", "_", lowered)
    return lowered[:20] or "unknown"


def _sanitize_asset_id(asset_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", str(asset_id or "").strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned[:64]


def _key(site_id: str, line_id: str, machine_id: str) -> str:
    return f"{_slug_part(site_id)}|{_slug_part(line_id)}|{_slug_part(machine_id)}"


@dataclass(frozen=True)
class AssetResolution:
    resolved_asset_id: str
    resolution_mode: str
    review_required: bool
    proposal_id: str | None
    proposal_status: str
    degraded_resolution: bool = False


class AssetIdentityResolver:
    """Resolves/learns asset identities with pending->approved lifecycle."""

    def __init__(self, sqlite_path: str | Path):
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self) -> None:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS asset_mappings (
                  identity_key TEXT PRIMARY KEY,
                  site_id TEXT NOT NULL,
                  line_id TEXT NOT NULL,
                  machine_id TEXT NOT NULL,
                  asset_id TEXT NOT NULL,
                  status TEXT NOT NULL,
                  proposal_id TEXT,
                  reviewer TEXT,
                  review_notes TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS asset_proposals (
                  proposal_id TEXT PRIMARY KEY,
                  identity_key TEXT NOT NULL,
                  site_id TEXT NOT NULL,
                  line_id TEXT NOT NULL,
                  machine_id TEXT NOT NULL,
                  suggested_asset_id TEXT NOT NULL,
                  status TEXT NOT NULL,
                  reason TEXT NOT NULL,
                  reviewer TEXT,
                  review_notes TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_asset_proposals_status ON asset_proposals(status)")
            conn.commit()
        finally:
            conn.close()

    def _next_proposal_id(self, conn: sqlite3.Connection) -> str:
        cur = conn.cursor()
        cur.execute("SELECT proposal_id FROM asset_proposals ORDER BY proposal_id DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            return "ASP-000001"
        try:
            number = int(str(row["proposal_id"]).split("-", 1)[1])
        except Exception:
            number = 0
        return f"ASP-{number + 1:06d}"

    def _suggested_asset_id(self, site_id: str, line_id: str, machine_id: str) -> str:
        candidate = f"tr_{_slug_part(site_id)}_{_slug_part(line_id)}_{_slug_part(machine_id)}"
        return _sanitize_asset_id(candidate)

    def _validate_provided_asset_id(self, asset_id: str) -> str:
        cleaned = _sanitize_asset_id(asset_id)
        if not ASSET_ID_PATTERN.fullmatch(cleaned):
            raise ValueError("asset_id must match [a-zA-Z0-9_-]{1,64}")
        return cleaned

    def resolve(
        self,
        *,
        asset_id: str | None,
        site_id: str | None,
        line_id: str | None,
        machine_id: str | None,
    ) -> AssetResolution:
        explicit = str(asset_id or "").strip()
        if explicit and explicit.lower() != "auto":
            provided = self._validate_provided_asset_id(explicit)
            return AssetResolution(
                resolved_asset_id=provided,
                resolution_mode="provided",
                review_required=False,
                proposal_id=None,
                proposal_status="none",
                degraded_resolution=False,
            )

        if not site_id or not line_id or not machine_id:
            raise ValueError("site_id, line_id and machine_id are required when asset_id is missing or auto.")

        site = _slug_part(site_id)
        line = _slug_part(line_id)
        machine = _slug_part(machine_id)
        identity_key = _key(site, line, machine)
        suggested = self._suggested_asset_id(site, line, machine)
        now = _now_iso()

        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM asset_mappings WHERE identity_key = ? LIMIT 1",
                (identity_key,),
            )
            mapping = cur.fetchone()
            if mapping and mapping["status"] == "approved":
                return AssetResolution(
                    resolved_asset_id=str(mapping["asset_id"]),
                    resolution_mode="approved_mapping",
                    review_required=False,
                    proposal_id=str(mapping["proposal_id"]) if mapping["proposal_id"] else None,
                    proposal_status="approved",
                )

            cur.execute(
                """
                SELECT * FROM asset_mappings
                WHERE machine_id = ? AND status = 'approved' AND (site_id != ? OR line_id != ?)
                LIMIT 1
                """,
                (machine, site, line),
            )
            collision = cur.fetchone()
            reason = "new_machine_pending"
            if collision:
                reason = "machine_id_conflict"

            cur.execute(
                """
                SELECT * FROM asset_proposals
                WHERE identity_key = ? AND status = 'pending'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (identity_key,),
            )
            existing = cur.fetchone()
            if existing:
                proposal_id = str(existing["proposal_id"])
            else:
                proposal_id = self._next_proposal_id(conn)
                cur.execute(
                    """
                    INSERT INTO asset_proposals (
                      proposal_id, identity_key, site_id, line_id, machine_id,
                      suggested_asset_id, status, reason, reviewer, review_notes, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        proposal_id,
                        identity_key,
                        site,
                        line,
                        machine,
                        suggested,
                        "pending",
                        reason,
                        None,
                        None,
                        now,
                        now,
                    ),
                )
                cur.execute(
                    """
                    INSERT INTO asset_mappings (
                      identity_key, site_id, line_id, machine_id, asset_id, status, proposal_id,
                      reviewer, review_notes, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(identity_key) DO UPDATE SET
                      asset_id=excluded.asset_id,
                      status=excluded.status,
                      proposal_id=excluded.proposal_id,
                      updated_at=excluded.updated_at
                    """,
                    (
                        identity_key,
                        site,
                        line,
                        machine,
                        suggested,
                        "pending",
                        proposal_id,
                        None,
                        None,
                        now,
                        now,
                    ),
                )
                conn.commit()

            return AssetResolution(
                resolved_asset_id=suggested,
                resolution_mode="pending_proposal",
                review_required=True,
                proposal_id=proposal_id,
                proposal_status="pending",
            )
        finally:
            conn.close()

    def list_pending(self, *, limit: int = 100) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT * FROM asset_proposals
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            )
            rows = cur.fetchall()
            return [self._proposal_row_to_payload(row) for row in rows]
        finally:
            conn.close()

    def review(
        self,
        *,
        proposal_id: str,
        action: str,
        reviewer: str,
        notes: str = "",
    ) -> dict[str, Any]:
        normalized = str(action or "").strip().lower()
        if normalized not in {"approve", "reject"}:
            raise ValueError("action must be approve or reject")
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM asset_proposals WHERE proposal_id = ? LIMIT 1", (proposal_id,))
            row = cur.fetchone()
            if not row:
                raise KeyError(f"asset proposal not found: {proposal_id}")
            now = _now_iso()
            status = "approved" if normalized == "approve" else "rejected"
            cur.execute(
                """
                UPDATE asset_proposals
                SET status = ?, reviewer = ?, review_notes = ?, updated_at = ?
                WHERE proposal_id = ?
                """,
                (status, reviewer, notes, now, proposal_id),
            )
            map_status = "approved" if normalized == "approve" else "rejected"
            cur.execute(
                """
                UPDATE asset_mappings
                SET status = ?, reviewer = ?, review_notes = ?, updated_at = ?
                WHERE identity_key = ?
                """,
                (map_status, reviewer, notes, now, row["identity_key"]),
            )
            conn.commit()
        finally:
            conn.close()
        return self.get_proposal(proposal_id)

    def get_proposal(self, proposal_id: str) -> dict[str, Any]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM asset_proposals WHERE proposal_id = ? LIMIT 1", (proposal_id,))
            row = cur.fetchone()
            if not row:
                raise KeyError(f"asset proposal not found: {proposal_id}")
            return self._proposal_row_to_payload(row)
        finally:
            conn.close()

    def lookup(self, *, site_id: str, line_id: str, machine_id: str) -> dict[str, Any]:
        site = _slug_part(site_id)
        line = _slug_part(line_id)
        machine = _slug_part(machine_id)
        identity_key = _key(site, line, machine)
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM asset_mappings WHERE identity_key = ? LIMIT 1", (identity_key,))
            row = cur.fetchone()
            if not row:
                return {
                    "identity_key": identity_key,
                    "site_id": site,
                    "line_id": line,
                    "machine_id": machine,
                    "asset_id": self._suggested_asset_id(site, line, machine),
                    "status": "none",
                    "proposal_id": None,
                }
            return {
                "identity_key": str(row["identity_key"]),
                "site_id": str(row["site_id"]),
                "line_id": str(row["line_id"]),
                "machine_id": str(row["machine_id"]),
                "asset_id": str(row["asset_id"]),
                "status": str(row["status"]),
                "proposal_id": str(row["proposal_id"]) if row["proposal_id"] else None,
            }
        finally:
            conn.close()

    def _proposal_row_to_payload(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "proposal_id": str(row["proposal_id"]),
            "identity_key": str(row["identity_key"]),
            "site_id": str(row["site_id"]),
            "line_id": str(row["line_id"]),
            "machine_id": str(row["machine_id"]),
            "suggested_asset_id": str(row["suggested_asset_id"]),
            "status": str(row["status"]),
            "reason": str(row["reason"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "reviewer": row["reviewer"],
            "review_notes": row["review_notes"],
        }
