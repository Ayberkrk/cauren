from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(value: Any, *, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


class CanonicalReviewStore:
    def __init__(self, sqlite_path: str | Path):
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_column(self, conn: sqlite3.Connection, name: str, ddl: str) -> None:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(unknown_cases)")
        existing = {str(row["name"]) for row in cur.fetchall()}
        if name not in existing:
            cur.execute(f"ALTER TABLE unknown_cases ADD COLUMN {ddl}")

    def _ensure_tables(self) -> None:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS unknown_cases (
                  unk_code TEXT PRIMARY KEY,
                  status TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  source_event_id TEXT NOT NULL,
                  reason TEXT NOT NULL,
                  draft_json TEXT NOT NULL,
                  approved_json TEXT,
                  event_json TEXT NOT NULL,
                  report_json TEXT NOT NULL,
                  reviewer TEXT,
                  review_notes TEXT
                )
                """
            )
            self._ensure_column(conn, "source_layer", "source_layer TEXT NOT NULL DEFAULT 'model'")
            self._ensure_column(conn, "source_layers_json", "source_layers_json TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column(conn, "fault_family_id", "fault_family_id TEXT NOT NULL DEFAULT 'unknown_anomaly'")
            self._ensure_column(conn, "fault_family_label", "fault_family_label TEXT NOT NULL DEFAULT 'Unknown Anomaly'")
            self._ensure_column(conn, "fault_subtype_id", "fault_subtype_id TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "fault_subtype_label", "fault_subtype_label TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "fault_signature_id", "fault_signature_id TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "fault_signature_text", "fault_signature_text TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "known_status", "known_status TEXT NOT NULL DEFAULT 'unknown'")
            self._ensure_column(conn, "similar_family_candidates_json", "similar_family_candidates_json TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column(conn, "review_required", "review_required INTEGER NOT NULL DEFAULT 1")
            self._ensure_column(conn, "decision_ids_json", "decision_ids_json TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column(conn, "review_action", "review_action TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "asset_class", "asset_class TEXT NOT NULL DEFAULT 'generic'")
            self._ensure_column(conn, "asset_norm_scope", "asset_norm_scope TEXT NOT NULL DEFAULT 'global'")
            self._ensure_column(conn, "asset_norm_confidence", "asset_norm_confidence REAL NOT NULL DEFAULT 0.0")
            self._ensure_column(conn, "norm_source", "norm_source TEXT NOT NULL DEFAULT 'global'")
            self._ensure_column(conn, "candidate_dimensions_json", "candidate_dimensions_json TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column(conn, "observing_dimensions_json", "observing_dimensions_json TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column(conn, "provisional_dimensions_json", "provisional_dimensions_json TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column(conn, "asset_critical_dimensions_json", "asset_critical_dimensions_json TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column(conn, "dimension_importance_by_asset_json", "dimension_importance_by_asset_json TEXT NOT NULL DEFAULT '{}'")
            self._ensure_column(conn, "dimension_lifecycle_status_json", "dimension_lifecycle_status_json TEXT NOT NULL DEFAULT '{}'")
            conn.commit()
        finally:
            conn.close()

    def _next_code(self, conn: sqlite3.Connection) -> str:
        cur = conn.cursor()
        cur.execute("SELECT unk_code FROM unknown_cases ORDER BY unk_code DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            return "UNK-000001"
        last = str(row["unk_code"])
        try:
            number = int(last.split("-", 1)[1])
        except Exception:
            number = 0
        return f"UNK-{number + 1:06d}"

    def submit_model_case(self, entry: dict[str, Any], *, decision_id: str = "") -> dict[str, Any]:
        event_id = str(entry.get("event_id") or "").strip()
        if not event_id:
            raise ValueError("event_id is required")
        draft = {
            "queue_id": "",
            "fault_label": str(entry.get("fault_family_label") or entry.get("fault_label") or "Unknown Anomaly"),
            "fault_name": str(entry.get("fault_family_label") or entry.get("fault_label") or "Unknown Anomaly"),
            "fault_family_id": str(entry.get("fault_family_id") or "unknown_anomaly"),
            "fault_family_label": str(entry.get("fault_family_label") or "Unknown Anomaly"),
            "fault_subtype_id": str(entry.get("fault_subtype_id") or ""),
            "fault_subtype_label": str(entry.get("fault_subtype_label") or ""),
            "fault_signature_id": str(entry.get("fault_signature_id") or ""),
            "fault_signature_text": str(entry.get("fault_signature_text") or ""),
            "known_status": str(entry.get("known_status") or "unknown"),
            "similar_family_candidates": entry.get("similar_family_candidates") if isinstance(entry.get("similar_family_candidates"), list) else [],
            "review_required": bool(entry.get("review_required", True)),
            "asset_class": str(entry.get("asset_class") or "generic"),
            "asset_norm_scope": str(entry.get("asset_norm_scope") or "global"),
            "asset_norm_confidence": float(entry.get("asset_norm_confidence") or 0.0),
            "norm_source": str(entry.get("norm_source") or "global"),
            "candidate_dimensions": entry.get("candidate_dimensions") if isinstance(entry.get("candidate_dimensions"), list) else [],
            "observing_dimensions": entry.get("observing_dimensions") if isinstance(entry.get("observing_dimensions"), list) else [],
            "provisional_dimensions": entry.get("provisional_dimensions") if isinstance(entry.get("provisional_dimensions"), list) else [],
            "asset_critical_dimensions": entry.get("asset_critical_dimensions") if isinstance(entry.get("asset_critical_dimensions"), list) else [],
            "dimension_importance_by_asset": entry.get("dimension_importance_by_asset") if isinstance(entry.get("dimension_importance_by_asset"), dict) else {},
            "dimension_lifecycle_status": entry.get("dimension_lifecycle_status") if isinstance(entry.get("dimension_lifecycle_status"), dict) else {},
            "severity": str(entry.get("status") or "UNKNOWN").lower(),
            "mission_phase": str(entry.get("mission_phase") or "unknown"),
            "source_layer": "model",
            "source_layers": ["model"],
            "decision_ids": [decision_id] if decision_id else [],
            "reason": str(entry.get("unknown_reason") or entry.get("reason") or "low_confidence_review"),
            "root_causes": [str(entry.get("fault_subtype_label") or entry.get("root_cause_label") or "Unknown Anomaly")],
            "symptoms": [],
            "solution_suggestions": [str(entry.get("procedure_ref") or "PROC-GENERAL-DIAGNOSTIC-999")],
            "citations": [str(entry.get("procedure_ref") or "PROC-GENERAL-DIAGNOSTIC-999")],
            "quality_flags": ["model_low_confidence_review"],
            "quality_score": float(entry.get("root_cause_confidence") or 0.0),
        }
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM unknown_cases WHERE source_event_id = ? ORDER BY created_at DESC LIMIT 1", (event_id,))
            existing = cur.fetchone()
            now = _now_iso()
            if existing:
                existing_draft = _load_json(existing["draft_json"], default={})
                if not isinstance(existing_draft, dict):
                    existing_draft = {}
                existing_decision_ids = _load_json(existing["decision_ids_json"], default=[])
                merged_decision_ids = list(
                    dict.fromkeys(
                        [str(item) for item in existing_decision_ids if str(item).strip()]
                        + ([decision_id] if decision_id else [])
                    )
                )
                existing_layers = _load_json(existing["source_layers_json"], default=[])
                merged_layers = list(dict.fromkeys([str(item) for item in existing_layers if str(item).strip()] + ["model"]))
                existing_draft.update({k: v for k, v in draft.items() if v not in ("", [], None)})
                existing_draft["source_layers"] = merged_layers
                existing_draft["decision_ids"] = merged_decision_ids
                conn.execute(
                    """
                    UPDATE unknown_cases
                    SET updated_at = ?, reason = ?, draft_json = ?, event_json = ?, report_json = ?,
                        source_layer = ?, source_layers_json = ?, fault_family_id = ?, fault_family_label = ?,
                        fault_subtype_id = ?, fault_subtype_label = ?, fault_signature_id = ?, fault_signature_text = ?,
                        known_status = ?, similar_family_candidates_json = ?, review_required = ?, decision_ids_json = ?,
                        asset_class = ?, asset_norm_scope = ?, asset_norm_confidence = ?, norm_source = ?,
                        candidate_dimensions_json = ?, observing_dimensions_json = ?, provisional_dimensions_json = ?,
                        asset_critical_dimensions_json = ?, dimension_importance_by_asset_json = ?, dimension_lifecycle_status_json = ?
                    WHERE unk_code = ?
                    """,
                    (
                        now,
                        draft["reason"],
                        json.dumps(existing_draft, ensure_ascii=True),
                        json.dumps(entry, ensure_ascii=True),
                        json.dumps({}, ensure_ascii=True),
                        "model",
                        json.dumps(merged_layers, ensure_ascii=True),
                        draft["fault_family_id"],
                        draft["fault_family_label"],
                        draft["fault_subtype_id"],
                        draft["fault_subtype_label"],
                        draft["fault_signature_id"],
                        draft["fault_signature_text"],
                        draft["known_status"],
                        json.dumps(draft["similar_family_candidates"], ensure_ascii=True),
                        1 if draft["review_required"] else 0,
                        json.dumps(merged_decision_ids, ensure_ascii=True),
                        draft["asset_class"],
                        draft["asset_norm_scope"],
                        draft["asset_norm_confidence"],
                        draft["norm_source"],
                        json.dumps(draft["candidate_dimensions"], ensure_ascii=True),
                        json.dumps(draft["observing_dimensions"], ensure_ascii=True),
                        json.dumps(draft["provisional_dimensions"], ensure_ascii=True),
                        json.dumps(draft["asset_critical_dimensions"], ensure_ascii=True),
                        json.dumps(draft["dimension_importance_by_asset"], ensure_ascii=True),
                        json.dumps(draft["dimension_lifecycle_status"], ensure_ascii=True),
                        existing["unk_code"],
                    ),
                )
                conn.commit()
                return {"queue_id": existing["unk_code"], "unk_code": existing["unk_code"], "source_event_id": event_id}

            unk_code = self._next_code(conn)
            draft["queue_id"] = unk_code
            conn.execute(
                """
                INSERT INTO unknown_cases (
                  unk_code, status, created_at, updated_at, source_event_id, reason,
                  draft_json, approved_json, event_json, report_json, reviewer, review_notes,
                  source_layer, source_layers_json, fault_family_id, fault_family_label,
                  fault_subtype_id, fault_subtype_label, fault_signature_id, fault_signature_text,
                  known_status, similar_family_candidates_json, review_required, decision_ids_json, review_action,
                  asset_class, asset_norm_scope, asset_norm_confidence, norm_source,
                  candidate_dimensions_json, observing_dimensions_json, provisional_dimensions_json,
                  asset_critical_dimensions_json, dimension_importance_by_asset_json, dimension_lifecycle_status_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    unk_code,
                    "pending_review",
                    now,
                    now,
                    event_id,
                    draft["reason"],
                    json.dumps(draft, ensure_ascii=True),
                    None,
                    json.dumps(entry, ensure_ascii=True),
                    json.dumps({}, ensure_ascii=True),
                    None,
                    None,
                    "model",
                    json.dumps(["model"], ensure_ascii=True),
                    draft["fault_family_id"],
                    draft["fault_family_label"],
                    draft["fault_subtype_id"],
                    draft["fault_subtype_label"],
                    draft["fault_signature_id"],
                    draft["fault_signature_text"],
                    draft["known_status"],
                    json.dumps(draft["similar_family_candidates"], ensure_ascii=True),
                    1 if draft["review_required"] else 0,
                    json.dumps(draft["decision_ids"], ensure_ascii=True),
                    "",
                    draft["asset_class"],
                    draft["asset_norm_scope"],
                    draft["asset_norm_confidence"],
                    draft["norm_source"],
                    json.dumps(draft["candidate_dimensions"], ensure_ascii=True),
                    json.dumps(draft["observing_dimensions"], ensure_ascii=True),
                    json.dumps(draft["provisional_dimensions"], ensure_ascii=True),
                    json.dumps(draft["asset_critical_dimensions"], ensure_ascii=True),
                    json.dumps(draft["dimension_importance_by_asset"], ensure_ascii=True),
                    json.dumps(draft["dimension_lifecycle_status"], ensure_ascii=True),
                ),
            )
            conn.commit()
            return {"queue_id": unk_code, "unk_code": unk_code, "source_event_id": event_id}
        finally:
            conn.close()
