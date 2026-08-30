from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


LEDGER_SCHEMA_VERSION = "1.0"


def _stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _safe_zoneinfo(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except Exception:
        return ZoneInfo("UTC")


def normalize_iso_utc(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except Exception:
        return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def to_local_iso(iso_utc: str, timezone_name: str) -> str:
    parsed = datetime.fromisoformat(normalize_iso_utc(iso_utc).replace("Z", "+00:00"))
    return parsed.astimezone(_safe_zoneinfo(timezone_name)).replace(microsecond=0).isoformat()


def _clip_text(value: str, *, limit: int = 360) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def build_model_decision_record(
    *,
    resp: dict[str, Any],
    timezone_name: str = "Europe/Istanbul",
    linked_decision_ids: list[str] | None = None,
) -> dict[str, Any]:
    diagnostics = resp.get("diagnostics") if isinstance(resp.get("diagnostics"), dict) else {}
    metrics = resp.get("metrics") if isinstance(resp.get("metrics"), dict) else {}
    meta = resp.get("meta") if isinstance(resp.get("meta"), dict) else {}
    event_id = str(diagnostics.get("event_id") or f"evt-{int(time.time() * 1000)}")
    timestamp_utc = normalize_iso_utc(diagnostics.get("timestamp") or meta.get("timestamp"))
    severity = str(diagnostics.get("classifier_severity") or diagnostics.get("status") or "UNKNOWN").strip().lower()
    severity = "critical" if severity == "critical" else ("warning" if severity == "warning" else "info")
    status = str(diagnostics.get("status") or "UNKNOWN")
    risk_class = "high" if severity == "critical" else ("medium" if severity == "warning" else "low")
    asset_id = str(meta.get("asset_id") or "")
    module_id = str(meta.get("module_id") or asset_id or "")
    site_id = str(meta.get("site_id") or "")
    primary_dim = str(diagnostics.get("primary_fault_dimension") or "unknown_sensor")
    top3 = diagnostics.get("top3_candidates") if isinstance(diagnostics.get("top3_candidates"), list) else []
    candidate_dimensions = metrics.get("candidate_dimensions") if isinstance(metrics.get("candidate_dimensions"), list) else []
    observing_dimensions = metrics.get("observing_dimensions") if isinstance(metrics.get("observing_dimensions"), list) else []
    provisional_dimensions = metrics.get("provisional_dimensions") if isinstance(metrics.get("provisional_dimensions"), list) else []
    asset_critical_dimensions = metrics.get("asset_critical_dimensions") if isinstance(metrics.get("asset_critical_dimensions"), list) else []
    dimension_importance_by_asset = (
        metrics.get("dimension_importance_by_asset") if isinstance(metrics.get("dimension_importance_by_asset"), dict) else {}
    )
    dimension_lifecycle_status = (
        metrics.get("dimension_lifecycle_status") if isinstance(metrics.get("dimension_lifecycle_status"), dict) else {}
    )
    asset_norm_scope = str(metrics.get("asset_norm_scope") or meta.get("asset_norm_scope") or "global")
    asset_norm_confidence = float(metrics.get("asset_norm_confidence") or meta.get("asset_norm_confidence") or 0.0)
    asset_class = str(metrics.get("asset_class") or meta.get("asset_class") or "generic")
    norm_source = str(metrics.get("norm_source") or meta.get("norm_source") or "global")
    human_summary = _clip_text(
        f"{module_id or asset_id or event_id}: Model {diagnostics.get('fault_family_label') or diagnostics.get('fault_label') or 'Unknown Anomaly'} için {severity} seviyesinde anomali işaretledi. "
        f"Durum={status}. Alt tip={diagnostics.get('fault_subtype_label') or diagnostics.get('root_cause_label') or 'UNKNOWN'}. "
        f"Güven={float(diagnostics.get('root_cause_confidence') or 0.0):.2f}.",
    )
    technical_summary = _clip_text(
        " | ".join(
            [
                f"event_id={event_id}",
                f"status={status}",
                f"severity={severity}",
                f"primary_dimension={primary_dim}",
                f"coverage={float(metrics.get('coverage_score') or metrics.get('observation_coverage') or 0.0):.3f}",
                f"missing={','.join(metrics.get('missing_dimensions') or []) or 'none'}",
                f"candidate_dimensions={','.join(candidate_dimensions) or 'none'}",
                f"asset_norm_scope={asset_norm_scope}",
                f"fault_family={diagnostics.get('fault_family_id') or 'unknown_anomaly'}",
                f"fault_subtype={diagnostics.get('fault_subtype_id') or 'none'}",
                f"model_version={meta.get('model_version') or meta.get('calibrator_model_version') or 'unknown'}",
            ]
        ),
        limit=600,
    )
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "decision_id": f"model-{event_id}-diagnose-{int(time.time() * 1000)}",
        "event_id": event_id,
        "timestamp_utc": timestamp_utc,
        "timestamp_local": to_local_iso(timestamp_utc, timezone_name),
        "asset_id": asset_id,
        "module_id": module_id,
        "site_id": site_id,
        "source_layer": "model",
        "status": status,
        "severity": severity,
        "risk_class": risk_class,
        "decision_type": "diagnose",
        "human_summary": human_summary,
        "technical_summary": technical_summary,
        "diagnostics_snapshot": {
            "fault_label": str(diagnostics.get("fault_label") or diagnostics.get("fault_family_label") or ""),
            "fault_family_id": str(diagnostics.get("fault_family_id") or ""),
            "fault_family_label": str(diagnostics.get("fault_family_label") or ""),
            "fault_subtype_id": str(diagnostics.get("fault_subtype_id") or ""),
            "fault_subtype_label": str(diagnostics.get("fault_subtype_label") or ""),
            "fault_signature_id": str(diagnostics.get("fault_signature_id") or ""),
            "fault_signature_text": str(diagnostics.get("fault_signature_text") or diagnostics.get("fault_descriptor") or ""),
            "known_status": str(diagnostics.get("known_status") or "unknown"),
            "similar_family_candidates": diagnostics.get("similar_family_candidates") if isinstance(diagnostics.get("similar_family_candidates"), list) else [],
            "failure_type": str(diagnostics.get("failure_type") or ""),
            "fault_name": str(diagnostics.get("fault_name") or diagnostics.get("fault_descriptor") or ""),
            "primary_fault_dimension": primary_dim,
            "root_cause_label": str(diagnostics.get("root_cause_label") or ""),
            "root_cause_confidence": float(diagnostics.get("root_cause_confidence") or 0.0),
            "top3_candidates": top3,
            "ttf_state": str(diagnostics.get("ttf_state") or "unknown"),
            "ttf_seconds": diagnostics.get("ttf_seconds"),
            "ttf_confidence": float(diagnostics.get("ttf_confidence") or 0.0),
        },
        "evidence_snapshot": {
            "active_dimensions": metrics.get("active_dimensions") if isinstance(metrics.get("active_dimensions"), list) else [],
            "core_active_dimensions": metrics.get("core_active_dimensions") if isinstance(metrics.get("core_active_dimensions"), list) else [],
            "aux_active_dimensions": metrics.get("aux_active_dimensions") if isinstance(metrics.get("aux_active_dimensions"), list) else [],
            "missing_dimensions": metrics.get("missing_dimensions") if isinstance(metrics.get("missing_dimensions"), list) else [],
            "candidate_dimensions": candidate_dimensions,
            "observing_dimensions": observing_dimensions,
            "provisional_dimensions": provisional_dimensions,
            "asset_critical_dimensions": asset_critical_dimensions,
            "asset_norm_scope": asset_norm_scope,
            "asset_norm_confidence": asset_norm_confidence,
            "asset_class": asset_class,
            "norm_source": norm_source,
            "dimension_importance_by_asset": dimension_importance_by_asset,
            "dimension_lifecycle_status": dimension_lifecycle_status,
        },
        "advisory_snapshot": {},
        "trace_snapshot": {
            "selected_path_summary": str(diagnostics.get("selected_path_summary") or ""),
            "relation_violations": diagnostics.get("relation_violations") if isinstance(diagnostics.get("relation_violations"), list) else [],
            "rule_hits": diagnostics.get("rule_hits") if isinstance(diagnostics.get("rule_hits"), list) else [],
            "prototype_hits": diagnostics.get("prototype_hits") if isinstance(diagnostics.get("prototype_hits"), list) else [],
        },
        "model_version": str(meta.get("model_version") or meta.get("calibrator_model_version") or "unknown"),
        "prompt_template_version": "",
        "constitution_version": "",
        "dataset_profile": str(meta.get("data_policy") or ""),
        "coverage_score": float(metrics.get("coverage_score") or metrics.get("observation_coverage") or 0.0),
        "missing_dimensions": metrics.get("missing_dimensions") if isinstance(metrics.get("missing_dimensions"), list) else [],
        "quality_flags": [str(item) for item in metrics.get("quality_flags", [])] if isinstance(metrics.get("quality_flags"), list) else [],
        "citations": [],
        "review_state": "not_required",
        "approval_state": "not_required",
        "linked_decision_ids": list(linked_decision_ids or []),
        "linked_review_queue_ids": [],
        "action_requested": False,
        "action_allowed": False,
        "action_executed": False,
        "action_block_reason": "",
        "requires_engineer_approval": False,
        "approved_by": "",
        "approved_at": "",
        "execution_target": "",
        "execution_result": "",
        "hash_prev": "",
        "hash_self": "",
    }


class DecisionLedgerStore:
    def __init__(
        self,
        *,
        sqlite_path: str | Path,
        jsonl_path: str | Path | None = None,
        retention_days: int = 365,
    ):
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.retention_days = max(30, int(retention_days))
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS decision_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id TEXT NOT NULL UNIQUE,
                    event_id TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    timestamp_local TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    module_id TEXT NOT NULL,
                    site_id TEXT NOT NULL,
                    source_layer TEXT NOT NULL,
                    status TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    risk_class TEXT NOT NULL,
                    decision_type TEXT NOT NULL,
                    human_summary TEXT NOT NULL,
                    technical_summary TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    prompt_template_version TEXT NOT NULL,
                    constitution_version TEXT NOT NULL,
                    dataset_profile TEXT NOT NULL,
                    coverage_score REAL NOT NULL,
                    missing_dimensions_json TEXT NOT NULL,
                    quality_flags_json TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    review_state TEXT NOT NULL,
                    approval_state TEXT NOT NULL,
                    linked_decision_ids_json TEXT NOT NULL,
                    hash_prev TEXT NOT NULL,
                    hash_self TEXT NOT NULL,
                    raw_record_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_model_decision_event_id ON decision_ledger(event_id);
                CREATE INDEX IF NOT EXISTS idx_model_decision_timestamp ON decision_ledger(timestamp_utc);
                CREATE INDEX IF NOT EXISTS idx_model_decision_module ON decision_ledger(module_id);
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _latest_hash(self, conn: sqlite3.Connection) -> str:
        row = conn.execute("SELECT hash_self FROM decision_ledger ORDER BY id DESC LIMIT 1").fetchone()
        return str(row["hash_self"]) if row and row["hash_self"] else ""

    def _hash_record(self, record: dict[str, Any], hash_prev: str) -> str:
        payload = dict(record)
        payload["hash_prev"] = hash_prev
        payload["hash_self"] = ""
        return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()

    def _decision_id_exists(self, conn: sqlite3.Connection, decision_id: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM decision_ledger WHERE decision_id = ? LIMIT 1",
            (decision_id,),
        ).fetchone()
        return row is not None

    def _ensure_unique_decision_id(self, conn: sqlite3.Connection, record: dict[str, Any]) -> None:
        base_decision_id = str(record.get("decision_id") or "").strip()
        if not base_decision_id:
            base_decision_id = f"decision-{int(time.time() * 1000)}"
        if not self._decision_id_exists(conn, base_decision_id):
            record["decision_id"] = base_decision_id
            return
        linked_ids = [str(item) for item in (record.get("linked_decision_ids") or []) if str(item)]
        if base_decision_id not in linked_ids:
            linked_ids.append(base_decision_id)
        counter = 2
        candidate = f"{base_decision_id}-r{counter}"
        while self._decision_id_exists(conn, candidate):
            counter += 1
            candidate = f"{base_decision_id}-r{counter}"
        record["decision_id"] = candidate
        record["linked_decision_ids"] = linked_ids

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        enriched = json.loads(json.dumps(record, ensure_ascii=True))
        conn = self._connect()
        try:
            self._ensure_unique_decision_id(conn, enriched)
            hash_prev = self._latest_hash(conn)
            hash_self = self._hash_record(enriched, hash_prev)
            enriched["hash_prev"] = hash_prev
            enriched["hash_self"] = hash_self
            now_ts = time.time()
            conn.execute(
                """
                INSERT INTO decision_ledger(
                    decision_id, event_id, timestamp_utc, timestamp_local, asset_id, module_id, site_id,
                    source_layer, status, severity, risk_class, decision_type, human_summary, technical_summary,
                    model_version, prompt_template_version, constitution_version, dataset_profile, coverage_score,
                    missing_dimensions_json, quality_flags_json, citations_json, review_state, approval_state,
                    linked_decision_ids_json, hash_prev, hash_self, raw_record_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(enriched.get("decision_id") or ""),
                    str(enriched.get("event_id") or ""),
                    str(enriched.get("timestamp_utc") or ""),
                    str(enriched.get("timestamp_local") or ""),
                    str(enriched.get("asset_id") or ""),
                    str(enriched.get("module_id") or ""),
                    str(enriched.get("site_id") or ""),
                    str(enriched.get("source_layer") or ""),
                    str(enriched.get("status") or ""),
                    str(enriched.get("severity") or ""),
                    str(enriched.get("risk_class") or ""),
                    str(enriched.get("decision_type") or ""),
                    str(enriched.get("human_summary") or ""),
                    str(enriched.get("technical_summary") or ""),
                    str(enriched.get("model_version") or ""),
                    str(enriched.get("prompt_template_version") or ""),
                    str(enriched.get("constitution_version") or ""),
                    str(enriched.get("dataset_profile") or ""),
                    float(enriched.get("coverage_score") or 0.0),
                    json.dumps(enriched.get("missing_dimensions") or [], ensure_ascii=True),
                    json.dumps(enriched.get("quality_flags") or [], ensure_ascii=True),
                    json.dumps(enriched.get("citations") or [], ensure_ascii=True),
                    str(enriched.get("review_state") or ""),
                    str(enriched.get("approval_state") or ""),
                    json.dumps(enriched.get("linked_decision_ids") or [], ensure_ascii=True),
                    hash_prev,
                    hash_self,
                    json.dumps(enriched, ensure_ascii=True),
                    now_ts,
                ),
            )
            cutoff = now_ts - float(self.retention_days * 24 * 3600)
            conn.execute("DELETE FROM decision_ledger WHERE created_at < ?", (cutoff,))
            conn.commit()
        finally:
            conn.close()
        if self.jsonl_path is not None:
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(enriched, ensure_ascii=True) + "\n")
        return enriched
