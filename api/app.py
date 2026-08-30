from __future__ import annotations

import copy
import glob
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import pickle
import queue
import re
import shutil
import socket
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

_BOOTSTRAP_CAUREN_CORE_RUNTIME = str(os.getenv("CAUREN_RUNTIME_MODE", "cauren_core")).strip().lower() in {
    "cauren",
    "cauren_core",
    "core_agents",
    "civil_agent",
}

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

try:
    import psutil
except ModuleNotFoundError:
    psutil = None

try:
    import torch
except ModuleNotFoundError:
    torch = None

HierarchicalRootCauseEngine = Any
extract_feature_vector = None
ComponentGraph = None
CALIBRATION_MODES = ("safe", "aggressive")
visualize_wavescan_panel = None
CANONICAL_BETA25_DATA_DIR = Path("runtime_outputs/civil_dataset")

from core_dimensions import (
    AUX_DIMENSIONS,
    CANONICAL_DIMENSIONS,
    CANONICAL_DIMENSION_ALIASES,
    CANONICAL_DIMENSION_INDEX,
    CANONICAL_DIMENSION_RANGES,
    CANONICAL_DIMENSION_UNITS,
    CORE_DIMENSIONS,
    LEGACY_CORE_DIMENSIONS,
    MODEL_INPUT_DIMENSIONS,
    canonical_dimension_name,
    dimension_class,
)
try:
    from api.fault_export import FaultDatasetExporter
except ModuleNotFoundError:
    # Fallback for direct script execution: python api/app.py
    from fault_export import FaultDatasetExporter

try:
    from api.memory_service import CONTEXT_DIMENSIONS, MEMORY_DIMENSIONS, SharedMemoryService
except ModuleNotFoundError:
    from memory_service import CONTEXT_DIMENSIONS, MEMORY_DIMENSIONS, SharedMemoryService

try:
    from api.asset_identity import ASSET_ID_PATTERN, AssetIdentityResolver, AssetResolution
except ModuleNotFoundError:
    from asset_identity import ASSET_ID_PATTERN, AssetIdentityResolver, AssetResolution

try:
    from api.root_cause_release import resolve_release_bundle, serialize_release_bundle
except ModuleNotFoundError:
    from root_cause_release import resolve_release_bundle, serialize_release_bundle

try:
    from api.decision_ledger import DecisionLedgerStore, build_model_decision_record
except ModuleNotFoundError:
    from decision_ledger import DecisionLedgerStore, build_model_decision_record

try:
    from api.fault_ontology import canonicalize_fault
except ModuleNotFoundError:
    from fault_ontology import canonicalize_fault

try:
    from api.canonical_review_store import CanonicalReviewStore
except ModuleNotFoundError:
    from canonical_review_store import CanonicalReviewStore

try:
    from api.asset_norms import AssetNormStore, AssetNormSummary
except ModuleNotFoundError:
    from asset_norms import AssetNormStore, AssetNormSummary

try:
    from api.operational_memory import OperationalMemoryStore
except ModuleNotFoundError:
    from operational_memory import OperationalMemoryStore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cauren_civil_api")

app = FastAPI(
    title="Cauren Civil Diagnostics API",
    description="Civil engineering diagnostics and decision-support runtime.",
)

PUBLIC_PATHS = {
    "/health",
    "/health/liveness",
    "/health/readiness",
}
def _security_enabled() -> bool:
    return str(os.getenv("GOV_PILOT_SECURITY_ENABLED", "true")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _configured_auth_tokens() -> tuple[str, ...]:
    # No baked-in fallback token: this is an open-source codebase, so any
    # hardcoded string here is public knowledge and would let anyone
    # authenticate against a deployment that forgot to set its own token.
    # With security enabled and nothing configured, every authenticated
    # route fails closed (see the middleware below) rather than accepting
    # a well-known default.
    raw = os.getenv("GOV_PILOT_API_TOKENS") or os.getenv("GOV_PILOT_API_TOKEN") or ""
    return tuple(token.strip() for token in raw.split(",") if token.strip())


if _security_enabled() and not _configured_auth_tokens():
    logger.warning(
        "{\"event\": \"gov_pilot_security_no_tokens_configured\", \"detail\": "
        "\"GOV_PILOT_SECURITY_ENABLED is on but no GOV_PILOT_API_TOKEN(S) is set; "
        "every authenticated request will be rejected until a token is configured.\"}"
    )


def _client_ip_allowed(request: Request) -> bool:
    allowlist = str(os.getenv("GOV_PILOT_ALLOWED_CIDRS", "")).strip()
    if not allowlist:
        return True
    raw_ip = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
    if not raw_ip and request.client is not None:
        raw_ip = request.client.host
    try:
        client_ip = ipaddress.ip_address(raw_ip)
    except ValueError:
        return False
    for item in allowlist.split(","):
        cidr = item.strip()
        if not cidr:
            continue
        try:
            if client_ip in ipaddress.ip_network(cidr, strict=False):
                return True
        except ValueError:
            continue
    return False


def _auth_role(request: Request) -> str:
    return str(request.headers.get("x-cauren-role") or request.headers.get("x-gov-role") or "operator").strip().lower()


def _has_valid_token(request: Request) -> bool:
    header = str(request.headers.get("authorization") or "").strip()
    token = ""
    if header.lower().startswith("bearer "):
        token = header.split(" ", 1)[1].strip()
    if not token:
        token = str(request.headers.get("x-cauren-api-key") or request.headers.get("x-api-key") or "").strip()
    return any(hmac.compare_digest(token, configured) for configured in _configured_auth_tokens())


def _role_allowed_for_path(role: str, path: str, method: str) -> bool:
    if role in {"admin", "government_admin", "operator"}:
        return True
    if role in {"viewer", "auditor"}:
        return method == "GET" and not path.endswith("/review") and "/ops/" not in path
    return False


@app.middleware("http")
async def government_pilot_security_gate(request: Request, call_next):
    path = request.url.path
    if _security_enabled() and path not in PUBLIC_PATHS:
        if not _client_ip_allowed(request):
            logger.warning("gov_pilot_auth_rejected reason=ip path=%s", path)
            return JSONResponse(status_code=403, content={"detail": "client network is not allowed"})
        if not _has_valid_token(request):
            logger.warning("gov_pilot_auth_rejected reason=token path=%s", path)
            return JSONResponse(status_code=401, content={"detail": "government pilot identity required"})
        role = _auth_role(request)
        if not _role_allowed_for_path(role, path, request.method):
            logger.warning("gov_pilot_auth_rejected reason=role role=%s path=%s", role, path)
            return JSONResponse(status_code=403, content={"detail": "role is not allowed for this endpoint"})
        content_length = int(request.headers.get("content-length") or 0)
        max_bytes = int(os.getenv("GOV_PILOT_MAX_BODY_BYTES", str(25 * 1024 * 1024)))
        if content_length > max_bytes:
            logger.warning("gov_pilot_auth_rejected reason=body_too_large path=%s bytes=%s", path, content_length)
            return JSONResponse(status_code=413, content={"detail": "request body is too large"})
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Cache-Control", "no-store")
    return response

DIMENSIONS = list(MODEL_INPUT_DIMENSIONS)
CANONICAL_DIMENSION_LIST = list(CANONICAL_DIMENSIONS)
DIMENSION_RANGES = dict(CANONICAL_DIMENSION_RANGES)
CORE_DIMENSION_SET = set(CORE_DIMENSIONS)
AUX_DIMENSION_SET = set(AUX_DIMENSIONS)

SEQ_MIN = 16
SEQ_MAX = 4096
# Civil engineering data doesn't arrive at industrial-telemetry rates: a
# building's real observation cadence is often one permit review, one
# inspection, or one CBS record update per quarter or per year, not per
# second. The floor here is ~1 sample per 50 years -- generous enough for
# any realistic civil reporting cycle -- while still rejecting zero or
# negative values. The ceiling stays high enough for structural-health
# accelerometer/vibration sensors (up to ~1 kHz).
SAMPLING_HZ_MIN = 1e-9
SAMPLING_HZ_MAX = 1000.0
DATA_POLICY_LABEL = "250k_beta25_only"

DIAGNOSTIC_REQUIRED_FIELDS = (
    "event_id",
    "timestamp",
    "status",
    "classifier_severity",
    "fault_label",
    "fault_family_id",
    "fault_family_label",
    "fault_subtype_id",
    "fault_subtype_label",
    "fault_signature_id",
    "fault_signature_text",
    "known_status",
    "similar_family_candidates",
    "review_required",
    "root_cause_label",
    "root_cause_text",
    "root_cause_confidence",
    "failure_type",
    "primary_fault_dimension",
    "trend_description",
    "severity_sigma",
    "procedure_ref",
    "solution_recommendation",
    "report_text",
    "kb_version",
    "top3_candidates",
    "secondary_diagnosis",
    "fault_descriptor",
    "ttf_seconds",
    "ttf_state",
    "ttf_confidence",
)

FAILURE_TYPE_CANONICAL_MAP = {
    "normal": "normal",
    "physics violation": "physics_violation",
    "high noise": "high_noise",
    "solar storm event": "solar_storm_event",
    "structural failure": "structural_failure",
    "ghost thermal peak": "thermal_impact",
    "insidious drift": "insidious_drift",
    "sudden failure": "sudden_failure",
}

TTF_STATES = {"stable", "recovering", "watch", "soon", "imminent", "collapse", "unknown"}
TRACE_TTL_SECONDS = 365 * 24 * 3600
TRACE_MAX_ITEMS = 200000
ARTIFACT_DEDUP_WINDOW_SEC_DEFAULT = 60.0
ARTIFACT_TTL_SEC_DEFAULT = 1800.0
ARTIFACT_FULL_RENDER_EVERY_N_DEFAULT = 10


class CalibrateIn(BaseModel):
    asset_id: Optional[str] = None
    site_id: Optional[str] = None
    line_id: Optional[str] = None
    machine_id: Optional[str] = None
    agent_id: Optional[str] = None
    sector: Optional[str] = None
    client_id: Optional[str] = None
    site_context: Optional[Dict[str, Any]] = None
    forecast_context: Optional[Dict[str, Any]] = None
    control_mode: Optional[str] = "guarded_auto"
    sensor_schema_version: Optional[str] = None
    timestamp: float
    mission_phase: Optional[str] = "unknown"
    seq_len: int
    sampling_hz: float
    runtime_mode: Optional[str] = None
    sensors: Optional[List[Dict[str, Any]]] = None
    ingress_tag: Optional[str] = None
    ingress_tags: Optional[List[str]] = None
    sector_hint: Optional[str] = None
    sector_hint_confidence: Optional[float] = None
    source_metadata: Optional[Dict[str, Any]] = None


class DiagnoseIn(BaseModel):
    asset_id: Optional[str] = None
    site_id: Optional[str] = None
    line_id: Optional[str] = None
    machine_id: Optional[str] = None
    agent_id: Optional[str] = None
    sector: Optional[str] = None
    client_id: Optional[str] = None
    site_context: Optional[Dict[str, Any]] = None
    forecast_context: Optional[Dict[str, Any]] = None
    control_mode: Optional[str] = "guarded_auto"
    sensor_schema_version: Optional[str] = None
    timestamp: float
    mission_phase: Optional[str] = "unknown"
    seq_len: int
    sampling_hz: float
    runtime_mode: Optional[str] = None
    sensors: Optional[List[Dict[str, Any]]] = None
    ingress_tag: Optional[str] = None
    ingress_tags: Optional[List[str]] = None
    sector_hint: Optional[str] = None
    sector_hint_confidence: Optional[float] = None
    source_metadata: Optional[Dict[str, Any]] = None
    building_id: Optional[str] = None
    geometry: Optional[Dict[str, Any]] = None
    address: Optional[Dict[str, Any]] = None
    administrative_unit: Optional[Dict[str, Any]] = None
    project_permit: Optional[Dict[str, Any]] = None
    construction_status: Optional[Dict[str, Any]] = None
    infrastructure_connections: Optional[Dict[str, Any]] = None
    risk_assessments: Optional[Dict[str, Any]] = None
    inspection_findings: Optional[List[Dict[str, Any]]] = None

class MemoryIngestIn(BaseModel):
    asset_id: Optional[str] = None
    site_id: Optional[str] = None
    line_id: Optional[str] = None
    machine_id: Optional[str] = None
    timestamp: float | str
    sensor_data: Optional[List[List[float]]] = None
    sensor_values: Optional[Dict[str, float]] = None
    dimension_names: Optional[List[str]] = None
    primary_dimension_sources: Optional[Dict[str, str]] = None
    primary_internal_pressure_sensor: Optional[str] = None
    status: Optional[str] = "NORMAL"
    include_in_learning: Optional[bool] = None
    correction_data: Optional[List[List[float]]] = None


class AssetProposalReviewIn(BaseModel):
    proposal_id: str
    action: str
    reviewer: str
    notes: Optional[str] = ""


class ArtifactClaimIn(BaseModel):
    worker_id: Optional[str] = "artifact-worker"
    limit: Optional[int] = 1


class ArtifactAckIn(BaseModel):
    claim_id: str
    status: Optional[str] = "ok"  # ok | failed
    rendered_images: Optional[int] = 0
    total_files_written: Optional[int] = 0
    lag_ms: Optional[float] = None
    error: Optional[str] = None


class CBSBuildingRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    building_id: str
    geometry: Optional[Dict[str, Any]] = None
    address: Optional[Dict[str, Any]] = None
    administrative_unit: Optional[Dict[str, Any]] = None
    project_permit: Optional[Dict[str, Any]] = None
    construction_status: Optional[Dict[str, Any]] = None
    infrastructure_connections: Optional[Dict[str, Any]] = None
    cadastral_reference: Optional[Dict[str, Any]] = None
    risk_assessments: Optional[Dict[str, Any]] = None
    inspection_findings: Optional[List[Dict[str, Any]]] = None
    sensors: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class CBSBuildingBatchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idempotency_key: str
    source: Optional[str] = "tucbs"
    records: List[CBSBuildingRecord]


class RuntimeMetrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.requests_total = 0
        self.errors_total = 0
        self.anomalies_total = 0
        self.queue_drops_total = 0
        self.backpressure_rejects_total = 0
        self.traffic_activity_transitions_total = 0
        self.artifact_jobs_rendered_total = 0
        self.artifact_jobs_deduped_total = 0
        self.artifact_jobs_skipped_total = 0
        self.artifact_jobs_failed_total = 0
        self.operational_memory_hits_total = 0
        self.operational_memory_misses_total = 0
        self.operational_memory_safe_fallbacks_total = 0
        self.operational_memory_invalidations_total = 0
        self.latency_ms = deque(maxlen=10000)
        self.hot_path_latency_ms = deque(maxlen=10000)
        self.full_path_latency_ms = deque(maxlen=10000)
        self.disk_write_events = deque(maxlen=10000)  # (ts, count)
        self.request_events = deque(maxlen=20000)  # (ts, success, anomaly)
        self.artifact_lag_events = deque(maxlen=10000)  # (ts, lag_ms)
        self.last_request_ts = 0.0
        self.queue_mode_counts = {
            "normal": 0,
            "warn": 0,
            "degrade": 0,
            "protect": 0,
        }

    def record_request(self, latency_ms: float, success: bool, anomaly: bool):
        with self._lock:
            now_ts = time.time()
            self.requests_total += 1
            if not success:
                self.errors_total += 1
            if anomaly:
                self.anomalies_total += 1
            self.latency_ms.append(latency_ms)
            self.request_events.append((now_ts, bool(success), bool(anomaly)))
            self.last_request_ts = now_ts

    def record_operational_memory_hit(self, *, latency_ms: float | None = None):
        with self._lock:
            self.operational_memory_hits_total += 1
            if latency_ms is not None:
                self.hot_path_latency_ms.append(float(latency_ms))

    def record_operational_memory_miss(self, *, latency_ms: float | None = None):
        with self._lock:
            self.operational_memory_misses_total += 1
            if latency_ms is not None:
                self.full_path_latency_ms.append(float(latency_ms))

    def record_operational_memory_safe_fallback(self):
        with self._lock:
            self.operational_memory_safe_fallbacks_total += 1

    def record_operational_memory_invalidation(self):
        with self._lock:
            self.operational_memory_invalidations_total += 1

    def record_queue_drop(self):
        with self._lock:
            self.queue_drops_total += 1

    def record_backpressure_reject(self):
        with self._lock:
            self.backpressure_rejects_total += 1

    def record_queue_mode_transition(self, mode: str):
        if mode not in self.queue_mode_counts:
            return
        with self._lock:
            self.queue_mode_counts[mode] += 1

    def record_traffic_activity_transition(self):
        with self._lock:
            self.traffic_activity_transitions_total += 1

    def record_disk_writes(self, count: int):
        if count <= 0:
            return
        with self._lock:
            self.disk_write_events.append((time.time(), int(count)))

    def record_artifact_rendered(self, *, lag_ms: Optional[float] = None):
        with self._lock:
            self.artifact_jobs_rendered_total += 1
            if lag_ms is not None and np.isfinite(lag_ms):
                self.artifact_lag_events.append((time.time(), float(lag_ms)))

    def record_artifact_deduped(self):
        with self._lock:
            self.artifact_jobs_deduped_total += 1

    def record_artifact_skipped(self):
        with self._lock:
            self.artifact_jobs_skipped_total += 1

    def record_artifact_failed(self):
        with self._lock:
            self.artifact_jobs_failed_total += 1

    def snapshot(self, report_queue_depth: int, inference_queue_depth: int = 0) -> Dict[str, Any]:
        with self._lock:
            latencies = list(self.latency_ms)
            requests_total = self.requests_total
            errors_total = self.errors_total
            anomalies_total = self.anomalies_total
            queue_drops_total = self.queue_drops_total
            backpressure_rejects_total = self.backpressure_rejects_total
            traffic_activity_transitions_total = self.traffic_activity_transitions_total
            artifact_jobs_rendered_total = self.artifact_jobs_rendered_total
            artifact_jobs_deduped_total = self.artifact_jobs_deduped_total
            artifact_jobs_skipped_total = self.artifact_jobs_skipped_total
            artifact_jobs_failed_total = self.artifact_jobs_failed_total
            operational_memory_hits_total = self.operational_memory_hits_total
            operational_memory_misses_total = self.operational_memory_misses_total
            operational_memory_safe_fallbacks_total = self.operational_memory_safe_fallbacks_total
            operational_memory_invalidations_total = self.operational_memory_invalidations_total
            hot_path_latencies = list(self.hot_path_latency_ms)
            full_path_latencies = list(self.full_path_latency_ms)
            disk_events = list(self.disk_write_events)
            request_events = list(self.request_events)
            artifact_lag_events = list(self.artifact_lag_events)
            queue_mode_counts = dict(self.queue_mode_counts)
            last_request_ts = float(self.last_request_ts)

        p50 = _percentile(latencies, 50)
        p95 = _percentile(latencies, 95)
        hot_path_p50 = _percentile(hot_path_latencies, 50)
        hot_path_p95 = _percentile(hot_path_latencies, 95)
        full_path_p50 = _percentile(full_path_latencies, 50)
        full_path_p95 = _percentile(full_path_latencies, 95)
        error_rate = (errors_total / requests_total) if requests_total else 0.0
        anomaly_rate = (anomalies_total / requests_total) if requests_total else 0.0
        operational_memory_lookup_total = operational_memory_hits_total + operational_memory_misses_total
        operational_memory_hit_rate = (
            operational_memory_hits_total / operational_memory_lookup_total
            if operational_memory_lookup_total
            else 0.0
        )
        operational_memory_safe_fallback_rate = (
            operational_memory_safe_fallbacks_total
            / (operational_memory_hits_total + operational_memory_safe_fallbacks_total)
            if (operational_memory_hits_total + operational_memory_safe_fallbacks_total) > 0
            else 0.0
        )

        now_ts = time.time()
        disk_write_rate = sum(c for ts, c in disk_events if now_ts - ts <= 60.0) / 60.0
        requests_1m = sum(1 for ts, _, _ in request_events if now_ts - ts <= 60.0)
        lag_values = [lag for ts, lag in artifact_lag_events if now_ts - ts <= 1800.0]
        artifact_lag_ms_p95 = _percentile(lag_values, 95) if lag_values else 0.0
        recent_window_sec = 30.0
        receiving_traffic_recently = (now_ts - last_request_ts) <= recent_window_sec

        return {
            "uptime_sec": round(now_ts - self.start_time, 3),
            "requests_total": requests_total,
            "errors_total": errors_total,
            "anomalies_total": anomalies_total,
            "error_rate": round(error_rate, 6),
            "anomaly_rate": round(anomaly_rate, 6),
            "latency_ms_p50": round(p50, 3),
            "latency_ms_p95": round(p95, 3),
            "queue_depth": report_queue_depth,
            "artifact_queue_depth": report_queue_depth,
            "inference_queue_depth": int(inference_queue_depth),
            "queue_drops_total": queue_drops_total,
            "backpressure_rejects_total": backpressure_rejects_total,
            "traffic_activity_transitions_total": traffic_activity_transitions_total,
            "disk_write_rate_per_sec": round(disk_write_rate, 6),
            "requests_1m": int(requests_1m),
            "receiving_traffic_recently": bool(receiving_traffic_recently),
            "artifact_lag_ms_p95": round(float(artifact_lag_ms_p95), 3),
            "artifact_jobs_rendered": int(artifact_jobs_rendered_total),
            "artifact_jobs_deduped": int(artifact_jobs_deduped_total),
            "artifact_jobs_skipped": int(artifact_jobs_skipped_total),
            "artifact_jobs_failed": int(artifact_jobs_failed_total),
            "queue_mode_counts": queue_mode_counts,
            "operational_memory_hit_rate": round(float(operational_memory_hit_rate), 6),
            "operational_memory_safe_fallback_rate": round(float(operational_memory_safe_fallback_rate), 6),
            "operational_memory_invalidations": int(operational_memory_invalidations_total),
            "hot_path_latency_ms_p50": round(float(hot_path_p50), 3),
            "hot_path_latency_ms_p95": round(float(hot_path_p95), 3),
            "full_path_latency_ms_p50": round(float(full_path_p50), 3),
            "full_path_latency_ms_p95": round(float(full_path_p95), 3),
        }


class AssetContext:
    def __init__(self, engine: Any):
        self.engine = engine
        self.lock = threading.RLock()
        self.last_used = time.time()


def _percentile(values: List[float], pct: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    idx = int(round((max(0, min(100, pct)) / 100.0) * (len(ordered) - 1)))
    return float(ordered[idx])


def _path_size_bytes(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += int(item.stat().st_size)
    return total


def _nearest_existing_ancestor(path: Path) -> Path:
    """Walk up from `path` to the closest directory that actually exists.

    `path` itself may not exist yet (e.g. a state-store file on a fresh
    checkout before the first write), which makes `shutil.disk_usage`
    raise `FileNotFoundError`. Any real ancestor is on the same
    filesystem, so its free-space figure is an equally valid readiness
    signal.
    """
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            return Path.cwd()
        candidate = parent
    return candidate


def _runtime_resource_snapshot(app_ref: FastAPI) -> Dict[str, Any]:
    artifact_root_raw = getattr(app_ref.state, "artifact_output_root", None)
    artifact_root = Path(str(artifact_root_raw)).resolve() if artifact_root_raw else None
    state_store_raw = str(getattr(app_ref.state, "state_store_path", "") or "")
    state_store_path = Path(state_store_raw).resolve() if state_store_raw else None
    disk_probe = artifact_root or state_store_path or Path.cwd()
    usage = shutil.disk_usage(_nearest_existing_ancestor(disk_probe))
    if psutil is None:
        rss_bytes = 0
        vms_bytes = 0
        open_files_count = 0
    else:
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        rss_bytes = int(mem.rss)
        vms_bytes = int(mem.vms)
        open_files_count = int(len(proc.open_files()))
    return {
        "process_rss_bytes": rss_bytes,
        "process_vms_bytes": vms_bytes,
        "open_files_count": open_files_count,
        "artifact_dir_size_bytes": _path_size_bytes(artifact_root),
        "state_store_size_bytes": _path_size_bytes(state_store_path),
        "disk_free_gb": round(usage.free / (1024 ** 3), 3),
        "restart_counter": int(getattr(app_ref.state, "restart_counter", 0) or 0),
        "readiness_false_streak": int(getattr(app_ref.state, "readiness_false_streak", 0) or 0),
        "readiness_false_streak_max": int(getattr(app_ref.state, "readiness_false_streak_max", 0) or 0),
    }


def _queue_pressure_snapshot(app_ref: FastAPI) -> Dict[str, Any]:
    queue_ref = getattr(app_ref.state, "report_queue", None)
    depth = int(queue_ref.qsize()) if queue_ref is not None else 0
    maxsize = int(getattr(queue_ref, "maxsize", 0)) if queue_ref is not None else 0
    warn = int(getattr(app_ref.state, "queue_warn_threshold", 1200))
    degrade = int(getattr(app_ref.state, "queue_degrade_threshold", 1600))
    protect = int(getattr(app_ref.state, "queue_protect_threshold", 1900))
    if depth >= protect:
        watermark = "protect"
    elif depth >= degrade:
        watermark = "degrade"
    elif depth >= warn:
        watermark = "warn"
    else:
        watermark = "normal"
    return {
        "queue_depth": depth,
        "artifact_queue_depth": depth,
        "queue_maxsize": maxsize,
        "queue_warn_threshold": warn,
        "queue_degrade_threshold": degrade,
        "queue_protect_threshold": protect,
        "queue_watermark": watermark,
    }


def _update_queue_pressure_state(app_ref: FastAPI, *, context: str) -> Dict[str, Any]:
    snapshot = _queue_pressure_snapshot(app_ref)
    previous = str(getattr(app_ref.state, "queue_watermark", "normal"))
    current = str(snapshot["queue_watermark"])
    if previous != current:
        app_ref.state.queue_watermark = current
        app_ref.state.queue_watermark_last_change_ts = time.time()
        app_ref.state.metrics.record_queue_mode_transition(current)
        logger.warning(
            json.dumps(
                {
                    "event": "queue_watermark_transition",
                    "from": previous,
                    "to": current,
                    "context": context,
                    "queue_depth": snapshot["queue_depth"],
                    "warn_threshold": snapshot["queue_warn_threshold"],
                    "degrade_threshold": snapshot["queue_degrade_threshold"],
                    "protect_threshold": snapshot["queue_protect_threshold"],
                }
            )
        )
    return snapshot


def _enforce_backpressure_or_raise(request: Request, endpoint: str) -> Dict[str, Any]:
    snapshot = _inference_pressure_snapshot(request.app)
    if snapshot["inference_watermark"] == "protect":
        request.app.state.metrics.record_backpressure_reject()
        raise HTTPException(
            status_code=503,
            detail=(
                "Inference protect mode active: request rejected to keep service healthy"
            ),
        )
    return snapshot


def _inference_pressure_snapshot(app_ref: FastAPI) -> Dict[str, Any]:
    queue_ref = getattr(app_ref.state, "inference_queue", None)
    depth = int(queue_ref.qsize()) if queue_ref is not None else 0
    maxsize = int(getattr(queue_ref, "maxsize", 0)) if queue_ref is not None else 0
    warn = int(getattr(app_ref.state, "inference_queue_warn_threshold", 0))
    degrade = int(getattr(app_ref.state, "inference_queue_degrade_threshold", 0))
    protect = int(getattr(app_ref.state, "inference_queue_protect_threshold", 0))
    if depth >= protect and protect > 0:
        watermark = "protect"
    elif depth >= degrade and degrade > 0:
        watermark = "degrade"
    elif depth >= warn and warn > 0:
        watermark = "warn"
    else:
        watermark = "normal"
    return {
        "inference_queue_depth": depth,
        "inference_queue_maxsize": maxsize,
        "inference_queue_warn_threshold": warn,
        "inference_queue_degrade_threshold": degrade,
        "inference_queue_protect_threshold": protect,
        "inference_watermark": watermark,
    }


def _acquire_inference_slot_or_raise(request: Request, endpoint: str) -> str:
    queue_ref = getattr(request.app.state, "inference_queue", None)
    if queue_ref is None:
        return ""
    token = f"{endpoint}:{time.time_ns()}:{uuid.uuid4().hex[:8]}"
    try:
        queue_ref.put_nowait(token)
    except queue.Full:
        request.app.state.metrics.record_backpressure_reject()
        raise HTTPException(
            status_code=503,
            detail="Inference queue saturated: request rejected to keep service healthy",
        )
    return token


def _release_inference_slot(request: Request, token: str) -> None:
    queue_ref = getattr(request.app.state, "inference_queue", None)
    if queue_ref is None or not token:
        return
    try:
        queue_ref.get_nowait()
        queue_ref.task_done()
    except Exception:
        pass


def _resolve_active_traffic_node(app_ref: FastAPI, metrics_snapshot: Optional[Dict[str, Any]] = None) -> str:
    override = str(getattr(app_ref.state, "active_traffic_node_override", "auto")).strip()
    if override and override.lower() not in {"auto", "unknown"}:
        return override
    queue_ref = getattr(app_ref.state, "report_queue", None)
    queue_depth = int(queue_ref.qsize()) if queue_ref is not None else 0
    inf_ref = getattr(app_ref.state, "inference_queue", None)
    inf_depth = int(inf_ref.qsize()) if inf_ref is not None else 0
    snap = metrics_snapshot or app_ref.state.metrics.snapshot(
        report_queue_depth=queue_depth,
        inference_queue_depth=inf_depth,
    )
    if bool(snap.get("receiving_traffic_recently", False)):
        return str(getattr(app_ref.state, "traffic_node", "unknown"))
    return "unknown"


def _update_traffic_activity_state(
    app_ref: FastAPI,
    *,
    metrics_snapshot: Dict[str, Any],
    context: str,
) -> bool:
    current = bool(metrics_snapshot.get("receiving_traffic_recently", False))
    previous = bool(getattr(app_ref.state, "receiving_traffic_recently_flag", False))
    if current != previous:
        app_ref.state.receiving_traffic_recently_flag = current
        app_ref.state.metrics.record_traffic_activity_transition()
        logger.info(
            json.dumps(
                {
                    "event": "traffic_activity_transition",
                    "context": context,
                    "traffic_node": str(getattr(app_ref.state, "traffic_node", "unknown")),
                    "active": current,
                    "requests_1m": int(metrics_snapshot.get("requests_1m", 0)),
                }
            )
        )
    return current


def _sanitize_asset_id(asset_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", asset_id)


def _file_version(path: Path) -> str:
    if not path.exists():
        return "missing"
    st = path.stat()
    h = hashlib.sha256(path.read_bytes()).hexdigest()[:10]
    return f"{path.name}:{int(st.st_mtime)}:{h}"


def _to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _evaluate_root_cause_consistency(
    *,
    artifact_path: Path,
    root_engine: HierarchicalRootCauseEngine,
) -> Dict[str, Any]:
    issues: List[str] = []
    artifact_root_count = int(len(getattr(root_engine, "root_classes", []) or []))
    artifact_fault_count = int(len(getattr(root_engine, "fault_name_classes", []) or []))
    metadata = getattr(root_engine, "model_metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}

    metadata_root_count = _to_int_or_none(metadata.get("root_class_count"))
    metadata_fault_count = _to_int_or_none(metadata.get("fault_class_count"))

    if metadata_root_count is not None and metadata_root_count != artifact_root_count:
        issues.append(
            f"metadata_root_class_count_mismatch:{metadata_root_count}!={artifact_root_count}"
        )
    if metadata_fault_count is not None and metadata_fault_count != artifact_fault_count:
        issues.append(
            f"metadata_fault_class_count_mismatch:{metadata_fault_count}!={artifact_fault_count}"
        )

    metrics_path = artifact_path.with_suffix(".metrics.json")
    metrics_fault_count: int | None = None
    metrics_prototype_count: int | None = None
    if not metrics_path.exists():
        issues.append(f"metrics_missing:{metrics_path}")
    else:
        try:
            metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics_fault_count = _to_int_or_none(
                metrics_payload.get("fault_name_class_count")
            )
            metrics_prototype_count = _to_int_or_none(
                metrics_payload.get("prototype_count")
            )
        except Exception as exc:
            issues.append(f"metrics_parse_failed:{type(exc).__name__}")

    if metrics_fault_count is not None and metrics_fault_count != artifact_fault_count:
        issues.append(
            f"metrics_fault_name_class_count_mismatch:{metrics_fault_count}!={artifact_fault_count}"
        )
    if metrics_prototype_count is not None and metrics_prototype_count != artifact_root_count:
        issues.append(
            f"metrics_prototype_count_mismatch:{metrics_prototype_count}!={artifact_root_count}"
        )

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "artifact_root_count": artifact_root_count,
        "artifact_fault_count": artifact_fault_count,
        "metadata_root_count": metadata_root_count,
        "metadata_fault_count": metadata_fault_count,
        "metrics_fault_name_class_count": metrics_fault_count,
        "metrics_prototype_count": metrics_prototype_count,
    }


def _serialize_analysis(result_obj):
    raw = asdict(result_obj)
    return json.loads(
        json.dumps(raw, default=lambda x: x.value if isinstance(x, Enum) else str(x))
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("%", "").replace("σ", "")
        if not cleaned:
            return default
        try:
            return float(cleaned)
        except ValueError:
            return default
    return default


def _safe_probability(value: Any, default: float = 0.0) -> float:
    raw = _safe_float(value, default=default)
    if raw > 1.0 and raw <= 100.0:
        raw = raw / 100.0
    return max(0.0, min(1.0, raw))


def _to_snake_case(text: Any, *, default: str = "unknown_anomaly") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(text or "").strip().lower()).strip("_")
    return cleaned or default


def _normalize_status_text(value: Any, *, fallback: str = "UNKNOWN") -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    return text.upper()


def _to_iso8601_utc(value: Any) -> str:
    if isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                parsed = parsed.astimezone(timezone.utc)
                return parsed.replace(microsecond=0).isoformat().replace("+00:00", "Z")
            except ValueError:
                try:
                    return _to_iso8601_utc(float(text))
                except ValueError:
                    pass

    dt = datetime.now(tz=timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _event_id_from_timestamp(timestamp_iso: str) -> str:
    try:
        parsed = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
    except ValueError:
        parsed = datetime.now(tz=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    date_part = parsed.strftime("%Y%m%d")
    millis = int(parsed.timestamp() * 1000.0)
    return f"evt-{date_part}-{millis}"


def _normalize_top3_candidates(
    raw_candidates: Any,
    *,
    fallback_label: str,
    fallback_probability: float,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    if isinstance(raw_candidates, list):
        for row in raw_candidates:
            if not isinstance(row, dict):
                continue
            label = str(
                row.get("label")
                or row.get("cause_id")
                or row.get("description")
                or ""
            ).strip()
            if not label:
                continue
            probability = _safe_probability(
                row.get("probability", row.get("confidence", 0.0)),
                default=0.0,
            )
            candidates.append({"label": label, "probability": round(probability, 6)})
    if not candidates:
        label = str(fallback_label or "Unknown Anomaly").strip() or "Unknown Anomaly"
        candidates = [
            {
                "label": label,
                "probability": round(_safe_probability(fallback_probability, default=0.0), 6),
            }
        ]
    return candidates[:3]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalized_ttf_fields(diagnostics: Dict[str, Any]) -> tuple[float | None, str, float]:
    raw_ttf = diagnostics.get("ttf_seconds", diagnostics.get("estimated_time_to_critical"))
    ttf_seconds: float | None = None
    if raw_ttf is not None:
        parsed = _safe_float(raw_ttf, default=float("nan"))
        if np.isfinite(parsed):
            ttf_seconds = -1.0 if parsed < 0 else float(parsed)

    trend_probe = str(diagnostics.get("trend_description") or "").strip().lower()
    provided_state = str(diagnostics.get("ttf_state") or "").strip().lower()

    if provided_state in TTF_STATES:
        ttf_state = provided_state
    elif ttf_seconds is not None:
        if ttf_seconds < 0:
            ttf_state = "collapse"
        elif ttf_seconds <= 60.0:
            ttf_state = "imminent"
        elif ttf_seconds <= 3600.0:
            ttf_state = "soon"
        else:
            ttf_state = "watch"
    elif "recover" in trend_probe:
        ttf_state = "recovering"
    elif "stable" in trend_probe:
        ttf_state = "stable"
    else:
        ttf_state = "unknown"

    raw_conf = diagnostics.get("ttf_confidence")
    if raw_conf is not None:
        ttf_conf = _safe_probability(raw_conf, default=0.0)
    else:
        root_conf = _safe_probability(diagnostics.get("root_cause_confidence"), default=0.0)
        sigma = abs(_safe_float(diagnostics.get("severity_sigma"), default=0.0))
        sigma_boost = min(1.0, sigma / 6.0)
        state_bonus = {
            "collapse": 0.20,
            "imminent": 0.15,
            "soon": 0.10,
            "watch": 0.05,
            "recovering": 0.08,
            "stable": 0.05,
            "unknown": 0.00,
        }.get(ttf_state, 0.0)
        if ttf_state == "unknown":
            ttf_conf = 0.0
        else:
            ttf_conf = _clip01(0.35 + (0.45 * root_conf) + (0.10 * sigma_boost) + state_bonus)
    return ttf_seconds, ttf_state, round(float(ttf_conf), 6)


def _canonical_failure_type(diagnostics: Dict[str, Any]) -> str:
    raw = str(diagnostics.get("failure_type") or "").strip()
    normalized = re.sub(r"\s+", " ", raw.lower())
    base = FAILURE_TYPE_CANONICAL_MAP.get(normalized, _to_snake_case(raw))

    primary_dim = str(diagnostics.get("primary_fault_dimension") or "").strip().lower()
    root_label = str(diagnostics.get("root_cause_label") or "").strip().upper()
    root_text = str(diagnostics.get("root_cause_text") or "").strip().lower()
    trend = str(diagnostics.get("trend_description") or "").strip().lower()
    context_text = f"{root_label} {root_text}"

    if primary_dim in {"external_pressure", "internal_pressure"}:
        if "regulator" in context_text or root_label.startswith("PROP-"):
            return "pressurization_instability"
        if "pressure drop" in context_text or ("drop" in context_text and "pressur" in context_text):
            return "pressure_drop"
        if "surge" in context_text or "spike" in context_text:
            return "pressure_surge"
        if "decreas" in trend and "pressur" in context_text:
            return "pressurization_instability"

    return base


def _apply_canonical_fault_ontology(diagnostics: Dict[str, Any]) -> None:
    canonical_fault = canonicalize_fault(
        root_cause_label=diagnostics.get("root_cause_label"),
        failure_type=diagnostics.get("failure_type"),
        primary_dimension=diagnostics.get("primary_fault_dimension"),
        fault_name=diagnostics.get("fault_name"),
        fault_descriptor=diagnostics.get("fault_descriptor"),
        top3_candidates=diagnostics.get("top3_candidates"),
        is_unknown=bool(diagnostics.get("is_unknown")),
    )
    diagnostics["fault_label"] = canonical_fault.fault_family_label
    diagnostics["fault_family_id"] = canonical_fault.fault_family_id
    diagnostics["fault_family_label"] = canonical_fault.fault_family_label
    diagnostics["fault_subtype_id"] = canonical_fault.fault_subtype_id
    diagnostics["fault_subtype_label"] = canonical_fault.fault_subtype_label
    diagnostics["fault_signature_id"] = canonical_fault.fault_signature_id
    diagnostics["fault_signature_text"] = canonical_fault.fault_signature_text
    diagnostics["known_status"] = canonical_fault.known_status
    diagnostics["similar_family_candidates"] = canonical_fault.similar_family_candidates
    diagnostics["review_required"] = bool(canonical_fault.review_required)


def _normalize_diagnose_response_payload(
    resp: Dict[str, Any],
    *,
    payload_timestamp: Any,
    asset_id: str,
    mission_phase: Optional[str],
) -> None:
    diagnostics = resp.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}

    metrics = resp.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}

    meta = resp.get("meta")
    if not isinstance(meta, dict):
        meta = {}

    calibration = resp.get("calibration")
    if not isinstance(calibration, dict):
        calibration = {}

    diag_timestamp = _to_iso8601_utc(diagnostics.get("timestamp", payload_timestamp))
    diagnostics["timestamp"] = diag_timestamp
    diagnostics["event_id"] = str(diagnostics.get("event_id") or "").strip() or _event_id_from_timestamp(diag_timestamp)
    diagnostics["status"] = _normalize_status_text(diagnostics.get("status"), fallback="UNKNOWN")
    diagnostics["classifier_severity"] = _normalize_status_text(
        diagnostics.get("classifier_severity"),
        fallback=diagnostics["status"],
    )

    active_dimensions = metrics.get("active_dimensions")
    if isinstance(active_dimensions, list):
        normalized_dims = [str(item).strip() for item in active_dimensions if str(item).strip()]
    else:
        normalized_dims = []
    core_active_dimensions = metrics.get("core_active_dimensions")
    if isinstance(core_active_dimensions, list):
        metrics["core_active_dimensions"] = [str(item).strip() for item in core_active_dimensions if str(item).strip()]
    else:
        metrics["core_active_dimensions"] = [dim for dim in normalized_dims if dim in CORE_DIMENSION_SET]
    aux_active_dimensions = metrics.get("aux_active_dimensions")
    if isinstance(aux_active_dimensions, list):
        metrics["aux_active_dimensions"] = [str(item).strip() for item in aux_active_dimensions if str(item).strip()]
    else:
        metrics["aux_active_dimensions"] = [dim for dim in normalized_dims if dim in AUX_DIMENSION_SET]
    missing_dimensions = metrics.get("missing_dimensions")
    if isinstance(missing_dimensions, list):
        metrics["missing_dimensions"] = [str(item).strip() for item in missing_dimensions if str(item).strip()]
    else:
        metrics["missing_dimensions"] = []
    for key in ("candidate_dimensions", "observing_dimensions", "provisional_dimensions", "asset_critical_dimensions"):
        value = metrics.get(key)
        if isinstance(value, list):
            metrics[key] = [str(item).strip() for item in value if str(item).strip()]
        else:
            metrics[key] = []
    lifecycle_value = metrics.get("dimension_lifecycle_status")
    metrics["dimension_lifecycle_status"] = dict(lifecycle_value) if isinstance(lifecycle_value, dict) else {}
    importance_value = metrics.get("dimension_importance_by_asset")
    metrics["dimension_importance_by_asset"] = (
        {str(k): str(v) for k, v in importance_value.items() if str(k).strip()}
        if isinstance(importance_value, dict)
        else {}
    )
    metrics["asset_norm_scope"] = str(metrics.get("asset_norm_scope") or meta.get("asset_norm_scope") or "global")
    metrics["asset_norm_confidence"] = float(
        _safe_probability(metrics.get("asset_norm_confidence"), default=_safe_probability(meta.get("asset_norm_confidence"), default=0.0))
    )
    metrics["asset_class"] = str(metrics.get("asset_class") or meta.get("asset_class") or "generic")
    metrics["norm_source"] = str(metrics.get("norm_source") or meta.get("norm_source") or "global")
    metrics["observation_coverage"] = float(_safe_float(metrics.get("observation_coverage"), default=0.0))
    metrics["coverage_score"] = float(_safe_float(metrics.get("coverage_score"), default=metrics["observation_coverage"]))
    metrics["partial_observation"] = bool(
        metrics.get("partial_observation", bool(metrics["missing_dimensions"]) or metrics["coverage_score"] < 0.999)
    )

    primary_dim = str(diagnostics.get("primary_fault_dimension") or "").strip()
    if not primary_dim:
        primary_dim = normalized_dims[0] if normalized_dims else "unknown_sensor"
    diagnostics["primary_fault_dimension"] = primary_dim
    if not normalized_dims and primary_dim != "unknown_sensor":
        normalized_dims = [primary_dim]
    metrics["active_dimensions"] = normalized_dims

    root_label_norm = str(diagnostics.get("root_cause_label") or "").strip()
    if not root_label_norm:
        root_label_norm = "UNKNOWN_FALLBACK"
    probe_upper = root_label_norm.upper()
    if probe_upper in {"UNKNOWN", "UNKNOWN ANOMALY"}:
        root_label_norm = "UNKNOWN_FALLBACK"
    diagnostics["root_cause_label"] = root_label_norm
    diagnostics["root_cause_text"] = str(diagnostics.get("root_cause_text") or "").strip()
    diagnostics["trend_description"] = str(diagnostics.get("trend_description") or "unknown").strip() or "unknown"
    diagnostics["report_text"] = str(diagnostics.get("report_text") or "").strip()
    diagnostics["solution_recommendation"] = str(diagnostics.get("solution_recommendation") or "").strip()
    diagnostics["procedure_ref"] = str(diagnostics.get("procedure_ref") or "").strip()
    diagnostics["kb_version"] = str(diagnostics.get("kb_version") or "unknown").strip() or "unknown"
    diagnostics["secondary_diagnosis"] = (
        str(diagnostics.get("secondary_diagnosis") or "").strip() or None
    )
    diagnostics["location_context"] = str(diagnostics.get("location_context") or "unknown_zone")
    diagnostics["mechanism_family"] = str(diagnostics.get("mechanism_family") or "unknown_mechanism")
    diagnostics["selected_path_summary"] = str(diagnostics.get("selected_path_summary") or "")
    if not isinstance(diagnostics.get("relation_violations"), list):
        diagnostics["relation_violations"] = []
    if not isinstance(diagnostics.get("rule_hits"), list):
        diagnostics["rule_hits"] = []
    if not isinstance(diagnostics.get("prototype_hits"), list):
        diagnostics["prototype_hits"] = []
    diagnostics["severity_sigma"] = round(_safe_float(diagnostics.get("severity_sigma"), default=0.0), 4)
    diagnostics["root_cause_confidence"] = round(
        _safe_probability(
            diagnostics.get(
                "root_cause_confidence",
                diagnostics.get("classifier_confidence", 0.0),
            ),
            default=0.0,
        ),
        6,
    )
    diagnostics["failure_type"] = _canonical_failure_type(diagnostics)
    diagnostics["fault_name"] = (
        str(
            diagnostics.get("fault_name")
            or diagnostics.get("root_cause_text")
            or diagnostics.get("failure_type")
            or diagnostics.get("root_cause_label")
            or "Unknown anomaly"
        ).strip()
        or "Unknown anomaly"
    )
    diagnostics["fault_descriptor"] = (
        str(diagnostics.get("fault_descriptor") or diagnostics["fault_name"]).strip()
        or diagnostics["fault_name"]
    )
    if not isinstance(diagnostics.get("fault_name_candidates"), list):
        diagnostics["fault_name_candidates"] = []

    fallback_candidate_label = (
        diagnostics["root_cause_label"]
        if diagnostics["root_cause_label"] != "UNKNOWN"
        else (diagnostics["root_cause_text"] or diagnostics["failure_type"] or "Unknown Anomaly")
    )
    diagnostics["top3_candidates"] = _normalize_top3_candidates(
        diagnostics.get("top3_candidates"),
        fallback_label=fallback_candidate_label,
        fallback_probability=diagnostics["root_cause_confidence"],
    )
    ttf_seconds, ttf_state, ttf_confidence = _normalized_ttf_fields(diagnostics)
    diagnostics["ttf_seconds"] = ttf_seconds
    diagnostics["ttf_state"] = ttf_state
    diagnostics["ttf_confidence"] = ttf_confidence
    diagnostics["estimated_time_to_critical"] = ttf_seconds
    _apply_canonical_fault_ontology(diagnostics)
    meta["asset_class"] = str(meta.get("asset_class") or metrics.get("asset_class") or "generic")
    meta["asset_norm_scope"] = str(meta.get("asset_norm_scope") or metrics.get("asset_norm_scope") or "global")
    meta["asset_norm_confidence"] = float(
        _safe_probability(meta.get("asset_norm_confidence"), default=_safe_probability(metrics.get("asset_norm_confidence"), default=0.0))
    )
    meta["norm_source"] = str(meta.get("norm_source") or metrics.get("norm_source") or "global")
    for key in ("asset_critical_dimensions", "candidate_dimensions", "observing_dimensions", "provisional_dimensions", "approved_dimensions"):
        value = meta.get(key)
        if isinstance(value, list):
            meta[key] = [str(item).strip() for item in value if str(item).strip()]
        else:
            meta[key] = list(metrics.get(key) or [])
    lifecycle_meta = meta.get("dimension_lifecycle_status")
    meta["dimension_lifecycle_status"] = (
        {str(k): str(v) for k, v in lifecycle_meta.items() if str(k).strip()}
        if isinstance(lifecycle_meta, dict)
        else dict(metrics.get("dimension_lifecycle_status") or {})
    )
    importance_meta = meta.get("dimension_importance_by_asset")
    meta["dimension_importance_by_asset"] = (
        {str(k): str(v) for k, v in importance_meta.items() if str(k).strip()}
        if isinstance(importance_meta, dict)
        else dict(metrics.get("dimension_importance_by_asset") or {})
    )
    resp["meta"] = meta
    resp["metrics"] = metrics
    resp["diagnostics"] = diagnostics

    for key in DIAGNOSTIC_REQUIRED_FIELDS:
        diagnostics.setdefault(key, None)

    calibration_data = calibration.get("calibrated_data")
    if not isinstance(calibration_data, list):
        calibration["calibrated_data"] = []
    resp["calibration"] = calibration

    meta["asset_id"] = str(meta.get("asset_id") or asset_id)
    meta["mission_phase"] = str(meta.get("mission_phase") or mission_phase or "unknown")
    meta["runtime_mode"] = str(meta.get("runtime_mode") or "standard")
    meta["calibration_mode_effective"] = str(meta.get("calibration_mode_effective") or "safe")
    meta["data_policy"] = str(meta.get("data_policy") or DATA_POLICY_LABEL)
    meta["sampling_hz_effective"] = float(_safe_float(meta.get("sampling_hz_effective"), default=0.0))
    if meta["sampling_hz_effective"] <= 0.0:
        meta["sampling_hz_effective"] = float(_safe_float(meta.get("sampling_hz"), default=0.0))
    meta["seq_len"] = int(_safe_float(meta.get("seq_len"), default=len(calibration.get("calibrated_data", []))))
    meta["timestamp"] = _to_iso8601_utc(meta.get("timestamp", payload_timestamp))

    for key in ("memory_snapshot", "memory_context", "calibration_memory"):
        if not isinstance(resp.get(key), dict):
            resp[key] = {}

    resp["diagnostics"] = diagnostics
    resp["metrics"] = metrics
    resp["meta"] = meta


def _iso_to_epoch_seconds(value: Any) -> float:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return float(parsed.astimezone(timezone.utc).timestamp())
    except Exception:
        return 0.0


def _build_reasoning_trace_item(resp: Dict[str, Any]) -> Dict[str, Any]:
    diagnostics = resp.get("diagnostics") if isinstance(resp.get("diagnostics"), dict) else {}
    meta = resp.get("meta") if isinstance(resp.get("meta"), dict) else {}
    return {
        "event_id": str(diagnostics.get("event_id") or ""),
        "timestamp": str(diagnostics.get("timestamp") or ""),
        "asset_id": str(meta.get("asset_id") or ""),
        "primary_fault_dimension": str(diagnostics.get("primary_fault_dimension") or "unknown_sensor"),
        "failure_type": str(diagnostics.get("failure_type") or "unknown_anomaly"),
        "fault_label": str(diagnostics.get("fault_label") or diagnostics.get("fault_family_label") or ""),
        "fault_family_id": str(diagnostics.get("fault_family_id") or ""),
        "fault_family_label": str(diagnostics.get("fault_family_label") or ""),
        "fault_subtype_id": str(diagnostics.get("fault_subtype_id") or ""),
        "fault_subtype_label": str(diagnostics.get("fault_subtype_label") or ""),
        "fault_signature_id": str(diagnostics.get("fault_signature_id") or ""),
        "fault_signature_text": str(diagnostics.get("fault_signature_text") or ""),
        "known_status": str(diagnostics.get("known_status") or "unknown"),
        "similar_family_candidates": diagnostics.get("similar_family_candidates")
        if isinstance(diagnostics.get("similar_family_candidates"), list)
        else [],
        "fault_name": str(diagnostics.get("fault_name") or ""),
        "fault_descriptor": str(diagnostics.get("fault_descriptor") or diagnostics.get("fault_name") or ""),
        "status": str(diagnostics.get("status") or "UNKNOWN"),
        "root_cause_label": str(diagnostics.get("root_cause_label") or "UNKNOWN_FALLBACK"),
        "root_cause_confidence": round(
            _safe_probability(diagnostics.get("root_cause_confidence"), default=0.0),
            6,
        ),
        "top3_candidates": diagnostics.get("top3_candidates")
        if isinstance(diagnostics.get("top3_candidates"), list)
        else [],
        "relation_violations": diagnostics.get("relation_violations")
        if isinstance(diagnostics.get("relation_violations"), list)
        else [],
        "rule_hits": diagnostics.get("rule_hits") if isinstance(diagnostics.get("rule_hits"), list) else [],
        "prototype_hits": diagnostics.get("prototype_hits")
        if isinstance(diagnostics.get("prototype_hits"), list)
        else [],
        "selected_path_summary": str(diagnostics.get("selected_path_summary") or ""),
        "unknown_reason": str(diagnostics.get("unknown_reason") or ""),
        "ttf_seconds": diagnostics.get("ttf_seconds"),
        "ttf_state": str(diagnostics.get("ttf_state") or "unknown"),
        "ttf_confidence": round(_safe_probability(diagnostics.get("ttf_confidence"), default=0.0), 6),
        "procedure_ref": str(diagnostics.get("procedure_ref") or ""),
    }


def _np_json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _probe_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".write_probe_{os.getpid()}_{time.time_ns()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _ensure_writable_dir(preferred: Path, *, label: str) -> Path:
    preferred = Path(preferred)
    if _probe_writable_dir(preferred):
        return preferred
    fallback = Path(tempfile.gettempdir()) / "hybrid_api_state" / label
    if _probe_writable_dir(fallback):
        logger.warning(
            "state_dir_fallback label=%s preferred=%s fallback=%s",
            label,
            preferred,
            fallback,
        )
        return fallback
    raise RuntimeError(f"no writable state dir for {label}: preferred={preferred} fallback={fallback}")


def _resolve_writable_file_path(
    preferred: Path,
    *,
    fallback_dir: Path,
    env_name: str,
) -> Path:
    preferred = Path(preferred)
    if _probe_writable_dir(preferred.parent):
        return preferred
    fallback = Path(fallback_dir) / preferred.name
    if _probe_writable_dir(fallback.parent):
        logger.warning(
            "state_file_fallback env=%s preferred=%s fallback=%s",
            env_name,
            preferred,
            fallback,
        )
        return fallback
    return preferred


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = Path(str(path) + ".tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True, default=_np_json_default), encoding="utf-8")
    os.replace(tmp_path, path)


def _persist_reasoning_trace(app_ref: FastAPI, item: Dict[str, Any]) -> None:
    path = getattr(app_ref.state, "xai_reasoning_trace_path", None)
    if not isinstance(path, Path):
        return
    now_iso = datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    now_epoch = time.time()
    cutoff = now_epoch - TRACE_TTL_SECONDS

    with app_ref.state.xai_local_store_lock:
        payload: Dict[str, Any]
        if path.exists():
            try:
                parsed = json.loads(path.read_text(encoding="utf-8"))
                payload = parsed if isinstance(parsed, dict) else {}
            except Exception:
                payload = {}
        else:
            payload = {}

        items = payload.get("items")
        if not isinstance(items, list):
            items = []
        items.append(item)
        filtered = []
        for row in items:
            if not isinstance(row, dict):
                continue
            row_ts = _iso_to_epoch_seconds(row.get("timestamp"))
            if row_ts > 0 and row_ts < cutoff:
                continue
            filtered.append(row)
        if len(filtered) > TRACE_MAX_ITEMS:
            filtered = filtered[-TRACE_MAX_ITEMS:]

        payload = {
            "version": str(payload.get("version") or "1.0"),
            "updated_at": now_iso,
            "items": filtered,
        }
        _atomic_write_json(path, payload)


def _build_unknown_queue_entry(
    resp: Dict[str, Any],
    *,
    feature_vector: Optional[List[float]],
) -> Dict[str, Any]:
    diagnostics = resp.get("diagnostics") if isinstance(resp.get("diagnostics"), dict) else {}
    meta = resp.get("meta") if isinstance(resp.get("meta"), dict) else {}
    metrics = resp.get("metrics") if isinstance(resp.get("metrics"), dict) else {}
    calibration = resp.get("calibration") if isinstance(resp.get("calibration"), dict) else {}
    calibrated_rows = calibration.get("calibrated_data")
    if not isinstance(calibrated_rows, list):
        calibrated_rows = []
    if len(calibrated_rows) > 64:
        calibrated_rows = calibrated_rows[-64:]
    entry = {
        "event_id": str(diagnostics.get("event_id") or ""),
        "timestamp": str(diagnostics.get("timestamp") or ""),
        "asset_id": str(meta.get("asset_id") or ""),
        "asset_class": str(meta.get("asset_class") or metrics.get("asset_class") or "generic"),
        "asset_norm_scope": str(meta.get("asset_norm_scope") or metrics.get("asset_norm_scope") or "global"),
        "asset_norm_confidence": round(
            _safe_probability(meta.get("asset_norm_confidence"), default=_safe_probability(metrics.get("asset_norm_confidence"), default=0.0)),
            6,
        ),
        "norm_source": str(meta.get("norm_source") or metrics.get("norm_source") or "global"),
        "asset_critical_dimensions": list(meta.get("asset_critical_dimensions") or metrics.get("asset_critical_dimensions") or []),
        "candidate_dimensions": list(meta.get("candidate_dimensions") or metrics.get("candidate_dimensions") or []),
        "observing_dimensions": list(meta.get("observing_dimensions") or metrics.get("observing_dimensions") or []),
        "provisional_dimensions": list(meta.get("provisional_dimensions") or metrics.get("provisional_dimensions") or []),
        "approved_dimensions": list(meta.get("approved_dimensions") or []),
        "dimension_importance_by_asset": dict(meta.get("dimension_importance_by_asset") or metrics.get("dimension_importance_by_asset") or {}),
        "dimension_lifecycle_status": dict(meta.get("dimension_lifecycle_status") or metrics.get("dimension_lifecycle_status") or {}),
        "mission_phase": str(meta.get("mission_phase") or "unknown"),
        "runtime_mode": str(meta.get("runtime_mode") or "standard"),
        "status": str(diagnostics.get("status") or "UNKNOWN"),
        "fault_label": str(diagnostics.get("fault_label") or diagnostics.get("fault_family_label") or "Unknown Anomaly"),
        "fault_family_id": str(diagnostics.get("fault_family_id") or "unknown_anomaly"),
        "fault_family_label": str(diagnostics.get("fault_family_label") or diagnostics.get("fault_label") or "Unknown Anomaly"),
        "fault_subtype_id": str(diagnostics.get("fault_subtype_id") or ""),
        "fault_subtype_label": str(diagnostics.get("fault_subtype_label") or ""),
        "fault_signature_id": str(diagnostics.get("fault_signature_id") or ""),
        "fault_signature_text": str(diagnostics.get("fault_signature_text") or ""),
        "known_status": str(diagnostics.get("known_status") or "unknown"),
        "similar_family_candidates": diagnostics.get("similar_family_candidates")
        if isinstance(diagnostics.get("similar_family_candidates"), list)
        else [],
        "review_required": bool(diagnostics.get("review_required", True)),
        "root_cause_label": str(diagnostics.get("root_cause_label") or "UNKNOWN_FALLBACK"),
        "root_cause_confidence": round(
            _safe_probability(diagnostics.get("root_cause_confidence"), default=0.0),
            6,
        ),
        "is_unknown": bool(diagnostics.get("is_unknown", False)),
        "unknown_reason": str(diagnostics.get("unknown_reason") or ""),
        "primary_fault_dimension": str(diagnostics.get("primary_fault_dimension") or "unknown_sensor"),
        "failure_type": str(diagnostics.get("failure_type") or "unknown_anomaly"),
        "fault_descriptor": str(diagnostics.get("fault_descriptor") or diagnostics.get("fault_name") or ""),
        "top3_candidates": diagnostics.get("top3_candidates")
        if isinstance(diagnostics.get("top3_candidates"), list)
        else [],
        "procedure_ref": str(diagnostics.get("procedure_ref") or ""),
        "ttf_seconds": diagnostics.get("ttf_seconds"),
        "ttf_state": str(diagnostics.get("ttf_state") or "unknown"),
        "ttf_confidence": round(_safe_probability(diagnostics.get("ttf_confidence"), default=0.0), 6),
        "calibrated_data": calibrated_rows,
    }
    if isinstance(feature_vector, list):
        entry["feature_vector"] = feature_vector
    return entry


def _append_unknown_queue(app_ref: FastAPI, entry: Dict[str, Any]) -> None:
    path = getattr(app_ref.state, "xai_unknown_queue_path", None)
    if not isinstance(path, Path):
        return
    with app_ref.state.xai_local_store_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry, ensure_ascii=True) + "\n")


def _persist_local_xai_records(
    app_ref: FastAPI,
    resp: Dict[str, Any],
    *,
    feature_vector: Optional[List[float]],
) -> None:
    queue_links: List[str] = []
    model_record = build_model_decision_record(
        resp=resp,
        timezone_name=os.getenv("XAI_DECISION_LEDGER_TIMEZONE", "Europe/Istanbul"),
    )

    try:
        diagnostics = resp.get("diagnostics") if isinstance(resp.get("diagnostics"), dict) else {}
        conf_threshold = float(getattr(app_ref.state, "xai_unknown_conf_threshold", 0.55))
        should_enqueue = bool(diagnostics.get("is_unknown", False)) or (
            _safe_probability(diagnostics.get("root_cause_confidence"), default=0.0) < conf_threshold
        )
        if should_enqueue:
            queue_entry = _build_unknown_queue_entry(resp, feature_vector=feature_vector)
            review_store = getattr(app_ref.state, "canonical_review_store", None)
            if review_store is not None:
                submitted = review_store.submit_model_case(
                    queue_entry,
                    decision_id=str(model_record.get("decision_id") or ""),
                )
                queue_id = str(submitted.get("queue_id") or submitted.get("unk_code") or "").strip()
                if queue_id:
                    queue_links.append(queue_id)
                    queue_entry["queue_id"] = queue_id
            _append_unknown_queue(app_ref, queue_entry)
    except Exception:
        logger.exception("xai_unknown_queue_append_failed")

    try:
        trace_item = _build_reasoning_trace_item(resp)
        _persist_reasoning_trace(app_ref, trace_item)
    except Exception:
        logger.exception("xai_reasoning_trace_persist_failed")

    try:
        ledger = getattr(app_ref.state, "xai_decision_ledger", None)
        if ledger is not None:
            if queue_links:
                model_record["linked_review_queue_ids"] = list(dict.fromkeys(queue_links))
            ledger.append(model_record)
    except Exception:
        logger.exception("xai_decision_ledger_append_failed")


def _collect_runtime_state_payload(app_ref: FastAPI) -> Dict[str, Any]:
    now_ts = time.time()
    ttl_sec = float(getattr(app_ref.state, "state_store_ttl_sec", 21600))
    max_guard = int(getattr(app_ref.state, "state_store_max_guard_entries", 2048))
    max_assets = int(getattr(app_ref.state, "state_store_max_asset_entries", 1024))

    with app_ref.state.calibration_guard_lock:
        guard_items = list(app_ref.state.calibration_guard_states.items())
    guard_items.sort(key=lambda item: float(item[1].get("_updated_at", 0.0)), reverse=True)
    guard_state: Dict[str, Dict[str, Any]] = {}
    for key, value in guard_items:
        updated_at = float(value.get("_updated_at", 0.0))
        if updated_at > 0 and (now_ts - updated_at) > ttl_sec:
            continue
        guard_state[str(key)] = dict(value)
        if len(guard_state) >= max_guard:
            break

    asset_summaries = dict(getattr(app_ref.state, "asset_context_summaries", {}))
    asset_items = sorted(
        asset_summaries.items(),
        key=lambda item: float(item[1].get("last_seen_ts", 0.0)),
        reverse=True,
    )
    asset_state: Dict[str, Dict[str, Any]] = {}
    for asset_id, summary in asset_items:
        last_seen = float(summary.get("last_seen_ts", 0.0))
        if last_seen > 0 and (now_ts - last_seen) > ttl_sec:
            continue
        asset_state[str(asset_id)] = dict(summary)
        if len(asset_state) >= max_assets:
            break

    return {
        "version": 1,
        "updated_at": now_ts,
        "ttl_sec": ttl_sec,
        "guard_state": guard_state,
        "asset_context_summaries": asset_state,
    }


def _persist_runtime_state_if_due(app_ref: FastAPI, *, force: bool = False) -> None:
    if not bool(getattr(app_ref.state, "state_store_enabled", False)):
        return
    path = getattr(app_ref.state, "state_store_path", None)
    if path is None:
        return

    now_ts = time.time()
    interval_sec = float(getattr(app_ref.state, "state_store_write_interval_sec", 2.0))
    last_persist = float(getattr(app_ref.state, "state_store_last_persist_ts", 0.0))
    if not force and (now_ts - last_persist) < interval_sec:
        return

    with app_ref.state.state_store_lock:
        now_ts = time.time()
        last_persist = float(getattr(app_ref.state, "state_store_last_persist_ts", 0.0))
        if not force and (now_ts - last_persist) < interval_sec:
            return
        payload = _collect_runtime_state_payload(app_ref)
        tmp_path = Path(str(path) + ".tmp")
        try:
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(json.dumps(payload, ensure_ascii=True, default=_np_json_default), encoding="utf-8")
            os.replace(tmp_path, path)
            app_ref.state.state_store_last_persist_ts = now_ts
            app_ref.state.state_store_connected = True
            app_ref.state.continuity_mode = "durable"
        except Exception:
            logger.exception("state_store_persist_failed path=%s", path)
            app_ref.state.state_store_connected = False
            app_ref.state.continuity_mode = "degraded"
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass


def _restore_runtime_state(app_ref: FastAPI) -> None:
    if not bool(getattr(app_ref.state, "state_store_enabled", False)):
        app_ref.state.state_store_connected = False
        app_ref.state.continuity_mode = "off"
        return

    path = getattr(app_ref.state, "state_store_path", None)
    if path is None:
        app_ref.state.state_store_connected = False
        app_ref.state.continuity_mode = "degraded"
        return

    if not path.exists():
        app_ref.state.state_store_connected = True
        app_ref.state.continuity_mode = "durable"
        return

    now_ts = time.time()
    ttl_sec = float(getattr(app_ref.state, "state_store_ttl_sec", 21600))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        restored_guard = {}
        for key, value in dict(payload.get("guard_state", {})).items():
            updated_at = float(value.get("_updated_at", 0.0))
            if updated_at > 0 and (now_ts - updated_at) > ttl_sec:
                continue
            restored_guard[str(key)] = dict(value)
        restored_assets = {}
        for asset_id, summary in dict(payload.get("asset_context_summaries", {})).items():
            last_seen = float(summary.get("last_seen_ts", 0.0))
            if last_seen > 0 and (now_ts - last_seen) > ttl_sec:
                continue
            restored_assets[str(asset_id)] = dict(summary)

        with app_ref.state.calibration_guard_lock:
            app_ref.state.calibration_guard_states.update(restored_guard)
        app_ref.state.asset_context_summaries.update(restored_assets)
        app_ref.state.state_store_connected = True
        app_ref.state.continuity_mode = "durable"
        logger.info(
            json.dumps(
                {
                    "event": "state_store_restore_ok",
                    "path": str(path),
                    "restored_guard_states": len(restored_guard),
                    "restored_asset_summaries": len(restored_assets),
                }
            )
        )
    except Exception:
        logger.exception("state_store_restore_failed path=%s", path)
        app_ref.state.state_store_connected = False
        app_ref.state.continuity_mode = "degraded"


def _validate_quality_policy(policy: Optional[str]) -> str:
    normalized = str(policy or "fail_closed").strip().lower()
    if normalized not in {"fail_closed", "mask_invalid"}:
        raise HTTPException(
            status_code=422,
            detail="quality_policy must be one of: fail_closed, mask_invalid",
        )
    return normalized


def _validate_calibration_mode(calibration_mode: Optional[str]) -> str:
    normalized = str(calibration_mode or "safe").strip().lower()
    if normalized not in CALIBRATION_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"calibration_mode must be one of: {', '.join(CALIBRATION_MODES)}",
        )
    return normalized


def _resolve_runtime_mode(runtime_mode: Optional[str], seq_len: int, sampling_hz: float) -> str:
    if runtime_mode is not None:
        normalized = str(runtime_mode).strip().lower()
        if normalized not in {"standard", "nano"}:
            raise HTTPException(status_code=422, detail="runtime_mode must be one of: standard, nano")
        return normalized
    # Auto mode selection: nano for short windows or high sampling rates.
    return "nano" if (seq_len <= 64 or sampling_hz >= 50.0) else "standard"


def _effective_sampling_hz(runtime_mode: str, sampling_hz: float) -> float:
    if runtime_mode == "nano":
        return float(max(sampling_hz, 10.0))
    return float(sampling_hz)


def _resolve_asset_identity(
    request: Request,
    *,
    asset_id: Optional[str],
    site_id: Optional[str],
    line_id: Optional[str],
    machine_id: Optional[str],
) -> AssetResolution:
    resolver = getattr(request.app.state, "asset_identity_resolver", None)
    explicit = str(asset_id or "").strip()
    if resolver is None:
        if explicit and explicit.lower() != "auto":
            if not ASSET_ID_PATTERN.fullmatch(explicit):
                raise HTTPException(status_code=422, detail="asset_id must match [a-zA-Z0-9_-]{1,64}")
            return AssetResolution(
                resolved_asset_id=explicit,
                resolution_mode="provided",
                review_required=False,
                proposal_id=None,
                proposal_status="none",
                degraded_resolution=True,
            )
        raise HTTPException(
            status_code=422,
            detail="asset resolution unavailable; provide explicit asset_id",
        )
    try:
        return resolver.resolve(
            asset_id=asset_id,
            site_id=site_id,
            line_id=line_id,
            machine_id=machine_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        if explicit and explicit.lower() != "auto" and ASSET_ID_PATTERN.fullmatch(explicit):
            logger.exception("asset_identity_resolver_degraded")
            return AssetResolution(
                resolved_asset_id=explicit,
                resolution_mode="provided",
                review_required=False,
                proposal_id=None,
                proposal_status="none",
                degraded_resolution=True,
            )
        raise HTTPException(
            status_code=422,
            detail="asset resolution unavailable; provide explicit asset_id",
        )


def _asset_resolution_meta(resolution: AssetResolution) -> Dict[str, Any]:
    return {
        "resolved_asset_id": resolution.resolved_asset_id,
        "asset_resolution_mode": resolution.resolution_mode,
        "asset_review_required": bool(resolution.review_required),
        "asset_proposal_id": resolution.proposal_id,
        "asset_proposal_status": resolution.proposal_status,
        "degraded_resolution": bool(resolution.degraded_resolution),
    }


def _is_operational_memory_request_eligible(payload: DiagnoseIn) -> bool:
    if not bool(payload.sensors):
        return False
    if bool(payload.site_context):
        return False
    if bool(payload.forecast_context):
        return False
    control_mode = str(payload.control_mode or "guarded_auto").strip().lower()
    return control_mode in {"", "guarded_auto"}


def _feature_map_summary(result: Any, pipeline) -> dict[str, Any]:
    try:
        schema = pipeline.registry.get(result.selected_agent).schema
    except Exception:
        return {}
    summary = {
        "agent_id": schema.agent_id,
        "sector": schema.sector,
        "schema_version": schema.version,
        "feature_order": list(schema.feature_order),
        "required_features": list(schema.required_features),
    }
    agent_outputs = getattr(result, "agent_outputs", {}) if hasattr(result, "agent_outputs") else {}
    if isinstance(agent_outputs, dict):
        for key in (
            "normalization_summary",
            "normalized_sensor_count",
            "unknown_sensor_count",
            "unknown_sensor_names",
            "ambiguous_sensor_names",
        ):
            if key in agent_outputs:
                summary[key] = agent_outputs[key]
    return summary


def _operational_memory_recordable(payload: DiagnoseIn, body: dict[str, Any]) -> bool:
    if not _is_operational_memory_request_eligible(payload):
        return False
    if bool(body.get("route_needs_context", False)):
        return False
    if float(body.get("route_confidence", 0.0) or 0.0) < 0.70:
        return False
    rejected = list(body.get("rejected_samples") or [])
    sensors = list(payload.sensors or [])
    if sensors and (len(rejected) / float(len(sensors))) > 0.25:
        return False
    if str(body.get("selected_agent") or "").strip() == "":
        return False
    if str(body.get("anomaly_family") or "").strip().lower() == "unknown":
        return False
    return True




def _asset_norm_meta(summary: Optional[AssetNormSummary]) -> Dict[str, Any]:
    if summary is None:
        return {
            "asset_class": "generic",
            "asset_norm_scope": "global",
            "asset_norm_confidence": 0.0,
            "norm_source": "global",
            "asset_critical_dimensions": [],
            "candidate_dimensions": [],
            "observing_dimensions": [],
            "provisional_dimensions": [],
            "approved_dimensions": [],
            "dimension_importance_by_asset": {},
            "dimension_lifecycle_status": {},
        }
    return {
        "asset_class": summary.asset_class,
        "asset_norm_scope": summary.asset_norm_scope,
        "asset_norm_confidence": float(summary.asset_norm_confidence),
        "norm_source": summary.norm_source,
        "asset_critical_dimensions": list(summary.asset_critical_dimensions),
        "candidate_dimensions": list(summary.candidate_dimensions),
        "observing_dimensions": list(summary.observing_dimensions),
        "provisional_dimensions": list(summary.provisional_dimensions),
        "approved_dimensions": list(summary.approved_dimensions),
        "dimension_importance_by_asset": dict(summary.dimension_importance_by_asset),
        "dimension_lifecycle_status": dict(summary.dimension_lifecycle_status),
    }




def _validate_common_fields(
    asset_id: str,
    timestamp: float,
    seq_len: int,
    sampling_hz: float,
    runtime_mode: Optional[str],
) -> tuple[str, float]:
    if not ASSET_ID_PATTERN.fullmatch(asset_id):
        raise HTTPException(status_code=422, detail="asset_id must match [a-zA-Z0-9_-]{1,64}")

    if seq_len < SEQ_MIN or seq_len > SEQ_MAX:
        raise HTTPException(status_code=422, detail=f"seq_len must be in [{SEQ_MIN}, {SEQ_MAX}]")

    if sampling_hz < SAMPLING_HZ_MIN or sampling_hz > SAMPLING_HZ_MAX:
        raise HTTPException(
            status_code=422,
            detail=f"sampling_hz must be in [{SAMPLING_HZ_MIN}, {SAMPLING_HZ_MAX}]",
        )

    now_ts = time.time()
    if abs(timestamp - now_ts) > 7 * 24 * 3600:
        raise HTTPException(status_code=422, detail="timestamp is outside the accepted 7-day window")
    mode = _resolve_runtime_mode(runtime_mode, seq_len, sampling_hz)
    return mode, _effective_sampling_hz(mode, sampling_hz)


@dataclass
class NormalizedTelemetry:
    core_matrix: np.ndarray
    presence_mask: np.ndarray
    observed_dimensions: List[str]
    core_dimensions: List[str]
    aux_dimensions: List[str]
    missing_dimensions: List[str]
    candidate_dimensions: List[str]
    source_dimension_map: Dict[str, str]
    internal_pressure_candidates: List[str]
    unused_input_columns: List[str]
    rejected_dimensions: List[str]
    observed_series: Dict[str, List[float]]
    selected_indices: Dict[str, int]


def _validate_data_matrix(
    matrix: List[List[float]],
    *,
    seq_len: int,
    field_name: str,
    expected_dim_count: Optional[int] = None,
) -> np.ndarray:
    if len(matrix) != seq_len:
        raise HTTPException(
            status_code=422,
            detail=f"seq_len ({seq_len}) does not match len({field_name}) ({len(matrix)})",
        )

    try:
        data_np = np.asarray(matrix, dtype=np.float32)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"invalid {field_name} matrix: {e}")

    required_dims = len(DIMENSIONS) if expected_dim_count is None else int(expected_dim_count)
    if data_np.ndim != 2 or data_np.shape[1] != required_dims:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be shaped [seq_len, {required_dims}]",
        )

    if not np.isfinite(data_np).all():
        raise HTTPException(status_code=422, detail=f"{field_name} contains NaN or inf")

    return data_np


def _validate_optional_bool_matrix(
    matrix: Optional[List[List[bool]]],
    expected_shape: tuple[int, int],
    *,
    field_name: str,
) -> Optional[np.ndarray]:
    if matrix is None:
        return None
    arr = np.asarray(matrix, dtype=bool)
    if arr.shape != expected_shape:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must match the exact shape of data [{expected_shape[0]}, {expected_shape[1]}]",
        )
    return arr


def _canonical_dimension_name(name: str) -> Optional[str]:
    return canonical_dimension_name(name)


def _validate_dimension_ranges(
    matrix: np.ndarray,
    *,
    field_name: str,
    dimensions: List[str],
) -> None:
    if matrix.ndim != 2:
        raise HTTPException(status_code=422, detail=f"{field_name} must be a 2D matrix")
    if matrix.shape[1] != len(dimensions):
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be shaped [seq_len, {len(dimensions)}]",
        )
    for idx, dim_name in enumerate(dimensions):
        low, high = DIMENSION_RANGES[dim_name]
        col = matrix[:, idx]
        if np.any(col < low) or np.any(col > high):
            raise HTTPException(
                status_code=422,
                detail=f"{field_name}.{dim_name} out of accepted envelope [{low}, {high}]",
            )


def _observed_series_from_matrix(matrix: np.ndarray, dimensions: List[str]) -> Dict[str, List[float]]:
    observed: Dict[str, List[float]] = {}
    for idx, dim in enumerate(dimensions):
        observed[dim] = [float(x) for x in matrix[:, idx].tolist()]
    return observed


def _project_boolean_mask(
    mask_np: np.ndarray,
    *,
    selected_indices: Dict[str, int],
    seq_len: int,
) -> np.ndarray:
    projected = np.zeros((seq_len, len(DIMENSIONS)), dtype=bool)
    for dim_name, src_idx in selected_indices.items():
        dst_idx = CANONICAL_DIMENSION_INDEX[dim_name]
        projected[:, dst_idx] = mask_np[:, int(src_idx)]
    return projected


def _apply_observation_mask(
    telemetry: NormalizedTelemetry,
    *,
    mask_np: Optional[np.ndarray],
) -> NormalizedTelemetry:
    if mask_np is None:
        effective_presence = telemetry.presence_mask.astype(bool, copy=True)
    else:
        projected_mask = _project_boolean_mask(
            mask_np.astype(bool, copy=False),
            selected_indices=telemetry.selected_indices,
            seq_len=int(telemetry.core_matrix.shape[0]),
        )
        effective_presence = np.logical_and(telemetry.presence_mask, projected_mask)

    if not bool(effective_presence.any()):
        raise HTTPException(
            status_code=422,
            detail="all observations are missing or invalid after applying quality/presence masks",
        )

    masked_matrix = telemetry.core_matrix.astype(np.float32, copy=True)
    masked_matrix[~effective_presence] = 0.0

    observed_dimensions: List[str] = []
    observed_series: Dict[str, List[float]] = {}
    for dim_name in CANONICAL_DIMENSION_LIST:
        dim_idx = CANONICAL_DIMENSION_INDEX[dim_name]
        dim_presence = effective_presence[:, dim_idx]
        if not bool(dim_presence.any()):
            continue
        observed_dimensions.append(dim_name)
        observed_series[dim_name] = [float(x) for x in masked_matrix[dim_presence, dim_idx].tolist()]

    return NormalizedTelemetry(
        core_matrix=masked_matrix,
        presence_mask=effective_presence,
        observed_dimensions=observed_dimensions,
        core_dimensions=[dim for dim in observed_dimensions if dim in CORE_DIMENSION_SET],
        aux_dimensions=[dim for dim in observed_dimensions if dim in AUX_DIMENSION_SET],
        missing_dimensions=[dim for dim in CANONICAL_DIMENSION_LIST if dim not in observed_dimensions],
        candidate_dimensions=list(telemetry.candidate_dimensions),
        source_dimension_map=dict(telemetry.source_dimension_map),
        internal_pressure_candidates=list(telemetry.internal_pressure_candidates),
        unused_input_columns=list(telemetry.unused_input_columns),
        rejected_dimensions=list(telemetry.rejected_dimensions),
        observed_series=observed_series,
        selected_indices=dict(telemetry.selected_indices),
    )


def _normalize_primary_source_map(
    raw_map: Optional[Dict[str, str]],
    *,
    primary_internal_pressure_sensor: Optional[str],
) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    if isinstance(raw_map, dict):
        for key, value in raw_map.items():
            canonical_key = _canonical_dimension_name(str(key))
            if not canonical_key:
                continue
            text = str(value or "").strip()
            if text:
                normalized[canonical_key] = text
    preferred_internal = str(primary_internal_pressure_sensor or "").strip()
    if preferred_internal and "internal_pressure" not in normalized:
        normalized["internal_pressure"] = preferred_internal
    return normalized


def _project_to_canonical_dimensions(
    data_np: np.ndarray,
    *,
    field_name: str,
    dimension_names: Optional[List[str]],
    primary_dimension_sources: Optional[Dict[str, str]],
    primary_internal_pressure_sensor: Optional[str],
    required_dimensions: Optional[List[str]] = None,
) -> NormalizedTelemetry:
    if data_np.ndim != 2:
        raise HTTPException(status_code=422, detail=f"{field_name} must be a 2D matrix")

    col_count = int(data_np.shape[1])
    if not dimension_names:
        if col_count not in {len(CORE_DIMENSIONS), len(LEGACY_CORE_DIMENSIONS)}:
            raise HTTPException(
                status_code=422,
                detail="dimension_names required for non-canonical or extended payloads",
            )
        default_dimensions = list(CORE_DIMENSIONS if col_count == len(CORE_DIMENSIONS) else LEGACY_CORE_DIMENSIONS)
        source_map = {dim: dim for dim in default_dimensions}
        projected = np.zeros((data_np.shape[0], len(DIMENSIONS)), dtype=np.float32)
        presence = np.zeros((data_np.shape[0], len(DIMENSIONS)), dtype=bool)
        selected_indices = {}
        for idx, dim_name in enumerate(default_dimensions):
            dst_idx = CANONICAL_DIMENSION_INDEX[dim_name]
            projected[:, dst_idx] = data_np[:, idx]
            presence[:, dst_idx] = True
            selected_indices[dim_name] = idx
        _validate_dimension_ranges(data_np, field_name=field_name, dimensions=default_dimensions)
        return NormalizedTelemetry(
            core_matrix=projected,
            presence_mask=presence,
            observed_dimensions=list(default_dimensions),
            core_dimensions=list(default_dimensions),
            aux_dimensions=[],
            missing_dimensions=[dim for dim in CANONICAL_DIMENSION_LIST if dim not in default_dimensions],
            candidate_dimensions=[],
            source_dimension_map=source_map,
            internal_pressure_candidates=[source_map["internal_pressure"]] if "internal_pressure" in source_map else [],
            unused_input_columns=[],
            rejected_dimensions=[],
            observed_series=_observed_series_from_matrix(data_np, list(default_dimensions)),
            selected_indices=selected_indices,
        )

    if len(dimension_names) != col_count:
        raise HTTPException(
            status_code=422,
            detail=f"len(dimension_names) ({len(dimension_names)}) must match {field_name} columns ({col_count})",
        )

    cleaned_names = [str(x).strip() for x in dimension_names]
    if any(not name for name in cleaned_names):
        raise HTTPException(status_code=422, detail="dimension_names cannot contain empty labels")

    candidate_by_dim: Dict[str, List[int]] = {dim: [] for dim in CANONICAL_DIMENSION_LIST}
    rejected_dimensions: List[str] = []
    candidate_dimensions: List[str] = []
    for idx, raw_name in enumerate(cleaned_names):
        canonical = _canonical_dimension_name(raw_name)
        if canonical is None:
            rejected_dimensions.append(raw_name)
            candidate_dimensions.append(raw_name)
            continue
        candidate_by_dim[canonical].append(idx)

    selected_idx: Dict[str, int] = {}
    source_map: Dict[str, str] = {}
    normalized_sources = _normalize_primary_source_map(
        primary_dimension_sources,
        primary_internal_pressure_sensor=primary_internal_pressure_sensor,
    )
    candidates_map: Dict[str, List[str]] = {}
    used_column_indices: set[int] = set()

    for dim in CANONICAL_DIMENSION_LIST:
        candidates = candidate_by_dim.get(dim, [])
        candidates_map[dim] = [cleaned_names[idx] for idx in candidates]
        if not candidates:
            continue
        if len(candidates) == 1:
            idx = candidates[0]
            selected_idx[dim] = idx
            source_map[dim] = cleaned_names[idx]
            used_column_indices.add(int(idx))
            continue

        explicit_source = str(normalized_sources.get(dim, "")).strip()
        if not explicit_source:
            preferred_idx = None
            for idx in candidates:
                if cleaned_names[idx] == dim:
                    preferred_idx = idx
                    break
            if preferred_idx is None:
                preferred_idx = candidates[0]
            selected_idx[dim] = int(preferred_idx)
            source_map[dim] = cleaned_names[int(preferred_idx)]
            used_column_indices.add(int(preferred_idx))
            continue
        exact_match_idx: Optional[int] = None
        for idx in candidates:
            if cleaned_names[idx] == explicit_source:
                exact_match_idx = idx
                break
        if exact_match_idx is None:
            explicit_norm = explicit_source.lower()
            for idx in candidates:
                if cleaned_names[idx].lower() == explicit_norm:
                    exact_match_idx = idx
                    break
        if exact_match_idx is None:
            candidate_labels = [cleaned_names[idx] for idx in candidates]
            raise HTTPException(
                status_code=422,
                detail=(
                    f"primary_dimension_sources.{dim}='{explicit_source}' does not match candidates: "
                    f"{candidate_labels}"
                ),
            )
        selected_idx[dim] = int(exact_match_idx)
        source_map[dim] = cleaned_names[int(exact_match_idx)]
        used_column_indices.add(int(exact_match_idx))

    if not selected_idx and not candidate_dimensions:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} does not contain any supported canonical dimensions",
        )

    if not selected_idx and candidate_dimensions:
        zero_matrix = np.zeros((data_np.shape[0], len(DIMENSIONS)), dtype=np.float32)
        zero_presence = np.zeros((data_np.shape[0], len(DIMENSIONS)), dtype=bool)
        return NormalizedTelemetry(
            core_matrix=zero_matrix,
            presence_mask=zero_presence,
            observed_dimensions=[],
            core_dimensions=[],
            aux_dimensions=[],
            missing_dimensions=list(CANONICAL_DIMENSION_LIST),
            candidate_dimensions=list(dict.fromkeys(candidate_dimensions)),
            source_dimension_map={},
            internal_pressure_candidates=[],
            unused_input_columns=list(cleaned_names),
            rejected_dimensions=list(dict.fromkeys(rejected_dimensions)),
            observed_series={},
            selected_indices={},
        )

    core_projected = np.zeros((data_np.shape[0], len(DIMENSIONS)), dtype=np.float32)
    presence_mask = np.zeros((data_np.shape[0], len(DIMENSIONS)), dtype=bool)
    for dim_name, src_idx in selected_idx.items():
        dim_idx = CANONICAL_DIMENSION_INDEX[dim_name]
        core_projected[:, dim_idx] = data_np[:, src_idx]
        presence_mask[:, dim_idx] = True

    observed_dimensions = [dim for dim in CANONICAL_DIMENSION_LIST if dim in selected_idx]
    aux_dimensions = [dim for dim in observed_dimensions if dim in AUX_DIMENSION_SET]
    observed_series = {
        dim: [float(x) for x in data_np[:, selected_idx[dim]].tolist()]
        for dim in observed_dimensions
    }
    _validate_dimension_ranges(
        np.stack([data_np[:, selected_idx[dim]] for dim in observed_dimensions], axis=1),
        field_name=field_name,
        dimensions=observed_dimensions,
    )

    internal_candidates = candidates_map.get("internal_pressure", [])
    return NormalizedTelemetry(
        core_matrix=core_projected,
        presence_mask=presence_mask,
        observed_dimensions=observed_dimensions,
        core_dimensions=[dim for dim in observed_dimensions if dim in CORE_DIMENSION_SET],
        aux_dimensions=aux_dimensions,
        missing_dimensions=[dim for dim in CANONICAL_DIMENSION_LIST if dim not in observed_dimensions],
        candidate_dimensions=list(dict.fromkeys(candidate_dimensions)),
        source_dimension_map=source_map,
        internal_pressure_candidates=internal_candidates,
        unused_input_columns=[cleaned_names[idx] for idx in range(col_count) if idx not in used_column_indices],
        rejected_dimensions=list(dict.fromkeys(rejected_dimensions)),
        observed_series=observed_series,
        selected_indices=selected_idx,
    )


def _normalize_sensor_values_payload(sensor_values: Dict[str, float]) -> NormalizedTelemetry:
    if not isinstance(sensor_values, dict) or not sensor_values:
        raise HTTPException(status_code=422, detail="sensor_values must be a non-empty object.")
    observed_series: Dict[str, List[float]] = {}
    source_map: Dict[str, str] = {}
    rejected: List[str] = []
    for raw_name, raw_value in sensor_values.items():
        canonical = _canonical_dimension_name(str(raw_name))
        if canonical is None:
            rejected.append(str(raw_name))
            continue
        if canonical in observed_series:
            raise HTTPException(
                status_code=422,
                detail=f"sensor_values contains duplicate canonical dimension '{canonical}' via multiple keys",
            )
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail=f"sensor_values.{raw_name} must be numeric")
        if not np.isfinite(value):
            raise HTTPException(status_code=422, detail=f"sensor_values.{raw_name} contains NaN or inf")
        low, high = DIMENSION_RANGES[canonical]
        if value < low or value > high:
            raise HTTPException(
                status_code=422,
                detail=f"sensor_values.{canonical} out of accepted envelope [{low}, {high}]",
            )
        observed_series[canonical] = [value]
        source_map[canonical] = str(raw_name)
    core_rows = np.zeros((1, len(DIMENSIONS)), dtype=np.float32)
    presence_mask = np.zeros((1, len(DIMENSIONS)), dtype=bool)
    selected_indices: Dict[str, int] = {}
    for dim, rows in observed_series.items():
        dim_idx = CANONICAL_DIMENSION_INDEX[dim]
        core_rows[0, dim_idx] = float(rows[0])
        presence_mask[0, dim_idx] = True
        selected_indices[dim] = dim_idx
    return NormalizedTelemetry(
        core_matrix=core_rows,
        presence_mask=presence_mask,
        observed_dimensions=[dim for dim in CANONICAL_DIMENSION_LIST if dim in observed_series],
        core_dimensions=[dim for dim in CORE_DIMENSIONS if dim in observed_series],
        aux_dimensions=[dim for dim in CANONICAL_DIMENSION_LIST if dim in AUX_DIMENSION_SET and dim in observed_series],
        missing_dimensions=[dim for dim in CANONICAL_DIMENSION_LIST if dim not in observed_series],
        candidate_dimensions=list(dict.fromkeys(rejected)),
        source_dimension_map=source_map,
        internal_pressure_candidates=[source_map["internal_pressure"]] if "internal_pressure" in source_map else [],
        unused_input_columns=[],
        rejected_dimensions=list(dict.fromkeys(rejected)),
        observed_series=observed_series,
        selected_indices=selected_indices,
    )


def _validate_quality(
    quality: Optional[List[List[bool]]],
    expected_shape: tuple[int, int],
    *,
    quality_policy: str = "fail_closed",
) -> tuple[Optional[np.ndarray], int]:
    if quality is None:
        return None, 0
    q = np.asarray(quality, dtype=bool)
    if q.shape != expected_shape:
        raise HTTPException(
            status_code=422,
            detail=f"quality must match the exact shape of data [{expected_shape[0]}, {expected_shape[1]}]",
        )
    invalid_count = int((~q).sum())
    if invalid_count >= q.size and quality_policy != "mask_invalid":
        raise HTTPException(
            status_code=422,
            detail="all observations are invalid after applying quality flags",
        )
    return q, invalid_count


def _apply_quality_mask(raw_np: np.ndarray, quality_mask: Optional[np.ndarray]) -> np.ndarray:
    if quality_mask is None:
        return raw_np
    if quality_mask.all():
        return raw_np
    masked = raw_np.copy()
    for dim_idx in range(masked.shape[1]):
        valid = quality_mask[:, dim_idx]
        if valid.any():
            fill_value = float(np.median(masked[valid, dim_idx]))
        else:
            fill_value = 0.0
        masked[~valid, dim_idx] = fill_value
    return masked








def _get_existing_asset_context(app_ref: FastAPI, asset_id: str) -> Optional[AssetContext]:
    with app_ref.state.asset_contexts_lock:
        return app_ref.state.asset_contexts.get(asset_id)


def _diagnostics_hint_from_analysis(analysis_result: Any) -> Dict[str, Any]:
    diagnostics_obj = getattr(analysis_result, "diagnostics", None)
    if diagnostics_obj is None:
        return {}
    status = (
        diagnostics_obj.status.value
        if isinstance(getattr(diagnostics_obj, "status", None), Enum)
        else str(getattr(diagnostics_obj, "status", "UNKNOWN"))
    )
    failure_type = (
        diagnostics_obj.failure_type.value
        if isinstance(getattr(diagnostics_obj, "failure_type", None), Enum)
        else str(getattr(diagnostics_obj, "failure_type", "unknown_anomaly"))
    )
    ts = _safe_float(getattr(diagnostics_obj, "timestamp", time.time()), default=time.time())
    return {
        "event_id": f"evt-{int(ts * 1000)}",
        "status": _normalize_status_text(status, fallback="UNKNOWN"),
        "classifier_severity": _normalize_status_text(status, fallback="UNKNOWN"),
        "root_cause_label": str(getattr(diagnostics_obj, "report_text", "") or "").strip(),
        "failure_type": str(failure_type or "unknown_anomaly"),
        "primary_fault_dimension": str(
            getattr(diagnostics_obj, "primary_fault_dimension", "") or "unknown_sensor"
        ).strip(),
    }


def _build_artifact_dedup_key(asset_id: str, diagnostics_hint: Dict[str, Any]) -> str:
    root_cause = str(
        diagnostics_hint.get("root_cause_label")
        or diagnostics_hint.get("failure_type")
        or "unknown"
    ).strip().lower()
    severity = str(
        diagnostics_hint.get("classifier_severity")
        or diagnostics_hint.get("status")
        or "unknown"
    ).strip().lower()
    primary_dimension = str(
        diagnostics_hint.get("primary_fault_dimension")
        or "unknown_sensor"
    ).strip().lower()
    return f"{asset_id}|{root_cause}|{severity}|{primary_dimension}"


def _artifact_mode_from_watermark(queue_watermark: str) -> str:
    mode = str(queue_watermark or "normal").strip().lower()
    if mode == "protect":
        return "summary-only"
    if mode in {"degrade", "warn"}:
        return "lite"
    return "full"


def _serialize_plot_data(plot_data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(plot_data, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in ("recon_error_map", "heatmap", "input_window", "dimensions", "primary_dim_idx", "details"):
        if key not in plot_data:
            continue
        value = plot_data.get(key)
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def _extract_latest_plot_payload(app_ref: FastAPI, asset_id: str) -> Dict[str, Any]:
    ctx = _get_existing_asset_context(app_ref, asset_id)
    if ctx is None:
        return {}
    with ctx.lock:
        buffer = getattr(ctx.engine, "anomaly_buffer", [])
        if not buffer:
            return {}
        item = buffer.pop()  # keep the newest anomaly payload only
        if buffer:
            buffer.clear()
        return _serialize_plot_data(item.get("plot_data") or {})


def _save_dual_attention_map(save_path: Path, sensor_name: str, signal: np.ndarray, cam: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    fig.patch.set_facecolor("#101216")

    axes[0].set_title(f"{sensor_name} Signal")
    axes[0].plot(signal, color="#4FC3F7", linewidth=1.8)
    axes[0].grid(alpha=0.25)

    heat = np.asarray(cam, dtype=np.float32).reshape(1, -1)
    axes[1].set_title("Attention Heatmap")
    im = axes[1].imshow(heat, aspect="auto", cmap="inferno", vmin=0.0, vmax=1.0)
    axes[1].set_yticks([])
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_artifact_job_payload(job: Dict[str, Any]) -> Dict[str, Any]:
    render_mode = str(job.get("render_mode", "full")).strip().lower()
    plot_data = job.get("plot_data") if isinstance(job.get("plot_data"), dict) else {}
    event_id = str(job.get("event_id") or f"evt-{int(time.time() * 1000)}")
    output_root = Path(str(job.get("output_root") or "runtime_outputs"))
    wavescan_dir = output_root / "wavescan_panels"
    heatmap_dir = output_root / "grad_cam_heatmaps"
    report_dir = output_root / "anomaly_reports"
    wavescan_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    details = plot_data.get("details") if isinstance(plot_data.get("details"), dict) else {}
    dimensions = plot_data.get("dimensions")
    if not isinstance(dimensions, list):
        dimensions = list(DIMENSIONS)
    primary_idx = int(_safe_float(plot_data.get("primary_dim_idx"), default=0.0))
    primary_idx = max(0, min(primary_idx, max(0, len(dimensions) - 1)))
    sensor_name = str(
        job.get("primary_dimension")
        or (dimensions[primary_idx] if dimensions else "unknown_sensor")
    )

    input_window = np.asarray(plot_data.get("input_window", []), dtype=np.float32)
    signal = np.asarray([], dtype=np.float32)
    if input_window.ndim == 3 and input_window.shape[0] > 0 and input_window.shape[2] > primary_idx:
        signal = np.asarray(input_window[0, :, primary_idx], dtype=np.float32)
    elif input_window.ndim == 2 and input_window.shape[1] > primary_idx:
        signal = np.asarray(input_window[:, primary_idx], dtype=np.float32)

    cam = np.asarray(plot_data.get("heatmap", []), dtype=np.float32).reshape(-1)
    if signal.size > 0:
        if cam.size <= 0:
            cam = np.zeros(signal.shape[0], dtype=np.float32)
        elif cam.size != signal.shape[0]:
            old_x = np.linspace(0.0, 1.0, cam.size)
            new_x = np.linspace(0.0, 1.0, signal.shape[0])
            cam = np.interp(new_x, old_x, cam).astype(np.float32)
        cam = np.clip(cam, 0.0, 1.0)

    image_files_written = 0
    rendered_images = 0
    if render_mode in {"full", "lite"} and signal.size > 0:
        panel_path = wavescan_dir / f"panel_{event_id}.png"
        visualize_wavescan_panel(
            sensor_name,
            signal,
            cam if cam.size > 0 else None,
            str(panel_path),
            extra_metrics=details,
        )
        image_files_written += 1
        rendered_images += 1

    if render_mode == "full" and signal.size > 0 and cam.size > 0:
        heatmap_path = heatmap_dir / f"attention_map_{event_id}.png"
        _save_dual_attention_map(heatmap_path, sensor_name, signal, cam)
        image_files_written += 1
        rendered_images += 1

    summary_payload = {
        "event_id": event_id,
        "asset_id": str(job.get("asset_id") or ""),
        "render_mode": render_mode,
        "queued_at_ts": _safe_float(job.get("queued_at_ts"), default=time.time()),
        "processed_at_ts": time.time(),
        "deduped": bool(job.get("deduped", False)),
        "queue_watermark": str(job.get("queue_watermark") or "normal"),
        "diagnostics": job.get("diagnostics") if isinstance(job.get("diagnostics"), dict) else {},
        "details": details,
        "rendered_images": rendered_images,
    }
    json_path = report_dir / f"consolidated_alerts_{event_id}.json"
    json_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False, default=_np_json_default),
        encoding="utf-8",
    )
    return {
        "alerts_written": 1,
        "image_files_written": image_files_written,
        "json_files_written": 1,
        "total_files_written": image_files_written + 1,
        "rendered_images": rendered_images,
    }


def _execute_artifact_job(app_ref: FastAPI, job: Dict[str, Any]) -> None:
    queued_at_ts = _safe_float(job.get("queued_at_ts"), default=time.time())
    lag_ms = max(0.0, (time.time() - queued_at_ts) * 1000.0)
    ttl_sec = _safe_float(job.get("ttl_sec"), default=float(app_ref.state.artifact_job_ttl_sec))
    if ttl_sec > 0 and (lag_ms / 1000.0) > ttl_sec:
        job["render_mode"] = "summary-only"
        job["expired_ttl"] = True

    flush_stats = render_artifact_job_payload(job)
    app_ref.state.metrics.record_disk_writes(flush_stats.get("total_files_written", 0))
    if bool(job.get("deduped", False)):
        app_ref.state.metrics.record_artifact_deduped()
    elif int(flush_stats.get("rendered_images", 0)) > 0:
        app_ref.state.metrics.record_artifact_rendered(lag_ms=lag_ms)
    else:
        app_ref.state.metrics.record_artifact_skipped()
    app_ref.state.artifact_last_job_ts = time.time()


def _requeue_expired_artifact_claims(app_ref: FastAPI) -> None:
    inflight = getattr(app_ref.state, "artifact_claims_inflight", {})
    if not isinstance(inflight, dict) or not inflight:
        return
    claim_ttl = float(getattr(app_ref.state, "artifact_claim_ttl_sec", 120.0))
    now_ts = time.time()
    expired_ids: List[str] = []
    for claim_id, row in list(inflight.items()):
        if not isinstance(row, dict):
            expired_ids.append(claim_id)
            continue
        claimed_at = _safe_float(row.get("claimed_at_ts"), default=now_ts)
        if (now_ts - claimed_at) > claim_ttl:
            expired_ids.append(claim_id)
    for claim_id in expired_ids:
        row = inflight.pop(claim_id, None)
        if not isinstance(row, dict):
            continue
        job = row.get("job")
        if isinstance(job, dict):
            try:
                app_ref.state.report_queue.put_nowait(job)
            except queue.Full:
                app_ref.state.metrics.record_queue_drop()
        try:
            app_ref.state.report_queue.task_done()
        except Exception:
            pass


def _report_worker(app_ref: FastAPI):
    while not app_ref.state.report_worker_stop.is_set():
        queue_ref = app_ref.state.report_queue
        try:
            job = queue_ref.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            if isinstance(job, dict):
                _execute_artifact_job(app_ref, job)
            else:
                # Legacy fallback: old queue payloads may contain asset_id only.
                asset_id = str(job)
                ctx = _get_existing_asset_context(app_ref, asset_id)
                if ctx is not None:
                    with ctx.lock:
                        flush_stats = ctx.engine.flush_reports()
                    app_ref.state.metrics.record_disk_writes(flush_stats.get("total_files_written", 0))
                    app_ref.state.metrics.record_artifact_rendered()
        except Exception:
            app_ref.state.metrics.record_artifact_failed()
            logger.exception("report_worker_failed")
        finally:
            queue_ref.task_done()


def _load_scaler_params(
    scaler_path: Path,
    *,
    expected_dimensions: Optional[List[str]] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not scaler_path.exists():
        logger.warning("scaler_params_missing path=%s", scaler_path)
        return None, None
    try:
        with scaler_path.open("rb") as fp:
            params = pickle.load(fp)
        mean = np.asarray(params.get("mean", []), dtype=np.float32)
        scale = np.asarray(params.get("scale", []), dtype=np.float32)
        expected_size = len(expected_dimensions or DIMENSIONS)
        if mean.size != expected_size or scale.size != expected_size:
            logger.warning(
                "scaler_params_shape_mismatch mean=%s scale=%s expected=%d",
                mean.size,
                scale.size,
                expected_size,
            )
            return None, None
        scale = np.where(np.abs(scale) < 1e-8, 1.0, scale)
        return mean, scale
    except Exception:
        logger.exception("scaler_params_load_failed path=%s", scaler_path)
        return None, None


def _checkpoint_input_dim_from_state_dict(state_dict: Dict[str, Any]) -> Optional[int]:
    probe = state_dict.get("mae_to_rnn.weight")
    if isinstance(probe, torch.Tensor) and probe.ndim == 2:
        return int(probe.shape[0])
    probe = state_dict.get("final_projection.2.weight")
    if isinstance(probe, torch.Tensor) and probe.ndim == 2:
        return int(probe.shape[0])
    return None


def _load_release_manifest(
    manifest_path: Path,
    *,
    checkpoint_path: Path,
    scaler_path: Path,
) -> Dict[str, Any]:
    if not manifest_path.exists():
        raise RuntimeError(
            f"Calibrator release manifest missing (fail-closed startup): {manifest_path}"
        )
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Calibrator release manifest unreadable: {manifest_path} ({exc})") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"Calibrator release manifest invalid object: {manifest_path}")
    if not bool(manifest.get("accepted", False)):
        raise RuntimeError(
            "Calibrator release manifest not accepted (fail-closed startup): "
            f"path={manifest_path} issues={manifest.get('issues', [])}"
        )
    if str(manifest.get("checkpoint_path", "")) and Path(str(manifest["checkpoint_path"])).name != checkpoint_path.name:
        raise RuntimeError(
            "Calibrator release manifest checkpoint mismatch (fail-closed startup): "
            f"manifest={manifest.get('checkpoint_path')} runtime={checkpoint_path}"
        )
    if str(manifest.get("scaler_path", "")) and Path(str(manifest["scaler_path"])).name != scaler_path.name:
        raise RuntimeError(
            "Calibrator release manifest scaler mismatch (fail-closed startup): "
            f"manifest={manifest.get('scaler_path')} runtime={scaler_path}"
        )
    return manifest


def _checkpoint_dimension_layout(
    checkpoint: Dict[str, Any],
    *,
    manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[str]]:
    manifest = dict(manifest or {})
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    inferred_input_dim = _checkpoint_input_dim_from_state_dict(state_dict) or 0
    manifest_train_dimensions = list(
        manifest.get("model_train_dimensions")
        or manifest.get("train_enabled_dimensions")
        or []
    )
    manifest_accepted_dimensions = list(manifest.get("accepted_dimensions") or [])
    manifest_context_dimensions = list(manifest.get("context_only_dimensions") or [])

    if inferred_input_dim == len(CANONICAL_DIMENSION_LIST):
        inferred_train_dimensions = list(CANONICAL_DIMENSION_LIST)
    elif inferred_input_dim == len(LEGACY_CORE_DIMENSIONS):
        inferred_train_dimensions = list(LEGACY_CORE_DIMENSIONS)
    elif inferred_input_dim == len(CORE_DIMENSIONS):
        inferred_train_dimensions = list(CORE_DIMENSIONS)
    else:
        inferred_train_dimensions = []

    model_train_dimensions = list(
        checkpoint.get("model_train_dimensions")
        or checkpoint.get("train_enabled_dimensions")
        or manifest_train_dimensions
        or inferred_train_dimensions
        or CORE_DIMENSIONS
    )
    accepted_dimensions = list(
        checkpoint.get("accepted_dimensions")
        or manifest_accepted_dimensions
        or CANONICAL_DIMENSION_LIST
    )
    context_only_dimensions = list(
        checkpoint.get("context_only_dimensions")
        or manifest_context_dimensions
        or [dim for dim in accepted_dimensions if dim not in model_train_dimensions]
    )
    model_train_dimensions = [dim for dim in model_train_dimensions if dim in accepted_dimensions]
    if not model_train_dimensions:
        raise RuntimeError("Calibrator checkpoint/manifest missing model_train_dimensions")
    return {
        "model_train_dimensions": model_train_dimensions,
        "accepted_dimensions": accepted_dimensions,
        "context_only_dimensions": context_only_dimensions,
    }


def _load_runtime_adjacency(data_dir: Path, *, dimensions: Optional[List[str]] = None) -> Optional[torch.Tensor]:
    try:
        target_dimensions = list(dimensions or DIMENSIONS)
        frames = []
        for sid in (1, 2, 3):
            csv_path = data_dir / f"sensor_{sid}_advanced_beta_25.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            required_cols = []
            for dim_name in target_dimensions:
                col_name = f"{dim_name}_drift"
                if dim_name == "external_pressure" and "pressure_drift" in df.columns:
                    col_name = "pressure_drift"
                required_cols.append(col_name)
            if any(col not in df.columns for col in required_cols):
                continue
            frames.append(df[required_cols].to_numpy(dtype=np.float32))

        if not frames:
            logger.warning("adjacency_build_skipped reason=no_valid_sensor_csv")
            return None

        drift_data = np.concatenate(frames, axis=0)
        graph = ComponentGraph(sensor_time_series=drift_data, correlation_threshold=0.85)
        return graph.get_adjacency_tensor().float()
    except Exception:
        logger.exception("adjacency_build_failed data_dir=%s", data_dir)
        return None


def _canonical_to_train_subset(matrix: np.ndarray, *, train_dimensions: List[str]) -> np.ndarray:
    indices = [CANONICAL_DIMENSION_INDEX[dim] for dim in train_dimensions]
    return matrix[:, indices].astype(np.float32, copy=True)


def _canonical_presence_to_train_subset(mask: Optional[np.ndarray], *, train_dimensions: List[str]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    indices = [CANONICAL_DIMENSION_INDEX[dim] for dim in train_dimensions]
    return mask[:, indices].astype(np.float32, copy=True)


def _merge_train_subset_into_canonical(
    raw_matrix: np.ndarray,
    calibrated_subset: np.ndarray,
    *,
    train_dimensions: List[str],
) -> np.ndarray:
    merged = raw_matrix.astype(np.float32, copy=True)
    for subset_idx, dim_name in enumerate(train_dimensions):
        dim_idx = CANONICAL_DIMENSION_INDEX[dim_name]
        merged[:, dim_idx] = calibrated_subset[:, subset_idx]
    return merged


def _guard_state_key(asset_id: str, runtime_mode: str, calibration_mode: str) -> str:
    return f"{asset_id}::{runtime_mode}::{calibration_mode}"


def _get_calibration_guard_state(app_ref: FastAPI, state_key: str) -> Dict[str, Any]:
    with app_ref.state.calibration_guard_lock:
        state = app_ref.state.calibration_guard_states.get(state_key)
        if state is None:
            state = {"step_count": 0, "alpha": 0.85}
            app_ref.state.calibration_guard_states[state_key] = state
        return dict(state)


def _set_calibration_guard_state(app_ref: FastAPI, state_key: str, state: Dict[str, Any]) -> None:
    state_copy = dict(state)
    state_copy["_updated_at"] = time.time()
    with app_ref.state.calibration_guard_lock:
        app_ref.state.calibration_guard_states[state_key] = state_copy
    _persist_runtime_state_if_due(app_ref)










def _meta_payload(
    request: Request,
    *,
    asset_id: str,
    timestamp: float,
    seq_len: int,
    sampling_hz: float,
    sampling_hz_effective: float,
    runtime_mode: str,
    processing_ms: float,
    mission_phase: Optional[str] = "unknown",
    calibration_mode_requested: str = "safe",
    calibration_mode_effective: str = "safe",
    fallback_applied: bool = False,
    masked_points_count: int = 0,
    source_dimension_map: Optional[Dict[str, str]] = None,
    internal_pressure_source: Optional[str] = None,
    internal_pressure_candidates: Optional[List[str]] = None,
    accepted_dimensions: Optional[List[str]] = None,
    observed_dimensions: Optional[List[str]] = None,
    model_train_dimensions: Optional[List[str]] = None,
    context_only_dimensions: Optional[List[str]] = None,
    core_dimensions: Optional[List[str]] = None,
    aux_dimensions: Optional[List[str]] = None,
    rejected_dimensions: Optional[List[str]] = None,
    unused_input_columns: Optional[List[str]] = None,
    asset_class: str = "generic",
    asset_norm_scope: str = "global",
    asset_norm_confidence: float = 0.0,
    norm_source: str = "global",
    asset_critical_dimensions: Optional[List[str]] = None,
    candidate_dimensions: Optional[List[str]] = None,
    observing_dimensions: Optional[List[str]] = None,
    provisional_dimensions: Optional[List[str]] = None,
    approved_dimensions: Optional[List[str]] = None,
    dimension_importance_by_asset: Optional[Dict[str, str]] = None,
    dimension_lifecycle_status: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    queue_snapshot = _queue_pressure_snapshot(request.app)
    inf_ref = getattr(request.app.state, "inference_queue", None)
    inf_depth = int(inf_ref.qsize()) if inf_ref is not None else 0
    metrics_snapshot = request.app.state.metrics.snapshot(
        report_queue_depth=queue_snapshot["queue_depth"],
        inference_queue_depth=inf_depth,
    )
    _update_traffic_activity_state(
        request.app,
        metrics_snapshot=metrics_snapshot,
        context="meta_payload",
    )
    active_traffic_node = _resolve_active_traffic_node(request.app, metrics_snapshot)
    return {
        "asset_id": asset_id,
        "received_timestamp": timestamp,
        "timestamp": _to_iso8601_utc(timestamp),
        "mission_phase": str(mission_phase or "unknown"),
        "seq_len": seq_len,
        "sampling_hz": sampling_hz,
        "sampling_hz_effective": float(sampling_hz_effective),
        "runtime_mode": runtime_mode,
        "calibration_mode_requested": calibration_mode_requested,
        "calibration_mode_effective": calibration_mode_effective,
        "fallback_applied": bool(fallback_applied),
        "processing_ms": round(float(processing_ms), 3),
        "data_policy": DATA_POLICY_LABEL,
        "data_dir": str(getattr(request.app.state, "canonical_data_dir", CANONICAL_BETA25_DATA_DIR)),
        "model_version_calibrator": getattr(request.app.state, "calibrator_model_version", "unknown"),
        "model_version_anomaly": getattr(request.app.state, "anomaly_model_version", "unknown"),
        "model_loaded": bool(getattr(request.app.state, "model_loaded", False)),
        "scaler_loaded": bool(getattr(request.app.state, "scaler_loaded", False)),
        "adj_loaded": bool(getattr(request.app.state, "adj_loaded", False)),
        "masked_points_count": int(masked_points_count),
        "kb_version": getattr(request.app.state, "kb_version", "unknown"),
        "model_version": getattr(request.app.state, "calibrator_model_version", "unknown"),
        "profile_version": getattr(request.app.state, "profile_version", "unknown"),
        "queue_depth": queue_snapshot["queue_depth"],
        "queue_maxsize": queue_snapshot["queue_maxsize"],
        "queue_watermark": queue_snapshot["queue_watermark"],
        "queue_warn_threshold": queue_snapshot["queue_warn_threshold"],
        "queue_degrade_threshold": queue_snapshot["queue_degrade_threshold"],
        "queue_protect_threshold": queue_snapshot["queue_protect_threshold"],
        "continuity_mode": str(getattr(request.app.state, "continuity_mode", "off")),
        "state_store_connected": bool(getattr(request.app.state, "state_store_connected", False)),
        "state_store_path": str(getattr(request.app.state, "state_store_path", "")),
        "state_store_ttl_sec": int(getattr(request.app.state, "state_store_ttl_sec", 0)),
        "traffic_node": str(getattr(request.app.state, "traffic_node", "unknown")),
        "traffic_role": str(getattr(request.app.state, "traffic_role", "unknown")),
        "active_traffic_node": active_traffic_node,
        "receiving_traffic_recently": bool(metrics_snapshot.get("receiving_traffic_recently", False)),
        "inference_profile": getattr(request.app.state, "inference_profile", "balanced"),
        "sensitivity_profile": getattr(request.app.state, "sensitivity_profile", "balanced"),
        "fast_model_enabled": bool(getattr(request.app.state, "fast_model_enabled", False)),
        "fault_export_path": (
            str(request.app.state.fault_exporter.output_path)
            if hasattr(request.app.state, "fault_exporter")
            else None
        ),
        "fault_export_format": (
            request.app.state.fault_exporter.export_format
            if hasattr(request.app.state, "fault_exporter")
            else None
        ),
        "source_dimension_map": dict(source_dimension_map or {}),
        "internal_pressure_source": str(internal_pressure_source or ""),
        "internal_pressure_candidates": list(internal_pressure_candidates or []),
        "accepted_dimensions": list(accepted_dimensions or []),
        "observed_dimensions": list(observed_dimensions or []),
        "model_train_dimensions": list(model_train_dimensions or []),
        "context_only_dimensions": list(context_only_dimensions or []),
        "core_dimensions": list(core_dimensions or []),
        "aux_dimensions": list(aux_dimensions or []),
        "rejected_dimensions": list(rejected_dimensions or []),
        "unused_input_columns": list(unused_input_columns or []),
        "asset_class": str(asset_class or "generic"),
        "asset_norm_scope": str(asset_norm_scope or "global"),
        "asset_norm_confidence": float(asset_norm_confidence),
        "norm_source": str(norm_source or "global"),
        "asset_critical_dimensions": list(asset_critical_dimensions or []),
        "candidate_dimensions": list(candidate_dimensions or []),
        "observing_dimensions": list(observing_dimensions or []),
        "provisional_dimensions": list(provisional_dimensions or []),
        "approved_dimensions": list(approved_dimensions or []),
        "dimension_importance_by_asset": dict(dimension_importance_by_asset or {}),
        "dimension_lifecycle_status": dict(dimension_lifecycle_status or {}),
        "memory_enabled": bool(getattr(request.app.state, "memory_enabled", False)),
        "memory_status": str(getattr(request.app.state, "memory_status", "unavailable")),
        "asset_identity_status": str(getattr(request.app.state, "asset_identity_status", "unavailable")),
    }


def _memory_unavailable_payload(asset_id: str) -> Dict[str, Any]:
    now = datetime.now(tz=timezone.utc).isoformat()
    return {
        "memory_snapshot": {
            "asset_id": asset_id,
            "timezone": "Europe/Istanbul",
            "generated_at": now,
            "memory_status": "unavailable",
            "dimensions": {},
        },
        "memory_context": {
            "asset_id": asset_id,
            "timezone": "Europe/Istanbul",
            "generated_at": now,
            "memory_status": "unavailable",
            "daypart": "unknown",
            "contexts": [],
        },
        "calibration_memory": {
            "asset_id": asset_id,
            "memory_status": "unavailable",
            "updated_at": now,
            "drift_summary": {},
        },
    }


def _current_values_from_matrix(matrix: np.ndarray) -> Dict[str, float]:
    if matrix.size == 0:
        return {}
    last_row = matrix[-1]
    values: Dict[str, float] = {}
    for idx, dim in enumerate(DIMENSIONS):
        if idx < len(last_row):
            values[dim] = float(last_row[idx])
    return values


def _current_values_from_series(series_by_dim: Optional[Dict[str, List[float]]]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for dim, rows in (series_by_dim or {}).items():
        if rows:
            values[dim] = float(rows[-1])
    return values


def _build_memory_outputs(
    request: Request,
    *,
    asset_id: str,
    timestamp: float | str,
    raw_matrix: np.ndarray,
    observed_series: Optional[Dict[str, List[float]]] = None,
    correction_matrix: np.ndarray | None,
    include_in_learning: bool,
) -> Dict[str, Any]:
    if not bool(getattr(request.app.state, "memory_enabled", False)):
        request.app.state.memory_status = "unavailable"
        return _memory_unavailable_payload(asset_id)

    service = getattr(request.app.state, "memory_service", None)
    if service is None:
        request.app.state.memory_status = "unavailable"
        return _memory_unavailable_payload(asset_id)

    try:
        if raw_matrix.size > 0:
            service.ingest_sensor_window(
                asset_id=asset_id,
                timestamp=timestamp,
                sensor_matrix=raw_matrix.tolist(),
                dimension_series=observed_series or _observed_series_from_matrix(raw_matrix, list(DIMENSIONS)),
                include_in_learning=include_in_learning,
            )
        if correction_matrix is not None and correction_matrix.size > 0:
            calibration_memory = service.update_calibration_memory(
                asset_id=asset_id,
                timestamp=timestamp,
                correction_matrix=correction_matrix.tolist(),
            )
        else:
            calibration_memory = service.get_calibration_memory(asset_id)

        memory_snapshot = service.get_norm_snapshot(asset_id)
        memory_context = service.get_context(
            asset_id=asset_id,
            timestamp=timestamp,
            current_values=_current_values_from_series(
                observed_series or _observed_series_from_matrix(raw_matrix, list(DIMENSIONS))
            ),
        )
        request.app.state.memory_status = "available"
        return {
            "memory_snapshot": memory_snapshot,
            "memory_context": memory_context,
            "calibration_memory": calibration_memory,
        }
    except Exception:
        request.app.state.memory_status = "unavailable"
        logger.exception("memory_service_failed asset_id=%s", asset_id)
        return _memory_unavailable_payload(asset_id)






def _is_cauren_runtime_state(request: Request) -> bool:
    return getattr(request.app.state, "runtime_architecture", "") == "cauren_core_civil_agent"


def _sync_cauren_calibrate(request: Request, payload: CalibrateIn, explain: bool):
    t0 = time.perf_counter()
    try:
        asset_id_input = payload.asset_id
        if not str(asset_id_input or "").strip() and _is_cauren_runtime_state(request):
            asset_id_input = "cauren_asset"
        resolution = _resolve_asset_identity(
            request,
            asset_id=asset_id_input,
            site_id=payload.site_id,
            line_id=payload.line_id,
            machine_id=payload.machine_id,
        )
        runtime_mode, sampling_hz_effective = _validate_common_fields(
            resolution.resolved_asset_id,
            float(payload.timestamp),
            int(payload.seq_len),
            float(payload.sampling_hz),
            payload.runtime_mode,
        )
        cauren_payload = {
            "asset_id": resolution.resolved_asset_id,
            "agent_id": payload.agent_id,
            "sector": payload.sector,
            "client_id": payload.client_id,
            "site_context": payload.site_context or {},
            "forecast_context": payload.forecast_context or {},
            "control_mode": payload.control_mode or "guarded_auto",
            "sensor_schema_version": payload.sensor_schema_version,
            "timestamp": float(payload.timestamp),
            "seq_len": int(payload.seq_len),
            "sampling_hz": float(sampling_hz_effective),
            "runtime_mode": runtime_mode,
            "mission_phase": payload.mission_phase,
            "sensors": payload.sensors or [],
            "ingress_tag": payload.ingress_tag,
            "ingress_tags": payload.ingress_tags or [],
            "sector_hint": payload.sector_hint,
            "sector_hint_confidence": payload.sector_hint_confidence,
            "source_metadata": payload.source_metadata or {},
            "site_id": payload.site_id,
            "line_id": payload.line_id,
            "machine_id": payload.machine_id,
        }
        result = _get_cauren_pipeline(request).calibrate(cauren_payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        request.app.state.metrics.record_request(elapsed_ms, success=True, anomaly=False)
        meta = {
            "architecture": "cauren_core_civil_agent",
            "asset_id": resolution.resolved_asset_id,
            "client_id": payload.client_id,
            "mission_phase": payload.mission_phase,
            "runtime_mode": runtime_mode,
            "sampling_hz_effective": sampling_hz_effective,
            "processing_ms": round(float(elapsed_ms), 3),
            "agent_id_requested": payload.agent_id,
            "sector_requested": payload.sector,
            "sensor_schema_version": payload.sensor_schema_version,
            "calibration_mode": "cauren_core_agent_schema",
                        "explain": bool(explain),
        }
        meta.update(_asset_resolution_meta(resolution))
        return {
            "calibrated_data": result["calibrated_data"],
            "feature_names": result["feature_names"],
            "feature_metadata": result["feature_metadata"],
            "presence_mask": result["presence_mask"],
            "timestamps": result["timestamps"],
            "sector_scores": result.get("sector_scores", []),
            "sector_score_breakdown": result.get("sector_score_breakdown", []),
            "selected_sector_score": result.get("selected_sector_score", 0.0),
            "selected_sector_prior": result.get("selected_sector_prior", 0.0),
            "sector_selection_strategy": result.get("sector_selection_strategy", "prior_weighted_score_fusion"),
            "sector_selection_needs_context": result.get("sector_selection_needs_context", False),
            "sector_prior_conflict": result.get("sector_prior_conflict", False),
            "sector_prior_conflict_reason": result.get("sector_prior_conflict_reason", ""),
            "quality": {
                "mode": "cauren_core_agent_schema",
                "raw_sensor_count": result["raw_sensor_count"],
                "rejected_sample_count": len(result["rejected_samples"]),
                "router_confidence": result["confidence"],
            },
            "selected_agent": result["selected_agent"],
            "candidate_agents": result["candidate_agents"],
            "rejected_samples": result["rejected_samples"],
            "meta": meta,
        }
    except HTTPException:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        request.app.state.metrics.record_request(elapsed_ms, success=False, anomaly=False)
        raise
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        request.app.state.metrics.record_request(elapsed_ms, success=False, anomaly=False)
        logger.exception("cauren_calibrate_failed asset_id=%s", getattr(payload, "asset_id", "unknown"))
        raise HTTPException(status_code=500, detail=str(e))




def _get_cauren_pipeline(request: Request):
    pipeline = getattr(request.app.state, "cauren_pipeline", None)
    if pipeline is not None:
        return pipeline
    from cauren_core import CaurenPipeline

    pipeline = CaurenPipeline.from_default_registry()
    request.app.state.cauren_pipeline = pipeline
    return pipeline


def _uses_cbs_building_payload(payload: DiagnoseIn) -> bool:
    return bool(
        str(payload.building_id or "").strip()
        or payload.geometry
        or payload.project_permit
        or payload.construction_status
        or payload.infrastructure_connections
        or payload.risk_assessments
        or payload.inspection_findings
    )


def _score_from_mapping(data: Optional[Dict[str, Any]], keys: tuple[str, ...], default: float) -> float:
    if not isinstance(data, dict):
        return default
    for key in keys:
        if key not in data:
            continue
        value = data.get(key)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            text = str(value or "").strip().lower()
            if text in {"ok", "valid", "approved", "uygun", "tamam", "ready", "complete"}:
                return 1.0
            if text in {"missing", "invalid", "rejected", "eksik", "uygunsuz", "risk", "not_ready"}:
                return 0.0
            continue
        if parsed > 1.0:
            parsed = parsed / 100.0
        return max(0.0, min(1.0, parsed))
    return default


def _inspection_score(items: Optional[List[Dict[str, Any]]]) -> float:
    if not items:
        return 0.15
    scores: list[float] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        scores.append(
            _score_from_mapping(
                item,
                ("risk_score", "severity_score", "finding_score", "score", "risk"),
                0.35,
            )
        )
    return max(scores) if scores else 0.35


def _building_sensors_from_payload(payload: DiagnoseIn | CBSBuildingRecord) -> list[dict[str, Any]]:
    existing = getattr(payload, "sensors", None)
    if existing:
        return list(existing)
    risk = getattr(payload, "risk_assessments", None)
    permit = getattr(payload, "project_permit", None)
    status = getattr(payload, "construction_status", None)
    infra = getattr(payload, "infrastructure_connections", None)
    findings = getattr(payload, "inspection_findings", None)
    timestamp = time.time()
    progress = _score_from_mapping(status, ("progress_pct", "completion_pct", "construction_progress_pct"), 0.0) * 100.0
    values = {
        "structural_risk_score": _score_from_mapping(
            risk,
            ("structural_risk_score", "structural_risk", "building_risk_score", "risk_score"),
            0.25,
        ),
        "inspection_finding_score": _inspection_score(findings),
        "permit_status_score": _score_from_mapping(
            permit,
            ("permit_status_score", "document_readiness_score", "approval_score", "approved"),
            0.5,
        ),
        "construction_progress_pct": progress,
        "infrastructure_connection_score": _score_from_mapping(
            infra,
            ("infrastructure_connection_score", "readiness_score", "utility_connection_score", "ready"),
            0.5,
        ),
        "natural_hazard_score": _score_from_mapping(
            risk,
            ("natural_hazard_score", "earthquake_risk_score", "flood_risk_score", "natural_risk_score"),
            0.2,
        ),
        "occupancy_safety_score": _score_from_mapping(
            risk,
            ("occupancy_safety_score", "life_safety_score", "occupancy_risk_score"),
            0.2,
        ),
        "ground_stability_score": _score_from_mapping(
            risk,
            ("ground_stability_score", "ground_risk_score", "geotechnical_risk_score"),
            0.2,
        ),
    }
    return [
        {
            "sensor_id": f"{getattr(payload, 'building_id', 'building')}_{name}",
            "name": name,
            "unit": "%" if name == "construction_progress_pct" else "ratio",
            "value": round(float(value), 6),
            "timestamp": timestamp,
            "quality": True,
        }
        for name, value in values.items()
    ]


def _building_site_context(payload: DiagnoseIn | CBSBuildingRecord, *, source: str = "diagnose") -> dict[str, Any]:
    return {
        "source": source,
        "domain": "asce jcce civil engineering building construction infrastructure cadastre permit inspection",
        "building_id": getattr(payload, "building_id", None),
        "geometry": getattr(payload, "geometry", None) or {},
        "address": getattr(payload, "address", None) or {},
        "administrative_unit": getattr(payload, "administrative_unit", None) or {},
        "project_permit": getattr(payload, "project_permit", None) or {},
        "construction_status": getattr(payload, "construction_status", None) or {},
        "infrastructure_connections": getattr(payload, "infrastructure_connections", None) or {},
        "risk_assessments": getattr(payload, "risk_assessments", None) or {},
        "cadastral_reference": getattr(payload, "cadastral_reference", None) or {},
    }


def _cbs_job_store(app_ref: FastAPI) -> dict[str, dict[str, Any]]:
    store = getattr(app_ref.state, "cbs_jobs", None)
    if isinstance(store, dict):
        return store
    app_ref.state.cbs_jobs = {}
    return app_ref.state.cbs_jobs


def _cbs_idempotency_index(app_ref: FastAPI) -> dict[str, str]:
    index = getattr(app_ref.state, "cbs_idempotency_index", None)
    if isinstance(index, dict):
        return index
    app_ref.state.cbs_idempotency_index = {}
    return app_ref.state.cbs_idempotency_index


def _cbs_job_id(idempotency_key: str) -> str:
    digest = hashlib.sha256(str(idempotency_key).encode("utf-8")).hexdigest()[:16]
    return f"cbs-job-{digest}"


def _record_unknown_fields(record: CBSBuildingRecord) -> list[str]:
    extras = getattr(record, "model_extra", None)
    if isinstance(extras, dict):
        return sorted(str(key) for key in extras)
    return []


def _validate_cbs_record(record: CBSBuildingRecord) -> list[str]:
    errors: list[str] = []
    if not str(record.building_id or "").strip():
        errors.append("building_id_missing")
    if not isinstance(record.geometry, dict) or not record.geometry:
        errors.append("geometry_missing")
    if not isinstance(record.address, dict) or not record.address:
        errors.append("address_missing")
    if not isinstance(record.administrative_unit, dict) or not record.administrative_unit:
        errors.append("administrative_unit_missing")
    if not isinstance(record.project_permit, dict) or not record.project_permit:
        errors.append("project_permit_missing")
    if not isinstance(record.construction_status, dict) or not record.construction_status:
        errors.append("construction_status_missing")
    return errors


def _process_cbs_building_batch(request: Request, payload: CBSBuildingBatchIn) -> dict[str, Any]:
    if len(payload.records) > 10000:
        raise HTTPException(status_code=413, detail="CBS building batch limit is 10000 records")
    index = _cbs_idempotency_index(request.app)
    store = _cbs_job_store(request.app)
    key = str(payload.idempotency_key or "").strip()
    if not key:
        raise HTTPException(status_code=422, detail="idempotency_key is required")
    if key in index and index[key] in store:
        previous = copy.deepcopy(store[index[key]])
        previous["idempotent_replay"] = True
        return previous

    job_id = _cbs_job_id(key)
    started = time.perf_counter()
    pipeline = _get_cauren_pipeline(request)
    accepted = 0
    rejected = 0
    risky = 0
    failures: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    sample_results: list[dict[str, Any]] = []

    for idx, record in enumerate(payload.records):
        errors = _validate_cbs_record(record)
        unknown_fields = _record_unknown_fields(record)
        if errors:
            rejected += 1
            failures.append(
                {
                    "index": idx,
                    "building_id": record.building_id,
                    "errors": errors,
                    "unknown_fields": unknown_fields,
                }
            )
            continue
        cauren_payload = {
            "asset_id": record.building_id,
            "agent_id": "cauren-civil",
            "sector": "civil",
            "client_id": payload.source,
            "site_context": _building_site_context(record, source=str(payload.source or "tucbs")),
            "timestamp": time.time(),
            "seq_len": SEQ_MIN,
            "sampling_hz": 1.0,
            "runtime_mode": None,
            "mission_phase": "government_cbs_building_batch",
            "sensors": _building_sensors_from_payload(record),
        }
        try:
            diagnosis = pipeline.diagnose(cauren_payload)
        except Exception as exc:
            rejected += 1
            failures.append(
                {
                    "index": idx,
                    "building_id": record.building_id,
                    "errors": [f"diagnosis_failed:{exc}"],
                    "unknown_fields": unknown_fields,
                }
            )
            continue
        accepted += 1
        risk_score = float(diagnosis.risk_score)
        if risk_score >= 0.65:
            risky += 1
        if unknown_fields:
            review_queue.append(
                {
                    "building_id": record.building_id,
                    "reason": "unknown_fields_preserved",
                    "unknown_fields": unknown_fields,
                }
            )
        if len(sample_results) < 20:
            sample_results.append(
                {
                    "building_id": record.building_id,
                    "selected_agent": diagnosis.selected_agent,
                    "risk_score": round(risk_score, 6),
                    "review_required": bool(risk_score >= 0.65 or unknown_fields),
                    "unknown_fields": unknown_fields,
                }
            )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    job = {
        "job_id": job_id,
        "status": "completed",
        "idempotency_key": key,
        "source": payload.source,
        "submitted_records": len(payload.records),
        "accepted_records": accepted,
        "rejected_records": rejected,
        "risky_records": risky,
        "processing_ms": round(elapsed_ms, 3),
        "failures": failures[:200],
        "failure_count": len(failures),
        "review_queue": review_queue[:200],
        "review_queue_count": len(review_queue),
        "sample_results": sample_results,
        "contract": {
            "max_records": 10000,
            "required_domains": ["building", "address", "administrative_unit", "permit", "construction_status"],
            "agent": "cauren-civil",
            "llm_decision_authority": "decision_support_only",
        },
        "idempotent_replay": False,
    }
    store[job_id] = copy.deepcopy(job)
    index[key] = job_id
    return job


@app.post("/cbs/buildings/batch")
async def cbs_buildings_batch(request: Request, payload: CBSBuildingBatchIn):
    if not payload.records:
        raise HTTPException(status_code=422, detail="records must contain at least one CBS building record")
    _enforce_backpressure_or_raise(request, "/cbs/buildings/batch")
    token = _acquire_inference_slot_or_raise(request, "/cbs/buildings/batch")
    try:
        return await run_in_threadpool(_process_cbs_building_batch, request, payload)
    finally:
        _release_inference_slot(request, token)


@app.get("/cbs/jobs/{job_id}")
async def cbs_job_status(request: Request, job_id: str):
    store = _cbs_job_store(request.app)
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="cbs job not found")
    return copy.deepcopy(job)


def _sync_cauren_diagnose(request: Request, payload: DiagnoseIn, explain: bool):
    t0 = time.perf_counter()
    try:
        asset_id_input = payload.asset_id
        if not str(asset_id_input or "").strip() and getattr(
            request.app.state,
            "runtime_architecture",
            "",
        ) == "cauren_core_civil_agent":
            asset_id_input = "cauren_asset"
        resolution = _resolve_asset_identity(
            request,
            asset_id=asset_id_input,
            site_id=payload.site_id,
            line_id=payload.line_id,
            machine_id=payload.machine_id,
        )
        runtime_mode, sampling_hz_effective = _validate_common_fields(
            resolution.resolved_asset_id,
            float(payload.timestamp),
            int(payload.seq_len),
            float(payload.sampling_hz),
            payload.runtime_mode,
        )
        cauren_payload = {
            "asset_id": resolution.resolved_asset_id,
            "agent_id": "cauren-civil" if _uses_cbs_building_payload(payload) else payload.agent_id,
            "sector": "civil" if _uses_cbs_building_payload(payload) else payload.sector,
            "client_id": payload.client_id,
            "site_context": (
                _building_site_context(payload)
                if _uses_cbs_building_payload(payload)
                else payload.site_context or {}
            ),
            "sensor_schema_version": payload.sensor_schema_version,
            "timestamp": float(payload.timestamp),
            "seq_len": int(payload.seq_len),
            "sampling_hz": float(sampling_hz_effective),
            "runtime_mode": runtime_mode,
            "mission_phase": payload.mission_phase,
            "sensors": _building_sensors_from_payload(payload) if _uses_cbs_building_payload(payload) else payload.sensors or [],
            "ingress_tag": payload.ingress_tag,
            "ingress_tags": payload.ingress_tags or [],
            "sector_hint": payload.sector_hint,
            "sector_hint_confidence": payload.sector_hint_confidence,
            "source_metadata": payload.source_metadata or {},
            "site_id": payload.site_id,
            "line_id": payload.line_id,
            "machine_id": payload.machine_id,
        }
        pipeline = _get_cauren_pipeline(request)
        operational_memory = getattr(request.app.state, "operational_memory_store", None)
        layout_key = ""
        operational_lookup = "disabled"
        operational_used = False
        operational_fallback_reason = ""
        result = None

        operational_memory_enabled = operational_memory is not None and _is_operational_memory_request_eligible(payload)
        if operational_memory_enabled:
            from cauren_core.adapter import sensor_layout_key

            layout_key = sensor_layout_key(cauren_payload["sensors"])
            memory_entry, operational_lookup = operational_memory.lookup(
                asset_id=resolution.resolved_asset_id,
                sensor_layout_key=layout_key,
                agent_registry_version=getattr(pipeline, "registry_version", ""),
                requested_agent_id=str(payload.agent_id or "").strip() or None,
                requested_sector=str(payload.sector or "").strip() or None,
            )
            if operational_lookup in {"expired", "registry_version_mismatch"}:
                request.app.state.metrics.record_operational_memory_invalidation()
            if memory_entry is not None:
                hot_started = time.perf_counter()
                hot_result = pipeline.diagnose_with_preselected_route(
                    cauren_payload,
                    selected_agent_id=memory_entry.selected_agent,
                    route_confidence=memory_entry.route_confidence,
                    route_reason=memory_entry.route_reason,
                    route_needs_context=memory_entry.route_needs_context,
                    route_ambiguity_reason=memory_entry.route_ambiguity_reason,
                    candidate_agents=list(memory_entry.candidate_agents),
                    sector_score_breakdown=list(memory_entry.sector_score_breakdown),
                    selected_sector_score=float(memory_entry.selected_sector_score),
                    selected_sector_prior=float(memory_entry.selected_sector_prior),
                    selection_strategy=memory_entry.selection_strategy,
                    prior_conflict=bool(memory_entry.prior_conflict),
                    prior_conflict_reason=memory_entry.prior_conflict_reason,
                )
                hot_body = hot_result.to_dict()
                rejected_ratio = (
                    len(hot_body.get("rejected_samples") or [])
                    / float(max(1, len(cauren_payload["sensors"])))
                )
                if (
                    hot_body.get("selected_agent") == memory_entry.selected_agent
                    and not bool(hot_body.get("route_needs_context", False))
                    and rejected_ratio <= 0.35
                ):
                    operational_used = True
                    request.app.state.metrics.record_operational_memory_hit(
                        latency_ms=(time.perf_counter() - hot_started) * 1000.0
                    )
                    operational_memory.mark_hot_path_success(
                        asset_id=resolution.resolved_asset_id,
                        sensor_layout_key=layout_key,
                    )
                    result = hot_result
                else:
                    operational_fallback_reason = "hot_path_validation_failed"
                    request.app.state.metrics.record_operational_memory_safe_fallback()
                    operational_memory.note_fallback(
                        asset_id=resolution.resolved_asset_id,
                        sensor_layout_key=layout_key,
                        reason=operational_fallback_reason,
                        invalidate=True,
                    )
                    request.app.state.metrics.record_operational_memory_invalidation()

        if result is None:
            full_started = time.perf_counter()
            result = pipeline.diagnose(cauren_payload)
            if operational_memory_enabled:
                request.app.state.metrics.record_operational_memory_miss(
                    latency_ms=(time.perf_counter() - full_started) * 1000.0
                )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        is_anomaly = result.anomaly_type != "nominal_variation"
        request.app.state.metrics.record_request(elapsed_ms, success=True, anomaly=is_anomaly)
        body = result.to_dict()
        if (
            operational_memory_enabled
            and layout_key
            and _operational_memory_recordable(payload, body)
        ):
            schema = pipeline.registry.get(result.selected_agent).schema
            rejected_ratio = len(body.get("rejected_samples") or []) / float(max(1, len(cauren_payload["sensors"])))
            operational_memory.remember_success(
                asset_id=resolution.resolved_asset_id,
                sensor_layout_key=layout_key,
                selected_agent=result.selected_agent,
                agent_schema_version=schema.version,
                agent_registry_version=getattr(pipeline, "registry_version", ""),
                route_confidence=float(body.get("route_confidence", 0.0) or 0.0),
                route_reason=str(body.get("route_reason") or ""),
                route_needs_context=bool(body.get("route_needs_context", False)),
                route_ambiguity_reason=str(body.get("route_ambiguity_reason") or ""),
                candidate_agents=list(body.get("candidate_agents") or []),
                sector_score_breakdown=list(body.get("sector_score_breakdown") or []),
                selected_sector_score=float(body.get("selected_sector_score", 0.0) or 0.0),
                selected_sector_prior=float(body.get("selected_sector_prior", 0.0) or 0.0),
                selection_strategy=str(body.get("sector_selection_strategy") or "prior_weighted_score_fusion"),
                prior_conflict=bool(body.get("sector_prior_conflict", False)),
                prior_conflict_reason=str(body.get("sector_prior_conflict_reason") or ""),
                feature_map_summary=_feature_map_summary(result, pipeline),
                rejected_sample_rate=float(rejected_ratio),
                anomaly_family=str(body.get("anomaly_family") or ""),
            )
        calibrated_rows = result.core_output.calibrated_matrix
        if hasattr(calibrated_rows, "tolist"):
            calibrated_data = calibrated_rows.tolist()
        else:
            calibrated_data = [list(row) for row in calibrated_rows]
        body["calibration"] = {
            "mode": "cauren_core_agent_schema",
            "calibrated_data": calibrated_data,
        }
        if explain:
            from cauren_core import render_diagnosis_explanation

            body["explanation"] = render_diagnosis_explanation(result)
        body["meta"] = {
            "architecture": "cauren_core_civil_agent",
            "asset_id": resolution.resolved_asset_id,
            "client_id": payload.client_id,
            "mission_phase": payload.mission_phase,
            "runtime_mode": runtime_mode,
            "sampling_hz_effective": sampling_hz_effective,
            "processing_ms": round(float(elapsed_ms), 3),
            "agent_id_requested": payload.agent_id,
            "sector_requested": payload.sector,
            "sensor_schema_version": payload.sensor_schema_version,
            "cbs_building_payload": bool(_uses_cbs_building_payload(payload)),
                        "explain": bool(explain),
            "operational_memory_lookup": operational_lookup,
            "operational_memory_used": bool(operational_used),
            "operational_memory_fallback_reason": operational_fallback_reason,
            "sensor_layout_key": layout_key,
        }
        body.update(_asset_resolution_meta(resolution))
        return body
    except HTTPException:
        raise
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        request.app.state.metrics.record_request(elapsed_ms, success=False, anomaly=False)
        logger.exception("cauren_diagnose_failed asset_id=%s", getattr(payload, "asset_id", "unknown"))
        raise HTTPException(status_code=500, detail=str(e))






CAUREN_CORE_RUNTIME_MODES = {"cauren", "cauren_core", "core_agents", "civil_agent"}


def _is_cauren_core_runtime_mode() -> bool:
    mode = str(os.getenv("CAUREN_RUNTIME_MODE", "")).strip().lower()
    return mode in CAUREN_CORE_RUNTIME_MODES


def _load_cauren_core_agent_runtime(base_dir: Path) -> None:
    from cauren_core import CaurenPipeline
    from cauren_core.tenant_runtime import load_tenant_runtime_config_from_env

    maintenance_dir = _ensure_writable_dir(
        Path(
            os.getenv(
                "MAINTENANCE_STATE_DIR",
                str(base_dir / "runtime_outputs" / "state"),
            )
        ),
        label="state",
    )
    tenant_runtime = load_tenant_runtime_config_from_env()
    if tenant_runtime is not None and tenant_runtime.lock_agent_routing:
        pipeline = CaurenPipeline.from_agent_ids([tenant_runtime.active_agent_id])
    else:
        pipeline = CaurenPipeline.from_default_registry()

    app.state.runtime_architecture = "cauren_core_civil_agent"
    app.state.cauren_runtime_mode = str(os.getenv("CAUREN_RUNTIME_MODE", "core_agents")).strip().lower()
    app.state.cauren_pipeline = pipeline
    app.state.cauren_agent_ids = pipeline.registry.ids()
    app.state.cauren_tenant_runtime = tenant_runtime.to_dict() if tenant_runtime is not None else None
    app.state.model_loaded = True
    app.state.cauren_core_loaded = True
    app.state.scaler_loaded = False
    app.state.adj_loaded = False
    app.state.checkpoint_loaded = False
    app.state.profile_loaded = False
    app.state.root_cause_label_alignment_ok = True
    app.state.root_cause_consistency_ok = True
    app.state.root_cause_unmapped_labels = []
    app.state.root_cause_consistency = {"ok": True, "issues": []}
    app.state.root_cause_release = {}
    app.state.root_cause_release_revision_tag = "agent-scoped"
    app.state.root_cause_release_source = "cauren_core"
    app.state.root_cause_release_validation_ok = True

    app.state.model = None
    app.state.fast_model = None
    app.state.fast_model_enabled = False
    app.state.model_lock = threading.RLock()
    app.state.asset_contexts = {}
    app.state.asset_contexts_lock = threading.RLock()
    app.state.metrics = RuntimeMetrics()
    app.state.restart_counter = int(os.getenv("RESTART_COUNTER", "0") or 0)
    app.state.readiness_false_streak = 0
    app.state.readiness_false_streak_max = 0
    app.state.xai_local_store_lock = threading.RLock()
    app.state.calibration_guard_states = {}
    app.state.calibration_guard_lock = threading.RLock()
    app.state.calibration_guard_enabled = False

    app.state.canonical_data_dir = None
    app.state.data_policy = "agent_schema"
    app.state.release_manifest = {
        "architecture": "cauren_core_civil_agent",
        "accepted": True,
        "agent_count": len(app.state.cauren_agent_ids),
        "tenant_runtime": dict(app.state.cauren_tenant_runtime or {}) if app.state.cauren_tenant_runtime else None,
    }
    app.state.release_manifest_path = None
    app.state.release_manifest_accepted = True
    app.state.accepted_dimensions = []
    app.state.model_train_dimensions = []
    app.state.context_only_dimensions = []
    app.state.dimension_units_canonical = {}
    app.state.model_version = "cauren-core-v0"
    app.state.calibrator_model_version = "cauren-core-v0"
    app.state.anomaly_model_version = "cauren-core-v0"
    app.state.kb_version = "agent-scoped"
    app.state.profile_version = "agent-schema-v1"

    app.state.reporting_enabled = os.getenv("REPORTING_ENABLED", "false").lower().strip() == "true"
    app.state.report_queue = queue.Queue(maxsize=int(os.getenv("REPORT_QUEUE_MAXSIZE", "2000")))
    app.state.queue_warn_threshold = int(os.getenv("QUEUE_WARN_THRESHOLD", "1200"))
    app.state.queue_degrade_threshold = int(os.getenv("QUEUE_DEGRADE_THRESHOLD", "1600"))
    app.state.queue_protect_threshold = int(os.getenv("QUEUE_PROTECT_THRESHOLD", "1900"))
    app.state.queue_watermark = "normal"
    app.state.queue_watermark_last_change_ts = time.time()
    app.state.artifact_claims_inflight = {}
    app.state.artifact_claim_ttl_sec = float(os.getenv("ARTIFACT_CLAIM_TTL_SEC", "300"))
    app.state.artifact_requeue_on_fail = os.getenv("ARTIFACT_REQUEUE_ON_FAIL", "true").lower() == "true"
    app.state.artifact_last_job_ts = 0.0

    inference_queue_maxsize = int(os.getenv("INFERENCE_QUEUE_MAXSIZE", "256"))
    app.state.inference_queue = queue.Queue(maxsize=inference_queue_maxsize)
    app.state.inference_queue_warn_threshold = int(os.getenv("INFERENCE_QUEUE_WARN_THRESHOLD", "180"))
    app.state.inference_queue_degrade_threshold = int(os.getenv("INFERENCE_QUEUE_DEGRADE_THRESHOLD", "220"))
    app.state.inference_queue_protect_threshold = int(os.getenv("INFERENCE_QUEUE_PROTECT_THRESHOLD", "250"))

    app.state.report_worker_mode = "external"
    app.state.report_worker_stop = threading.Event()
    app.state.report_worker = None
    app.state.traffic_node = os.getenv("TRAFFIC_NODE_NAME", socket.gethostname())
    app.state.traffic_role = os.getenv("TRAFFIC_NODE_ROLE", "cauren_core")
    app.state.active_traffic_node_override = os.getenv("ACTIVE_TRAFFIC_NODE", "auto")
    app.state.receiving_traffic_recently_flag = False

    app.state.state_store_enabled = False
    app.state.state_store_path = _resolve_writable_file_path(
        Path(os.getenv("STATE_STORE_PATH", str(maintenance_dir / "runtime_state_store.json"))),
        fallback_dir=maintenance_dir,
        env_name="STATE_STORE_PATH",
    )
    app.state.state_store_ttl_sec = int(os.getenv("STATE_STORE_TTL_SEC", "21600"))
    app.state.state_store_max_guard_entries = int(os.getenv("STATE_STORE_MAX_GUARD_ENTRIES", "2048"))
    app.state.state_store_max_asset_entries = int(os.getenv("STATE_STORE_MAX_ASSET_ENTRIES", "1024"))
    app.state.state_store_write_interval_sec = float(os.getenv("STATE_STORE_WRITE_INTERVAL_SEC", "2.0"))
    app.state.state_store_last_persist_ts = 0.0
    app.state.state_store_lock = threading.RLock()
    app.state.state_store_connected = True
    app.state.continuity_mode = "cauren_core"
    app.state.asset_context_summaries = {}

    app.state.xai_decision_ledger = None
    app.state.canonical_review_store = None
    app.state.asset_norm_store = None
    app.state.memory_enabled = False
    app.state.memory_status = "unavailable"
    app.state.shared_memory = None
    app.state.asset_identity_resolver = None
    app.state.asset_identity_status = "unavailable"
    app.state.operational_memory_store = None
    app.state.operational_memory_status = "unavailable"
    try:
        operational_memory_path = _resolve_writable_file_path(
            Path(
                os.getenv(
                    "OPERATIONAL_MEMORY_STATE_PATH",
                    str(maintenance_dir / "operational_memory_state.json"),
                )
            ),
            fallback_dir=maintenance_dir,
            env_name="OPERATIONAL_MEMORY_STATE_PATH",
        )
        app.state.operational_memory_store = OperationalMemoryStore(
            state_path=operational_memory_path,
            ttl_sec=float(os.getenv("OPERATIONAL_MEMORY_TTL_SEC", "21600")),
            max_entries=int(os.getenv("OPERATIONAL_MEMORY_MAX_ENTRIES", "4096")),
        )
        app.state.operational_memory_status = "available"
    except Exception:
        app.state.operational_memory_store = None
        app.state.operational_memory_status = "unavailable"
        logger.exception("operational_memory_init_failed")
    logger.info(
        json.dumps(
            {
                "event": "cauren_core_runtime_loaded",
                "architecture": app.state.runtime_architecture,
                "agents": app.state.cauren_agent_ids,
                "operational_memory_status": app.state.operational_memory_status,
            }
        )
    )


@app.on_event("startup")
def load_model():
    base_dir = Path(__file__).resolve().parent.parent
    _load_cauren_core_agent_runtime(base_dir)


@app.on_event("shutdown")
def shutdown_worker():
    _persist_runtime_state_if_due(app, force=True)
    if hasattr(app.state, "report_worker_stop"):
        app.state.report_worker_stop.set()
    if hasattr(app.state, "report_worker") and app.state.report_worker is not None:
        app.state.report_worker.join(timeout=2.0)


@app.get("/health/liveness")
def liveness():
    return {"status": "alive", "timestamp": time.time()}


@app.get("/health/readiness")
def readiness(request: Request):
    queue_snapshot = _update_queue_pressure_state(request.app, context="readiness")
    queue_depth = int(queue_snapshot["queue_depth"])
    inf_snapshot = _inference_pressure_snapshot(request.app)
    inference_queue_depth = int(inf_snapshot["inference_queue_depth"])
    metrics_snapshot = request.app.state.metrics.snapshot(
        report_queue_depth=queue_depth,
        inference_queue_depth=inference_queue_depth,
    )
    _update_traffic_activity_state(
        request.app,
        metrics_snapshot=metrics_snapshot,
        context="readiness",
    )
    active_traffic_node = _resolve_active_traffic_node(request.app, metrics_snapshot)
    ready = (
        bool(getattr(request.app.state, "cauren_core_loaded", False))
        and inf_snapshot["inference_watermark"] != "protect"
        and bool(getattr(request.app.state, "state_store_connected", True))
    )
    if ready:
        request.app.state.readiness_false_streak = 0
    else:
        streak = int(getattr(request.app.state, "readiness_false_streak", 0) or 0) + 1
        request.app.state.readiness_false_streak = streak
        request.app.state.readiness_false_streak_max = max(
            int(getattr(request.app.state, "readiness_false_streak_max", 0) or 0),
            streak,
        )
    return {
        "status": "ready" if ready else "not_ready",
        "ready": bool(ready),
        "inference_ready": bool(ready),
        "runtime_architecture": "cauren_core_civil_agent",
        "cauren_runtime_mode": str(getattr(request.app.state, "cauren_runtime_mode", "core_agents")),
        "cauren_core_loaded": bool(getattr(request.app.state, "cauren_core_loaded", False)),
        "agent_count": int(len(getattr(request.app.state, "cauren_agent_ids", []))),
        "agents": list(getattr(request.app.state, "cauren_agent_ids", [])),
        "tenant_runtime": dict(getattr(request.app.state, "cauren_tenant_runtime", {}) or {}),
        "model_loaded": bool(getattr(request.app.state, "model_loaded", False)),
        "data_policy": str(getattr(request.app.state, "data_policy", "agent_schema")),
        "artifact_degraded": queue_snapshot["queue_watermark"] in {"degrade", "protect"},
        "queue_depth": queue_depth,
        "artifact_queue_depth": queue_depth,
        "queue_maxsize": queue_snapshot["queue_maxsize"],
        "queue_watermark": queue_snapshot["queue_watermark"],
        "queue_warn_threshold": queue_snapshot["queue_warn_threshold"],
        "queue_degrade_threshold": queue_snapshot["queue_degrade_threshold"],
        "queue_protect_threshold": queue_snapshot["queue_protect_threshold"],
        "inference_queue_depth": inference_queue_depth,
        "inference_queue_maxsize": inf_snapshot["inference_queue_maxsize"],
        "inference_watermark": inf_snapshot["inference_watermark"],
        "inference_queue_warn_threshold": inf_snapshot["inference_queue_warn_threshold"],
        "inference_queue_degrade_threshold": inf_snapshot["inference_queue_degrade_threshold"],
        "inference_queue_protect_threshold": inf_snapshot["inference_queue_protect_threshold"],
        "continuity_mode": str(getattr(request.app.state, "continuity_mode", "cauren_core")),
        "state_store_connected": bool(getattr(request.app.state, "state_store_connected", True)),
        "state_store_path": str(getattr(request.app.state, "state_store_path", "")),
        "state_store_ttl_sec": int(getattr(request.app.state, "state_store_ttl_sec", 0)),
        "state_store_enabled": bool(getattr(request.app.state, "state_store_enabled", False)),
        "traffic_node": str(getattr(request.app.state, "traffic_node", "unknown")),
        "traffic_role": str(getattr(request.app.state, "traffic_role", "unknown")),
        "active_traffic_node": active_traffic_node,
        "receiving_traffic_recently": bool(metrics_snapshot.get("receiving_traffic_recently", False)),
        "memory_enabled": bool(getattr(request.app.state, "memory_enabled", False)),
        "memory_status": str(getattr(request.app.state, "memory_status", "unavailable")),
        "asset_identity_status": str(getattr(request.app.state, "asset_identity_status", "unavailable")),
        "report_worker_mode": str(getattr(request.app.state, "report_worker_mode", "external")),
        "model_version": getattr(request.app.state, "model_version", "unknown"),
        "calibrator_model_version": getattr(request.app.state, "calibrator_model_version", "unknown"),
        "anomaly_model_version": getattr(request.app.state, "anomaly_model_version", "unknown"),
        "kb_version": getattr(request.app.state, "kb_version", "unknown"),
        "profile_version": getattr(request.app.state, "profile_version", "unknown"),
        **_runtime_resource_snapshot(request.app),
    }
@app.get("/agents")
def list_cauren_agents(request: Request):
    pipeline = _get_cauren_pipeline(request)
    agents = []
    for agent in pipeline.registry.all():
        schema = agent.schema
        agents.append(
            {
                "agent_id": schema.agent_id,
                "sector": schema.sector,
                "display_name": schema.display_name,
                "schema_version": schema.version,
                "required_features": list(schema.required_features),
                "optional_features": list(schema.optional_features),
                "feature_order": list(schema.feature_order),
            }
        )
    return {
        "architecture": "cauren_core_civil_agent",
        "measurement_contract": "agent_schema",
        "routing_contract": "payload_plus_classifier",
        "output_taxonomy": "anomaly_type_plus_risk_score",
        "agents": agents,
    }


@app.post("/ops/artifact/claim")
def artifact_claim(request: Request, payload: ArtifactClaimIn):
    _requeue_expired_artifact_claims(request.app)
    limit = int(payload.limit or 1)
    limit = max(1, min(limit, 8))
    worker_id = str(payload.worker_id or "artifact-worker")
    items: List[Dict[str, Any]] = []
    for _ in range(limit):
        try:
            job = request.app.state.report_queue.get_nowait()
        except queue.Empty:
            break
        if not isinstance(job, dict):
            job = {
                "job_id": str(uuid.uuid4()),
                "event_id": f"evt-{int(time.time() * 1000)}",
                "asset_id": str(job),
                "render_mode": "summary-only",
                "queued_at_ts": time.time(),
                "diagnostics": {},
                "plot_data": {},
                "output_root": str(getattr(request.app.state, "artifact_output_root", "runtime_outputs")),
            }
        claim_id = str(uuid.uuid4())
        request.app.state.artifact_claims_inflight[claim_id] = {
            "claim_id": claim_id,
            "worker_id": worker_id,
            "claimed_at_ts": time.time(),
            "job": job,
        }
        items.append({"claim_id": claim_id, "job": job})
    return {
        "items": items,
        "queue_depth": int(request.app.state.report_queue.qsize()),
        "inflight": int(len(getattr(request.app.state, "artifact_claims_inflight", {}))),
    }


@app.post("/ops/artifact/ack")
def artifact_ack(request: Request, payload: ArtifactAckIn):
    inflight = getattr(request.app.state, "artifact_claims_inflight", {})
    row = inflight.pop(payload.claim_id, None) if isinstance(inflight, dict) else None
    if row is None:
        return {"acked": False, "reason": "claim_not_found"}

    try:
        request.app.state.report_queue.task_done()
    except Exception:
        pass

    job = row.get("job") if isinstance(row, dict) else {}
    status = str(payload.status or "ok").strip().lower()
    lag_ms = payload.lag_ms
    if lag_ms is None:
        queued_at = _safe_float((job or {}).get("queued_at_ts"), default=time.time())
        lag_ms = max(0.0, (time.time() - queued_at) * 1000.0)
    if status == "ok":
        request.app.state.metrics.record_disk_writes(int(payload.total_files_written or 0))
        rendered = int(payload.rendered_images or 0)
        if rendered > 0:
            request.app.state.metrics.record_artifact_rendered(lag_ms=float(lag_ms))
        elif bool((job or {}).get("deduped", False)):
            request.app.state.metrics.record_artifact_deduped()
        else:
            request.app.state.metrics.record_artifact_skipped()
        return {"acked": True, "status": "ok"}

    request.app.state.metrics.record_artifact_failed()
    if bool(getattr(request.app.state, "artifact_requeue_on_fail", True)):
        try:
            request.app.state.report_queue.put_nowait(job)
        except Exception:
            request.app.state.metrics.record_queue_drop()
    return {"acked": True, "status": "failed", "error": str(payload.error or "")}


@app.get("/ops/artifact/status")
def artifact_status(request: Request):
    _requeue_expired_artifact_claims(request.app)
    queue_snapshot = _queue_pressure_snapshot(request.app)
    inf_snapshot = _inference_pressure_snapshot(request.app)
    snapshot = request.app.state.metrics.snapshot(
        report_queue_depth=int(queue_snapshot["queue_depth"]),
        inference_queue_depth=int(inf_snapshot["inference_queue_depth"]),
    )
    return {
        "report_worker_mode": str(getattr(request.app.state, "report_worker_mode", "external")),
        "artifact_queue_depth": int(queue_snapshot["queue_depth"]),
        "artifact_queue_maxsize": int(queue_snapshot["queue_maxsize"]),
        "artifact_queue_watermark": str(queue_snapshot["queue_watermark"]),
        "artifact_queue_warn_threshold": int(queue_snapshot["queue_warn_threshold"]),
        "artifact_queue_degrade_threshold": int(queue_snapshot["queue_degrade_threshold"]),
        "artifact_queue_protect_threshold": int(queue_snapshot["queue_protect_threshold"]),
        "artifact_claims_inflight": int(len(getattr(request.app.state, "artifact_claims_inflight", {}))),
        "artifact_last_job_ts": float(getattr(request.app.state, "artifact_last_job_ts", 0.0)),
        "artifact_lag_ms_p95": float(snapshot.get("artifact_lag_ms_p95", 0.0)),
        "artifact_jobs_rendered": int(snapshot.get("artifact_jobs_rendered", 0)),
        "artifact_jobs_deduped": int(snapshot.get("artifact_jobs_deduped", 0)),
        "artifact_jobs_skipped": int(snapshot.get("artifact_jobs_skipped", 0)),
        "artifact_jobs_failed": int(snapshot.get("artifact_jobs_failed", 0)),
        "inference_queue_depth": int(inf_snapshot["inference_queue_depth"]),
        "inference_watermark": str(inf_snapshot["inference_watermark"]),
    }


@app.post("/calibrate")
async def calibrate(request: Request, payload: CalibrateIn, explain: bool = False):
    if not bool(payload.sensors):
        raise HTTPException(status_code=422, detail="civil calibrate requests require sensors")
    _enforce_backpressure_or_raise(request, "/calibrate")
    token = _acquire_inference_slot_or_raise(request, "/calibrate")
    try:
        return await run_in_threadpool(_sync_cauren_calibrate, request, payload, explain)
    finally:
        _release_inference_slot(request, token)


@app.post("/diagnose")
async def diagnose(request: Request, payload: DiagnoseIn, explain: bool = False):
    if not bool(payload.sensors) and not _uses_cbs_building_payload(payload):
        raise HTTPException(status_code=422, detail="civil diagnose requests require sensors or CBS building fields")
    _enforce_backpressure_or_raise(request, "/diagnose")
    token = _acquire_inference_slot_or_raise(request, "/diagnose")
    try:
        return await run_in_threadpool(_sync_cauren_diagnose, request, payload, explain)
    finally:
        _release_inference_slot(request, token)


@app.post("/memory/ingest")
async def memory_ingest(request: Request, payload: MemoryIngestIn):
    resolution = _resolve_asset_identity(
        request,
        asset_id=payload.asset_id,
        site_id=payload.site_id,
        line_id=payload.line_id,
        machine_id=payload.machine_id,
    )
    telemetry: Optional[NormalizedTelemetry] = None
    sensor_data = payload.sensor_data
    if sensor_data is None:
        if not payload.sensor_values:
            raise HTTPException(status_code=422, detail="Provide either sensor_data or sensor_values.")
        telemetry = _normalize_sensor_values_payload(payload.sensor_values)
        sensor_data = telemetry.core_matrix.tolist()
    else:
        if not sensor_data:
            raise HTTPException(status_code=422, detail="sensor_data cannot be empty.")
        expected_cols = len(payload.dimension_names) if payload.dimension_names else len(sensor_data[0])
        for idx, row in enumerate(sensor_data):
            if not isinstance(row, list) or len(row) != expected_cols:
                raise HTTPException(
                    status_code=422,
                    detail=f"sensor_data row#{idx} must have {expected_cols} values.",
                )
        raw_sensor_np = _validate_data_matrix(
            sensor_data,
            seq_len=len(sensor_data),
            field_name="sensor_data",
            expected_dim_count=expected_cols,
        )
        telemetry = _project_to_canonical_dimensions(
            raw_sensor_np,
            field_name="sensor_data",
            dimension_names=payload.dimension_names,
            primary_dimension_sources=payload.primary_dimension_sources,
            primary_internal_pressure_sensor=payload.primary_internal_pressure_sensor,
            required_dimensions=[],
        )
        sensor_data = telemetry.core_matrix.tolist()

    include_in_learning = payload.include_in_learning
    if include_in_learning is None:
        include_in_learning = str(payload.status or "NORMAL").strip().upper() == "NORMAL"

    correction = payload.correction_data
    correction_np = None if correction is None else np.asarray(correction, dtype=np.float32)
    memory_outputs = _build_memory_outputs(
        request,
        asset_id=resolution.resolved_asset_id,
        timestamp=payload.timestamp,
        raw_matrix=np.asarray(sensor_data, dtype=np.float32),
        observed_series=telemetry.observed_series if telemetry is not None else None,
        correction_matrix=correction_np,
        include_in_learning=bool(include_in_learning) and not resolution.review_required,
    )
    memory_outputs["asset_resolution"] = _asset_resolution_meta(resolution)
    if telemetry is not None:
        memory_outputs["meta"] = {
            "accepted_dimensions": list(getattr(request.app.state, "accepted_dimensions", CANONICAL_DIMENSION_LIST)),
            "observed_dimensions": list(telemetry.observed_dimensions),
            "model_train_dimensions": list(getattr(request.app.state, "model_train_dimensions", DIMENSIONS)),
            "context_only_dimensions": list(getattr(request.app.state, "context_only_dimensions", [])),
            "core_dimensions": list(telemetry.core_dimensions),
            "aux_dimensions": list(telemetry.aux_dimensions),
            "missing_dimensions": list(telemetry.missing_dimensions),
            "source_dimension_map": dict(telemetry.source_dimension_map),
            "unused_input_columns": list(telemetry.unused_input_columns),
            "rejected_dimensions": list(telemetry.rejected_dimensions),
        }
    return memory_outputs


@app.get("/memory/norms/{asset_id}")
async def memory_norms(request: Request, asset_id: str):
    if not bool(getattr(request.app.state, "memory_enabled", False)):
        return _memory_unavailable_payload(asset_id)["memory_snapshot"]
    service = getattr(request.app.state, "memory_service", None)
    if service is None:
        return _memory_unavailable_payload(asset_id)["memory_snapshot"]
    try:
        snapshot = service.get_norm_snapshot(asset_id)
        request.app.state.memory_status = "available"
        return snapshot
    except Exception:
        request.app.state.memory_status = "unavailable"
        logger.exception("memory_norms_failed asset_id=%s", asset_id)
        return _memory_unavailable_payload(asset_id)["memory_snapshot"]


@app.get("/memory/context/{asset_id}")
async def memory_context(
    request: Request,
    asset_id: str,
    dimension: str | None = Query(default=None),
    current_value: float | None = Query(default=None),
    timestamp: float | None = Query(default=None),
):
    if not bool(getattr(request.app.state, "memory_enabled", False)):
        return _memory_unavailable_payload(asset_id)["memory_context"]
    service = getattr(request.app.state, "memory_service", None)
    if service is None:
        return _memory_unavailable_payload(asset_id)["memory_context"]
    current_values: Dict[str, float] = {}
    if dimension is not None and current_value is not None:
        dim = str(dimension).strip()
        canonical_dim = _canonical_dimension_name(dim)
        if canonical_dim is None:
            raise HTTPException(status_code=422, detail=f"Unsupported dimension: {dim}")
        current_values[canonical_dim] = float(current_value)
    try:
        context_payload = service.get_context(
            asset_id=asset_id,
            timestamp=timestamp or time.time(),
            current_values=current_values,
        )
        if dimension:
            context_payload["contexts"] = [
                row for row in context_payload.get("contexts", []) if row.get("dimension") == dimension
            ]
        request.app.state.memory_status = "available"
        return context_payload
    except Exception:
        request.app.state.memory_status = "unavailable"
        logger.exception("memory_context_failed asset_id=%s", asset_id)
        return _memory_unavailable_payload(asset_id)["memory_context"]


@app.get("/asset/proposals/pending")
async def asset_pending(request: Request, limit: int = Query(default=100)):
    resolver = getattr(request.app.state, "asset_identity_resolver", None)
    if resolver is None:
        return {"status": "unavailable", "items": []}
    try:
        return {"status": "ok", "items": resolver.list_pending(limit=max(1, int(limit)))}
    except Exception:
        logger.exception("asset_proposals_pending_failed")
        return {"status": "error", "items": []}


@app.post("/asset/proposals/review")
async def asset_review(request: Request, payload: AssetProposalReviewIn):
    resolver = getattr(request.app.state, "asset_identity_resolver", None)
    if resolver is None:
        raise HTTPException(status_code=503, detail="asset identity resolver unavailable")
    try:
        updated = resolver.review(
            proposal_id=str(payload.proposal_id),
            action=str(payload.action),
            reviewer=str(payload.reviewer),
            notes=str(payload.notes or ""),
        )
        return {"status": "ok", "entry": updated}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/asset/registry/{site_id}/{line_id}/{machine_id}")
async def asset_registry_lookup(request: Request, site_id: str, line_id: str, machine_id: str):
    resolver = getattr(request.app.state, "asset_identity_resolver", None)
    if resolver is None:
        return {"status": "unavailable"}
    try:
        payload = resolver.lookup(site_id=site_id, line_id=line_id, machine_id=machine_id)
        return {"status": "ok", "entry": payload}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
