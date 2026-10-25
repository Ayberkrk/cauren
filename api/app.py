"""Cauren Civil Diagnostics API.

A thin FastAPI wrapper around cauren_core.CaurenPipeline. It accepts either
raw sensor readings or a CBS ("bina/insaat" GIS) building record, runs the
full normalize -> route -> anomaly-core -> physics -> compose pipeline, and
returns the result (including the quality-control report and uncertainty
band -- see cauren_core.quality_control / cauren_core.uncertainty) as JSON.

This is a single-operator API. It stores diagnosis responses in a local
SQLite history (90 days by default); raw requests are not retained. Set
CAUREN_API_KEY to require Bearer authentication on non-health routes.
This is not a multi-tenant service.
"""

from __future__ import annotations

import hmac
import json
import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from cauren_core.baseline import MIN_REFERENCE_SAMPLES, fit_feature_baseline, parse_baselines

from .history import persist_history, read_baseline, read_history, store_baseline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cauren_civil_api")

app = FastAPI(
    title="Cauren Civil Diagnostics API",
    description="Civil engineering diagnostics and decision-support runtime.",
)

MAX_REQUEST_BYTES = 1_048_576
MAX_SENSOR_ROWS = 8192

ASSET_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
SEQ_MIN, SEQ_MAX = 1, 4096
SAMPLING_HZ_MIN, SAMPLING_HZ_MAX = 1e-9, 1000.0


class RequestGuardMiddleware:
    """Enforce bounded request bodies and optional Bearer authentication."""

    def __init__(self, app, max_body_bytes: int = MAX_REQUEST_BYTES):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        api_key = os.getenv("CAUREN_API_KEY", "").strip()
        if api_key and path not in {"/health/liveness", "/health/readiness"}:
            authorization = headers.get(b"authorization", b"").decode("latin-1")
            scheme, _, supplied = authorization.partition(" ")
            if scheme.lower() != "bearer" or not hmac.compare_digest(supplied, api_key):
                await self._respond(send, 401, {"detail": "Bearer authentication required"}, [(b"www-authenticate", b"Bearer")])
                return

        try:
            declared_length = int(headers.get(b"content-length", b"0") or b"0")
        except ValueError:
            declared_length = self.max_body_bytes + 1
        if declared_length > self.max_body_bytes:
            await self._respond(send, 413, {"detail": "request body exceeds 1 MiB limit"})
            return

        body = bytearray()
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.max_body_bytes:
                await self._respond(send, 413, {"detail": "request body exceeds 1 MiB limit"})
                return
            more_body = message.get("more_body", False)

        sent = False

        async def replay_body():
            nonlocal sent
            if sent:
                # The body is fully replayed; later receives must still report
                # a real client disconnect instead of an endless empty request.
                return await receive()
            sent = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay_body, send)

    @staticmethod
    async def _respond(send, status_code: int, body: dict[str, Any], extra_headers=()):
        encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
        headers = [(b"content-type", b"application/json"), (b"content-length", str(len(encoded)).encode("ascii"))]
        headers.extend(extra_headers)
        await send({"type": "http.response.start", "status": status_code, "headers": headers})
        await send({"type": "http.response.body", "body": encoded})


app.add_middleware(RequestGuardMiddleware)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


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
    timestamp: Optional[float] = None
    seq_len: int = 16
    sampling_hz: float = 1.0
    runtime_mode: Optional[str] = None
    sensors: Optional[List[Dict[str, Any]]] = Field(default=None, max_length=MAX_SENSOR_ROWS)
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
    timestamp: Optional[float] = None
    seq_len: int = 16
    sampling_hz: float = 1.0
    runtime_mode: Optional[str] = None
    sensors: Optional[List[Dict[str, Any]]] = Field(default=None, max_length=MAX_SENSOR_ROWS)
    ingress_tag: Optional[str] = None
    ingress_tags: Optional[List[str]] = None
    sector_hint: Optional[str] = None
    sector_hint_confidence: Optional[float] = None
    source_metadata: Optional[Dict[str, Any]] = None
    oma_frequency_drift: Optional[Dict[str, Any]] = None
    # Per-asset reference: {feature: {center, scale, n, expected_variation,
    # unit}}. Overrides a baseline stored with POST /baseline/{asset_id}.
    baseline: Optional[Dict[str, Dict[str, Any]]] = None
    # CBS ("bina/insaat" GIS) fields. When `sensors` is not supplied, only
    # features supported by observed source values are converted into readings.
    building_id: Optional[str] = None
    geometry: Optional[Dict[str, Any]] = None
    address: Optional[Dict[str, Any]] = None
    administrative_unit: Optional[Dict[str, Any]] = None
    project_permit: Optional[Dict[str, Any]] = None
    construction_status: Optional[Dict[str, Any]] = None
    infrastructure_connections: Optional[Dict[str, Any]] = None
    risk_assessments: Optional[Dict[str, Any]] = None
    inspection_findings: Optional[List[Dict[str, Any]]] = None


class BaselineFitIn(BaseModel):
    agent_id: Optional[str] = None
    sector: Optional[str] = None
    # A reference window in the same row format as /diagnose. Every reading
    # needs a timestamp so repeated readings of a feature stay separate.
    sensors: List[Dict[str, Any]] = Field(max_length=MAX_SENSOR_ROWS)
    # Floor for a feature's scale, e.g. its measurement resolution. Required
    # for a feature that never varies in the reference window.
    min_scale: Optional[Dict[str, float]] = None
    # Override the inferred "does this feature normally vary" flag.
    expected_variation: Optional[Dict[str, bool]] = None


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CBS building record -> sensor reading conversion
# ---------------------------------------------------------------------------


def _score_from_mapping(data: Optional[Dict[str, Any]], keys: tuple[str, ...], default: float | None = None) -> float | None:
    if not isinstance(data, dict):
        return default
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        # max(0, min(1, nan)) is 1.0, so a NaN would read as maximum risk.
        if not math.isfinite(score):
            continue
        return max(0.0, min(1.0, score))
    return default


def _progress_pct_from_mapping(
    data: Optional[Dict[str, Any]],
    keys: tuple[str, ...],
    default: float | None = None,
) -> float | None:
    if not isinstance(data, dict):
        return default
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, bool):
            return 100.0 if value else 0.0
        try:
            progress = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(progress):
            continue
        if 0.0 <= progress <= 1.0:
            progress *= 100.0
        return max(0.0, min(100.0, progress))
    return default


def _inspection_score(items: Optional[List[Dict[str, Any]]]) -> float | None:
    scores = [
        score
        for item in (items or [])
        if isinstance(item, dict)
        for score in [_score_from_mapping(item, ("finding_score", "severity_score", "score"))]
        if score is not None
    ]
    return max(scores) if scores else None


def _uses_cbs_building_payload(payload: DiagnoseIn) -> bool:
    return bool(str(payload.building_id or "").strip()) and not payload.sensors


def _building_sensors_from_payload(payload: DiagnoseIn) -> list[dict[str, Any]]:
    risk = payload.risk_assessments
    permit = payload.project_permit
    status = payload.construction_status
    infra = payload.infrastructure_connections
    findings = payload.inspection_findings
    timestamp = float(payload.timestamp) if payload.timestamp is not None else time.time()
    values = {
        "structural_risk_score": _score_from_mapping(
            risk, ("structural_risk_score", "structural_risk", "building_risk_score", "risk_score")
        ),
        "inspection_finding_score": _inspection_score(findings),
        "permit_status_score": _score_from_mapping(
            permit, ("permit_status_score", "document_readiness_score", "approval_score", "approved")
        ),
        "construction_progress_pct": _progress_pct_from_mapping(
            status, ("progress_pct", "completion_pct", "construction_progress_pct")
        ),
        "infrastructure_connection_score": _score_from_mapping(
            infra, ("infrastructure_connection_score", "readiness_score", "utility_connection_score", "ready")
        ),
        "natural_hazard_score": _score_from_mapping(
            risk, ("natural_hazard_score", "earthquake_risk_score", "flood_risk_score", "natural_risk_score")
        ),
        "occupancy_safety_score": _score_from_mapping(
            risk, ("occupancy_safety_score", "life_safety_score", "occupancy_risk_score")
        ),
        "ground_stability_score": _score_from_mapping(
            risk, ("ground_stability_score", "ground_risk_score", "geotechnical_risk_score")
        ),
    }
    building_id = str(payload.building_id or "building")
    return [
        {
            "sensor_id": f"{building_id}_{name}",
            "name": name,
            "unit": "%" if name == "construction_progress_pct" else "ratio",
            "value": round(float(value), 6),
            "timestamp": timestamp,
            "metadata": {
                "source": "cbs_payload",
                "source_section": _CBS_FEATURE_SECTIONS[name],
                "observed": True,
            },
        }
        for name, value in values.items()
        if value is not None
    ]


_CBS_FEATURE_SECTIONS = {
    "structural_risk_score": "risk_assessments",
    "inspection_finding_score": "inspection_findings",
    "permit_status_score": "project_permit",
    "construction_progress_pct": "construction_status",
    "infrastructure_connection_score": "infrastructure_connections",
    "natural_hazard_score": "risk_assessments",
    "occupancy_safety_score": "risk_assessments",
    "ground_stability_score": "risk_assessments",
}


def _building_site_context(payload: DiagnoseIn) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if payload.geometry:
        context["geometry"] = payload.geometry
    if payload.address:
        context["address"] = payload.address
    if payload.administrative_unit:
        context["administrative_unit"] = payload.administrative_unit
    return context


# ---------------------------------------------------------------------------
# Pipeline access
# ---------------------------------------------------------------------------


def _get_cauren_pipeline():
    pipeline = getattr(app.state, "cauren_pipeline", None)
    if pipeline is not None:
        return pipeline
    from cauren_core import CaurenPipeline

    pipeline = CaurenPipeline.from_default_registry()
    app.state.cauren_pipeline = pipeline
    return pipeline


# ---------------------------------------------------------------------------
# Route handlers (sync bodies, dispatched through a threadpool -- see the
# concurrency note in CONTRIBUTING.md: CaurenPipeline is shared across
# requests, so nothing here may stash per-request state on it)
# ---------------------------------------------------------------------------


def _sync_calibrate(payload: CalibrateIn) -> dict:
    t0 = time.perf_counter()
    try:
        asset_id = str(payload.asset_id or "cauren_asset").strip()
        timestamp = float(payload.timestamp if payload.timestamp is not None else time.time())
        runtime_mode, sampling_hz_effective = _validate_common_fields(
            asset_id, timestamp, int(payload.seq_len), float(payload.sampling_hz), payload.runtime_mode
        )
        cauren_payload = {
            "asset_id": asset_id,
            "agent_id": payload.agent_id,
            "sector": payload.sector,
            "client_id": payload.client_id,
            "site_context": payload.site_context or {},
            "forecast_context": payload.forecast_context or {},
            "control_mode": payload.control_mode or "guarded_auto",
            "sensor_schema_version": payload.sensor_schema_version,
            "timestamp": timestamp,
            "seq_len": int(payload.seq_len),
            "sampling_hz": float(sampling_hz_effective),
            "runtime_mode": runtime_mode,
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
        result = _get_cauren_pipeline().calibrate(cauren_payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        response = {
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
            "quality_control": result.get("quality_control", {}),
            "selected_agent": result["selected_agent"],
            "candidate_agents": result["candidate_agents"],
            "rejected_samples": result["rejected_samples"],
            "meta": {
                "asset_id": asset_id,
                "client_id": payload.client_id,
                "runtime_mode": runtime_mode,
                "sampling_hz_effective": sampling_hz_effective,
                "processing_ms": round(elapsed_ms, 3),
                "raw_sensor_count": result["raw_sensor_count"],
                "rejected_sample_count": len(result["rejected_samples"]),
                "router_confidence": result["confidence"],
            },
        }
        persist_history(asset_id=asset_id, kind="calibrate", result=response)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("cauren_calibrate_failed asset_id=%s", getattr(payload, "asset_id", "unknown"))
        raise HTTPException(status_code=500, detail="calibration could not be processed") from exc


def _resolve_baseline(payload: DiagnoseIn, asset_id: str) -> dict[str, Any]:
    """Reference resolution order: request payload, then the baseline stored
    for an explicitly named asset, then none (the core falls back to the
    window itself and reports reference "none")."""
    if payload.baseline:
        return {"baseline": payload.baseline, "baseline_source": "payload"}
    if payload.asset_id:
        stored = read_baseline(asset_id=asset_id)
        if stored and stored["features"]:
            return {"baseline": stored["features"], "baseline_source": "stored"}
    return {"baseline": None, "baseline_source": "none"}


def _sync_fit_baseline(asset_id: str, payload: BaselineFitIn) -> dict:
    if any(row.get("timestamp") is None for row in payload.sensors if isinstance(row, dict)):
        raise HTTPException(status_code=422, detail="every reference reading needs a timestamp")
    agent_id, values, units = _get_cauren_pipeline().reference_values(
        {"agent_id": payload.agent_id, "sector": payload.sector, "sensors": payload.sensors}
    )
    fitted: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    for feature, observed in sorted(values.items()):
        try:
            fitted[feature] = fit_feature_baseline(
                observed,
                unit=units.get(feature, ""),
                expected_variation=(payload.expected_variation or {}).get(feature),
                min_scale=(payload.min_scale or {}).get(feature),
            ).to_dict()
        except ValueError as exc:
            skipped[feature] = str(exc)
    if not fitted:
        raise HTTPException(
            status_code=422,
            detail={
                "message": f"no feature had a usable reference window (at least {MIN_REFERENCE_SAMPLES} samples)",
                "skipped": skipped,
            },
        )
    fitted_at = time.time()
    store_baseline(asset_id=asset_id, agent_id=agent_id, baselines=fitted, fitted_at=fitted_at)
    return {
        "asset_id": asset_id,
        "agent_id": agent_id,
        "fitted_at": fitted_at,
        "features": fitted,
        "skipped": skipped,
        "method": "median and MAD x 1.4826 (mean absolute deviation when MAD is zero)",
    }


def _sync_diagnose(payload: DiagnoseIn, explain: bool) -> dict:
    t0 = time.perf_counter()
    try:
        asset_id = str(payload.asset_id or "cauren_asset").strip()
        timestamp = float(payload.timestamp if payload.timestamp is not None else time.time())
        runtime_mode, sampling_hz_effective = _validate_common_fields(
            asset_id, timestamp, int(payload.seq_len), float(payload.sampling_hz), payload.runtime_mode
        )
        uses_cbs = _uses_cbs_building_payload(payload)
        cbs_sensors = _building_sensors_from_payload(payload) if uses_cbs else None
        cauren_payload = {
            "asset_id": asset_id,
            "agent_id": "cauren-civil" if uses_cbs else payload.agent_id,
            "sector": "civil" if uses_cbs else payload.sector,
            "client_id": payload.client_id,
            "site_context": _building_site_context(payload) if uses_cbs else (payload.site_context or {}),
            "forecast_context": payload.forecast_context or {},
            "control_mode": payload.control_mode or "guarded_auto",
            "sensor_schema_version": payload.sensor_schema_version,
            "timestamp": timestamp,
            "seq_len": int(payload.seq_len),
            "sampling_hz": float(sampling_hz_effective),
            "runtime_mode": runtime_mode,
            "sensors": cbs_sensors if uses_cbs else (payload.sensors or []),
            "ingress_tag": payload.ingress_tag,
            "ingress_tags": payload.ingress_tags or [],
            "sector_hint": payload.sector_hint,
            "sector_hint_confidence": payload.sector_hint_confidence,
            "source_metadata": payload.source_metadata or {},
            "site_id": payload.site_id,
            "line_id": payload.line_id,
            "machine_id": payload.machine_id,
            "oma_frequency_drift": payload.oma_frequency_drift or {},
        }
        cauren_payload.update(_resolve_baseline(payload, asset_id))
        result = _get_cauren_pipeline().diagnose(cauren_payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        body = result.to_dict()
        calibrated_rows = result.core_output.calibrated_matrix
        body["calibration"] = {
            "mode": "cauren_core_agent_schema",
            "calibrated_data": [list(row) for row in calibrated_rows],
        }
        if explain:
            from cauren_core import render_diagnosis_explanation

            body["explanation"] = render_diagnosis_explanation(result)
        body["meta"] = {
            "asset_id": asset_id,
            "client_id": payload.client_id,
            "runtime_mode": runtime_mode,
            "sampling_hz_effective": sampling_hz_effective,
            "processing_ms": round(elapsed_ms, 3),
            "agent_id_requested": payload.agent_id,
            "sector_requested": payload.sector,
            "sensor_schema_version": payload.sensor_schema_version,
            "cbs_building_payload": uses_cbs,
            "explain": bool(explain),
        }
        if uses_cbs:
            observed = {row["name"] for row in (cbs_sensors or [])}
            body["meta"]["cbs_observed_features"] = sorted(observed)
            body["meta"]["cbs_missing_features"] = sorted(set(_CBS_FEATURE_SECTIONS) - observed)
            body["meta"]["cbs_feature_sources"] = {
                row["name"]: row["metadata"]["source_section"] for row in (cbs_sensors or [])
            }
        persist_history(asset_id=asset_id, kind="diagnose", result=body)
        return body
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("cauren_diagnose_failed asset_id=%s", getattr(payload, "asset_id", "unknown"))
        raise HTTPException(status_code=500, detail="diagnosis could not be processed") from exc


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.on_event("startup")
def load_pipeline() -> None:
    pipeline = _get_cauren_pipeline()
    app.state.cauren_agent_ids = pipeline.registry.ids()
    logger.info(
        "cauren_pipeline_loaded agents=%s",
        app.state.cauren_agent_ids,
    )


@app.get("/health/liveness")
def liveness():
    return {"status": "ok"}


@app.get("/health/readiness")
def readiness():
    pipeline = getattr(app.state, "cauren_pipeline", None)
    agent_ids = list(getattr(app.state, "cauren_agent_ids", []))
    ready = pipeline is not None and bool(agent_ids)
    return {
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "agent_count": len(agent_ids),
        "agents": agent_ids,
    }


@app.get("/agents")
def list_cauren_agents():
    pipeline = _get_cauren_pipeline()
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


@app.get("/history/{asset_id}")
def get_asset_history(asset_id: str, limit: int = Query(default=100, ge=1, le=1000)):
    if not ASSET_ID_PATTERN.fullmatch(asset_id):
        raise HTTPException(status_code=422, detail="asset_id must match [a-zA-Z0-9_-]{1,64}")
    return {"asset_id": asset_id, "records": read_history(asset_id=asset_id, limit=limit)}


@app.get("/baseline/{asset_id}")
def get_asset_baseline(asset_id: str):
    if not ASSET_ID_PATTERN.fullmatch(asset_id):
        raise HTTPException(status_code=422, detail="asset_id must match [a-zA-Z0-9_-]{1,64}")
    stored = read_baseline(asset_id=asset_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="no baseline stored for this asset")
    return stored


@app.post("/baseline/{asset_id}")
async def fit_asset_baseline(asset_id: str, payload: BaselineFitIn):
    if not ASSET_ID_PATTERN.fullmatch(asset_id):
        raise HTTPException(status_code=422, detail="asset_id must match [a-zA-Z0-9_-]{1,64}")
    if not payload.sensors:
        raise HTTPException(status_code=422, detail="sensors is required")
    return await run_in_threadpool(_sync_fit_baseline, asset_id, payload)


@app.post("/calibrate")
async def calibrate(payload: CalibrateIn):
    if not payload.sensors:
        raise HTTPException(status_code=422, detail="sensors is required")
    return await run_in_threadpool(_sync_calibrate, payload)


def _validate_oma_frequency_drift(drift: Optional[Dict[str, Any]]) -> None:
    if not drift or drift.get("max_drop_pct") is None:
        return
    value = drift["max_drop_pct"]
    try:
        number = float(value) if not isinstance(value, bool) else math.nan
    except (TypeError, ValueError):
        number = math.nan
    if not math.isfinite(number):
        raise HTTPException(status_code=422, detail="oma_frequency_drift.max_drop_pct must be a finite number")


@app.post("/diagnose")
async def diagnose(payload: DiagnoseIn, explain: bool = False):
    if not payload.sensors and not str(payload.building_id or "").strip():
        raise HTTPException(status_code=422, detail="either sensors or a CBS building_id payload is required")
    _validate_oma_frequency_drift(payload.oma_frequency_drift)
    try:
        parse_baselines(payload.baseline)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return await run_in_threadpool(_sync_diagnose, payload, explain)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
