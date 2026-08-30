import csv
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from api.fault_ontology import canonicalize_fault


@dataclass(frozen=True)
class FaultRecord:
    fault_code: str
    fault_label: str
    root_cause: str
    procedure_ref: str


class FaultDatasetExporter:
    """
    Writes anomaly outputs to a persistent dataset file for downstream model training.
    """

    FIELDNAMES = ["fault_code", "fault_label", "root_cause", "procedure_ref"]

    # Keyed by `CanonicalFault.fault_family_id` (see api/fault_ontology.py),
    # which is the taxonomy actually produced by canonicalize_fault(). This
    # used to be keyed by a standalone FailureType enum's `.value` strings
    # (PHYSICS_VIOLATION, HIGH_NOISE, ...), but that enum no longer exists
    # anywhere in the codebase, so every lookup silently fell through to
    # the FC-9999 "unknown" fallback.
    CODE_MAP: Dict[str, str] = {
        "nominal": "FC-0000",
        "review_required": "FC-0100",
        "pressure_instability": "FC-1000",
        "thermal_anomaly": "FC-2000",
        "radiation_anomaly": "FC-3000",
        "structural_anomaly": "FC-4000",
        "sensor_drift": "FC-5000",
        "vibration_mechanical": "FC-6000",
        "electrical_anomaly": "FC-7000",
        "flow_instability": "FC-8000",
        "process_disturbance": "FC-8500",
        "signal_noise": "FC-9000",
        "multi_fault": "FC-9500",
        "unknown_anomaly": "FC-9999",
    }

    PROCEDURE_MAP: Dict[str, str] = {
        "FC-0000": "PROC-NORMAL-OPS-000",
        "FC-0100": "PROC-MANUAL-REVIEW-001",
        "FC-1000": "PROC-PRESSURE-INSTABILITY-INSPECTION-001",
        "FC-2000": "PROC-THERMAL-ANOMALY-INSPECTION-001",
        "FC-3000": "PROC-RADIATION-ANOMALY-INSPECTION-001",
        "FC-4000": "PROC-STRUCTURAL-ANOMALY-EMERGENCY-001",
        "FC-5000": "PROC-SENSOR-DRIFT-CALIBRATION-001",
        "FC-6000": "PROC-VIBRATION-MECHANICAL-INSPECTION-001",
        "FC-7000": "PROC-ELECTRICAL-ANOMALY-INSPECTION-001",
        "FC-8000": "PROC-FLOW-INSTABILITY-INSPECTION-001",
        "FC-8500": "PROC-PROCESS-DISTURBANCE-REVIEW-001",
        "FC-9000": "PROC-SIGNAL-NOISE-INSPECTION-001",
        "FC-9500": "PROC-MULTI-FAULT-ESCALATION-001",
        "FC-9999": "PROC-GENERAL-DIAGNOSTIC-999",
    }

    def __init__(self, output_path: Path, export_format: str = "csv"):
        self.output_path = Path(output_path)
        self.export_format = (export_format or "csv").lower().strip()
        if self.export_format not in {"csv", "json"}:
            raise ValueError("export_format must be either 'csv' or 'json'")
        self._lock = threading.RLock()
        self._ensure_target_file()

    @staticmethod
    def _normalize_fault_label(value: str) -> str:
        text = (value or "").strip()
        if not text:
            return "Unknown Fault"
        return re.sub(r"\s+", " ", text)

    @classmethod
    def _fault_code_for(cls, fault_family_id: str) -> str:
        key = re.sub(r"[^a-z0-9]+", "_", (fault_family_id or "").strip().lower()).strip("_")
        return cls.CODE_MAP.get(key, "FC-9999")

    @classmethod
    def _procedure_ref_for(cls, fault_code: str) -> str:
        return cls.PROCEDURE_MAP.get(fault_code, "PROC-GENERAL-DIAGNOSTIC-999")

    def _ensure_target_file(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            return

        if self.export_format == "csv":
            with self.output_path.open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=self.FIELDNAMES)
                writer.writeheader()
        else:
            with self.output_path.open("w", encoding="utf-8") as fp:
                json.dump([], fp, ensure_ascii=False, indent=2)

    def build_record(self, analysis_result) -> FaultRecord:
        diagnostics = analysis_result.diagnostics
        failure_type = (
            diagnostics.failure_type.value
            if hasattr(diagnostics.failure_type, "value")
            else str(diagnostics.failure_type)
        )
        canonical_fault = canonicalize_fault(
            root_cause_label=getattr(diagnostics, "root_cause_label", ""),
            failure_type=failure_type,
            primary_dimension=getattr(diagnostics, "primary_fault_dimension", ""),
            fault_name=getattr(diagnostics, "fault_name", ""),
            fault_descriptor=getattr(diagnostics, "fault_descriptor", ""),
            top3_candidates=getattr(diagnostics, "top3_candidates", []),
            is_unknown=False,
        )
        fault_label = self._normalize_fault_label(canonical_fault.fault_family_label or failure_type)
        fault_code = self._fault_code_for(canonical_fault.fault_family_id)
        trend_description = getattr(diagnostics, "trend_description", "")
        root_cause = self._normalize_fault_label(
            canonical_fault.fault_subtype_label or trend_description or "Unknown root cause"
        )
        procedure_ref = self._procedure_ref_for(fault_code)
        return FaultRecord(
            fault_code=fault_code,
            fault_label=fault_label,
            root_cause=root_cause,
            procedure_ref=procedure_ref,
        )

    def write_fault(self, analysis_result):
        record = self.build_record(analysis_result)
        record_dict = {
            "fault_code": record.fault_code,
            "fault_label": record.fault_label,
            "root_cause": record.root_cause,
            "procedure_ref": record.procedure_ref,
        }
        with self._lock:
            if self.export_format == "csv":
                with self.output_path.open("a", newline="", encoding="utf-8") as fp:
                    writer = csv.DictWriter(fp, fieldnames=self.FIELDNAMES)
                    writer.writerow(record_dict)
            else:
                with self.output_path.open("r", encoding="utf-8") as fp:
                    current = json.load(fp)
                if not isinstance(current, list):
                    current = []
                current.append(record_dict)
                with self.output_path.open("w", encoding="utf-8") as fp:
                    json.dump(current, fp, ensure_ascii=False, indent=2)
