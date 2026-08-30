import csv
import json
import os
import sys
from types import SimpleNamespace


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.fault_export import FaultDatasetExporter


def _analysis_result(failure_type_value: str, trend_description: str):
    diagnostics = SimpleNamespace(
        failure_type=SimpleNamespace(value=failure_type_value),
        trend_description=trend_description,
    )
    return SimpleNamespace(diagnostics=diagnostics)


def test_fault_export_csv_same_type_same_code(tmp_path):
    export_path = tmp_path / "fault_dataset.csv"
    exporter = FaultDatasetExporter(export_path, export_format="csv")

    # These both hit the "pressure_instability" family via the
    # PRESSURE/REGULATOR keyword rule in api/fault_ontology.py.
    exporter.write_fault(_analysis_result("Pressure Regulator Fault", "Pressure mismatch"))
    exporter.write_fault(_analysis_result("Pressure Cavitation Event", "Temperature mismatch"))

    with export_path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))

    assert len(rows) == 2
    assert rows[0]["fault_code"] == rows[1]["fault_code"] == "FC-1000"
    assert rows[0]["fault_label"] == "Pressure Instability"
    assert rows[0]["procedure_ref"] == "PROC-PRESSURE-INSTABILITY-INSPECTION-001"


def test_fault_export_json_appends_records(tmp_path):
    export_path = tmp_path / "fault_dataset.json"
    exporter = FaultDatasetExporter(export_path, export_format="json")

    # Both hit "signal_noise" via the NOISE keyword rule.
    exporter.write_fault(_analysis_result("High Noise", "Sensor blindness"))
    exporter.write_fault(_analysis_result("Sensor Noise Spike", "Radiation burst"))

    with export_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["fault_code"] == "FC-9000"
    assert data[1]["fault_code"] == "FC-9000"
    assert data[0]["fault_label"] == "Signal Noise"
