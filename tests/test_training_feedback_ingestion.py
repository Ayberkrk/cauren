from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# This test exercises LLM/src/rocket_llm, a separate sub-package (see
# LLM/pyproject.toml) that isn't installed into the root environment and
# isn't on sys.path by default. Without this, `pytest tests/` fails to
# even collect this file (ModuleNotFoundError: rocket_llm), which aborts
# the entire root test run rather than just this one module.
LLM_SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LLM", "src")
if os.path.isdir(LLM_SRC_DIR) and LLM_SRC_DIR not in sys.path:
    sys.path.insert(0, LLM_SRC_DIR)

pytest.importorskip("yaml", reason="rocket_llm.pipelines.training requires PyYAML")
training = pytest.importorskip(
    "rocket_llm.pipelines.training",
    reason="LLM/ sub-package (rocket_llm) not available on sys.path",
)


def test_prepare_sft_dataset_appends_feedback_training_rows(tmp_path: Path, monkeypatch) -> None:
    configs_dir = tmp_path / "configs"
    processed_dir = tmp_path / "data" / "processed"
    panel_dir = tmp_path / "data" / "feedback"
    configs_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    structured_input = processed_dir / "sft_fault_kb_merged.jsonl"
    structured_input.write_text(
        json.dumps(
            {
                "input": {
                    "diagnostic_event": {
                        "fault_code": "CIV-01",
                        "fault_label": "Structural Risk Drift",
                        "severity": "warning",
                        "mission_phase": "inspection",
                        "sensor_summary": [{"sensor_id": "STR-1", "reading": 0.82, "unit": "score", "trend": "down"}],
                    },
                    "evidence_bundle": {"telemetry_snippets": ["structural risk drift"]},
                },
                "target": {
                    "fault_summary": "Structural risk drift elevated.",
                    "recommended_checks": ["Check structural risk telemetry."],
                    "approved_next_steps": ["Hold the construction package for review."],
                    "do_not_do": ["Do not advance the construction stage before review."],
                    "confidence_score": 0.71,
                    "confidence_statement": "Structured evidence available.",
                    "citations": ["PROC-ENG-01"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    feedback_jsonl = panel_dir / "decision_feedback_training.jsonl"
    feedback_jsonl.write_text(
        json.dumps(
            {
                "decision_id": "llm-civil-1",
                "review_id": "review-civil-1",
                "asset_class": "cauren-civil",
                "fault_family_id": "structural_risk_escalation",
                "verdict": "corrected",
                "notes": "Check structural risk drift before generic checks.",
                "original_advisory_snapshot": {"fault_summary": "Original."},
                "corrected_advisory_snapshot": {
                    "fault_summary": "Engineer-corrected summary.",
                    "recommended_checks": ["Verify structural risk instability first."],
                    "approved_next_steps": ["Keep the site under review pending sign-off."],
                    "do_not_do": ["Do not advance site work before engineering review."],
                    "confidence_statement": "Engineer-reviewed correction.",
                },
                "training_ref": "feedback://review-civil-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    training_config = configs_dir / "training.yaml"
    training_config.write_text(
        "\n".join(
            [
                "metadata:",
                '  schema_version: "1.2.0"',
                '  constitution_version: "2.0.0"',
                '  model_version: "training-test"',
                '  prompt_template_version: "1.1.0"',
                "jobs:",
                "  prepare_sft_dataset:",
                "    enabled: true",
                '    input_jsonl: "data/processed/sft_fault_kb_merged.jsonl"',
                '    output_jsonl: "data/processed/qwen_sft_chat.jsonl"',
                '    feedback_training_jsonl: "data/feedback/decision_feedback_training.jsonl"',
                '    system_prompt: "You are a test assistant."',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(training, "TRAINING_CONFIG", training_config)
    training.prepare_sft_dataset()

    output_jsonl = processed_dir / "qwen_sft_chat.jsonl"
    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert rows[0]["messages"][0]["content"] == "You are a test assistant."
    assert rows[1]["metadata"]["source"] == "decision_feedback_training"
    assert "engineer_verdict: corrected" in rows[1]["messages"][1]["content"]
    assert "Verify structural risk instability first." in rows[1]["messages"][2]["content"]
