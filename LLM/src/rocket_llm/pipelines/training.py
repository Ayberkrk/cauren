from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


TRAINING_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "training.yaml"


class TrainingGuardError(RuntimeError):
    pass


class TrainingPipelineRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.payload = self._load_yaml(self.path)
        self.root_dir = self.path.parent.parent

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Expected mapping in {path}")
        return data

    def ensure_enabled(self, job_name: str) -> dict[str, Any]:
        jobs = self.payload.get("jobs")
        if not isinstance(jobs, dict):
            raise TrainingGuardError("Training config is missing jobs map.")
        job = jobs.get(job_name)
        if not isinstance(job, dict):
            raise TrainingGuardError(f"Training job {job_name} is not defined in {self.path}.")
        if not bool(job.get("enabled", False)):
            raise TrainingGuardError(f"Training job {job_name} is disabled.")
        return job

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.root_dir / path


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _append_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _structured_to_chat_rows(*, input_structured_jsonl: Path, system_prompt: str) -> list[dict[str, Any]]:
    rows = _read_jsonl_rows(input_structured_jsonl)
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        input_payload = row.get("input") if isinstance(row.get("input"), dict) else {}
        target_payload = row.get("target") if isinstance(row.get("target"), dict) else {}
        out.append(
            {
                "record_id": f"structured-{index:06d}",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(input_payload, ensure_ascii=True, sort_keys=True)},
                    {"role": "assistant", "content": json.dumps(target_payload, ensure_ascii=True)},
                ],
                "metadata": {"source": "structured_sft"},
            }
        )
    return out


def _feedback_user_prompt(row: dict[str, Any]) -> str:
    asset_class = str(row.get("asset_class") or "cauren-civil")
    fault_family_id = str(row.get("fault_family_id") or "unknown_anomaly")
    verdict = str(row.get("verdict") or "reviewed")
    notes = str(row.get("notes") or "")
    original_snapshot = row.get("original_advisory_snapshot")
    original_json = json.dumps(
        original_snapshot if isinstance(original_snapshot, dict) else {},
        ensure_ascii=True,
        sort_keys=True,
    )
    return "\n".join(
        [
            "Revise the advisory using engineer-reviewed corrections and return strict advisory JSON.",
            f"asset_class: {asset_class}",
            f"fault_family_id: {fault_family_id}",
            f"engineer_verdict: {verdict}",
            f"engineer_notes: {notes or 'none'}",
            f"original_advisory_json: {original_json}",
        ]
    )


def _convert_feedback_records_to_sft_rows(*, feedback_jsonl: Path, system_prompt: str) -> list[dict[str, Any]]:
    rows = _read_jsonl_rows(feedback_jsonl)
    chat_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        corrected = row.get("corrected_advisory_snapshot")
        if not isinstance(corrected, dict) or not corrected:
            continue
        review_id = str(row.get("review_id") or row.get("decision_id") or f"feedback-{index:06d}")
        chat_rows.append(
            {
                "record_id": f"feedback-{review_id}-{index:06d}",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _feedback_user_prompt(row)},
                    {"role": "assistant", "content": json.dumps(corrected, ensure_ascii=True)},
                ],
                "metadata": {
                    "source": "decision_feedback_training",
                    "decision_id": str(row.get("decision_id") or ""),
                    "review_id": str(row.get("review_id") or ""),
                    "asset_class": str(row.get("asset_class") or ""),
                    "fault_family_id": str(row.get("fault_family_id") or ""),
                    "verdict": str(row.get("verdict") or ""),
                    "training_ref": str(row.get("training_ref") or ""),
                },
            }
        )
    return chat_rows


def prepare_sft_dataset() -> None:
    registry = TrainingPipelineRegistry(TRAINING_CONFIG)
    job = registry.ensure_enabled("prepare_sft_dataset")
    input_jsonl = registry.resolve_path(str(job.get("input_jsonl", "data/processed/sft_fault_kb_merged.jsonl")))
    output_jsonl = registry.resolve_path(str(job.get("output_jsonl", "data/processed/qwen_sft_chat.jsonl")))
    feedback_training_jsonl = str(job.get("feedback_training_jsonl", "data/feedback/decision_feedback_training.jsonl")).strip()
    feedback_path = registry.resolve_path(feedback_training_jsonl) if feedback_training_jsonl else None
    system_prompt = str(
        job.get(
            "system_prompt",
            "You are a civil engineering advisory assistant. Return strict JSON and only evidence-grounded recommendations.",
        )
    )
    structured_rows = _structured_to_chat_rows(input_structured_jsonl=input_jsonl, system_prompt=system_prompt)
    _write_jsonl_rows(output_jsonl, structured_rows)
    if feedback_path is not None and feedback_path.exists():
        _append_jsonl_rows(
            output_jsonl,
            _convert_feedback_records_to_sft_rows(feedback_jsonl=feedback_path, system_prompt=system_prompt),
        )
