from __future__ import annotations

import json
from pathlib import Path

import pytest

from rocket_llm import cli
from rocket_llm.pipelines.training import TrainingGuardError, prepare_sft_dataset


def _fixture_config(tmp_path: Path, *, enabled: bool = True) -> tuple[Path, Path, Path]:
    config_dir = tmp_path / "configs"
    data_dir = tmp_path / "data" / "processed"
    config_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    input_path = data_dir / "input.jsonl"
    input_path.write_text(
        json.dumps({"input": {"evidence": "inspection record"}, "target": {"summary": "review required"}}) + "\n",
        encoding="utf-8",
    )
    config_path = config_dir / "training.yaml"
    config_path.write_text(
        "jobs:\n"
        "  prepare_sft_dataset:\n"
        f"    enabled: {'true' if enabled else 'false'}\n"
        '    input_jsonl: "data/processed/input.jsonl"\n'
        '    output_jsonl: "data/processed/output.jsonl"\n'
        '    feedback_training_jsonl: "data/feedback/feedback.jsonl"\n'
        '    system_prompt: "Return evidence-grounded JSON."\n',
        encoding="utf-8",
    )
    return config_path, input_path, data_dir / "output.jsonl"


def test_prepare_cli_writes_rows_and_reports_counts(tmp_path: Path, monkeypatch, capsys):
    config_path, _input_path, output_path = _fixture_config(tmp_path)
    monkeypatch.setattr("sys.argv", ["rocket-llm-prepare-sft", "--config", str(config_path)])

    assert cli.prepare_sft_dataset_main() == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["total_rows"] == 1
    assert summary["engineer_reviewed_feedback_rows"] == 0
    assert output_path.exists()
    row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["messages"][0]["content"] == "Return evidence-grounded JSON."


def test_sft_preparation_refuses_disabled_job(tmp_path: Path):
    config_path, _input_path, _output_path = _fixture_config(tmp_path, enabled=False)

    with pytest.raises(TrainingGuardError, match="is disabled"):
        prepare_sft_dataset(config_path)


def test_sft_preparation_resolves_monkeypatched_default_config_at_call_time(tmp_path: Path, monkeypatch):
    config_path, _input_path, output_path = _fixture_config(tmp_path)
    monkeypatch.setattr("rocket_llm.pipelines.training.TRAINING_CONFIG", config_path)

    summary = prepare_sft_dataset()

    assert summary["total_rows"] == 1
    assert output_path.exists()


def test_sft_preparation_rejects_malformed_jsonl_instead_of_dropping_rows(tmp_path: Path):
    config_path, input_path, output_path = _fixture_config(tmp_path)
    input_path.write_text('{"input": {}}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"input.jsonl:2"):
        prepare_sft_dataset(config_path)

    assert not output_path.exists()
