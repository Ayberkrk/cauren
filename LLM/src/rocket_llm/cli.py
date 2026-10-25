"""Command-line entry points for implemented, non-model LLM utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rocket_llm.pipelines.training import prepare_sft_dataset


def prepare_sft_dataset_main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert reviewed structured advisories into chat-style SFT JSONL; does not train a model."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to the rocket_llm training YAML")
    args = parser.parse_args()
    summary = prepare_sft_dataset(args.config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
