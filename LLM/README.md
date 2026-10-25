# rocket-llm

Advisory LLM scaffold for Cauren (fault/diagnosis explanation and briefing
generation), packaged separately from the root `cauren` package.

**Status: pre-alpha.** The only executable slice currently implemented is
`rocket-llm-prepare-sft`, which converts structured records and
engineer-reviewed corrections into chat-style SFT JSONL. It does not call a
model, train weights, or generate engineering advice. The job stays disabled
in the default training config until an operator enables it.

## Development setup

```bash
cd LLM
pip install -e '.[dev]'
pytest
```

After enabling `prepare_sft_dataset` in the training YAML, run
`rocket-llm-prepare-sft --config configs/training.yaml` from this directory.
The command reports row counts and stops on malformed JSONL instead of
silently dropping records.
