# rocket-llm

Advisory LLM scaffold for Cauren (fault/diagnosis explanation and briefing
generation), packaged separately from the root `cauren` package.

**Status: pre-alpha.** `pyproject.toml` declares dependency groups and a
package layout, but only `src/rocket_llm/pipelines/training.py` is
implemented so far; there is no CLI yet (see the root repository's issue
tracker for the entry-point rollout plan).

## Development setup

```bash
cd LLM
pip install -e '.[dev]'
pytest
```
