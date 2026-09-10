## What changed

## Why

## Testing

- [ ] `python3 -m pytest tests/` passes locally
- [ ] Added or updated a test covering this change
- [ ] If this touches the civil feature list, all four places are
      updated (`cauren_agents/civil/agent.py`,
      `tools/build_cauren_civil_dataset.py`,
      `tools/audit_cauren_data_quality.py`,
      `data/public_sources/README.md`)
- [ ] If this touches the pipeline or the core runtime, no per-request
      state was added to anything shared across requests (see
      CONTRIBUTING.md and `tests/test_pipeline_concurrency.py`)

## Notes for the reviewer

Anything that needs extra attention, known limitations, or follow-up
work this PR intentionally leaves out.
