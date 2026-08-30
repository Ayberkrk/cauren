# Contributing to Cauren

Thanks for considering a contribution. This project is a research
prototype, not a finished product, so honest, verifiable changes are
worth more here than polish.

## Getting set up

```bash
pip install fastapi pydantic pytest httpx numpy pandas psutil uvicorn pyyaml torch
python3 -m pytest tests/
```

All 36 tests should pass with no network access and no GPU. If you are
working on the `cauren-bridge` backbone, you also need `torch` and, to
rebuild the dataset from scratch, `pandas`, `pyarrow`, and `scipy`.

Read `README.md` and `operations/CAUREN_CORE_AGENT_ARCHITECTURE.md`
first. Together they document the architecture, the data flow, and a
key invariant: the civil feature list is duplicated across
`cauren_agents/civil/agent.py`, `tools/build_cauren_civil_dataset.py`,
`tools/audit_cauren_data_quality.py`, and `data/public_sources/README.md`.
If you touch one, update all four.

## What kind of contributions are useful right now

- **Bug fixes with a reproduction.** State the input, the wrong output,
  and the expected output. A failing test that your fix makes pass is
  the strongest form of this.
- **New public data sources** for features the `cauren-civil` schema
  cannot fill from any public source yet (see the README's honest
  limitations section). A source with a real, independent outcome label
  is especially valuable, the kind of thing that let `cauren-bridge` be
  trained and evaluated against a genuine 5-year deterioration outcome
  instead of a heuristic label.
- **Physics-layer relations** grounded in an actual civil/structural
  engineering principle, with a citation or a clear justification, not
  just a plausible-looking formula.
- **Tests that catch a real failure mode**, especially ones exercising
  partial or malformed real-world input rather than only the happy path.

## What to avoid

- Do not invent or approximate values for features you do not have data
  for. If a score is missing, it should show up as missing, not as a
  guess. This project treats that distinction as a hard rule.
- Do not claim a model is trained or validated unless you can point to
  the dataset, the training run, and the evaluation numbers behind that
  claim.
- Avoid adding a new required dependency for a small feature. Prefer
  the existing soft-import pattern (see the `try/except ModuleNotFoundError`
  guards at the top of `api/app.py`) if the dependency is optional.

## Pull requests

1. Run `python3 -m pytest tests/` and make sure it is green before
   opening the PR.
2. Describe what changed and why in plain terms. If the change affects
   model behavior or output shape, include a before/after example.
3. If you touched the civil feature list, confirm you updated all four
   places listed above.
4. Keep PRs scoped to one change. Large, mixed PRs are harder to review
   and harder to revert if something is wrong.

## Reporting issues

Open a GitHub issue with:

- What you expected to happen.
- What actually happened, including the exact error message if there
  is one.
- The smallest input or command that reproduces it.

Security-sensitive findings (for example, anything touching the
`GOV_PILOT_*` auth gate) should not go into a public issue. Reach out to
a maintainer directly instead.
