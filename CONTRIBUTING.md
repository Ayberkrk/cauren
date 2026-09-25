# Contributing to Cauren

Thanks for considering a contribution. This project is a research
prototype, not a finished product, so honest, verifiable changes are
worth more here than polish.

## Getting set up

Running the test suite needs only these:

```bash
pip install -e '.[test]'
python3 -m pytest tests/
```

(equivalent to `pip install fastapi pydantic pytest httpx` if you'd
rather not use the `pyproject.toml` extra)

All 115 tests should pass with no network access, no GPU, and without
`numpy` or `torch` installed. Add `uvicorn` to run the API itself. If
you are working on the `cauren-bridge` backbone you also need `torch`,
`pyyaml` for the `LLM/` sub-project, and, to rebuild the bridge dataset
from scratch, `pandas` and `scipy` (`pip install -e '.[backbone,dataset]'`
covers both of the latter two cases) -- installing all three brings 15
more tests into scope, for 130 total (see `tests/test_docs_match_test_count.py`,
which fails CI if this number ever drifts from reality).

Read `README.md` and `operations/CAUREN_CORE_AGENT_ARCHITECTURE.md`
first. Together they document the architecture and the data flow.

## Two invariants worth knowing before you change anything

**The civil feature list is duplicated on purpose.** It lives in
`cauren_agents/civil/agent.py`, `tools/build_cauren_civil_dataset.py`,
`tools/audit_cauren_data_quality.py`, and
`data/public_sources/README.md`. If you touch one, update all four.

**The pipeline must stay safe to call concurrently.** The API keeps a
single `CaurenPipeline` on `app.state` and dispatches every
diagnose/calibrate call through a threadpool, so anything reachable
from `CaurenPipeline.diagnose` is shared across in-flight requests. Do
not store per-request state on `CaurenPipeline`, `CaurenCoreRuntime`, or
anything they hold; return it instead. This was a real bug once (one
request could read another's backbone metadata), and
`tests/test_pipeline_concurrency.py` exists to keep it from coming back.

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

Security-sensitive findings should not go into a public issue.
Reach out to a maintainer directly instead. Note that `api/app.py`
ships with no authentication by design (see README's API section) --
that is a documented design choice for a research prototype, not
something to report as a bug on its own.
