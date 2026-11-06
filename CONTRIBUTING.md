# Contributing to Cauren

Thanks for considering a contribution. Cauren is a research study on
vibration-based damage detection for bridges, so honest, reproducible
changes are worth more here than polish.

## Getting set up

```bash
pip install -e '.[test]'
python3 -m pytest tests/
```

The 34 tests in the minimal set need only `pytest`: no network, no
numpy. Installing the optional study dependencies
(`pip install -e '.[study]' timoshenko-engine`) brings the suite to
76 total (see `tests/test_docs_match_test_count.py`, which fails CI if
these numbers drift from reality).

Read `README.md` and the study pages in `docs/` first.

## Ground rules for studies

- **Fit on reference data only.** Mode selection, regression
  coefficients, covariance estimates and control limits come from the
  reference period. Holdout and post-change data are only scored.
- **Say what was decided after looking.** A choice made after seeing
  post-change results (a mode subset, a threshold) is reported as
  post hoc, never as the main result.
- **Report false alarms next to detections.** A detector that finds every
  change but alarms on healthy days is not a result.
- **Never commit monitoring data.** Each dataset has its own license, and
  Z24's terms forbid passing the data on. Studies download or read data
  into gitignored folders under `data/`.

## What kind of contributions are useful right now

- **Bug fixes with a reproduction.** State the input, the wrong output,
  and the expected output. A failing test that your fix makes pass is
  the strongest form of this.
- **New detectors or normalizations** that can run under the existing
  protocol, with a reference to the method.
- **New public bridge datasets** with an independently documented damage
  or change history.
- **Tests that catch a real failure mode**, especially synthetic signals
  with a known answer.

## What to avoid

- Do not claim a method works unless you can point to the dataset, the
  protocol, and the numbers behind that claim.
- Avoid adding a new required dependency. Prefer the existing soft-import
  pattern (see `cauren_physics/oma.py` and
  `cauren_physics/timoshenko_adapter.py`) if the dependency is optional.

## Pull requests

1. Run `python3 -m pytest tests/` and make sure it is green before
   opening the PR.
2. Describe what changed and why in plain terms. If the change affects a
   study's results, update its page in `docs/` with the new numbers.
3. Keep PRs scoped to one change. Large, mixed PRs are harder to review
   and harder to revert if something is wrong.

## Reporting issues

Open a GitHub issue with:

- What you expected to happen.
- What actually happened, including the exact error message if there
  is one.
- The smallest input or command that reproduces it.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
taking part, you agree to uphold it.
