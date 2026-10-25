# Statistical core: synthetic alarm-rate benchmark

`tools/benchmark_core_anomaly.py` measures how often the full diagnosis
pipeline raises an alarm on synthetic civil windows whose ground truth is
known. It exists so changes to the statistical anomaly core
(`cauren_core/runtime.py`, `cauren_core/risk_calibration.py`) are judged on
false-alarm and detection rates rather than on a handful of hand-picked
examples.

This is a synthetic benchmark. It checks that the core behaves sensibly on
independent Gaussian noise with one controlled change; it says nothing about
how often real civil records drift, how autocorrelated they are, or whether
an alarm corresponds to an engineering problem. It is decision-support
evidence about the detector, not a validation of any structure.

## Protocol

- Schema: the eight required features of `cauren-civil`
  (`cauren_agents/civil/agent.py`), 16 time steps per window (`seq_len` 16).
- Reference: each feature has a per-asset reference center and scale (see
  `REFERENCE` in the tool). Clean data is independent Gaussian noise around
  the center with standard deviation equal to the scale.
- Change: at most one change, applied to ONE feature
  (`structural_risk_score` unless stated). Magnitudes are in reference-scale
  units ("sigma"):
  - drift: linear ramp starting at the reference level and ending `k` sigma
    away on the last step;
  - step: `+k` sigma over the second half of the window;
  - spike: one sample at a random step, `+k` sigma.
- Constant scenarios: `constant_expected_variation` freezes a feature that
  normally varies (a stuck sensor); `constant_naturally_constant` freezes
  `permit_status_score`, whose reference marks it as legitimately constant;
  `constant_all_features` freezes the whole feed.
- Trials: 500 per scenario, seeded numpy (`--seed 20261025`); each scenario
  draws from its own stream so adding scenarios does not change the others.
- Alarm: `diagnosis.risk_score >= 0.34`, the civil decision threshold, from
  `CaurenPipeline.from_default_registry().diagnose(payload)`.
- The reference is sent in every payload under `baseline`. Rates are reported
  with Wilson 95% intervals.

Reproduce with:

```bash
pip install -e '.[test]' numpy
python3 tools/benchmark_core_anomaly.py
```

## Before: window-normalized core (main at 7bd185a)

The core at this commit has no notion of a per-asset reference and ignores
the `baseline` key. Every pattern score is normalized by the window's own
spread, and the diagnosis risk is `0.52 x core risk` because the physics
contribution is suppressed until rules are validated.

| Scenario | Alarm expected | Alarm rate | Wilson 95% | Most common anomaly_type |
|---|---|---|---|---|
| clean_noise | no | 3.6% | 2.3% to 5.6% | spike |
| constant_expected_variation | yes | 0.4% | 0.1% to 1.4% | spike |
| constant_naturally_constant | no | 0.4% | 0.1% to 1.4% | spike |
| constant_all_features | yes | 100.0% | 99.2% to 100.0% | flatline |
| drift_3sigma | yes | 1.0% | 0.4% to 2.3% | spike |
| step_3sigma | yes | 1.2% | 0.6% to 2.6% | spike |
| spike_6sigma | yes | 5.8% | 4.1% to 8.2% | spike |
| drift_1sigma | yes | 3.6% | 2.3% to 5.6% | spike |
| step_1sigma | yes | 2.0% | 1.1% to 3.6% | spike |
| spike_1sigma | yes | 3.6% | 2.3% to 5.6% | spike |

What this shows:

- Alarms are essentially independent of the injected magnitude. A 3 sigma
  step or drift alarms about as often as clean noise, and a 6 sigma spike
  barely more. Window-normalized scores cannot see how large a change is
  relative to the asset's normal behavior.
- The dominant label is `spike` in every scenario, including clean noise,
  because the spike term takes the maximum over features of a per-feature
  95th percentile.
- A whole frozen feed always alarms as `flatline`, but one stuck feature
  among eight is invisible, and the core cannot tell a stuck sensor from a
  feature that is constant by nature.
