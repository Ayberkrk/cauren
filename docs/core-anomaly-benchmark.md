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

## After: reference-based scoring with Holm correction

With a per-asset reference (the `baseline` from the request or from
`POST /baseline/{asset_id}`, see
[API history and access controls](api-history-and-access.md)), each feature is
tested against its own reference center and scale:

| Test | Statistic | Null distribution | Pattern key |
|---|---|---|---|
| spike / dip | largest distance from the window median, in reference scales, inflated by sqrt(1 + 1/n) | Sidak over the n samples | `spike` or `dip` |
| step | difference of half-window medians | normal with median variance pi / (2 n) per half | `level_shift` |
| drift | Theil-Sen slope x observed span | least-squares slope spread / sqrt(0.91) | `drift` |
| level | window mean minus reference center | 1/n + pi / (2 n_ref) | `level_shift` |
| flatline | window spread below 0.1 x scale, only when `expected_variation` | chi-square lower tail | `flatline` |

The reference is itself estimated, so every tail comes from Student's t with
0.37 x (n_ref - 1) degrees of freedom (the efficiency of the MAD scale)
rather than the normal. All tests of a diagnosis are corrected together with
Holm's step-down procedure at alpha 0.05, and the smallest adjusted p-value
p is mapped to `risk = 1 - (1 - 0.34) ** (ln p / ln 0.05)`, so the 0.34 civil
threshold is exactly "significant at a 5% family-wise rate". Physics
evidence is still suppressed; it can raise the diagnosis risk but no longer
dilutes the core risk below its threshold. The anomaly label comes from the
most significant shape test (spike, step, drift, flatline); when both drift
and step are significant for the same feature, the better-fitting shape
(line or two-level step) names it. Patterns without a reference test
(oscillation, variance change, correlation break, and the others) are still
reported as window-relative shape descriptors but no longer drive risk.

Without a reference the core reports `insufficient_validated_evidence`,
risk 0, and reference `none`.

| Scenario | Alarm expected | Before | After | After, Wilson 95% | After, most common anomaly_type | Target |
|---|---|---|---|---|---|---|
| clean_noise | no | 3.6% | 1.4% | 0.7% to 2.9% | nominal_variation | <= 5% |
| constant_expected_variation | yes | 0.4% | 100.0% | 99.2% to 100.0% | flatline | >= 90% |
| constant_naturally_constant | no | 0.4% | 2.2% | 1.2% to 3.9% | nominal_variation | <= 2% |
| constant_all_features | yes | 100.0% | 100.0% | 99.2% to 100.0% | flatline | |
| drift_3sigma | yes | 1.0% | 99.4% | 98.3% to 99.8% | drift | >= 90% |
| step_3sigma | yes | 1.2% | 100.0% | 99.2% to 100.0% | level_shift | >= 90% |
| spike_6sigma | yes | 5.8% | 92.6% | 90.0% to 94.6% | spike | >= 85% |
| drift_1sigma | yes | 3.6% | 5.2% | 3.6% to 7.5% | nominal_variation | |
| step_1sigma | yes | 2.0% | 9.8% | 7.5% to 12.7% | nominal_variation | |
| spike_1sigma | yes | 3.6% | 1.4% | 0.7% to 2.9% | nominal_variation | |

Without a reference (`--no-reference`) every scenario alarms 0.0% of the time
(Wilson upper bound 0.8%): the core reports insufficient evidence instead.

Reading the table:

- `tests/test_core_anomaly_benchmark.py` checks each target against its
  Wilson 95% interval. Every target holds at its point estimate except
  `constant_naturally_constant`, at 2.2% against 2%. That rate is not caused
  by the constant feature (it never fires): it is the family-wise false-alarm
  rate of the other seven noisy features, statistically indistinguishable
  from the 1.4% clean rate.
- 1 sigma changes are mostly not detected. Sixteen samples cannot separate a
  1 sigma shift from noise at a 5% family-wise rate; detecting them would
  need a longer window or a looser rate.
- Labels are less certain than alarms. About half of the 3 sigma drifts are
  labeled `level_shift`, because over 16 noisy samples a ramp and a step fit
  almost equally well; 3 sigma steps are labeled `level_shift` about 90% of
  the time.
- The null model is independent Gaussian noise around an exactly known
  reference. Real civil records are often autocorrelated, seasonal, or
  revised in batches; on such data the false-alarm rate will be higher than
  shown here, and the reference should be refitted from a reviewed period.
  These rates describe the detector, not any structure's condition.
