# OMA and Timoshenko integration

This note records why and how Cauren's operational modal analysis was changed when it adopted Timoshenko Engine 2.0 as an optional backend. Results remain decision support for engineering review, not damage verdicts.

Timoshenko is optional. `cauren_physics/timoshenko_adapter.py` loads it only when version 2.0 or newer is installed; otherwise every calculation below runs on Cauren's own pure-Python path, and both paths are tested to agree. Cauren-specific risk thresholds and interpretation stay in Cauren.

### 1. Mode pairing and resolution-limited drift

- Why: one observed mode was being reused for several reference modes. This could turn a missing mode into a false large frequency drop and a civil risk score of 1.0. Small differences below the FFT resolution could also create a watch finding.
- How: test cases were added first and run against the old implementation. They reproduced the false alarm and the resolution-limited watch finding. Pairing now uses nearest log frequency, preserves original indices, and retains at most one observation per reference. The existing absolute matching tolerance is applied after pairing. A measured difference within one frequency-resolution bin is retained in `drift_pct`, marked `resolution_limited`, assigned severity `none`, and excluded from `max_drop_pct`.
- Files: `cauren_physics/oma.py`, `cauren_physics/timoshenko_adapter.py`, `tests/test_oma_identification.py`.

### 2. Damping estimate

- Why: a single full-record periodogram reported Hann window width as physical damping and substantially underestimated damping in a noise-driven oscillator.
- How: tests for an undamped tone, a short decaying record, and a deterministic noise-driven single-degree system were run before changing the estimator. The fallback now uses a Welch power spectrum with symmetric Hann segments, half overlap, at least eight averages, interpolated half-power crossings, and a four-bin minimum bandwidth. The no-NumPy path uses a bounded 2048-sample pure-Python DFT. Timoshenko 1.x is no longer loaded because its old mode pairing and damping behavior differ from the 2.x contract.
- Files: `cauren_physics/oma.py`, `cauren_physics/timoshenko_adapter.py`, `tests/test_oma_identification.py`.

### 3. Vibration CSV loading

- Why: blank cells and blank one-column records were silently dropped, shifting the regular sample time axis. The two tools had duplicate loaders and numeric conversion could expose a raw traceback.
- How: the old behavior was covered by tests first. `csv.reader` now resolves header positions, rejects blank rows, blank selected cells, non-numeric values, and non-finite values with a file name and physical reader line number. Both tools share this loader. Multi-column loading preserves the requested channel order.
- Files: `tools/run_explainable_review.py`, `tools/run_oma_identification.py`, `tests/test_explainable_review.py`.

### 4. Multi-channel FDD

- Why: Cauren previously exposed only single-channel peak picking even though Timoshenko 2.0 already provides FDD and mode shapes.
- How: channel-major series are transposed to sample-major observations for `tm.MultiChannelData`. FDD modes become `OMAResult` modes with singular-value amplitude, relative confidence, and channel-indexed complex shape values. Empty shape output is omitted from the JSON to preserve existing single-channel output keys. Both command tools accept multi-column options and show an explicit Timoshenko version error when unavailable.
- Files: `cauren_physics/oma.py`, `cauren_physics/timoshenko_adapter.py`, `tools/run_explainable_review.py`, `tools/run_oma_identification.py`, `tests/test_oma_identification.py`, `tests/test_explainable_review.py`.
- Validation: a deterministic 100 Hz, 8192-sample, three-channel, two-mode FDD test checks frequency resolution and modal assurance criterion above 0.99 when Timoshenko is installed.

### 5. FE model consistency

- Why: the FE reference model was not exposed by the OMA command, and FE model error must not be interpreted as measured drift or damage.
- How: a separate `model_consistency` report pairs measured modes with model natural frequencies, reports one-based mode numbers, and calculates the median squared frequency ratio and its median absolute deviation percentage. The OMA CLI accepts paired mass and stiffness arrays and writes a separate `fe_model_consistency` object. The measured baseline remains the only input to drift comparison.
- Files: `cauren_physics/fe_reference_model.py`, `tools/run_oma_identification.py`, `tests/test_fe_physics_consistency.py`, `tests/test_oma_identification.py`.
- Validation: tests cover a three-story structure at stiffness ratio 0.81, a missing second mode, no measured modes, CLI output, and Timoshenko update parity when installed.

### 6. Shared stiffness update

repository-wide search confirmed that OMA, mode pairing, FDD, and shear-building frequencies already delegate through the optional adapter. The remaining duplicated shared calculation was the global stiffness update summary. It now delegates to `tm.update` when Timoshenko 2.x is available, with the existing pure-Python calculation retained when it is absent.

## Validation

The suite runs with Timoshenko absent (including without NumPy, as in CI) and with Timoshenko 2.x installed. Tests that need Timoshenko are skipped when it is absent. Each defect above was reproduced by a failing test before it was fixed.
