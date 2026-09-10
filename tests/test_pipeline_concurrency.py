"""The API shares one CaurenPipeline across a threadpool (api/app.py wraps
every diagnose/calibrate call in run_in_threadpool against
app.state.cauren_pipeline), so the pipeline and the core runtime under it
have to be safe to call concurrently.
"""

import threading

from cauren_core import CaurenPipeline


def _payload(base_value: float):
    return {
        "agent_id": "cauren-civil",
        "sensors": [
            {
                "sensor_id": f"s{step}",
                "name": "structural_risk_score",
                "unit": "ratio",
                "value": base_value + 0.01 * step,
                "timestamp": float(step),
            }
            for step in range(8)
        ],
    }


def test_concurrent_diagnoses_do_not_cross_contaminate_backbone_metadata():
    """Regression: the runtime used to stash the last calibration's backbone
    metadata on itself, so one thread resetting it mid-flight left another
    thread reporting empty (or another request's) backbone_metadata.
    """
    pipeline = CaurenPipeline.from_default_registry()
    empty_metadata_count = 0
    errors: list[BaseException] = []
    counter_lock = threading.Lock()

    def worker(base_value: float) -> None:
        nonlocal empty_metadata_count
        try:
            for _ in range(40):
                diagnosis = pipeline.diagnose(_payload(base_value))
                if not diagnosis.core_output.backbone_metadata:
                    with counter_lock:
                        empty_metadata_count += 1
        except BaseException as exc:  # noqa: BLE001 - surfaced via the assert below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(0.1 * idx,)) for idx in range(1, 9)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"concurrent diagnose raised: {errors[0]!r}"
    assert empty_metadata_count == 0


def test_concurrent_diagnoses_stay_deterministic_per_input():
    """Same input must produce the same risk score whether it runs alone or
    alongside other requests on the shared pipeline.
    """
    pipeline = CaurenPipeline.from_default_registry()
    expected = pipeline.diagnose(_payload(0.5)).risk_score

    observed: list[float] = []
    observed_lock = threading.Lock()

    def worker(base_value: float) -> None:
        for _ in range(25):
            diagnosis = pipeline.diagnose(_payload(base_value))
            if base_value == 0.5:
                with observed_lock:
                    observed.append(diagnosis.risk_score)

    threads = [threading.Thread(target=worker, args=(value,)) for value in (0.5, 0.2, 0.5, 0.9, 0.5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert observed, "expected the 0.5 workers to record results"
    assert all(score == expected for score in observed)
