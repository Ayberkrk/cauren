"""The runtime state store and the in-process report worker are both opt-in.

They used to be unreachable: `state_store_enabled` was hardcoded False and
`_restore_runtime_state` / `_report_worker` were never called, so the service
persisted state it could never read back and shipped a worker that never ran.
These tests pin down both the default (off, nothing started) and the opt-in
path (on, and actually round-tripping).
"""

import json
import os
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ.setdefault("GOV_PILOT_API_TOKEN", "test-only-cauren-pilot-token")

from api.app import (  # noqa: E402
    _persist_runtime_state_if_due,
    _restore_runtime_state,
    app,
)


class _FakeLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _fake_app(**state):
    """Minimal stand-in for the FastAPI app: the state-store helpers only ever
    touch `.state`, so a real app (and its startup cost) isn't needed here.
    """
    return SimpleNamespace(state=SimpleNamespace(**state))


def _app_with_store(path, *, enabled: bool):
    return _fake_app(
        state_store_enabled=enabled,
        state_store_path=path,
        state_store_ttl_sec=21600,
        state_store_write_interval_sec=0.0,
        state_store_last_persist_ts=0.0,
        state_store_lock=_FakeLock(),
        state_store_max_guard_entries=2048,
        state_store_max_asset_entries=1024,
        calibration_guard_lock=_FakeLock(),
        calibration_guard_states={},
        asset_context_summaries={},
        asset_contexts={},
        asset_contexts_lock=_FakeLock(),
    )


def test_state_store_is_disabled_by_default_and_writes_nothing(tmp_path):
    path = tmp_path / "runtime_state_store.json"
    app_ref = _app_with_store(path, enabled=False)

    _persist_runtime_state_if_due(app_ref, force=True)

    assert not path.exists()


def test_disabled_restore_reports_off_and_leaves_state_untouched(tmp_path):
    path = tmp_path / "runtime_state_store.json"
    app_ref = _app_with_store(path, enabled=False)

    _restore_runtime_state(app_ref)

    assert app_ref.state.continuity_mode == "off"
    assert app_ref.state.state_store_connected is False
    assert app_ref.state.calibration_guard_states == {}


def test_enabled_store_round_trips_guard_state_across_a_restart(tmp_path):
    """The point of the store: state written before a restart comes back after
    it. Before this was wired, the write happened and the read never did.
    """
    path = tmp_path / "runtime_state_store.json"

    writer = _app_with_store(path, enabled=True)
    writer.state.calibration_guard_states = {"asset-1|civil": {"mode": "safe", "_updated_at": 1e12}}
    _persist_runtime_state_if_due(writer, force=True)

    assert path.exists(), "enabling the store should actually write it"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "guard_state" in payload

    # A fresh process boots with empty in-memory state and restores from disk.
    reader = _app_with_store(path, enabled=True)
    _restore_runtime_state(reader)

    assert reader.state.continuity_mode == "durable"
    assert reader.state.state_store_connected is True
    assert "asset-1|civil" in reader.state.calibration_guard_states


def test_enabled_store_drops_entries_past_their_ttl(tmp_path):
    path = tmp_path / "runtime_state_store.json"

    writer = _app_with_store(path, enabled=True)
    writer.state.calibration_guard_states = {"stale|civil": {"mode": "safe", "_updated_at": 1.0}}
    _persist_runtime_state_if_due(writer, force=True)

    reader = _app_with_store(path, enabled=True)
    reader.state.state_store_ttl_sec = 1  # anything written at t=1.0 is long expired
    _restore_runtime_state(reader)

    assert "stale|civil" not in reader.state.calibration_guard_states
    assert reader.state.continuity_mode == "durable"


def test_startup_runs_the_restore_path_and_starts_no_worker_by_default():
    """Regression: `_restore_runtime_state` and `_report_worker` were both
    defined but never invoked. Startup must now call restore (which, with the
    store off, settles continuity_mode to "off" rather than leaving the
    boot-time placeholder) and must still not start an in-process worker
    unless REPORT_WORKER_MODE=internal.
    """
    assert os.getenv("REPORT_WORKER_MODE", "external").strip().lower() != "internal"

    with TestClient(app):
        assert app.state.report_worker_mode == "external"
        assert app.state.report_worker is None, "external mode must not spawn an in-process worker"
        # Only _restore_runtime_state sets this to "off"; the initializer
        # leaves it as "cauren_core".
        assert app.state.continuity_mode == "off"
        assert app.state.state_store_enabled is False
