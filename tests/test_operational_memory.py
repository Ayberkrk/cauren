from __future__ import annotations

from api.operational_memory import OperationalMemoryStore


def test_operational_memory_second_lookup_hits_and_persists(tmp_path) -> None:
    state_path = tmp_path / "operational_memory.json"
    store = OperationalMemoryStore(state_path=state_path, ttl_sec=3600.0, max_entries=64)
    store.remember_success(
        asset_id="asset_1",
        sensor_layout_key="layout:civil",
        selected_agent="cauren-civil",
        agent_schema_version="v1",
        agent_registry_version="reg-v1",
        route_confidence=0.91,
        route_reason="civil_schema_hits=8",
        route_needs_context=False,
        route_ambiguity_reason="",
        candidate_agents=[{"agent_id": "cauren-civil", "score": 0.91, "reason": "civil_schema_hits=8"}],
        sector_score_breakdown=[
            {
                "agent_id": "cauren-civil",
                "sector": "civil",
                "model_evidence_score": 0.82,
                "metadata_prior_score": 0.0,
                "context_prior_score": 0.12,
                "penalty_score": 0.0,
                "final_sector_score": 0.91,
            }
        ],
        selected_sector_score=0.91,
        selected_sector_prior=0.0,
        selection_strategy="prior_weighted_score_fusion",
        prior_conflict=False,
        prior_conflict_reason="",
        feature_map_summary={"sector": "civil", "feature_order": ["structural_risk_score"]},
        rejected_sample_rate=0.0,
        anomaly_family="construction_risk",
    )

    hit, reason = store.lookup(
        asset_id="asset_1",
        sensor_layout_key="layout:civil",
        agent_registry_version="reg-v1",
    )
    assert reason == "hit"
    assert hit is not None
    assert hit.selected_agent == "cauren-civil"

    reloaded = OperationalMemoryStore(state_path=state_path, ttl_sec=3600.0, max_entries=64)
    hit_after_restart, reason_after_restart = reloaded.lookup(
        asset_id="asset_1",
        sensor_layout_key="layout:civil",
        agent_registry_version="reg-v1",
    )
    assert reason_after_restart == "hit"
    assert hit_after_restart is not None
    assert hit_after_restart.selected_agent == "cauren-civil"


def test_operational_memory_invalidates_on_registry_mismatch(tmp_path) -> None:
    store = OperationalMemoryStore(state_path=tmp_path / "operational_memory.json", ttl_sec=3600.0, max_entries=64)
    store.remember_success(
        asset_id="asset_1",
        sensor_layout_key="layout:civil",
        selected_agent="cauren-civil",
        agent_schema_version="v1",
        agent_registry_version="reg-v1",
        route_confidence=0.91,
        route_reason="civil_schema_hits=8",
        route_needs_context=False,
        route_ambiguity_reason="",
        candidate_agents=[],
        sector_score_breakdown=[],
        selected_sector_score=0.91,
        selected_sector_prior=0.0,
        selection_strategy="prior_weighted_score_fusion",
        prior_conflict=False,
        prior_conflict_reason="",
        feature_map_summary={"sector": "civil"},
        rejected_sample_rate=0.0,
        anomaly_family="construction_risk",
    )

    hit, reason = store.lookup(
        asset_id="asset_1",
        sensor_layout_key="layout:civil",
        agent_registry_version="reg-v2",
    )
    assert hit is None
    assert reason == "registry_version_mismatch"
    assert store.get_entry(asset_id="asset_1", sensor_layout_key="layout:civil") is None


def test_operational_memory_recovers_from_corrupt_state(tmp_path) -> None:
    state_path = tmp_path / "operational_memory.json"
    state_path.write_text("{bad json", encoding="utf-8")

    store = OperationalMemoryStore(state_path=state_path, ttl_sec=3600.0, max_entries=64)
    assert store.lookup(
        asset_id="asset_1",
        sensor_layout_key="layout:civil",
        agent_registry_version="reg-v1",
    ) == (None, "miss")
    assert state_path.with_suffix(state_path.suffix + ".corrupt").exists()
