from __future__ import annotations

import os
import sys
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.asset_identity import AssetIdentityResolver


def test_asset_identity_pending_then_approved(tmp_path: Path) -> None:
    resolver = AssetIdentityResolver(sqlite_path=tmp_path / "asset_identity.sqlite3")
    first = resolver.resolve(
        asset_id="auto",
        site_id="plant_1",
        line_id="line_a",
        machine_id="machine_01",
    )
    assert first.resolution_mode == "pending_proposal"
    assert first.review_required is True
    assert first.proposal_id

    reviewed = resolver.review(
        proposal_id=str(first.proposal_id),
        action="approve",
        reviewer="engineer-1",
        notes="Looks good.",
    )
    assert reviewed["status"] == "approved"

    second = resolver.resolve(
        asset_id="auto",
        site_id="plant_1",
        line_id="line_a",
        machine_id="machine_01",
    )
    assert second.resolution_mode == "approved_mapping"
    assert second.review_required is False


def test_asset_identity_conflict_creates_pending(tmp_path: Path) -> None:
    resolver = AssetIdentityResolver(sqlite_path=tmp_path / "asset_identity.sqlite3")
    first = resolver.resolve(
        asset_id="auto",
        site_id="plant_1",
        line_id="line_a",
        machine_id="machine_01",
    )
    resolver.review(
        proposal_id=str(first.proposal_id),
        action="approve",
        reviewer="engineer-1",
    )
    second = resolver.resolve(
        asset_id="auto",
        site_id="plant_1",
        line_id="line_b",
        machine_id="machine_01",
    )
    assert second.resolution_mode == "pending_proposal"
    assert second.review_required is True


def test_explicit_asset_id_bypasses_pending(tmp_path: Path) -> None:
    resolver = AssetIdentityResolver(sqlite_path=tmp_path / "asset_identity.sqlite3")
    resolved = resolver.resolve(
        asset_id="line_1",
        site_id=None,
        line_id=None,
        machine_id=None,
    )
    assert resolved.resolution_mode == "provided"
    assert resolved.resolved_asset_id == "line_1"
