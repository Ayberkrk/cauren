from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cauren_agents.registry import build_registry


@dataclass(frozen=True)
class TenantRuntimeConfig:
    tenant_id: str
    tenant_label: str
    active_agent_id: str
    active_sector: str
    workspace_title: str
    lock_agent_routing: bool = True
    facility_profile_path: str = ""
    can_signal_profile_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _agent_sector(agent_id: str) -> str:
    registry = build_registry([agent_id])
    return registry.get(agent_id).schema.sector


def load_tenant_runtime_config(path: str | Path | None) -> TenantRuntimeConfig | None:
    if path is None:
        return None
    file_path = Path(path).expanduser()
    if not file_path.exists():
        return None
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Tenant runtime config must be an object: {file_path}")
    active_agent_id = str(raw.get("active_agent_id") or "").strip()
    if not active_agent_id:
        raise ValueError(f"Tenant runtime config is missing active_agent_id: {file_path}")
    tenant_id = str(raw.get("tenant_id") or file_path.stem).strip() or file_path.stem
    tenant_label = str(raw.get("tenant_label") or tenant_id).strip() or tenant_id
    active_sector = str(raw.get("active_sector") or "").strip() or _agent_sector(active_agent_id)
    workspace_title = str(raw.get("workspace_title") or f"{tenant_label} Civil Workspace").strip() or f"{tenant_label} Civil Workspace"
    return TenantRuntimeConfig(
        tenant_id=tenant_id,
        tenant_label=tenant_label,
        active_agent_id=active_agent_id,
        active_sector=active_sector,
        workspace_title=workspace_title,
        lock_agent_routing=bool(raw.get("lock_agent_routing", True)),
        facility_profile_path=str(raw.get("facility_profile_path") or "").strip(),
        can_signal_profile_path=str(raw.get("can_signal_profile_path") or raw.get("facility_profile_path") or "").strip(),
    )


def load_tenant_runtime_config_from_env() -> TenantRuntimeConfig | None:
    raw = str(os.getenv("CAUREN_TENANT_CONFIG_PATH", "")).strip()
    if not raw:
        return None
    return load_tenant_runtime_config(raw)
