from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


def _taxonomy_dir() -> Path:
    return Path(
        os.getenv(
            "CAUREN_AGENT_TAXONOMY_DIR",
            str(Path(__file__).resolve().parent / "taxonomy"),
        )
    )


def _tuple_of_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return ()


def _human_label(value: str) -> str:
    return str(value or "").replace("_", " ").strip().title()


@lru_cache(maxsize=32)
def load_agent_taxonomy(agent_id: str) -> tuple[dict[str, Any], ...]:
    path = _taxonomy_dir() / f"{agent_id}.json"
    if not path.exists():
        return ()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return ()
    out: list[dict[str, Any]] = []
    for group in data.get("groups", []):
        if not isinstance(group, dict):
            continue
        group_id = str(group.get("group_id") or "general")
        core_patterns = _tuple_of_strings(group.get("core_patterns"))
        required_features = _tuple_of_strings(group.get("required_features"))
        optional_context = _tuple_of_strings(group.get("optional_context"))
        for anomaly_id in _tuple_of_strings(group.get("anomaly_ids")):
            out.append(
                {
                    "agent_id": str(data.get("agent_id") or agent_id),
                    "taxonomy_version": str(data.get("version") or "v1"),
                    "anomaly_id": anomaly_id,
                    "group": group_id,
                    "core_patterns": core_patterns,
                    "required_features": required_features,
                    "optional_context": optional_context,
                    "human_label_tr": _human_label(anomaly_id),
                }
            )
    return tuple(out)


def load_all_agent_taxonomies(agent_ids: list[str] | tuple[str, ...]) -> dict[str, tuple[dict[str, Any], ...]]:
    return {agent_id: load_agent_taxonomy(agent_id) for agent_id in agent_ids}
