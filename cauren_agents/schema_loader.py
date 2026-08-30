from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from cauren_core.contracts import AgentSchema


def _schema_dir() -> Path:
    return Path(
        os.getenv(
            "CAUREN_AGENT_SCHEMA_DIR",
            str(Path(__file__).resolve().parent / "schemas"),
        )
    )


def _tuple_of_strings(value: Any, fallback: tuple[str, ...] = ()) -> tuple[str, ...]:
    if value is None:
        return fallback
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return fallback


def _aliases(value: Any, fallback: Mapping[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        return dict(fallback)
    return {str(key): _tuple_of_strings(items) for key, items in value.items()}


def _units(value: Any, fallback: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, dict):
        return dict(fallback)
    return {str(key): str(item) for key, item in value.items()}


def _feature_metadata(
    value: Any,
    fallback: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {str(key): dict(item) for key, item in fallback.items()}
    out: dict[str, dict[str, Any]] = {}
    for key, item in value.items():
        if not isinstance(item, dict):
            continue
        out[str(key)] = {str(meta_key): meta_value for meta_key, meta_value in item.items()}
    return out or {str(key): dict(item) for key, item in fallback.items()}


def load_schema(default: AgentSchema) -> AgentSchema:
    path = _schema_dir() / f"{default.agent_id}.json"
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return default
    return AgentSchema(
        agent_id=str(data.get("agent_id") or default.agent_id),
        sector=str(data.get("sector") or default.sector),
        display_name=str(data.get("display_name") or default.display_name),
        required_features=_tuple_of_strings(data.get("required_features"), default.required_features),
        optional_features=_tuple_of_strings(data.get("optional_features"), default.optional_features),
        aliases=_aliases(data.get("aliases"), default.aliases),
        units=_units(data.get("units"), default.units),
        feature_metadata=_feature_metadata(data.get("feature_metadata"), default.feature_metadata),
        version=str(data.get("version") or default.version),
    )
