from __future__ import annotations

from pathlib import Path
from typing import Any


def serialize_release_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in bundle.items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def resolve_release_bundle(base_dir: Path) -> dict[str, Any]:
    base_dir = Path(base_dir)
    civil_bundle = base_dir / 'cauren_agents' / 'taxonomy' / 'cauren-civil.json'
    return {
        'mode': 'civil_only',
        'source': 'cauren_civil_stub',
        'release_dir': None,
        'manifest_path': None,
        'artifact_path': civil_bundle,
        'metrics_path': civil_bundle,
        'kb_path': civil_bundle,
        'revision_tag': 'cauren-civil-v1',
        'validation_ok': civil_bundle.exists(),
        'issues': [] if civil_bundle.exists() else ['civil_taxonomy_missing'],
        'missing_files': [] if civil_bundle.exists() else [str(civil_bundle)],
        'pointer_path': base_dir / 'cauren_agents' / 'taxonomy' / 'cauren-civil.json',
    }
