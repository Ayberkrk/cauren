from __future__ import annotations

from cauren_agents.base import SectorAgent
from cauren_agents.schema_loader import load_schema
from cauren_core.contracts import AgentSchema
from cauren_physics.bridge import BridgePhysics


def build_bridge_agent() -> SectorAgent:
    """Bridge-vertical sector agent.

    Scoped to the 3 features that a real, public, per-bridge ground-truth
    source actually supplies -- see data/cauren_bridge/dataset_summary.json
    (FHWA National Bridge Inventory condition ratings + USGS seismic and
    FEMA flood hazard). This is intentionally narrower than cauren-civil's
    8-feature building schema: it exists so the backbone can be trained
    and validated against a real, independently-observed outcome
    (deck_drop_5yr), not to replace the building schema.
    """

    schema = AgentSchema(
        agent_id="cauren-bridge",
        sector="civil",
        display_name="Cauren Bridge",
        required_features=(
            "structural_risk_score",
            "ground_stability_score",
            "natural_hazard_score",
        ),
        units={
            "structural_risk_score": "ratio",
            "ground_stability_score": "ratio",
            "natural_hazard_score": "ratio",
        },
        aliases={
            "structural_risk_score": ("deck_condition_risk", "condition_risk_score", "structural_risk"),
            "ground_stability_score": ("scour_stability_score", "foundation_stability_score", "scour_score"),
            "natural_hazard_score": ("seismic_flood_hazard_score", "hazard_exposure_score"),
        },
    )
    return SectorAgent(schema=load_schema(schema), physics=BridgePhysics())
