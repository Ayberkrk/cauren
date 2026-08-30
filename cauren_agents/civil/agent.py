from __future__ import annotations

from cauren_agents.base import SectorAgent
from cauren_agents.schema_loader import load_schema
from cauren_core.contracts import AgentSchema
from cauren_physics.civil import CivilPhysics


def build_civil_agent() -> SectorAgent:
    schema = AgentSchema(
        agent_id="cauren-civil",
        sector="civil",
        display_name="Cauren Civil",
        required_features=(
            "structural_risk_score",
            "inspection_finding_score",
            "permit_status_score",
            "construction_progress_pct",
            "infrastructure_connection_score",
            "natural_hazard_score",
            "occupancy_safety_score",
            "ground_stability_score",
        ),
        optional_features=(
            "building_height_m",
            "footprint_area_m2",
            "soil_settlement_mm",
            "utility_disruption_score",
        ),
        units={
            "structural_risk_score": "ratio",
            "inspection_finding_score": "ratio",
            "permit_status_score": "ratio",
            "construction_progress_pct": "%",
            "infrastructure_connection_score": "ratio",
            "natural_hazard_score": "ratio",
            "occupancy_safety_score": "ratio",
            "ground_stability_score": "ratio",
            "building_height_m": "m",
            "footprint_area_m2": "m2",
            "soil_settlement_mm": "mm",
            "utility_disruption_score": "ratio",
        },
        aliases={
            "structural_risk_score": ("structural_risk", "building_risk_score", "risk_score"),
            "inspection_finding_score": ("inspection_score", "finding_score", "field_review_score"),
            "permit_status_score": ("approval_score", "document_readiness_score", "permit_score"),
            "construction_progress_pct": ("progress_pct", "completion_pct", "construction_progress"),
            "infrastructure_connection_score": (
                "utility_connection_score",
                "readiness_score",
                "infrastructure_readiness",
            ),
            "natural_hazard_score": ("earthquake_risk_score", "flood_risk_score", "natural_risk_score"),
            "occupancy_safety_score": ("life_safety_score", "occupancy_risk_score"),
            "ground_stability_score": ("ground_risk_score", "geotechnical_risk_score", "soil_stability_score"),
            "building_height_m": ("height_m", "structure_height_m"),
            "footprint_area_m2": ("area_m2", "building_area_m2"),
            "soil_settlement_mm": ("settlement_mm", "foundation_settlement_mm"),
            "utility_disruption_score": ("service_disruption_score", "utility_risk_score"),
        },
    )
    return SectorAgent(schema=load_schema(schema), physics=CivilPhysics())
