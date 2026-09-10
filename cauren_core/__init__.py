from typing import TYPE_CHECKING

from .contracts import (
    AgentCandidate,
    AgentSchema,
    CaurenDiagnosis,
    CoreOutput,
    FeatureMetadata,
    NormalizationDecision,
    NormalizationRegistryEntry,
    NormalizationTrace,
    NormalizedSensorReading,
    PhysicsEvidence,
    SensorReading,
    SensorWindow,
)
from .quality_control import QualityControlReport, QualityFinding, evaluate_quality
from .uncertainty import UncertaintyFactor, UncertaintyReport, estimate_uncertainty

if TYPE_CHECKING:
    # Imported eagerly for type checkers only. At runtime these stay behind
    # the module __getattr__ below so importing cauren_core does not pull in
    # torch (backbone/training) or the whole agent registry (orchestrator).
    from .backbone import CaurenHybridBackbone, HybridBackboneUnavailable
    from .explanations import render_diagnosis_explanation, render_physics_evidence
    from .orchestrator import CaurenPipeline
    from .runtime import CaurenCoreRuntime
    from .training import CaurenCoreTrainConfig, train_cauren_core

__all__ = [
    "AgentCandidate",
    "AgentSchema",
    "CaurenCoreRuntime",
    "CaurenDiagnosis",
    "CaurenHybridBackbone",
    "CaurenCoreTrainConfig",
    "CaurenPipeline",
    "CoreOutput",
    "evaluate_quality",
    "FeatureMetadata",
    "HybridBackboneUnavailable",
    "NormalizationDecision",
    "NormalizationRegistryEntry",
    "NormalizationTrace",
    "NormalizedSensorReading",
    "PhysicsEvidence",
    "QualityControlReport",
    "QualityFinding",
    "render_diagnosis_explanation",
    "render_physics_evidence",
    "SensorReading",
    "SensorWindow",
    "train_cauren_core",
    "UncertaintyFactor",
    "UncertaintyReport",
    "estimate_uncertainty",
]


def __getattr__(name: str):
    if name == "CaurenCoreRuntime":
        from .runtime import CaurenCoreRuntime

        return CaurenCoreRuntime
    if name == "CaurenPipeline":
        from .orchestrator import CaurenPipeline

        return CaurenPipeline
    if name in {"render_diagnosis_explanation", "render_physics_evidence"}:
        from .explanations import render_diagnosis_explanation, render_physics_evidence

        return {
            "render_diagnosis_explanation": render_diagnosis_explanation,
            "render_physics_evidence": render_physics_evidence,
        }[name]
    if name in {"CaurenHybridBackbone", "HybridBackboneUnavailable"}:
        from .backbone import CaurenHybridBackbone, HybridBackboneUnavailable

        return {
            "CaurenHybridBackbone": CaurenHybridBackbone,
            "HybridBackboneUnavailable": HybridBackboneUnavailable,
        }[name]
    if name in {"CaurenCoreTrainConfig", "train_cauren_core"}:
        from .training import CaurenCoreTrainConfig, train_cauren_core

        return {
            "CaurenCoreTrainConfig": CaurenCoreTrainConfig,
            "train_cauren_core": train_cauren_core,
        }[name]
    raise AttributeError(f"module 'cauren_core' has no attribute {name!r}")
