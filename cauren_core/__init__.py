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

__all__ = [
    "AgentCandidate",
    "AgentSchema",
    "CaurenCoreRuntime",
    "CaurenDiagnosis",
    "CaurenHybridBackbone",
    "CaurenCoreTrainConfig",
    "CaurenPipeline",
    "CoreOutput",
    "FeatureMetadata",
    "HybridBackboneUnavailable",
    "NormalizationDecision",
    "NormalizationRegistryEntry",
    "NormalizationTrace",
    "NormalizedSensorReading",
    "PhysicsEvidence",
    "render_diagnosis_explanation",
    "render_physics_evidence",
    "SensorReading",
    "SensorWindow",
    "train_cauren_core",
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
