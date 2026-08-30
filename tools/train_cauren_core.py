from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cauren_core.training import CaurenCoreTrainConfig, train_cauren_core


LOCAL_PROFILES = {
    "civil_cpu_small": {
        "dataset_dir": "data/cauren_civil",
        "output_path": "cauren_core/checkpoints/cauren_civil_backbone_bundle.pt",
        "agents": ["cauren-civil"],
        "training_roles": ["mae_reconstruction"],
        "splits": ["train"],
        "validation_splits": ["validation"],
        "epochs": 8,
        "batch_size": 2,
        "learning_rate": 4e-4,
        "max_windows_per_agent": 256,
        "device": "cpu",
        "seed": 42,
        "early_stopping_patience": 4,
        "min_epochs": 4,
    }
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the civil-only Cauren Core backbone bundle.")
    parser.add_argument("--profile", choices=sorted(LOCAL_PROFILES.keys()), default=None)
    parser.add_argument("--dataset-dir", default="data/cauren_civil")
    parser.add_argument("--output-path", default="cauren_core/checkpoints/cauren_civil_backbone_bundle.pt")
    parser.add_argument("--agents", nargs="*", default=["cauren-civil"])
    parser.add_argument("--training-roles", nargs="*", default=["mae_reconstruction"])
    parser.add_argument("--splits", nargs="*", default=["train"])
    parser.add_argument("--validation-splits", nargs="*", default=["validation"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-windows-per-agent", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--min-epochs", type=int, default=4)
    explicit_flags = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
    args = parser.parse_args()

    defaults = {field: parser.get_default(field) for field in [
        "agents", "dataset_dir", "output_path", "training_roles", "splits", "validation_splits",
        "epochs", "batch_size", "learning_rate", "max_windows_per_agent", "device", "seed",
        "early_stopping_patience", "min_epochs",
    ]}
    if args.profile:
        for field, profile_value in LOCAL_PROFILES[args.profile].items():
            if f"--{field.replace('_', '-')}" not in explicit_flags and getattr(args, field) == defaults[field]:
                setattr(args, field, profile_value)

    summary = train_cauren_core(
        CaurenCoreTrainConfig(
            dataset_dir=Path(args.dataset_dir),
            output_path=Path(args.output_path),
            agents=tuple(args.agents),
            training_roles=tuple(args.training_roles),
            splits=tuple(args.splits),
            validation_splits=tuple(args.validation_splits),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_windows_per_agent=args.max_windows_per_agent,
            device=args.device,
            seed=args.seed,
            dry_run=bool(args.dry_run),
            early_stopping_patience=args.early_stopping_patience,
            min_epochs=args.min_epochs,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
