from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .backbone_model import create_cauren_backbone_model
from .contracts import SensorWindow


class HybridBackboneUnavailable(RuntimeError):
    pass


class CaurenHybridBackbone:
    """
    Optional MAE + SCNN + RNN-Twin adapter for Cauren Core.

    It consumes the new agent-schema feature matrix and presence mask directly.
    No canonical dimension list is required. The adapter stays opt-in unless a
    Cauren-compatible checkpoint is supplied or untrained execution is explicitly allowed.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        allow_untrained: bool = False,
        mae_config: dict[str, Any] | None = None,
        rnn_config: dict[str, Any] | None = None,
        scnn_config: dict[str, Any] | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path).expanduser() if checkpoint_path else None
        self.device_name = device or os.getenv("CAUREN_CORE_BACKBONE_DEVICE", "cpu")
        self.allow_untrained = bool(allow_untrained)
        self.mae_config = mae_config or {
            "d_model": 64,
            "n_encoder_layers": 2,
            "n_heads": 4,
            "d_ff_encoder": 128,
            "d_ff_decoder": 64,
        }
        self.rnn_config = rnn_config or {"hidden_dim": 128, "num_layers": 1}
        self.scnn_config = scnn_config or {"hidden_channels": 32}
        self._torch = None
        self._model = None
        self._input_dim: int | None = None
        self._layout_key: str | None = None
        self._normalization: dict[str, Any] | None = None
        self._loaded_checkpoint = False
        self._unavailable_reason = ""

    @classmethod
    def from_env(cls) -> "CaurenHybridBackbone | None":
        mode = str(os.getenv("CAUREN_CORE_BACKBONE", "")).strip().lower()
        if mode not in {"civil", "lightweight", "hybrid", "mae_scnn_rnn", "mae_scnn_rnn_twin"}:
            return None
        allow_untrained = os.getenv("CAUREN_CORE_ALLOW_UNTRAINED_BACKBONE", "false").lower() == "true"
        checkpoint = os.getenv("CAUREN_CORE_BACKBONE_CHECKPOINT", "").strip() or None
        if checkpoint is None and not allow_untrained:
            return cls(checkpoint_path=None, allow_untrained=False)
        return cls(checkpoint_path=checkpoint, allow_untrained=allow_untrained)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": "mae_scnn_rnn_twin",
            "available": self._model is not None,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else "",
            "checkpoint_loaded": bool(self._loaded_checkpoint),
            "allow_untrained": bool(self.allow_untrained),
            "input_dim": self._input_dim,
            "layout_key": self._layout_key,
            "device": self.device_name,
            "unavailable_reason": self._unavailable_reason,
        }

    def calibrate(self, window: SensorWindow, *, sampling_hz: float = 1.0, runtime_mode: str | None = None):
        feature_names = tuple(feature.name for feature in window.features)
        if not self._ensure_model(input_dim=len(window.features), feature_names=feature_names):
            raise HybridBackboneUnavailable(self._unavailable_reason or "civil backbone unavailable")
        torch = self._torch
        assert torch is not None
        matrix = torch.tensor(window.matrix, dtype=torch.float32, device=self.device_name).unsqueeze(0)
        mask = torch.tensor(window.presence_mask, dtype=torch.float32, device=self.device_name).unsqueeze(0)
        model_input = self._normalize_tensor(matrix)
        with torch.no_grad():
            output = self._model(
                model_input,
                return_intermediates=True,
                sampling_hz=float(sampling_hz),
                runtime_mode=runtime_mode,
                presence_mask=mask,
            )
        calibrated_tensor = self._denormalize_tensor(output["calibrated"])
        calibrated = calibrated_tensor.detach().cpu().tolist()[0]
        return tuple(tuple(float(value) for value in row) for row in calibrated), output

    @staticmethod
    def layout_key(feature_names: tuple[str, ...]) -> str:
        return "feature_layout:" + "|".join(str(name) for name in feature_names)

    def _normalize_tensor(self, tensor):
        if not self._normalization or self._torch is None:
            return tensor
        mean = self._normalization.get("mean")
        std = self._normalization.get("std")
        if not isinstance(mean, list) or not isinstance(std, list):
            return tensor
        torch = self._torch
        mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, 1, -1)
        std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, 1, -1).clamp_min(1e-6)
        return (tensor - mean_t) / std_t

    def _denormalize_tensor(self, tensor):
        if not self._normalization or self._torch is None:
            return tensor
        mean = self._normalization.get("mean")
        std = self._normalization.get("std")
        if not isinstance(mean, list) or not isinstance(std, list):
            return tensor
        torch = self._torch
        mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, 1, -1)
        std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, 1, -1).clamp_min(1e-6)
        return (tensor * std_t) + mean_t

    def _select_state_dict(
        self,
        checkpoint: Any,
        *,
        input_dim: int,
        feature_names: tuple[str, ...],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
        if not isinstance(checkpoint, dict):
            return checkpoint, None, None
        if checkpoint.get("format") == "cauren_core_backbone_bundle_v1":
            layouts = checkpoint.get("layouts", {})
            if not isinstance(layouts, dict):
                self._unavailable_reason = "invalid_bundle_layouts"
                return None, None, None
            key = self.layout_key(feature_names)
            layout = layouts.get(key)
            if not isinstance(layout, dict):
                for candidate_key, candidate_layout in layouts.items():
                    if (
                        isinstance(candidate_layout, dict)
                        and int(candidate_layout.get("input_dim", -1)) == int(input_dim)
                        and tuple(candidate_layout.get("feature_names", ())) == tuple(feature_names)
                    ):
                        key = str(candidate_key)
                        layout = candidate_layout
                        break
            if not isinstance(layout, dict):
                self._unavailable_reason = f"layout_missing:{key}"
                return None, None, None
            state_dict = layout.get("model_state_dict")
            if not isinstance(state_dict, dict):
                self._unavailable_reason = f"layout_state_missing:{key}"
                return None, None, None
            normalization = layout.get("normalization") if isinstance(layout.get("normalization"), dict) else None
            return state_dict, normalization, key
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        normalization = checkpoint.get("normalization") if isinstance(checkpoint.get("normalization"), dict) else None
        return state_dict, normalization, None

    def _ensure_model(self, *, input_dim: int, feature_names: tuple[str, ...]) -> bool:
        requested_key = self.layout_key(feature_names)
        if self._model is not None and self._input_dim == input_dim and self._layout_key == requested_key:
            return True
        if self.checkpoint_path is None and not self.allow_untrained:
            self._unavailable_reason = "checkpoint_required"
            return False
        try:
            import torch
        except Exception as exc:
            self._unavailable_reason = f"torch_unavailable:{exc.__class__.__name__}"
            return False
        try:
            model = create_cauren_backbone_model(
                torch,
                mae_config=self.mae_config,
                rnn_config=self.rnn_config,
                scnn_config=self.scnn_config,
                input_dim=int(input_dim),
            )
            model.to(torch.device(self.device_name))
            selected_normalization = None
            selected_layout_key = requested_key
            if self.checkpoint_path is not None:
                if not self.checkpoint_path.exists():
                    self._unavailable_reason = f"checkpoint_missing:{self.checkpoint_path}"
                    return False
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device_name)
                state_dict, selected_normalization, bundle_layout_key = self._select_state_dict(
                    checkpoint,
                    input_dim=input_dim,
                    feature_names=feature_names,
                )
                if state_dict is None:
                    return False
                if bundle_layout_key:
                    selected_layout_key = bundle_layout_key
                model.load_state_dict(state_dict, strict=False)
                self._loaded_checkpoint = True
            model.eval()
            self._torch = torch
            self._model = model
            self._input_dim = int(input_dim)
            self._layout_key = selected_layout_key
            self._normalization = selected_normalization
            self._unavailable_reason = ""
            return True
        except Exception as exc:
            self._model = None
            self._input_dim = None
            self._layout_key = None
            self._normalization = None
            self._unavailable_reason = f"{exc.__class__.__name__}:{exc}"
            return False
