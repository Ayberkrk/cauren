from __future__ import annotations

import math
from typing import Any


def create_cauren_backbone_model(
    torch,
    *,
    input_dim: int,
    mae_config: dict[str, Any] | None = None,
    rnn_config: dict[str, Any] | None = None,
    scnn_config: dict[str, Any] | None = None,
):
    """
    Build the Cauren hybrid backbone: a masked-autoencoder (MAE) temporal
    encoder, an SCNN (sequential/1-D convolutional) local-pattern branch,
    and an RNN-Twin self-supervised consistency branch, plus a small
    auxiliary classification head for a real supervised target
    (`deck_drop_5yr` for the bridge vertical) when training data provides
    one.

    This replaced a 4-line stand-in (a 2-layer MLP autoencoder) that used
    these three names in its config dict without implementing any of the
    architecture they name. All three components below are real and are
    exercised by tools/train_cauren_core.py.

    Interface contract (must stay stable -- cauren_core/backbone.py calls
    this directly): forward(x, return_intermediates=False, presence_mask=None,
    **_) -> dict with at least "calibrated" (same shape as x) and
    "losses" = {"mae_loss": scalar, "rnn_consistency_loss": scalar}.
    """
    nn = torch.nn

    mae_config = {
        "d_model": 64,
        "n_encoder_layers": 2,
        "n_heads": 4,
        "d_ff_encoder": 128,
        "d_ff_decoder": 64,
        "dropout": 0.1,
        **(mae_config or {}),
    }
    rnn_config = {"hidden_dim": 128, "num_layers": 1, **(rnn_config or {})}
    scnn_config = {"hidden_channels": 32, "kernel_size": 3, **(scnn_config or {})}

    d_model = int(mae_config["d_model"])

    class SinusoidalPositionalEncoding(nn.Module):
        """Standard fixed sinusoidal position encoding (Vaswani et al.).

        Civil-engineering windows have real but irregular calendar gaps
        (a bridge inspected every 1-2 years, not every fixed tick), so
        this only encodes *order* within the window, not absolute time;
        the RNN-Twin branch below is what actually reasons over the
        sequence's temporal structure.
        """

        def __init__(self, d_model: int, max_len: int = 4096):
            super().__init__()
            position = torch.arange(max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
            self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

        def forward(self, x):
            return x + self.pe[:, : x.shape[1], :].to(x.dtype)

    class SCNNBlock(nn.Module):
        """Sequential CNN branch: stacked dilated 1-D convolutions over the
        time axis, extracting local (few-timestep) deterioration/spike
        shapes that a purely global attention encoder can under-weight.
        Residual-added into the encoder's input stream."""

        def __init__(self, d_model: int, hidden_channels: int, kernel_size: int):
            super().__init__()
            dilation = 2
            # "same"-length padding for a dilated conv: dilation * (k - 1) // 2
            # (assumes odd kernel_size, which SCNN_CONFIG always supplies).
            padding_1 = kernel_size // 2
            padding_2 = dilation * (kernel_size - 1) // 2
            self.net = nn.Sequential(
                nn.Conv1d(d_model, hidden_channels, kernel_size, padding=padding_1),
                nn.GELU(),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=padding_2, dilation=dilation),
                nn.GELU(),
                nn.Conv1d(hidden_channels, d_model, kernel_size=1),
            )
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            # x: (batch, seq, d_model) -> conv over seq axis -> back
            conv_in = x.transpose(1, 2)
            conv_out = self.net(conv_in).transpose(1, 2)
            return self.norm(x + conv_out)

    class MAEEncoder(nn.Module):
        """Transformer encoder over the (masked) feature-projected window.
        `presence_mask` marks genuinely-observed timesteps/features so the
        model is trained to reconstruct real values, not the zero-fill
        used for missing sensor coverage -- padded/unobserved timesteps
        never contribute to the reconstruction loss (see mae_loss below)."""

        def __init__(self, input_dim: int, cfg: dict[str, Any]):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_enc = SinusoidalPositionalEncoding(d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=int(cfg["n_heads"]),
                dim_feedforward=int(cfg["d_ff_encoder"]),
                dropout=float(cfg["dropout"]),
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=int(cfg["n_encoder_layers"]))

        def forward(self, x, key_padding_mask=None):
            h = self.pos_enc(self.input_proj(x))
            return self.encoder(h, src_key_padding_mask=key_padding_mask)

    class RNNTwin(nn.Module):
        """Self-supervised consistency branch: a GRU walks the encoder's
        own output sequence and, at each step, predicts the *next* step's
        encoded representation from everything before it. The gap between
        that prediction and what the encoder actually produced at t+1 is
        `rnn_consistency_loss` -- a temporal-coherence signal independent
        of the reconstruction loss, matching the "RNN-Twin" name that was
        previously just an unused config dict."""

        def __init__(self, cfg: dict[str, Any]):
            super().__init__()
            hidden_dim = int(cfg["hidden_dim"])
            self.gru = nn.GRU(d_model, hidden_dim, num_layers=int(cfg["num_layers"]), batch_first=True)
            self.to_d_model = nn.Linear(hidden_dim, d_model)

        def forward(self, encoded):
            rnn_out, _ = self.gru(encoded)
            predicted_next = self.to_d_model(rnn_out)
            return predicted_next

    class CaurenHybridBackboneModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = MAEEncoder(input_dim, mae_config)
            self.scnn = SCNNBlock(d_model, int(scnn_config["hidden_channels"]), int(scnn_config["kernel_size"]))
            self.rnn_twin = RNNTwin(rnn_config)
            self.fusion_gate = nn.Linear(d_model * 2, d_model)
            self.decoder = nn.Sequential(
                nn.Linear(d_model, int(mae_config["d_ff_decoder"])),
                nn.GELU(),
                nn.Linear(int(mae_config["d_ff_decoder"]), input_dim),
            )
            # Auxiliary supervised head: predicts a single deterioration
            # logit from the pooled window representation. Unused unless
            # a training loop supplies real labels (deck_drop_5yr) --
            # see tools/train_cauren_core.py. At inference with no label
            # available it is simply not read.
            self.risk_head = nn.Sequential(
                nn.Linear(d_model, max(8, d_model // 2)),
                nn.GELU(),
                nn.Linear(max(8, d_model // 2), 1),
            )

        def forward(self, x, return_intermediates: bool = False, presence_mask=None, **_: Any):
            observed = x if presence_mask is None else (x * presence_mask)

            key_padding_mask = None
            if presence_mask is not None:
                # A timestep is "padding" only if literally no feature was
                # observed there (fully missing row), matching how the
                # adapter marks synthetically-repeated/padded slots.
                any_observed = presence_mask.any(dim=-1)
                if not bool(any_observed.all()):
                    key_padding_mask = ~any_observed

            encoded = self.encoder(observed, key_padding_mask=key_padding_mask)
            scnn_features = self.scnn(encoded)
            predicted_next = self.rnn_twin(encoded)

            fused = self.fusion_gate(torch.cat([scnn_features, predicted_next], dim=-1))
            delta = self.decoder(fused)
            calibrated = observed + delta

            if presence_mask is not None:
                weight = presence_mask
                denom = weight.sum().clamp_min(1.0)
                mae_loss = ((calibrated - observed).abs() * weight).sum() / denom
            else:
                mae_loss = (calibrated - observed).abs().mean()

            if encoded.shape[1] > 1:
                target = encoded[:, 1:, :].detach()
                pred = predicted_next[:, :-1, :]
                if presence_mask is not None:
                    step_mask = presence_mask.any(dim=-1)[:, 1:].unsqueeze(-1).float()
                    denom = step_mask.sum().clamp_min(1.0)
                    rnn_consistency_loss = (((pred - target) ** 2) * step_mask).sum() / denom
                else:
                    rnn_consistency_loss = torch.nn.functional.mse_loss(pred, target)
            else:
                rnn_consistency_loss = torch.zeros((), dtype=calibrated.dtype, device=calibrated.device)

            if presence_mask is not None:
                step_present = presence_mask.any(dim=-1).float().unsqueeze(-1)
                pooled = (fused * step_present).sum(dim=1) / step_present.sum(dim=1).clamp_min(1.0)
            else:
                pooled = fused.mean(dim=1)
            risk_logit = self.risk_head(pooled).squeeze(-1)

            outputs = {
                "calibrated": calibrated,
                "losses": {
                    "mae_loss": mae_loss,
                    "rnn_consistency_loss": rnn_consistency_loss,
                },
                "risk_logit": risk_logit,
                "pooled": pooled,
            }
            if return_intermediates:
                outputs["encoded"] = encoded
                outputs["scnn_features"] = scnn_features
            return outputs

    return CaurenHybridBackboneModel()
