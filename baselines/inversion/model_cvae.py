"""Conditional VAE for descriptor -> parameter inversion (Task 5).

Encoder: q(z | params_encoded, descriptors) -> (mu, logvar)
Decoder: p(params | z, descriptors) with MIXED output heads matching the
         ParamCodec block layout:
           - numeric blocks (float, numeric_enum): single Gaussian/MSE head
           - bool blocks: single BCE head
           - one-hot enum blocks: softmax head with cross-entropy

Loss = MSE_numeric + sum CE_onehot + sum BCE_bool + beta * KL

Why this is the headline: the descriptor->param mapping is one-to-many. A
deterministic regressor can only emit the mean of plausible patches, which is
often a "boring" patch. A CVAE generates diverse, plugin-valid configurations
conditioned on the target descriptor vector, and round-trip re-rendering then
measures whether those configurations actually reproduce the target timbre.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common.encoding import ParamCodec, CodecBlock


@dataclass
class BlockLayout:
    """Concrete slicing of the 754-dim codec vector into typed heads."""
    numeric_slots: list[int]          # column indices for float + numeric_enum
    bool_slots: list[int]             # column indices for bool
    onehot_ranges: list[tuple[int, int]]  # (start, end) per one-hot enum block
    encoded_dim: int

    @classmethod
    def from_codec(cls, codec: ParamCodec) -> "BlockLayout":
        numeric_slots: list[int] = []
        bool_slots: list[int] = []
        onehot_ranges: list[tuple[int, int]] = []
        for b in codec.blocks:
            if b.kind in ("float", "numeric_enum"):
                numeric_slots.append(b.start)
            elif b.kind == "bool":
                bool_slots.append(b.start)
            elif b.kind == "onehot_enum":
                onehot_ranges.append((b.start, b.end))
        return cls(numeric_slots, bool_slots, onehot_ranges, codec.encoded_dim)


def _mlp(dims: Sequence[int], dropout: float = 0.1, batchnorm: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            if batchnorm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class CVAE(nn.Module):
    def __init__(self,
                 layout: BlockLayout,
                 desc_dim: int = 7,
                 z_dim: int = 32,
                 hidden: Sequence[int] = (512, 256),
                 dropout: float = 0.1):
        super().__init__()
        self.layout = layout
        self.x_dim = layout.encoded_dim
        self.desc_dim = desc_dim
        self.z_dim = z_dim

        self.encoder = _mlp((self.x_dim + desc_dim, *hidden), dropout=dropout)
        self.fc_mu = nn.Linear(hidden[-1], z_dim)
        self.fc_lv = nn.Linear(hidden[-1], z_dim)

        # Decoder shares trunk and emits raw logits sized to the codec dim;
        # we slice/loss them according to layout in `compute_loss`.
        dec_hidden = tuple(reversed(hidden))
        self.decoder = _mlp((z_dim + desc_dim, *dec_hidden), dropout=dropout)
        self.head = nn.Linear(dec_hidden[-1], self.x_dim)

    def encode(self, x: torch.Tensor, d: torch.Tensor):
        h = self.encoder(torch.cat([x, d], dim=-1))
        return self.fc_mu(h), self.fc_lv(h)

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self.head(self.decoder(torch.cat([z, d], dim=-1)))

    def forward(self, x: torch.Tensor, d: torch.Tensor):
        mu, lv = self.encode(x, d)
        z = self.reparameterise(mu, lv)
        recon = self.decode(z, d)
        return recon, mu, lv

    # ----- loss / decoding helpers ----------------------------------------

    def compute_loss(self, recon: torch.Tensor, target: torch.Tensor,
                     mu: torch.Tensor, logvar: torch.Tensor,
                     beta: float = 1.0) -> dict:
        L = self.layout
        # Numeric block: simple MSE.
        if L.numeric_slots:
            idx = torch.tensor(L.numeric_slots, device=recon.device, dtype=torch.long)
            num_pred = recon.index_select(1, idx)
            num_tgt = target.index_select(1, idx)
            num_loss = F.mse_loss(num_pred, num_tgt, reduction="mean")
        else:
            num_loss = torch.zeros((), device=recon.device)

        # Bool block: BCE-with-logits (target column is {0,1}).
        if L.bool_slots:
            idx = torch.tensor(L.bool_slots, device=recon.device, dtype=torch.long)
            b_pred = recon.index_select(1, idx)
            b_tgt = target.index_select(1, idx)
            bool_loss = F.binary_cross_entropy_with_logits(b_pred, b_tgt, reduction="mean")
        else:
            bool_loss = torch.zeros((), device=recon.device)

        # One-hot blocks: per-block cross-entropy on the row's argmax index.
        ce_loss = torch.zeros((), device=recon.device)
        if L.onehot_ranges:
            n = recon.shape[0]
            ce_terms = []
            for (s, e) in L.onehot_ranges:
                logits = recon[:, s:e]
                tgt_idx = target[:, s:e].argmax(dim=1)
                ce_terms.append(F.cross_entropy(logits, tgt_idx, reduction="mean"))
            ce_loss = torch.stack(ce_terms).mean()

        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        recon_loss = num_loss + bool_loss + ce_loss
        return {
            "loss": recon_loss + beta * kl,
            "num_loss": num_loss.detach(),
            "bool_loss": bool_loss.detach(),
            "ce_loss": ce_loss.detach(),
            "kl": kl.detach(),
            "recon_loss": recon_loss.detach(),
        }

    def to_codec_vector(self, recon: torch.Tensor) -> np.ndarray:
        """Convert one decoded batch row into the codec's expected encoded form.

        - Numeric & bool slots: passed through as-is (the codec.decode does
          inverse z-score and 0.5 threshold).
        - One-hot slots: argmax -> 1-hot, so codec.decode picks the right
          enum option.

        For best-of-N (where the codec needs the actual numeric values), the
        sigmoid bools are NOT thresholded here; codec.decode does the >=0.5
        threshold.
        """
        x = recon.detach().cpu().numpy().astype(np.float32)
        if x.ndim == 1:
            x = x[None, :]
        L = self.layout
        for (s, e) in L.onehot_ranges:
            block = x[:, s:e]
            arg = np.argmax(block, axis=1)
            block[:] = 0.0
            block[np.arange(block.shape[0]), arg] = 1.0
            x[:, s:e] = block
        if L.bool_slots:
            for j in L.bool_slots:
                x[:, j] = 1.0 / (1.0 + np.exp(-x[:, j]))  # sigmoid; codec.decode thresholds at 0.5
        return x
