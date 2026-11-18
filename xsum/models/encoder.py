"""Permutation-invariant set encoder over the primary tokens with a global token that carries the true vertex.
Ported from XSuM-MINERvA; the token features are MicroBooNE's (kinetic energy in GeV and a unit direction)."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..constants import N_CLASSES

KE_FLOOR = 1e-3  # GeV, below the 10 MeV cut; keeps log(T) finite on padding


def token_features(feat):
    """feat [..., 4] = (T [GeV], dx, dy, dz) -> (log T, asinh(T / 0.3 GeV), dx, dy, dz)."""
    T = feat[..., :1].clamp_min(KE_FLOOR)
    return torch.cat([torch.log(T), torch.asinh(T / 0.3), feat[..., 1:4]], -1)


class SetEncoder(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=4, d_ff=256, dropout=0.0, n_ctx=3, n_classes=N_CLASSES):
        super().__init__()
        self.cls_emb = nn.Embedding(n_classes, d_model)
        self.feat_mlp = nn.Sequential(nn.Linear(5, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.ctx_mlp = nn.Sequential(nn.Linear(n_ctx, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.global_tok = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        # true vertex (cm): centre and half-width of the cryostat volume the sample is generated in
        self.ctx_loc = nn.Parameter(torch.tensor([128.0, 0.0, 520.0]), requires_grad=False)
        self.ctx_scale = nn.Parameter(torch.tensor([150.0, 120.0, 330.0]), requires_grad=False)

    def forward(self, cls, feat, mask, ctx):
        """Returns the event embedding z [B,d] and the token states h [B,n,d]. An empty set is the global token alone."""
        tok = self.cls_emb(cls) + self.feat_mlp(token_features(feat))
        g = self.global_tok + self.ctx_mlp((ctx - self.ctx_loc) / self.ctx_scale)[:, None]
        x = torch.cat([g, tok], 1)
        pad = torch.cat([torch.zeros_like(mask[:, :1]), ~mask], 1)
        h = self.norm(self.tf(x, src_key_padding_mask=pad))
        return h[:, 0], h[:, 1:]
