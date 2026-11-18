"""The surrogate (docs/technote Figure 2): set encoder -> flag heads (slice found; given a slice: matched, nu_e
hypothesis, flash found), PFParticle multiplicity head, 9-d event flow, and (pfp=True, the second stage) the
PFParticle set flow with cross-attention to the primary tokens."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import SetEncoder
from .flow import FlowMatcher
from .prongflow import ProngFlow
from ..data.pfp_tf import PFP_DIM
from ..data.transform import VTX_COLS, FLASH_COLS

N_FLAGS = 4            # slice, matched | slice, nu_e | slice, flash | slice
N_COND_FLAGS = 3       # the three conditional ones enter the flow conditioning
N_PFP_CLASSES = 10     # N_PFP = 1..10
EVENT_DIM = 9


class Surrogate(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=4, flow_hidden=512, flow_layers=4, n_classes=15, pfp=False, pfp_layers=3):
        super().__init__()
        self.pfp = pfp
        self.enc = SetEncoder(d_model, n_heads, n_layers, n_classes=n_classes)
        self.flags = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, N_FLAGS))
        self.flag_emb = nn.Linear(N_COND_FLAGS, 32)
        self.card = nn.Sequential(nn.Linear(d_model + 32, d_model), nn.SiLU(), nn.Linear(d_model, N_PFP_CLASSES))
        self.n_emb = nn.Embedding(N_PFP_CLASSES, 32)
        self.flow = FlowMatcher(EVENT_DIM, d_model + 64, flow_hidden, flow_layers)
        if pfp:   # conditioned on the flow conditioning and the event vector; cross-attends to [global token, primaries]
            self.pfpflow = ProngFlow(PFP_DIM, d_model, d_model + 64 + EVENT_DIM, pfp_layers)

    @staticmethod
    def tokens(z, h, mask):
        """Cross-attention keys: the event embedding first, so that an empty primary set still has one key."""
        return torch.cat([z[:, None], h], 1), torch.cat([torch.ones_like(mask[:, :1]), mask], 1)

    def encode(self, b, return_tokens=False):
        z, h = self.enc(b["cls"], b["feat"], b["mask"], b["ctx"])
        return (z, h) if return_tokens else z

    @staticmethod
    def dim_mask(flags):
        """[n, 9]: the vertex residual exists only for a matched slice, the flash columns only with a flash."""
        m = torch.ones(len(flags), EVENT_DIM, device=flags.device)
        m[:, VTX_COLS] = flags[:, 0:1]; m[:, FLASH_COLS] = flags[:, 2:3]
        return m

    def flow_cond(self, z, cflags, npfp):
        return torch.cat([z, self.flag_emb(cflags), self.n_emb(npfp.clamp(0, N_PFP_CLASSES - 1))], -1)

    def losses(self, b):
        z, h = self.encode(b, return_tokens=True)
        f = b["flags"]; has = f[:, 0] > 0
        logits = self.flags(z)
        zero = logits.sum() * 0
        out = {"slice": F.binary_cross_entropy_with_logits(logits[:, 0], f[:, 0])}
        if not has.any():
            out.update(matched=zero, nue=zero, flash=zero, card=zero, flow=zero)
            if self.pfp: out["pfp"] = zero
            return out
        for j, k in enumerate(("matched", "nue", "flash"), start=1):
            out[k] = F.binary_cross_entropy_with_logits(logits[has, j], f[has, j])
        cf = f[has, 1:4]; zc = z[has]
        out["card"] = F.cross_entropy(self.card(torch.cat([zc, self.flag_emb(cf)], -1)), b["npfp"][has])
        cond = self.flow_cond(zc, cf, b["npfp"][has])
        out["flow"] = self.flow.loss(b["x"][has], cond, b["dmask"][has] * self.dim_mask(cf)).mean()
        if self.pfp and "pfp" in b:
            keys, kmask = self.tokens(zc, h[has], b["mask"][has])
            out["pfp"] = self.pfpflow.loss(b["pfp"][has], torch.cat([cond, b["x"][has]], -1), keys, kmask, b["pmask"][has])
        return out

    @torch.no_grad()
    def predict_probs(self, b):
        z = self.encode(b)
        return torch.sigmoid(self.flags(z)), z

    @torch.no_grad()
    def sample(self, b, n_steps=64, teacher_flags=None, teacher_npfp=None, pfp=True):
        """Flags, N_PFP, the event vector and (a pfp model) the PFParticle rows, all in model space, for every event;
        `slice` is the mask for the caller, the rest describes the event *if* it has a slice (the heads are conditional
        probabilities). pfp=False skips the set flow (most of the sampling cost)."""
        z, h = self.encode(b, return_tokens=True); p = torch.sigmoid(self.flags(z))
        u = torch.rand_like(p)
        has = u[:, 0] < p[:, 0]
        cflags = (u[:, 1:4] < p[:, 1:4]).float() if teacher_flags is None else teacher_flags
        pn = F.softmax(self.card(torch.cat([z, self.flag_emb(cflags)], -1)), -1)
        npfp = torch.multinomial(pn, 1)[:, 0] if teacher_npfp is None else teacher_npfp
        cond = self.flow_cond(z, cflags, npfp)
        dm = self.dim_mask(cflags)
        x = self.flow.sample(cond, n_steps, dim_mask=dm)
        out = {"slice": has, "cflags": cflags, "npfp": npfp, "x": x, "dmask": dm, "p": p, "pn": pn}
        if self.pfp and pfp:
            N = int(npfp.max().item()) + 1
            pmask = torch.arange(N, device=z.device)[None, :] <= npfp[:, None]
            keys, kmask = self.tokens(z, h, b["mask"])
            out["pfp"] = self.pfpflow.sample(torch.cat([cond, x], -1), keys, kmask, pmask, n_steps); out["pmask"] = pmask
        return out
