"""Second stage: the PFParticle set flow on top of the event-stage model (warm start), every parameter fine-tuned."""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from ..data.dataset import EventDataset, PFP_FIELDS
from ..data.pfp_tf import PfpTransform, PFP_MODEL_NAMES, PFP_DIM, CONT_COLS
from ..data.torch_data import BucketBatchSampler, TorchEvents, collate
from ..data.transform import EventTransform
from ..eval.metrics import sample_vs_real_1d, confusion
from ..models.surrogate import Surrogate, N_PFP_CLASSES
from .event_stage import to_dev


def make_loaders(ds, tf, ptf, bs, seed, workers=2, bucket=True):
    idx = {k: np.where(ds.split == v)[0] for k, v in (("train", 0), ("val", 1), ("test", 2))}
    tds = {k: TorchEvents(ds, v, tf, ptf, seed) for k, v in idx.items()}
    ld = {k: DataLoader(tds[k], batch_size=bs, shuffle=False, collate_fn=collate, num_workers=workers) for k in tds if k != "train"}
    if bucket:
        ld["train"] = DataLoader(tds["train"], batch_sampler=BucketBatchSampler(ds.mask[idx["train"]].sum(1), bs, seed), collate_fn=collate, num_workers=workers)
    else:
        ld["train"] = DataLoader(tds["train"], batch_size=bs, shuffle=True, collate_fn=collate, num_workers=workers, drop_last=True)
    return idx, tds, ld


def train(stems, out_dir, init_from, epochs=12, bs=512, lr=2e-4, seed=0, device="cuda", pfp_layers=3, log_every=100,
          init_seed=None, bucket=True, workers=2):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    init_seed = seed if init_seed is None else init_seed; torch.manual_seed(init_seed); np.random.seed(init_seed); t0 = time.time()
    ck = torch.load(init_from, map_location=device, weights_only=False); c = ck["config"]
    ds = EventDataset(stems, active_only=c["active_only"])
    assert ds.has_hits, "the PFParticle stage needs <stem>.hits.parquet for every stem (scripts/curate/slim_hits.py)"
    tf = EventTransform.from_state(ck["transform"])
    tr_rows = ds.pfp_mask & ((ds.split == 0) & ds.has_slice)[:, None]
    ptf = PfpTransform().fit(ds.pfp[tr_rows], seed)
    idx, tds, ld = make_loaders(ds, tf, ptf, bs, init_seed, workers, bucket)
    print(f"data: {len(ds)} events, {int(ds.pfp_mask.sum())} PFParticles, {time.time()-t0:.0f} s", flush=True)
    model = Surrogate(c["d_model"], 4, c["n_layers"], c["flow_hidden"], c["flow_layers"], n_classes=c["n_classes"], pfp=True, pfp_layers=pfp_layers).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print(f"warm start from {init_from}: {len(missing)} new tensors (the set flow), {len(unexpected)} unexpected", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, epochs * len(ld["train"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    hist, best, step = [], np.inf, 0
    cfg = dict(c, pfp=True, pfp_layers=pfp_layers)
    for ep in range(epochs):
        model.train(); agg = {}
        for b in ld["train"]:
            b = to_dev(b, device); L = model.losses(b); loss = sum(L.values())
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
            for k, v in L.items(): agg[k] = agg.get(k, 0) + v.item()
            if step % log_every == 0: print(f"  step {step} " + " ".join(f"{k} {v.item():.4f}" for k, v in L.items()), flush=True)
        model.eval(); va = {}
        with torch.no_grad():
            for b in ld["val"]:
                L = model.losses(to_dev(b, device))
                for k, v in L.items(): va[k] = va.get(k, 0) + v.item() / len(ld["val"])
        tot = sum(va.values()); hist.append({"epoch": ep, "train": {k: v / max(1, len(ld["train"])) for k, v in agg.items()}, "val": va, "time": time.time() - t0})
        print(f"epoch {ep}: val " + " ".join(f"{k} {v:.4f}" for k, v in va.items()) + f"  total {tot:.4f}  [{time.time()-t0:.0f} s]", flush=True)
        if np.isfinite(tot) and tot < best:
            best = tot
            torch.save({"model": model.state_dict(), "transform": tf.state(), "pfp_transform": ptf.state(), "config": cfg}, out / "model.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    return ds, tf, ptf, idx, ld, out


def load_model(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False); c = ck["config"]
    m = Surrogate(c["d_model"], 4, c["n_layers"], c["flow_hidden"], c["flow_layers"], n_classes=c["n_classes"], pfp=True, pfp_layers=c.get("pfp_layers", 3)).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m, EventTransform.from_state(ck["transform"]), PfpTransform.from_state(ck["pfp_transform"]), c


def sample_loader(model, ld, n_steps=64, device="cuda", seed=0):
    torch.manual_seed(seed); acc = {}; N = 0
    with torch.no_grad():
        for b in ld:
            s = model.sample(to_dev(b, device), n_steps)
            for k, v in s.items(): acc.setdefault(k, []).append(v.cpu())
            N = max(N, s["pfp"].shape[1])
    out = {k: torch.cat(v).numpy() for k, v in acc.items() if k not in ("pfp", "pmask")}
    out["pfp"] = torch.cat([torch.nn.functional.pad(p, (0, 0, 0, N - p.shape[1])) for p in acc["pfp"]]).numpy()
    out["pmask"] = torch.cat([torch.nn.functional.pad(p, (0, N - p.shape[1])) for p in acc["pmask"]]).numpy().astype(bool)
    return out


@torch.no_grad()
def evaluate(model, tf, ptf, ds, idx_test, ld_test, out_dir, device="cuda", n_steps=64, seed=0):
    """PFParticle-level closure on the test split: per-column marginals real vs sampled (flat over PFParticles),
    per-event sums (hits, charge) and a classifier on the flat PFParticle rows."""
    out = pathlib.Path(out_dir); (out / "figures").mkdir(exist_ok=True, parents=True)
    S = sample_loader(model, ld_test, n_steps, device, seed)
    has = ds.has_slice[idx_test]; sl = S["slice"]
    rng = np.random.default_rng(seed)
    pr = ds.pfp[idx_test][ds.pfp_mask[idx_test]]                       # real records, flat
    xr = ptf.forward(pr, rng); xf = S["pfp"][sl][S["pmask"][sl]]; pf = ptf.inverse(xf)
    M = {"n_test": int(len(idx_test)), "n_pfp_real": int(len(pr)), "n_pfp_sampled": int(len(pf))}
    M["pfp_model_space"] = {n: sample_vs_real_1d(xr[:, j], xf[:, j]) for j, n in enumerate(PFP_MODEL_NAMES)}
    M["pfp_raw"] = {n: sample_vs_real_1d(pr[:, j], pf[:, j]) for j, n in enumerate(PFP_FIELDS)}
    M["pfp_flags"] = {f"has_p{p}": {"real": float((pr[:, 4 + 4 * p] > 0).mean()), "sampled": float((pf[:, 4 + 4 * p] > 0).mean())} for p in range(3)}
    M["pfp_flags"]["score_is1"] = {"real": float((pr[:, 0] >= 1 - 1e-6).mean()), "sampled": float((pf[:, 0] >= 1 - 1e-6).mean())}
    M["pfp_flags"]["score_is0"] = {"real": float((pr[:, 0] <= 1e-6).mean()), "sampled": float((pf[:, 0] <= 1e-6).mean())}
    # per-event sums over the PFParticles: total hits and charge on the collection plane, and the leading PFParticle's charge
    def per_event(P, mask):
        n_ev = len(mask); seg = np.repeat(np.arange(n_ev), mask.sum(1))
        tot_n = np.zeros(n_ev); tot_q = np.zeros(n_ev); lead = np.zeros(n_ev)
        np.add.at(tot_n, seg, P[:, 12]); np.add.at(tot_q, seg, P[:, 13]); np.maximum.at(lead, seg, P[:, 13])
        return np.stack([tot_n, tot_q, lead], 1)
    er = per_event(pr, ds.pfp_mask[idx_test][has]); ef = per_event(pf, S["pmask"][sl])
    M["event_sums_p2"] = {n: sample_vs_real_1d(np.log1p(er[:, j]), np.log1p(ef[:, j])) for j, n in enumerate(["log_nhits", "log_charge", "log_lead_charge"])}
    # classifier on flat PFParticle rows, split by row: on the decoded records as an analysis sees them (log of the
    # counts and charges, zeros where a plane has no hits) and on the continuous model-space columns
    def _dec(P):
        # one dtype for both sides: the integer columns' logs differ in the low bits between float32 and float64,
        # which a tree classifier separates perfectly
        D = P.astype(np.float32).copy(); D[:, 4:] = np.log1p(np.clip(D[:, 4:], 0, None)); return D
    Dr, Df = _dec(pr), _dec(pf)
    n = min(len(xr), len(xf)); ir, if_ = rng.permutation(len(xr))[:n], rng.permutation(len(xf))[:n]
    yc = np.concatenate([np.ones(n), np.zeros(n)]); half = n // 2
    tr = np.r_[np.arange(half), n + np.arange(half)]; ts = np.r_[np.arange(half, n), n + np.arange(half, n)]
    def _auc(Xc, cols):
        c = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, early_stopping=True, random_state=seed)
        c.fit(Xc[tr][:, cols], yc[tr]); return float(roc_auc_score(yc[ts], c.predict_proba(Xc[ts][:, cols])[:, 1]))
    Xd = np.concatenate([Dr[ir], Df[if_]]); Xm = np.concatenate([xr[ir], xf[if_]])
    M["classifier_auc_pfp"] = _auc(Xd, list(range(Dr.shape[1])))
    M["classifier_auc_pfp_model_continuous"] = _auc(Xm, CONT_COLS)
    M["classifier_auc_by_variable"] = {k: _auc(Xd, [j]) for j, k in enumerate(PFP_FIELDS)}
    M["corr_real"] = np.corrcoef(xr.T).round(3).tolist(); M["corr_fake"] = np.corrcoef(xf.T).round(3).tolist()
    (out / "metrics_pfp.json").write_text(json.dumps(M, indent=1, default=float))
    _figures(M, xr, xf, out / "figures")
    return M


def _figures(M, xr, xf, fdir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(5, 5, figsize=(17, 13))
    for j, (ax, nm) in enumerate(zip(axs.flat, PFP_MODEL_NAMES)):
        bins = np.linspace(-6, 6, 70)
        ax.hist(xr[:, j], bins, histtype="step", color="k", label="open sample", density=True)
        ax.hist(xf[:, j], bins, histtype="step", color="C1", label="surrogate", density=True)
        ax.set_title(f"{nm}  W1 {M['pfp_model_space'][nm]['w1']:.3f}", fontsize=9); ax.set_yscale("log")
    axs.flat[0].legend(frameon=False, fontsize=8)
    for ax in axs.flat[len(PFP_MODEL_NAMES):]: ax.axis("off")
    fig.tight_layout(); fig.savefig(fdir / "pfp_marginals.png", dpi=120); plt.close(fig)
