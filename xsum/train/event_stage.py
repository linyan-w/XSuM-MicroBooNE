"""Event-stage training and evaluation: set encoder + flag heads + multiplicity head + event flow (XSuM-MINERvA's
event stage, on the MicroBooNE targets)."""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from ..data.dataset import EventDataset, EVENT_FIELDS
from ..data.torch_data import BucketBatchSampler, TorchEvents, collate
from ..data.transform import EventTransform, MODEL_NAMES, design_mask
from ..eval.metrics import binary_metrics, confusion, multiclass_metrics, sample_vs_real_1d
from ..models.surrogate import Surrogate, N_PFP_CLASSES
from ..constants import N_CLASSES

FLAG_NAMES = ["slice", "matched", "nue", "flash"]


def make_loaders(ds, tf, bs, seed, workers=2, bucket=True):
    idx = {k: np.where(ds.split == v)[0] for k, v in (("train", 0), ("val", 1), ("test", 2))}
    tds = {k: TorchEvents(ds, v, tf) for k, v in idx.items()}
    ld = {k: DataLoader(tds[k], batch_size=bs, shuffle=False, collate_fn=collate, num_workers=workers) for k in tds if k != "train"}
    if bucket:
        ld["train"] = DataLoader(tds["train"], batch_sampler=BucketBatchSampler(ds.mask[idx["train"]].sum(1), bs, seed), collate_fn=collate, num_workers=workers)
    else:
        ld["train"] = DataLoader(tds["train"], batch_size=bs, shuffle=True, collate_fn=collate, num_workers=workers, drop_last=True)
    return idx, tds, ld


def to_dev(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


def train(stems, out_dir, epochs=20, bs=512, lr=3e-4, seed=0, device="cuda", d_model=128, n_layers=4,
          flow_hidden=512, flow_layers=4, active_only=False, log_every=100, init_seed=None, bucket=True, workers=2):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    init_seed = seed if init_seed is None else init_seed; torch.manual_seed(init_seed); np.random.seed(init_seed)
    t0 = time.time()
    ds = EventDataset(stems, active_only=active_only)
    ok_tr = (ds.split == 0) & ds.has_slice
    tf = EventTransform().fit(np.where(design_mask(ds.matched[ok_tr], ds.flash_found[ok_tr]) > 0, ds.event[ok_tr], np.nan))
    idx, tds, ld = make_loaders(ds, tf, bs, init_seed, workers, bucket)
    print(ds.summary(), flush=True)
    print(f"data: {len(ds)} events ({len(idx['train'])} train / {len(idx['val'])} val / {len(idx['test'])} test), {time.time()-t0:.0f} s", flush=True)
    model = Surrogate(d_model, 4, n_layers, flow_hidden, flow_layers, n_classes=N_CLASSES).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, epochs * len(ld["train"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    hist, best, step = [], np.inf, 0
    print(f"model: {n_par/1e6:.2f} M parameters, {steps} steps", flush=True)
    config = {"d_model": d_model, "n_layers": n_layers, "flow_hidden": flow_hidden, "flow_layers": flow_layers,
              "active_only": active_only, "n_classes": N_CLASSES, "stems": list(stems) if not isinstance(stems, str) else [stems]}
    for ep in range(epochs):
        model.train(); agg = {}
        for b in ld["train"]:
            b = to_dev(b, device)
            L = model.losses(b); loss = sum(L.values())
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            for k, v in L.items(): agg[k] = agg.get(k, 0) + v.item()
            if step % log_every == 0:
                print(f"  step {step} " + " ".join(f"{k} {v.item():.4f}" for k, v in L.items()), flush=True)
        model.eval(); va = {}
        with torch.no_grad():
            for b in ld["val"]:
                L = model.losses(to_dev(b, device))
                for k, v in L.items(): va[k] = va.get(k, 0) + v.item() / len(ld["val"])
        rec = {"epoch": ep, "train": {k: v / max(1, len(ld["train"])) for k, v in agg.items()}, "val": va, "time": time.time() - t0}
        hist.append(rec); tot = sum(va.values())
        print(f"epoch {ep}: val " + " ".join(f"{k} {v:.4f}" for k, v in va.items()) + f"  total {tot:.4f}  [{rec['time']:.0f} s]", flush=True)
        if np.isfinite(tot) and tot < best:
            best = tot
            torch.save({"model": model.state_dict(), "transform": tf.state(), "config": config}, out / "model.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    return ds, tf, idx, ld, out


def load_model(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    c = ck["config"]
    m = Surrogate(c["d_model"], 4, c["n_layers"], c["flow_hidden"], c["flow_layers"], n_classes=c["n_classes"]).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m, EventTransform.from_state(ck["transform"]), c


def sample_loader(model, ld, n_steps=64, device="cuda", seed=0):
    torch.manual_seed(seed); acc = {}
    with torch.no_grad():
        for b in ld:
            s = model.sample(to_dev(b, device), n_steps)
            for k, v in s.items(): acc.setdefault(k, []).append(v.cpu())
    return {k: torch.cat(v).numpy() for k, v in acc.items()}


def truth_features(ds: EventDataset, idx):
    """Summary of the input for the conditional classifier test: vertex, n primaries, summed KE, class counts, max KE."""
    m = ds.mask[idx]; ke = ds.feat[idx, :, 0] * m
    cnt = np.stack([((ds.cls[idx] == c) & m).sum(1) for c in range(1, N_CLASSES)], 1)
    return np.concatenate([ds.ctx[idx], m.sum(1)[:, None], ke.sum(1)[:, None], ke.max(1)[:, None], cnt], 1).astype(np.float32)


@torch.no_grad()
def evaluate(model, tf, ds, idx_test, ld_test, out_dir, device="cuda", n_steps=64, seed=0):
    out = pathlib.Path(out_dir); (out / "figures").mkdir(exist_ok=True, parents=True)
    S = sample_loader(model, ld_test, n_steps, device, seed)
    flags = np.stack([ds.has_slice, ds.matched, ds.is_nue_slice, ds.flash_found], 1)[idx_test]
    has = flags[:, 0]; P = S["p"]
    M = {"n_test": int(len(idx_test)), "flags": {"slice": binary_metrics(has.astype(int), P[:, 0])}}
    for j, k in enumerate(FLAG_NAMES[1:], start=1):
        M["flags"][k] = binary_metrics(flags[has, j].astype(int), P[has, j])
    yn = np.clip(ds.n_pfp[idx_test] - 1, 0, 9)
    M["multiplicity"] = multiclass_metrics(yn[has], S["pn"][has])
    sn = S["npfp"]; sl = S["slice"]
    M["multiplicity"]["confusion_true_vs_sampled"] = confusion(yn[has], sn[has], N_PFP_CLASSES).tolist()
    M["multiplicity"]["marginal_true"] = np.bincount(yn[has], minlength=N_PFP_CLASSES).tolist()
    M["multiplicity"]["marginal_sampled"] = np.bincount(sn[sl], minlength=N_PFP_CLASSES).tolist()
    M["rates"] = {"slice_true": float(has.mean()), "slice_sampled": float(sl.mean())}
    for j, k in enumerate(FLAG_NAMES[1:], start=1):
        M["rates"][k + "_true"] = float(flags[has, j].mean()); M["rates"][k + "_sampled"] = float(S["cflags"][sl, j - 1].mean())
    # event vector: real (events with a slice) vs sampled (events the surrogate gave a slice), per column with its mask
    xr, mr = tf.forward(ds.event[idx_test[has]]); xf, mf = S["x"][sl], S["dmask"][sl]
    mr *= design_mask(ds.matched[idx_test[has]], ds.flash_found[idx_test[has]]); xr[mr == 0] = 0.0
    M["event_model_space"] = {n: sample_vs_real_1d(xr[mr[:, j] > 0, j], xf[mf[:, j] > 0, j]) for j, n in enumerate(MODEL_NAMES)}
    yr = tf.inverse(xr, mr); yf = tf.inverse(xf, mf)
    M["event_raw"] = {n: sample_vs_real_1d(yr[np.isfinite(yr[:, j]), j], yf[np.isfinite(yf[:, j]), j]) for j, n in enumerate(EVENT_FIELDS)}
    # classifier closure: real vs surrogate rows, marginal (flags, N, x) and conditional (+ truth features), split by event
    Xt = truth_features(ds, idx_test)
    Rr = np.concatenate([flags[has, 1:4], yn[has][:, None], xr], 1); Rf = np.concatenate([S["cflags"][sl], sn[sl][:, None], xf], 1)
    n = min(len(Rr), len(Rf)); rng = np.random.default_rng(seed)
    ir, if_ = rng.permutation(len(Rr))[:n], rng.permutation(len(Rf))[:n]
    Xr, Xf = np.concatenate([Xt[has][ir], Rr[ir]], 1), np.concatenate([Xt[sl][if_], Rf[if_]], 1)
    Xc = np.concatenate([Xr, Xf]); yc = np.concatenate([np.ones(n), np.zeros(n)])
    half = n // 2; tr = np.r_[np.arange(half), n + np.arange(half)]; ts = np.r_[np.arange(half, n), n + np.arange(half, n)]
    def _auc(cols):
        c = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, early_stopping=True, random_state=seed)
        c.fit(Xc[tr][:, cols], yc[tr]); return float(roc_auc_score(yc[ts], c.predict_proba(Xc[ts][:, cols])[:, 1]))
    nX = Xt.shape[1]; nR = Rr.shape[1]
    M["classifier_auc_marginal"] = _auc(list(range(nX, nX + nR)))
    M["classifier_auc_conditional"] = _auc(list(range(nX + nR)))
    M["classifier_auc_by_variable"] = {k: _auc([nX + j]) for j, k in enumerate(["matched", "nue", "flash", "npfp"] + MODEL_NAMES)}
    M["corr_real"] = np.corrcoef(xr.T).round(3).tolist(); M["corr_fake"] = np.corrcoef(xf.T).round(3).tolist()
    (out / "metrics.json").write_text(json.dumps(M, indent=1, default=float))
    _figures(M, xr, mr, xf, mf, out / "figures")
    return M


def _figures(M, xr, mr, xf, mf, fdir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 5, figsize=(17, 6))
    for j, (ax, nm) in enumerate(zip(axs.flat, MODEL_NAMES)):
        r, f = xr[mr[:, j] > 0, j], xf[mf[:, j] > 0, j]
        bins = np.linspace(-6, 6, 70)
        ax.hist(r, bins, histtype="step", color="k", label="open sample", density=True)
        ax.hist(f, bins, histtype="step", color="C1", label="surrogate", density=True)
        ax.set_title(f"{nm}  W1 {M['event_model_space'][nm]['w1']:.3f}", fontsize=9); ax.set_yscale("log")
    ax = axs.flat[-1]
    mt, ms = np.array(M["multiplicity"]["marginal_true"]), np.array(M["multiplicity"]["marginal_sampled"])
    ax.bar(np.arange(1, 11) - 0.2, mt / mt.sum(), 0.4, color="k", label="open sample"); ax.bar(np.arange(1, 11) + 0.2, ms / ms.sum(), 0.4, color="C1", label="surrogate")
    ax.set_title("N PFParticles", fontsize=9); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(fdir / "event_marginals.png", dpi=130); plt.close(fig)
