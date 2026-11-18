#!/usr/bin/env python
"""Train and evaluate the PFParticle stage (set flow), warm-started from an event-stage checkpoint.
Usage: scripts/train/train_pfp.py OUT_DIR --init reports/event00/model.pt --stems data/slim/BNB_All_NoWire_00 [--epochs 12] [--eval-only]"""
import sys, pathlib, argparse, json, glob
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("--init", required=True); ap.add_argument("--stems", nargs="*", default=None); ap.add_argument("--slim-dir", default="data/slim")
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--bs", type=int, default=512); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--pfp-layers", type=int, default=3); ap.add_argument("--steps", type=int, default=64); ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--eval-only", action="store_true"); ap.add_argument("--device", default="cuda"); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only-pfp", action="store_true", help="train the set flow alone, the event stage frozen")
    a = ap.parse_args()
    from xsum.train.pfp_stage import train, evaluate, load_model, make_loaders
    from xsum.data.dataset import EventDataset
    stems = a.stems or sorted(p[:-len(".truth.parquet")] for p in glob.glob(f"{a.slim_dir}/*.truth.parquet"))
    if a.eval_only:
        model, tf, ptf, c = load_model(pathlib.Path(a.out_dir) / "model.pt", a.device)
        ds = EventDataset(stems, active_only=c["active_only"]); idx, tds, ld = make_loaders(ds, tf, ptf, a.bs, 0, a.workers)
    else:
        ds, tf, ptf, idx, ld, out = train(stems, a.out_dir, a.init, a.epochs, a.bs, a.lr, 0, a.device, a.pfp_layers, init_seed=a.seed, workers=a.workers, only_pfp=a.only_pfp)
        model, tf, ptf, c = load_model(pathlib.Path(a.out_dir) / "model.pt", a.device)
    M = evaluate(model, tf, ptf, ds, idx["test"], ld["test"], a.out_dir, a.device, a.steps)
    print(json.dumps({"n_pfp": [M["n_pfp_real"], M["n_pfp_sampled"]], "auc_pfp_decoded": round(M["classifier_auc_pfp"], 4), "auc_pfp_model_continuous": round(M["classifier_auc_pfp_model_continuous"], 4),
                      "w1": {k: round(v["w1"], 3) for k, v in M["pfp_model_space"].items()}, "flags": M["pfp_flags"],
                      "event_sums_p2": {k: round(v["w1"], 3) for k, v in M["event_sums_p2"].items()}}, indent=1))
