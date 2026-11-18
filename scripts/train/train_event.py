#!/usr/bin/env python
"""Train and evaluate the event-stage surrogate on slim files.
Usage: scripts/train/train_event.py OUT_DIR --stems data/slim/BNB_All_NoWire_00 [...] [--epochs 20] [--bs 512] [--eval-only]"""
import sys, pathlib, argparse, json, glob
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("--stems", nargs="*", default=None); ap.add_argument("--slim-dir", default="data/slim")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--bs", type=int, default=512); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128); ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--flow-hidden", type=int, default=512); ap.add_argument("--flow-layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=64); ap.add_argument("--seed", type=int, default=None, help="initialisation and shuffling seed (ensemble members); the split stays at seed 0")
    ap.add_argument("--active-only", action="store_true", help="train and evaluate on events with the true vertex in the active volume")
    ap.add_argument("--eval-only", action="store_true"); ap.add_argument("--device", default="cuda"); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    from xsum.train.event_stage import train, evaluate, load_model, make_loaders
    from xsum.data.dataset import EventDataset
    stems = a.stems or sorted(p[:-len(".truth.parquet")] for p in glob.glob(f"{a.slim_dir}/*.truth.parquet"))
    print(f"{len(stems)} files: {stems}")
    if a.eval_only:
        model, tf, c = load_model(pathlib.Path(a.out_dir) / "model.pt", a.device)
        ds = EventDataset(stems, active_only=c["active_only"]); idx, tds, ld = make_loaders(ds, tf, a.bs, 0, a.workers)
    else:
        ds, tf, idx, ld, out = train(stems, a.out_dir, a.epochs, a.bs, a.lr, 0, a.device, a.d_model, a.n_layers, a.flow_hidden, a.flow_layers, a.active_only, init_seed=a.seed, workers=a.workers)
        model, tf, c = load_model(pathlib.Path(a.out_dir) / "model.pt", a.device)
    M = evaluate(model, tf, ds, idx["test"], ld["test"], a.out_dir, a.device, a.steps)
    print(json.dumps({"flags_logloss": {k: round(v["logloss"], 4) for k, v in M["flags"].items()}, "flags_auc": {k: round(v["auc"], 3) for k, v in M["flags"].items()},
                      "mult_logloss": round(M["multiplicity"]["logloss"], 4), "rates": {k: round(v, 3) for k, v in M["rates"].items()},
                      "auc_marginal": round(M["classifier_auc_marginal"], 4), "auc_conditional": round(M["classifier_auc_conditional"], 4),
                      "w1": {k: round(v["w1"], 4) for k, v in M["event_model_space"].items()}}, indent=1))
