#!/usr/bin/env python
"""Hit-level slim of one open-sample file: per-PFParticle and per-slice summaries of the TPC hits of the neutrino
slice, from `hit_table` and `pandoraHit_table` (one row per hit in the same order; `pfp_id` is -1 for a hit of the
slice that no PFParticle owns, and for every hit outside the neutrino slice).

  <stem>.hits.parquet  one row per row of <stem>.reco.parquet (same order, joined by event_index):
    slice_nhits[3], slice_charge[3]         hits and summed integral (ADC) of the neutrino slice per plane (0, 1, 2)
    slice_free_nhits[3], slice_free_charge[3]   the slice's hits that no PFParticle owns
    pfp_nhits[n_pfp,3], pfp_charge[n_pfp,3]  per PFParticle, in the order of the reco file's pfp_id list
    pfp_dwire[n_pfp,3], pfp_dtick[n_pfp,3]   wire and time span (max - min) of the PFParticle's hits per plane
    pfp_wire0[n_pfp,3], pfp_tick0[n_pfp,3]   charge-weighted mean wire and time per plane (-1 where no hit)

The two hit tables hold 6.7e8 rows per file; they are read in chunks of --chunk events (about 8 M rows per 500 events).
Usage: scripts/curate/slim_hits.py FILE data/slim/<stem> [--events N] [--chunk 500]
"""
from __future__ import annotations
import argparse, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import awkward as ak
import numpy as np
import pyarrow.parquet as pq
from xsum.io.h5 import Sample

N_PLANES = 3


def aggregate(keys_ev, keys_pfp, plane, integral, wire, tick):
    """Group (event, pfp, plane) rows: returns dict of per-group arrays and the group keys."""
    key = np.stack([keys_ev, keys_pfp, plane], 1)
    uk, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
    g = len(uk)
    n = np.bincount(inv, minlength=g)
    q = np.bincount(inv, weights=integral, minlength=g)
    wmin = np.full(g, np.inf); wmax = np.full(g, -np.inf); tmin = np.full(g, np.inf); tmax = np.full(g, -np.inf)
    np.minimum.at(wmin, inv, wire); np.maximum.at(wmax, inv, wire); np.minimum.at(tmin, inv, tick); np.maximum.at(tmax, inv, tick)
    w0 = np.bincount(inv, weights=integral * wire, minlength=g) / np.where(q > 0, q, 1)
    t0 = np.bincount(inv, weights=integral * tick, minlength=g) / np.where(q > 0, q, 1)
    return uk, {"n": n, "q": q, "dw": wmax - wmin, "dt": tmax - tmin, "w0": w0, "t0": t0}


def slim_hits(src: str, stem: str, events: int | None, chunk: int):
    t0 = time.time()
    s = Sample(src)
    reco = ak.from_parquet(stem + ".reco.parquet")
    r_ev = ak.to_numpy(reco["event_index"]); r_sid = ak.to_numpy(reco["slice_id"])
    r_pfp = reco["pfp_id"]; r_npfp = ak.to_numpy(reco["n_pfp"])
    n_ev = s.n_events if events is None else min(events, s.n_events)
    nu_sid = np.full(n_ev, -2, np.int64); keep = r_ev < n_ev; nu_sid[r_ev[keep]] = r_sid[keep]
    row_of = np.full(n_ev, -1); row_of[r_ev[keep]] = np.flatnonzero(keep)
    n_rows = int(keep.sum())
    S = {k: np.zeros((n_rows, N_PLANES), np.float64) for k in ("slice_nhits", "slice_charge", "slice_free_nhits", "slice_free_charge")}
    P = {k: [None] * n_rows for k in ("pfp_nhits", "pfp_charge", "pfp_dwire", "pfp_dtick", "pfp_wire0", "pfp_tick0")}
    for e0 in range(0, n_ev, chunk):
        e1 = min(e0 + chunk, n_ev)
        ph = s.read("pandoraHit_table", ["pfp_id", "slice_id"], e0, e1)
        ht = s.read("hit_table", ["integral", "local_plane", "local_wire", "local_time"], e0, e1)
        assert len(ph["pfp_id"]) == len(ht["integral"]) and np.array_equal(ph["event_index"], ht["event_index"])
        ev = ph["event_index"]
        in_nu = ph["slice_id"] == nu_sid[ev]
        ev, pfp, pl = ev[in_nu], ph["pfp_id"][in_nu], ht["local_plane"][in_nu]
        q, w, t = ht["integral"][in_nu].astype(np.float64), ht["local_wire"][in_nu].astype(np.float64), ht["local_time"][in_nu].astype(np.float64)
        rows = row_of[ev]
        # slice totals and the free hits
        np.add.at(S["slice_nhits"], (rows, pl), 1); np.add.at(S["slice_charge"], (rows, pl), q)
        free = pfp < 0
        np.add.at(S["slice_free_nhits"], (rows[free], pl[free]), 1); np.add.at(S["slice_free_charge"], (rows[free], pl[free]), q[free])
        # per PFParticle
        own = ~free
        if own.any():
            uk, agg = aggregate(ev[own], pfp[own], pl[own], q[own], w[own], t[own])
            lut = {(int(a), int(b), int(c)): i for i, (a, b, c) in enumerate(uk)}
            for r in np.unique(rows):
                e = r_ev[r]; ids = ak.to_numpy(r_pfp[r]); k = len(ids)
                arr = {kk: np.full((k, N_PLANES), -1.0 if kk in ("pfp_wire0", "pfp_tick0") else 0.0) for kk in P}
                for j, pid in enumerate(ids):
                    for p in range(N_PLANES):
                        i = lut.get((int(e), int(pid), p))
                        if i is None: continue
                        arr["pfp_nhits"][j, p] = agg["n"][i]; arr["pfp_charge"][j, p] = agg["q"][i]
                        arr["pfp_dwire"][j, p] = agg["dw"][i]; arr["pfp_dtick"][j, p] = agg["dt"][i]
                        arr["pfp_wire0"][j, p] = agg["w0"][i]; arr["pfp_tick0"][j, p] = agg["t0"][i]
                for kk in P: P[kk][r] = arr[kk]
        print(f"  events {e0}-{e1}: {in_nu.sum()} hits in neutrino slices of {len(np.unique(rows))} events, {time.time()-t0:.0f} s", flush=True)
    for r in range(n_rows):   # reco rows whose events had no owned hit in any chunk
        if P["pfp_nhits"][r] is None:
            k = r_npfp[r]
            for kk in P: P[kk][r] = np.full((k, N_PLANES), -1.0 if kk in ("pfp_wire0", "pfp_tick0") else 0.0)
    out = {"event_index": r_ev[keep], "event_id": reco["event_id"][keep]}
    out.update({k: v.astype(np.float32) if "charge" in k else v.astype(np.int32) for k, v in S.items()})
    for kk in P:
        out[kk] = ak.Array([a.astype(np.float32) if kk in ("pfp_charge", "pfp_wire0", "pfp_tick0") else a.astype(np.int32) for a in P[kk]])
    ak.to_parquet(ak.Array(out), stem + ".hits.parquet")
    print(f"{stem}.hits.parquet: {n_rows} rows, {time.time()-t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src"); ap.add_argument("stem"); ap.add_argument("--events", type=int); ap.add_argument("--chunk", type=int, default=500)
    a = ap.parse_args()
    slim_hits(a.src, a.stem, a.events, a.chunk)
