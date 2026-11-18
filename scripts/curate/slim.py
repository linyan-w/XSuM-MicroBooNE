#!/usr/bin/env python
"""Slim one open-sample file into two Parquet files, the analogue of the MINERvA truth/reco split:

  <stem>.truth.parquet  one row per event: event-table columns (nu_vtx is the model input; nu_vtx_corr, the space-charge
                        corrected vertex, is for validation only) and the jagged list of Geant4 primaries above the
                        kinetic-energy cut (pdg, class, |p|, kinetic energy, start/end positions, direction proxy)
  <stem>.reco.parquet   one row per event that has a Pandora neutrino slice (slice_pdg 12 or 14, at most one per event):
                        the slice (vertex, nu_score, flash-match score, slice_pdg), the jagged list of its PFParticles
                        (pdg, track/shower score, vertex), and the beam-window flash (time, PE, y/z barycentre)

Events are cross-linked by `event_index` (position in the file's event table) and `event_id` (run, subrun, event).
The file is processed in chunks of --chunk events (one slice per column per chunk, so memory stays at a few hundred MB
whatever the file size) and each chunk is one Parquet row group.
Usage: scripts/curate/slim.py FILE_OR_URL data/slim/<stem> [--events N] [--chunk 5000] [--ke-cut-gev 0.01]
"""
from __future__ import annotations
import argparse, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import awkward as ak
import numpy as np
import pyarrow.parquet as pq
from xsum.io.h5 import Sample
from xsum.io.tables import SLIM_TABLES
from xsum.prep.particles import select_primaries
from xsum.prep.volumes import in_active_volume
from xsum.constants import BEAM_WINDOW_US, SLICE_PDG_NUMU, SLICE_PDG_NUE


def jagged(flat: dict, event_index: np.ndarray, n_events: int, fields, offset: int = 0) -> dict:
    counts = np.bincount(event_index - offset, minlength=n_events)
    order = np.argsort(event_index, kind="stable")
    return {f: ak.unflatten(flat[f][order], counts) for f in fields}


def slim_chunk(s: Sample, e0: int, e1: int, ke_cut: float) -> tuple[ak.Array, ak.Array]:
    n = e1 - e0
    ev = s.read("event_table", SLIM_TABLES["event_table"], e0, e1)
    truth = dict(ev)
    truth["in_active_volume"] = in_active_volume(ev["nu_vtx"])
    prim = select_primaries(s.read("particle_table", SLIM_TABLES["particle_table"], e0, e1), ke_cut)
    pj = jagged(prim, prim["event_index"], n, ["g4_id", "pdg", "cls", "p", "ke", "dir", "start", "end"], e0)
    truth.update({"prim_" + k: v for k, v in pj.items()})
    truth["n_prim"] = ak.num(pj["pdg"])
    # reco: the neutrino slice (at most one per event), its PFPs, the beam flash
    pr = s.read("pandoraPrimary_table", SLIM_TABLES["pandoraPrimary_table"], e0, e1)
    nu = np.isin(pr["slice_pdg"], [SLICE_PDG_NUMU, SLICE_PDG_NUE])
    ei = pr["event_index"][nu]
    assert len(np.unique(ei)) == len(ei), "more than one neutrino slice in an event"
    nsl = np.bincount(pr["event_index"] - e0, minlength=n)
    reco = {"event_index": ei, "event_id": ev["event_id"][ei - e0], "n_slices": nsl[ei - e0]}
    for k in ("slice_id", "slice_pdg", "nu_score", "flashmatch_score", "vtx", "vtx_wire_pos", "vtx_wire_time"):
        reco["slice_" + k.replace("slice_", "")] = pr[k][nu]
    pf = s.read("pandoraPfp_table", SLIM_TABLES["pandoraPfp_table"], e0, e1)
    pfj = jagged(pf, pf["event_index"], n, ["pfp_id", "pfp_pdg", "trkshr_score", "vtx", "vtx_wire_time"], e0)
    for k, v in pfj.items():
        reco["pfp_" + k.replace("pfp_", "")] = v[ei - e0]
    reco["n_pfp"] = ak.num(pfj["pfp_id"])[ei - e0]
    fl = s.read("opflash_table", SLIM_TABLES["opflash_table"], e0, e1)
    inwin = (fl["time"] > BEAM_WINDOW_US[0]) & (fl["time"] < BEAM_WINDOW_US[1])
    best = np.full(n, -1); pe = np.full(n, -1.0)   # the brightest in-window flash per event
    for i in np.flatnonzero(inwin):
        e = fl["event_index"][i] - e0
        if fl["totalpe"][i] > pe[e]:
            pe[e], best[e] = fl["totalpe"][i], i
    b = best[ei - e0]; has = b >= 0
    reco["flash_found"] = has
    for k in ("time", "totalpe", "y_center", "y_width", "z_center", "z_width"):
        v = np.full(len(ei), np.nan, np.float32); v[has] = fl[k][b[has]]
        reco["flash_" + k] = v
    return ak.Array(truth), ak.Array(reco)


def slim(src: str, stem: str, events: int | None, ke_cut: float, chunk: int = 5000):
    t0 = time.time()
    s = Sample(src)
    n = s.n_events if events is None else min(events, s.n_events)
    writers = {}
    n_truth = n_reco = 0; n_act = 0
    for e0 in range(0, n, chunk):
        e1 = min(e0 + chunk, n)
        truth, reco = slim_chunk(s, e0, e1, ke_cut)
        for name, arr in (("truth", truth), ("reco", reco)):
            tab = ak.to_arrow_table(arr)
            if name not in writers:
                writers[name] = pq.ParquetWriter(f"{stem}.{name}.parquet", tab.schema)
            writers[name].write_table(tab)
        n_truth += len(truth); n_reco += len(reco); n_act += int(ak.sum(truth["in_active_volume"]))
        print(f"  events {e0}-{e1}: {len(reco)} reco rows, {time.time()-t0:.0f} s", flush=True)
    for w in writers.values():
        w.close()
    print(f"{stem}: {n_truth} truth rows, {n_reco} reco rows ({n_reco/n_truth:.3f}), "
          f"{n_act/n_truth:.3f} in the active volume, {time.time()-t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src"); ap.add_argument("stem")
    ap.add_argument("--events", type=int); ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--ke-cut-gev", type=float, default=0.01)
    a = ap.parse_args()
    pathlib.Path(a.stem).parent.mkdir(parents=True, exist_ok=True)
    slim(a.src, a.stem, a.events, a.ke_cut_gev, a.chunk)
