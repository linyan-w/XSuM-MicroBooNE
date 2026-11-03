#!/usr/bin/env python
"""Slim one open-sample file into two Parquet files, the analogue of the MINERvA truth/reco split:

  <stem>.truth.parquet  one row per event: event-table columns and the jagged list of Geant4 primaries above the
                        kinetic-energy cut (pdg, class, |p|, kinetic energy, start/end positions, direction proxy)
  <stem>.reco.parquet   one row per event that has a Pandora neutrino slice (slice_pdg 12 or 14, at most one per event): the slice (vertex, nu_score, flash-match
                        score, slice_pdg), the jagged list of its PFParticles (pdg, track/shower score, vertex), and the
                        beam-window flash (time, PE, y/z barycentre)

Events are cross-linked by `event_index` (position in the file's event table) and `event_id` (run, subrun, event).
Usage: scripts/curate/slim.py FILE_OR_URL data/slim/<stem> [--events N] [--ke-cut-gev 0.01]
"""
from __future__ import annotations
import argparse, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import awkward as ak
import numpy as np
from xsum.io.h5 import Sample
from xsum.io.tables import SLIM_TABLES
from xsum.prep.particles import select_primaries
from xsum.prep.volumes import in_active_volume
from xsum.constants import BEAM_WINDOW_US, SLICE_PDG_NUMU, SLICE_PDG_NUE


def jagged(flat: dict, event_index: np.ndarray, n_events: int, fields) -> dict:
    counts = np.bincount(event_index, minlength=n_events)
    order = np.argsort(event_index, kind="stable")
    return {f: ak.unflatten(flat[f][order], counts) for f in fields}


def slim(src: str, stem: str, events: int | None, ke_cut: float):
    t0 = time.time()
    s = Sample(src)
    n = s.n_events if events is None else min(events, s.n_events)
    ev = s.read("event_table", SLIM_TABLES["event_table"], event_stop=n)
    truth = {k: ev[k] for k in ev}
    truth["in_active_volume"] = in_active_volume(ev["nu_vtx"])
    prim = select_primaries(s.read("particle_table", SLIM_TABLES["particle_table"], event_stop=n), ke_cut)
    pj = jagged(prim, prim["event_index"], n, ["g4_id", "pdg", "cls", "p", "ke", "dir", "start", "end"])
    truth.update({"prim_" + k: v for k, v in pj.items()})
    truth["n_prim"] = ak.num(pj["pdg"])
    ak.to_parquet(ak.Array(truth), stem + ".truth.parquet")
    # reco: the neutrino slice (Pandora keeps at most one per event), its PFPs, the beam flash
    pr = s.read("pandoraPrimary_table", SLIM_TABLES["pandoraPrimary_table"], event_stop=n)
    nu = np.isin(pr["slice_pdg"], [SLICE_PDG_NUMU, SLICE_PDG_NUE])
    ei = pr["event_index"][nu]
    assert len(np.unique(ei)) == len(ei), "more than one neutrino slice in an event"
    nsl = np.bincount(pr["event_index"], minlength=n)
    reco = {"event_index": ei, "event_id": ev["event_id"][ei], "n_slices": nsl[ei]}
    for k in ("slice_id", "slice_pdg", "nu_score", "flashmatch_score", "vtx", "vtx_wire_pos", "vtx_wire_time"):
        reco["slice_" + k.replace("slice_", "")] = pr[k][nu]
    pf = s.read("pandoraPfp_table", SLIM_TABLES["pandoraPfp_table"], event_stop=n)
    pfj = jagged(pf, pf["event_index"], n, ["pfp_id", "pfp_pdg", "trkshr_score", "vtx", "vtx_wire_time"])
    for k, v in pfj.items():
        reco["pfp_" + k.replace("pfp_", "")] = v[ei]
    reco["n_pfp"] = ak.num(pfj["pfp_id"])[ei]
    fl = s.read("opflash_table", SLIM_TABLES["opflash_table"], event_stop=n)
    inwin = (fl["time"] > BEAM_WINDOW_US[0]) & (fl["time"] < BEAM_WINDOW_US[1])
    # the brightest in-window flash per event
    best = np.full(n, -1); pe = np.full(n, -1.0)
    for i in np.flatnonzero(inwin):
        e = fl["event_index"][i]
        if fl["totalpe"][i] > pe[e]:
            pe[e], best[e] = fl["totalpe"][i], i
    has = best[ei] >= 0
    reco["flash_found"] = has
    for k in ("time", "totalpe", "y_center", "y_width", "z_center", "z_width"):
        v = np.full(len(ei), np.nan, np.float32); v[has] = fl[k][best[ei][has]]
        reco["flash_" + k] = v
    ak.to_parquet(ak.Array(reco), stem + ".reco.parquet")
    print(f"{stem}: {n} truth rows, {len(ei)} reco rows ({len(ei)/n:.3f}), "
          f"{truth['in_active_volume'].mean():.3f} in the active volume, {time.time()-t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src"); ap.add_argument("stem")
    ap.add_argument("--events", type=int); ap.add_argument("--ke-cut-gev", type=float, default=0.01)
    a = ap.parse_args()
    pathlib.Path(a.stem).parent.mkdir(parents=True, exist_ok=True)
    slim(a.src, a.stem, a.events, a.ke_cut_gev)
