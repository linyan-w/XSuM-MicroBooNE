#!/usr/bin/env python
"""Print the structure of an open-sample file (local path or Zenodo URL: only metadata and the first chunks are read)
and per-event statistics over the first --events events. Writes a Markdown report with --out.

Usage: scripts/curate/h5_summary.py FILE_OR_URL [--events 2000] [--out notes.md]   (the numbers of 2026-11-03 are in docs/technote/technote.pdf, Section 2.2)
"""
import argparse, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
from xsum.io.h5 import Sample
from xsum.prep.particles import select_primaries, is_primary
from xsum.prep.volumes import in_active_volume
from xsum.constants import CLASS_NAMES, G4_CATEGORY, FILL_NU_SCORE


def pct(x, q=(50, 90, 99)):
    return "/".join(f"{v:g}" for v in np.percentile(x, q))


def main(a):
    t0 = time.time()
    s = Sample(a.file)
    parts = [x for x in a.file.split("?")[0].split("/") if x and x != "content"]
    L = [f"# Measured content of `{parts[-1]}`", "",
         f"Source `{a.file}`, read {time.strftime('%Y-%m-%d')} with `scripts/curate/h5_summary.py --events {a.events}`.", "",
         "## Tables", "", "| table | rows | events with rows | columns |", "|---|---|---|---|"]
    for t in s.tables:
        g = s.h5[t]
        cols = ", ".join(f"{c}[{g[c].shape[1]}]" if g[c].shape[1] > 1 else c for c in s.columns(t))
        nev = s.n_events if t == "event_table" else len(s.seq_cnt(t)[0])
        L.append(f"| `{t}` | {g['event_id'].shape[0]:,} | {nev:,} | {cols} |")
    L += ["", f"Events in the file: {s.n_events:,}. Statistics below on the first {a.events:,}.", ""]
    n = min(a.events, s.n_events)
    ev = s.read("event_table", event_stop=n)
    L += ["## Event table", ""]
    pdg, cnt = np.unique(ev["nu_pdg"], return_counts=True)
    L.append("- neutrino flavour: " + ", ".join(f"{p}: {c/n:.3f}" for p, c in zip(pdg, cnt)))
    L.append(f"- charged current: {ev['is_cc'].mean():.3f}")
    L.append(f"- neutrino energy (GeV) p50/90/99: {pct(ev['nu_energy'])}; lepton energy p50/90/99: {pct(ev['lep_energy'])}")
    act = in_active_volume(ev["nu_vtx"])
    L.append(f"- true vertex inside the active volume: {act.mean():.3f} (the sample is generated in the whole cryostat)")
    for i, c in enumerate("xyz"):
        L.append(f"- vertex {c} (cm) min/p1/p50/p99/max: {ev['nu_vtx'][:, i].min():.0f}/{pct(ev['nu_vtx'][:, i], (1, 50, 99))}/{ev['nu_vtx'][:, i].max():.0f}")
    dv = ev["nu_vtx_corr"] - ev["nu_vtx"]
    L.append(f"- space-charge correction (corr - true, cm), p50/p99 of |d|: x {pct(np.abs(dv[:,0]), (50, 99))}, y {pct(np.abs(dv[:,1]), (50, 99))}, z {pct(np.abs(dv[:,2]), (50, 99))}")
    L.append(f"- run/subrun: {len(np.unique(ev['event_id'][:, 0]))} runs, {len(np.unique(ev['event_id'][:, :2], axis=0))} subruns in the first {n} events")
    # particles
    parts = s.read("particle_table", event_stop=n)
    prim = is_primary(parts["start_process"])
    L += ["", "## Particle table", ""]
    L.append(f"- rows per event p50/90/99: {pct(np.bincount(parts['event_index'], minlength=n))}; primaries per event p50/90/99: {pct(np.bincount(parts['event_index'][prim], minlength=n))}")
    L.append(f"- primaries have parent_id values {np.unique(parts['parent_id'][prim]).tolist()[:5]}; non-primary start processes: {', '.join(x.decode() for x in np.unique(parts['start_process'][~prim])[:12])}")
    pp, pc = np.unique(parts["g4_pdg"][prim], return_counts=True)
    order = np.argsort(-pc)[:16]
    L.append("- primary PDG counts: " + ", ".join(f"{pp[i]}: {pc[i]}" for i in order))
    sel = select_primaries(parts, ke_cut_gev=0.01)
    L.append(f"- primaries above 10 MeV kinetic energy (neutrinos dropped) per event p50/90/99/max: {pct(np.bincount(sel['event_index'], minlength=n))}/{np.bincount(sel['event_index'], minlength=n).max()}")
    cc, cn = np.unique(sel["cls"], return_counts=True)
    L.append("- model classes: " + ", ".join(f"{CLASS_NAMES[c]}: {k}" for c, k in zip(cc, cn)))
    cat, catn = np.unique(parts["category"], return_counts=True)
    L.append("- categories (all rows): " + ", ".join(f"{G4_CATEGORY[c] if 0 <= c < len(G4_CATEGORY) else c}: {k}" for c, k in zip(cat, catn)))
    zero = np.linalg.norm(sel["dir"], axis=1) == 0
    L.append(f"- primaries whose start and end position coincide (no direction proxy): {zero.mean():.4f}")
    # pandora
    pr = s.read("pandoraPrimary_table", event_stop=n)
    L += ["", "## Pandora", ""]
    nsl = np.bincount(pr["event_index"], minlength=n)
    L.append(f"- slices per event p50/90/99: {pct(nsl)}; slice_pdg counts: {dict(zip(*[x.tolist() for x in np.unique(pr['slice_pdg'], return_counts=True)]))}")
    scored = pr["nu_score"] != FILL_NU_SCORE
    L.append(f"- slices with a nu_score: {scored.mean():.3f}; nu_score p50/90/99 among them: {pct(pr['nu_score'][scored])}")
    nu = np.isin(pr["slice_pdg"], [12, 14])
    assert len(np.unique(pr["event_index"][nu])) == nu.sum(), "more than one neutrino slice in an event"
    has_nu = np.zeros(n, bool); has_nu[pr["event_index"][nu]] = True
    L.append(f"- events with a neutrino slice (slice_pdg 12 or 14; never more than one per event): {has_nu.mean():.3f}; among events with the true vertex in the active volume: {has_nu[act].mean():.3f}; outside: {has_nu[~act].mean():.3f}")
    p11 = pr["slice_pdg"] == 11
    L.append(f"- rows with the undocumented slice_pdg 11: {p11.sum()} ({(pr['vtx'][p11][:, 0] == -999).mean():.2f} of them with vtx = -999); they are not the neutrino candidate and are ignored")
    L.append(f"- nu_score of the neutrino slice p10/50/90: {pct(pr['nu_score'][nu], (10, 50, 90))}; flash-match score p50/90: {pct(pr['flashmatch_score'][nu], (50, 90))}")
    fm = pr["flashmatch_score"] != 9999.0
    L.append(f"- slices with a flash-match score: {fm.mean():.3f}; score p50/90/99: {pct(pr['flashmatch_score'][fm]) if fm.any() else 'none'}")
    # reco vertex of the neutrino slice against the truth
    ei = pr["event_index"][nu]
    d = pr["vtx"][nu] - ev["nu_vtx_corr"][ei]
    ina = act[ei]
    L.append(f"- neutrino-slice vertex minus nu_vtx_corr (cm), |d| p50/90/99: {pct(np.linalg.norm(d, axis=1))} (all); {pct(np.linalg.norm(d[ina], axis=1))} (true vertex in the active volume); per axis p50 of |d| in the active volume: {', '.join(f'{c} {np.percentile(np.abs(d[ina][:, i]), 50):.1f}' for i, c in enumerate('xyz'))}")
    pf = s.read("pandoraPfp_table", event_stop=n)
    npf = np.bincount(pf["event_index"], minlength=n)
    L.append(f"- PFParticles of the neutrino slice per event (events with a slice) p50/90/99/max: {pct(npf[has_nu])}/{npf.max()}; events with a slice but no PFP: {(npf[has_nu] == 0).mean():.3f}")
    L.append(f"- pfp_pdg counts: {dict(zip(*[x.tolist() for x in np.unique(pf['pfp_pdg'], return_counts=True)]))}; trkshr_score p10/50/90: {pct(pf['trkshr_score'], (10, 50, 90))}")
    fl = s.read("opflash_table", event_stop=n)
    nfl = np.bincount(fl["event_index"], minlength=n)
    L += ["", "## Optical flashes", ""]
    L.append(f"- flashes per event p50/90/99: {pct(nfl)}; events without a flash: {(nfl == 0).mean():.3f}")
    L.append(f"- flash time (us) p1/50/99: {pct(fl['time'], (1, 50, 99))}; total PE p50/90/99: {pct(fl['totalpe'])}")
    L += ["", f"Read in {time.time()-t0:.0f} s."]
    txt = "\n".join(L)
    print(txt)
    if a.out:
        pathlib.Path(a.out).write_text(txt + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--events", type=int, default=2000)
    ap.add_argument("--out")
    main(ap.parse_args())
