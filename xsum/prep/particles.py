"""The generator-level input of one event: the Geant4 primaries of the particle table.

`start_process == "primary"` marks the particles that emerge from the neutrino interaction (the generator's final state
after final-state interactions, as handed to Geant4); everything else in the table is a secondary. Species and the
momentum magnitude are the particle's initial state (generator level). The table stores no momentum vector:
`direction()` is the unit start-to-end displacement of the particle's own Geant4 trajectory, a DEVELOPMENT STOPGAP
that carries Geant4 propagation (the end point is where the track stopped, converted or interacted); the deployed
model must take the generator's initial momentum vector from the artroot companion file (docs/technote, Section 3 and
the open questions). A neutral pion decays at the vertex (zero displacement for 42% of them), so its direction is
rebuilt from its two decay photons: the momentum-weighted sum of the photons' displacement directions, which are
straight lines to their conversion points (`pi0_direction_from_daughters`).
"""
from __future__ import annotations

import numpy as np

from ..constants import kinetic_energy, pdg_class


def is_primary(start_process: np.ndarray) -> np.ndarray:
    sp = start_process.astype("S") if start_process.dtype.kind != "S" else start_process
    return sp == b"primary"


def direction(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Unit vector from start to end position; zero where the two coincide."""
    d = end - start
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    return np.where(n > 0, d / np.where(n > 0, n, 1.0), 0.0)


def pi0_direction_from_daughters(parts: dict, idx: np.ndarray, d: np.ndarray) -> np.ndarray:
    """For the primaries `idx` (rows of `parts`) that are neutral pions, replace their direction `d` by the
    momentum-weighted sum of their decay daughters' displacement directions (rows with parent_id == the pion's g4_id
    in the same event). Pions without daughters in the table keep the displacement proxy."""
    pdg = parts["g4_pdg"]; ev = parts["event_index"]; par = parts["parent_id"]; g4 = parts["g4_id"]
    pi = np.flatnonzero(pdg[idx] == 111)
    if not len(pi):
        return d
    dd = direction(parts["start_position"], parts["end_position"]); p = parts["momentum"].astype(np.float64)
    key = (ev.astype(np.int64) << 32) | par.astype(np.int64)            # (event, parent) of every row
    order = np.argsort(key, kind="stable"); ks = key[order]
    out = d.copy()
    for j in pi:
        r = idx[j]; k = (int(ev[r]) << 32) | int(g4[r])
        a, b = np.searchsorted(ks, k, "left"), np.searchsorted(ks, k, "right")
        rows = order[a:b]
        if len(rows) == 0:
            continue
        v = (p[rows, None] * dd[rows]).sum(0); n = np.linalg.norm(v)
        if n > 0:
            out[j] = v / n
    return out


def select_primaries(parts: dict, ke_cut_gev: float = 0.01, keep_neutrons: bool = True) -> dict:
    """Primaries above the kinetic-energy cut, as flat arrays with the model class, kinetic energy and direction proxy.
    `parts` is `Sample.read("particle_table", ...)`."""
    prim = is_primary(parts["start_process"])
    pdg = parts["g4_pdg"][prim]
    p = parts["momentum"][prim].astype(np.float64)
    ke = kinetic_energy(pdg, p)
    keep = ke >= ke_cut_gev
    if not keep_neutrons:
        keep &= pdg != 2112
    keep &= ~np.isin(np.abs(pdg), [12, 14, 16])  # the outgoing neutrino of an NC event is not visible
    idx = np.flatnonzero(prim)[keep]
    d = pi0_direction_from_daughters(parts, idx, direction(parts["start_position"][idx], parts["end_position"][idx]))
    return {
        "event_index": parts["event_index"][idx],
        "g4_id": parts["g4_id"][idx],
        "pdg": pdg[keep],
        "cls": pdg_class(pdg[keep]),
        "p": p[keep],
        "ke": ke[keep],
        "dir": d,
        "start": parts["start_position"][idx],
        "end": parts["end_position"][idx],
    }
