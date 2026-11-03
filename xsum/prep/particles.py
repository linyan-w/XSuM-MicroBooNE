"""The generator-level input of one event: the Geant4 primaries of the particle table.

`start_process == "primary"` marks the particles that emerge from the neutrino interaction (the generator's final state
after final-state interactions, as handed to Geant4); everything else in the table is a secondary. The table stores the
momentum magnitude and the start/end positions, not the direction: `direction()` is the start-to-end displacement, a
proxy that is exact for particles that stop or exit without scattering and approximate otherwise (docs/dataset.md).
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
    return {
        "event_index": parts["event_index"][idx],
        "g4_id": parts["g4_id"][idx],
        "pdg": pdg[keep],
        "cls": pdg_class(pdg[keep]),
        "p": p[keep],
        "ke": ke[keep],
        "dir": direction(parts["start_position"][idx], parts["end_position"][idx]),
        "start": parts["start_position"][idx],
        "end": parts["end_position"][idx],
    }
