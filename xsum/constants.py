"""MicroBooNE conventions used across the package. Units of the open-sample HDF5 files: GeV, cm, ticks, PE.

Sources: uboone/OpenSamples v02 (microboone_utils.py, file-content-hdf5.md) and the measurements in docs/technote (Section 4).
"""
import numpy as np

# Detector. One TPC, three wire planes: 0 and 1 induction (+/-60 degrees to vertical, 2400 wires each, 3 mm pitch),
# 2 collection (vertical, 3456 wires). 32 PMTs. 6400 time ticks per readout, 0.5 us per tick.
N_PLANES = 3
N_WIRES = (2400, 2400, 3456)
N_TICKS = 6400
N_PMTS = 32
TICK_US = 0.5

# Active volume (cm), as approximated by microboone_utils.isPosInActiveVolume: x drift, y vertical, z along the beam.
ACTIVE_X = (0.0, 255.0)
ACTIVE_Y = (-116.0, 116.0)
ACTIVE_Z = (10.0, 1036.0)
# The sample was generated in the whole cryostat; events with nu_vtx outside the active volume are a large fraction and
# are never reconstructed as a neutrino slice. They play the role MINERvA's out-of-volume events played: Truth only.

# Drift: x = (tick - trigger_offset) * TIME2CM, microboone_utils.tpcTimeToX
TIME2CM = 0.0548965   # cm per tick (sampling x drift velocity)
TRIG_OFFSET_TICKS = 800

# Beam window of the optical flash (us): the BNB spill, after which cosmic flashes dominate (docs/technote, Table 5)
BEAM_WINDOW_US = (3.0, 5.0)

# Sentinels of the Pandora tables
FILL_NU_SCORE = -999.0      # pandoraPrimary_table.nu_score for cosmic slices without a score
FILL_FLASHMATCH = 9999.0    # pandoraPrimary_table.flashmatch_score when no flash match was attempted
SLICE_PDG_NUMU, SLICE_PDG_NUE, SLICE_PDG_COSMIC = 14, 12, 13
PFP_PDG_TRACK, PFP_PDG_SHOWER = 13, 11

# Masses (GeV) for the kinetic energy of the Geant4 primaries (the table stores |p| only)
PDG_MASS = {
    13: 0.1056584, -13: 0.1056584, 11: 0.000511, -11: 0.000511, 22: 0.0,
    211: 0.1395704, -211: 0.1395704, 111: 0.1349768,
    2212: 0.9382721, 2112: 0.9395654,
    321: 0.493677, -321: 0.493677, 311: 0.497611, -311: 0.497611, 130: 0.497611, 310: 0.497611,
    3122: 1.115683, 3222: 1.18937, 3212: 1.192642, 3112: 1.197449,
    12: 0.0, -12: 0.0, 14: 0.0, -14: 0.0,
}

# Geant4 semantic categories of particle_table.category (OpenSamples microboone_utils.category)
G4_CATEGORY = ["pion", "muon", "kaon", "proton", "electron", "michel", "delta", "other", "photon"]

# Species classes for the model input (one token per primary). 0 is padding.
CLASS_NAMES = ["pad", "mu-", "mu+", "e-", "e+", "gamma", "p", "n", "pi+", "pi-", "pi0", "K+-", "K0", "hyperon", "other"]
PDG_CLASS = {
    13: 1, -13: 2, 11: 3, -11: 4, 22: 5, 2212: 6, 2112: 7, 211: 8, -211: 9, 111: 10,
    321: 11, -321: 11, 311: 12, -311: 12, 130: 12, 310: 12,
    3122: 13, 3222: 13, 3212: 13, 3112: 13, 3322: 13, 3312: 13, 3334: 13,
}
N_CLASSES = len(CLASS_NAMES)


def pdg_class(pdg: np.ndarray) -> np.ndarray:
    """Model class per PDG code; unknown codes (nuclei, exotics) map to 'other'."""
    other = CLASS_NAMES.index("other")
    return np.array([PDG_CLASS.get(int(p), other) for p in pdg], dtype=np.int64)


def kinetic_energy(pdg: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Kinetic energy (GeV) from |p| (GeV) and the PDG mass; particles of unknown mass get their momentum."""
    m = np.array([PDG_MASS.get(int(x), np.nan) for x in pdg])
    ke = np.sqrt(p * p + m * m) - m
    return np.where(np.isnan(m), p, ke)
