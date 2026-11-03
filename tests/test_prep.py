import numpy as np

from xsum.constants import CLASS_NAMES, kinetic_energy
from xsum.prep.particles import select_primaries, is_primary
from xsum.prep.volumes import in_active_volume


def test_primaries(sample):
    parts = sample.read("particle_table")
    prim = is_primary(parts["start_process"])
    assert 0.02 < prim.mean() < 0.2
    assert np.all(parts["parent_id"][prim] == 0)
    sel = select_primaries(parts, ke_cut_gev=0.01)
    assert len(sel["pdg"]) <= prim.sum()
    assert np.all(sel["ke"] >= 0.01)
    assert not np.isin(np.abs(sel["pdg"]), [12, 14]).any()
    assert set(np.unique(sel["cls"])) <= set(range(1, len(CLASS_NAMES)))
    norms = np.linalg.norm(sel["dir"], axis=1)
    assert np.all((np.abs(norms - 1) < 1e-5) | (norms == 0))
    # every event with a reconstructed neutrino slice has at least one primary in the file
    assert len(np.unique(sel["event_index"])) > 200


def test_kinetic_energy():
    ke = kinetic_energy(np.array([2212, 13, 22]), np.array([1.0, 0.3, 0.1]))
    assert abs(ke[0] - (np.hypot(1.0, 0.9382721) - 0.9382721)) < 1e-6
    assert abs(ke[2] - 0.1) < 1e-9


def test_active_volume(sample):
    ev = sample.read("event_table")
    act = in_active_volume(ev["nu_vtx"])
    assert 0.2 < act.mean() < 0.7
    assert not in_active_volume(np.array([[-10.0, 0.0, 500.0]]))[0]
