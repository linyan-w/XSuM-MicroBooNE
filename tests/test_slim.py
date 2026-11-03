import importlib.util, pathlib

import awkward as ak
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _slim():
    spec = importlib.util.spec_from_file_location("slim", ROOT / "scripts/curate/slim.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.slim


def test_slim_round_trip(sample, tmp_path):
    stem = str(tmp_path / "ex")
    _slim()(sample.path, stem, None, 0.01)
    t = ak.from_parquet(stem + ".truth.parquet"); r = ak.from_parquet(stem + ".reco.parquet")
    assert len(t) == sample.n_events
    nu = np.isin(sample.read("pandoraPrimary_table", ["slice_pdg"])["slice_pdg"], [12, 14])
    assert len(r) == nu.sum()
    # cross-link and PFP ownership
    ei = ak.to_numpy(r["event_index"])
    assert np.array_equal(ak.to_numpy(r["event_id"]), ak.to_numpy(t["event_id"])[ei])
    assert np.array_equal(ak.to_numpy(r["n_pfp"]), sample.counts("pandoraPfp_table")[ei])
    assert ak.all(ak.num(r["pfp_pdg"]) == r["n_pfp"])
    assert np.array_equal(ak.to_numpy(ak.num(t["prim_pdg"])), ak.to_numpy(t["n_prim"]))
    # some events have no primary above the cut (and 17% of events have no particle rows at all, docs/dataset.md),
    # with or without a slice; the flash columns are NaN exactly where no flash was found
    assert ak.sum(t["n_prim"] == 0) > 0
    assert np.array_equal(np.isnan(ak.to_numpy(r["flash_time"])), ~ak.to_numpy(r["flash_found"]))
    eff = len(r) / len(t)
    assert 0.3 < eff < 0.5
