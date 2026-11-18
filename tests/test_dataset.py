import importlib.util, pathlib

import numpy as np
import pytest

from xsum.data.dataset import EventDataset, split_by_subrun, EVENT_FIELDS, PFP_FIELDS, N_PFP_MAX

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def slim_stem(sample, tmp_path_factory):
    spec = importlib.util.spec_from_file_location("slim", ROOT / "scripts/curate/slim.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    stem = str(tmp_path_factory.mktemp("slim") / "ex")
    m.slim(sample.path, stem, None, 0.01, chunk=128)
    return stem


def test_dataset(slim_stem, sample):
    ds = EventDataset(slim_stem, n_max=32)
    assert len(ds) == sample.n_events
    assert ds.has_slice.sum() == 123
    it = ds[1]
    assert it["cls"].shape == (32,) and it["feat"].shape == (32, 4) and it["ctx"].shape == (3,)
    assert it["event"].shape == (len(EVENT_FIELDS),) and it["pfp"].shape == (N_PFP_MAX, len(PFP_FIELDS))
    assert (ds.mask.sum(1) == np.minimum(ds.n_prim, 32)).all()
    ok = ds.has_slice
    assert np.isfinite(ds.event[ok, :5]).all() and np.isnan(ds.event[~ok]).all()
    assert (ds.n_pfp[~ok] == 0).all() and (ds.n_pfp[ok] >= 1).all()
    assert (np.isfinite(ds.pfp[..., 0]) == ds.pfp_mask).all()
    assert np.isnan(ds.event[ok & ~ds.flash_found, 5]).all() and np.isfinite(ds.event[ds.flash_found, 5]).all()
    # matched slices sit at the truth, wrong ones far away
    assert np.nanmedian(np.abs(ds.event[ds.matched, :3])) < 2.0
    assert ds.matched[ok].mean() > 0.5
    print("\n" + ds.summary())


def test_split_by_subrun():
    run = np.repeat(np.arange(5), 100); sub = np.tile(np.repeat(np.arange(10), 10), 5)
    s = split_by_subrun(run, sub, seed=1)
    key = run * 100000 + sub
    for k in np.unique(key):
        assert len(np.unique(s[key == k])) == 1
    assert abs(np.bincount(s, minlength=3)[0] / len(s) - 0.8) < 0.1
