"""Smoke test of the event stage on the 300-event fixture (skipped without torch)."""
import importlib.util, pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def slim_stem(sample, tmp_path_factory):
    spec = importlib.util.spec_from_file_location("slim", ROOT / "scripts/curate/slim.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    stem = str(tmp_path_factory.mktemp("slim") / "ex"); m.slim(sample.path, stem, None, 0.01, chunk=128); return stem


def test_transform_round_trip(slim_stem):
    from xsum.data.dataset import EventDataset
    from xsum.data.transform import EventTransform
    ds = EventDataset(slim_stem); y = ds.event[ds.has_slice]
    tf = EventTransform().fit(y); x, m = tf.forward(y)
    assert np.isfinite(x).all() and (x[m == 0] == 0).all()
    yb = tf.inverse(x, m)
    fin = np.isfinite(y) & (np.abs(x) < tf.clip - 1e-3)
    assert np.allclose(yb[fin], y[fin], rtol=1e-4, atol=1e-3)
    assert np.isnan(yb[m == 0]).all()


def test_train_and_sample(slim_stem, tmp_path):
    from xsum.train.event_stage import train, evaluate, load_model
    ds, tf, idx, ld, out = train(slim_stem, tmp_path / "m", epochs=2, bs=32, device="cpu", d_model=32, n_layers=1, flow_hidden=64, flow_layers=2, workers=0, bucket=False)
    model, tf2, c = load_model(out / "model.pt", "cpu")
    M = evaluate(model, tf2, ds, idx["test"], ld["test"], out, "cpu", n_steps=8)
    assert 0 < M["rates"]["slice_sampled"] < 1 and len(M["event_model_space"]) == 9
    assert np.isfinite([v["w1"] for v in M["event_model_space"].values()]).all()
    assert (out / "figures" / "event_marginals.png").exists()


def test_pfp_stage(slim_stem, tmp_path):
    """Hit-level slim on the fixture (hit tables for its first 20 events), then the set-flow stage warm-started from a
    tiny event-stage model."""
    spec = importlib.util.spec_from_file_location("slim_hits", ROOT / "scripts/curate/slim_hits.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    from xsum.io.h5 import Sample
    src = str(ROOT / "tests/data/BNB_All_NoWire_00.first300.h5")
    m.slim_hits(src, slim_stem, None, 64)
    from xsum.data.dataset import EventDataset
    ds = EventDataset(slim_stem)
    assert ds.has_hits
    nh = ds.pfp[ds.pfp_mask][:, 4::4]
    first20 = ds.has_slice & (np.arange(len(ds)) < 20)
    assert (ds.pfp[first20][ds.pfp_mask[first20]][:, 4::4].sum(1) > 0).all()     # every PFParticle of the first 20 events owns hits
    assert ds.slice_nhits[first20].sum() > 0 and (ds.slice_nhits[~first20] == 0).all()
    from xsum.train import event_stage, pfp_stage
    _, _, _, _, out = event_stage.train(slim_stem, tmp_path / "m", epochs=1, bs=32, device="cpu", d_model=32, n_layers=1, flow_hidden=64, flow_layers=2, workers=0, bucket=False)
    ds2, tf, ptf, idx, ld, out2 = pfp_stage.train(slim_stem, tmp_path / "p", out / "model.pt", epochs=2, bs=32, device="cpu", pfp_layers=1, workers=0, bucket=False)
    model, tf, ptf, c = pfp_stage.load_model(out2 / "model.pt", "cpu")
    M = pfp_stage.evaluate(model, tf, ptf, ds2, idx["test"], ld["test"], out2, "cpu", n_steps=8)
    assert M["n_pfp_sampled"] > 0 and np.isfinite([v["w1"] for v in M["pfp_model_space"].values()]).all()
    # round trip of the PFParticle transform on real rows
    rng = np.random.default_rng(0); P = ds2.pfp[ds2.pfp_mask]; x = ptf.forward(P, rng); Pb = ptf.inverse(x)
    inside = (np.abs(x[:, 3:6]) < 7.99).all(1)          # the model space is clipped at +-8
    assert inside.mean() > 0.9 and np.allclose(Pb[inside, 1:4], P[inside, 1:4], atol=0.05, rtol=1e-3)
    assert np.array_equal(Pb[:, 0] >= 1 - 1e-6, P[:, 0] >= 1 - 1e-6) and np.array_equal(Pb[:, 4::4], P[:, 4::4])   # score edges and hit counts round-trip
    has = P[:, 4::4] > 0
    assert np.array_equal(Pb[:, 4::4] > 0, has)
