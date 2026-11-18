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
