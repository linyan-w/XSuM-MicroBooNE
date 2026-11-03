import numpy as np

from xsum.io.h5 import Sample
from xsum.io.tables import SLIM_TABLES


def test_tables_and_counts(sample: Sample):
    assert sample.n_events == 300
    for t in SLIM_TABLES:
        assert t in sample.tables
        seq, cnt = (np.arange(300), np.ones(300)) if t == "event_table" else sample.seq_cnt(t)
        assert cnt.sum() == sample.h5[t]["event_id"].shape[0]


def test_event_index_matches_event_id(sample: Sample):
    """The seq_cnt bookkeeping must reproduce event_id row by row."""
    ev = sample.read("event_table")
    for t in SLIM_TABLES:
        rows = sample.read(t, ["event_id"])
        assert np.array_equal(rows["event_id"], ev["event_id"][rows["event_index"]]), t


def test_event_range_slicing(sample: Sample):
    full = sample.read("particle_table", ["g4_id"])
    part = sample.read("particle_table", ["g4_id"], event_start=100, event_stop=150)
    sel = (full["event_index"] >= 100) & (full["event_index"] < 150)
    assert np.array_equal(part["g4_id"], full["g4_id"][sel])
    assert np.array_equal(part["event_index"], full["event_index"][sel])
    c = sample.counts("pandoraPfp_table")
    assert c.shape == (300,) and c.sum() == sample.h5["pandoraPfp_table"]["pfp_id"].shape[0]
