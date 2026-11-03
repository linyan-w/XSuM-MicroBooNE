#!/usr/bin/env python
"""Write a small open-sample file with the first --events events of the event-level tables (event, particle,
pandoraPrimary, pandoraPfp, opflash), in the layout of the released files, for the tests. Hit-level tables are
left out (about 16k TPC hits and 16k Pandora hit links per event).

Usage: scripts/curate/make_example.py SRC_FILE_OR_URL tests/data/BNB_All_NoWire_00.first300.h5 --events 300
"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import h5py
import numpy as np
from xsum.io.h5 import Sample
from xsum.io.tables import SLIM_TABLES


def main(a):
    s = Sample(a.src)
    n = min(a.events, s.n_events)
    with h5py.File(a.out, "w") as o:
        o.attrs["source"] = a.src.split("?")[0]
        o.attrs["events"] = n
        for t in SLIM_TABLES:
            g = o.create_group(t)
            r0, r1 = s.row_range(t, 0, n)
            for c in s.columns(t):
                g.create_dataset(c, data=s.h5[t][c][r0:r1], compression="gzip")
            if t == "event_table":
                g.create_dataset("event_id.seq_cnt", data=np.stack([np.arange(n), np.ones(n, np.int64)], 1))
            else:
                seq, cnt = s.seq_cnt(t)
                k = int(np.searchsorted(seq, n))
                g.create_dataset("event_id.seq_cnt", data=np.stack([seq[:k], cnt[:k]], 1))
            print(f"{t}: {r1 - r0} rows")
    print(f"{a.out}: {pathlib.Path(a.out).stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--events", type=int, default=300)
    main(ap.parse_args())
