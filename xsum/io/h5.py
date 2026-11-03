"""Reader for the open-sample HDF5 files (local path or Zenodo URL).

Every table is a group of column datasets with one row per object, plus `event_id.seq_cnt`: rows `[event_seq, count]`
giving, for each event that has rows in the table, the event's position in `event_table` and its number of rows, in
file order. Rows of one event are contiguous, so a table can be read for a range of events with one slice per column.
A URL is read over HTTP range requests through fsspec (metadata and the first chunks of each column cost a few MB).
"""
from __future__ import annotations

import numpy as np

EVENT_TABLE = "event_table"
ROW_TABLES = ("particle_table", "pandoraPrimary_table", "pandoraPfp_table", "opflash_table", "opflashsumpe_table",
              "ophit_table", "hit_table", "edep_table", "pandoraHit_table")


def open_h5(path_or_url: str, block_size: int = 4 * 2**20):
    """h5py.File on a local path, or on a Zenodo/HTTP URL through fsspec's block cache."""
    import h5py
    if str(path_or_url).startswith(("http://", "https://")):
        import fsspec
        fs = fsspec.filesystem("http", block_size=block_size)
        return h5py.File(fs.open(path_or_url, "rb", cache_type="blockcache"), "r")
    return h5py.File(path_or_url, "r")


class Sample:
    """One open-sample file. `tables` lists the groups; `n_events` the rows of the event table."""

    def __init__(self, path_or_url: str):
        self.path = str(path_or_url)
        self.h5 = open_h5(path_or_url)
        self.tables = [k for k in self.h5.keys() if k.endswith("_table")]
        self.n_events = self.h5[EVENT_TABLE]["event_id"].shape[0]
        self._seq = {}

    def close(self):
        self.h5.close()

    def columns(self, table: str):
        return [k for k in self.h5[table].keys() if not k.endswith(".seq_cnt")]

    def seq_cnt(self, table: str) -> tuple[np.ndarray, np.ndarray]:
        """(event index into event_table, row count) per event present in `table`, in file order."""
        if table not in self._seq:
            sc = self.h5[table]["event_id.seq_cnt"][:]
            seq, cnt = sc[:, 0].astype(np.int64), sc[:, 1].astype(np.int64)
            assert np.all(np.diff(seq) > 0), f"{table}: event sequence is not increasing"
            self._seq[table] = (seq, cnt)
        return self._seq[table]

    def row_range(self, table: str, event_start: int = 0, event_stop: int | None = None) -> tuple[int, int]:
        """Row interval [r0, r1) of `table` holding the events with index in [event_start, event_stop)."""
        if table == EVENT_TABLE:
            return event_start, (self.n_events if event_stop is None else min(event_stop, self.n_events))
        seq, cnt = self.seq_cnt(table)
        offs = np.concatenate([[0], np.cumsum(cnt)])
        i0 = int(np.searchsorted(seq, event_start, side="left"))
        i1 = len(seq) if event_stop is None else int(np.searchsorted(seq, event_stop, side="left"))
        return int(offs[i0]), int(offs[i1])

    def read(self, table: str, columns=None, event_start: int = 0, event_stop: int | None = None) -> dict:
        """Columns of `table` for an event range as numpy arrays (shape [rows] or [rows, k]), plus `event_index`:
        the row's event position in event_table."""
        r0, r1 = self.row_range(table, event_start, event_stop)
        g = self.h5[table]
        out = {}
        for c in (columns or self.columns(table)):
            a = g[c][r0:r1]
            out[c] = a[:, 0] if (a.ndim == 2 and a.shape[1] == 1) else a
        if table == EVENT_TABLE:
            out["event_index"] = np.arange(r0, r1)
        else:
            seq, cnt = self.seq_cnt(table)
            offs = np.concatenate([[0], np.cumsum(cnt)])
            i0, i1 = np.searchsorted(offs, r0), np.searchsorted(offs, r1)
            out["event_index"] = np.repeat(seq[i0:i1], cnt[i0:i1])
        return out

    def counts(self, table: str) -> np.ndarray:
        """Rows of `table` per event, for every event (zero for events absent from the table)."""
        seq, cnt = self.seq_cnt(table)
        n = np.zeros(self.n_events, dtype=np.int64)
        n[seq] = cnt
        return n
