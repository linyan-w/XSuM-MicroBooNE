"""Fixtures on the 300-event example file extracted from BNB_All_NoWire_00.h5 (scripts/curate/make_example.py)."""
import pathlib
import pytest

from xsum.io.h5 import Sample

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "tests/data/BNB_All_NoWire_00.first300.h5"


@pytest.fixture(scope="session")
def sample():
    if not EXAMPLE.exists():
        pytest.skip("example file missing")
    return Sample(str(EXAMPLE))
