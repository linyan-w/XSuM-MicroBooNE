# XSuM-MicroBooNE

A surrogate detector model for the [MicroBooNE open samples](https://microboone.fnal.gov/documents-publications/public-datasets/),
built the way [XSuM-MINERvA](https://github.com/linyan-w/XSuM-MINERvA) was built for the MINERvA open dataset: a
generative model that maps the final-state particles of a simulated neutrino interaction to the reconstructed
quantities of the open sample, trained on the sample itself, so that events from any generator can be run through it.
The target sample is the **BNB inclusive overlay sample without wire information**
([10.5281/zenodo.8370883](https://doi.org/10.5281/zenodo.8370883), 18 HDF5 files, 195 GB, 753,467 events).

State of the work: the sample is characterised, file 00 is on disk, slimmed at event level (Geant4 primaries; Pandora's
neutrino slice, PFParticles and beam flash) and at hit level (per-PFParticle hits, charge and spans per plane), and
both stages of the model (set encoder, flag heads, multiplicity head, event flow; PFParticle set flow) are written,
tested and trained on it. The decoder and output file, the ensemble and the generator-input path are not written. The design, the measurements and the status are in the technote
[`docs/technote/technote.pdf`](docs/technote/technote.pdf).

## Layout

| path | content |
|---|---|
| `xsum/` | the package: `io/h5.py` reads the open-sample files (local or straight from Zenodo over HTTP ranges), `io/tables.py` lists the columns used, `prep/` selects the primaries and defines the detector volume, `data/` the dataset over the slim files, the event-vector transform and the torch view, `models/` the set encoder, the flow matcher and the surrogate, `train/` the event-stage loop and its evaluation, `constants.py` the conventions |
| `scripts/curate/` | `fetch_zenodo.py` (download with md5 and a disk reserve), `h5_summary.py` (structure and statistics of a file), `slim.py` (HDF5 to truth/reco Parquet, streamed in event chunks), `slim_hits.py` (per-PFParticle hit summaries from the hit tables), `make_example.py` (the test file), `pipeline_file00.sh` (file 00 end to end) |
| `scripts/train/` | `train_event.py` (the event stage) and `train_pfp.py` (the PFParticle set flow on top of it), each with its evaluation |
| `configs/` | the Zenodo file manifest (names, sizes, md5s, URLs) |
| `docs/technote/` | the technote (`technote.tex`, `technote.pdf`; `build.sh` rebuilds it): the sample, what the files contain and what was measured on them, the surrogate's input and output, the differences from MINERvA, the schematic of the chain with the status of every step, open questions; the collaboration's column documentation (`h5_content.md`) becomes its appendix |
| `tests/` | reader and preparation tests on a 300-event extract of the first file (`tests/data/`, 2.3 MB) |

## Quick start

```bash
bash scripts/setup_env.sh                      # Python >= 3.9 venv with h5py, fsspec, awkward, torch (CUDA), ...
source .venv/bin/activate
pytest -q

# look at a file without downloading it (metadata and the first chunks, a few MB)
python scripts/curate/h5_summary.py "https://zenodo.org/api/records/8370883/files/BNB_All_NoWire_00.h5/content" --events 2000

# fetch one file (11 GB; the script keeps --reserve-gb free and verifies the md5), then slim it
python scripts/curate/fetch_zenodo.py data/h5 --files 00 --reserve-gb 20
python scripts/curate/slim.py data/h5/BNB_All_NoWire_00.h5 data/slim/BNB_All_NoWire_00

# or slim straight from Zenodo (every column of the event-level tables streams in; hit tables are not read)
python scripts/curate/slim.py "https://zenodo.org/api/records/8370883/files/BNB_All_NoWire_00.h5/content" data/slim/BNB_All_NoWire_00 --events 5000

# train the event stage on the slim files and evaluate on the test split (metrics.json, figures/)
python scripts/train/train_event.py reports/event00 --stems data/slim/BNB_All_NoWire_00 --epochs 30
# hit-level slim (per-PFParticle hits, charge, spans per plane), then the PFParticle set flow on top, event stage frozen
python scripts/curate/slim_hits.py data/h5/BNB_All_NoWire_00.h5 data/slim/BNB_All_NoWire_00
python scripts/train/train_pfp.py reports/pfp00 --init reports/event00/model.pt --stems data/slim/BNB_All_NoWire_00 --epochs 120 --only-pfp
```

The tests and the training need torch; on dune-gpu01 the XSuM-MINERvA environment is reused
(`/home/lwan/AI/Diffusion/.venv/bin/python`, with h5py and fsspec added) instead of a second 6 GB venv.

## What the files give the surrogate

Input: the Geant4 primaries of each event (species, momentum magnitude, start and end position; the direction is not
stored and is taken from the displacement), the true and space-charge-corrected vertex, flavour, CC/NC and neutrino
energy. Output: whether Pandora found a neutrino slice (39% of events, 66% of those with the vertex in the active
volume), the slice vertex and scores, the beam flash, and the PFParticles of the slice (track/shower score, start
point). Per-PFParticle charge and extent must be derived from the hit tables (674 M hit links per file); there are no
track fits, calorimetric energies or particle-identification scores in the HDF5 files, and no exposure. Details and
open questions in the technote, [`docs/technote/technote.pdf`](docs/technote/technote.pdf).

## License

Copyright 2026 Fermi Forward Discovery Group, LLC. The code in this repository is released under the Apache License,
Version 2.0 (see `LICENSE`). Fermilab is operated by Fermi Forward Discovery Group, LLC under Contract No.
89243024CSC000002 with the U.S. Department of Energy.

The data are the MicroBooNE Collaboration's, released under CC BY 4.0. Suggested acknowledgment (from the record):
*We acknowledge the MicroBooNE Collaboration for making publicly available the data sets 10.5281/zenodo.8370883 employed
in this work. These data sets consist of simulated neutrino interactions from the Booster Neutrino Beamline overlaid on
top of cosmic data collected with the MicroBooNE detector [2017 JINST 12 P02017].*
