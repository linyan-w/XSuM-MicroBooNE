# The sample and how the surrogate reads it

## The open sample

MicroBooNE's *BNB Inclusive Overlay Sample (No Wire Info)*, Zenodo record
[10.5281/zenodo.8370883](https://doi.org/10.5281/zenodo.8370883) (version "large", 2023-09-26, CC BY 4.0): simulated
Booster Neutrino Beam interactions of every flavour and interaction type, with the relative abundances of the nominal flux
and cross-section models, generated in the whole cryostat and overlaid on beam-off data of the detector (cosmic rays and
noise). Documentation and example notebooks: [uboone/OpenSamples v02](https://github.com/uboone/OpenSamples/tree/v02);
the column-by-column content is copied in [`h5_content.md`](h5_content.md), what the files actually contain is measured
in [`h5_notes.md`](h5_notes.md). The record holds 18 HDF5 files of 10.3-11.1 GB (195 GB, 753,467 events; about 42.7k
events per file) and a list of xrootd URLs of the artroot files; [`../configs/zenodo_8370883_files.txt`](../configs/zenodo_8370883_files.txt)
has the names, sizes, md5s and URLs, `scripts/curate/fetch_zenodo.py` downloads and verifies them one at a time.

The files carry no generator-level record beyond the event table (flavour, energy, direction, CC/NC, lepton energy,
vertex) and no exposure (POT). The generator (GENIE, MicroBooNE's MCC9 tune), the flux and the POT are not in the files
or the record; they are in the artroot files and the collaboration's documentation and must be taken from there.

## What one event contains

One HDF5 "event" is one detector readout with one simulated neutrino interaction (plus the overlaid cosmic activity).
The tables are flat columns with one row per object; `event_id.seq_cnt` groups rows by event (`xsum.io.h5.Sample`).

| side | table | used as |
|---|---|---|
| truth | `event_table` | the interaction: `nu_pdg`, `nu_energy`, `nu_dir`, `is_cc`, `lep_energy`, `nu_vtx` (true) and `nu_vtx_corr` (space-charge corrected, the one to compare with the reconstruction) |
| truth | `particle_table`, rows with `start_process == "primary"` | the final state handed to Geant4: the surrogate input (see below) |
| truth | `edep_table` | energy deposits per hit and Geant4 particle; the hit-level ground truth, not needed by the surrogate |
| reco | `pandoraPrimary_table` | every Pandora slice of the readout (about 30 per event, almost all cosmic, `slice_pdg = 13`); the one with `slice_pdg` 12 or 14 is Pandora's neutrino candidate, at most one per event, with its vertex, `nu_score` and flash-match score |
| reco | `pandoraPfp_table` | the PFParticles of the neutrino candidate: track/shower label (`pfp_pdg` 13/11), `trkshr_score`, start point |
| reco | `pandoraHit_table`, `hit_table` | hit-to-PFP links and the hits (integral, plane, wire, time); about 16k hits per event. The per-PFP charge, hit counts and extents that stand in for track momentum and calorimetry have to be derived from these two tables (not yet done) |
| reco | `opflash_table`, `opflashsumpe_table`, `ophit_table` | the optical flashes; the beam-window flash (3-5 us) and its PE and barycentre |

There are no track fits, no calorimetric energies, no particle-identification scores and no event-level kinematics in
the HDF5 files: the reconstruction is Pandora's pattern recognition (slices, PFParticles, hit clustering) and nothing
downstream of it. The surrogate's output is therefore the Pandora level: whether a neutrino slice is found, its vertex
and scores, the number of PFParticles, and per PFParticle the track/shower score, the start point and the hit-derived
charge and extent.

## Differences from the MINERvA surrogate that shape the design

| | MINERvA (XSuM-MINERvA) | MicroBooNE (this repository) |
|---|---|---|
| input particles | `mc_FSPart*`: species and three-momentum of every post-FSI particle | Geant4 primaries: species, momentum magnitude, start and end position. **No direction is stored**; the start-to-end displacement is the proxy (`xsum.prep.particles.direction`), exact for particles that stop or leave without scattering, approximate otherwise (1.8% of primaries have no displacement at all) |
| vertex | `mc_vtx` (mm) | `nu_vtx` / `nu_vtx_corr` (cm); space-charge shifts the corrected x by about 1 cm (p50) and up to 5-11 cm |
| population | CC nu_mu (later all events) in the inner detector, z 4000-8700 mm | every flavour; the active volume holds 42% of the generated vertices, the rest is cryostat and never reconstructed as a neutrino slice. Flavour mix: 98.1% nu_mu, 1.1% nu_e, 0.7% anti-nu_mu; 71% CC |
| "reconstructed" flag | a muon candidate exists (32% of Truth events) | a neutrino slice exists (39% of events; 66% inside the active volume, 20% outside: those are cosmic slices mislabelled as the neutrino, with vertices far from the truth) |
| reco vertex | `MasterAnaDev_vtx`, 10 mm (x, y) / 40 mm (z) resolution | neutrino-slice `vtx`: 0.2-0.4 cm per axis at p50 inside the active volume, with a long tail of wrong slices (|d| p90 = 123 cm) |
| muon | momentum, angle, MINOS match, charge | nothing muon-specific: PFParticles with a track score; momentum only by range or multiple scattering, neither in the files |
| calorimetry | three recoil energies | hit integrals (ADC) per PFParticle per plane, to be derived from the hit tables |
| prongs | up to 10 hadron prongs with pion and proton fits | 1-10 PFParticles (p50 2, p90 4) with `trkshr_score` and start point |
| detector conditions | overlay dead time | overlay cosmics: about 30 cosmic slices per readout, the beam flash |
| exposure | `Meta/POT_Used` per file | not in the files |
| units | MeV, mm | GeV, cm, ticks |

## The surrogate's input and output (draft)

**Input**, one event: `nu_vtx_corr` (3), `nu_pdg`, `is_cc`, `nu_energy`, and the set of primaries above 10 MeV kinetic
energy with the outgoing neutrino of NC events dropped: class (`xsum.constants.CLASS_NAMES`), kinetic energy, direction
proxy. Neutrons are kept (they are the most common primary) and so are the primaries outside the active volume. The slim
truth file (`scripts/curate/slim.py`) holds exactly this plus the event identifiers and an `in_active_volume` flag.

**Output**, one event:

1. `has_nu_slice` (Bernoulli): Pandora selected a neutrino slice.
2. Given a slice: `vtx` (3), `nu_score`, `flashmatch_score`, `slice_pdg` (12/11-like vs 14), the beam-flash PE and
   y/z barycentre, the number of PFParticles.
3. Per PFParticle: `trkshr_score`, start point, and the hit-derived summary (charge per plane, hit count, extent along
   the direction, end point) once the hit tables are curated.

The slim reco file holds 1 and 2 and the PFParticle columns of the HDF5 file; the hit-derived columns are the next
curation step. The model structure can then follow XSuM-MINERvA: set encoder on the primaries with the vertex and
flavour in the global token, Bernoulli heads for the flags, a multiplicity head for the number of PFParticles, an event
flow for the slice-level continuous quantities and a prong set flow for the PFParticles.

## Open questions

- **Direction of the primaries.** Whether the start-to-end proxy is good enough, or whether the artroot `MCTruth` /
  `MCParticle` momenta must be extracted once (one pass over the artroot files via xrootd) and distributed as a
  companion truth file. The artroot file list is in the record.
- **Exposure.** A POT per file (or per event weight) is needed for any cross-section use; it is not in the HDF5 files.
- **Hit-level targets.** Which per-PFParticle quantities to derive from the 674 M hit links per file (streaming, one file
  at a time), and the disk budget for doing so: the machine has about 30 GB free, one file is 11 GB.
- **The 20% of out-of-volume events with a "neutrino" slice.** They are cosmic slices; the surrogate must either model
  them (a wrong-slice class with a vertex drawn from the cosmic distribution) or be restricted to the active volume.
- **Split.** Events are grouped by run and subrun (120 subruns in the first 2000 events of one file); split by subrun
  as in MINERvA so that overlay-correlated readouts stay together.
