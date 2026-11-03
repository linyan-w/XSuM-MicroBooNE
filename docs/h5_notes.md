# Measured content of `BNB_All_NoWire_00.h5`

Source `https://zenodo.org/api/records/8370883/files/BNB_All_NoWire_00.h5/content`, read 2026-11-03 with `scripts/curate/h5_summary.py --events 2000`.

## Tables

| table | rows | events with rows | columns |
|---|---|---|---|
| `edep_table` | 40,439,430 | 35,430 | energy, energy_fraction, event_id[3], g4_id, hit_id |
| `event_table` | 42,674 | 42,674 | event_id[3], is_cc, lep_energy, nu_dir[3], nu_energy, nu_pdg, nu_vtx[3], nu_vtx_corr[3], nu_vtx_wire_pos[3], nu_vtx_wire_time |
| `hit_table` | 674,335,781 | 42,674 | event_id[3], hit_id, integral, local_plane, local_time, local_wire, rms, tpc |
| `opflash_table` | 42,059 | 36,900 | event_id[3], flash_id, time, time_width, totalpe, wire_pos[3], y_center, y_width, z_center, z_width |
| `opflashsumpe_table` | 1,027,947 | 36,900 | event_id[3], flash_id, pmt_channel, sumpe, sumpe_id |
| `ophit_table` | 15,317,703 | 42,674 | amplitude, area, event_id[3], hit_channel, hit_id, pe, peaktime, sumpe_id, width, wire_pos[3] |
| `pandoraHit_table` | 674,335,781 | 42,674 | event_id[3], hit_id, pfp_id, slice_id |
| `pandoraPfp_table` | 45,068 | 18,375 | event_id[3], pfp_id, pfp_pdg, trkshr_score, vtx[3], vtx_wire_pos[3], vtx_wire_time |
| `pandoraPrimary_table` | 1,352,085 | 42,674 | event_id[3], flashmatch_score, nu_score, slice_id, slice_pdg, vtx[3], vtx_wire_pos[3], vtx_wire_time |
| `particle_table` | 4,011,145 | 35,430 | category, end_position[3], end_position_corr[3], end_process, end_wire_pos[3], end_wire_time, event_id[3], g4_id, g4_pdg, instance, momentum, parent_id, start_position[3], start_position_corr[3], start_process, start_wire_pos[3], start_wire_time |

Events in the file: 42,674. Statistics below on the first 2,000.

## Event table

- neutrino flavour: -14: 0.007, -12: 0.001, 12: 0.011, 14: 0.981
- charged current: 0.709
- neutrino energy (GeV) p50/90/99: 1.08526/1.76068/4.2295; lepton energy p50/90/99: 0.563839/1.21863/2.82237
- true vertex inside the active volume: 0.419 (the sample is generated in the whole cryostat)
- vertex x (cm) min/p1/p50/p99/max: -63/-52.9909/132.67/309.039/319
- vertex y (cm) min/p1/p50/p99/max: -191/-180.968/-2.34008/165.251/191
- vertex z (cm) min/p1/p50/p99/max: -82/-63.5997/508.025/1089.7/1118
- space-charge correction (corr - true, cm), p50/p99 of |d|: x 1.12164/4.55501, y 0.0255394/11.014, z 0.00261688/3.51344
- run/subrun: 29 runs, 120 subruns in the first 2000 events

## Particle table

- rows per event p50/90/99: 40/224/717.13; primaries per event p50/90/99: 2/6/22
- primaries have parent_id values [0]; non-primary start processes: CoulombScat, Decay, annihil, compt, conv, dInelastic, eBrem, eIoni, electronNuclear, hBertiniCaptureAtRest, hIoni, hadElastic
- primary PDG counts: 2112: 2406, 2212: 2166, 13: 850, 211: 243, 111: 221, -211: 103, 11: 14, 22: 7, -13: 4, 321: 2, -321: 1, 1000180400: 1
- primaries above 10 MeV kinetic energy (neutrinos dropped) per event p50/90/99/max: 2/5/14.01/22
- model classes: mu-: 847, mu+: 4, e-: 14, gamma: 6, p: 1702, n: 1864, pi+: 241, pi-: 101, pi0: 217, K+-: 3, other: 1
- categories (all rows): pion: 2924, muon: 38899, proton: 6665, electron: 10056, michel: 2549, delta: 4637, other: 77813, photon: 41827
- primaries whose start and end position coincide (no direction proxy): 0.0184

## Pandora

- slices per event p50/90/99: 30/43/62; slice_pdg counts: {11: 721, 12: 79, 13: 61647, 14: 705}
- slices with a nu_score: 0.411; nu_score p50/90/99 among them: 0.0279327/0.341043/0.987047
- events with a neutrino slice (slice_pdg 12 or 14; never more than one per event): 0.392; among events with the true vertex in the active volume: 0.657; outside: 0.201
- rows with the undocumented slice_pdg 11: 721 (0.88 of them with vtx = -999); they are not the neutrino candidate and are ignored
- nu_score of the neutrino slice p10/50/90: 0.010735/0.324803/0.988651; flash-match score p50/90: 5.14938/52.0537
- slices with a flash-match score: 0.436; score p50/90/99: 45.4947/200.723/423.711
- neutrino-slice vertex minus nu_vtx_corr (cm), |d| p50/90/99: 4.55622/196.72/335.602 (all); 0.732969/123.174/312.099 (true vertex in the active volume); per axis p50 of |d| in the active volume: x 0.2, y 0.4, z 0.4
- PFParticles of the neutrino slice per event (events with a slice) p50/90/99/max: 2/4/6/10; events with a slice but no PFP: 0.000
- pfp_pdg counts: {11: 355, 13: 1482}; trkshr_score p10/50/90: 0.0596302/0.929463/1

## Optical flashes

- flashes per event p50/90/99: 1/2/2; events without a flash: 0.151
- flash time (us) p1/50/99: 2.56969/4.54625/22.6216; total PE p50/90/99: 697.068/4152.81/9413.28

Read in 84 s.
