"""The tables and columns the surrogate work reads from the NoWire files (docs/h5_content.md); hit-level tables are
read only by the curation step that derives per-PFP charge (not yet written)."""

EVENT_COLUMNS = ["event_id", "is_cc", "lep_energy", "nu_dir", "nu_energy", "nu_pdg", "nu_vtx", "nu_vtx_corr",
                 "nu_vtx_wire_pos", "nu_vtx_wire_time"]
PARTICLE_COLUMNS = ["event_id", "g4_id", "g4_pdg", "parent_id", "momentum", "category", "instance",
                    "start_process", "end_process", "start_position", "end_position",
                    "start_position_corr", "end_position_corr", "start_wire_time", "end_wire_time"]
PRIMARY_COLUMNS = ["event_id", "slice_id", "slice_pdg", "nu_score", "flashmatch_score", "vtx", "vtx_wire_pos", "vtx_wire_time"]
PFP_COLUMNS = ["event_id", "pfp_id", "pfp_pdg", "trkshr_score", "vtx", "vtx_wire_pos", "vtx_wire_time"]
OPFLASH_COLUMNS = ["event_id", "flash_id", "time", "time_width", "totalpe", "y_center", "y_width", "z_center", "z_width"]

SLIM_TABLES = {
    "event_table": EVENT_COLUMNS,
    "particle_table": PARTICLE_COLUMNS,
    "pandoraPrimary_table": PRIMARY_COLUMNS,
    "pandoraPfp_table": PFP_COLUMNS,
    "opflash_table": OPFLASH_COLUMNS,
}
