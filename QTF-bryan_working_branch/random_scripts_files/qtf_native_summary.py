#!/usr/bin/env python3
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])

keep = [
    "name",
    "forcefield",
    "chi_mode",
    "total_energy",
    "native_end_to_end",
    "rebuilt_end_to_end",
    "native_rg",
    "rebuilt_rg",
    "term_constraint",
    "term_sasa",
    "term_hbond",
    "term_electrostatics",
    "term_vdw_repulsion",
    "term_rotamer",
    "term_pi_stacking",
    "term_rama",
    "term_geometry",
]

keep = [c for c in keep if c in df.columns]
print(df[keep].to_string(index=False))
