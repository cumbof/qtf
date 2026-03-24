#!/usr/bin/env python3
import json
import pandas as pd

BEAM_CSV = "/home/raubenb/gitrepos/QTF/beam_search_results/beam_chig_selective_hbond0.6_sasa0.8_ljattr0.15/beamsearch_ranked.csv"
NATIVE_JSON = "/home/raubenb/gitrepos/QTF/native_scores/chig_amber_selective_hbond0.6_scalesasa0.8_ljattract0.15.json"

TERM_COLS = [
    "term_sasa",
    "term_vdw_repulsion",
    "term_hbond",
    "term_pi_stacking",
    "term_electrostatics",
    "term_rotamer",
    "term_constraint",
    "term_rama",
    "term_geometry",
]

with open(NATIVE_JSON, "r") as f:
    native = json.load(f)
if isinstance(native, list):
    native = native[0]

df = pd.read_csv(BEAM_CSV).copy()
df_r = df[pd.notnull(df["rmsd_to_reference_A"])].copy()

best_energy = df.sort_values("energy", ascending=True).iloc[0]
best_rmsd = df_r.sort_values("rmsd_to_reference_A", ascending=True).iloc[0]

rows = []

native_row = {
    "label": "rebuilt_native",
    "energy_rank": None,
    "energy": float(native["total_energy"]),
    "rmsd_to_reference_A": float(native["rebuilt_vs_native_ca_rmsd"]),
    "pred_e2e_A": float(native["rebuilt_end_to_end"]),
    "pred_rg_A": float(native["rebuilt_rg"]),
}
for c in TERM_COLS:
    native_row[c] = float(native[c]) if c in native else None
rows.append(native_row)

for label, row in [("best_energy_beam", best_energy), ("best_rmsd_beam", best_rmsd)]:
    r = {
        "label": label,
        "energy_rank": int(row["energy_rank"]) if pd.notnull(row.get("energy_rank")) else None,
        "energy": float(row["energy"]),
        "rmsd_to_reference_A": float(row["rmsd_to_reference_A"]) if pd.notnull(row.get("rmsd_to_reference_A")) else None,
        "pred_e2e_A": float(row["pred_e2e_A"]) if pd.notnull(row.get("pred_e2e_A")) else None,
        "pred_rg_A": float(row["pred_rg_A"]) if pd.notnull(row.get("pred_rg_A")) else None,
    }
    for c in TERM_COLS:
        r[c] = float(row[c]) if c in row and pd.notnull(row[c]) else None
    rows.append(r)

out = pd.DataFrame(rows)

keep = ["label", "energy_rank", "energy", "rmsd_to_reference_A", "pred_e2e_A", "pred_rg_A"] + TERM_COLS
out = out[keep]

print(out.to_string(index=False))
out.to_csv("native_vs_beam_comparison.csv", index=False)