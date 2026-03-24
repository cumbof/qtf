#!/usr/bin/env python3
import json
import pandas as pd
import matplotlib.pyplot as plt

# Inputs
beam_csv = "/home/raubenb/gitrepos/QTF/beam_chig_selective_chi2/beamsearch_ranked.csv"
native_json = "/home/raubenb/gitrepos/QTF/native_scores/chig_amber_selective.json"

# Load beam-search ranked results
df = pd.read_csv(beam_csv)

# Load native scoring result
with open(native_json, "r") as f:
    native = json.load(f)

# If native json is a list with one row, unwrap it
if isinstance(native, list):
    native = native[0]

native_energy = float(native["total_energy"])
native_rebuilt_rmsd = float(native["rebuilt_vs_native_ca_rmsd"])

# Basic cleanup
df = df.copy()
df = df[pd.notnull(df["energy"]) & pd.notnull(df["rmsd_to_reference_A"])]

# Plot 1: Energy vs RMSD
plt.figure(figsize=(7, 5))
plt.scatter(df["rmsd_to_reference_A"], df["energy"], alpha=0.7)
plt.axhline(native_energy, linestyle="--", label=f"Native rebuilt energy = {native_energy:.1f}")
plt.axvline(2.0, linestyle=":", label="Native-like threshold = 2.0 Å")
plt.xlabel("RMSD to experimental structure (Å)")
plt.ylabel("Energy")
plt.title("QTF landscape: Energy vs RMSD (Chignolin)")
plt.legend()
plt.tight_layout()
plt.savefig("energy_vs_rmsd_funnel.png", dpi=200)
plt.close()

# Plot 2: E2E vs RMSD
if "pred_e2e_A" in df.columns:
    plt.figure(figsize=(7, 5))
    plt.scatter(df["rmsd_to_reference_A"], df["pred_e2e_A"], alpha=0.7)
    plt.axhline(5.5086, linestyle="--", label="Experimental E2E")
    plt.axvline(2.0, linestyle=":", label="Native-like threshold = 2.0 Å")
    plt.xlabel("RMSD to experimental structure (Å)")
    plt.ylabel("Predicted end-to-end distance (Å)")
    plt.title("QTF landscape: E2E vs RMSD (Chignolin)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("e2e_vs_rmsd.png", dpi=200)
    plt.close()

# Plot 3: Term correlations in the near-native basin
term_cols = [c for c in df.columns if c.startswith("term_")]
near = df[df["rmsd_to_reference_A"] <= 3.5].copy()

if len(near) >= 5:
    corrs = []
    for c in term_cols + ["energy"]:
        if c in near.columns:
            r = near[[c, "rmsd_to_reference_A"]].corr(method="pearson").iloc[0, 1]
            corrs.append((c, r))
    corr_df = pd.DataFrame(corrs, columns=["feature", "pearson_r"]).sort_values("pearson_r")
    corr_df.to_csv("near_native_term_correlations.csv", index=False)

print("Wrote:")
print("  energy_vs_rmsd_funnel.png")
print("  e2e_vs_rmsd.png")
print("  near_native_term_correlations.csv")
print()
print(f"Native rebuilt CA RMSD: {native_rebuilt_rmsd:.3f} Å")
print(f"Native energy: {native_energy:.3f}")
