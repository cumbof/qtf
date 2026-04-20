#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("grid_analysis/grid_v3_rot_pi_largegrid")
BEAM_CSV = ROOT / "master_beam_rows.csv"
NATIVE_CSV = ROOT / "master_native_rows.csv"
SUMMARY_CSV = ROOT / "master_experiment_summary.csv"
OUTDIR = ROOT / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

GOOD_REBUILD_THRESH = 1.5

beam = pd.read_csv(BEAM_CSV)
native = pd.read_csv(NATIVE_CSV)
summary = pd.read_csv(SUMMARY_CSV)

# Normalize reference PDB columns if present
for df in (beam, native, summary):
    if "reference_pdb" in df.columns:
        df["reference_pdb_path"] = df["reference_pdb"]
        df["reference_pdb_id"] = df["reference_pdb"].astype(str).map(lambda s: Path(s).stem.upper())
    elif "reference_pdb_path" in df.columns:
        df["reference_pdb_id"] = df["reference_pdb_path"].astype(str).map(lambda s: Path(s).stem.upper())

# 1) rebuild quality summary
rebuild_cols = [c for c in ["protein_name", "reference_pdb_id", "native_rebuilt_rmsd"] if c in summary.columns]
if rebuild_cols:
    rebuild = summary[rebuild_cols].copy().drop_duplicates()
    rebuild.to_csv(OUTDIR / "native_rebuild_quality.csv", index=False)

    plt.figure(figsize=(8, 4.5))
    plt.bar(rebuild["protein_name"], rebuild["native_rebuilt_rmsd"])
    plt.axhline(GOOD_REBUILD_THRESH, linestyle="--")
    plt.ylabel("Native rebuild RMSD (Å)")
    plt.title("Rebuild quality by protein")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTDIR / "native_rebuild_quality.png", dpi=200)
    plt.close()

# 2) funnels per protein
for protein, g in beam.groupby("protein_name"):
    if "rmsd_to_reference_A" not in g.columns:
        continue
    g = g[pd.notnull(g["rmsd_to_reference_A"]) & pd.notnull(g["energy"])].copy()
    if g.empty:
        continue

    plt.figure(figsize=(6.5, 5))
    plt.scatter(g["rmsd_to_reference_A"], g["energy"], alpha=0.35)
    plt.xlabel("RMSD to reference (Å)")
    plt.ylabel("Energy")
    plt.title(f"Energy funnel — {protein}")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"funnel_{protein}.png", dpi=200)
    plt.close()

# 3) term correlations overall and filtered by rebuild quality
term_cols = [c for c in beam.columns if c.startswith("term_")]
if "energy" in beam.columns:
    term_cols = term_cols + ["energy"]

if "rmsd_to_reference_A" in beam.columns and term_cols:
    corr_rows = []
    for c in term_cols:
        sub = beam[[c, "rmsd_to_reference_A"]].dropna()
        if len(sub) < 3:
            continue
        corr_rows.append({
            "feature": c,
            "pearson_r": sub[c].corr(sub["rmsd_to_reference_A"], method="pearson"),
            "spearman_r": sub[c].corr(sub["rmsd_to_reference_A"], method="spearman"),
        })
    corr_df = pd.DataFrame(corr_rows).sort_values("spearman_r")
    corr_df.to_csv(OUTDIR / "term_correlations_overall.csv", index=False)

    plt.figure(figsize=(8, max(4, 0.28 * len(corr_df))))
    plt.barh(corr_df["feature"], corr_df["spearman_r"])
    plt.xlabel("Spearman correlation with RMSD")
    plt.title("Term correlations vs RMSD (all panel beam rows)")
    plt.tight_layout()
    plt.savefig(OUTDIR / "term_correlations_overall.png", dpi=200)
    plt.close()

# 4) good-rebuild subset
good_proteins = set()
if "native_rebuilt_rmsd" in summary.columns:
    good_proteins = set(summary.loc[summary["native_rebuilt_rmsd"] <= GOOD_REBUILD_THRESH, "protein_name"].dropna().tolist())

if good_proteins:
    beam_good = beam[beam["protein_name"].isin(good_proteins)].copy()
    corr_rows = []
    for c in term_cols:
        sub = beam_good[[c, "rmsd_to_reference_A"]].dropna()
        if len(sub) < 3:
            continue
        corr_rows.append({
            "feature": c,
            "pearson_r": sub[c].corr(sub["rmsd_to_reference_A"], method="pearson"),
            "spearman_r": sub[c].corr(sub["rmsd_to_reference_A"], method="spearman"),
        })
    corr_good = pd.DataFrame(corr_rows).sort_values("spearman_r")
    corr_good.to_csv(OUTDIR / "term_correlations_good_rebuild_only.csv", index=False)

    plt.figure(figsize=(8, max(4, 0.28 * len(corr_good))))
    plt.barh(corr_good["feature"], corr_good["spearman_r"])
    plt.xlabel("Spearman correlation with RMSD")
    plt.title("Term correlations vs RMSD (good native rebuilds only)")
    plt.tight_layout()
    plt.savefig(OUTDIR / "term_correlations_good_rebuild_only.png", dpi=200)
    plt.close()

# 5) ranking gap plot
if {"protein_name", "best_rmsd", "best_energy_rmsd"}.issubset(summary.columns):
    keep = [
        "protein_name",
        "protein_label",
        "reference_pdb_id",
        "hbond_scale",
        "sasa_scale",
        "vdw_rep_scale",
        "vdw_attr_scale",
        "rotamer_scale",
        "pi_stack_scale",
        "best_rmsd",
        "best_energy_rmsd",
    ]
    keep = [c for c in keep if c in summary.columns]
    gap_df = summary[keep].copy()
    gap_df["ranking_gap_A"] = gap_df["best_energy_rmsd"] - gap_df["best_rmsd"]
    gap_df.to_csv(OUTDIR / "ranking_gap_summary.csv", index=False)

    plt.figure(figsize=(8, 4.5))
    label_col = "protein_label" if "protein_label" in rebuild.columns else "protein_name"
    plt.bar(rebuild[label_col], rebuild["native_rebuilt_rmsd"])
    plt.ylabel("Best-energy RMSD − Best RMSD (Å)")
    plt.title("Ranking gap by protein")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTDIR / "ranking_gap_by_protein.png", dpi=200)
    plt.close()

print(f"Wrote analysis plots to {OUTDIR.resolve()}")
