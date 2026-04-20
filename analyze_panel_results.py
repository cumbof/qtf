#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

PARAM_COLS: List[str] = [
    "hbond_scale", "sasa_scale", "vdw_rep_scale", "vdw_attr_scale", "rotamer_scale", "pi_stack_scale",
]
RUN_KEY_COLS: List[str] = [
    "protein_name", "reference_pdb_id", "reference_pdb_path", "sequence", "forcefield", "chi_mode",
    "window_deg", "step_deg", *PARAM_COLS,
]
GOOD_REBUILD_THRESH = 1.5


def summarize_beam_group(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values(["energy", "energy_rank"], kind="stable").reset_index(drop=True)
    best_energy_row = g.iloc[0]
    best_rmsd_idx = g["rmsd_to_reference_A"].astype(float).idxmin()
    best_rmsd_row = g.loc[best_rmsd_idx]
    native_like_count = int(g["native_like"].fillna(False).astype(bool).sum()) if "native_like" in g.columns else 0
    return pd.Series({
        "n_ranked_rows": int(len(g)),
        "best_energy": float(best_energy_row["energy"]),
        "best_energy_rmsd": float(best_energy_row["rmsd_to_reference_A"]),
        "best_energy_rank": int(best_energy_row["energy_rank"]) if "energy_rank" in best_energy_row else 1,
        "best_rmsd": float(best_rmsd_row["rmsd_to_reference_A"]),
        "best_rmsd_energy": float(best_rmsd_row["energy"]),
        "best_rmsd_rank": int(best_rmsd_row["energy_rank"]) if "energy_rank" in best_rmsd_row else -1,
        "ranking_gap": float(best_energy_row["rmsd_to_reference_A"] - best_rmsd_row["rmsd_to_reference_A"]),
        "native_like_count": native_like_count,
        "native_like_fraction": float(native_like_count / len(g)) if len(g) else 0.0,
        "mean_rmsd_topk": float(g["rmsd_to_reference_A"].mean()),
        "median_rmsd_topk": float(g["rmsd_to_reference_A"].median()),
        "mean_energy_topk": float(g["energy"].mean()),
    })


def build_corrected_summary(beam_csv: Path, native_csv: Path, manifest_csv: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    beam = pd.read_csv(beam_csv)
    native = pd.read_csv(native_csv)
    manifest = pd.read_csv(manifest_csv)
    if "status" in manifest.columns:
        manifest = manifest.loc[manifest["status"].fillna("").eq("ok")].copy()

    # Ensure window/step exist on raw tables; if missing, pull from manifest by experiment_id
    for raw in (beam, native):
        for col in ("window_deg", "step_deg"):
            if col not in raw.columns:
                raw[col] = pd.NA

    if {"experiment_id", "window_deg", "step_deg"}.issubset(manifest.columns):
        lookup = manifest[["experiment_id", "window_deg", "step_deg"]].drop_duplicates()
        for raw_name, raw in [("beam", beam), ("native", native)]:
            if "experiment_id" in raw.columns:
                raw2 = raw.merge(lookup, on="experiment_id", how="left", suffixes=("", "_mf"))
                for col in ("window_deg", "step_deg"):
                    raw2[col] = raw2[col].fillna(raw2[f"{col}_mf"])
                    raw2 = raw2.drop(columns=[f"{col}_mf"])
                if raw_name == "beam":
                    beam = raw2
                else:
                    native = raw2

    beam_summary = (
        beam.groupby(RUN_KEY_COLS, dropna=False)
        .apply(summarize_beam_group, include_groups=False)
        .reset_index()
    )

    native_keep_cols = [
        c for c in native.columns if c in RUN_KEY_COLS or c in {
            "name", "pdb_path", "chain", "residue_start", "residue_end", "n_residues", "total_energy",
            "rebuilt_vs_native_ca_rmsd", "native_end_to_end", "native_rg", "rebuilt_end_to_end", "rebuilt_rg",
            "rebuilt_ca_pdb_path", "rebuilt_ca_centroid_pdb_path"
        } or c.startswith("term_")
    ]
    native2 = native[native_keep_cols].copy().rename(columns={
        "total_energy": "native_total_energy",
        "rebuilt_vs_native_ca_rmsd": "native_rebuild_ca_rmsd",
        "rebuilt_ca_pdb_path": "native_rebuilt_ca_pdb_path",
        "rebuilt_ca_centroid_pdb_path": "native_rebuilt_ca_centroid_pdb_path",
    })
    native2 = native2.drop_duplicates(subset=[c for c in RUN_KEY_COLS if c in native2.columns], keep="first")

    merged = beam_summary.merge(
        native2,
        on=[c for c in RUN_KEY_COLS if c in native2.columns],
        how="left",
        validate="one_to_one",
    )

    manifest_keep = [c for c in manifest.columns if c in RUN_KEY_COLS or c in {"experiment_id", "run_dir", "status", "error"}]
    merged = merged.merge(
        manifest[manifest_keep].drop_duplicates(),
        on=[c for c in RUN_KEY_COLS if c in manifest_keep],
        how="left",
        validate="one_to_one",
    )

    merged["native_beats_best_energy"] = merged["native_total_energy"] < merged["best_energy"]
    merged["native_beats_best_rmsd_energy"] = merged["native_total_energy"] < merged["best_rmsd_energy"]
    merged.to_csv(outdir / "master_experiment_summary_corrected.csv", index=False)

    protein_summary = (
        merged.groupby("protein_name", dropna=False)
        .agg(
            n_runs=("protein_name", "size"),
            mean_best_rmsd=("best_rmsd", "mean"),
            median_best_rmsd=("best_rmsd", "median"),
            min_best_rmsd=("best_rmsd", "min"),
            mean_best_energy_rmsd=("best_energy_rmsd", "mean"),
            median_best_energy_rmsd=("best_energy_rmsd", "median"),
            mean_ranking_gap=("ranking_gap", "mean"),
            median_ranking_gap=("ranking_gap", "median"),
            min_ranking_gap=("ranking_gap", "min"),
            total_native_like=("native_like_count", "sum"),
            mean_native_rebuild_ca_rmsd=("native_rebuild_ca_rmsd", "mean"),
        )
        .reset_index()
    )
    protein_summary.to_csv(outdir / "protein_level_summary_corrected.csv", index=False)
    return beam, native, merged


def _bar(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, outpath: Path):
    if df.empty:
        return
    plt.figure(figsize=(8, 4.8))
    plt.bar(df[x].astype(str), df[y].astype(float))
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _save_term_correlations(beam: pd.DataFrame, summary: pd.DataFrame, outdir: Path):
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
        corr_df.to_csv(outdir / "term_correlations_overall.csv", index=False)

        if not corr_df.empty:
            plt.figure(figsize=(8, max(4, 0.28 * len(corr_df))))
            plt.barh(corr_df["feature"], corr_df["spearman_r"])
            plt.xlabel("Spearman correlation with RMSD")
            plt.title("Term correlations vs RMSD (all beam rows)")
            plt.tight_layout()
            plt.savefig(outdir / "term_correlations_overall.png", dpi=200)
            plt.close()

    good_proteins = set()
    if "native_rebuild_ca_rmsd" in summary.columns:
        good_proteins = set(
            summary.loc[summary["native_rebuild_ca_rmsd"] <= GOOD_REBUILD_THRESH, "protein_name"].dropna().tolist()
        )

    if good_proteins and "rmsd_to_reference_A" in beam.columns and term_cols:
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
        corr_good.to_csv(outdir / "term_correlations_good_rebuild_only.csv", index=False)

        if not corr_good.empty:
            plt.figure(figsize=(8, max(4, 0.28 * len(corr_good))))
            plt.barh(corr_good["feature"], corr_good["spearman_r"])
            plt.xlabel("Spearman correlation with RMSD")
            plt.title("Term correlations vs RMSD (good native rebuilds only)")
            plt.tight_layout()
            plt.savefig(outdir / "term_correlations_good_rebuild_only.png", dpi=200)
            plt.close()



def save_ranked_parameter_sets(summary: pd.DataFrame, outdir: Path):
    """
    Recycle the old rank_grid_search_results.py logic, but operate on the
    corrected summary dataframe produced in this script.
    """
    needed = {"best_rmsd", "best_energy_rmsd", *PARAM_COLS}
    if summary.empty or not needed.issubset(summary.columns):
        return

    df = summary.copy()
    if "ranking_gap" not in df.columns:
        df["ranking_gap"] = df["best_energy_rmsd"] - df["best_rmsd"]

    grouped = df.groupby(PARAM_COLS, dropna=False).agg(
        n_proteins=("protein_name", "nunique"),
        mean_best_rmsd=("best_rmsd", "mean"),
        median_best_rmsd=("best_rmsd", "median"),
        mean_best_energy_rmsd=("best_energy_rmsd", "mean"),
        median_best_energy_rmsd=("best_energy_rmsd", "median"),
        mean_ranking_gap=("ranking_gap", "mean"),
        median_ranking_gap=("ranking_gap", "median"),
        mean_native_rebuild=("native_rebuild_ca_rmsd", "mean"),
        min_best_rmsd=("best_rmsd", "min"),
        min_ranking_gap=("ranking_gap", "min"),
        total_native_like=("native_like_count", "sum"),
    ).reset_index()

    grouped["score"] = grouped["mean_ranking_gap"] * 2.0 + grouped["mean_best_rmsd"]
    grouped = grouped.sort_values(
        ["score", "mean_ranking_gap", "mean_best_rmsd", "min_best_rmsd"]
    ).reset_index(drop=True)

    grouped.to_csv(outdir / "grid_ranked_parameter_sets.csv", index=False)

def make_plots(beam: pd.DataFrame, summary: pd.DataFrame, outdir: Path):
    if not summary.empty and "native_rebuild_ca_rmsd" in summary.columns:
        rebuild = summary[["protein_name", "reference_pdb_id", "native_rebuild_ca_rmsd"]].drop_duplicates().sort_values("protein_name")
        rebuild.to_csv(outdir / "native_rebuild_quality.csv", index=False)
        plt.figure(figsize=(8, 4.5))
        plt.bar(rebuild["protein_name"], rebuild["native_rebuild_ca_rmsd"])
        plt.axhline(GOOD_REBUILD_THRESH, linestyle="--")
        plt.ylabel("Native rebuild RMSD (Å)")
        plt.title("Rebuild quality by protein")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(outdir / "native_rebuild_quality.png", dpi=200)
        plt.close()

    if not summary.empty:
        _bar(
            summary.groupby("protein_name", dropna=False)["best_rmsd"].min().reset_index().sort_values("protein_name"),
            "protein_name", "best_rmsd", "Best sampled RMSD by protein", "Best RMSD (Å)",
            outdir / "best_rmsd_by_protein.png"
        )
        _bar(
            summary.groupby("protein_name", dropna=False)["best_energy_rmsd"].min().reset_index().sort_values("protein_name"),
            "protein_name", "best_energy_rmsd", "Best-energy RMSD by protein", "Best-energy RMSD (Å)",
            outdir / "best_energy_rmsd_by_protein.png"
        )
        _bar(
            summary.groupby("protein_name", dropna=False)["ranking_gap"].min().reset_index().sort_values("protein_name"),
            "protein_name", "ranking_gap", "Best ranking gap by protein", "Ranking gap (Å)",
            outdir / "ranking_gap_by_protein.png"
        )

        gap_cols = [c for c in [
            "protein_name", "reference_pdb_id", "chi_mode", "window_deg", "step_deg",
            "hbond_scale", "sasa_scale", "vdw_rep_scale", "vdw_attr_scale",
            "rotamer_scale", "pi_stack_scale", "best_rmsd", "best_energy_rmsd", "ranking_gap"
        ] if c in summary.columns]
        summary[gap_cols].copy().to_csv(outdir / "ranking_gap_summary.csv", index=False)
        summary.sort_values(["protein_name", "best_rmsd", "ranking_gap"]).groupby("protein_name", dropna=False).head(5).to_csv(
            outdir / "slide_top5_by_best_rmsd.csv", index=False
        )
        summary.sort_values(["protein_name", "ranking_gap", "best_rmsd"]).groupby("protein_name", dropna=False).head(5).to_csv(
            outdir / "slide_top5_by_ranking_gap.csv", index=False
        )

    if not beam.empty and "rmsd_to_reference_A" in beam.columns and "energy" in beam.columns:
        for protein, g in beam.groupby("protein_name", dropna=False):
            g = g[pd.notnull(g["rmsd_to_reference_A"]) & pd.notnull(g["energy"])].copy()
            if g.empty:
                continue
            plt.figure(figsize=(6.5, 5))
            plt.scatter(g["rmsd_to_reference_A"], g["energy"], alpha=0.3)
            plt.xlabel("RMSD to reference (Å)")
            plt.ylabel("Energy")
            plt.title(f"Energy funnel — {protein}")
            plt.tight_layout()
            plt.savefig(outdir / f"funnel_{protein}.png", dpi=200)
            plt.close()

    _save_term_correlations(beam, summary, outdir)


def main():
    ap = argparse.ArgumentParser(description="Analyze panel/grid results from a collected directory.")
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    indir = Path(args.indir)
    beam_csv = indir / "master_beam_rows.csv"
    native_csv = indir / "master_native_rows.csv"
    manifest_csv = indir / "master_grid_manifest.csv"
    for p in [beam_csv, native_csv, manifest_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Required input file not found: {p}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    beam, native, summary = build_corrected_summary(beam_csv, native_csv, manifest_csv, outdir)
    make_plots(beam, summary, outdir)
    save_ranked_parameter_sets(summary, outdir)
    (outdir / "analysis_run_info.json").write_text(json.dumps({
        "indir": str(indir),
        "outdir": str(outdir),
        "files_written": sorted([p.name for p in outdir.iterdir() if p.is_file()]),
    }, indent=2))


if __name__ == "__main__":
    main()
