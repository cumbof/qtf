#!/usr/bin/env python3
import pandas as pd
from pathlib import Path


def main():
    summary_path = Path("/home/raubenb/gitrepos/QTF/grid_analysis/grid_v2_rot_pi/master_experiment_summary.csv")
    df = pd.read_csv(summary_path)

    # ---- define ranking gap ----
    df["ranking_gap"] = df["best_energy_rmsd"] - df["best_rmsd"]

    # ---- parameter key ----
    param_cols = [
        "hbond_scale",
        "sasa_scale",
        "vdw_rep_scale",
        "vdw_attr_scale",
        "rotamer_scale",
        "pi_stack_scale",
    ]

    # ---- aggregate across proteins ----
    grouped = df.groupby(param_cols).agg(
        n_proteins=("protein_name", "nunique"),

        mean_best_rmsd=("best_rmsd", "mean"),
        median_best_rmsd=("best_rmsd", "median"),

        mean_best_energy_rmsd=("best_energy_rmsd", "mean"),
        median_best_energy_rmsd=("best_energy_rmsd", "median"),

        mean_ranking_gap=("ranking_gap", "mean"),
        median_ranking_gap=("ranking_gap", "median"),

        mean_native_rebuild=("native_rebuilt_rmsd", "mean"),
    ).reset_index()

    # ---- scoring ----
    # prioritize:
    #   1) low ranking gap
    #   2) low best RMSD
    grouped["score"] = (
        grouped["mean_ranking_gap"] * 2.0 +
        grouped["mean_best_rmsd"]
    )

    grouped = grouped.sort_values("score")

    # ---- save ----
    outdir = Path("/home/raubenb/gitrepos/QTF/grid_analysis/grid_v2_rot_pi")
    outdir.mkdir(exist_ok=True)

    grouped.to_csv(outdir / "grid_ranked_parameter_sets.csv", index=False)

    # ---- print top results ----
    print("\n===== TOP PARAMETER SETS =====\n")
    print(grouped.head(10).to_string(index=False))

    print("\nSaved to:")
    print(outdir / "grid_ranked_parameter_sets.csv")


if __name__ == "__main__":
    main()
