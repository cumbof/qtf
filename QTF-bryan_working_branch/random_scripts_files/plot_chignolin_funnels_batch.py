#!/usr/bin/env python3
"""
Batch funnel plotting for Chignolin using separate native-score and beam-search directories.

This version matches a layout like:

beam_search_results/
  beam_chig_selective_hbond15/
    beamsearch_ranked.csv
  beam_chig_selective_hbond15_scalesasa0.7/
    beamsearch_ranked.csv
  ...

native_score_results/
  chig_amber_selective_hbond15.json
  chig_amber_selective_hbond15_scalesasa0.7.json
  ...

Edit ROOT_NATIVE, ROOT_BEAM, and EXPERIMENTS below.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT_NATIVE = Path("/home/raubenb/gitrepos/QTF/native_scores")
ROOT_BEAM = Path("/home/raubenb/gitrepos/QTF/beam_search_results")
OUTDIR = Path("chignolin_funnel_plots")
OUTDIR.mkdir(parents=True, exist_ok=True)

REF_E2E = 5.5086
NATIVE_THRESH = 2.0

EXPERIMENTS = [
    {
        "label": "selective_baseline",
        "slug": "E5_selective_chi_baseline",
        "beam_dir": "beam_chig_selective_chi2",
        "beam_file": "beamsearch_ranked.csv",
        "native_file": "chig_amber_selective.json",
    },
    {
        "label": "hbond1.5",
        "slug": "E6_hbond_1p5",
        "beam_dir": "beam_chig_selective_hbond15",
        "beam_file": "beamsearch_ranked.csv",
        "native_file": "chig_amber_selective_hbond15.json",
    },
    {
        "label": "sasa0.7",
        "slug": "E7_sasa_0p7",
        "beam_dir": "beam_chig_selective_hbond15_scalesasa0.7",
        "beam_file": "beamsearch_ranked.csv",
        "native_file": "chig_amber_selective_hbond15_scalesasa0.7.json",
    },
    {
        "label": "ljattr_added",
        "slug": "E8_ljattr_added",
        "beam_dir": "beam_chig_selective_hbond15_sasa07_ljattr",
        "beam_file": "beamsearch_ranked.csv",
        "native_file": "chig_amber_selective_hbond15_sasa07_ljattr.json",
    },
    {
        "label": "hb0.75_sasa0.7_lj0.1",
        "slug": "E9_hbond0p75_sasa0p7_lj0p1",
        "beam_dir": "beam_chig_selective_hbond0.75_sasa0.7_ljattr0.1",
        "beam_file": "beamsearch_ranked.csv",
        "native_file": "chig_amber_selective_hbond0.75_sasa0.7_ljattr0.1.json",
    },
    {
        "label": "hb0.6_sasa0.8_lj0.15",
        "slug": "E10_hbond0p6_sasa0p8_lj0p15",
        "beam_dir": "beam_chig_selective_hbond0.6_sasa0.8_ljattr0.15",
        "beam_file": "beamsearch_ranked.csv",
        "native_file": "chig_amber_selective_hbond0.6_scalesasa0.8_ljattract0.15.json",
    },
]

summary_rows = []

def load_native(native_json_path):
    with open(native_json_path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        obj = obj[0]
    return obj

def resolve_paths(exp):
    beam_csv = ROOT_BEAM / exp["beam_dir"] / exp["beam_file"]
    native_json = ROOT_NATIVE / exp["native_file"]
    return beam_csv, native_json

def plot_one(exp):
    beam_csv, native_json = resolve_paths(exp)
    slug = exp["slug"]
    label = exp["label"]

    if not beam_csv.exists():
        print(f"[warn] missing beam CSV: {beam_csv}")
        return None
    if not native_json.exists():
        print(f"[warn] missing native JSON: {native_json}")
        return None

    df = pd.read_csv(beam_csv)
    native = load_native(native_json)

    if "rmsd_to_reference_A" not in df.columns:
        print(f"[warn] no rmsd_to_reference_A in {beam_csv}")
        return None

    df = df[pd.notnull(df["energy"]) & pd.notnull(df["rmsd_to_reference_A"])].copy()
    if df.empty:
        print(f"[warn] no valid rows in {beam_csv}")
        return None

    native_energy = float(native["total_energy"])
    native_rebuilt_rmsd = float(native["rebuilt_vs_native_ca_rmsd"])

    best_energy_row = df.sort_values("energy", ascending=True).iloc[0]
    best_rmsd_row = df.sort_values("rmsd_to_reference_A", ascending=True).iloc[0]

    summary_rows.append({
        "slug": slug,
        "label": label,
        "best_rmsd_A": float(best_rmsd_row["rmsd_to_reference_A"]),
        "best_energy_rmsd_A": float(best_energy_row["rmsd_to_reference_A"]),
        "best_energy": float(best_energy_row["energy"]),
        "best_rmsd_energy": float(best_rmsd_row["energy"]),
        "native_energy": native_energy,
        "native_rebuilt_rmsd_A": native_rebuilt_rmsd,
        "beam_csv": str(beam_csv),
        "native_json": str(native_json),
        "n_points": int(len(df)),
    })

    # Energy vs RMSD
    plt.figure(figsize=(7, 5))
    plt.scatter(df["rmsd_to_reference_A"], df["energy"], alpha=0.7)
    plt.axhline(native_energy, linestyle="--", label=f"Native energy = {native_energy:.1f}")
    plt.axvline(NATIVE_THRESH, linestyle=":", label=f"Native-like threshold = {NATIVE_THRESH:.1f} Å")
    plt.xlabel("RMSD to experimental structure (Å)")
    plt.ylabel("Energy")
    plt.title(f"Energy vs RMSD — {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{slug}_energy_vs_rmsd.png", dpi=200)
    plt.close()

    # E2E vs RMSD
    if "pred_e2e_A" in df.columns:
        plt.figure(figsize=(7, 5))
        plt.scatter(df["rmsd_to_reference_A"], df["pred_e2e_A"], alpha=0.7)
        plt.axhline(REF_E2E, linestyle="--", label=f"Experimental E2E = {REF_E2E:.2f} Å")
        plt.axvline(NATIVE_THRESH, linestyle=":", label=f"Native-like threshold = {NATIVE_THRESH:.1f} Å")
        plt.xlabel("RMSD to experimental structure (Å)")
        plt.ylabel("Predicted end-to-end distance (Å)")
        plt.title(f"E2E vs RMSD — {label}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTDIR / f"{slug}_e2e_vs_rmsd.png", dpi=200)
        plt.close()

    return df, native

plotted = []
for exp in EXPERIMENTS:
    result = plot_one(exp)
    if result is not None:
        plotted.append((exp, *result))

if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTDIR / "funnel_summary.csv", index=False)

    # Progress bar
    plt.figure(figsize=(10, 5))
    plt.bar(summary_df["label"], summary_df["best_rmsd_A"])
    plt.ylabel("Best RMSD (Å)")
    plt.title("Chignolin progress by experiment")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTDIR / "combined_progress_bar.png", dpi=200)
    plt.close()

    # Progress lines
    plt.figure(figsize=(10, 5))
    x = range(len(summary_df))
    plt.plot(x, summary_df["best_rmsd_A"], marker="o", label="Best RMSD in beam")
    plt.plot(x, summary_df["best_energy_rmsd_A"], marker="s", label="RMSD of best-energy structure")
    plt.xticks(x, summary_df["label"], rotation=45, ha="right")
    plt.ylabel("RMSD (Å)")
    plt.title("Chignolin progress across experiments")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "combined_progress_lines.png", dpi=200)
    plt.close()

if plotted:
    n = len(plotted)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]
    ax_list = [ax for row in axes for ax in row]
    for ax in ax_list[n:]:
        ax.axis("off")

    for ax, (exp, df, native) in zip(ax_list, plotted):
        native_energy = float(native["total_energy"])
        ax.scatter(df["rmsd_to_reference_A"], df["energy"], alpha=0.55)
        ax.axhline(native_energy, linestyle="--")
        ax.axvline(NATIVE_THRESH, linestyle=":")
        ax.set_title(exp["label"])
        ax.set_xlabel("RMSD (Å)")
        ax.set_ylabel("Energy")

    fig.suptitle("Chignolin energy funnels across experiments", y=0.995)
    fig.tight_layout()
    fig.savefig(OUTDIR / "combined_funnel_grid.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

print(f"Wrote plots to {OUTDIR.resolve()}")
