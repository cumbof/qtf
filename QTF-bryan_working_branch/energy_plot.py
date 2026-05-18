import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv",        required=True)
parser.add_argument("--out",        default="funnel_plot.png")
parser.add_argument("--title",      default="Energy Funnel")
parser.add_argument("--rmsd_col",   default="rmsd_to_reference")
parser.add_argument("--energy_col", default="energy")
args = parser.parse_args()

df = pd.read_csv(args.csv).dropna(subset=[args.rmsd_col, args.energy_col])

# clip outliers
e_lo = np.percentile(df[args.energy_col], 2)
e_hi = np.percentile(df[args.energy_col], 98)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df[args.rmsd_col], df[args.energy_col],
           s=12, alpha=0.4, color="steelblue", linewidths=0)

ax.set_ylim(e_lo, e_hi)
ax.set_xlabel("RMSD to reference (Å)")
ax.set_ylabel("Energy")
ax.set_title(args.title)
ax.grid(True, linewidth=0.4, alpha=0.5)

plt.tight_layout()
plt.savefig(args.out, dpi=150)
print(f"Saved → {args.out}")
