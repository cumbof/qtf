#!/usr/bin/env python3
"""
QTF Energy-RMSD Funnel Plot
Stage 3 energy vs Stage 3 RMSD across all replicas.

USAGE:
    python3 qtf_funnel_plot.py \
        --results_dir outputs/no_skip/slurm_YYDPETGTWY_amber \
        --n_replicas 400 \
        --out funnel_plot.png
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', required=True)
parser.add_argument('--n_replicas',  type=int, default=400)
parser.add_argument('--out',         default='funnel_plot.png')
parser.add_argument('--title',       default='Stage 3 Energy vs RMSD')
args = parser.parse_args()

# ── Load data ──────────────────────────────────────────────────────────────────
energies, rmsds = [], []
for i in range(args.n_replicas):
    path = os.path.join(args.results_dir, f"replica_{i}",
                        f"replica_{i}_result.json")
    if not os.path.exists(path):
        continue
    with open(path) as f:
        r = json.load(f)
    e    = r.get('energy')        # ← Stage 3 final energy
    rmsd = r.get('rmsd_stage3')   # ← Stage 3 statevector RMSD
    if e is not None and rmsd is not None:
        energies.append(e)
        rmsds.append(rmsd)

energies = np.array(energies)
rmsds    = np.array(rmsds)

print(f"Loaded       : {len(energies)} replicas")
print(f"Energy range : {energies.min():.2f} to {energies.max():.2f}")
print(f"RMSD range   : {rmsds.min():.3f} to {rmsds.max():.3f} Å")

# ── Clip energy outliers (2-98%) ───────────────────────────────────────────────
e_lo   = np.percentile(energies, 2)
e_hi   = np.percentile(energies, 98)
mask   = (energies >= e_lo) & (energies <= e_hi)
e_plot = energies[mask]
r_plot = rmsds[mask]

corr = np.corrcoef(e_plot, r_plot)[0, 1]
print(f"Pearson r    : {corr:.4f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#0f0f1a')

# ── Left: RMSD (x) vs Energy (y) scatter ──────────────────────────────────────
ax = axes[0]
ax.set_facecolor('#1a1a2e')

sc = ax.scatter(r_plot, e_plot,
                c=r_plot, cmap='RdYlGn_r',
                s=20, alpha=0.7, edgecolors='none')
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('RMSD (Å)', color='white', fontsize=9)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

# Best RMSD
bi = np.argmin(r_plot)
ax.scatter(r_plot[bi], e_plot[bi], color='gold', s=200,
           zorder=5, marker='*',
           label=f'Best RMSD: {r_plot[bi]:.3f} Å\nEnergy: {e_plot[bi]:.1f}')

# Best energy
ei = np.argmin(e_plot)
ax.scatter(r_plot[ei], e_plot[ei], color='cyan', s=150,
           zorder=5, marker='D',
           label=f'Best Energy: {e_plot[ei]:.1f}\nRMSD: {r_plot[ei]:.3f} Å')

ax.set_xlabel('RMSD (Å) — Stage 3 Statevector', color='white', fontsize=11)
ax.set_ylabel('Energy — Stage 3 Final', color='white', fontsize=11)
ax.set_title(f'{args.title}\nPearson r = {corr:.4f}',
             color='white', fontsize=11, fontweight='bold')
ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_color('#444')
ax.grid(True, alpha=0.15, color='white')
ax.legend(facecolor='#2a2a3e', edgecolor='#555',
          labelcolor='white', fontsize=8)

# ── Right: sort by energy → does RMSD decrease? ───────────────────────────────
ax2 = axes[1]
ax2.set_facecolor('#1a1a2e')

sort_idx    = np.argsort(e_plot)
sorted_e    = e_plot[sort_idx]
sorted_rmsd = r_plot[sort_idx]

window      = max(1, len(sorted_e) // 20)
run_mean    = np.convolve(sorted_rmsd,
                          np.ones(window)/window, mode='valid')

ax2.scatter(np.arange(len(sorted_e)), sorted_rmsd,
            c=sorted_e, cmap='RdYlGn', s=10,
            alpha=0.5, edgecolors='none')
ax2.plot(np.arange(len(run_mean)), run_mean,
         color='gold', linewidth=2.5,
         label=f'Running mean (window={window})')

ax2.set_xlabel('Replicas sorted by Energy (lowest → highest)',
               color='white', fontsize=11)
ax2.set_ylabel('RMSD (Å)', color='white', fontsize=11)
ax2.set_title('Does lower energy → lower RMSD?\n'
              '(left = best energy, right = worst energy)',
              color='white', fontsize=11, fontweight='bold')
ax2.tick_params(colors='white')
for sp in ax2.spines.values(): sp.set_color('#444')
ax2.grid(True, alpha=0.15, color='white')
ax2.legend(facecolor='#2a2a3e', edgecolor='#555',
           labelcolor='white', fontsize=9)

if corr < -0.3:
    verdict = 'YES — energy predicts RMSD'
    col = '#44ff88'
elif corr < 0:
    verdict = 'WEAK — slight correlation'
    col = '#ffaa00'
else:
    verdict = 'NO — energy does not predict RMSD'
    col = '#ff4444'

ax2.text(0.05, 0.95, verdict, transform=ax2.transAxes,
         color=col, fontsize=10, va='top', fontweight='bold')

plt.tight_layout()
plt.savefig(args.out, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f"Saved → {args.out}")
