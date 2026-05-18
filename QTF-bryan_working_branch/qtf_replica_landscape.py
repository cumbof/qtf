#!/usr/bin/env python3
"""
QTF Per-Replica Energy Landscape
Shows energy at every function call across all 3 stages for one replica.

USAGE:
    python3 qtf_replica_landscape.py \
        --results_dir outputs/no_skip/slurm_YYDPETGTWY_amber \
        --replica_id 0 \
        --out replica_0_landscape.png
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', required=True)
parser.add_argument('--replica_id',  type=int, default=0)
parser.add_argument('--out',         default='replica_landscape.png')
args = parser.parse_args()

# ── Load tracker ───────────────────────────────────────────────────────────────
tracker_path = os.path.join(args.results_dir,
    f"replica_{args.replica_id}",
    f"replica_{args.replica_id}_tracker.json")
result_path  = os.path.join(args.results_dir,
    f"replica_{args.replica_id}",
    f"replica_{args.replica_id}_result.json")

with open(tracker_path) as f: tracker = json.load(f)
with open(result_path)  as f: result  = json.load(f)

history  = np.array(tracker['history'])
markers  = tracker['stage_markers']   # [[iter, "Stage1"], [iter, "Stage2"], ...]
n        = len(history)

print(f"Replica {args.replica_id}")
print(f"  Total function calls : {n:,}")
print(f"  Initial energy       : {history[0]:.2f}")
print(f"  Final energy         : {history[-1]:.2f}")
print(f"  RMSD stage1          : {result.get('rmsd_stage1', 'N/A')}")
print(f"  RMSD stage2          : {result.get('rmsd_stage2', 'N/A')}")
print(f"  RMSD stage3          : {result.get('rmsd_stage3', 'N/A')}")

# ── Stage boundaries ───────────────────────────────────────────────────────────
COL = {'Stage1': '#ff6b35', 'Stage2': '#4ecdc4', 'Stage3': '#a855f7'}
LAB = {'Stage1': 'Stage 1 — COBYLA collapse',
       'Stage2': 'Stage 2 — SLSQP refine',
       'Stage3': 'Stage 3 — SLSQP relax'}

stage_ranges = []
for idx, (start, name) in enumerate(markers):
    end = markers[idx+1][0] if idx+1 < len(markers) else n
    stage_ranges.append((name, start, end))

# ── Figure: 3 panels ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.patch.set_facecolor('#0f0f1a')

iters = np.arange(n)

# ── Panel 1: Full landscape every call ────────────────────────────────────────
ax = axes[0]
ax.set_facecolor('#1a1a2e')
for name, start, end in stage_ranges:
    ax.plot(iters[start:end], history[start:end],
            color=COL[name], linewidth=0.6, alpha=0.7)
    ax.axvline(start, color=COL[name], linestyle='--', alpha=0.4, linewidth=1)
    mid = start + (end - start) // 2
    ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] != 1 else history.max(),
            name, color=COL[name], fontsize=8, ha='center')

ax.set_xlabel('Function Evaluation #', color='white', fontsize=10)
ax.set_ylabel('Energy', color='white', fontsize=10)
ax.set_title(f'Replica {args.replica_id} — Every Function Call\n'
             f'Initial: {history[0]:.1f}  →  Final: {history[-1]:.1f}  '
             f'(Δ = {history[0]-history[-1]:.1f})',
             color='white', fontsize=11, fontweight='bold')
ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_color('#444')
ax.grid(True, alpha=0.15, color='white')

# ── Panel 2: Running minimum (best so far at each call) ───────────────────────
ax = axes[1]
ax.set_facecolor('#1a1a2e')
running_min = np.minimum.accumulate(history)
for name, start, end in stage_ranges:
    ax.plot(iters[start:end], history[start:end],
            color=COL[name], linewidth=0.5, alpha=0.25)
    ax.plot(iters[start:end], running_min[start:end],
            color=COL[name], linewidth=2.5, alpha=0.95,
            label=f"{LAB[name]}  ({history[start]:.1f} → {history[end-1]:.1f})")
    ax.axvline(start, color=COL[name], linestyle='--', alpha=0.3, linewidth=1)

ax.set_xlabel('Function Evaluation #', color='white', fontsize=10)
ax.set_ylabel('Best Energy So Far', color='white', fontsize=10)
ax.set_title('Running Minimum — Is it consistently decreasing?',
             color='white', fontsize=11, fontweight='bold')
ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_color('#444')
ax.grid(True, alpha=0.15, color='white')
ax.legend(facecolor='#2a2a3e', edgecolor='#555',
          labelcolor='white', fontsize=8, loc='upper right')

# ── Panel 3: Per-stage zoom — each stage starts at 0 ─────────────────────────
ax = axes[2]
ax.set_facecolor('#1a1a2e')

rmsd_vals = {
    'Stage1': result.get('rmsd_stage1'),
    'Stage2': result.get('rmsd_stage2'),
    'Stage3': result.get('rmsd_stage3'),
}

for name, start, end in stage_ranges:
    seg   = history[start:end]
    xi    = np.arange(len(seg))
    rm    = np.minimum.accumulate(seg)
    rmsd  = rmsd_vals.get(name)
    label = f"{LAB[name]}"
    if rmsd: label += f"  →  RMSD: {rmsd:.3f} Å"

    ax.plot(xi, seg, color=COL[name], linewidth=0.5, alpha=0.25)
    ax.plot(xi, rm,  color=COL[name], linewidth=2.5, alpha=0.95, label=label)

    # Annotate start and end energy
    ax.annotate(f'{seg[0]:.1f}', xy=(0, seg[0]),
                color=COL[name], fontsize=8, ha='left')
    ax.annotate(f'{seg[-1]:.1f}', xy=(len(seg)-1, rm[-1]),
                color=COL[name], fontsize=8, ha='right')

ax.set_xlabel('Iteration within Stage', color='white', fontsize=10)
ax.set_ylabel('Energy', color='white', fontsize=10)
ax.set_title('Per-Stage Energy (each stage resets x-axis)',
             color='white', fontsize=11, fontweight='bold')
ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_color('#444')
ax.grid(True, alpha=0.15, color='white')
ax.legend(facecolor='#2a2a3e', edgecolor='#555',
          labelcolor='white', fontsize=8)

plt.tight_layout()
plt.savefig(args.out, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f"Saved → {args.out}")
