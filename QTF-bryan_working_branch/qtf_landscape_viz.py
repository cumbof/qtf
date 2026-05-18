#!/usr/bin/env python3
"""
QTF Energy Landscape Visualizer
=================================
Plots energy vs iteration across all 3 optimization stages.

USAGE:
    # After a fold run, pass the tracker object:
    from qtf_landscape_viz import plot_energy_landscape
    plot_energy_landscape(tracker, sequence="YYDPETGTWY", save_path="landscape.png")

    # Or run standalone with a saved tracker JSON:
    python3 qtf_landscape_viz.py --tracker_json tracker.json
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import argparse
import os


def plot_energy_landscape(
    tracker,
    sequence: str = "",
    forcefield: str = "",
    save_path: str = None,
    show: bool = True,
    title_extra: str = "",
):
    """
    Plot energy landscape from a LandscapeTracker object.

    Parameters
    ----------
    tracker     : LandscapeTracker — from folder.fold()
    sequence    : str — protein sequence for title
    forcefield  : str — force field name for title
    save_path   : str — path to save PNG (None = don't save)
    show        : bool — whether to call plt.show()
    title_extra : str — extra info to append to title (e.g. RMSD)
    """
    history = np.array(tracker.history)
    markers = tracker.stage_markers   # list of (iter_idx, name)
    n_iters = len(history)

    if n_iters == 0:
        print("[WARNING] No energy history to plot.")
        return

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0f0f1a')

    # Stage colors
    STAGE_COLORS = {
        'Stage1': '#ff6b35',   # orange  — COBYLA collapse
        'Stage2': '#4ecdc4',   # teal    — SLSQP refine
        'Stage3': '#a855f7',   # purple  — SLSQP relax
    }
    STAGE_LABELS = {
        'Stage1': 'Stage 1 — COBYLA Collapse',
        'Stage2': 'Stage 2 — SLSQP Refinement',
        'Stage3': 'Stage 3 — SLSQP Relaxation',
    }

    # Build per-stage iteration arrays
    stage_ranges = []
    for idx, (start_iter, name) in enumerate(markers):
        end_iter = markers[idx + 1][0] if idx + 1 < len(markers) else n_iters
        stage_ranges.append((name, start_iter, end_iter))

    iters = np.arange(n_iters)

    # ── Plot 1: Full landscape ─────────────────────────────────────────────────
    ax1 = axes[0, 0]
    ax1.set_facecolor('#1a1a2e')

    for name, start, end in stage_ranges:
        color = STAGE_COLORS.get(name, '#ffffff')
        seg_iters = iters[start:end]
        seg_energy = history[start:end]
        ax1.plot(seg_iters, seg_energy, color=color, linewidth=1.2, alpha=0.9)
        ax1.axvline(x=start, color=color, linestyle='--', alpha=0.4, linewidth=0.8)
        if len(seg_iters) > 0:
            mid = start + (end - start) // 2
            ax1.text(mid, ax1.get_ylim()[1] if ax1.get_ylim()[1] != 1.0 else max(history) * 0.95,
                     name, color=color, fontsize=8, ha='center', alpha=0.7)

    ax1.set_xlabel('Function Evaluation', color='white', fontsize=10)
    ax1.set_ylabel('Energy', color='white', fontsize=10)
    ax1.set_title('Full Energy Landscape', color='white', fontsize=12, fontweight='bold')
    ax1.tick_params(colors='white')
    ax1.spines['bottom'].set_color('#444')
    ax1.spines['left'].set_color('#444')
    ax1.spines['top'].set_color('#444')
    ax1.spines['right'].set_color('#444')
    ax1.grid(True, alpha=0.15, color='white')

    # ── Plot 2: Per-stage zoom ─────────────────────────────────────────────────
    ax2 = axes[0, 1]
    ax2.set_facecolor('#1a1a2e')

    for name, start, end in stage_ranges:
        color = STAGE_COLORS.get(name, '#ffffff')
        seg_energy = history[start:end]
        seg_iters  = np.arange(len(seg_energy))

        # Running minimum (envelope)
        running_min = np.minimum.accumulate(seg_energy)

        ax2.plot(seg_iters, seg_energy,    color=color, linewidth=0.8,
                 alpha=0.4, label=f'_{name}')
        ax2.plot(seg_iters, running_min,   color=color, linewidth=2.0,
                 alpha=0.95, label=STAGE_LABELS.get(name, name))

    ax2.set_xlabel('Iteration within Stage', color='white', fontsize=10)
    ax2.set_ylabel('Energy', color='white', fontsize=10)
    ax2.set_title('Per-Stage Energy (running minimum)', color='white',
                  fontsize=12, fontweight='bold')
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('#444')
    ax2.spines['left'].set_color('#444')
    ax2.spines['top'].set_color('#444')
    ax2.spines['right'].set_color('#444')
    ax2.grid(True, alpha=0.15, color='white')
    legend = ax2.legend(facecolor='#2a2a3e', edgecolor='#555',
                        labelcolor='white', fontsize=8)

    # ── Plot 3: Stage-by-stage energy drop bar chart ───────────────────────────
    ax3 = axes[1, 0]
    ax3.set_facecolor('#1a1a2e')

    stage_names  = []
    stage_starts = []
    stage_ends_e = []
    stage_drops  = []
    stage_cols   = []

    prev_energy = history[0]
    for name, start, end in stage_ranges:
        seg = history[start:end]
        if len(seg) == 0:
            continue
        start_e = seg[0]
        end_e   = seg[-1]
        drop    = start_e - end_e
        stage_names.append(name)
        stage_starts.append(start_e)
        stage_ends_e.append(end_e)
        stage_drops.append(drop)
        stage_cols.append(STAGE_COLORS.get(name, '#ffffff'))

    x = np.arange(len(stage_names))
    bars = ax3.bar(x, stage_drops, color=stage_cols, alpha=0.85, edgecolor='white',
                   linewidth=0.5)

    for bar, drop, se, ee in zip(bars, stage_drops, stage_starts, stage_ends_e):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + abs(max(stage_drops)) * 0.02,
                 f'Δ{drop:.1f}\n({se:.1f}→{ee:.1f})',
                 ha='center', va='bottom', color='white', fontsize=8)

    ax3.set_xticks(x)
    ax3.set_xticklabels([STAGE_LABELS.get(n, n) for n in stage_names],
                         color='white', fontsize=8)
    ax3.set_ylabel('Energy Drop (higher = better)', color='white', fontsize=10)
    ax3.set_title('Energy Reduction per Stage', color='white',
                  fontsize=12, fontweight='bold')
    ax3.tick_params(colors='white')
    ax3.spines['bottom'].set_color('#444')
    ax3.spines['left'].set_color('#444')
    ax3.spines['top'].set_color('#444')
    ax3.spines['right'].set_color('#444')
    ax3.grid(True, alpha=0.15, color='white', axis='y')

    # ── Plot 4: Stats summary ──────────────────────────────────────────────────
    ax4 = axes[1, 1]
    ax4.set_facecolor('#1a1a2e')
    ax4.axis('off')

    stats_lines = [
        f"Sequence       : {sequence}",
        f"Force field    : {forcefield.upper()}",
        f"",
        f"Total evaluations : {n_iters:,}",
        f"Initial energy    : {history[0]:.2f}",
        f"Final energy      : {history[-1]:.2f}",
        f"Total drop        : {history[0] - history[-1]:.2f}",
        f"",
    ]

    for name, start, end in stage_ranges:
        seg = history[start:end]
        if len(seg) == 0:
            continue
        color = STAGE_COLORS.get(name, '#ffffff')
        label = STAGE_LABELS.get(name, name)
        stats_lines.append(
            f"{label}\n"
            f"  Evaluations : {len(seg):,}\n"
            f"  Start E     : {seg[0]:.2f}\n"
            f"  End E       : {seg[-1]:.2f}\n"
            f"  Drop        : {seg[0]-seg[-1]:.2f}\n"
        )

    if title_extra:
        stats_lines.append(f"\n{title_extra}")

    y_pos = 0.95
    for line in stats_lines:
        if line == "":
            y_pos -= 0.025
            continue
        # Color stage lines
        color = 'white'
        for sname, scol in STAGE_COLORS.items():
            if sname in line:
                color = scol
                break
        ax4.text(0.05, y_pos, line, transform=ax4.transAxes,
                 color=color, fontsize=9, va='top',
                 fontfamily='monospace')
        y_pos -= 0.06 if '\n' in line else 0.04

    ax4.set_title('Run Summary', color='white', fontsize=12, fontweight='bold')

    # ── Final layout ───────────────────────────────────────────────────────────
    title = f"QTF Energy Landscape — {sequence}"
    if forcefield:
        title += f" ({forcefield.upper()})"
    fig.suptitle(title, color='white', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"[SAVED] {save_path}")

    if show:
        plt.show()

    plt.close()
    return fig


def plot_ensemble_landscape(results, sequence="", forcefield="", save_path=None):
    """
    Plot energy landscapes for ALL replicas overlaid on one plot.
    Good for visualizing ensemble diversity.

    Parameters
    ----------
    results : list of dicts — from manager.get_ranked_results()
              each dict must have 'tracker', 'energy', 'type', 'id'
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#0f0f1a')

    INIT_COLORS = {
        'helix':  '#ff6b35',
        'sheet':  '#4ecdc4',
        'random': '#a855f7',
    }

    # ── Plot 1: All trajectories overlaid ─────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor('#1a1a2e')

    for res in results:
        tracker = res.get('tracker')
        if tracker is None or not tracker.history:
            continue
        history = np.array(tracker.history)
        color   = INIT_COLORS.get(res.get('type', 'random'), '#ffffff')
        running_min = np.minimum.accumulate(history)
        ax1.plot(running_min, color=color, linewidth=1.0, alpha=0.5)

    # Highlight best replica
    best = sorted(results, key=lambda x: x['energy'])[0]
    if best.get('tracker') and best['tracker'].history:
        h = np.minimum.accumulate(best['tracker'].history)
        ax1.plot(h, color='#ffd700', linewidth=2.5, alpha=1.0,
                 label=f"Best (E={best['energy']:.1f})")

    patches = [mpatches.Patch(color=c, label=f'{k} init')
               for k, c in INIT_COLORS.items()]
    patches.append(mpatches.Patch(color='#ffd700', label='Best replica'))
    ax1.legend(handles=patches, facecolor='#2a2a3e',
               edgecolor='#555', labelcolor='white', fontsize=9)
    ax1.set_xlabel('Function Evaluation', color='white', fontsize=10)
    ax1.set_ylabel('Energy (running min)', color='white', fontsize=10)
    ax1.set_title('All Replica Trajectories', color='white',
                  fontsize=12, fontweight='bold')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_color('#444')
    ax1.grid(True, alpha=0.15, color='white')

    # ── Plot 2: Final energy distribution ─────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('#1a1a2e')

    for init_type, color in INIT_COLORS.items():
        energies = [r['energy'] for r in results
                    if r.get('type') == init_type]
        if energies:
            ax2.hist(energies, bins=15, color=color, alpha=0.6,
                     label=f'{init_type} (n={len(energies)})',
                     edgecolor='white', linewidth=0.3)

    ax2.axvline(best['energy'], color='#ffd700', linewidth=2,
                linestyle='--', label=f"Best: {best['energy']:.1f}")
    ax2.set_xlabel('Final Energy', color='white', fontsize=10)
    ax2.set_ylabel('Count', color='white', fontsize=10)
    ax2.set_title('Final Energy Distribution by Init Strategy',
                  color='white', fontsize=12, fontweight='bold')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_color('#444')
    ax2.grid(True, alpha=0.15, color='white')
    ax2.legend(facecolor='#2a2a3e', edgecolor='#555',
               labelcolor='white', fontsize=9)

    title = f"QTF Ensemble Landscape — {sequence}"
    if forcefield:
        title += f" ({forcefield.upper()})"
    fig.suptitle(title, color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"[SAVED] {save_path}")

    plt.show()
    plt.close()
    return fig


# ── Standalone CLI ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracker_json', default=None,
                        help='Path to saved tracker JSON')
    parser.add_argument('--sequence',     default="")
    parser.add_argument('--forcefield',   default="amber")
    parser.add_argument('--save',         default="energy_landscape.png")
    args = parser.parse_args()

    if args.tracker_json:
        with open(args.tracker_json) as f:
            data = json.load(f)

        class FakeTracker:
            def __init__(self, d):
                self.history       = d['history']
                self.stage_markers = [tuple(x) for x in d['stage_markers']]
                self.current_iter  = d['current_iter']

        tracker = FakeTracker(data)
        plot_energy_landscape(
            tracker,
            sequence=args.sequence,
            forcefield=args.forcefield,
            save_path=args.save,
        )
    else:
        print("Provide --tracker_json to plot from saved data.")
        print("Or import plot_energy_landscape() in your code.")