#!/usr/bin/env python3
"""
QTF Cross-Replica Energy Analysis
===================================
Clean, readable output with exact numbers.

USAGE:
    python3 qtf_replica_analysis.py \
        --results_dir outputs/energy_reps_Brickwork_entanglement11/slurm_YYDPETGTWY_amber \
        --n_replicas 23 \
        --sequence YYDPETGTWY \
        --save_dir outputs/analysis
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')


# ── Colors ─────────────────────────────────────────────────────────────────────
BG_DARK  = '#0f0f1a'
BG_PANEL = '#1a1a2e'
COL_S1   = '#ff6b35'
COL_S2   = '#4ecdc4'
COL_S3   = '#a855f7'
COL_GOLD = '#ffd700'
COL_WHITE = '#ffffff'
COL_GRAY  = '#888888'
INIT_COLORS = {'helix': '#ff6b35', 'sheet': '#4ecdc4', 'random': '#a855f7'}


def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=COL_WHITE, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.grid(True, alpha=0.15, color=COL_WHITE)
    if title:  ax.set_title(title, color=COL_WHITE, fontsize=11, fontweight='bold', pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=COL_WHITE, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=COL_WHITE, fontsize=9)


def load_all_data(results_dir, n_replicas):
    trackers = []
    results  = []
    missing  = []

    for i in range(n_replicas):
        replica_dir  = os.path.join(results_dir, f"replica_{i}")
        tracker_path = os.path.join(replica_dir,  f"replica_{i}_tracker.json")
        result_path  = os.path.join(replica_dir,  f"replica_{i}_result.json")

        t_data = r_data = None
        if os.path.exists(tracker_path):
            with open(tracker_path) as f: t_data = json.load(f)
        if os.path.exists(result_path):
            with open(result_path)  as f: r_data = json.load(f)

        if t_data and r_data:
            trackers.append(t_data)
            results.append(r_data)
        else:
            missing.append(i)

    print(f"  Loaded : {len(trackers)} / {n_replicas} replicas")
    if missing:
        print(f"  Missing: {missing[:10]}{'...' if len(missing)>10 else ''}")
    return trackers, results


def extract_stage_info(tracker):
    history = tracker['history']
    markers = tracker['stage_markers']
    if len(history) == 0 or len(markers) < 3:
        return None

    s1 = min(markers[0][0], len(history)-1)
    s2 = min(markers[1][0], len(history)-1)
    s3 = min(markers[2][0], len(history)-1)

    e_init  = history[s1]
    e_s1end = history[s2-1] if s2 > 0 else history[s1]
    e_s2end = history[s3-1] if s3 > 0 else history[s2]
    e_final = history[-1]

    return {
        'e_init':     e_init,
        'e_s1end':    e_s1end,
        'e_s2end':    e_s2end,
        'e_final':    e_final,
        's1_drop':    e_init  - e_s1end,
        's2_drop':    e_s1end - e_s2end,
        's3_drop':    e_s2end - e_final,
        'total_drop': e_init  - e_final,
        's1_evals':   s2 - s1,
        's2_evals':   s3 - s2,
        's3_evals':   len(history) - s3,
        'history':    history,
        's1_idx':     s1,
        's2_idx':     s2,
        's3_idx':     s3,
    }


def print_replica_table(results, stage_list, sequence):
    print(f"\n{'='*110}")
    print(f"  REPLICA-BY-REPLICA BREAKDOWN  —  {sequence}")
    print(f"{'='*110}")
    print(f"  {'Rep':>4}  {'Init':>7}  {'E_start':>9}  {'S1 drop':>9}  {'E_s1':>9}  "
          f"{'S2 drop':>9}  {'E_s2':>9}  {'S3 drop':>9}  {'E_final':>9}  "
          f"{'RMSD':>8}  {'Time':>7}")
    print(f"  {'-'*4}  {'-'*7}  {'-'*9}  {'-'*9}  {'-'*9}  "
          f"{'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  "
          f"{'-'*8}  {'-'*7}")

    for res, stage in zip(results, stage_list):
        if stage is None:
            continue
        rmsd    = res.get('rmsd_to_reference')
        runtime = res.get('runtime_s', 0) / 60.0
        rmsd_str = f"{rmsd:.4f}" if rmsd else "   N/A  "

        print(f"  {res['replica_id']:>4}  {res.get('init_type','?'):>7}  "
              f"{stage['e_init']:>9.2f}  "
              f"{stage['s1_drop']:>+9.2f}  "
              f"{stage['e_s1end']:>9.2f}  "
              f"{stage['s2_drop']:>+9.2f}  "
              f"{stage['e_s2end']:>9.2f}  "
              f"{stage['s3_drop']:>+9.2f}  "
              f"{stage['e_final']:>9.2f}  "
              f"{rmsd_str:>8}  "
              f"{runtime:>5.1f}m")

    print(f"{'='*110}")


def print_stage_summary(stage_list, results):
    valid = [s for s in stage_list if s is not None]
    if not valid:
        return

    s1d = [s['s1_drop']    for s in valid]
    s2d = [s['s2_drop']    for s in valid]
    s3d = [s['s3_drop']    for s in valid]
    tot = [s['total_drop'] for s in valid]
    fin = [s['e_final']    for s in valid]

    print(f"\n{'='*65}")
    print(f"  STAGE SUMMARY  ({len(valid)} replicas)")
    print(f"{'='*65}")
    print(f"  {'Stage':<25}  {'Mean':>8}  {'Best':>8}  {'Worst':>8}  {'Std':>8}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'Stage 1  (COBYLA collapse)':<25}  {np.mean(s1d):>8.2f}  "
          f"{max(s1d):>8.2f}  {min(s1d):>8.2f}  {np.std(s1d):>8.2f}")
    print(f"  {'Stage 2  (SLSQP refine)':<25}  {np.mean(s2d):>8.2f}  "
          f"{max(s2d):>8.2f}  {min(s2d):>8.2f}  {np.std(s2d):>8.2f}")
    print(f"  {'Stage 3  (SLSQP relax)':<25}  {np.mean(s3d):>8.2f}  "
          f"{max(s3d):>8.2f}  {min(s3d):>8.2f}  {np.std(s3d):>8.2f}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'Total':<25}  {np.mean(tot):>8.2f}  "
          f"{max(tot):>8.2f}  {min(tot):>8.2f}  {np.std(tot):>8.2f}")

    print(f"\n  Final energy stats:")
    print(f"    Best  : {min(fin):.4f}")
    print(f"    Mean  : {np.mean(fin):.4f}")
    print(f"    Worst : {max(fin):.4f}")
    print(f"    Std   : {np.std(fin):.4f}")

    mean_total = np.mean(tot)
    if mean_total > 0:
        print(f"\n  % contribution to total drop:")
        print(f"    Stage 1 : {np.mean(s1d):>8.2f}  →  {100*np.mean(s1d)/mean_total:>5.1f}%")
        print(f"    Stage 2 : {np.mean(s2d):>8.2f}  →  {100*np.mean(s2d)/mean_total:>5.1f}%")
        print(f"    Stage 3 : {np.mean(s3d):>8.2f}  →  {100*np.mean(s3d)/mean_total:>5.1f}%")
    print(f"{'='*65}")


def print_rmsd_summary(results):
    has_rmsd = [r for r in results if r.get('rmsd_to_reference') is not None]
    if not has_rmsd:
        print("\n  No RMSD data available.")
        return

    rmsds = [r['rmsd_to_reference'] for r in has_rmsd]

    print(f"\n{'='*65}")
    print(f"  RMSD SUMMARY  ({len(has_rmsd)} replicas vs ground truth)")
    print(f"{'='*65}")
    print(f"  Best RMSD  : {min(rmsds):.4f} Å")
    print(f"  Mean RMSD  : {np.mean(rmsds):.4f} Å")
    print(f"  Std RMSD   : {np.std(rmsds):.4f} Å")
    print(f"  Worst RMSD : {max(rmsds):.4f} Å")

    print(f"\n  By init strategy:")
    print(f"  {'Strategy':>10}  {'N':>4}  {'Best':>8}  {'Mean':>8}  {'Std':>8}")
    print(f"  {'-'*10}  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}")
    for strat in ['helix', 'sheet', 'random']:
        vals = [r['rmsd_to_reference'] for r in has_rmsd
                if r.get('init_type') == strat]
        if vals:
            print(f"  {strat:>10}  {len(vals):>4}  "
                  f"{min(vals):>8.4f}  {np.mean(vals):>8.4f}  {np.std(vals):>8.4f}")

    print(f"\n  Top 10 by RMSD:")
    print(f"  {'Rank':>5}  {'Rep':>5}  {'Init':>8}  {'RMSD (Å)':>10}  {'Energy':>10}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*10}  {'-'*10}")
    for rank, r in enumerate(sorted(has_rmsd,
                             key=lambda x: x['rmsd_to_reference'])[:10], 1):
        print(f"  {rank:>5}  {r['replica_id']:>5}  "
              f"{r.get('init_type','?'):>8}  "
              f"{r['rmsd_to_reference']:>10.4f}  "
              f"{r['energy']:>10.4f}")
    print(f"{'='*65}")


def plot_clean(trackers, results, stage_list, save_dir, sequence):
    valid = [(t, r, s) for t, r, s in
             zip(trackers, results, stage_list) if s is not None]
    if not valid:
        return

    e_init  = [s['e_init']  for _, _, s in valid]
    e_s1    = [s['e_s1end'] for _, _, s in valid]
    e_s2    = [s['e_s2end'] for _, _, s in valid]
    e_final = [s['e_final'] for _, _, s in valid]
    rep_ids = [str(r['replica_id']) for _, r, _ in valid]
    x       = np.arange(len(valid))

    # ── Plot 1: Energy flow ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(14, len(valid)*0.5), 6))
    fig.patch.set_facecolor(BG_DARK)
    style_ax(ax, f"Energy at Each Stage per Replica — {sequence}",
             "Replica", "Energy")

    ax.plot(x, e_init,  'o--', color='#ffffff', lw=1,   ms=4, alpha=0.5, label='Start')
    ax.plot(x, e_s1,    's-',  color=COL_S1,    lw=1.5, ms=4, alpha=0.8, label='After Stage 1 (COBYLA)')
    ax.plot(x, e_s2,    '^-',  color=COL_S2,    lw=1.5, ms=4, alpha=0.8, label='After Stage 2 (SLSQP)')
    ax.plot(x, e_final, 'D-',  color=COL_S3,    lw=2,   ms=5, alpha=1.0, label='Final (Stage 3)')

    # Annotate exact final values
    for i, (xi, val) in enumerate(zip(x, e_final)):
        ax.text(xi, val - abs(max(e_final)-min(e_final))*0.05,
                f'{val:.1f}', ha='center', va='top',
                color=COL_S3, fontsize=6, rotation=90)

    best_i = np.argmin(e_final)
    ax.annotate(f"★ Best: {e_final[best_i]:.2f}",
                xy=(x[best_i], e_final[best_i]),
                xytext=(x[best_i], e_final[best_i] - abs(max(e_final)-min(e_final))*0.25),
                color=COL_GOLD, fontsize=9, fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color=COL_GOLD, lw=1.5))

    ax.set_xticks(x)
    ax.set_xticklabels(rep_ids, rotation=90, fontsize=7)
    ax.legend(facecolor='#2a2a3e', edgecolor='#555',
              labelcolor=COL_WHITE, fontsize=9)
    plt.tight_layout()
    p = os.path.join(save_dir, "1_energy_flow_per_replica.png")
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"[SAVED] {p}")
    plt.close()

    # ── Plot 2: Stage drops with exact numbers ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(max(18, len(valid)*0.6), 6))
    fig.patch.set_facecolor(BG_DARK)

    for col, (key, color, label) in enumerate([
        ('s1_drop', COL_S1, 'Stage 1  COBYLA  drop'),
        ('s2_drop', COL_S2, 'Stage 2  SLSQP   drop'),
        ('s3_drop', COL_S3, 'Stage 3  Relax   drop'),
    ]):
        ax = axes[col]
        style_ax(ax, label, "Replica", "Energy Drop")
        drops = [s[key] for _, _, s in valid]
        bars  = ax.bar(x, drops, color=color, alpha=0.8,
                       edgecolor=COL_WHITE, linewidth=0.3)

        for bar, val in zip(bars, drops):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    f'{val:.1f}', ha='center', va='bottom',
                    color=COL_WHITE, fontsize=6.5, rotation=90)

        ax.axhline(np.mean(drops), color=COL_GOLD, lw=2,
                   linestyle='--', label=f'Mean: {np.mean(drops):.2f}')
        ax.set_xticks(x)
        ax.set_xticklabels(rep_ids, rotation=90, fontsize=7)
        ax.legend(facecolor='#2a2a3e', edgecolor='#555',
                  labelcolor=COL_WHITE, fontsize=8)

    fig.suptitle(f"Stage-by-Stage Energy Drops — {sequence}",
                 color=COL_WHITE, fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(save_dir, "2_stage_drops_exact.png")
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"[SAVED] {p}")
    plt.close()

    # ── Plot 3: Best replica full landscape ────────────────────────────────────
    best_i = np.argmin(e_final)
    bt, br, bs = valid[best_i]
    history = np.array(bt['history'])
    s1i, s2i, s3i = bs['s1_idx'], bs['s2_idx'], bs['s3_idx']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(BG_DARK)

    ax = axes[0]
    style_ax(ax, f"Best Replica #{br['replica_id']} — Full Energy Landscape",
             "Function Evaluation", "Energy")

    ax.plot(np.arange(s2i),              history[:s2i],   color=COL_S1, lw=1.2, label='Stage 1 COBYLA')
    ax.plot(np.arange(s2i, s3i),         history[s2i:s3i],color=COL_S2, lw=1.2, label='Stage 2 SLSQP')
    ax.plot(np.arange(s3i, len(history)),history[s3i:],   color=COL_S3, lw=1.5, label='Stage 3 Relax')

    ax.axvline(s2i, color=COL_S1, ls='--', alpha=0.4)
    ax.axvline(s3i, color=COL_S2, ls='--', alpha=0.4)

    for xi, val, lbl, col in [
        (0,             bs['e_init'],  f"Start\n{bs['e_init']:.2f}",   COL_WHITE),
        (s2i-1,         bs['e_s1end'], f"S1 end\n{bs['e_s1end']:.2f}", COL_S1),
        (s3i-1,         bs['e_s2end'], f"S2 end\n{bs['e_s2end']:.2f}", COL_S2),
        (len(history)-1,bs['e_final'], f"Final\n{bs['e_final']:.2f}",  COL_S3),
    ]:
        ax.annotate(lbl, xy=(xi, history[xi]),
                    xytext=(xi, history[xi] + abs(history.max()-history.min())*0.08),
                    color=col, fontsize=8, fontweight='bold', ha='center',
                    arrowprops=dict(arrowstyle='->', color=col, lw=1))

    ax.legend(facecolor='#2a2a3e', edgecolor='#555',
              labelcolor=COL_WHITE, fontsize=9)

    ax2 = axes[1]
    style_ax(ax2, f"Best Replica #{br['replica_id']} — Per-Stage Running Min",
             "Iteration within Stage", "Energy")

    for seg, color, label in [
        (history[s1i:s2i], COL_S1, f"Stage 1  {bs['e_init']:.1f}→{bs['e_s1end']:.1f}  (Δ{bs['s1_drop']:.1f})"),
        (history[s2i:s3i], COL_S2, f"Stage 2  {bs['e_s1end']:.1f}→{bs['e_s2end']:.1f}  (Δ{bs['s2_drop']:.1f})"),
        (history[s3i:],    COL_S3, f"Stage 3  {bs['e_s2end']:.1f}→{bs['e_final']:.1f}  (Δ{bs['s3_drop']:.1f})"),
    ]:
        if len(seg) == 0: continue
        xi = np.arange(len(seg))
        rm = np.minimum.accumulate(seg)
        ax2.plot(xi, seg, color=color, lw=0.6, alpha=0.3)
        ax2.plot(xi, rm,  color=color, lw=2.0, label=label)

    ax2.legend(facecolor='#2a2a3e', edgecolor='#555',
               labelcolor=COL_WHITE, fontsize=9)

    plt.tight_layout()
    p = os.path.join(save_dir, "3_best_replica_landscape.png")
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"[SAVED] {p}")
    plt.close()

    # ── Plot 4: RMSD vs Energy ─────────────────────────────────────────────────
    has_rmsd = [(r, s) for _, r, s in valid
                if r.get('rmsd_to_reference') is not None]
    if not has_rmsd:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BG_DARK)
    style_ax(ax, f"Energy vs RMSD — {sequence}",
             "Final Energy", "RMSD to Ground Truth (Å)")

    for r, s in has_rmsd:
        color = INIT_COLORS.get(r.get('init_type', 'random'), COL_GRAY)
        ax.scatter(r['energy'], r['rmsd_to_reference'],
                   color=color, s=80, alpha=0.75,
                   edgecolors=COL_WHITE, linewidth=0.4)
        ax.annotate(f"#{r['replica_id']}\n{r['rmsd_to_reference']:.3f}",
                    xy=(r['energy'], r['rmsd_to_reference']),
                    xytext=(4, 4), textcoords='offset points',
                    color=COL_WHITE, fontsize=6.5, alpha=0.85)

    best_r = min(has_rmsd, key=lambda x: x[0]['rmsd_to_reference'])
    ax.scatter(best_r[0]['energy'], best_r[0]['rmsd_to_reference'],
               color=COL_GOLD, s=250, zorder=5, marker='*',
               label=f"★ Best RMSD: {best_r[0]['rmsd_to_reference']:.4f} Å  "
                     f"(#{best_r[0]['replica_id']})")

    energies_r = [r['energy'] for r, _ in has_rmsd]
    rmsds_r    = [r['rmsd_to_reference'] for r, _ in has_rmsd]
    if len(energies_r) > 2:
        corr = np.corrcoef(energies_r, rmsds_r)[0, 1]
        ax.text(0.05, 0.95, f"Pearson r = {corr:.4f}",
                transform=ax.transAxes, color=COL_GOLD,
                fontsize=12, va='top', fontweight='bold')

    patches = [mpatches.Patch(color=c, label=k)
               for k, c in INIT_COLORS.items()
               if any(r.get('init_type') == k for r, _ in has_rmsd)]
    ax.legend(handles=patches + [mpatches.Patch(color=COL_GOLD, label='Best RMSD')],
              facecolor='#2a2a3e', edgecolor='#555',
              labelcolor=COL_WHITE, fontsize=9)

    plt.tight_layout()
    p = os.path.join(save_dir, "4_rmsd_vs_energy.png")
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"[SAVED] {p}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--n_replicas',  type=int, default=23)
    parser.add_argument('--sequence',    default="YYDPETGTWY")
    parser.add_argument('--save_dir',    default="outputs/analysis")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  QTF REPLICA ANALYSIS")
    print(f"  Sequence : {args.sequence}")
    print(f"  Replicas : {args.n_replicas}")
    print(f"  Dir      : {args.results_dir}")
    print(f"{'='*65}")

    trackers, results = load_all_data(args.results_dir, args.n_replicas)
    if not trackers:
        print("No data found! Check --results_dir path.")
        return

    stage_list = [extract_stage_info(t) for t in trackers]

    print_replica_table(results, stage_list, args.sequence)
    print_stage_summary(stage_list, results)
    print_rmsd_summary(results)

    print(f"\n── Generating plots ──")
    plot_clean(trackers, results, stage_list, args.save_dir, args.sequence)

    print(f"\n✅ Done!  Plots → {args.save_dir}")
    print(f"   1_energy_flow_per_replica.png   — energy at each stage per replica")
    print(f"   2_stage_drops_exact.png         — exact drop number per stage")
    print(f"   3_best_replica_landscape.png    — full curve of best replica")
    print(f"   4_rmsd_vs_energy.png            — RMSD vs energy labeled")


if __name__ == "__main__":
    main()