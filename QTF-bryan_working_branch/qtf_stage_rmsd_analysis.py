#!/usr/bin/env python3
"""
QTF Stage RMSD Analysis + Visualization
=========================================
Analyzes Stage 1, Stage 2, Stage 3 RMSD across all replicas.
Works for both skip_stage2=True and skip_stage2=False runs.

USAGE:
    # With all 3 stages (no_skip):
    python3 qtf_stage_rmsd_analysis.py \
        --results_dir outputs/no_skip/slurm_YYDPETGTWY_amber \
        --n_replicas 400 \
        --save_dir outputs/analysis_no_skip

    # Skip stage 2 run:
    python3 qtf_stage_rmsd_analysis.py \
        --results_dir outputs/Skip1/slurm_YYDPETGTWY_amber \
        --n_replicas 400 \
        --save_dir outputs/analysis_skip1
"""
import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ── Colors ─────────────────────────────────────────────────────────────────────
BG_DARK   = '#0f0f1a'
BG_PANEL  = '#1a1a2e'
COL_S1    = '#ff6b35'
COL_S2    = '#4ecdc4'
COL_S3    = '#a855f7'
COL_GOLD  = '#ffd700'
COL_WHITE = '#ffffff'
COL_GREEN = '#44ff88'
COL_RED   = '#ff4444'


def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=COL_WHITE, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.grid(True, alpha=0.15, color=COL_WHITE)
    if title:  ax.set_title(title,  color=COL_WHITE, fontsize=11, fontweight='bold', pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=COL_WHITE, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=COL_WHITE, fontsize=9)


def load_data(results_dir, n_replicas):
    data    = []
    missing = []
    for i in range(n_replicas):
        path = os.path.join(results_dir, f"replica_{i}", f"replica_{i}_result.json")
        if not os.path.exists(path):
            missing.append(i); continue
        with open(path) as f:
            data.append(json.load(f))
    print(f"  Loaded : {len(data)} / {n_replicas} replicas")
    if missing:
        print(f"  Missing: {len(missing)} → {missing[:5]}{'...' if len(missing)>5 else ''}")
    return data


def print_summary(data):
    has_s2 = any(r.get('rmsd_stage2') is not None for r in data)

    s1 = [(r['replica_id'], r['rmsd_stage1']) for r in data if r.get('rmsd_stage1')]
    s2 = [(r['replica_id'], r['rmsd_stage2']) for r in data if r.get('rmsd_stage2')]
    s3 = [(r['replica_id'], r['rmsd_stage3']) for r in data if r.get('rmsd_stage3')]
    en = [(r['replica_id'], r['energy'])       for r in data]

    print(f"\n{'='*70}")
    print(f"  STAGE RMSD ANALYSIS  ({len(data)} replicas)  "
          f"[{'3-stage' if has_s2 else 'skip-stage2'}]")
    print(f"{'='*70}")

    for label, pairs, color_tag in [
        ("Stage 1  (COBYLA collapse)",   s1, "S1"),
        ("Stage 2  (SLSQP refine)",      s2, "S2"),
        ("Stage 3  (SLSQP relax)",       s3, "S3"),
    ]:
        if not pairs:
            if color_tag == "S2" and not has_s2:
                print(f"\n  Stage 2 : SKIPPED (skip_stage2=True)")
            continue
        vals    = [v for _, v in pairs]
        best_id = min(pairs, key=lambda x: x[1])[0]
        print(f"\n  {label}:")
        print(f"    Best   : {min(vals):.4f} Å  → Replica #{best_id}")
        print(f"    Mean   : {np.mean(vals):.4f} Å")
        print(f"    Worst  : {max(vals):.4f} Å")
        print(f"    Std    : {np.std(vals):.4f} Å")

    # ── Stage-to-stage comparisons ─────────────────────────────────────────────
    s1d = dict(s1); s2d = dict(s2); s3d = dict(s3); ed = dict(en)

    def compare(d_from, d_to, label_from, label_to):
        common = set(d_from) & set(d_to)
        if not common: return
        improved = [i for i in common if d_to[i] < d_from[i]]
        changes  = [d_to[i] - d_from[i] for i in common]
        print(f"\n  {label_from} → {label_to}  ({len(common)} replicas):")
        print(f"    Improved : {len(improved):>4}  ({100*len(improved)/len(common):.1f}%)")
        print(f"    Worsened : {len(common)-len(improved):>4}  ({100*(len(common)-len(improved))/len(common):.1f}%)")
        print(f"    Mean Δ   : {np.mean(changes):>+.4f} Å  "
              f"({'↓ better' if np.mean(changes)<0 else '↑ worse'})")

    compare(s1d, s2d, "Stage 1", "Stage 2")
    compare(s2d, s3d, "Stage 2", "Stage 3")
    compare(s1d, s3d, "Stage 1", "Stage 3 (overall)")

    # ── Top 10 table ───────────────────────────────────────────────────────────
    final_stage = s3d if s3d else s2d if s2d else s1d
    final_label = "S3" if s3d else "S2" if s2d else "S1"

    print(f"\n  Top 10 by {final_label} RMSD:")
    hdr = f"  {'Rank':>4}  {'Rep':>5}  {'S1':>8}  "
    if has_s2: hdr += f"{'S2':>8}  "
    hdr += f"{'S3':>8}  {'Energy':>10}"
    print(hdr)
    print("  " + "-"*65)

    for rank, (i, v) in enumerate(sorted(final_stage.items(), key=lambda x:x[1])[10:100], 1):
        row = f"  {rank:>4}  {i:>5}  {s1d.get(i, float('nan')):>8.4f}  "
        if has_s2: row += f"{s2d.get(i, float('nan')):>8.4f}  "
        row += f"{s3d.get(i, float('nan')):>8.4f}  {ed.get(i,0):>10.2f}"
        print(row)

    print(f"{'='*70}")


def plot_all(data, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    has_s2 = any(r.get('rmsd_stage2') is not None for r in data)

    s1p = [(r['replica_id'], r['rmsd_stage1']) for r in data if r.get('rmsd_stage1')]
    s2p = [(r['replica_id'], r['rmsd_stage2']) for r in data if r.get('rmsd_stage2')]
    s3p = [(r['replica_id'], r['rmsd_stage3']) for r in data if r.get('rmsd_stage3')]
    enp = [(r['replica_id'], r['energy'])       for r in data]

    s1d = dict(s1p); s2d = dict(s2p); s3d = dict(s3p); ed = dict(enp)

    # Common replicas across all available stages
    if has_s2:
        common = sorted(set(s1d) & set(s2d) & set(s3d))
    else:
        common = sorted(set(s1d) & set(s3d))

    if not common:
        print("Not enough common replicas to plot!")
        return

    s1v = [s1d[i] for i in common]
    s2v = [s2d[i] for i in common] if has_s2 else None
    s3v = [s3d[i] for i in common]
    ev  = [ed.get(i, 0) for i in common]

    # ── Figure 1: Distributions + scatter ─────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor(BG_DARK)

    # Plot 1: RMSD distributions all stages
    ax = axes[0, 0]
    style_ax(ax, "RMSD Distribution — All Stages", "RMSD (Å)", "Count")
    ax.hist(s1v, bins=25, color=COL_S1, alpha=0.7,
            label=f'Stage 1  mean={np.mean(s1v):.3f}', edgecolor=COL_WHITE, lw=0.3)
    if has_s2 and s2v:
        ax.hist(s2v, bins=25, color=COL_S2, alpha=0.7,
                label=f'Stage 2  mean={np.mean(s2v):.3f}', edgecolor=COL_WHITE, lw=0.3)
    ax.hist(s3v, bins=25, color=COL_S3, alpha=0.7,
            label=f'Stage 3  mean={np.mean(s3v):.3f}', edgecolor=COL_WHITE, lw=0.3)
    ax.axvline(min(s1v), color=COL_S1, lw=1.5, linestyle='--')
    if has_s2 and s2v:
        ax.axvline(min(s2v), color=COL_S2, lw=1.5, linestyle='--')
    ax.axvline(min(s3v), color=COL_S3, lw=1.5, linestyle='--')
    ax.legend(facecolor='#2a2a3e', edgecolor='#555', labelcolor=COL_WHITE, fontsize=8)

    # Plot 2: S1 vs S3 scatter (color = improved/worsened)
    ax = axes[0, 1]
    style_ax(ax, "Stage 1 vs Stage 3 RMSD", "Stage 1 RMSD (Å)", "Stage 3 RMSD (Å)")
    changes = [s3d[i] - s1d[i] for i in common]
    colors  = [COL_GREEN if c < 0 else COL_RED for c in changes]
    ax.scatter(s1v, s3v, c=colors, alpha=0.6, s=20, edgecolors='none')
    lim = max(max(s1v), max(s3v)) * 1.05
    ax.plot([0, lim], [0, lim], color=COL_WHITE, lw=1, linestyle='--', alpha=0.4)
    best_i = min(s3d, key=s3d.get)
    ax.scatter(s1d.get(best_i,0), s3d[best_i], color=COL_GOLD, s=150,
               zorder=5, marker='*', label=f'Best S3: {s3d[best_i]:.3f} Å (#{best_i})')
    improved = sum(1 for c in changes if c < 0)
    ax.text(0.05, 0.95, f"↓ Improved: {improved}/{len(common)} ({100*improved/len(common):.0f}%)",
            transform=ax.transAxes, color=COL_GREEN, fontsize=9, va='top', fontweight='bold')
    ax.text(0.05, 0.88, f"↑ Worsened: {len(common)-improved}/{len(common)} ({100*(len(common)-improved)/len(common):.0f}%)",
            transform=ax.transAxes, color=COL_RED, fontsize=9, va='top', fontweight='bold')
    ax.legend(facecolor='#2a2a3e', edgecolor='#555', labelcolor=COL_WHITE, fontsize=8)

    # Plot 3: RMSD change distribution S1→S3
    ax = axes[1, 0]
    style_ax(ax, "RMSD Change S1 → S3\n(negative = improved)", "Δ RMSD (Å)", "Count")
    neg = [c for c in changes if c < 0]
    pos = [c for c in changes if c >= 0]
    if neg: ax.hist(neg, bins=20, color=COL_GREEN, alpha=0.8,
                    label=f'Improved ({len(neg)})', edgecolor=COL_WHITE, lw=0.3)
    if pos: ax.hist(pos, bins=20, color=COL_RED, alpha=0.8,
                    label=f'Worsened ({len(pos)})', edgecolor=COL_WHITE, lw=0.3)
    ax.axvline(0,                color=COL_WHITE, lw=1.5, linestyle='--', alpha=0.5)
    ax.axvline(np.mean(changes), color=COL_GOLD,  lw=2,   linestyle='--',
               label=f'Mean: {np.mean(changes):+.3f} Å')
    ax.legend(facecolor='#2a2a3e', edgecolor='#555', labelcolor=COL_WHITE, fontsize=8)

    # Plot 4: Energy vs S3 RMSD
    ax = axes[1, 1]
    style_ax(ax, "Energy vs Stage 3 RMSD", "Final Energy", "Stage 3 RMSD (Å)")
    ax.scatter(ev, s3v, c=COL_S3, alpha=0.5, s=15, edgecolors='none')
    ax.scatter(ed.get(best_i,0), s3d[best_i], color=COL_GOLD, s=200,
               zorder=5, marker='*', label=f'Best: {s3d[best_i]:.3f} Å (#{best_i})')
    if len(ev) > 2:
        corr = np.corrcoef(ev, s3v)[0, 1]
        ax.text(0.05, 0.95, f"Pearson r = {corr:.4f}",
                transform=ax.transAxes, color=COL_GOLD, fontsize=10, va='top', fontweight='bold')
    ax.legend(facecolor='#2a2a3e', edgecolor='#555', labelcolor=COL_WHITE, fontsize=8)

    tag = "3-Stage" if has_s2 else "Skip Stage 2"
    fig.suptitle(f"QTF Stage RMSD Analysis — {tag}  |  {len(common)} replicas",
                 color=COL_WHITE, fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(save_dir, "1_stage_rmsd_analysis.png")
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"[SAVED] {p}")
    plt.close()

    # ── Figure 2: Per-replica bars for all stages ──────────────────────────────
    n_plots = 3 if has_s2 else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(max(16, len(common)*0.05), 4*n_plots))
    fig.patch.set_facecolor(BG_DARK)
    x = np.arange(len(common))

    for idx, (vals, color, label) in enumerate([
        (s1v, COL_S1, "Stage 1 RMSD (COBYLA)"),
        (s2v, COL_S2, "Stage 2 RMSD (SLSQP refine)") if has_s2 else (None, None, None),
        (s3v, COL_S3, "Stage 3 RMSD (SLSQP relax)"),
    ]):
        if vals is None: continue
        ax = axes[idx] if n_plots > 1 else axes
        style_ax(ax, label, "Replica Index", "RMSD (Å)")
        ax.bar(x, vals, color=color, alpha=0.8, width=1.0)
        ax.axhline(np.mean(vals), color=COL_GOLD, lw=2, linestyle='--',
                   label=f'Mean: {np.mean(vals):.3f} Å')
        ax.axhline(min(vals), color=COL_WHITE, lw=1.5, linestyle=':',
                   label=f'Best: {min(vals):.3f} Å')
        ax.legend(facecolor='#2a2a3e', edgecolor='#555', labelcolor=COL_WHITE, fontsize=9)

    fig.suptitle("Per-Replica RMSD by Stage",
                 color=COL_WHITE, fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(save_dir, "2_per_replica_rmsd.png")
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"[SAVED] {p}")
    plt.close()

    # ── Figure 3: Stage progression per replica (line plot) ───────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(BG_DARK)
    style_ax(ax, "RMSD Progression Across Stages — All Replicas",
             "Stage", "RMSD (Å)")

    stage_labels = ['Stage 1', 'Stage 2', 'Stage 3'] if has_s2 else ['Stage 1', 'Stage 3']

    for i in common:
        if has_s2:
            pts = [s1d.get(i), s2d.get(i), s3d.get(i)]
            x_pts = [0, 1, 2]
        else:
            pts = [s1d.get(i), s3d.get(i)]
            x_pts = [0, 1]
        if any(p is None for p in pts): continue
        improved_overall = pts[-1] < pts[0]
        ax.plot(x_pts, pts, color=COL_GREEN if improved_overall else COL_RED,
                alpha=0.2, linewidth=0.8)

    # Highlight best final
    best_i = min(s3d, key=s3d.get)
    if has_s2:
        best_pts = [s1d.get(best_i), s2d.get(best_i), s3d.get(best_i)]
        x_pts = [0, 1, 2]
    else:
        best_pts = [s1d.get(best_i), s3d.get(best_i)]
        x_pts = [0, 1]
    ax.plot(x_pts, best_pts, color=COL_GOLD, linewidth=3, zorder=5,
            marker='*', markersize=12,
            label=f'Best #{best_i}: {s3d[best_i]:.3f} Å final')

    # Mean line
    if has_s2:
        means = [np.mean(s1v), np.mean(s2v), np.mean(s3v)]
        x_pts = [0, 1, 2]
    else:
        means = [np.mean(s1v), np.mean(s3v)]
        x_pts = [0, 1]
    ax.plot(x_pts, means, color=COL_WHITE, linewidth=2.5, linestyle='--',
            marker='o', markersize=8, label=f'Mean progression')

    ax.set_xticks(range(len(stage_labels)))
    ax.set_xticklabels(stage_labels, color=COL_WHITE, fontsize=11)

    for xi, (sl, mv) in enumerate(zip(stage_labels, means)):
        ax.text(xi, mv + 0.05, f'{mv:.3f}', ha='center', color=COL_WHITE,
                fontsize=9, fontweight='bold')

    patches = [mpatches.Patch(color=COL_GREEN, label='Improved overall'),
               mpatches.Patch(color=COL_RED,   label='Worsened overall')]
    ax.legend(handles=patches + [
        mpatches.Patch(color=COL_GOLD,  label=f'Best #{best_i}'),
        mpatches.Patch(color=COL_WHITE, label='Mean')],
              facecolor='#2a2a3e', edgecolor='#555', labelcolor=COL_WHITE, fontsize=9)

    plt.tight_layout()
    p = os.path.join(save_dir, "3_stage_progression.png")
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"[SAVED] {p}")
    plt.close()

    # ── Figure 4: Top 20 horizontal bars ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(BG_DARK)

    for col, (sort_dict, sort_label, sort_col) in enumerate([
        (s1d, "Stage 1", COL_S1),
        (s3d, "Stage 3", COL_S3),
    ]):
        ax = axes[col]
        style_ax(ax, f"Top 20 by {sort_label} RMSD", "RMSD (Å)", "Replica")
        top20 = sorted(sort_dict.items(), key=lambda x: x[1])[:20]
        y = np.arange(len(top20))
        ids = [f"#{i}" for i, _ in top20]

        ax.barh(y - 0.2, [s1d.get(i, 0) for i, _ in top20],
                height=0.25, color=COL_S1, alpha=0.8, label='Stage 1')
        if has_s2:
            ax.barh(y, [s2d.get(i, 0) for i, _ in top20],
                    height=0.25, color=COL_S2, alpha=0.8, label='Stage 2')
        ax.barh(y + 0.2, [s3d.get(i, 0) for i, _ in top20],
                height=0.25, color=COL_S3, alpha=0.8, label='Stage 3')

        ax.set_yticks(y)
        ax.set_yticklabels(ids, color=COL_WHITE, fontsize=8)
        ax.legend(facecolor='#2a2a3e', edgecolor='#555', labelcolor=COL_WHITE, fontsize=8)

    fig.suptitle("Top 20 Replicas — Stage 1 / 2 / 3 RMSD",
                 color=COL_WHITE, fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(save_dir, "4_top20_all_stages.png")
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"[SAVED] {p}")
    plt.close()

    print(f"\n✅ All plots saved to: {save_dir}")
    print(f"   1_stage_rmsd_analysis.png  — distributions, scatter, energy vs RMSD")
    print(f"   2_per_replica_rmsd.png     — per replica bars for each stage")
    print(f"   3_stage_progression.png    — RMSD progression lines S1→S2→S3")
    print(f"   4_top20_all_stages.png     — top 20 replicas all stages side by side")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--n_replicas',  type=int, default=400)
    parser.add_argument('--save_dir',    default='outputs/analysis')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  QTF STAGE RMSD ANALYSIS")
    print(f"  Results dir : {args.results_dir}")
    print(f"  Replicas    : {args.n_replicas}")
    print(f"{'='*70}")

    data = load_data(args.results_dir, args.n_replicas)
    if not data:
        print("No data found! Check --results_dir path.")
        return

    print_summary(data)
    print(f"\n── Generating plots ──")
    plot_all(data, args.save_dir)


if __name__ == "__main__":
    main()