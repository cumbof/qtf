#!/usr/bin/env python3
"""
QTF Results Collector
======================
Run this AFTER all SLURM array jobs finish.
Collects all replica JSON files, merges them, sorts by RMSD,
and prints a summary table.

USAGE:
    python3 qtf_collect_results.py \
        --results_dir outputs/slurm_YYDPETGTWY_amber \
        --n_replicas 300
"""

import os
import json
import argparse
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True,
                        help='Directory containing replica_X subdirs')
    parser.add_argument('--n_replicas',  type=int, default=300)
    parser.add_argument('--top_k',       type=int, default=10,
                        help='Print top K by RMSD')
    args = parser.parse_args()

    # ── Collect all result JSONs ─────────────────────────────────────────────────
    results = []
    missing = []

    for i in range(args.n_replicas):
        result_path = os.path.join(
            args.results_dir, f"replica_{i}", f"replica_{i}_result.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                results.append(json.load(f))
        else:
            missing.append(i)

    print(f"\n=== QTF ENSEMBLE RESULTS ===")
    print(f"  Collected : {len(results)} / {args.n_replicas} replicas")
    if missing:
        print(f"  Missing   : {len(missing)} replicas → {missing[:10]}{'...' if len(missing)>10 else ''}")

    if not results:
        print("No results found!")
        return

    # ── Sort by RMSD ─────────────────────────────────────────────────────────────
    has_rmsd   = [r for r in results if r.get('rmsd_to_reference') is not None]
    no_rmsd    = [r for r in results if r.get('rmsd_to_reference') is None]

    by_rmsd    = sorted(has_rmsd,  key=lambda x: x['rmsd_to_reference'])
    by_energy  = sorted(results,   key=lambda x: x['energy'])

    # ── Summary stats ────────────────────────────────────────────────────────────
    energies = [r['energy'] for r in results]
    print(f"\n── Energy Stats ──────────────────────────────────────")
    print(f"  Best    : {min(energies):.3f}")
    print(f"  Worst   : {max(energies):.3f}")
    print(f"  Mean    : {np.mean(energies):.3f}")
    print(f"  Std     : {np.std(energies):.3f}")

    if has_rmsd:
        rmsds = [r['rmsd_to_reference'] for r in has_rmsd]
        print(f"\n── RMSD Stats (vs ground truth) ──────────────────────")
        print(f"  Best    : {min(rmsds):.3f} Å")
        print(f"  Worst   : {max(rmsds):.3f} Å")
        print(f"  Mean    : {np.mean(rmsds):.3f} Å")
        print(f"  Std     : {np.std(rmsds):.3f} Å")

        # ── Top K by RMSD ────────────────────────────────────────────────────────
        print(f"\n── Top {args.top_k} by RMSD ───────────────────────────────────")
        print(f"  {'Replica':>8} {'Init':>8} {'Energy':>10} {'RMSD (Å)':>10} {'E2E (Å)':>9} {'Rg (Å)':>8}")
        print("  " + "-" * 60)
        for r in by_rmsd[:args.top_k]:
            print(f"  #{r['replica_id']:>6}   {r['init_type']:>8}  "
                  f"{r['energy']:>10.2f}  {r['rmsd_to_reference']:>8.3f} Å  "
                  f"{r.get('pred_e2e_A', 0):>7.2f}  {r.get('pred_rg_A', 0):>6.2f}")

        # ── Check if best RMSD == best energy ────────────────────────────────────
        best_rmsd_replica   = by_rmsd[0]['replica_id']
        best_energy_replica = by_energy[0]['replica_id']
        print(f"\n── Correlation Check ─────────────────────────────────")
        print(f"  Best RMSD   : Replica #{best_rmsd_replica} "
              f"(RMSD={by_rmsd[0]['rmsd_to_reference']:.3f} Å, "
              f"E={by_rmsd[0]['energy']:.2f})")
        print(f"  Best Energy : Replica #{best_energy_replica} "
              f"(E={by_energy[0]['energy']:.2f}, "
              f"RMSD={by_energy[0].get('rmsd_to_reference', 'N/A')})")
        if best_rmsd_replica == best_energy_replica:
            print(f"  ✅ Best RMSD and best energy are the SAME replica!")
        else:
            print(f"  ⚠️  Best RMSD and best energy are DIFFERENT replicas.")

    # ── Save merged CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(args.results_dir, "all_replicas.csv")
    if HAS_PANDAS:
        import pandas as pd
        df = pd.DataFrame(results).sort_values('rmsd_to_reference' if has_rmsd else 'energy')
        df.to_csv(csv_path, index=False)
        print(f"\n  Saved merged CSV → {csv_path}")
    else:
        # Manual CSV
        keys = list(results[0].keys())
        with open(csv_path, 'w') as f:
            f.write(','.join(keys) + '\n')
            for r in (by_rmsd if has_rmsd else by_energy):
                f.write(','.join(str(r.get(k, '')) for k in keys) + '\n')
        print(f"\n  Saved merged CSV → {csv_path}")

    # ── Runtime stats ────────────────────────────────────────────────────────────
    runtimes = [r.get('runtime_s', 0) for r in results]
    print(f"\n── Runtime Stats ──────────────────────────────────────")
    print(f"  Mean per replica : {np.mean(runtimes)/60:.1f} min")
    print(f"  Max per replica  : {max(runtimes)/60:.1f} min")
    print(f"  Total (serial)   : {sum(runtimes)/3600:.1f} hours")
    print(f"  Actual (parallel): ~{max(runtimes)/60:.1f} min  ← SLURM parallel!")
    print("=" * 55)


if __name__ == "__main__":
    main()
