#!/usr/bin/env python3
"""
QTF Single Replica Runner
==========================
Called by SLURM array — runs exactly ONE replica and saves results.
Each SLURM array task gets a unique replica_id which seeds the RNG,
ensuring all replicas start from genuinely different points.

USAGE (called by SLURM, not directly):
    python3 qtf_single_replica.py \
        --predict "YYDPETGTWY" \
        --reference_structure "5AWL" \
        --forcefield amber \
        --replica_id 42 \
        --maxiter 2000 \
        --hw_backend aer \
        --hw_shots 4096 \
        --outdir outputs/slurm_YYDPETGTWY_amber/replica_42
"""

import os
import sys
import json
import time
import argparse
import numpy as np

# ── QTF imports ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import QTF.runner_hardware2 as runner
import QTF.evaluator as evaluator


def _jsonify(x):
    try:
        import numpy as _np
        if isinstance(x, (_np.floating, _np.integer)):
            return x.item()
        if isinstance(x, _np.ndarray):
            return x.tolist()
    except Exception:
        pass
    return x


def get_hw_backend(hw_backend_name):
    if hw_backend_name == "none" or hw_backend_name is None:
        return None
    if hw_backend_name == "aer":
        from qiskit_aer import AerSimulator
        return AerSimulator()
    # Real IBM backend
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    if hw_backend_name == "least_busy":
        backends = service.backends(simulator=False, operational=True, min_num_qubits=4)
        return sorted(backends, key=lambda b: b.status().pending_jobs)[0]
    return service.backend(hw_backend_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict',             required=True)
    parser.add_argument('--reference_structure', default=None)
    parser.add_argument('--forcefield',          default='amber')
    parser.add_argument('--chi_mode',            default='selective')
    parser.add_argument('--replica_id',          type=int, required=True)
    parser.add_argument('--maxiter',             type=int, default=2000)
    parser.add_argument('--hw_backend',          default='aer')
    parser.add_argument('--hw_shots',            type=int, default=4096)
    parser.add_argument('--outdir',              default='outputs/replica')
    parser.add_argument('--average_reference_backbone', default=False, type=bool)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    time_start = time.time()

    print(f"[REPLICA {args.replica_id}] Starting...")
    print(f"  Sequence   : {args.predict}")
    print(f"  Forcefield : {args.forcefield}")
    print(f"  Backend    : {args.hw_backend}")
    print(f"  Max iter   : {args.maxiter}")

    selective_chi_map = {
        "Y": ["chi1", "chi2"], "W": ["chi1", "chi2"],
        "F": ["chi1", "chi2"], "H": ["chi1", "chi2"],
        "D": ["chi1"], "E": ["chi1"], "N": ["chi1"], "Q": ["chi1"],
        "T": ["chi1"], "S": ["chi1"], "V": ["chi1"], "I": ["chi1"],
        "L": ["chi1"], "M": ["chi1"], "K": ["chi1"], "R": ["chi1"],
        "C": ["chi1"], "P": ["chi1"], "A": [],        "G": [],
    }

    # ── Build folder ────────────────────────────────────────────────────────────
    folder = runner.QuantumBiophysicsFolder(
        args.predict,
        force_field=args.forcefield,
        chi_mode=args.chi_mode,
        selective_chi_map=selective_chi_map,
    )
    folder.current_stage = 3

    # ── Resolve backend ─────────────────────────────────────────────────────────
    hw_backend = get_hw_backend(args.hw_backend)

    # ── Determine initialization strategy from replica_id ───────────────────────
    # Cycle through helix / sheet / random so the ensemble has diversity
    strategies = ['random', 'random', 'random']
    strat = strategies[args.replica_id % 3]

    # ── Get starting parameters ─────────────────────────────────────────────────
    # Use replica_id as seed — guarantees every replica is different
    base_seed = int(__import__('hashlib').sha256(
        args.predict.encode()).hexdigest(), 16) % (2**32)
    replica_seed = base_seed + args.replica_id

    manager = runner.EnsembleFoldingManager(folder)

    if strat == 'random':
        start_params = folder.get_smart_initialization(
            n_attempts=50, seed=replica_seed)
    else:
        start_params = manager.prime_circuit(
            target_type=strat, seed=replica_seed)

    print(f"[REPLICA {args.replica_id}] Init strategy: {strat}, seed: {replica_seed}")

    # ── Fold ────────────────────────────────────────────────────────────────────
    coords, labels, bonds, tracker, opt_params, final_energy = folder.fold(
        max_iter=args.maxiter,
        initial_params=start_params,
        hw_backend=hw_backend,
        hw_shots=args.hw_shots,
    )
    # ── Save tracker to JSON ─────────────────────────────────────────────────────
    tracker_path = os.path.join(args.outdir, f"replica_{args.replica_id}_tracker.json")
    with open(tracker_path, "w") as f:
        json.dump({
            "history":       tracker.history,
            "stage_markers": tracker.stage_markers,
            "current_iter":  tracker.current_iter,
        }, f)
    print(f"[REPLICA {args.replica_id}] Tracker saved → {tracker_path}")

    # ── Save energy landscape plot ───────────────────────────────────────────────
    try:
        from qtf_landscape_viz import plot_energy_landscape
        plot_energy_landscape(
            tracker,
            sequence=args.predict,
            forcefield=args.forcefield,
            save_path=os.path.join(args.outdir, f"replica_{args.replica_id}_landscape.png"),
            show=False,
            title_extra=f"Replica {args.replica_id} | Init: {strat}"
        )
    except Exception as e:
        print(f"[REPLICA {args.replica_id}] Landscape plot failed: {e}")

    runtime = time.time() - time_start

    # ── Extract CA coordinates ───────────────────────────────────────────────────
    pred_ca = np.array([
        coords[i] for i, lbl in enumerate(folder.static_labels)
        if lbl[1] == 'CA'
    ])

    # ── RMSD against ground truth ────────────────────────────────────────────────
    rmsd = None
    t_e2e = None
    t_rg  = None
    if args.reference_structure:
        try:
            true_ca = evaluator.get_ground_truth_backbone(
                args.reference_structure, args.average_reference_backbone)
            n = min(len(pred_ca), len(true_ca))
            rmsd, _ = runner.StabilityAnalyzer.kabsch_rmsd(
                pred_ca[:n], true_ca[:n])
            rmsd = float(rmsd)
            t_e2e, t_rg = evaluator.calculate_physics_metrics(true_ca[:n])
            print(f"[REPLICA {args.replica_id}] RMSD: {rmsd:.3f} Å")
        except Exception as e:
            print(f"[REPLICA {args.replica_id}] RMSD failed: {e}")

    # ── Physics metrics ──────────────────────────────────────────────────────────
    p_e2e, p_rg = evaluator.calculate_physics_metrics(pred_ca)

    # ── Save PDB ─────────────────────────────────────────────────────────────────
    pdb_path = os.path.join(args.outdir, f"replica_{args.replica_id}.pdb")
    ca_pdb_path = os.path.join(args.outdir, f"replica_{args.replica_id}_ca.pdb")
    folder.save_pdb(coords, labels, filename=pdb_path, energy=final_energy)
    folder.save_reduced_pdb(pred_ca, filename=ca_pdb_path, energy=final_energy)

    # ── Save results JSON ────────────────────────────────────────────────────────
    result = {
        "replica_id":        args.replica_id,
        "seed":              replica_seed,
        "init_type":         strat,
        "sequence":          args.predict,
        "forcefield":        args.forcefield,
        "energy":            float(final_energy),
        "rmsd_to_reference": rmsd,
        "pred_e2e_A":        float(p_e2e),
        "pred_rg_A":         float(p_rg),
        "ref_e2e_A":         float(t_e2e) if t_e2e is not None else None,
        "ref_rg_A":          float(t_rg)  if t_rg  is not None else None,
        "runtime_s":         round(runtime, 2),
        "hw_backend":        args.hw_backend,
        "hw_shots":          args.hw_shots,
        "pdb_path":          pdb_path,
        "ca_pdb_path":       ca_pdb_path,
    }

    result_path = os.path.join(args.outdir, f"replica_{args.replica_id}_result.json")
    with open(result_path, "w") as f:
        json.dump({k: _jsonify(v) for k, v in result.items()}, f, indent=4)

    print(f"[REPLICA {args.replica_id}] Done!")
    print(f"  Energy     : {final_energy:.4f}")
    print(f"  RMSD       : {rmsd:.3f} Å" if rmsd else "  RMSD       : N/A")
    print(f"  Runtime    : {runtime:.1f}s")
    print(f"  Saved to   : {result_path}")


if __name__ == "__main__":
    main()
