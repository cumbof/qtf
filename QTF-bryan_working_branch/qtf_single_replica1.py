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

    # Skip stage 2:
    python3 qtf_single_replica.py \
        --predict "YYDPETGTWY" \
        --reference_structure "5AWL" \
        --forcefield amber \
        --replica_id 42 \
        --skip_stage2 \
        --outdir outputs/slurm_YYDPETGTWY_amber_no_s2/replica_42
"""

import os
import sys
import json
import time
import argparse
import numpy as np

# ── QTF imports ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import QTF.runner_hardware3 as runner
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
    parser.add_argument('--skip_stage2',         action='store_true', default=False,
                        help='Skip Stage 2 SLSQP — go directly Stage1→Stage3')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    time_start = time.time()

    print(f"[REPLICA {args.replica_id}] Starting...")
    print(f"  Sequence     : {args.predict}")
    print(f"  Forcefield   : {args.forcefield}")
    print(f"  Backend      : {args.hw_backend}")
    print(f"  Max iter     : {args.maxiter}")
    print(f"  Skip Stage 2 : {args.skip_stage2}")

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

    # ── Fetch ground truth CA once (for per-stage RMSD tracking) ────────────────
    true_ca = None
    if args.reference_structure:
        try:
            true_ca = evaluator.get_ground_truth_backbone(
                args.reference_structure, args.average_reference_backbone)
            print(f"[REPLICA {args.replica_id}] Ground truth loaded — "
                  f"{len(true_ca)} CA atoms")
        except Exception as e:
            print(f"[REPLICA {args.replica_id}] Ground truth load failed: {e}")

    # ── Determine initialization strategy ───────────────────────────────────────
    strategies   = ['random', 'random', 'random']
    strat        = strategies[args.replica_id % 3]
    base_seed    = int(__import__('hashlib').sha256(
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

    # ── Fold with per-stage RMSD tracking ───────────────────────────────────────
    coords, labels, bonds, tracker, opt_params, final_energy = folder.fold(
        max_iter=args.maxiter,
        initial_params=start_params,
        hw_backend=hw_backend,
        hw_shots=args.hw_shots,
        true_ca=true_ca,              # ← ground truth for per-stage RMSD
        skip_stage2=args.skip_stage2, # ← skip stage 2 flag
    )

    # ── Save statevector PDB (true 1.708 Å structure) ────────────────────────────
    sv_angles   = folder._get_angles(opt_params, mode="statevector")
    sv_coords, sv_labels, _ = folder.build_full_structure(sv_angles)
    sv_ca       = np.array([
        sv_coords[i] for i, lbl in enumerate(folder.static_labels)
        if lbl[1] == "CA"
    ])
    if true_ca is not None:
        P = sv_ca - sv_ca.mean(0)
        Q = true_ca - true_ca.mean(0)
        H = P.T @ Q
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(U) * np.linalg.det(Vt) < 0
        if d:
            S[-1] = -S[-1]; U[:,-1] = -U[:,-1]
        R = U @ Vt
        sv_ca_aligned = P @ R + true_ca.mean(0)
    else:
        sv_ca_aligned = sv_ca
    sv_pdb_path = os.path.join(args.outdir, f"replica_{args.replica_id}_sv.pdb")
    with open(sv_pdb_path, "w") as f:
        f.write(f"REMARK   STATEVECTOR ALIGNED\n")
        for k, (pos, res) in enumerate(zip(sv_ca_aligned, args.predict)):
            f.write(f"ATOM  {k+1:>5}  CA  {res:>3} A{k+1:>4}    "
                    f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           C\n")
        f.write("END\n")
    print(f"[REPLICA {args.replica_id}] Statevector PDB saved → {sv_pdb_path}")

    # ── Save tracker to JSON ─────────────────────────────────────────────────────
    tracker_path = os.path.join(args.outdir,
                                f"replica_{args.replica_id}_tracker.json")
    with open(tracker_path, "w") as f:
        json.dump({
            "history":       tracker.history,
            "stage_markers": [[m[0], m[1]] for m in tracker.stage_markers],
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
            save_path=os.path.join(args.outdir,
                                   f"replica_{args.replica_id}_landscape.png"),
            show=False,
            title_extra=f"Replica {args.replica_id} | Init: {strat} | "
                        f"skip_s2={args.skip_stage2}"
        )
    except Exception as e:
        print(f"[REPLICA {args.replica_id}] Landscape plot failed: {e}")

    runtime = time.time() - time_start

    # ── Extract CA coordinates ───────────────────────────────────────────────────
    pred_ca = np.array([
        coords[i] for i, lbl in enumerate(folder.static_labels)
        if lbl[1] == 'CA'
    ])

    # ── Final RMSD against ground truth ─────────────────────────────────────────
    rmsd  = None
    t_e2e = None
    t_rg  = None
    if true_ca is not None:
        try:
            n       = min(len(pred_ca), len(true_ca))
            rmsd, _ = runner.StabilityAnalyzer.kabsch_rmsd(
                pred_ca[:n], true_ca[:n])
            rmsd    = float(rmsd)
            t_e2e, t_rg = evaluator.calculate_physics_metrics(true_ca[:n])
            print(f"[REPLICA {args.replica_id}] Final RMSD: {rmsd:.3f} Å")
        except Exception as e:
            print(f"[REPLICA {args.replica_id}] Final RMSD failed: {e}")

    # ── Physics metrics ──────────────────────────────────────────────────────────
    p_e2e, p_rg = evaluator.calculate_physics_metrics(pred_ca)

    # ── Save PDB ─────────────────────────────────────────────────────────────────
    pdb_path    = os.path.join(args.outdir, f"replica_{args.replica_id}.pdb")
    ca_pdb_path = os.path.join(args.outdir, f"replica_{args.replica_id}_ca.pdb")
    folder.save_pdb(coords, labels, filename=pdb_path, energy=final_energy)
    folder.save_reduced_pdb(pred_ca, filename=ca_pdb_path, energy=final_energy)

    # ── Pull per-stage RMSDs saved by fold() ────────────────────────────────────
    stage_rmsds = getattr(folder, 'stage_rmsds', {
        's1': None, 's2': None, 's3': None})

    # ── Save results JSON ────────────────────────────────────────────────────────
    result = {
        "replica_id":        args.replica_id,
        "seed":              replica_seed,
        "init_type":         strat,
        "sequence":          args.predict,
        "forcefield":        args.forcefield,
        "skip_stage2":       args.skip_stage2,

        # energies
        "energy":            float(final_energy),

        # final RMSD
        "rmsd_to_reference": rmsd,

        # per-stage RMSDs ← NEW
        "rmsd_stage1":       stage_rmsds.get('s1'),
        "rmsd_stage2":       stage_rmsds.get('s2'),
        "rmsd_stage3":       stage_rmsds.get('s3'),

        # physics
        "pred_e2e_A":        float(p_e2e),
        "pred_rg_A":         float(p_rg),
        "ref_e2e_A":         float(t_e2e) if t_e2e is not None else None,
        "ref_rg_A":          float(t_rg)  if t_rg  is not None else None,

        # meta
        "runtime_s":         round(runtime, 2),
        "hw_backend":        args.hw_backend,
        "hw_shots":          args.hw_shots,
        "pdb_path":          pdb_path,
        "ca_pdb_path":       ca_pdb_path,
    }

    result_path = os.path.join(args.outdir,
                               f"replica_{args.replica_id}_result.json")
    with open(result_path, "w") as f:
        json.dump({k: _jsonify(v) for k, v in result.items()}, f, indent=4)

    # ── Final summary print ──────────────────────────────────────────────────────
    print(f"\n[REPLICA {args.replica_id}] ✓ Done!")
    print(f"  Energy       : {final_energy:.4f}")
    print(f"  RMSD final   : {rmsd:.3f} Å"   if rmsd else   "  RMSD final   : N/A")
    print(f"  RMSD stage1  : {stage_rmsds['s1']:.3f} Å"
          if stage_rmsds.get('s1') else "  RMSD stage1  : N/A")
    if args.skip_stage2:
        print(f"  RMSD stage2  : skipped")
    else:
        print(f"  RMSD stage2  : {stage_rmsds['s2']:.3f} Å"
              if stage_rmsds.get('s2') else "  RMSD stage2  : N/A")
    print(f"  RMSD stage3  : {stage_rmsds['s3']:.3f} Å"
          if stage_rmsds.get('s3') else "  RMSD stage3  : N/A")
    print(f"  Runtime      : {runtime:.1f}s")
    print(f"  Saved to     : {result_path}")


if __name__ == "__main__":
    main()