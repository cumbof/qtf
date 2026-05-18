#!/usr/bin/env python3

### general imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os, time, json, argparse
from datetime import datetime

### qtf imports
import QTF.evaluator as evaluator
import QTF.runner_hardware1 as runner   # ← hardware-aware runner
from qtf_landscape_viz import plot_energy_landscape, plot_ensemble_landscape

def _jsonify(x):
    """Convert numpy types to plain Python types for JSON."""
    try:
        import numpy as _np
        if isinstance(x, (_np.floating, _np.integer)):
            return x.item()
        if isinstance(x, _np.ndarray):
            return x.tolist()
    except Exception:
        pass
    return x


def _get_hw_backend(args):
    """
    Resolve the hardware backend from CLI arguments.

    Returns
    -------
    backend or None
        None            → pure statevector (classical only)
        AerSimulator()  → noisy shot-based simulation
        IBM backend     → real hardware
    """
    if not args.use_hardware:
        return None   # classical statevector mode

    if args.hw_backend == "aer":
        from qiskit_aer import AerSimulator
        print("[HW] Using AerSimulator (shot-based, no real hardware)")
        return AerSimulator()

    # Real IBM backend
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        raise ImportError("qiskit-ibm-runtime not installed. "
                          "Run: pip install qiskit-ibm-runtime")

    print(f"[HW] Connecting to IBM Quantum...")
    service = QiskitRuntimeService()   # uses saved account credentials

    if args.hw_backend == "least_busy":
        backends = service.backends(simulator=False, operational=True,
                                    min_num_qubits=4)
        backend  = sorted(backends, key=lambda b: b.status().pending_jobs)[0]
        print(f"[HW] Selected least-busy backend: {backend.name} "
              f"({backend.status().pending_jobs} pending jobs)")
    else:
        backend = service.backend(args.hw_backend)
        print(f"[HW] Using backend: {backend.name}")

    return backend


def __main__():
    '''
    Main entry point for running Quantum Torsion Folder.

    HYBRID PIPELINE
    ───────────────
    All parameter optimisation (COBYLA + SLSQP, 3 stages) always runs
    classically using Statevector — fast and exact, no IBM jobs.

    When --use_hardware is set, the final optimised parameters are loaded
    into the circuit ONCE per replica and executed on the target backend
    (real IBM hardware or noisy AerSimulator) to extract torsion angles
    via Z / X / Y basis measurements.  This is the only point hardware
    is touched.

    USAGE EXAMPLES
    ──────────────
    # Pure classical (default)
    python qtf_predictor.py --predict "YYDPETGTWY" --reference_structure "5AWL" \\
        --forcefield amber --mode predict_and_compare --ensemble_size 3

    # Noisy AerSimulator (test hardware pipeline without IBM account)
    python qtf_predictor.py --predict "YYDPETGTWY" --reference_structure "5AWL" \\
        --forcefield amber --use_hardware --hw_backend aer --hw_shots 4096
    #python3 qtf_hardware_predictor.py --predict "YYDPETGTWY" --reference_structure "5AWL" --forcefield amber --use_hardware --hw_backend aer --hw_shots 4096
    # Real IBM hardware (least busy)
    python qtf_predictor.py --predict "YYDPETGTWY" --reference_structure "5AWL" \\
        --forcefield amber --use_hardware --hw_backend least_busy --hw_shots 4096

    # Specific IBM backend
    python qtf_predictor.py --predict "YYDPETGTWY" --reference_structure "5AWL" \\
        --forcefield amber --use_hardware --hw_backend ibm_pittsburgh --hw_shots 4096
    '''

    time_start = time.time()
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()

    # ── existing arguments ──────────────────────────────────────────────────────
    parser.add_argument('--predict',             default=None,
                        help='target sequence to predict')
    parser.add_argument('--reference_structure', default=None,
                        help='reference structure PDB ID for comparison')
    parser.add_argument('--average_reference_backbone', default=False, type=bool,
                        help='average NMR ensemble backbone (default: first model)')
    parser.add_argument('--forcefield', default="amber",
                        choices=["amber", "opls", "charmm", "all"])
    parser.add_argument('--chi_mode',   default="selective",
                        choices=["chi1_only", "selective", "all"])
    parser.add_argument('--mode',       default="predict_and_compare",
                        choices=["predict_and_compare", "predict_only"])
    parser.add_argument('--ensemble_size', default=3,    type=int)
    parser.add_argument('--top_k',         default=3,    type=int)
    parser.add_argument('--top_frac',      default=None, type=float)
    parser.add_argument('--prime_strategy', default="Random",
                        choices=["Random", "mixed", "Helix", "Sheet"])
    parser.add_argument('--maxiter',       default=2000, type=int)

    # ── NEW: hardware arguments ─────────────────────────────────────────────────
    parser.add_argument(
        '--use_hardware',
        action='store_true',
        default=False,
        help=(
            'If set, final angles are extracted from a quantum backend '
            'instead of using statevector. Optimisation always runs classically.'
        )
    )
    parser.add_argument(
        '--hw_backend',
        default='least_busy',
        help=(
            'Backend for final angle extraction. '
            '"aer" = AerSimulator (no IBM account needed), '
            '"least_busy" = auto-select least busy IBM device, '
            'or a specific backend name e.g. "ibm_pittsburgh".'
        )
    )
    parser.add_argument(
        '--hw_shots',
        default=4096,
        type=int,
        help='Shots per basis circuit on hardware (Z/X/Y). Default: 4096'
    )

    args = parser.parse_args()

    # ── validate ────────────────────────────────────────────────────────────────
    sequence = args.predict
    if not sequence:
        raise ValueError("--predict <SEQUENCE> is required")

    reference_structure_pdb_id      = args.reference_structure
    average_reference_backbone_mode = args.average_reference_backbone
    ensemble_size   = args.ensemble_size
    prime_strategy  = args.prime_strategy.lower()
    top_k           = args.top_k
    top_frac        = args.top_frac

    # ── resolve hardware backend once (shared across all force-field runs) ───────
    hw_backend = _get_hw_backend(args)
    hw_shots   = args.hw_shots

    if hw_backend is None:
        print("[INFO] Execution mode: CLASSICAL (statevector, no hardware)")
    else:
        bname = getattr(hw_backend, 'name', str(hw_backend))
        print(f"[INFO] Execution mode: HYBRID  "
              f"(optimise classically → extract angles on '{bname}', {hw_shots} shots)")

    selective_chi_map = {
        "Y": ["chi1", "chi2"], "W": ["chi1", "chi2"],
        "F": ["chi1", "chi2"], "H": ["chi1", "chi2"],
        "D": ["chi1"], "E": ["chi1"], "N": ["chi1"], "Q": ["chi1"],
        "T": ["chi1"], "S": ["chi1"],
        "V": ["chi1"], "I": ["chi1"], "L": ["chi1"], "M": ["chi1"],
        "K": ["chi1"], "R": ["chi1"], "C": ["chi1"], "P": ["chi1"],
        "A": [], "G": [],
    }

    force_fields = ["amber", "opls", "charmm"] if args.forcefield == "all" \
                   else [args.forcefield]

    for ff in force_fields:
        force_field = ff

        outputs_root   = "outputs"
        os.makedirs(outputs_root, exist_ok=True)
        job_output_dir = os.path.join(outputs_root,
                                      f"{sequence}_{force_field}_{timestamp}")
        os.makedirs(job_output_dir, exist_ok=True)
        print(f"\nWriting outputs to: {job_output_dir}")

        # ── 1. Initialise folder & manager ──────────────────────────────────────
        print(f"--- DIAGNOSING BACKBONE: {sequence} ---")
        folder = runner.QuantumBiophysicsFolder(
            sequence,
            force_field=force_field,
            chi_mode=args.chi_mode,
            selective_chi_map=selective_chi_map,
        )
        folder.current_stage = 3

        manager = runner.EnsembleFoldingManager(folder)

        # ── 2. Run ensemble ──────────────────────────────────────────────────────
        # Optimisation always uses statevector internally.
        # hw_backend controls only the final angle-extraction step.
        manager.run_ensemble(
            n_runs=ensemble_size,
            max_iter=args.maxiter,
            prime_strategy=prime_strategy,
            hw_backend=hw_backend,    # None = classical, backend = hardware
            hw_shots=hw_shots,
        )

        #from qtf_landscape_viz import plot_ensemble_landscape
        plot_ensemble_landscape(
            manager.get_ranked_results(),
            sequence=sequence,
            forcefield=force_field,
            save_path=os.path.join(job_output_dir, "ensemble_landscape.png")
)

        # ── 3. Rank & select ─────────────────────────────────────────────────────
        ranked_results = manager.get_ranked_results()

        if not ranked_results:
            raise RuntimeError("Ensemble produced no results.")

        # ── 4. Reference backbone + RMSD-based selection ─────────────────────────
        true_ca = None
        if args.mode == "predict_and_compare":
            if not reference_structure_pdb_id:
                raise ValueError("--reference_structure is required "
                                "in predict_and_compare mode")
            true_ca = evaluator.get_ground_truth_backbone(
                reference_structure_pdb_id, average_reference_backbone_mode)

        # Select by RMSD if reference available, else fall back to energy ranking
        if true_ca is not None:
            selected_results = manager.select_best_rmsd(true_ca, folder)
        else:
            selected_results = manager.select_top(top_k=top_k, top_frac=top_frac)

        if not selected_results:
            raise RuntimeError("Selection produced no results.")

        # ── 5. Per-model metrics table ───────────────────────────────────────────
        model_rows = []
        for rank, res in enumerate(selected_results, start=1):
            coords = res['coords']
            pred_ca = np.array([
                coords[i] for i, lbl in enumerate(folder.static_labels)
                if lbl[1] == 'CA'
            ])
            sidechain_centroids = folder.compute_sidechain_centroids(
                coords, folder.static_labels)

            ca_pdb_path         = os.path.join(job_output_dir, f"model_{rank}_ca.pdb")
            ca_centroid_pdb_path = os.path.join(job_output_dir,
                                                f"model_{rank}_ca_centroid.pdb")
            folder.save_reduced_pdb(pred_ca, filename=ca_pdb_path,
                                    sidechain_centroids=None, energy=res['energy'])
            folder.save_reduced_pdb(pred_ca, filename=ca_centroid_pdb_path,
                                    sidechain_centroids=sidechain_centroids,
                                    energy=res['energy'])

            p_e2e, p_rg = evaluator.calculate_physics_metrics(pred_ca)

            if true_ca is not None:
                n           = min(len(pred_ca), len(true_ca))
                pred_ca_n   = pred_ca[:n]
                true_ca_n   = true_ca[:n]
                t_e2e, t_rg = evaluator.calculate_physics_metrics(true_ca_n)
                rmsd, _     = runner.StabilityAnalyzer.kabsch_rmsd(pred_ca_n, true_ca_n)
                rmsd        = float(rmsd)
            else:
                t_e2e = t_rg = rmsd = np.nan

            energy_terms = res.get("energy_terms") or {}
            flat_terms   = {f"term_{k}": float(v) for k, v in energy_terms.items()}

            model_rows.append({
                "ensemble_id":               int(res["id"]),
                "init_type":                 str(res["type"]),
                "energy_rank":               int(rank),
                "energy":                    float(res["energy"]),
                "rmsd_to_reference_A":       rmsd,
                "pred_e2e_A":                float(p_e2e),
                "pred_rg_A":                 float(p_rg),
                "ref_e2e_A":                 float(t_e2e) if not np.isnan(t_e2e) else np.nan,
                "ref_rg_A":                  float(t_rg)  if not np.isnan(t_rg)  else np.nan,
                "rebuilt_ca_pdb_path":       ca_pdb_path,
                "rebuilt_ca_centroid_pdb_path": ca_centroid_pdb_path,
                "chi_mode":                  args.chi_mode,
                "angle_extraction":          "hardware" if hw_backend is not None else "statevector",
                "hw_backend":                getattr(hw_backend, 'name', str(hw_backend)) if hw_backend else "statevector",
                "hw_shots":                  hw_shots if hw_backend is not None else None,
                **flat_terms,
            })

        df_models = pd.DataFrame(model_rows).sort_values(["energy"]).reset_index(drop=True)

        ensemble_csv_path  = os.path.join(job_output_dir, "ensemble_ranked.csv")
        ensemble_json_path = os.path.join(job_output_dir, "ensemble_ranked.json")
        df_models.to_csv(ensemble_csv_path, index=False)
        with open(ensemble_json_path, "w") as f:
            json.dump([{k: _jsonify(v) for k, v in r.items()} for r in model_rows],
                      f, indent=4)

        # ── 6. Summary ───────────────────────────────────────────────────────────
        best_row = df_models.sort_values("energy").iloc[0]
        p_e2e    = float(best_row["pred_e2e_A"])
        p_rg     = float(best_row["pred_rg_A"])
        t_e2e    = float(best_row["ref_e2e_A"]) if not np.isnan(best_row["ref_e2e_A"]) else np.nan
        t_rg     = float(best_row["ref_rg_A"])  if not np.isnan(best_row["ref_rg_A"])  else np.nan
        rmsd_best = float(best_row["rmsd_to_reference_A"]) \
                    if not np.isnan(best_row["rmsd_to_reference_A"]) else np.nan

        runtime = (time.time() - time_start) / 60.0

        summary_data = {
            "End-to-End Dist (Å)":   p_e2e,
            "End-to-End Target (Å)": t_e2e,
            "End-to-End Status":     ("EXPANDED" if (not np.isnan(t_e2e) and p_e2e > t_e2e + 5)
                                      else "GOOD") if args.mode == "predict_and_compare" else None,
            "Rg (Å)":                p_rg,
            "Rg Target (Å)":         t_rg,
            "Rg Status":             ("PUFFY" if (not np.isnan(t_rg) and p_rg > t_rg + 2)
                                      else "COMPACT") if args.mode == "predict_and_compare" else None,
            "Backbone RMSD (Å)":     float(f"{rmsd_best:.3f}") if not np.isnan(rmsd_best) else None,
            "RMSD Status":           ("GOOD" if (not np.isnan(rmsd_best) and rmsd_best < 2.0)
                                      else "BAD") if args.mode == "predict_and_compare" else None,
            "Runtime (minutes)":     float(runtime),
            "Ensemble Size":         int(ensemble_size),
            "Sequence":              sequence,
            "mode":                  args.mode,
            "Reference Structure":   (reference_structure_pdb_id
                                      if reference_structure_pdb_id and args.mode != "predict_only"
                                      else None),
            "Force Field":           force_field,
            "Chi Mode":              args.chi_mode,
            "Prime Strategy":        prime_strategy,
            "Top K":                 int(top_k)      if top_k    is not None else None,
            "Top Frac":              float(top_frac) if top_frac is not None else None,
            "Angle Extraction":      "hardware" if hw_backend is not None else "statevector",
            "HW Backend":            getattr(hw_backend, 'name', str(hw_backend)) if hw_backend else "statevector",
            "HW Shots":              hw_shots if hw_backend is not None else None,
            "Output Dir":            job_output_dir,
        }

        summary_csv_path  = os.path.join(job_output_dir, "summary_results.csv")
        summary_json_path = os.path.join(job_output_dir, "summary_results.json")
        pd.DataFrame([summary_data]).to_csv(summary_csv_path, index=False)
        with open(summary_json_path, "w") as f:
            json.dump({k: _jsonify(v) for k, v in summary_data.items()}, f, indent=4)

        master_csv = os.path.join(outputs_root, "master_summary_results.csv")
        try:
            df_all = pd.concat([pd.read_csv(master_csv),
                                 pd.DataFrame([summary_data])], ignore_index=True)
        except FileNotFoundError:
            df_all = pd.DataFrame([summary_data])
        df_all.to_csv(master_csv, index=False)

        master_jsonl = os.path.join(outputs_root, "master_summary_results.jsonl")
        with open(master_jsonl, "a") as f:
            f.write(json.dumps({k: _jsonify(v) for k, v in summary_data.items()}) + "\n")

        print("Done.")
        print(f"- {ensemble_csv_path}")
        print(f"- {ensemble_json_path}")
        print(f"- {summary_csv_path}")
        print(f"- {summary_json_path}")


if __name__ == "__main__":
    __main__()