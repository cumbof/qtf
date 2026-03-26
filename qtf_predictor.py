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
import QTF.runner_exports as runner


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





def __main__():
    '''
    Main entry point for running Quantum Torsion Folder.
    Produces per-run outputs in outputs/<sequence>_<forcefield>_<timestamp>/.
    '''

    # start tracking time
    time_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # parse command line arguments
    # example:
    # python qtf_predictor.py --predict "YYDPETGTWY" --reference_structure "5AWL" --average_reference_backbone False
    #   --forcefield amber --mode predict_and_compare --ensemble_size 20 --prime_strategy Random --top_k 5
    parser = argparse.ArgumentParser()

    parser.add_argument('--predict', default=None, help='target sequence to predict')
    parser.add_argument('--reference_structure', default=None, help='reference structure PDB ID for comparison')

    parser.add_argument(
        '--average_reference_backbone',
        default=False,
        type=bool,
        help=('How to select backbone from reference structure, either first model or average for NMR ensembles. '
              'Defaults to first model, which automatically works with Xray structures.')
    )

    parser.add_argument('--forcefield', default="amber", choices=["amber", "opls", "charmm", "all"],
                        help='choice of force field for scoring')
    parser.add_argument('--chi_mode', default="selective", choices=["chi1_only", "selective", "all"],
                        help='sidechain DOF exposure mode (match beam/native defaults with selective)')
    parser.add_argument('--mode', default="predict_and_compare", choices=["predict_and_compare", "predict_only"],
                        help='which mode to run script in')

    parser.add_argument('--ensemble_size', default=3, type=int, help='ensemble size')

    parser.add_argument('--top_k', default=1, type=int,
                        help='how many lowest-energy ensemble models to compare/save (default 1)')
    parser.add_argument('--top_frac', default=None, type=float,
                        help='fraction (0-1] of lowest-energy models to compare/save; overrides --top_k if set')

    parser.add_argument('--prime_strategy', default="Random",
                        choices=["Random", "mixed", "Helix", "Sheet"],
                        help='prime strategy for initialization')
    parser.add_argument('--maxiter', default=2000, type=int, help='max iterations for each trajectory')

    args = parser.parse_args()
    
    # required inputs
    sequence = args.predict
    if not sequence:
        raise ValueError("--predict <SEQUENCE> is required")

    # set the arguments that are passed in, which can then be applied to both modes
    reference_structure_pdb_id = args.reference_structure
    average_reference_backbone_mode = args.average_reference_backbone
    ensemble_size = args.ensemble_size
    prime_strategy = args.prime_strategy.lower()
    top_k = args.top_k
    top_frac = args.top_frac

    selective_chi_map = {
        "Y": ["chi1", "chi2"], "W": ["chi1", "chi2"], "F": ["chi1", "chi2"], "H": ["chi1", "chi2"],
        "D": ["chi1"], "E": ["chi1"], "N": ["chi1"], "Q": ["chi1"],
        "T": ["chi1"], "S": ["chi1"],
        "V": ["chi1"], "I": ["chi1"], "L": ["chi1"], "M": ["chi1"],
        "K": ["chi1"], "R": ["chi1"], "C": ["chi1"], "P": ["chi1"],
        "A": [], "G": [],
    }
    
    # iterate over force fields (if "all" is selected) or just the specified one
    force_fields = []
    if args.forcefield == "all":
        force_fields = ["amber", "opls", "charmm"]
    else:
        force_fields = [args.forcefield]    
    for ff in force_fields:
        force_field = ff 

        # output directories (DO NOT chdir; write explicitly)
        outputs_root = "outputs"
        os.makedirs(outputs_root, exist_ok=True)
        job_output_dir = os.path.join(outputs_root, f"{sequence}_{force_field}_{timestamp}")
        os.makedirs(job_output_dir, exist_ok=True)
        print(f"Writing outputs to: {job_output_dir}")

        # 1. Initialize Folder & Manager
        print(f"--- DIAGNOSING BACKBONE: {sequence} ---")
        if force_field == "all":
            folder = runner.QuantumBiophysicsFolder(
                sequence,
                force_field=["amber", "opls", "charmm"],
                chi_mode=args.chi_mode,
                selective_chi_map=selective_chi_map,
            )
        else:
            folder = runner.QuantumBiophysicsFolder(
                sequence,
                force_field=force_field,
                chi_mode=args.chi_mode,
                selective_chi_map=selective_chi_map,
            )
        folder.current_stage = 3

        manager = runner.EnsembleFoldingManager(folder)

        # 2. Run Ensemble
        manager.run_ensemble(n_runs=ensemble_size, max_iter=args.maxiter, prime_strategy=prime_strategy)

        # 3. Rank Ensemble by Energy and select low-energy candidates
        ranked_results = manager.get_ranked_results()
        selected_results = manager.select_top(top_k=top_k, top_frac=top_frac)
            
        if not ranked_results:
            raise RuntimeError("Ensemble produced no results.")
        if not selected_results:
            raise RuntimeError("Selection produced no results (check --top_k/--top_frac).")

        # 4. Compute reference backbone once (if needed)
        true_ca = None
        if args.mode == "predict_and_compare":
            if not reference_structure_pdb_id:
                raise ValueError("--reference_structure is required in predict_and_compare mode")
            true_ca = evaluator.get_ground_truth_backbone(reference_structure_pdb_id, average_reference_backbone_mode)

        # 5. Build per-model ranked table
        model_rows = []
        for rank, res in enumerate(selected_results, start=1):
            coords = res['coords']
            pred_ca = np.array([coords[i] for i, lbl in enumerate(folder.static_labels) if lbl[1] == 'CA'])
            sidechain_centroids = folder.compute_sidechain_centroids(coords, folder.static_labels)
            ca_pdb_path = os.path.join(job_output_dir, f"model_{rank}_ca.pdb")
            ca_centroid_pdb_path = os.path.join(job_output_dir, f"model_{rank}_ca_centroid.pdb")
            folder.save_reduced_pdb(pred_ca, filename=ca_pdb_path, sidechain_centroids=None, energy=res['energy'])
            folder.save_reduced_pdb(pred_ca, filename=ca_centroid_pdb_path, sidechain_centroids=sidechain_centroids, energy=res['energy'])
            p_e2e, p_rg = evaluator.calculate_physics_metrics(pred_ca)
            if true_ca is not None:
                n = min(len(pred_ca), len(true_ca))
                pred_ca_n = pred_ca[:n]
                true_ca_n = true_ca[:n]
                t_e2e, t_rg = evaluator.calculate_physics_metrics(true_ca_n)
                rmsd, _ = runner.StabilityAnalyzer.kabsch_rmsd(pred_ca_n, true_ca_n)
                rmsd = float(rmsd)
            else:
                t_e2e, t_rg, rmsd = np.nan, np.nan, np.nan

            # flatten energy term decomposition (if available)
            energy_terms = res.get("energy_terms") or {}
            flat_terms = {f"term_{k}": float(v) for k, v in energy_terms.items()}

            model_rows.append({
                "ensemble_id": int(res["id"]),
                "init_type": str(res["type"]),
                "energy_rank": int(rank),
                "energy": float(res["energy"]),
                "rmsd_to_reference_A": rmsd,
                "pred_e2e_A": float(p_e2e),
                "pred_rg_A": float(p_rg),
                "ref_e2e_A": float(t_e2e) if not np.isnan(t_e2e) else np.nan,
                "ref_rg_A": float(t_rg) if not np.isnan(t_rg) else np.nan,
                "rebuilt_ca_pdb_path": ca_pdb_path,
                "rebuilt_ca_centroid_pdb_path": ca_centroid_pdb_path,
                "chi_mode": args.chi_mode,
                **flat_terms,
            })

        df_models = pd.DataFrame(model_rows).sort_values(["energy"]).reset_index(drop=True)

        # Save ranked ensemble table
        ensemble_csv_path = os.path.join(job_output_dir, "ensemble_ranked.csv")
        ensemble_json_path = os.path.join(job_output_dir, "ensemble_ranked.json")
        df_models.to_csv(ensemble_csv_path, index=False)

        model_rows_json = [{k: _jsonify(v) for k, v in row.items()} for row in model_rows]
        with open(ensemble_json_path, "w") as f:
            json.dump(model_rows_json, f, indent=4)

        # 6. Best-model metrics for the one-row summary
        best_row = df_models.sort_values("energy").iloc[0]
        p_e2e = float(best_row["pred_e2e_A"])
        p_rg = float(best_row["pred_rg_A"])
        t_e2e = float(best_row["ref_e2e_A"]) if not np.isnan(best_row["ref_e2e_A"]) else np.nan
        t_rg = float(best_row["ref_rg_A"]) if not np.isnan(best_row["ref_rg_A"]) else np.nan
        rmsd_best = float(best_row["rmsd_to_reference_A"]) if not np.isnan(best_row["rmsd_to_reference_A"]) else np.nan

        # runtime
        runtime = (time.time() - time_start) / 60.0

        summary_data = {
            # metrics (best-by-energy)
            "End-to-End Dist (Å)": p_e2e,
            "End-to-End Target (Å)": t_e2e,
            "End-to-End Status": ("EXPANDED" if (not np.isnan(t_e2e) and p_e2e > t_e2e + 5) else "GOOD") if args.mode=="predict_and_compare" else None,

            "Rg (Å)": p_rg,
            "Rg Target (Å)": t_rg,
            "Rg Status": ("PUFFY" if (not np.isnan(t_rg) and p_rg > t_rg + 2) else "COMPACT") if args.mode=="predict_and_compare" else None,

            "Backbone RMSD (Å)": float(f"{rmsd_best:.3f}") if not np.isnan(rmsd_best) else None,
            "RMSD Status": ("GOOD" if (not np.isnan(rmsd_best) and rmsd_best < 2.0) else "BAD") if args.mode=="predict_and_compare" else None,

            # run-level meta/settings
            "Runtime (minutes)": float(runtime),
            "Ensemble Size": int(ensemble_size),
            "Sequence": sequence,
            "mode": args.mode,
            "Reference Structure": (reference_structure_pdb_id if reference_structure_pdb_id is not None and args.mode != "predict_only" else None),
            "Force Field": force_field,
            "Chi Mode": args.chi_mode,
            "Prime Strategy": prime_strategy,
            "Top K": int(top_k) if top_k is not None else None,
            "Top Frac": float(top_frac) if top_frac is not None else None,
            "Output Dir": job_output_dir,
        }

        # write per-run summary
        summary_csv_path = os.path.join(job_output_dir, "summary_results.csv")
        summary_json_path = os.path.join(job_output_dir, "summary_results.json")
        df_summary = pd.DataFrame([summary_data])
        df_summary.to_csv(summary_csv_path, index=False)
        with open(summary_json_path, "w") as f:
            json.dump({k: _jsonify(v) for k, v in summary_data.items()}, f, indent=4)

        # append to master results in outputs_root
        master_csv_path = os.path.join(outputs_root, "master_summary_results.csv")
        try:
            df_all = pd.read_csv(master_csv_path)
            df_all = pd.concat([df_all, df_summary], ignore_index=True)
        except FileNotFoundError:
            df_all = df_summary
        df_all.to_csv(master_csv_path, index=False)

        master_jsonl_path = os.path.join(outputs_root, "master_summary_results.jsonl")
        with open(master_jsonl_path, "a") as f:
            f.write(json.dumps({k: _jsonify(v) for k, v in summary_data.items()}) + "\n")

        print("Done.")
        print(f"- {ensemble_csv_path}")
        print(f"- {ensemble_json_path}")
        print(f"- {summary_csv_path}")
        print(f"- {summary_json_path}")


if __name__ == "__main__":
    __main__()
