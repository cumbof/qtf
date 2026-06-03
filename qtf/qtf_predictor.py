#!/usr/bin/env python3

### general imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os, time, json, argparse
from datetime import datetime

### qtf imports
import qtf.runner as runner
import qtf.workflow_utils as utils
import qtf.qtf_gromacs as qtf_gromacs


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



def core_ca_slice(coords: np.ndarray) -> np.ndarray:
    """Return CA coordinates excluding flexible terminal residues.

    Uses residues 2..N-1 (1-indexed), i.e. drops the first and last CA.
    Falls back to all coordinates for very short chains.
    """
    arr = np.asarray(coords)
    if arr.shape[0] > 2:
        return arr[1:-1]
    return arr

def core_ca_range_metadata(n_residues: int) -> dict:
    use_core = n_residues > 2
    return {
        "rmsd_ca_excludes_terminal_residues": bool(use_core),
        "rmsd_ca_start_residue_1indexed": 2 if use_core else 1,
        "rmsd_ca_end_residue_1indexed": (n_residues - 1) if use_core else n_residues,
        "rmsd_ca_n_aligned": (n_residues - 2) if use_core else n_residues,
    }


def adjacent_heavy_clash_metrics(coords, labels, min_allowed_A: float = 1.35, threshold_frac: float = 0.55):
    """Detect the closest heavy-atom contact between adjacent residues."""
    coords = np.asarray(coords, dtype=float)
    res_ids = np.array([int(r) for r, _, _ in labels], dtype=int)
    atom_names = [str(atom) for _, atom, _ in labels]
    heavy_mask = np.array([str(elem).upper() != "H" and not str(atom).upper().startswith("H") for _, atom, elem in labels], dtype=bool)

    elem_radii = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}
    radii = np.array([elem_radii.get(str(elem).upper()[0], 1.75) for _, _, elem in labels], dtype=float)

    idx = np.where(heavy_mask)[0]
    if idx.size < 2:
        return {
            "local_clash_min_heavy_dist": np.nan,
            "local_clash_flag": False,
            "local_clash_pair": "",
        }

    best_dist = float("inf")
    best_pair = ""
    worst_margin = float("inf")
    worst_pair = ""
    any_clash = False
    for ii, i in enumerate(idx[:-1]):
        ri = res_ids[i]
        ai = atom_names[i]
        for j in idx[ii + 1:]:
            if abs(ri - res_ids[j]) != 1:
                continue
            aj = atom_names[j]
            if ai == "C" and aj == "N":
                continue
            d = float(np.linalg.norm(coords[i] - coords[j]))
            threshold_A = max(min_allowed_A, threshold_frac * (float(radii[i]) + float(radii[j])))
            if d < best_dist:
                best_dist = d
                best_pair = f"{labels[i][0]}:{labels[i][1]}-{labels[j][0]}:{labels[j][1]}"
            margin = d - threshold_A
            if margin < worst_margin:
                worst_margin = margin
                worst_pair = f"{labels[i][0]}:{labels[i][1]}-{labels[j][0]}:{labels[j][1]}"
            if d < threshold_A:
                any_clash = True

    if best_dist == float("inf"):
        return {
            "local_clash_min_heavy_dist": np.nan,
            "local_clash_flag": False,
            "local_clash_pair": "",
        }

    return {
        "local_clash_min_heavy_dist": float(best_dist),
        "local_clash_flag": bool(any_clash),
        "local_clash_pair": worst_pair if any_clash else best_pair,
    }


def nonlocal_heavy_clash_metrics(coords, labels, min_allowed_A: float = 1.75):
    """Detect the closest heavy-atom contact between residues separated by at least 2."""
    coords = np.asarray(coords, dtype=float)
    res_ids = np.array([int(r) for r, _, _ in labels], dtype=int)
    heavy_mask = np.array([str(elem).upper() != "H" and not str(atom).upper().startswith("H") for _, atom, elem in labels], dtype=bool)

    idx = np.where(heavy_mask)[0]
    if idx.size < 2:
        return {
            "clash_min_heavy_dist": np.nan,
            "clash_flag": False,
            "clash_pair": "",
        }

    best_dist = float("inf")
    best_pair = ""
    for ii, i in enumerate(idx[:-1]):
        ri = res_ids[i]
        for j in idx[ii + 1:]:
            if abs(ri - res_ids[j]) < 2:
                continue
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < best_dist:
                best_dist = d
                best_pair = f"{labels[i][0]}:{labels[i][1]}-{labels[j][0]}:{labels[j][1]}"

    if best_dist == float("inf"):
        return {
            "clash_min_heavy_dist": np.nan,
            "clash_flag": False,
            "clash_pair": "",
        }

    return {
        "clash_min_heavy_dist": float(best_dist),
        "clash_flag": bool(best_dist < float(min_allowed_A)),
        "clash_pair": best_pair,
    }


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
    parser.add_argument('--reference_pdb', default=None, help='optional local reference PDB path for comparison')
    parser.add_argument('--rmsd_mode', default='ca', choices=['ca', 'heavy'],
                        help='RMSD atom selection: all CA atoms or all heavy atoms')
    parser.add_argument('--rmsd_residue_scope', default='core', choices=['core', 'all'],
                        help='Residue range used for RMSD; core excludes the first and last residues')

    parser.add_argument(
        '--average_reference_backbone',
        default=False,
        type=bool,
        help=('How to select backbone from reference structure, either first model or average for NMR ensembles. '
              'Defaults to first model, which automatically works with Xray structures.')
    )

    parser.add_argument('--forcefield', default="amber", choices=["amber", "opls", "charmm", "all"],
                        help='choice of force field for scoring')
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
    parser.add_argument('--energy_backend', default='custom', choices=['custom', 'rosetta', 'openmm'],
                        help='stage-3 scoring backend used by runner')
    parser.add_argument('--use_e2e_constraint', default=1, type=int,
                        help='1 to use length-scaled E2E constraint in custom scorer, 0 to disable')
    parser.add_argument('--e2e_scale', default=1.0, type=float,
                        help='multiplier for E2E constraint if enabled')
    parser.add_argument('--rosetta_repack', default=0, type=int)
    parser.add_argument('--rosetta_fa_min', default=0, type=int)
    parser.add_argument('--rosetta_cen_min', default=0, type=int)
    parser.add_argument('--gromacs_minimize', default=None, type=int,
                        help='1 to add hydrogens/topology and minimize each saved full PDB with GROMACS')
    parser.add_argument('--gromacs_forcefield', default='amber99sb-ildn')
    parser.add_argument('--gromacs_water', default='tip3p')
    parser.add_argument('--gromacs_nsteps', default=5000, type=int,
                        help='maximum GROMACS minimization steps; minimization may stop earlier at --gromacs_emtol')
    parser.add_argument('--gromacs_emtol', default=100.0, type=float,
                        help='GROMACS steepest-descent force tolerance in kJ/mol/nm')
    parser.add_argument('--gromacs_maxwarn', default=2, type=int)
    parser.add_argument('--gromacs_rerank', default=None, type=int,
                        help='1 to rerank final minimized outputs by GROMACS potential energy when available')

    args = parser.parse_args()
    if args.gromacs_minimize is None:
        args.gromacs_minimize = 1
    if args.gromacs_rerank is None:
        args.gromacs_rerank = 1

    # required inputs
    sequence = args.predict
    if not sequence:
        raise ValueError("--predict <SEQUENCE> is required")

    # set the arguments that are passed in, which can then be applied to both modes
    reference_structure_pdb_id = args.reference_structure
    reference_pdb_path = args.reference_pdb
    average_reference_backbone_mode = args.average_reference_backbone
    ensemble_size = args.ensemble_size
    prime_strategy = args.prime_strategy.lower()
    top_k = args.top_k
    top_frac = args.top_frac
    
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
            folder = utils.make_folder(
                sequence=sequence,
                force_field=["amber", "opls", "charmm"],
                energy_backend=args.energy_backend,
                use_e2e_constraint=bool(args.use_e2e_constraint),
                e2e_scale=args.e2e_scale,
                rosetta_repack=bool(args.rosetta_repack),
                rosetta_fa_min=bool(args.rosetta_fa_min),
                rosetta_cen_min=bool(args.rosetta_cen_min),
            )
        else:
            folder = utils.make_folder(
                sequence=sequence,
                force_field=force_field,
                energy_backend=args.energy_backend,
                use_e2e_constraint=bool(args.use_e2e_constraint),
                e2e_scale=args.e2e_scale,
                rosetta_repack=bool(args.rosetta_repack),
                rosetta_fa_min=bool(args.rosetta_fa_min),
                rosetta_cen_min=bool(args.rosetta_cen_min),
            )

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
        true_rmsd_coords = None
        true_rmsd_labels = None
        true_ca = None
        if args.mode == "predict_and_compare":
            reference_source = reference_pdb_path or reference_structure_pdb_id
            if not reference_source:
                raise ValueError("--reference_structure or --reference_pdb is required in predict_and_compare mode")
            true_rmsd_coords, true_rmsd_labels, ref_rmsd_meta = utils.load_reference_rmsd_coords(
                reference_source,
                args.rmsd_mode,
                average_backbone=average_reference_backbone_mode,
            )
            true_ca, _, _ = utils.load_reference_rmsd_coords(
                reference_source,
                "ca",
                average_backbone=average_reference_backbone_mode,
            )

        # 5. Build per-model ranked table
        model_rows = []
        os.makedirs(os.path.join(job_output_dir, "raw_pdbs"), exist_ok=True)
        if bool(args.gromacs_minimize):
            os.makedirs(os.path.join(job_output_dir, "gromacs_pdbs"), exist_ok=True)
        for rank, res in enumerate(selected_results, start=1):
            coords = res['coords']
            labels = res.get('labels') or folder.static_labels
            pred_ca = np.array([coords[i] for i, lbl in enumerate(labels) if lbl[1] == 'CA'])
            sidechain_centroids = folder.compute_sidechain_centroids(coords, labels)
            nonlocal_clash_metrics = nonlocal_heavy_clash_metrics(coords, labels)
            local_clash_metrics = adjacent_heavy_clash_metrics(coords, labels)
            ring_penetration_metrics = qtf_gromacs.ring_penetration_metrics(coords, labels)
            ca_pdb_path = os.path.join(job_output_dir, "raw_pdbs", f"model_{rank}_ca.pdb")
            ca_centroid_pdb_path = os.path.join(job_output_dir, "raw_pdbs", f"model_{rank}_ca_centroid.pdb")
            full_pdb_path = os.path.join(job_output_dir, "raw_pdbs", f"model_{rank}_full.pdb")
            folder.save_reduced_pdb(pred_ca, filename=ca_pdb_path, sidechain_centroids=None, energy=res['energy'])
            folder.save_reduced_pdb(pred_ca, filename=ca_centroid_pdb_path, sidechain_centroids=sidechain_centroids, energy=res['energy'])
            folder.save_pdb(
                coords,
                labels,
                filename=full_pdb_path,
                energy=res['energy'],
                remarks=["QTF heavy-atom rebuilt structure from optimized ensemble model"],
                include_hydrogens=False,
            )

            gromacs_result = utils.gromacs_postprocess_structure(
                enabled=bool(args.gromacs_minimize),
                full_pdb_path=full_pdb_path,
                gromacs_dir=os.path.join(job_output_dir, "gromacs_pdbs", f"model_{rank}"),
                forcefield=args.gromacs_forcefield,
                water=args.gromacs_water,
                nsteps=args.gromacs_nsteps,
                emtol=args.gromacs_emtol,
                maxwarn=args.gromacs_maxwarn,
                coords=coords,
                labels=labels,
                ca_coords=pred_ca,
                sidechain_centroid_fn=folder.compute_sidechain_centroids,
                nonlocal_clash_fn=nonlocal_heavy_clash_metrics,
                local_clash_fn=adjacent_heavy_clash_metrics,
            )
            coords = gromacs_result["coords"]
            labels = gromacs_result["labels"]
            pred_ca = gromacs_result["ca_coords"]
            sidechain_centroids = gromacs_result["sidechain_centroids"]
            nonlocal_clash_metrics = gromacs_result["nonlocal_clash_metrics"]
            local_clash_metrics = gromacs_result["local_clash_metrics"]
            ring_penetration_metrics = gromacs_result["ring_penetration_metrics"]
            gromacs_info = gromacs_result["gromacs_info"]
            p_e2e, p_rg = utils.calculate_physics_metrics(pred_ca)
            if true_rmsd_coords is not None:
                n = min(len(pred_ca), len(true_ca))
                pred_ca_n = pred_ca[:n]
                true_ca_n = true_ca[:n]
                t_e2e, t_rg = utils.calculate_physics_metrics(true_ca_n)
                rmsd, rmsd_meta = utils.rmsd_between_structures(
                    coords,
                    labels,
                    true_rmsd_coords,
                    true_rmsd_labels,
                    args.rmsd_mode,
                    args.rmsd_residue_scope,
                )
            else:
                t_e2e, t_rg, rmsd = np.nan, np.nan, np.nan
                rmsd_meta = utils.rmsd_selection_metadata(
                    args.rmsd_mode,
                    args.rmsd_residue_scope,
                    n_atoms=0,
                    n_residues=0,
                )

            # flatten energy term decomposition (if available)
            energy_terms = res.get("energy_terms") or {}
            flat_terms = {f"term_{k}": float(v) for k, v in energy_terms.items()}

            model_rows.append({
                "ensemble_id": int(res["id"]),
                "init_type": str(res["type"]),
                "energy_rank": int(rank),
                "energy": float(res["energy"]),
                "rmsd_to_reference_A": rmsd,
                "rmsd_mode": args.rmsd_mode,
                "rmsd_residue_scope": args.rmsd_residue_scope,
                **rmsd_meta,
                "pred_e2e_A": float(p_e2e),
                "pred_rg_A": float(p_rg),
                "ref_e2e_A": float(t_e2e) if not np.isnan(t_e2e) else np.nan,
                "ref_rg_A": float(t_rg) if not np.isnan(t_rg) else np.nan,
                "rebuilt_ca_pdb_path": ca_pdb_path,
                "rebuilt_ca_centroid_pdb_path": ca_centroid_pdb_path,
                "rebuilt_full_pdb_path": full_pdb_path,
                **gromacs_info,
                **nonlocal_clash_metrics,
                **local_clash_metrics,
                **ring_penetration_metrics,
                **flat_terms,
            })

        df_models = pd.DataFrame(model_rows).sort_values(["energy"]).reset_index(drop=True)
        if "clash_flag" in df_models.columns:
            clean_df = df_models.loc[~df_models["clash_flag"]].copy()
            if len(clean_df) > 0:
                df_models = clean_df
            else:
                print("[predictor][warn] all ranked candidates triggered the nonlocal clash filter; keeping unfiltered set")
        if "local_clash_flag" in df_models.columns:
            clean_df = df_models.loc[~df_models["local_clash_flag"]].copy()
            if len(clean_df) > 0:
                df_models = clean_df
            else:
                print("[predictor][warn] all ranked candidates triggered the local backbone clash filter; keeping unfiltered set")
        if "ring_penetration_flag" in df_models.columns:
            clean_df = df_models.loc[~df_models["ring_penetration_flag"]].copy()
            if len(clean_df) > 0:
                df_models = clean_df
            else:
                print("[predictor][warn] all ranked candidates triggered the ring penetration filter; keeping unfiltered set")

        if bool(args.gromacs_minimize) and bool(args.gromacs_rerank) and "gromacs_potential_kj_mol" in df_models.columns:
            df_models = df_models.sort_values(["gromacs_potential_kj_mol", "energy"], na_position="last").reset_index(drop=True)
            df_models["gromacs_energy_rank"] = np.arange(1, len(df_models) + 1)
        else:
            df_models = df_models.sort_values(["energy"]).reset_index(drop=True)

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
            "Reference Structure": ((reference_pdb_path or reference_structure_pdb_id) if (reference_pdb_path or reference_structure_pdb_id) is not None and args.mode != "predict_only" else None),
            "Force Field": force_field,
            "Prime Strategy": prime_strategy,
            "rmsd_mode": args.rmsd_mode,
            "rmsd_residue_scope": args.rmsd_residue_scope,
            "energy_backend": args.energy_backend,
            "use_e2e_constraint": bool(args.use_e2e_constraint),
            "e2e_scale": float(args.e2e_scale),
            "rosetta_repack": bool(args.rosetta_repack),
            "rosetta_fa_min": bool(args.rosetta_fa_min),
            "rosetta_cen_min": bool(args.rosetta_cen_min),
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


def main():
    __main__()


if __name__ == "__main__":
    main()
