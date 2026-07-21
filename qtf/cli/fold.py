#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os, time, json, argparse
import logging
from datetime import datetime

from qtf.core.ensemble import EnsembleFoldingManager
from qtf.core.folder import QuantumBiophysicsFolder
from qtf.utils import workflow as utils
from qtf.utils import gromacs as qtf_gromacs
from qtf.utils.workflow import (
    adjacent_heavy_clash_metrics,
    nonlocal_heavy_clash_metrics,
)


def _jsonify(x):
    try:
        import numpy as _np
        if isinstance(x, (_np.floating, _np.integer)):
            return x.item()
        if isinstance(x, _np.ndarray):
            return x.tolist()
    except Exception:
        pass
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    return x


def _format_rank_value(value):
    try:
        if pd.isna(value):
            return "NA"
        return str(int(value))
    except Exception:
        return "NA"


def _format_score_value(value):
    try:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
    except Exception:
        pass
    return str(value)


def _first_existing_pdb_path(row, columns):
    for column in columns:
        path = row.get(column, "")
        if isinstance(path, str) and path and os.path.isfile(path):
            return path
    return None


def _single_parameter_vector_from_file(path, expected_size=None):
    path = os.path.abspath(path)
    if path.endswith(".npy"):
        params = np.load(path)
    elif path.endswith(".npz"):
        payload = np.load(path)
        key = "params" if "params" in payload else ("circuit_parameters" if "circuit_parameters" in payload else None)
        if key is None:
            keys = list(payload.keys())
            if not keys:
                raise ValueError(f"no arrays found in parameter file: {path}")
            key = keys[0]
        params = payload[key]
    else:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            for key in ("params", "circuit_parameters", "parameters"):
                if key in payload:
                    params = payload[key]
                    break
            else:
                raise ValueError(f"no parameter vector found in JSON file: {path}")
        else:
            params = payload
    params = np.asarray(params, dtype=float).reshape(-1)
    if expected_size is not None and params.size != int(expected_size):
        raise ValueError(
            f"parameter vector in {path} has length {params.size}, expected {int(expected_size)}"
        )
    return params


def _find_parameter_manifest(path):
    path = os.path.abspath(path)
    if os.path.isdir(path):
        manifest = os.path.join(path, "circuit_parameters.json")
        nested_manifest = os.path.join(path, "circuit_parameters", "circuit_parameters.json")
        if os.path.isfile(manifest):
            return manifest
        elif os.path.isfile(nested_manifest):
            return nested_manifest
    return path


def _manifest_job_dir(manifest_path):
    params_dir = os.path.dirname(os.path.abspath(manifest_path))
    if os.path.basename(params_dir) == "circuit_parameters":
        return os.path.dirname(params_dir)
    return params_dir


def _selected_parameter_file_from_previous_run(path, select):
    path = os.path.abspath(path)
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, "ensemble_ranked.csv")):
        job_dir = path
    else:
        manifest_path = _find_parameter_manifest(path)
        job_dir = _manifest_job_dir(manifest_path)
    ensemble_csv_path = os.path.join(job_dir, "ensemble_ranked.csv")
    if not os.path.isfile(ensemble_csv_path):
        raise FileNotFoundError(
            f"--initial_params_select {select!r} requires ensemble_ranked.csv next to the saved parameters: "
            f"{ensemble_csv_path}"
        )

    df = pd.read_csv(ensemble_csv_path)
    if df.empty:
        raise ValueError(f"ensemble ranking is empty: {ensemble_csv_path}")
    if select == "best_energy":
        if "energy" not in df.columns:
            raise ValueError(f"ensemble ranking lacks required 'energy' column: {ensemble_csv_path}")
        idx = pd.to_numeric(df["energy"], errors="coerce").idxmin()
    elif select == "best_rmsd":
        if "rmsd_to_reference_A" not in df.columns:
            raise ValueError(f"ensemble ranking lacks required 'rmsd_to_reference_A' column: {ensemble_csv_path}")
        rmsd = pd.to_numeric(df["rmsd_to_reference_A"], errors="coerce")
        if not rmsd.notna().any():
            raise ValueError(f"ensemble ranking has no finite RMSD values: {ensemble_csv_path}")
        idx = rmsd.idxmin()
    else:
        raise ValueError(f"unsupported initial parameter selection mode: {select}")

    row = df.loc[idx]
    for column in ("circuit_params_npz_path", "circuit_params_json_path"):
        candidate = row.get(column, "")
        if isinstance(candidate, str) and candidate:
            if not os.path.isabs(candidate):
                candidate = os.path.join(job_dir, candidate)
            if os.path.isfile(candidate):
                return candidate

    if "ensemble_id" in row:
        replica_number = int(row["ensemble_id"]) + 1
        candidate = os.path.join(job_dir, "circuit_parameters", f"replica_{replica_number}_params.npz")
        if os.path.isfile(candidate):
            return candidate
        candidate = os.path.join(job_dir, "circuit_parameters", f"replica_{replica_number}_params.json")
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(f"could not resolve saved circuit parameters from selected row in {ensemble_csv_path}")


def _load_initial_parameter_vectors(path, *, expected_size, n_runs, select="first"):
    path = os.path.abspath(path)
    if select != "first":
        selected_path = _selected_parameter_file_from_previous_run(path, select)
        vector = _single_parameter_vector_from_file(selected_path, expected_size)
        return _expand_warm_start_vectors([vector], n_runs)

    if os.path.isdir(path):
        manifest = os.path.join(path, "circuit_parameters.json")
        nested_manifest = os.path.join(path, "circuit_parameters", "circuit_parameters.json")
        if os.path.isfile(manifest):
            path = manifest
        elif os.path.isfile(nested_manifest):
            path = nested_manifest
        else:
            files = sorted(
                os.path.join(path, name)
                for name in os.listdir(path)
                if name.endswith("_params.json") or name.endswith("_params.npz")
            )
            if not files:
                raise FileNotFoundError(f"no parameter files found in directory: {path}")
            vectors = [_single_parameter_vector_from_file(file_path, expected_size) for file_path in files]
            return _expand_warm_start_vectors(vectors, n_runs)

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and "replicas" in payload:
            vectors = []
            base_dir = os.path.dirname(path)
            for replica in payload.get("replicas") or []:
                if not isinstance(replica, dict):
                    continue
                for key in ("params", "circuit_parameters", "parameters"):
                    if key in replica:
                        vectors.append(np.asarray(replica[key], dtype=float).reshape(-1))
                        break
                else:
                    file_path = replica.get("npz_path") or replica.get("json_path")
                    if isinstance(file_path, str) and file_path:
                        if not os.path.isabs(file_path):
                            file_path = os.path.join(base_dir, file_path)
                        vectors.append(_single_parameter_vector_from_file(file_path, expected_size))
            if not vectors:
                raise ValueError(f"no replica parameter vectors found in manifest: {path}")
            for vector in vectors:
                if vector.size != int(expected_size):
                    raise ValueError(
                        f"parameter vector in {path} has length {vector.size}, expected {int(expected_size)}"
                    )
            return _expand_warm_start_vectors(vectors, n_runs)

    vector = _single_parameter_vector_from_file(path, expected_size)
    return _expand_warm_start_vectors([vector], n_runs)


def _expand_warm_start_vectors(vectors, n_runs):
    if len(vectors) == 1:
        return [vectors[0].copy() for _ in range(n_runs)]
    if len(vectors) < n_runs:
        raise ValueError(
            f"only {len(vectors)} warm-start vector(s) were provided for {n_runs} replica(s)"
        )
    return [np.asarray(vectors[i], dtype=float).reshape(-1).copy() for i in range(n_runs)]


def _write_circuit_parameter_artifacts(job_output_dir, *, sequence, args, folder, results):
    params_dir = os.path.join(job_output_dir, "circuit_parameters")
    os.makedirs(params_dir, exist_ok=True)
    manifest = {
        "format": "qtf.circuit_parameters.v1",
        "sequence": sequence,
        "energy_backend": args.energy_backend,
        "omega_mode": args.omega_mode,
        "chi_mode": folder.chi_mode,
        "n_qubits": int(folder.n_qubits),
        "n_params": int(folder.n_params),
        "ansatz": type(folder.ansatz).__name__,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "replicas": [],
    }
    for res in sorted(results, key=lambda item: int(item["id"])):
        replica_number = int(res["id"]) + 1
        params = np.asarray(res["params"], dtype=float).reshape(-1)
        if params.size != int(folder.n_params):
            raise ValueError(
                f"replica {replica_number} parameter vector has length {params.size}, expected {folder.n_params}"
            )
        stem = f"replica_{replica_number}_params"
        json_path = os.path.join(params_dir, f"{stem}.json")
        npz_path = os.path.join(params_dir, f"{stem}.npz")
        payload = {
            "format": "qtf.circuit_parameters.v1",
            "sequence": sequence,
            "energy_backend": args.energy_backend,
            "omega_mode": args.omega_mode,
            "chi_mode": folder.chi_mode,
            "ensemble_id": int(res["id"]),
            "replica": replica_number,
            "seed": int(res["seed"]),
            "energy": float(res["energy"]),
            "n_qubits": int(folder.n_qubits),
            "n_params": int(folder.n_params),
            "ansatz": type(folder.ansatz).__name__,
            "params": params.tolist(),
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        np.savez_compressed(
            npz_path,
            params=params,
            ensemble_id=int(res["id"]),
            replica=replica_number,
            seed=int(res["seed"]),
            energy=float(res["energy"]),
        )
        manifest["replicas"].append(
            {
                "ensemble_id": int(res["id"]),
                "replica": replica_number,
                "seed": int(res["seed"]),
                "energy": float(res["energy"]),
                "n_params": int(params.size),
                "json_path": os.path.relpath(json_path, params_dir),
                "npz_path": os.path.relpath(npz_path, params_dir),
            }
        )
    manifest_path = os.path.join(params_dir, "circuit_parameters.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def _write_ranked_multimodel_pdb(
    df,
    output_path,
    *,
    path_columns,
    energy_column,
    rmsd_column,
    remark_factory,
):
    if df.empty:
        return
    energy_values = df[energy_column] if energy_column in df.columns else pd.Series(np.nan, index=df.index)
    rmsd_values = df[rmsd_column] if rmsd_column in df.columns else pd.Series(np.nan, index=df.index)
    energy_numeric = pd.to_numeric(energy_values, errors="coerce")
    rmsd_numeric = pd.to_numeric(rmsd_values, errors="coerce")
    energy_ranks = energy_numeric.rank(
        method="min",
        ascending=True,
    ).where(energy_numeric.notna(), np.nan)
    rmsd_ranks = rmsd_numeric.rank(
        method="min",
        ascending=True,
    ).where(rmsd_numeric.notna(), np.nan)
    model_index = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for row_index, row in df.iterrows():
            pdb_path = _first_existing_pdb_path(row, path_columns)
            if not pdb_path:
                continue
            model_index += 1
            out.write(f"MODEL     {model_index:4d}\n")
            remarks = remark_factory(
                row,
                model_index,
                _format_rank_value(energy_ranks.loc[row_index]),
                _format_rank_value(rmsd_ranks.loc[row_index]),
                pdb_path,
            )
            for remark in remarks:
                out.write(f"REMARK {remark}\n")
            with open(pdb_path, "r", encoding="utf-8") as inp:
                for line in inp:
                    record = line[:6].strip().upper()
                    if record in {"MODEL", "ENDMDL", "END"}:
                        continue
                    out.write(line if line.endswith("\n") else f"{line}\n")
            out.write("ENDMDL\n")
    if model_index == 0 and os.path.exists(output_path):
        os.remove(output_path)


def main(argv=None):
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        root_logger.setLevel(logging.INFO)

    time_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(
        prog="qtf-fold",
        description="Run quantum folding prediction for a target sequence."
    )

    parser.add_argument('--predict', default=None, help='target sequence to predict')
    parser.add_argument('--reference_structure', default=None, help='reference structure PDB ID for comparison')
    parser.add_argument('--reference_pdb', default=None, help='optional local reference PDB path for comparison')
    parser.add_argument('--rmsd_mode', default='ca', choices=['ca', 'heavy'],
                        help='RMSD atom selection: all CA atoms or all heavy atoms')
    parser.add_argument('--rmsd_residue_scope', default='core', choices=['core', 'all'],
                        help='Residue range used for RMSD; core excludes the first and last residues')
    parser.add_argument(
        '--average_reference_backbone',
        action='store_true',
        help=('How to select backbone from reference structure, either first model or average for NMR ensembles. '
              'Defaults to first model, which automatically works with Xray structures.')
    )
    parser.add_argument('--mode', default="predict_and_compare", choices=["predict_and_compare", "predict_only"],
                        help='which mode to run script in')
    parser.add_argument('--ensemble_size', default=3, type=int, help='ensemble size')
    parser.add_argument('--top_k', default=1, type=int,
                        help='how many lowest-energy ensemble models to compare/save (default 1)')
    parser.add_argument('--top_frac', default=None, type=float,
                        help='fraction (0-1] of lowest-energy models to compare/save; overrides --top_k if set')
    parser.add_argument('--maxiter', default=2000, type=int, help='max iterations for each trajectory')
    parser.add_argument('--energy_backend', default='custom', choices=['custom', 'rosetta', 'openmm'],
                        help='energy backend used for all optimization stages')
    parser.add_argument('--use_e2e_constraint', default=1, type=int,
                        help='1 to use length-scaled E2E constraint in custom scorer, 0 to disable')
    parser.add_argument('--e2e_scale', default=1.0, type=float,
                        help='multiplier for E2E constraint if enabled')
    parser.add_argument('--rosetta_repack', default=0, type=int)
    parser.add_argument('--rosetta_fa_min', default=0, type=int)
    parser.add_argument('--rosetta_cen_min', default=0, type=int)
    parser.add_argument('--omega_mode', default='window', choices=['free', 'fixed', 'window'],
                        help='omega handling: window=default map DOF into 170-190 degree trans band, free=raw full-range DOF with penalty, fixed=trans/no DOF')
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
    parser.add_argument('--output_root', default=os.path.join("run_outputs", "quantum_simulations"),
                        help='root directory for predictor outputs')
    parser.add_argument('--top_k_snapshots', default=0, type=int,
                        help='save the K lowest-energy intermediate structures encountered during optimisation (0 = disabled)')
    parser.add_argument('--snapshot_energy_gap', default=0.25, type=float,
                        help='minimum raw QTF energy spacing between saved best snapshots; 0 disables energy-spacing filtering')
    parser.add_argument(
        '--snapshot_sort_by',
        default='energy',
        choices=['energy', 'rmsd'],
        help='sort exported snapshot pool by energy or by RMSD to the reference structure',
    )
    parser.add_argument(
        '--initial_params',
        default=None,
        help=(
            'warm-start circuit parameters from a saved QTF parameter JSON/NPZ/NPY file, '
            'a circuit_parameters.json manifest, or a circuit_parameters directory'
        ),
    )
    parser.add_argument(
        '--initial_params_select',
        default='first',
        choices=['first', 'best_energy', 'best_rmsd'],
        help=(
            'selection mode when --initial_params points at a previous multi-replica run: '
            'first=replica order, best_energy=lowest raw QTF/backend energy in ensemble_ranked.csv, '
            'best_rmsd=lowest rmsd_to_reference_A in ensemble_ranked.csv'
        ),
    )

    args = parser.parse_args(argv)
    if args.gromacs_minimize is None:
        args.gromacs_minimize = 1
    if args.gromacs_rerank is None:
        args.gromacs_rerank = 1

    sequence = args.predict
    if not sequence:
        raise ValueError("--predict <SEQUENCE> is required")

    reference_structure_pdb_id = args.reference_structure
    reference_pdb_path = args.reference_pdb
    average_reference_backbone_mode = args.average_reference_backbone
    ensemble_size = args.ensemble_size
    top_k = args.top_k
    top_frac = args.top_frac
    snapshot_sort_by = args.snapshot_sort_by

    outputs_root = args.output_root
    os.makedirs(outputs_root, exist_ok=True)
    job_output_dir = os.path.join(outputs_root, f"{sequence}_{args.energy_backend}_{timestamp}")
    os.makedirs(job_output_dir, exist_ok=True)
    print(f"Writing outputs to: {job_output_dir}")

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

    print(f"--- FOLDING SEQUENCE: {sequence} ---")
    folder = utils.make_folder(
        sequence=sequence,
        energy_backend=args.energy_backend,
        use_e2e_constraint=bool(args.use_e2e_constraint),
        e2e_scale=args.e2e_scale,
        rosetta_repack=bool(args.rosetta_repack),
        rosetta_fa_min=bool(args.rosetta_fa_min),
        rosetta_cen_min=bool(args.rosetta_cen_min),
        omega_mode=args.omega_mode,
    )

    initial_params_list = None
    if args.initial_params:
        initial_params_list = _load_initial_parameter_vectors(
            args.initial_params,
            expected_size=folder.n_params,
            n_runs=ensemble_size,
            select=args.initial_params_select,
        )
        print(
            "Loaded warm-start circuit parameters "
            f"for {len(initial_params_list)} replica(s): {args.initial_params} "
            f"(select={args.initial_params_select})"
        )

    manager = EnsembleFoldingManager(folder)
    run_kwargs = {
        "n_runs": ensemble_size,
        "max_iter": args.maxiter,
        "top_k_snapshots": args.top_k_snapshots,
        "snapshot_energy_gap": args.snapshot_energy_gap,
    }
    if initial_params_list is not None:
        run_kwargs["initial_params_list"] = initial_params_list
    manager.run_ensemble(**run_kwargs)

    ranked_results = manager.get_results(ranked=True)
    parameter_manifest_path = _write_circuit_parameter_artifacts(
        job_output_dir,
        sequence=sequence,
        args=args,
        folder=folder,
        results=manager.get_results(ranked=False),
    )
    with open(parameter_manifest_path, "r", encoding="utf-8") as handle:
        parameter_manifest = json.load(handle)
    parameter_artifacts_by_id = {
        int(entry["ensemble_id"]): {
            "json_path": os.path.join(os.path.dirname(parameter_manifest_path), entry["json_path"]),
            "npz_path": os.path.join(os.path.dirname(parameter_manifest_path), entry["npz_path"]),
        }
        for entry in parameter_manifest["replicas"]
    }
    selected_results = manager.select_top(top_k=top_k, top_frac=top_frac)

    if not ranked_results:
        raise RuntimeError("Ensemble produced no results.")
    if not selected_results:
        raise RuntimeError("Selection produced no results (check --top_k/--top_frac).")

    model_rows = []
    snapshot_rows = []
    os.makedirs(os.path.join(job_output_dir, "raw_models"), exist_ok=True)
    if bool(args.gromacs_minimize):
        os.makedirs(os.path.join(job_output_dir, "gromacs_minimized_models"), exist_ok=True)
    for rank, res in enumerate(selected_results, start=1):
        replica_stem = f"replica_{rank}"
        coords = res['coords']
        labels = res.get('labels') or folder.static_labels
        pred_ca = np.array([coords[i] for i, lbl in enumerate(labels) if lbl[1] == 'CA'])
        sidechain_centroids = folder.compute_sidechain_centroids(coords, labels)
        nonlocal_clash_metrics = nonlocal_heavy_clash_metrics(coords, labels)
        local_clash_metrics = adjacent_heavy_clash_metrics(coords, labels)
        ring_penetration_metrics = qtf_gromacs.ring_penetration_metrics(coords, labels)
        ca_pdb_path = os.path.join(job_output_dir, "raw_models", f"{replica_stem}_ca.pdb")
        ca_centroid_pdb_path = os.path.join(job_output_dir, "raw_models", f"{replica_stem}_ca_centroid.pdb")
        full_pdb_path = os.path.join(job_output_dir, "raw_models", f"{replica_stem}_full.pdb")
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

        if true_rmsd_coords is not None:
            coords, raw_rmsd, raw_rmsd_meta, raw_alignment = utils.align_structure_to_reference(
                coords,
                labels,
                true_rmsd_coords,
                true_rmsd_labels,
                args.rmsd_mode,
                args.rmsd_residue_scope,
            )
        else:
            raw_rmsd = np.nan
            raw_rmsd_meta = utils.rmsd_selection_metadata(
                args.rmsd_mode,
                args.rmsd_residue_scope,
                n_atoms=0,
                n_residues=0,
            )
        if true_rmsd_coords is not None:
            pred_ca = utils.apply_alignment_transform(pred_ca, raw_alignment)
            sidechain_centroids = utils.apply_alignment_transform(sidechain_centroids, raw_alignment)
            folder.save_reduced_pdb(pred_ca, filename=ca_pdb_path, sidechain_centroids=None, energy=res['energy'])
            folder.save_reduced_pdb(
                pred_ca,
                filename=ca_centroid_pdb_path,
                sidechain_centroids=sidechain_centroids,
                energy=res['energy'],
            )
            folder.save_pdb(
                coords,
                labels,
                filename=full_pdb_path,
                energy=res['energy'],
                remarks=[
                    "QTF heavy-atom rebuilt structure from optimized ensemble model",
                    "Coordinates rigidly aligned to reference for RMSD/visualization",
                ],
                include_hydrogens=False,
            )

        # Save best-K intermediate snapshots if available
        snapshots = res.get("best_snapshots", [])
        if snapshots:
            snap_dir = os.path.join(job_output_dir, "raw_models", "snapshots", replica_stem)
            os.makedirs(snap_dir, exist_ok=True)
            for si, snap in enumerate(snapshots, start=1):
                snap_coords = snap["coords"]
                snap_labels = snap["labels"]
                snap_pdb = os.path.join(snap_dir, f"snapshot_{si}.pdb")
                folder.save_pdb(
                    snap_coords,
                    snap_labels,
                    filename=snap_pdb,
                    energy=snap["energy"],
                    remarks=[f"Best snapshot #{si} during optimisation (E={snap['energy']:.4f})"],
                    include_hydrogens=False,
                )
                if true_rmsd_coords is not None:
                    snap_coords, snap_rmsd, snap_rmsd_meta, _ = utils.align_structure_to_reference(
                        snap_coords,
                        snap_labels,
                        true_rmsd_coords,
                        true_rmsd_labels,
                        args.rmsd_mode,
                        args.rmsd_residue_scope,
                    )
                else:
                    snap_rmsd = np.nan
                    snap_rmsd_meta = utils.rmsd_selection_metadata(
                        args.rmsd_mode,
                        args.rmsd_residue_scope,
                        n_atoms=0,
                        n_residues=0,
                    )
                if true_rmsd_coords is not None:
                    folder.save_pdb(
                        snap_coords,
                        snap_labels,
                        filename=snap_pdb,
                        energy=snap["energy"],
                        remarks=[
                            f"Best snapshot #{si} during optimisation (E={snap['energy']:.4f})",
                            "Coordinates rigidly aligned to reference for RMSD/visualization",
                        ],
                        include_hydrogens=False,
                    )
                snap_gromacs_status = "not_run"
                snap_gromacs_potential_kj_mol = np.nan
                snap_gromacs_potential_kcal_mol = np.nan
                snap_gromacs_rmsd = np.nan
                snap_gromacs_final_max_force = np.nan
                snap_gromacs_converged_fmax_lt_100 = False
                snap_gromacs_minimized_full_pdb_path = ""
                snap_raw_pdb_retained = True
                run_snapshot_gromacs = bool(args.gromacs_minimize)
                if run_snapshot_gromacs:
                    snap_gromacs_dir = os.path.join(
                        job_output_dir,
                        "gromacs_minimized_models",
                        "snapshots",
                        replica_stem,
                        f"snapshot_{si}",
                    )
                    snap_gromacs_result = qtf_gromacs.minimize_pdb_with_gromacs(
                        snap_pdb,
                        snap_gromacs_dir,
                        forcefield=args.gromacs_forcefield,
                        water=args.gromacs_water,
                        nsteps=args.gromacs_nsteps,
                        emtol=args.gromacs_emtol,
                        maxwarn=args.gromacs_maxwarn,
                    )
                    snap_gromacs_status = str(snap_gromacs_result.get("gromacs_status", "unknown"))
                    snap_gromacs_potential_kj_mol = float(snap_gromacs_result.get("gromacs_potential_kj_mol", np.nan))
                    snap_gromacs_potential_kcal_mol = float(snap_gromacs_result.get("gromacs_potential_kcal_mol", np.nan))
                    snap_gromacs_final_max_force = float(snap_gromacs_result.get("gromacs_final_max_force", np.nan))
                    snap_gromacs_converged_fmax_lt_100 = bool(
                        snap_gromacs_result.get("gromacs_converged_fmax_lt_100", False)
                    )
                    snap_gromacs_minimized_full_pdb_path = str(
                        snap_gromacs_result.get("gromacs_minimized_full_pdb_path") or ""
                    )
                    if snap_gromacs_minimized_full_pdb_path and os.path.isfile(snap_gromacs_minimized_full_pdb_path):
                        gromacs_coords, gromacs_labels = qtf_gromacs.parse_pdb_atoms(snap_gromacs_minimized_full_pdb_path)
                        if true_rmsd_coords is not None:
                            gromacs_coords, snap_gromacs_rmsd, _snap_gromacs_meta, _ = utils.align_structure_to_reference(
                                gromacs_coords,
                                gromacs_labels,
                                true_rmsd_coords,
                                true_rmsd_labels,
                                args.rmsd_mode,
                                args.rmsd_residue_scope,
                            )
                            folder.save_pdb(
                                gromacs_coords,
                                gromacs_labels,
                                filename=snap_gromacs_minimized_full_pdb_path,
                                energy=snap_gromacs_potential_kcal_mol,
                                remarks=[
                                    f"GROMACS-minimized snapshot #{si}",
                                    "Coordinates rigidly aligned to reference for RMSD/visualization",
                                ],
                                include_hydrogens=True,
                            )
                    if snap_gromacs_status == "ok" and os.path.exists(snap_pdb):
                        os.remove(snap_pdb)
                snap_raw_pdb_retained = os.path.exists(snap_pdb)
                snap_effective_rmsd = snap_gromacs_rmsd if not np.isnan(snap_gromacs_rmsd) else snap_rmsd
                snapshot_rows.append({
                    "ensemble_id": int(res["id"]),
                    "ensemble_rank": int(rank),
                    "snapshot_rank_within_replica": int(si),
                    "snapshot_energy": float(snap["energy"]),
                    "snapshot_rmsd_to_reference_A": float(snap_rmsd) if not np.isnan(snap_rmsd) else np.nan,
                    "snapshot_effective_rmsd_to_reference_A": (
                        float(snap_effective_rmsd) if not np.isnan(snap_effective_rmsd) else np.nan
                    ),
                    "snapshot_gromacs_enabled": bool(args.gromacs_minimize),
                    "snapshot_gromacs_status": snap_gromacs_status,
                    "snapshot_gromacs_potential_kj_mol": snap_gromacs_potential_kj_mol,
                    "snapshot_gromacs_potential_kcal_mol": snap_gromacs_potential_kcal_mol,
                    "snapshot_gromacs_rmsd_to_reference_A": float(snap_gromacs_rmsd) if not np.isnan(snap_gromacs_rmsd) else np.nan,
                    "snapshot_gromacs_final_max_force": snap_gromacs_final_max_force,
                    "snapshot_gromacs_converged_fmax_lt_100": snap_gromacs_converged_fmax_lt_100,
                    "snapshot_gromacs_minimized_full_pdb_path": snap_gromacs_minimized_full_pdb_path,
                    "rmsd_mode": args.rmsd_mode,
                    "rmsd_residue_scope": args.rmsd_residue_scope,
                    **snap_rmsd_meta,
                    "snapshot_pdb_path": snap_pdb if snap_raw_pdb_retained else "",
                    "snapshot_raw_pdb_retained": bool(snap_raw_pdb_retained),
                    "replica_final_energy": float(res["energy"]),
                    "replica_final_rmsd_to_reference_A": float(raw_rmsd) if not np.isnan(raw_rmsd) else np.nan,
                })

        gromacs_result = utils.gromacs_postprocess_structure(
            enabled=bool(args.gromacs_minimize),
            full_pdb_path=full_pdb_path,
            gromacs_dir=os.path.join(job_output_dir, "gromacs_minimized_models", replica_stem),
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
        final_rmsd = np.nan
        final_rmsd_meta = None
        if true_rmsd_coords is not None:
            coords, final_rmsd, final_rmsd_meta, _ = utils.align_structure_to_reference(
                coords,
                labels,
                true_rmsd_coords,
                true_rmsd_labels,
                args.rmsd_mode,
                args.rmsd_residue_scope,
            )
            pred_ca = qtf_gromacs.ca_coords(coords, labels)
            sidechain_centroids = folder.compute_sidechain_centroids(coords, labels)
            nonlocal_clash_metrics = nonlocal_heavy_clash_metrics(coords, labels)
            local_clash_metrics = adjacent_heavy_clash_metrics(coords, labels)
            ring_penetration_metrics = qtf_gromacs.ring_penetration_metrics(coords, labels)
            final_aligned_path = str(gromacs_info.get("gromacs_minimized_full_pdb_path") or full_pdb_path)
            if final_aligned_path and os.path.isfile(final_aligned_path):
                folder.save_pdb(
                    coords,
                    labels,
                    filename=final_aligned_path,
                    energy=res["energy"],
                    remarks=[
                        "QTF final ensemble structure",
                        "Coordinates rigidly aligned to reference for RMSD/visualization",
                    ],
                    include_hydrogens=bool(
                        gromacs_info.get("gromacs_status") == "ok"
                        and gromacs_info.get("gromacs_minimized_full_pdb_path")
                    ),
                )
        p_metrics = utils.calculate_physics_metrics(pred_ca)
        p_e2e = p_metrics["end_to_end"]
        p_rg = p_metrics["radius_of_gyration"]
        if true_rmsd_coords is not None:
            n = min(len(pred_ca), len(true_ca))
            pred_ca_n = pred_ca[:n]
            true_ca_n = true_ca[:n]
            t_metrics = utils.calculate_physics_metrics(true_ca_n)
            t_e2e = t_metrics["end_to_end"]
            t_rg = t_metrics["radius_of_gyration"]
            rmsd = final_rmsd
            rmsd_meta = final_rmsd_meta
            gromacs_rmsd = (
                rmsd
                if bool(args.gromacs_minimize) and gromacs_info.get("gromacs_status") == "ok"
                else np.nan
            )
        else:
            t_e2e, t_rg, rmsd = np.nan, np.nan, np.nan
            gromacs_rmsd = np.nan
            rmsd_meta = utils.rmsd_selection_metadata(
                args.rmsd_mode,
                args.rmsd_residue_scope,
                n_atoms=0,
                n_residues=0,
            )

        energy_terms = res.get("energy_terms") or {}
        flat_terms = {f"term_{k}": float(v) for k, v in energy_terms.items()}

        model_rows.append({
            "ensemble_id": int(res["id"]),
            "init_type": str(res["type"]),
            "energy_rank": int(rank),
            "energy": float(res["energy"]),
            "circuit_params_json_path": parameter_artifacts_by_id.get(int(res["id"]), {}).get("json_path", ""),
            "circuit_params_npz_path": parameter_artifacts_by_id.get(int(res["id"]), {}).get("npz_path", ""),
            "rmsd_to_reference_A": rmsd,
            "raw_rmsd_to_reference_A": raw_rmsd,
            "gromacs_rmsd_to_reference_A": gromacs_rmsd,
            "rmsd_mode": args.rmsd_mode,
            "rmsd_residue_scope": args.rmsd_residue_scope,
            **rmsd_meta,
            **{f"raw_{k}": v for k, v in raw_rmsd_meta.items()},
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
            print("[fold][warn] all ranked candidates triggered the nonlocal clash filter; keeping unfiltered set")
    if "local_clash_flag" in df_models.columns:
        clean_df = df_models.loc[~df_models["local_clash_flag"]].copy()
        if len(clean_df) > 0:
            df_models = clean_df
        else:
            print("[fold][warn] all ranked candidates triggered the local backbone clash filter; keeping unfiltered set")
    if "ring_penetration_flag" in df_models.columns:
        clean_df = df_models.loc[~df_models["ring_penetration_flag"]].copy()
        if len(clean_df) > 0:
            df_models = clean_df
        else:
            print("[fold][warn] all ranked candidates triggered the ring penetration filter; keeping unfiltered set")

    if bool(args.gromacs_minimize) and bool(args.gromacs_rerank) and "gromacs_potential_kj_mol" in df_models.columns:
        df_models = df_models.sort_values(["gromacs_potential_kj_mol", "energy"], na_position="last").reset_index(drop=True)
        df_models["gromacs_energy_rank"] = np.arange(1, len(df_models) + 1)
    else:
        df_models = df_models.sort_values(["energy"]).reset_index(drop=True)

    ensemble_csv_path = os.path.join(job_output_dir, "ensemble_ranked.csv")
    ensemble_json_path = os.path.join(job_output_dir, "ensemble_ranked.json")
    ensemble_pdb_path = os.path.join(job_output_dir, "ensemble_ranked.pdb")
    df_models.to_csv(ensemble_csv_path, index=False)
    df_models.to_json(ensemble_json_path, orient="records", indent=4)
    _write_ranked_multimodel_pdb(
        df_models,
        ensemble_pdb_path,
        path_columns=["gromacs_minimized_full_pdb_path", "rebuilt_full_pdb_path"],
        energy_column="energy",
        rmsd_column="rmsd_to_reference_A",
        remark_factory=lambda row, file_rank, energy_rank, rmsd_rank, pdb_path: [
            (
                f"QTF_SOURCE replica=replica_{int(row['energy_rank'])} "
                f"ensemble_id={int(row['ensemble_id'])}"
            ),
            f"QTF_RANK file_rank={file_rank} energy_rank={energy_rank} rmsd_rank={rmsd_rank}",
            (
                f"QTF_SCORE energy={_format_score_value(row.get('energy'))} "
                f"gromacs_potential_kj_mol={_format_score_value(row.get('gromacs_potential_kj_mol'))} "
                f"gromacs_potential_kcal_mol={_format_score_value(row.get('gromacs_potential_kcal_mol'))} "
                f"rmsd_A={_format_score_value(row.get('rmsd_to_reference_A'))} "
                f"raw_rmsd_A={_format_score_value(row.get('raw_rmsd_to_reference_A'))} "
                f"gromacs_rmsd_A={_format_score_value(row.get('gromacs_rmsd_to_reference_A'))}"
            ),
            f"QTF_PDB_SOURCE {pdb_path}",
        ],
    )

    if snapshot_rows:
        df_snapshots = pd.DataFrame(snapshot_rows)
        if snapshot_sort_by == "rmsd" and true_rmsd_coords is not None:
            df_snapshots = df_snapshots.sort_values(
                ["snapshot_effective_rmsd_to_reference_A", "snapshot_energy"],
                na_position="last",
            ).reset_index(drop=True)
        else:
            df_snapshots = df_snapshots.sort_values(["snapshot_energy"], na_position="last").reset_index(drop=True)
        df_snapshots["snapshot_global_rank"] = np.arange(1, len(df_snapshots) + 1)
        snapshot_csv_path = os.path.join(job_output_dir, "snapshot_ranked.csv")
        snapshot_json_path = os.path.join(job_output_dir, "snapshot_ranked.json")
        snapshot_pdb_path = os.path.join(job_output_dir, "snapshot_ranked.pdb")
        df_snapshots.to_csv(snapshot_csv_path, index=False)
        df_snapshots.to_json(snapshot_json_path, orient="records", indent=4)
        _write_ranked_multimodel_pdb(
            df_snapshots,
            snapshot_pdb_path,
            path_columns=["snapshot_gromacs_minimized_full_pdb_path", "snapshot_pdb_path"],
            energy_column="snapshot_energy",
            rmsd_column="snapshot_effective_rmsd_to_reference_A",
            remark_factory=lambda row, file_rank, energy_rank, rmsd_rank, pdb_path: [
                (
                    f"QTF_SOURCE replica=replica_{int(row['ensemble_rank'])} "
                    f"ensemble_id={int(row['ensemble_id'])} "
                    f"snapshot_rank_within_replica={int(row['snapshot_rank_within_replica'])}"
                ),
                (
                    f"QTF_RANK file_rank={file_rank} energy_rank={energy_rank} "
                    f"rmsd_rank={rmsd_rank} snapshot_global_rank={int(row['snapshot_global_rank'])}"
                ),
                (
                    f"QTF_SCORE energy={_format_score_value(row.get('snapshot_energy'))} "
                    f"gromacs_potential_kj_mol={_format_score_value(row.get('snapshot_gromacs_potential_kj_mol'))} "
                    f"gromacs_potential_kcal_mol={_format_score_value(row.get('snapshot_gromacs_potential_kcal_mol'))} "
                    f"effective_rmsd_A={_format_score_value(row.get('snapshot_effective_rmsd_to_reference_A'))} "
                    f"raw_rmsd_A={_format_score_value(row.get('snapshot_rmsd_to_reference_A'))} "
                    f"gromacs_rmsd_A={_format_score_value(row.get('snapshot_gromacs_rmsd_to_reference_A'))}"
                ),
                f"QTF_PDB_SOURCE {pdb_path}",
            ],
        )

    best_row = df_models.sort_values("energy").iloc[0]
    p_e2e = float(best_row["pred_e2e_A"])
    p_rg = float(best_row["pred_rg_A"])
    t_e2e = float(best_row["ref_e2e_A"]) if not np.isnan(best_row["ref_e2e_A"]) else np.nan
    t_rg = float(best_row["ref_rg_A"]) if not np.isnan(best_row["ref_rg_A"]) else np.nan
    rmsd_best = float(best_row["rmsd_to_reference_A"]) if not np.isnan(best_row["rmsd_to_reference_A"]) else np.nan
    raw_rmsd_best = float(best_row["raw_rmsd_to_reference_A"]) if not np.isnan(best_row["raw_rmsd_to_reference_A"]) else np.nan
    gromacs_rmsd_best = float(best_row["gromacs_rmsd_to_reference_A"]) if not np.isnan(best_row["gromacs_rmsd_to_reference_A"]) else np.nan

    runtime = (time.time() - time_start) / 60.0

    summary_data = {
        "End-to-End Dist (Å)": p_e2e,
        "End-to-End Target (Å)": t_e2e,
        "End-to-End Status": ("EXPANDED" if (not np.isnan(t_e2e) and p_e2e > t_e2e + 5) else "GOOD") if args.mode=="predict_and_compare" else None,
        "Rg (Å)": p_rg,
        "Rg Target (Å)": t_rg,
        "Rg Status": ("PUFFY" if (not np.isnan(t_rg) and p_rg > t_rg + 2) else "COMPACT") if args.mode=="predict_and_compare" else None,
        "Backbone RMSD (Å)": float(f"{rmsd_best:.3f}") if not np.isnan(rmsd_best) else None,
        "Raw Backbone RMSD (Å)": float(f"{raw_rmsd_best:.3f}") if not np.isnan(raw_rmsd_best) else None,
        "GROMACS Backbone RMSD (Å)": float(f"{gromacs_rmsd_best:.3f}") if not np.isnan(gromacs_rmsd_best) else None,
        "RMSD Status": ("GOOD" if (not np.isnan(rmsd_best) and rmsd_best < 2.0) else "BAD") if args.mode=="predict_and_compare" else None,
        "Runtime (minutes)": float(runtime),
        "Ensemble Size": int(ensemble_size),
        "Sequence": sequence,
        "mode": args.mode,
        "Reference Structure": ((reference_pdb_path or reference_structure_pdb_id) if (reference_pdb_path or reference_structure_pdb_id) is not None and args.mode != "predict_only" else None),
        "rmsd_mode": args.rmsd_mode,
        "rmsd_residue_scope": args.rmsd_residue_scope,
        "energy_backend": args.energy_backend,
        "use_e2e_constraint": bool(args.use_e2e_constraint),
        "e2e_scale": float(args.e2e_scale),
        "rosetta_repack": bool(args.rosetta_repack),
        "rosetta_fa_min": bool(args.rosetta_fa_min),
        "rosetta_cen_min": bool(args.rosetta_cen_min),
        "omega_mode": args.omega_mode,
        "Top K": int(top_k) if top_k is not None else None,
        "Top Frac": float(top_frac) if top_frac is not None else None,
        "Snapshot Sort By": snapshot_sort_by,
        "Snapshot Energy Gap": args.snapshot_energy_gap,
        "Snapshot GROMACS Enabled": bool(args.gromacs_minimize),
        "Initial Params": args.initial_params,
        "Initial Params Select": args.initial_params_select,
        "Circuit Parameters": parameter_manifest_path,
        "Output Dir": job_output_dir,
    }

    summary_csv_path = os.path.join(job_output_dir, "summary_results.csv")
    summary_json_path = os.path.join(job_output_dir, "summary_results.json")
    df_summary = pd.DataFrame([summary_data])
    df_summary.to_csv(summary_csv_path, index=False)
    with open(summary_json_path, "w") as f:
        json.dump({k: _jsonify(v) for k, v in summary_data.items()}, f, indent=4)

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
    main()
