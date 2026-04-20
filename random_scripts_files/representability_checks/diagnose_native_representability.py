#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mdtraj as md
import numpy as np
import pandas as pd


SELECTIVE_CHI_MAP: Dict[str, List[str]] = {
    "Y": ["chi1", "chi2"], "W": ["chi1", "chi2"], "F": ["chi1", "chi2"], "H": ["chi1", "chi2"],
    "D": ["chi1"], "E": ["chi1"], "N": ["chi1"], "Q": ["chi1"],
    "T": ["chi1"], "S": ["chi1"],
    "V": ["chi1"], "I": ["chi1"], "L": ["chi1"], "M": ["chi1"],
    "K": ["chi1"], "R": ["chi1"], "C": ["chi1"], "P": ["chi1"],
    "A": [], "G": [],
}

BASIN_CENTERS_DEG: List[Tuple[int, int]] = [
    (-60, -45), (-135, 135), (-75, 145), (-60, 120), (60, 45), (-90, 90)
]

CHI_ROTAMERS_DEG = [-60.0, 60.0, 180.0]


def wrap_deg(x: float) -> float:
    return ((x + 180.0) % 360.0) - 180.0


def circ_abs_diff_deg(a: float, b: float) -> float:
    return abs(wrap_deg(a - b))


def make_backbone_pairs_deg(window_deg: int, step_deg: int) -> List[Tuple[float, float]]:
    offsets = list(range(-window_deg, window_deg + 1, step_deg))
    pairs = []
    for cphi, cpsi in BASIN_CENTERS_DEG:
        for dphi in offsets:
            for dpsi in offsets:
                pairs.append((float(cphi + dphi), float(cpsi + dpsi)))
    return pairs


def allowed_chis_for_residue(aa: str, chi_mode: str) -> List[str]:
    all_chis_by_res = {
        "R": ["chi1", "chi2", "chi3", "chi4", "chi5"],
        "K": ["chi1", "chi2", "chi3", "chi4", "chi5"],
        "Q": ["chi1", "chi2", "chi3"],
        "E": ["chi1", "chi2", "chi3"],
        "M": ["chi1", "chi2", "chi3"],
        "D": ["chi1", "chi2"],
        "N": ["chi1", "chi2"],
        "F": ["chi1", "chi2"],
        "Y": ["chi1", "chi2"],
        "W": ["chi1", "chi2"],
        "H": ["chi1", "chi2"],
        "L": ["chi1", "chi2"],
        "I": ["chi1", "chi2"],
        "P": ["chi1", "chi2"],
        "C": ["chi1"],
        "S": ["chi1"],
        "T": ["chi1"],
        "V": ["chi1"],
    }
    available = all_chis_by_res.get(aa, [])
    if chi_mode == "all":
        return available
    if chi_mode == "chi1_only":
        return [c for c in available if c == "chi1"]
    if chi_mode == "selective":
        allowed = set(SELECTIVE_CHI_MAP.get(aa, ["chi1"]))
        return [c for c in available if c in allowed]
    raise ValueError(f"Unknown chi_mode: {chi_mode}")


def load_subset(pdb_path: str, chain: Optional[str]) -> tuple[md.Trajectory, list[int], str]:
    traj = md.load(pdb_path)
    top = traj.topology

    residues = []
    seq = []
    resseqs = []
    for res in top.residues:
        if chain and res.chain.chain_id != chain:
            continue
        if res.is_water:
            continue
        name3 = res.name.upper()
        aa3_to_1 = {
            "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H",
            "ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W",
            "TYR":"Y","VAL":"V"
        }
        if name3 not in aa3_to_1:
            continue
        residues.append(res.index)
        seq.append(aa3_to_1[name3])
        resseqs.append(res.resSeq)

    atom_indices = [a.index for a in top.atoms if a.residue.index in set(residues)]
    sub = traj.atom_slice(atom_indices)
    return sub, resseqs, "".join(seq)


def torsion_map_deg(traj: md.Trajectory) -> Dict[str, Dict[int, float]]:
    out: Dict[str, Dict[int, float]] = {"phi": {}, "psi": {}, "chi1": {}, "chi2": {}, "chi3": {}, "chi4": {}, "chi5": {}}
    top = traj.topology

    funcs = {
        "phi": md.compute_phi,
        "psi": md.compute_psi,
        "chi1": md.compute_chi1,
        "chi2": md.compute_chi2,
        "chi3": md.compute_chi3,
        "chi4": md.compute_chi4,
        "chi5": md.compute_chi5,
    }

    for name, func in funcs.items():
        idxs, vals = func(traj)
        if vals.size == 0:
            continue
        vals_deg = np.rad2deg(vals[0])
        for torsion_atoms, ang in zip(idxs, vals_deg):
            # Use the 2nd atom's residue for phi/psi/chi association, which matches mdtraj conventions well enough here
            atom_idx = int(torsion_atoms[1])
            res_idx = top.atom(atom_idx).residue.index
            out[name][res_idx] = float(wrap_deg(float(ang)))
    return out


def nearest_backbone_pair(phi: float, psi: float, pairs_deg: List[Tuple[float, float]]) -> tuple[Tuple[float, float], float, float]:
    best_pair = None
    best_phi_err = None
    best_psi_err = None
    best_score = None
    for pphi, ppsi in pairs_deg:
        ephi = circ_abs_diff_deg(phi, pphi)
        epsi = circ_abs_diff_deg(psi, ppsi)
        score = (ephi ** 2 + epsi ** 2) ** 0.5
        if best_score is None or score < best_score:
            best_score = score
            best_pair = (pphi, ppsi)
            best_phi_err = ephi
            best_psi_err = epsi
    return best_pair, float(best_phi_err), float(best_psi_err)


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose whether native torsions are representable in current QTF beam-search space.")
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--chain", default=None)
    ap.add_argument("--protein_name", default=None)
    ap.add_argument("--chi_mode", default="selective", choices=["chi1_only", "selective", "all"])
    ap.add_argument("--window_deg", type=int, default=30)
    ap.add_argument("--step_deg", type=int, default=15)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    traj, resseqs, seq = load_subset(args.pdb, args.chain)
    top = traj.topology
    tors = torsion_map_deg(traj)
    pairs_deg = make_backbone_pairs_deg(args.window_deg, args.step_deg)

    rows = []
    backbone_pair_rows = []
    chi_rows = []

    # map filtered residue order to mdtraj residue indices
    kept_residues = []
    for res in top.residues:
        if args.chain and res.chain.chain_id != args.chain:
            continue
        if res.is_water:
            continue
        if res.name.upper() not in {
            "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS",
            "MET","PHE","PRO","SER","THR","TRP","TYR","VAL"
        }:
            continue
        kept_residues.append(res)

    for i, res in enumerate(kept_residues):
        aa = seq[i]
        res_idx = res.index
        resseq = res.resSeq

        phi = tors["phi"].get(res_idx)
        psi = tors["psi"].get(res_idx)

        has_phi = phi is not None
        has_psi = psi is not None

        if has_phi and has_psi:
            best_pair, phi_err, psi_err = nearest_backbone_pair(phi, psi, pairs_deg)
            representable_backbone = (phi_err < 1e-9 and psi_err < 1e-9)
            pair_in_window = any(
                circ_abs_diff_deg(phi, pphi) <= args.window_deg + 1e-9 and
                circ_abs_diff_deg(psi, ppsi) <= args.window_deg + 1e-9
                for pphi, ppsi in BASIN_CENTERS_DEG
            )
            backbone_pair_rows.append({
                "res_index0": i,
                "resseq": resseq,
                "aa": aa,
                "native_phi_deg": phi,
                "native_psi_deg": psi,
                "nearest_phi_deg": best_pair[0],
                "nearest_psi_deg": best_pair[1],
                "phi_error_deg": phi_err,
                "psi_error_deg": psi_err,
                "backbone_pair_error_deg_euclid": float((phi_err ** 2 + psi_err ** 2) ** 0.5),
                "inside_any_basin_window": bool(pair_in_window),
                "exactly_on_backbone_grid": bool(representable_backbone),
            })
        elif has_phi or has_psi:
            # termini / partial availability
            native_val = phi if has_phi else psi
            axis = "phi" if has_phi else "psi"
            allowed_vals = sorted({pphi if has_phi else ppsi for pphi, ppsi in pairs_deg})
            nearest = min(allowed_vals, key=lambda x: circ_abs_diff_deg(native_val, x))
            err = circ_abs_diff_deg(native_val, nearest)
            backbone_pair_rows.append({
                "res_index0": i,
                "resseq": resseq,
                "aa": aa,
                "native_phi_deg": phi,
                "native_psi_deg": psi,
                "nearest_phi_deg": nearest if has_phi else None,
                "nearest_psi_deg": nearest if has_psi else None,
                "phi_error_deg": err if has_phi else None,
                "psi_error_deg": err if has_psi else None,
                "backbone_pair_error_deg_euclid": err,
                "inside_any_basin_window": None,
                "exactly_on_backbone_grid": bool(err < 1e-9),
                "partial_backbone_axis": axis,
            })
        else:
            backbone_pair_rows.append({
                "res_index0": i,
                "resseq": resseq,
                "aa": aa,
                "native_phi_deg": None,
                "native_psi_deg": None,
                "nearest_phi_deg": None,
                "nearest_psi_deg": None,
                "phi_error_deg": None,
                "psi_error_deg": None,
                "backbone_pair_error_deg_euclid": None,
                "inside_any_basin_window": None,
                "exactly_on_backbone_grid": None,
            })

        allowed = allowed_chis_for_residue(aa, args.chi_mode)
        available_all = allowed_chis_for_residue(aa, "all")
        row = {
            "res_index0": i,
            "resseq": resseq,
            "aa": aa,
            "allowed_chis_under_mode": ",".join(allowed),
            "available_chis_full": ",".join(available_all),
        }

        for chi_name in ["chi1", "chi2", "chi3", "chi4", "chi5"]:
            native = tors[chi_name].get(res_idx)
            row[f"native_{chi_name}_deg"] = native
            row[f"{chi_name}_exists_in_search_space"] = chi_name in allowed

            if native is None:
                row[f"{chi_name}_nearest_rotamer_deg"] = None
                row[f"{chi_name}_error_to_nearest_rotamer_deg"] = None
                row[f"{chi_name}_exactly_on_rotamer_grid"] = None
            elif chi_name in allowed:
                nearest = min(CHI_ROTAMERS_DEG, key=lambda x: circ_abs_diff_deg(native, x))
                err = circ_abs_diff_deg(native, nearest)
                row[f"{chi_name}_nearest_rotamer_deg"] = nearest
                row[f"{chi_name}_error_to_nearest_rotamer_deg"] = err
                row[f"{chi_name}_exactly_on_rotamer_grid"] = bool(err < 1e-9)
                chi_rows.append({
                    "res_index0": i,
                    "resseq": resseq,
                    "aa": aa,
                    "chi_name": chi_name,
                    "native_deg": native,
                    "nearest_rotamer_deg": nearest,
                    "error_deg": err,
                    "exists_in_search_space": True,
                })
            else:
                row[f"{chi_name}_nearest_rotamer_deg"] = None
                row[f"{chi_name}_error_to_nearest_rotamer_deg"] = None
                row[f"{chi_name}_exactly_on_rotamer_grid"] = None
                if native is not None:
                    chi_rows.append({
                        "res_index0": i,
                        "resseq": resseq,
                        "aa": aa,
                        "chi_name": chi_name,
                        "native_deg": native,
                        "nearest_rotamer_deg": None,
                        "error_deg": None,
                        "exists_in_search_space": False,
                    })

        rows.append(row)

    df = pd.DataFrame(rows)
    df_backbone = pd.DataFrame(backbone_pair_rows)
    df_chi = pd.DataFrame(chi_rows)

    protein_name = args.protein_name or Path(args.pdb).stem

    df.to_csv(outdir / f"{protein_name}_native_torsion_representability.csv", index=False)
    df_backbone.to_csv(outdir / f"{protein_name}_backbone_grid_diagnostics.csv", index=False)
    df_chi.to_csv(outdir / f"{protein_name}_chi_diagnostics.csv", index=False)

    summary = {
        "protein_name": protein_name,
        "pdb": args.pdb,
        "chain": args.chain,
        "chi_mode": args.chi_mode,
        "window_deg": args.window_deg,
        "step_deg": args.step_deg,
        "n_residues": int(len(df)),
        "n_backbone_with_both_phi_psi": int(df_backbone["native_phi_deg"].notna().fillna(False).astype(bool).mul(df_backbone["native_psi_deg"].notna().fillna(False).astype(bool)).sum()) if not df_backbone.empty else 0,
        "mean_backbone_pair_error_deg": float(df_backbone["backbone_pair_error_deg_euclid"].dropna().mean()) if not df_backbone.empty else None,
        "max_backbone_pair_error_deg": float(df_backbone["backbone_pair_error_deg_euclid"].dropna().max()) if not df_backbone.empty else None,
        "n_backbone_exactly_on_grid": int(df_backbone["exactly_on_backbone_grid"].fillna(False).astype(bool).sum()) if not df_backbone.empty else 0,
        "n_backbone_inside_any_basin_window": int(df_backbone["inside_any_basin_window"].fillna(False).astype(bool).sum()) if "inside_any_basin_window" in df_backbone else 0,
        "n_observed_chis": int(df_chi["native_deg"].notna().sum()) if not df_chi.empty else 0,
        "n_observed_chis_in_search_space": int(df_chi["exists_in_search_space"].fillna(False).astype(bool).sum()) if not df_chi.empty else 0,
        "mean_chi_error_deg_for_exposed_chis": float(df_chi.loc[df_chi["exists_in_search_space"] == True, "error_deg"].dropna().mean()) if not df_chi.empty else None,
        "max_chi_error_deg_for_exposed_chis": float(df_chi.loc[df_chi["exists_in_search_space"] == True, "error_deg"].dropna().max()) if not df_chi.empty else None,
        "n_missing_native_chis_due_to_mode": int(((df_chi["native_deg"].notna()) & (df_chi["exists_in_search_space"] == False)).sum()) if not df_chi.empty else 0,
    }

    (outdir / f"{protein_name}_representability_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
