#!/usr/bin/env python3
"""
Beam-search benchmarking for QTF energy function.

Goal: find a near-exhaustive (but tractable) discrete "ground truth" minimum
for short peptides (10–20 aa) using:
- Ramachandran basin grids (5 basins, ±window, step)
- chi rotamers (3-state)
- beam search width B

Scoring is done via the SAME QuantumBiophysicsFolder.energy_function used by the
quantum folding pipeline (stage 3), and energy term decompositions are saved.

Outputs (in --outdir):
- beamsearch_ranked.csv
- beamsearch_ranked.json
- beamsearch_partial_states.csv   (beam survivors at each residue depth)
- beamsearch_best.json
- (optional, if --reference_pdb) native-like labels and RMSD columns
"""
import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import QTF.runner as runner
import QTF.evaluator as evaluator


def _jsonify(x):
    """Convert numpy types to JSON-safe Python types."""
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def deg(vals):
    return [np.deg2rad(v) for v in vals]


@dataclass
class State:
    """A partial or full conformer state."""
    angles: np.ndarray              # angle vector (radians) for prefix (residues 0..depth-1)
    depth: int                      # number of residues included
    energy: float                   # energy score at this depth (stage 3 energy on truncated folder)
    terms: Dict[str, float]         # energy term decomposition
    # optional bookkeeping
    parent_id: Optional[int] = None
    local_choice: Optional[str] = None


def make_backbone_pairs(window_deg: int, step_deg: int, centers_deg: List[Tuple[int, int]]):
    """Return list of (phi, psi) pairs (radians) from basin centers + local grid."""
    offsets = list(range(-window_deg, window_deg + 1, step_deg))
    pairs = []
    for (cphi, cpsi) in centers_deg:
        for dphi in offsets:
            for dpsi in offsets:
                pairs.append((np.deg2rad(cphi + dphi), np.deg2rad(cpsi + dpsi)))
    return pairs


def residue_assignments(dof_entries: List[Dict[str, Any]],
                        backbone_pairs: List[Tuple[float, float]],
                        chi_rotamers: List[float],
                        include_chi: bool = False) -> List[np.ndarray]:
    """
    Build a list of per-residue angle assignments (in the order of dof_entries).
    - Couples phi/psi via backbone_pairs when both are present.
    - chi angles get 3-state rotamer options.
    """
    # Identify which dofs are present
    types = [d["type"] for d in dof_entries]
    has_phi = "phi" in types
    has_psi = "psi" in types

    # backbone assignment list as dict type->value
    backbone_assigns: List[Dict[str, float]] = []
    if has_phi and has_psi:
        for phi, psi in backbone_pairs:
            backbone_assigns.append({"phi": phi, "psi": psi})
    elif has_phi and not has_psi:
        # use phi values from pairs, psi ignored
        for phi, _psi in backbone_pairs:
            backbone_assigns.append({"phi": phi})
    elif has_psi and not has_phi:
        for _phi, psi in backbone_pairs:
            backbone_assigns.append({"psi": psi})
    else:
        backbone_assigns = [{}]

    # sidechain rotamers (optional)
    # In the coarse/centroid model, treat sidechains as having at most ONE chi DOF (chi1-like).
    # For benchmarking backbone landscapes (recommended), keep this OFF by default.
    chi_types = [t for t in types if "chi" in t]
    chi_types = chi_types[:1]

    if include_chi and chi_types:
        assigns = []
        chi_t = chi_types[0]
        for b in backbone_assigns:
            for v in chi_rotamers:   # only 3 options
                d = dict(b)
                d[chi_t] = float(v)
                assigns.append(d)
    else:
        assigns = backbone_assigns

    # convert dict assigns into angle vectors aligned with dof_entries order
    out = []
    for a in assigns:
        vec = []
        for d in dof_entries:
            t = d["type"]
            if t in a:
                vec.append(a[t])
            else:
                # if missing (e.g. residue lacks phi/psi), default 0
                vec.append(0.0)
        out.append(np.array(vec, dtype=float))
    return out


def eval_energy_terms(folder: runner.QuantumBiophysicsFolder, angle_vec: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate energy_function at stage 3 for a given angle vector by temporarily
    overriding _get_angles. Returns (energy, terms).
    """
    dummy_params = np.zeros(folder.n_params, dtype=float)
    orig_get_angles = folder._get_angles
    try:
        folder._get_angles = lambda _params: angle_vec
        folder.current_stage = 3
        E = float(folder.energy_function(dummy_params, return_terms=True))
        terms = getattr(folder, "last_energy_terms", {}) or {}
        terms = {str(k): float(v) for k, v in terms.items()}
        return E, terms
    finally:
        folder._get_angles = orig_get_angles


def build_ca_coords(folder: runner.QuantumBiophysicsFolder, angle_vec: np.ndarray) -> np.ndarray:
    """Build full structure and extract CA coordinates."""
    orig_get_angles = folder._get_angles
    try:
        folder._get_angles = lambda _params: angle_vec
        dummy_params = np.zeros(folder.n_params, dtype=float)
        angle_vec2 = folder._get_angles(dummy_params)
        coords, _, _ = folder.build_full_structure(angle_vec2)
        ca = np.array([coords[i] for i, lbl in enumerate(folder.static_labels) if lbl[1] == "CA"])
        return ca
    finally:
        folder._get_angles = orig_get_angles



def _backbone_key(folder_k: runner.QuantumBiophysicsFolder, angles_rad: np.ndarray) -> Tuple[int, ...]:
    """Key for deduplicating by backbone angles (phi/psi only), rounded to nearest degree."""
    key = []
    for i, d in enumerate(folder_k.dof_map):
        if i >= len(angles_rad):
            break
        if d.get("type") in ("phi", "psi"):
            key.append(int(round(np.rad2deg(angles_rad[i]))))
    return tuple(key)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--forcefield", default="amber")
    ap.add_argument("--beam_width", type=int, default=1000)
    ap.add_argument("--window_deg", type=int, default=15)
    ap.add_argument("--step_deg", type=int, default=15)
    ap.add_argument("--include_chi", action="store_true", help="include 1 chi rotamer DOF per residue (default: backbone-only)")
    ap.add_argument("--dedup_backbone", action="store_true", help="deduplicate expanded states by backbone (phi/psi) at each depth")
    ap.add_argument("--top_k", type=int, default=200, help="save top_k ranked conformers (post-search)")
    ap.add_argument("--reference_pdb", default=None, help="optional PDB id for RMSD comparison")
    ap.add_argument("--average_reference_backbone", action="store_true")
    ap.add_argument("--native_thresh", type=float, default=2.0, help="RMSD threshold (Å) for native-like")
    ap.add_argument("--outdir", default="beamsearch_outputs")
    ap.add_argument("--save_partial", action="store_true", help="save beam survivors at each depth to CSV")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    seq = args.sequence.strip()
    L = len(seq)

    # 5 basin centers (deg): alpha, beta, PPII, turn/loop, left-handed
    basin_centers = [(-60, -45), (-135, 135), (-75, 145), (-60, 120), (60, 45), (-90, 90)]
    backbone_pairs = make_backbone_pairs(args.window_deg, args.step_deg, basin_centers)
    chi_rot = deg([-60, 60, 180])

    # Build a full-length folder and per-length truncated folders (for partial scoring)
    folders_by_k: Dict[int, runner.QuantumBiophysicsFolder] = {}
    for k in range(1, L + 1):
        folders_by_k[k] = runner.QuantumBiophysicsFolder(sequence=seq[:k], force_field=args.forcefield)
        folders_by_k[k].current_stage = 3

    full_folder = folders_by_k[L]

    # Build dof_map entries grouped by residue for the FULL folder (assume prefix-consistent)
    dofs_by_res: Dict[int, List[Dict[str, Any]]] = {}
    for d in full_folder.dof_map:
        r = int(d["res"])
        dofs_by_res.setdefault(r, []).append(d)

    # For each residue, precompute per-residue assignments (vectors aligned with dofs_by_res[r] order)
    residue_opts: Dict[int, List[np.ndarray]] = {}
    for r in range(L):
        residue_opts[r] = residue_assignments(dofs_by_res.get(r, []), backbone_pairs, chi_rot, include_chi=args.include_chi)

    # We also need to know how many dofs are included up to each residue (prefix length)
    dof_count_by_res = [len(dofs_by_res.get(r, [])) for r in range(L)]
    prefix_counts = np.cumsum(dof_count_by_res)  # length L

    # Beam search
    beam: List[State] = []
    partial_rows = []

    # Initialize at depth 0 with empty angles, zero energy
    beam = [State(angles=np.zeros((0,), dtype=float), depth=0, energy=0.0, terms={})]

    for r in range(L):
        k = r + 1
        folder_k = folders_by_k[k]
        new_states: List[State] = []

        # Expand each beam state with residue r assignments
        for s_idx, st in enumerate(beam):
            for opt_idx, opt_vec in enumerate(residue_opts[r]):
                angles_new = np.concatenate([st.angles, opt_vec], axis=0)

                # Ensure angle vector length matches folder_k.total_angles
                # If mismatch (due to implementation details), skip safely
                if angles_new.shape[0] != folder_k.total_angles:
                    continue

                E, terms = eval_energy_terms(folder_k, angles_new)
                new_states.append(State(
                    angles=angles_new,
                    depth=k,
                    energy=E,
                    terms=terms,
                    parent_id=s_idx,
                    local_choice=f"res{r}_opt{opt_idx}"
                ))


        # Optionally deduplicate by backbone (phi/psi) to avoid filling the beam with chi-variants
        if args.dedup_backbone:
            uniq = {}
            for st2 in new_states:
                kkey = _backbone_key(folder_k, st2.angles)
                prev = uniq.get(kkey)
                if (prev is None) or (st2.energy < prev.energy):
                    uniq[kkey] = st2
            new_states = list(uniq.values())


        # Prune to beam width
        new_states.sort(key=lambda x: x.energy)
        beam = new_states[:args.beam_width]

        if args.save_partial:
            for rank, st in enumerate(beam, start=1):
                row = {
                    "depth": st.depth,
                    "beam_rank": rank,
                    "energy": st.energy,
                    **{f"term_{k}": v for k, v in st.terms.items()},
                }
                partial_rows.append(row)

        print(f"[beam] depth {k}/{L}: expanded {len(new_states):,} -> kept {len(beam):,}. best E={beam[0].energy:.3f}")

    # Final full-length candidates are in beam
    final_states = beam
    final_states.sort(key=lambda x: x.energy)

    # Optional reference backbone
    true_ca = None
    if args.reference_pdb:
        true_ca = evaluator.get_ground_truth_backbone(args.reference_pdb, average_backbone=args.average_reference_backbone)

    # Build final dataframe rows (include energy terms + optional metrics)
    rows = []
    for rank, st in enumerate(final_states, start=1):
        row = {
            "energy_rank": rank,
            "energy": float(st.energy),
            "depth": st.depth,
        }
        # terms
        for kterm, v in st.terms.items():
            row[f"term_{kterm}"] = float(v)

        # angles for reproducibility / later reconstruction
        row["angles_deg_json"] = json.dumps([float(np.rad2deg(a)) for a in st.angles])

        # Optional RMSD + physics metrics
        if true_ca is not None:
            pred_ca = build_ca_coords(full_folder, st.angles)
            n = min(len(pred_ca), len(true_ca))
            pred_ca_n = pred_ca[:n]
            true_ca_n = true_ca[:n]
            rmsd, _aligned = runner.StabilityAnalyzer.kabsch_rmsd(pred_ca_n, true_ca_n)
            row["rmsd_to_reference_A"] = float(rmsd)

            p_e2e, p_rg = evaluator.calculate_physics_metrics(pred_ca_n)
            t_e2e, t_rg = evaluator.calculate_physics_metrics(true_ca_n)
            row["pred_e2e_A"] = float(p_e2e)
            row["pred_rg_A"] = float(p_rg)
            row["ref_e2e_A"] = float(t_e2e)
            row["ref_rg_A"] = float(t_rg)

            row["native_like"] = bool(float(rmsd) <= float(args.native_thresh))

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("energy").reset_index(drop=True)

    # Save ranked (top_k if desired)
    df_out = df.head(args.top_k).copy()

    csv_path = os.path.join(args.outdir, "beamsearch_ranked.csv")
    json_path = os.path.join(args.outdir, "beamsearch_ranked.json")
    df_out.to_csv(csv_path, index=False)

    payload = [{k: _jsonify(v) for k, v in rec.items()} for rec in df_out.to_dict(orient="records")]
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Save partial survivors
    if args.save_partial and partial_rows:
        pd.DataFrame(partial_rows).to_csv(os.path.join(args.outdir, "beamsearch_partial_states.csv"), index=False)

    # Save best summary JSON
    best = df.iloc[0].to_dict()
    best_summary = {
        "sequence": seq,
        "forcefield": args.forcefield,
        "beam_width": args.beam_width,
        "window_deg": args.window_deg,
        "step_deg": args.step_deg,
        "basin_centers_deg": basin_centers,
        "chi_rotamers_deg": [-60, 60, 180],
        "native_thresh_A": args.native_thresh,
        "reference_pdb": args.reference_pdb,
        "best": {k: _jsonify(v) for k, v in best.items()},
    }
    with open(os.path.join(args.outdir, "beamsearch_best.json"), "w") as f:
        json.dump(best_summary, f, indent=2)

    # If reference is present, add extra summary stats
    if args.reference_pdb:
        native_count = int(df_out["native_like"].sum()) if "native_like" in df_out.columns else 0
        with open(os.path.join(args.outdir, "native_like_counts.json"), "w") as f:
            json.dump({
                "native_like": native_count,
                "non_native": int(len(df_out) - native_count),
                "native_thresh_A": args.native_thresh
            }, f, indent=2)

        # term deltas: native vs non-native
        term_cols = [c for c in df_out.columns if c.startswith("term_") and c not in ("term_total",)]
        if term_cols and "native_like" in df_out.columns:
            rows2 = []
            for c in term_cols + ["energy"]:
                mean_nat = df_out.loc[df_out["native_like"], c].mean()
                mean_non = df_out.loc[~df_out["native_like"], c].mean()
                rows2.append({
                    "feature": c,
                    "mean_native_like": float(mean_nat) if pd.notna(mean_nat) else None,
                    "mean_non_native": float(mean_non) if pd.notna(mean_non) else None,
                    "delta_native_minus_non": float(mean_nat - mean_non) if (pd.notna(mean_nat) and pd.notna(mean_non)) else None
                })
            pd.DataFrame(rows2).sort_values("delta_native_minus_non").to_csv(
                os.path.join(args.outdir, "native_like_term_differences.csv"), index=False
            )

        # correlations vs rmsd
        if term_cols and "rmsd_to_reference_A" in df_out.columns:
            corr_rows = []
            target = "rmsd_to_reference_A"
            for c in term_cols + ["energy"]:
                pearson = df_out[[c, target]].corr(method="pearson").iloc[0, 1]
                spearman = df_out[[c, target]].corr(method="spearman").iloc[0, 1]
                corr_rows.append({"feature": c, "pearson_r": float(pearson), "spearman_r": float(spearman)})
            pd.DataFrame(corr_rows).sort_values("spearman_r").to_csv(
                os.path.join(args.outdir, "term_correlations_vs_rmsd.csv"), index=False
            )

    print(f"[beam] wrote: {csv_path}")
    print(f"[beam] wrote: {json_path}")
    print(f"[beam] wrote: {os.path.join(args.outdir, 'beamsearch_best.json')}")

df = pd.read_csv("beam200_dedup/beamsearch_ranked.csv")
print(df["rmsd_to_reference_A"].nunique())
print(df["energy"].nunique())
print(df["term_geometry"].nunique())
print(df["term_sasa"].nunique())

if __name__ == "__main__":
    main()
