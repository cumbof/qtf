#!/usr/bin/env python3
"""
Beam-search benchmarking for QTF energy function.

Goal: find a near-exhaustive (but tractable) discrete "ground truth" minimum
for short peptides (10–20 aa) using:
- Ramachandran basin grids
- residue-specific chi rotamers
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
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd

import QTF.runner as runner
import QTF.utils as utils
import QTF.qtf_gromacs as qtf_gromacs


def _jsonify(x):
    """Convert numpy types to JSON-safe Python types."""
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

def deg(vals):
    return [np.deg2rad(v) for v in vals]


def kabsch_rmsd(P, Q):
    """
    Calculates RMSD between two coordinate sets after optimal alignment.
    Returns RMSD only.
    """
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch: {P.shape} vs {Q.shape}")

    P_centered = P - np.mean(P, axis=0)
    Q_centered = Q - np.mean(Q, axis=0)

    H = np.dot(P_centered.T, Q_centered)
    V, S, Wt = np.linalg.svd(H)

    d = (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0
    if d:
        V[:, -1] = -V[:, -1]

    R = np.dot(V, Wt)

    P_rotated = np.dot(P_centered, R)
    diff = P_rotated - Q_centered
    rms = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    return float(rms)


def core_ca_slice(coords: np.ndarray) -> np.ndarray:
    """Return CA coordinates excluding flexible terminal residues.

    Uses residues 2..N-1 (1-indexed), i.e. drops the first and last CA.
    Falls back to all coordinates for very short chains.
    """
    arr = np.asarray(coords)
    if arr.shape[0] > 2:
        return arr[1:-1]
    return arr


def core_ca_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Kabsch RMSD over the core CA range: residue 2 through second-to-last."""
    return kabsch_rmsd(core_ca_slice(P), core_ca_slice(Q))


def core_ca_range_metadata(n_residues: int) -> Dict[str, object]:
    use_core = n_residues > 2
    return {
        "rmsd_ca_excludes_terminal_residues": bool(use_core),
        "rmsd_ca_start_residue_1indexed": 2 if use_core else 1,
        "rmsd_ca_end_residue_1indexed": (n_residues - 1) if use_core else n_residues,
        "rmsd_ca_n_aligned": (n_residues - 2) if use_core else n_residues,
    }


def load_reference_ca_coords(pdb_path: str) -> np.ndarray:
    """
    Load CA coordinates from the reference PDB in Angstroms.
    Uses the first model only if multiple models are present.
    """
    traj = md.load(pdb_path)
    ca_idx = traj.topology.select("name CA")
    if ca_idx is None or len(ca_idx) == 0:
        raise ValueError(f"No CA atoms found in reference PDB: {pdb_path}")
    return traj.xyz[0, ca_idx, :] * 10.0  # nm -> Angstrom


def get_tuning_settings():
    return {
        "hbond_scale": float(os.getenv("QTF_HBOND_SCALE", "0.75")),
        "sasa_scale": float(os.getenv("QTF_SASA_SCALE", "0.7")),
        "vdw_rep_scale": float(os.getenv("QTF_VDW_REP_SCALE", "0.01")),
        "vdw_attr_scale": float(os.getenv("QTF_VDW_ATTR_SCALE", "0.1")),
        "rotamer_scale": float(os.getenv("QTF_ROTAMER_SCALE", "1.0")),
        "pi_stack_scale": float(os.getenv("QTF_PI_STACK_SCALE", "1.0")),
    }

def pdb_id_from_path(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    s = str(p).strip()
    if not s:
        return None
    return Path(s).stem.upper()

def infer_protein_name(explicit_name: Optional[str], reference_pdb: Optional[str], sequence: str) -> str:
    """
    Prefer an explicitly supplied protein name. Otherwise fall back to PDB basename,
    then sequence.
    """
    if explicit_name and str(explicit_name).strip():
        return str(explicit_name).strip()
    if reference_pdb and str(reference_pdb).strip():
        return os.path.splitext(os.path.basename(str(reference_pdb).strip()))[0]
    return sequence


@dataclass
class State:
    """A partial or full conformer state."""
    angles: np.ndarray
    depth: int
    energy: float
    terms: Dict[str, float]
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


def residue_assignments(
    dof_entries: List[Dict[str, Any]],
    backbone_pairs: List[Tuple[float, float]],
    chi_rotamer_map: Dict[str, List[float]],
    omega_values: Optional[List[float]] = None,
    max_options_per_residue: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """
    Build per-residue assignment vectors in dof_entries order.

    IMPORTANT: this function is deliberately lazy when max_options_per_residue
    is set. With large windows (e.g. 125/5) and chi_mode='all', the local
    Cartesian product can be millions of combinations for a single residue.
    The old implementation materialized that full list and then downsampled it,
    which can blow up memory/time before beam search even starts.

    Behavior:
      * max_options_per_residue is None or <=0: enumerate the full local space.
      * otherwise: uniformly sample up to max_options_per_residue unique local
        combinations from the implicit Cartesian product without materializing
        the full product.

    The returned options are still real combinations from the requested
    window/step/chi-mode space; we just avoid constructing combinations that
    will never be scored.
    """
    types = [d["type"] for d in dof_entries]
    has_phi = "phi" in types
    has_psi = "psi" in types
    has_omega = "omega" in types

    if has_phi and has_psi:
        backbone_assigns = [{"phi": float(phi), "psi": float(psi)} for phi, psi in backbone_pairs]
    elif has_phi:
        backbone_assigns = [{"phi": float(phi)} for phi, _ in backbone_pairs]
    elif has_psi:
        backbone_assigns = [{"psi": float(psi)} for _, psi in backbone_pairs]
    else:
        backbone_assigns = [{}]

    omega_assigns = [{"omega": float(v)} for v in (omega_values or [np.pi])] if has_omega else [{}]

    chi_types = sorted(
        [t for t in types if t.startswith("chi")],
        key=lambda x: int(x.replace("chi", ""))
    )
    chi_values = [list(map(float, chi_rotamer_map.get(t, [0.0]))) for t in chi_types]

    axes = [backbone_assigns, omega_assigns] + chi_values
    axis_lengths = [len(ax) for ax in axes]
    total = int(np.prod(axis_lengths, dtype=np.int64)) if axis_lengths else 1

    if max_options_per_residue is None or int(max_options_per_residue) <= 0 or total <= int(max_options_per_residue):
        selected = range(total)
    else:
        if rng is None:
            rng = np.random.default_rng()
        selected = sorted(rng.choice(total, size=int(max_options_per_residue), replace=False).tolist())

    def decode(flat_idx: int) -> np.ndarray:
        idx = int(flat_idx)
        per_axis = []
        for n in reversed(axis_lengths):
            per_axis.append(idx % n)
            idx //= n
        per_axis = list(reversed(per_axis))

        assign = dict(backbone_assigns[per_axis[0]])
        assign.update(omega_assigns[per_axis[1]])
        for chi_t, vals, axis_i in zip(chi_types, chi_values, per_axis[2:]):
            assign[chi_t] = float(vals[axis_i])

        return np.array([assign.get(d["type"], 0.0) for d in dof_entries], dtype=float)

    return [decode(i) for i in selected]


def maybe_downsample_options(
    options: List[np.ndarray],
    max_options_per_residue: Optional[int],
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Backward-compatible helper retained for older call sites.

    New code should pass max_options_per_residue into residue_assignments()
    directly so huge local option lists are never materialized. This helper only
    handles already-built lists.
    """
    if max_options_per_residue is None or int(max_options_per_residue) <= 0 or len(options) <= int(max_options_per_residue):
        return options
    idx = rng.choice(len(options), size=int(max_options_per_residue), replace=False)
    idx = sorted(idx.tolist())
    return [options[i] for i in idx]

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
        builder = getattr(folder, "build_output_structure", folder.build_full_structure)
        coords, _, _ = builder(angle_vec2)
        ca = np.array([coords[i] for i, lbl in enumerate(folder.static_labels) if lbl[1] == "CA"])
        return ca
    finally:
        folder._get_angles = orig_get_angles

def build_full_coords(folder: runner.QuantumBiophysicsFolder, angle_vec: np.ndarray):
    """Build full QTF structure and return coordinates, labels, and bonds."""
    orig_get_angles = folder._get_angles
    try:
        folder._get_angles = lambda _params: angle_vec
        dummy_params = np.zeros(folder.n_params, dtype=float)
        angle_vec2 = folder._get_angles(dummy_params)
        builder = getattr(folder, "build_output_structure", folder.build_full_structure)
        coords, labels, bonds = builder(angle_vec2)
        return coords, labels, bonds
    finally:
        folder._get_angles = orig_get_angles





def backbone_signature(folder: runner.QuantumBiophysicsFolder, angle_vec: np.ndarray, round_deg: float = 15.0):
    """
    Signature using ONLY backbone torsions, rounded into bins.
    This preserves sidechain diversity within a backbone family.
    """
    sig = []
    for dof, ang in zip(folder.dof_map, angle_vec):
        if dof["type"] in ("phi", "psi", "omega"):
            degv = np.rad2deg(ang)
            sig.append((int(dof["res"]), dof["type"], int(np.round(degv / round_deg))))
    return tuple(sig)


def dedup_states_by_backbone(
    folder: runner.QuantumBiophysicsFolder,
    states: List[State],
    round_deg: float = 15.0,
) -> List[State]:
    best = {}
    for st in states:
        sig = backbone_signature(folder, st.angles, round_deg=round_deg)
        prev = best.get(sig)
        if prev is None or st.energy < prev.energy:
            best[sig] = st
    out = list(best.values())
    out.sort(key=lambda s: s.energy)
    return out


def nonlocal_heavy_clash_metrics(coords: np.ndarray, labels: List[Tuple[int, str, str]], min_allowed_A: float = 1.75):
    """
    Find the closest heavy-atom contact between residues separated by at least 2.

    This is a cheap post-filter for beam ranking. It removes obvious steric
    overlaps that would show up as fake bonds in viewers even when the score is
    otherwise low.
    """
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


def adjacent_heavy_clash_metrics(coords: np.ndarray, labels: List[Tuple[int, str, str]], min_allowed_A: float = 1.35, threshold_frac: float = 0.55):
    """
    Find the closest heavy-atom contact between adjacent residues.

    This is a narrower post-filter for the local peptide-bond region and is
    meant to catch false local overlaps that can still survive the score.
    """
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--protein_name", default=None, help="optional stable protein identifier for metadata")
    ap.add_argument("--forcefield", default="amber")
    ap.add_argument("--beam_width", type=int, default=1000)
    ap.add_argument("--window_deg", type=int, default=15)
    ap.add_argument("--step_deg", type=int, default=15)
    ap.add_argument("--omega_step_deg", type=int, default=10,
                    help="Step size for bounded omega sampling in the fixed 170-190 degree band.")
    ap.add_argument("--top_k", type=int, default=200, help="save top_k ranked conformers (post-search)")
    ap.add_argument("--reference_pdb", default=None, help="optional PDB path/id for RMSD comparison")
    ap.add_argument("--rmsd_mode", default="ca", choices=["ca", "heavy"],
                    help="RMSD atom selection: all CA atoms or all heavy atoms")
    ap.add_argument("--rmsd_residue_scope", default="core", choices=["core", "all"],
                    help="Residue range used for RMSD: core excludes the first and last residues")
    ap.add_argument("--average_reference_backbone", action="store_true")
    ap.add_argument("--native_thresh", type=float, default=2.0, help="RMSD threshold (Å) for native-like")
    ap.add_argument("--outdir", default="beamsearch_outputs")
    ap.add_argument("--save_partial", action="store_true", help="save beam survivors at each residue depth to CSV")
    ap.add_argument("--chi_mode", default="selective", choices=["chi1_only", "selective", "all"])
    ap.add_argument("--max_sidechain_opts_per_residue", type=int, default=9,
                    help="Max sampled local torsion choices per residue. 0 means exhaustive local enumeration and can explode for 125/5/all.")
    ap.add_argument("--energy_backend", default="custom", choices=["custom", "rosetta", "openmm"],
                    help="Stage-3 scorer: custom QTF energy, PyRosetta-backed score, or OpenMM-backed score.")
    ap.add_argument("--use_e2e_constraint", type=int, default=1,
                    help="1 to use length-scaled E2E constraint in custom scorer, 0 to disable.")
    ap.add_argument("--e2e_scale", type=float, default=1.0,
                    help="Multiplier for the length-scaled E2E constraint when enabled.")
    ap.add_argument("--rosetta_repack", type=int, default=0)
    ap.add_argument("--rosetta_fa_min", type=int, default=0)
    ap.add_argument("--rosetta_cen_min", type=int, default=0)
    ap.add_argument("--gromacs_minimize", type=int, default=None,
                    help="1 to add hydrogens/topology and minimize each saved full PDB with GROMACS")
    ap.add_argument("--gromacs_forcefield", default="amber99sb-ildn")
    ap.add_argument("--gromacs_water", default="tip3p")
    ap.add_argument("--gromacs_nsteps", type=int, default=5000,
                    help="maximum GROMACS minimization steps; minimization may stop earlier at --gromacs_emtol")
    ap.add_argument("--gromacs_emtol", type=float, default=100.0,
                    help="GROMACS steepest-descent force tolerance in kJ/mol/nm")
    ap.add_argument("--gromacs_maxwarn", type=int, default=2)
    ap.add_argument("--gromacs_rerank", type=int, default=None,
                    help="1 to rerank final minimized outputs by GROMACS potential energy when available")
    ap.add_argument("--hard_clash_reject_A", type=float, default=0.75,
                    help="Reject beam candidates whose QTF hard-clash minimum distance is below this Angstrom threshold. 0 disables.")
    ap.add_argument("--random_seed", type=int, default=123)
    args = ap.parse_args()
    if args.gromacs_minimize is None:
        args.gromacs_minimize = 1
    if args.gromacs_rerank is None:
        args.gromacs_rerank = 1

    # Keep command-line args as source of truth, but mirror them into env vars
    # because some lower-level helper code still uses env fallbacks.
    os.environ["QTF_STAGE3_BACKEND"] = str(args.energy_backend)
    os.environ["QTF_USE_E2E_CONSTRAINT"] = "1" if int(args.use_e2e_constraint) else "0"
    os.environ["QTF_E2E_SCALE"] = str(args.e2e_scale)
    os.environ["QTF_ROSETTA_REPACK"] = "1" if int(args.rosetta_repack) else "0"
    os.environ["QTF_ROSETTA_FA_MIN"] = "1" if int(args.rosetta_fa_min) else "0"
    os.environ["QTF_ROSETTA_CEN_MIN"] = "1" if int(args.rosetta_cen_min) else "0"

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    seq = args.sequence.strip().upper()
    L = len(seq)
    tuning = get_tuning_settings()
    protein_name = infer_protein_name(args.protein_name, args.reference_pdb, seq)
    reference_pdb_id = pdb_id_from_path(args.reference_pdb)
    reference_pdb_path = str(args.reference_pdb) if args.reference_pdb else None
    experiment_id = (
        f"{protein_name}_ff-{args.forcefield}_chi-{args.chi_mode}"
        f"_rmsd-{args.rmsd_mode}_scope-{args.rmsd_residue_scope}"
        f"_backend-{args.energy_backend}_e2e-{args.use_e2e_constraint}"
        f"_hb-{tuning['hbond_scale']}_sasa-{tuning['sasa_scale']}"
        f"_vdwr-{tuning['vdw_rep_scale']}_vdwa-{tuning['vdw_attr_scale']}"
    )

    # 6 basin centers (deg): alpha, beta, PPII, turn/loop, left-handed, canonical hairpin-ish
    basin_centers = [(-60, -45), (-135, 135), (-75, 145), (-60, 120), (60, 45), (-90, 90)]
    backbone_pairs = make_backbone_pairs(args.window_deg, args.step_deg, basin_centers)
    omega_values = deg(list(range(170, 191, max(1, int(args.omega_step_deg)))))
    if not omega_values or abs(np.rad2deg(omega_values[-1]) - 190.0) > 1e-6:
        omega_values.append(np.deg2rad(190.0))

    chi_rotamer_map = {
        "chi1": deg([-60, 60, 180]),
        "chi2": deg([-60, 60, 180]),
        "chi3": deg([-60, 60, 180]),
        "chi4": deg([-60, 60, 180]),
        "chi5": deg([-60, 60, 180]),
    }

    selective_chi_map = {
        "Y": ["chi1", "chi2"], "W": ["chi1", "chi2"], "F": ["chi1", "chi2"], "H": ["chi1", "chi2"],
        "D": ["chi1"], "E": ["chi1"], "N": ["chi1"], "Q": ["chi1"],
        "T": ["chi1"], "S": ["chi1"],
        "V": ["chi1"], "I": ["chi1"], "L": ["chi1"], "M": ["chi1"],
        "K": ["chi1"], "R": ["chi1"], "C": ["chi1"], "P": ["chi1"],
        "A": [], "G": [],
    }

    folders_by_k: Dict[int, runner.QuantumBiophysicsFolder] = {}
    for k in range(1, L + 1):
        folders_by_k[k] = utils.make_folder(
            sequence=seq[:k],
            force_field=args.forcefield,
            chi_mode=args.chi_mode,
            selective_chi_map=selective_chi_map,
            energy_backend=args.energy_backend,
            use_e2e_constraint=bool(args.use_e2e_constraint),
            e2e_scale=args.e2e_scale,
            rosetta_repack=bool(args.rosetta_repack),
            rosetta_fa_min=bool(args.rosetta_fa_min),
            rosetta_cen_min=bool(args.rosetta_cen_min),
        )
        folders_by_k[k].current_stage = 3

    full_folder = folders_by_k[L]

    dofs_by_res: Dict[int, List[Dict[str, Any]]] = {}
    for d in full_folder.dof_map:
        r = int(d["res"])
        dofs_by_res.setdefault(r, []).append(d)

    residue_opts: Dict[int, List[np.ndarray]] = {}
    for r in range(L):
        residue_opts[r] = residue_assignments(
            dofs_by_res.get(r, []),
            backbone_pairs,
            chi_rotamer_map,
            omega_values=omega_values,
            max_options_per_residue=args.max_sidechain_opts_per_residue,
            rng=rng,
        )

    dof_count_by_res = [len(dofs_by_res.get(r, [])) for r in range(L)]
    prefix_counts = np.cumsum(dof_count_by_res)

    beam: List[State] = [State(angles=np.zeros(0, dtype=float), depth=0, energy=0.0, terms={}, parent_id=None, local_choice=None)]
    partial_rows = []

    def stable_id(obj) -> int:
        raw = json.dumps(obj, sort_keys=True, default=lambda z: z.tolist() if isinstance(z, np.ndarray) else z).encode("utf-8")
        return int(hashlib.sha1(raw).hexdigest()[:12], 16)

    for depth in range(1, L + 1):
        folder_k = folders_by_k[depth]
        local_opts = residue_opts[depth - 1]
        n_prefix = int(prefix_counts[depth - 1])

        new_states: List[State] = []

        for parent in beam:
            for opt_idx, local_vec in enumerate(local_opts):
                candidate = np.zeros(n_prefix, dtype=float)
                if parent.angles.size > 0:
                    candidate[:parent.angles.size] = parent.angles
                candidate[parent.angles.size:n_prefix] = local_vec

                E, terms = eval_energy_terms(folder_k, candidate)
                hard_min = float(terms.get("hard_clash_min_dist", 0.0) or 0.0)
                if float(args.hard_clash_reject_A) > 0.0 and hard_min > 0.0 and hard_min < float(args.hard_clash_reject_A):
                    continue
                st = State(
                    angles=candidate,
                    depth=depth,
                    energy=float(E),
                    terms=terms,
                    parent_id=stable_id({
                        "depth": parent.depth,
                        "angles": np.round(parent.angles, 6).tolist(),
                        "energy": round(parent.energy, 6),
                    }),
                    local_choice=f"res{depth-1}_opt{opt_idx}",
                )
                new_states.append(st)

        if not new_states:
            raise RuntimeError(
                f"All beam candidates were rejected by hard-clash filter at depth {depth}; "
                f"try lowering --hard_clash_reject_A or increasing the beam/search space."
            )

        new_states = dedup_states_by_backbone(folder_k, new_states, round_deg=float(args.step_deg))
        new_states.sort(key=lambda x: x.energy)
        beam = new_states[:args.beam_width]

        if args.save_partial:
            for rank_i, st in enumerate(beam[: min(50, len(beam))], start=1):
                row = {
                    "protein_name": protein_name,
                    "reference_pdb_id": reference_pdb_id,
                    "reference_pdb_path": reference_pdb_path,
                    "experiment_id": experiment_id,
                    "sequence": seq,
                    "forcefield": args.forcefield,
                    "chi_mode": args.chi_mode,
                    "depth": depth,
                    "rmsd_mode": args.rmsd_mode,
                    "rmsd_residue_scope": args.rmsd_residue_scope,
                    "rank_at_depth": rank_i,
                    "energy": float(st.energy),
                    "parent_id": st.parent_id,
                    "local_choice": st.local_choice,
                    "angles_deg_json": json.dumps([float(np.rad2deg(a)) for a in st.angles]),
                    "hbond_scale": tuning["hbond_scale"],
                    "sasa_scale": tuning["sasa_scale"],
                    "vdw_rep_scale": tuning["vdw_rep_scale"],
                    "vdw_attr_scale": tuning["vdw_attr_scale"],
                    "rotamer_scale": tuning["rotamer_scale"],
                    "pi_stack_scale": tuning["pi_stack_scale"],
                }
                for k, v in st.terms.items():
                    row[f"term_{k}"] = float(v)
                partial_rows.append(row)

        print(f"[beam] depth={depth:2d} expanded={len(new_states):6d} kept={len(beam):4d} bestE={beam[0].energy:10.4f}")

    rows = []
    ref_coords = None
    ref_labels = None
    ref_meta = {}
    ref_e2e = None
    ref_rg = None

    if args.reference_pdb:
        try:
            ref_coords, ref_labels, ref_meta = utils.load_reference_rmsd_coords(
                args.reference_pdb,
                args.rmsd_mode,
                average_backbone=bool(args.average_reference_backbone),
            )
            if ref_coords is not None and len(ref_coords) >= 2 and args.rmsd_mode == "ca":
                ref_e2e = float(np.linalg.norm(ref_coords[0] - ref_coords[-1]))
                ref_rg = float(np.sqrt(np.mean(np.sum((ref_coords - np.mean(ref_coords, axis=0))**2, axis=1))))
            print(f"[beam] loaded reference {args.rmsd_mode} coords from {args.reference_pdb}: shape={ref_coords.shape}")
        except Exception as e:
            print(f"[warn] failed to load reference backbone for {args.reference_pdb}: {e}")
            ref_coords = None
            ref_meta = {}

    pdb_dir = os.path.join(args.outdir, "raw_pdbs")
    gromacs_pdb_dir = os.path.join(args.outdir, "gromacs_pdbs")
    os.makedirs(pdb_dir, exist_ok=True)
    os.makedirs(gromacs_pdb_dir, exist_ok=True)

    for i, st in enumerate(beam, start=1):
        angle_full = st.angles
        E, terms = eval_energy_terms(full_folder, angle_full)
        coords_full, labels_full, _ = getattr(full_folder, "build_output_structure", full_folder.build_full_structure)(angle_full)
        ca = np.array([coords_full[j] for j, lbl in enumerate(labels_full) if lbl[1] == "CA"])
        sidechain_centroids = full_folder.compute_sidechain_centroids(coords_full, labels_full)
        clash_metrics = nonlocal_heavy_clash_metrics(coords_full, labels_full)
        local_clash_metrics = adjacent_heavy_clash_metrics(coords_full, labels_full)
        ring_penetration_metrics = qtf_gromacs.ring_penetration_metrics(coords_full, labels_full)

        ca_pdb_path = os.path.join(pdb_dir, f"rank_{i:04d}_ca.pdb")
        ca_centroid_pdb_path = os.path.join(pdb_dir, f"rank_{i:04d}_ca_centroid.pdb")
        full_pdb_path = os.path.join(pdb_dir, f"rank_{i:04d}_full.pdb")
        full_folder.save_reduced_pdb(ca, filename=ca_pdb_path, sidechain_centroids=None, energy=E)
        full_folder.save_reduced_pdb(ca, filename=ca_centroid_pdb_path, sidechain_centroids=sidechain_centroids, energy=E)
        full_folder.save_pdb(
            coords_full,
            labels_full,
            filename=full_pdb_path,
            energy=E,
            chain_id="A",
            remarks=["QTF heavy-atom rebuilt structure from beam-search torsions"],
            include_hydrogens=False,
        )

        gromacs_result = utils.gromacs_postprocess_structure(
            enabled=bool(args.gromacs_minimize),
            full_pdb_path=full_pdb_path,
            gromacs_dir=os.path.join(gromacs_pdb_dir, f"rank_{i:04d}"),
            forcefield=args.gromacs_forcefield,
            water=args.gromacs_water,
            nsteps=args.gromacs_nsteps,
            emtol=args.gromacs_emtol,
            maxwarn=args.gromacs_maxwarn,
            coords=coords_full,
            labels=labels_full,
            ca_coords=ca,
            sidechain_centroid_fn=full_folder.compute_sidechain_centroids,
            nonlocal_clash_fn=nonlocal_heavy_clash_metrics,
            local_clash_fn=adjacent_heavy_clash_metrics,
        )
        coords_full = gromacs_result["coords"]
        labels_full = gromacs_result["labels"]
        ca = gromacs_result["ca_coords"]
        sidechain_centroids = gromacs_result["sidechain_centroids"]
        clash_metrics = gromacs_result["nonlocal_clash_metrics"]
        local_clash_metrics = gromacs_result["local_clash_metrics"]
        ring_penetration_metrics = gromacs_result["ring_penetration_metrics"]
        gromacs_info = gromacs_result["gromacs_info"]

        row = {
            "protein_name": protein_name,
            "reference_pdb_id": reference_pdb_id,
            "reference_pdb_path": reference_pdb_path,
            "experiment_id": experiment_id,
            "sequence": seq,
            "forcefield": args.forcefield,
            "chi_mode": args.chi_mode,
            "rmsd_mode": args.rmsd_mode,
            "hbond_scale": tuning["hbond_scale"],
            "sasa_scale": tuning["sasa_scale"],
            "vdw_rep_scale": tuning["vdw_rep_scale"],
            "vdw_attr_scale": tuning["vdw_attr_scale"],
            "rotamer_scale": tuning["rotamer_scale"],
            "pi_stack_scale": tuning["pi_stack_scale"],
            "energy_rank": i,
            "energy": float(E),
            "depth": int(st.depth),
            "angles_deg_json": json.dumps([float(np.rad2deg(a)) for a in angle_full]),
            "rebuilt_ca_pdb_path": ca_pdb_path,
            "rebuilt_ca_centroid_pdb_path": ca_centroid_pdb_path,
            "rebuilt_full_pdb_path": full_pdb_path,
            **gromacs_info,
            **clash_metrics,
            **local_clash_metrics,
            **ring_penetration_metrics,
        }
        for k, v in terms.items():
            row[f"term_{k}"] = float(v)

        if ca is not None and len(ca) >= 2:
            row["pred_e2e_A"] = float(np.linalg.norm(ca[0] - ca[-1]))
            row["pred_rg_A"] = float(np.sqrt(np.mean(np.sum((ca - np.mean(ca, axis=0))**2, axis=1))))

        if ref_coords is not None:
            try:
                rmsd, rmsd_meta = utils.rmsd_between_structures(
                    coords_full,
                    labels_full,
                    ref_coords,
                    ref_labels,
                    args.rmsd_mode,
                    args.rmsd_residue_scope,
                )
                row["rmsd_to_reference_A"] = rmsd
                row.update(rmsd_meta)
                row["ref_e2e_A"] = ref_e2e
                row["ref_rg_A"] = ref_rg
                row["native_like"] = bool(float(rmsd) <= float(args.native_thresh))
                row["rmsd_error"] = ""
            except Exception as e:
                row["rmsd_to_reference_A"] = np.nan
                row["native_like"] = False
                row["rmsd_error"] = str(e)
                row.update(utils.rmsd_selection_metadata(args.rmsd_mode, args.rmsd_residue_scope, n_atoms=0, n_residues=0))

        rows.append(row)

    df = pd.DataFrame(rows)
    if "clash_flag" in df.columns:
        clean_df = df.loc[~df["clash_flag"]].copy()
        if len(clean_df) > 0:
            df = clean_df
        else:
            print("[beam][warn] all ranked candidates triggered the clash filter; keeping unfiltered set")
    if "local_clash_flag" in df.columns:
        clean_df = df.loc[~df["local_clash_flag"]].copy()
        if len(clean_df) > 0:
            df = clean_df
        else:
            print("[beam][warn] all ranked candidates triggered the local backbone clash filter; keeping prior set")
    if "ring_penetration_flag" in df.columns:
        clean_df = df.loc[~df["ring_penetration_flag"]].copy()
        if len(clean_df) > 0:
            df = clean_df
        else:
            print("[beam][warn] all ranked candidates triggered the ring penetration filter; keeping prior set")
    if bool(args.gromacs_minimize) and bool(args.gromacs_rerank) and "gromacs_potential_kj_mol" in df.columns:
        df = df.sort_values(["gromacs_potential_kj_mol", "energy"], na_position="last").reset_index(drop=True)
        df["gromacs_energy_rank"] = np.arange(1, len(df) + 1)
    else:
        df = df.sort_values("energy").reset_index(drop=True)

    df_out = df.head(args.top_k).copy()

    csv_path = os.path.join(args.outdir, "beamsearch_ranked.csv")
    json_path = os.path.join(args.outdir, "beamsearch_ranked.json")
    df_out.to_csv(csv_path, index=False)

    payload = [{k: _jsonify(v) for k, v in rec.items()} for rec in df_out.to_dict(orient="records")]
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    if args.save_partial and partial_rows:
        pd.DataFrame(partial_rows).to_csv(os.path.join(args.outdir, "beamsearch_partial_states.csv"), index=False)

    best = df.iloc[0].to_dict()
    best_summary = {
        "protein_name": protein_name,
        "reference_pdb_id": reference_pdb_id,
        "reference_pdb_path": reference_pdb_path,
        "experiment_id": experiment_id,
        "sequence": seq,
            "forcefield": args.forcefield,
            "beam_width": args.beam_width,
            "window_deg": args.window_deg,
            "step_deg": args.step_deg,
            "basin_centers_deg": basin_centers,
            "chi_mode": args.chi_mode,
            "rmsd_mode": args.rmsd_mode,
            "rmsd_residue_scope": args.rmsd_residue_scope,
            "chi_rotamers_deg": [-60, 60, 180],
                "native_thresh_A": args.native_thresh,
                "energy_backend": args.energy_backend,
        "rmsd_mode": args.rmsd_mode,
        "use_e2e_constraint": bool(args.use_e2e_constraint),
        "e2e_scale": args.e2e_scale,
        "tuning": tuning,
        "best": {k: _jsonify(v) for k, v in best.items()},
    }
    with open(os.path.join(args.outdir, "beamsearch_best.json"), "w") as f:
        json.dump(best_summary, f, indent=2)

    if args.reference_pdb:
        native_count = int(df_out["native_like"].sum()) if "native_like" in df_out.columns else 0
        with open(os.path.join(args.outdir, "native_like_counts.json"), "w") as f:
            json.dump({
                "native_like": native_count,
                "non_native": int(len(df_out) - native_count),
                "native_thresh_A": args.native_thresh
            }, f, indent=2)

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


if __name__ == "__main__":
    main()
