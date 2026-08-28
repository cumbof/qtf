"""Shared utility helpers for QTF entrypoints.

This module centralizes the duplicated runner, beam, native-score, and GROMACS
plumbing so the root scripts can stay thin wrappers for now.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import mdtraj as md
except ImportError:
    md = None  # type: ignore[assignment]
import numpy as np
import pandas as pd
from scipy.stats import circmean
try:
    from Bio.PDB import PDBIO, PDBParser, Select
except ImportError:
    PDBIO = PDBParser = Select = None  # type: ignore[assignment]

from qtf.analysis.stability import kabsch_rmsd
from qtf.core.folder import QuantumBiophysicsFolder
from qtf.utils.pdb import calculate_physics_metrics
from qtf.utils import gromacs as qtf_gromacs


Coords = np.ndarray
Labels = List[Tuple[int, str, str]]


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


_PDB_ID_PATTERN = re.compile(r"^[1-9][A-Z0-9]{3}$")


def pdb_id_from_path(p: Optional[str]) -> str:
    if p is None:
        return ""
    stem = Path(str(p)).stem.upper()
    m = _PDB_ID_PATTERN.match(stem)
    return m.group() if m else ""



def adjacent_heavy_clash_metrics(
    coords: Coords,
    labels: Labels,
    min_allowed_A: float = 1.35,
    threshold_frac: float = 0.55,
) -> Dict[str, object]:
    """Detect the closest heavy-atom contact between adjacent residues."""
    coords = np.asarray(coords, dtype=float)
    res_ids = np.array([int(r) for r, _, _ in labels], dtype=int)
    atom_names = [str(atom) for _, atom, _ in labels]
    heavy_mask = np.array(
        [str(elem).upper() != "H" and not str(atom).upper().startswith("H") for _, atom, elem in labels],
        dtype=bool,
    )

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


def nonlocal_heavy_clash_metrics(coords: Coords, labels: Labels, min_allowed_A: float = 1.75) -> Dict[str, object]:
    """Detect the closest heavy-atom contact between residues separated by at least 2."""
    coords = np.asarray(coords, dtype=float)
    res_ids = np.array([int(r) for r, _, _ in labels], dtype=int)
    heavy_mask = np.array(
        [str(elem).upper() != "H" and not str(atom).upper().startswith("H") for _, atom, elem in labels],
        dtype=bool,
    )

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


if Select is not None:

    class _SelectChainResidues(Select):
        def __init__(self, chain_id: Optional[str], start: Optional[int], end: Optional[int]):
            self.chain_id = chain_id
            self.start = start
            self.end = end

        def accept_model(self, model):
            return 1 if model.id == 0 else 0

        def accept_chain(self, chain):
            if self.chain_id is None:
                return 1
            return 1 if chain.id == self.chain_id else 0

        def accept_residue(self, residue):
            if residue.id[0] != " ":
                return 0
            resseq = residue.id[1]
            if self.start is not None and resseq < self.start:
                return 0
            if self.end is not None and resseq > self.end:
                return 0
            return 1


def extract_subset_pdb(
    src_pdb: str,
    chain_id: Optional[str],
    start: Optional[int],
    end: Optional[int],
) -> Tuple[str, List[int], str, str]:
    src_path = Path(src_pdb)
    if not src_path.is_file():
        candidate = Path("experimental_structures/pdb_files") / src_path.name
        if candidate.is_file():
            src_path = candidate
    src_pdb = str(src_path)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("native", src_pdb)
    model = next(structure.get_models())

    chains = [c for c in model]
    if chain_id is None:
        protein_chains = [c for c in chains if any(r.id[0] == " " for r in c)]
        if len(protein_chains) != 1:
            raise ValueError(f"PDB has multiple protein chains {[c.id for c in protein_chains]}; pass --chain")
        chain_id = protein_chains[0].id

    if chain_id and chain_id in model:
        chain = model[chain_id]
    else:
        chains = list(model.get_chains())
        if not chains:
            raise ValueError("No chains found in PDB")
        chain = model["A"] if "A" in model else chains[0]
        chain_id = chain.id

    seq_chars = []
    pdb_resseqs = []
    for res in chain:
        if res.id[0] != " ":
            continue
        resseq = res.id[1]
        if start is not None and resseq < start:
            continue
        if end is not None and resseq > end:
            continue
        if res.resname not in AA3_TO_1:
            raise ValueError(f"Unsupported residue {res.resname} at {chain_id}:{resseq}")
        pdb_resseqs.append(resseq)
        seq_chars.append(AA3_TO_1[res.resname])

    if not seq_chars:
        raise ValueError("No standard protein residues selected")

    out = tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False)
    out.close()

    io = PDBIO()
    io.set_structure(structure)
    io.save(out.name, _SelectChainResidues(chain_id, start, end))
    return out.name, pdb_resseqs, "".join(seq_chars), chain_id


def compute_qtf_angle_vector(
    trimmed_pdb: str,
    folder: QuantumBiophysicsFolder,
    chi_mode: str = "all",
) -> Tuple[np.ndarray, Dict[str, float]]:
    traj = md.load(trimmed_pdb)
    topo = traj.topology

    angle_map = {(res.index, "phi"): 0.0 for res in topo.residues}
    angle_map.update({(res.index, "psi"): 0.0 for res in topo.residues})

    torsion_fns = [
        ("phi", md.compute_phi),
        ("psi", md.compute_psi),
        ("chi1", md.compute_chi1),
        ("chi2", md.compute_chi2),
        ("chi3", md.compute_chi3),
        ("chi4", md.compute_chi4),
        ("chi5", md.compute_chi5),
    ]

    observed = {}
    for name, fn in torsion_fns:
        atom_quads, values = fn(traj)
        if values.size == 0:
            continue
        if values.ndim == 2 and values.shape[0] > 1:
            vals = circmean(np.mod(values, 2 * np.pi), axis=0)
        else:
            vals = values[0]
        for quad, val in zip(atom_quads, vals):
            res_idx = topo.atom(int(quad[1])).residue.index
            angle_map[(res_idx, name)] = float(val)
            observed[(res_idx, name)] = float(val)

    omega_quads, omega_values = md.compute_omega(traj)
    fixed_omegas = np.full(max(0, folder.n_residues - 1), np.pi, dtype=float)
    if omega_values.size:
        if omega_values.ndim == 2 and omega_values.shape[0] > 1:
            omega_vals = circmean(np.mod(omega_values, 2 * np.pi), axis=0)
        else:
            omega_vals = omega_values[0]
        for quad, val in zip(omega_quads, omega_vals):
            res_idx = topo.atom(int(quad[1])).residue.index
            if 0 <= res_idx < len(fixed_omegas):
                fixed_omegas[res_idx] = float(val)
                angle_map[(res_idx, "omega")] = float(val)
                observed[(res_idx, "omega")] = float(val)
    folder.fixed_omegas = fixed_omegas

    if chi_mode == "beam":
        for (res_idx, torsion_name) in list(angle_map.keys()):
            if torsion_name.startswith("chi") and torsion_name != "chi1":
                angle_map[(res_idx, torsion_name)] = 0.0
    elif chi_mode == "selective":
        pass
    elif chi_mode != "all":
        raise ValueError("chi_mode must be 'beam', 'selective', or 'all'")

    angle_vec = np.zeros(folder.total_angles, dtype=float)
    for i, dof in enumerate(folder.dof_map):
        res = int(dof["res"])
        typ = str(dof["type"]).replace("_branch", "")
        angle_vec[i] = angle_map.get((res, typ), 0.0)

    return angle_vec, observed


def eval_energy_terms(folder: QuantumBiophysicsFolder, angle_vec: np.ndarray):
    dummy_params = np.zeros(folder.n_params, dtype=float)
    folder.current_stage = 3
    total = float(folder.energy_function(dummy_params, return_terms=True, angle_override=angle_vec))
    terms = {str(k): float(v) for k, v in (getattr(folder, "last_energy_terms", {}) or {}).items()}
    return total, terms


def build_full_coords(folder: QuantumBiophysicsFolder, angle_vec: np.ndarray):
    coords, labels, bonds = folder.build_full_structure(angle_vec)
    return coords, labels, bonds


def make_rebuilt_output_paths(output_dir: Path, spec_name: str, start: Optional[int], end: Optional[int]) -> Tuple[Path, Path, Path]:
    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in spec_name)
    if start is not None or end is not None:
        start_str = start if start is not None else "NA"
        end_str = end if end is not None else "NA"
        stem = f"{safe_name}_{start_str}_{end_str}_rebuilt"
    else:
        stem = f"{safe_name}_rebuilt"
    return (
        output_dir / f"{stem}_ca.pdb",
        output_dir / f"{stem}_ca_centroid.pdb",
        output_dir / f"{stem}_full.pdb",
    )



def core_ca_slice(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords)
    if arr.shape[0] > 2:
        return arr[1:-1]
    return arr


def core_ca_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    rmsd, _ = kabsch_rmsd(core_ca_slice(P), core_ca_slice(Q))
    return float(rmsd)


def rmsd_selection_metadata(
    atom_mode: str,
    residue_scope: str,
    *,
    n_atoms: int,
    n_residues: int,
    n_matched: int = 0,
    n_missing: int = 0,
) -> Dict[str, object]:
    if atom_mode == "ca":
        atom_selection = "name CA"
    elif atom_mode == "heavy":
        atom_selection = "not element H"
    else:
        raise ValueError("rmsd atom mode must be 'ca' or 'heavy'")

    if residue_scope not in {"core", "all"}:
        raise ValueError("rmsd residue scope must be 'core' or 'all'")

    use_core = residue_scope == "core" and n_residues > 2
    return {
        "rmsd_mode": atom_mode,
        "rmsd_residue_scope": residue_scope,
        "rmsd_atom_selection": atom_selection,
        "rmsd_excludes_terminal_residues": bool(use_core),
        "rmsd_start_residue_1indexed": 2 if use_core else 1,
        "rmsd_end_residue_1indexed": (n_residues - 1) if use_core else n_residues,
        "rmsd_n_selected_residues": (n_residues - 2) if use_core else n_residues,
        "rmsd_n_selected_atoms": int(n_atoms),
        "rmsd_n_aligned": int(n_matched),
        "rmsd_n_matched": int(n_matched),
        "rmsd_n_missing": int(n_missing),
    }


def _reference_pdb_path(reference: str) -> Tuple[Path, bool]:
    ref = str(reference).strip()
    path = Path(ref)
    if path.is_file():
        return path, False
    if len(ref) == 4 and ref.isalnum():
        tmp = tempfile.NamedTemporaryFile("wb", suffix=".pdb", delete=False)
        url = f"https://files.rcsb.org/download/{ref.upper()}.pdb"
        req = urllib.request.Request(url, headers={"User-Agent": "QTF/0.4.2"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            tmp.write(resp.read())
        tmp.close()
        return Path(tmp.name), True
    raise FileNotFoundError(f"reference structure not found: {reference}")


def load_reference_rmsd_coords(
    reference: str,
    rmsd_mode: str,
    average_backbone: bool = False,
) -> Tuple[np.ndarray, List[Tuple[int, str, str]], Dict[str, object]]:
    ref_path, is_temp = _reference_pdb_path(reference)
    traj = md.load(str(ref_path))
    try:
        if rmsd_mode == "ca":
            atom_sel = traj.topology.select("name CA")
        elif rmsd_mode == "heavy":
            atom_sel = traj.topology.select("not element H")
        else:
            raise ValueError("rmsd mode must be 'ca' or 'heavy'")
        if atom_sel is None or len(atom_sel) == 0:
            raise ValueError(f"No atoms found for rmsd_mode={rmsd_mode} in {reference}")
        coords = traj.xyz[:, atom_sel, :] * 10.0
        labels: List[Tuple[int, str, str]] = []
        for atom_idx in atom_sel:
            atom = traj.topology.atom(int(atom_idx))
            element = atom.element.symbol if atom.element is not None else ""
            labels.append((int(atom.residue.index), str(atom.name), str(element)))
        if average_backbone:
            coords = np.mean(coords, axis=0, keepdims=False)
        else:
            coords = coords[0]
        meta = {
            "reference_pdb_path": str(ref_path),
            "reference_pdb_id": pdb_id_from_path(reference),
            "reference_source": "downloaded" if is_temp else "local",
            "reference_rmsd_mode": rmsd_mode,
            "reference_rmsd_n_atoms": int(len(atom_sel)),
        }
        return np.asarray(coords, dtype=float), labels, meta
    finally:
        if is_temp:
            try:
                os.unlink(ref_path)
            except OSError:
                pass


def select_rmsd_coords(
    coords: np.ndarray,
    labels: List[Tuple[int, str, str]],
    rmsd_mode: str,
    rmsd_residue_scope: str,
) -> Tuple[np.ndarray, List[Tuple[int, str]], List[int]]:
    arr = np.asarray(coords, dtype=float)
    if rmsd_mode == "ca":
        selected = np.array([i for i, lbl in enumerate(labels) if lbl[1] == "CA"], dtype=int)
    elif rmsd_mode == "heavy":
        selected = np.array([
            i for i, (_, atom, elem) in enumerate(labels)
            if str(elem).upper() != "H" and not str(atom).upper().startswith("H")
        ], dtype=int)
    else:
        raise ValueError("rmsd atom mode must be 'ca' or 'heavy'")

    if selected.size == 0:
        return arr[selected], [], []

    residue_ids = [int(labels[i][0]) for i in selected]
    unique_residues = sorted(set(residue_ids))
    if rmsd_residue_scope == "core" and len(unique_residues) > 2:
        allowed_residues = set(unique_residues[1:-1])
    elif rmsd_residue_scope == "all":
        allowed_residues = set(unique_residues)
    else:
        allowed_residues = set(unique_residues)

    filtered_idx = [i for i in selected if int(labels[i][0]) in allowed_residues]
    keys = [(int(labels[i][0]), str(labels[i][1])) for i in filtered_idx]
    filtered_residues = sorted(set(int(labels[i][0]) for i in filtered_idx))
    return arr[filtered_idx], keys, filtered_residues


def rmsd_between_structures(
    model_coords: np.ndarray,
    model_labels: List[Tuple[int, str, str]],
    reference_coords: np.ndarray,
    reference_labels: List[Tuple[int, str, str]],
    rmsd_mode: str,
    rmsd_residue_scope: str,
) -> Tuple[float, Dict[str, object]]:
    model_sel, model_keys, model_residues = select_rmsd_coords(model_coords, model_labels, rmsd_mode, rmsd_residue_scope)
    ref_sel, ref_keys, ref_residues = select_rmsd_coords(reference_coords, reference_labels, rmsd_mode, rmsd_residue_scope)
    ref_map = {key: coord for key, coord in zip(ref_keys, ref_sel)}
    model_common = []
    ref_common = []
    missing = []
    for key, coord in zip(model_keys, model_sel):
        ref_coord = ref_map.get(key)
        if ref_coord is None:
            missing.append(key)
            continue
        model_common.append(coord)
        ref_common.append(ref_coord)
    if not model_common:
        raise ValueError(f"No common atoms found for rmsd_mode={rmsd_mode}")
    if missing:
        # Keep the comparison robust by aligning the shared atom set only.
        # This matters for terminal atoms like OXT that can exist in the
        # reference but not in the rebuilt QTF structure.
        pass
    model_common_arr = np.asarray(model_common, dtype=float)
    ref_common_arr = np.asarray(ref_common, dtype=float)
    if model_common_arr.shape != ref_common_arr.shape:
        raise ValueError(f"Shape mismatch for rmsd_mode={rmsd_mode}: {model_common_arr.shape} vs {ref_common_arr.shape}")
    n_residues = len(model_residues) if model_residues else len(ref_residues)
    meta = rmsd_selection_metadata(
        rmsd_mode,
        rmsd_residue_scope,
        n_atoms=len(model_sel),
        n_residues=n_residues,
        n_matched=len(model_common_arr),
        n_missing=len(missing),
    )
    rmsd, _ = kabsch_rmsd(model_common_arr, ref_common_arr)
    return float(rmsd), meta


def align_structure_to_reference(
    model_coords: np.ndarray,
    model_labels: List[Tuple[int, str, str]],
    reference_coords: np.ndarray,
    reference_labels: List[Tuple[int, str, str]],
    rmsd_mode: str,
    rmsd_residue_scope: str,
) -> Tuple[np.ndarray, float, Dict[str, object], Dict[str, np.ndarray]]:
    """Rigidly align a complete model using its matched RMSD atom subset."""
    model_sel, model_keys, model_residues = select_rmsd_coords(
        model_coords, model_labels, rmsd_mode, rmsd_residue_scope
    )
    ref_sel, ref_keys, ref_residues = select_rmsd_coords(
        reference_coords, reference_labels, rmsd_mode, rmsd_residue_scope
    )
    ref_map = {key: coord for key, coord in zip(ref_keys, ref_sel)}
    matched_model = []
    matched_reference = []
    missing = []
    for key, coord in zip(model_keys, model_sel):
        reference_coord = ref_map.get(key)
        if reference_coord is None:
            missing.append(key)
            continue
        matched_model.append(coord)
        matched_reference.append(reference_coord)
    if not matched_model:
        raise ValueError(f"No common atoms found for rmsd_mode={rmsd_mode}")

    moving = np.asarray(matched_model, dtype=float)
    target = np.asarray(matched_reference, dtype=float)
    moving_centroid = moving.mean(axis=0)
    target_centroid = target.mean(axis=0)
    covariance = (moving - moving_centroid).T @ (target - target_centroid)
    left, _singular_values, right_transpose = np.linalg.svd(covariance)
    if np.linalg.det(left) * np.linalg.det(right_transpose) < 0.0:
        left[:, -1] *= -1.0
    rotation = left @ right_transpose
    translation = target_centroid - moving_centroid @ rotation
    aligned = np.asarray(model_coords, dtype=float) @ rotation + translation
    aligned_matched = moving @ rotation + translation
    rmsd = float(np.sqrt(np.mean(np.sum((aligned_matched - target) ** 2, axis=1))))

    n_residues = len(model_residues) if model_residues else len(ref_residues)
    metadata = rmsd_selection_metadata(
        rmsd_mode,
        rmsd_residue_scope,
        n_atoms=len(model_sel),
        n_residues=n_residues,
        n_matched=len(moving),
        n_missing=len(missing),
    )
    transform = {"rotation": rotation, "translation": translation}
    return aligned, rmsd, metadata, transform


def calculate_metrics(ca_coords: np.ndarray) -> Dict[str, float]:
    end_to_end = float(np.linalg.norm(ca_coords[0] - ca_coords[-1]))
    centroid = np.mean(ca_coords, axis=0)
    rg = float(np.sqrt(np.mean(np.sum((ca_coords - centroid) ** 2, axis=1))))
    return {"end_to_end": end_to_end, "radius_of_gyration": rg}


def load_panel(panel_path: str) -> List[Dict]:
    path = Path(panel_path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    raise ValueError("panel must be .json or .csv")


def _compute_native_metrics(
    folder: QuantumBiophysicsFolder,
    angle_vec: np.ndarray,
    observed: Dict,
    trimmed_pdb: str,
    pdb_resseqs: List[int],
    selected_chain_id: str,
    pdb_path: str,
    name: str,
    chi_mode: str,
    rmsd_mode: str,
    rmsd_residue_scope: str,
    energy_backend: str,
    use_e2e_constraint: bool,
    e2e_scale: float,
    gromacs_minimize: bool,
    gromacs_forcefield: str,
    gromacs_water: str,
    gromacs_nsteps: int,
    gromacs_emtol: float,
    gromacs_maxwarn: int,
    rebuilt_ca_pdb_path: Optional[str],
    rebuilt_ca_centroid_pdb_path: Optional[str],
    rebuilt_full_pdb_path: Optional[str],
    start: Optional[int],
    end: Optional[int],
) -> Dict[str, Any]:
    total_energy, terms = eval_energy_terms(folder, angle_vec)
    rebuilt_coords, rebuilt_labels, _ = build_full_coords(folder, angle_vec)
    rebuilt_ca = np.array([rebuilt_coords[i] for i, lbl in enumerate(rebuilt_labels) if lbl[1] == "CA"])
    rebuilt_metrics = calculate_metrics(rebuilt_ca)
    sidechain_centroids = folder.compute_sidechain_centroids(rebuilt_coords, rebuilt_labels)

    if rebuilt_ca_pdb_path is not None:
        folder.save_reduced_pdb(
            rebuilt_ca,
            filename=str(rebuilt_ca_pdb_path),
            sidechain_centroids=None,
            energy=total_energy,
            chain_id=selected_chain_id,
            resseqs=pdb_resseqs,
        )

    if rebuilt_ca_centroid_pdb_path is not None:
        folder.save_reduced_pdb(
            rebuilt_ca,
            filename=str(rebuilt_ca_centroid_pdb_path),
            sidechain_centroids=sidechain_centroids,
            energy=total_energy,
            chain_id=selected_chain_id,
            resseqs=pdb_resseqs,
        )

    if rebuilt_full_pdb_path is not None:
        folder.save_pdb(
            rebuilt_coords,
            rebuilt_labels,
            filename=str(rebuilt_full_pdb_path),
            energy=total_energy,
            chain_id=selected_chain_id,
            resseqs=pdb_resseqs,
            remarks=["QTF heavy-atom rebuilt structure from native torsions"],
            include_hydrogens=False,
        )

    gromacs_info: Dict[str, Any] = {}
    if gromacs_minimize and rebuilt_full_pdb_path is not None:
        rebuilt_full_path = Path(str(rebuilt_full_pdb_path))
        gromacs_result = gromacs_postprocess_structure(
            enabled=True,
            full_pdb_path=str(rebuilt_full_pdb_path),
            gromacs_dir=str(rebuilt_full_path.parent.parent / "gromacs_pdbs" / rebuilt_full_path.stem),
            forcefield=gromacs_forcefield,
            water=gromacs_water,
            nsteps=gromacs_nsteps,
            emtol=gromacs_emtol,
            maxwarn=gromacs_maxwarn,
            coords=rebuilt_coords,
            labels=rebuilt_labels,
            ca_coords=rebuilt_ca,
            sidechain_centroid_fn=folder.compute_sidechain_centroids,
            nonlocal_clash_fn=nonlocal_heavy_clash_metrics,
            local_clash_fn=adjacent_heavy_clash_metrics,
        )
        rebuilt_coords = gromacs_result["coords"]
        rebuilt_labels = gromacs_result["labels"]
        rebuilt_ca = gromacs_result["ca_coords"]
        sidechain_centroids = gromacs_result["sidechain_centroids"]
        gromacs_info = gromacs_result["gromacs_info"]

    native_traj = md.load(trimmed_pdb)
    native_ca_idx = native_traj.topology.select("name CA")
    native_ca = native_traj.xyz[0, native_ca_idx, :] * 10.0
    native_metrics = calculate_metrics(native_ca)

    rebuilt_metrics = calculate_metrics(rebuilt_ca)
    rebuilt_vs_native_ca_rmsd = core_ca_rmsd(rebuilt_ca, native_ca)
    native_rmsd_coords, native_rmsd_labels, rmsd_meta = load_reference_rmsd_coords(trimmed_pdb, rmsd_mode, average_backbone=False)
    rebuilt_vs_native_rmsd_A, rmsd_meta = rmsd_between_structures(
        rebuilt_coords,
        rebuilt_labels,
        native_rmsd_coords,
        native_rmsd_labels,
        rmsd_mode,
        rmsd_residue_scope,
    )
    rmsd_meta = {
        **rmsd_meta,
        "rebuilt_vs_native_rmsd_A": float(rebuilt_vs_native_rmsd_A),
    }

    tuning = {
        "hbond_scale": 0.75,
        "sasa_scale": 0.7,
        "vdw_rep_scale": 0.01,
        "vdw_attr_scale": 0.1,
        "rotamer_scale": 1.0,
        "pi_stack_scale": 1.0,
    }
    protein_name = name
    reference_pdb_id = pdb_id_from_path(pdb_path)
    reference_pdb_path = str(pdb_path)
    experiment_id = (
        f"{protein_name}_chi-{chi_mode}"
        f"_backend-{energy_backend}_e2e-{int(bool(use_e2e_constraint))}"
        f"_hb-{tuning['hbond_scale']}_sasa-{tuning['sasa_scale']}"
        f"_vdwr-{tuning['vdw_rep_scale']}_vdwa-{tuning['vdw_attr_scale']}"
    )

    row = {
        "protein_name": protein_name,
        "reference_pdb_id": reference_pdb_id,
        "reference_pdb_path": reference_pdb_path,
        "experiment_id": experiment_id,
        "name": name,
        "pdb_path": str(pdb_path),
        "chain": selected_chain_id or "",
        "residue_start": start if start is not None else pdb_resseqs[0],
        "residue_end": end if end is not None else pdb_resseqs[-1],
        "sequence": folder.sequence,
        "chi_mode": chi_mode,
        "rmsd_mode": rmsd_mode,
        "rmsd_residue_scope": rmsd_residue_scope,
        "energy_backend": energy_backend,
        "use_e2e_constraint": bool(use_e2e_constraint),
        "e2e_scale": float(e2e_scale),
        "gromacs_minimize": bool(gromacs_minimize),
        **gromacs_info,
        "hbond_scale": tuning["hbond_scale"],
        "sasa_scale": tuning["sasa_scale"],
        "vdw_rep_scale": tuning["vdw_rep_scale"],
        "vdw_attr_scale": tuning["vdw_attr_scale"],
        "rotamer_scale": tuning["rotamer_scale"],
        "pi_stack_scale": tuning["pi_stack_scale"],
        "n_residues": len(folder.sequence),
        "total_energy": total_energy,
        "native_end_to_end": native_metrics["end_to_end"],
        "native_rg": native_metrics["radius_of_gyration"],
        "rebuilt_end_to_end": rebuilt_metrics["end_to_end"],
        "rebuilt_rg": rebuilt_metrics["radius_of_gyration"],
        "rebuilt_vs_native_ca_rmsd": rebuilt_vs_native_ca_rmsd,
        **rmsd_meta,
        "n_observed_torsions": len(observed),
        "n_qtf_angles": int(folder.total_angles),
        "rebuilt_ca_pdb_path": rebuilt_ca_pdb_path or "",
        "rebuilt_ca_centroid_pdb_path": rebuilt_ca_centroid_pdb_path or "",
        "rebuilt_full_pdb_path": rebuilt_full_pdb_path or "",
    }
    for k, v in terms.items():
        row[f"term_{k}"] = v
    return row


def score_native_structure(
    *,
    name: str,
    pdb_path: str,
    chain: Optional[str],
    start: Optional[int],
    end: Optional[int],
    chi_mode: str,
    rmsd_mode: str = "ca",
    rmsd_residue_scope: str = "core",
    energy_backend: str = "custom",
    use_e2e_constraint: bool = True,
    e2e_scale: float = 1.0,
    gromacs_minimize: bool = False,
    gromacs_forcefield: str = "amber99sb-ildn",
    gromacs_water: str = "tip3p",
    gromacs_nsteps: int = 5000,
    gromacs_emtol: float = 100.0,
    gromacs_maxwarn: int = 2,
    selective_chi_map: Optional[Dict[str, List[str]]] = None,
    rebuilt_ca_pdb_path: Optional[str] = None,
    rebuilt_ca_centroid_pdb_path: Optional[str] = None,
    rebuilt_full_pdb_path: Optional[str] = None,
) -> Dict[str, Any]:
    trimmed_pdb, pdb_resseqs, sequence, selected_chain_id = extract_subset_pdb(pdb_path, chain, start, end)

    try:
        if selective_chi_map is None:
            selective_chi_map = {
            "Y": ["chi1", "chi2"], "W": ["chi1", "chi2"], "F": ["chi1", "chi2"], "H": ["chi1", "chi2"],
            "D": ["chi1"], "E": ["chi1"], "N": ["chi1"], "Q": ["chi1"],
            "T": ["chi1"], "S": ["chi1"],
            "V": ["chi1"], "I": ["chi1"], "L": ["chi1"], "M": ["chi1"],
            "K": ["chi1"], "R": ["chi1"], "C": ["chi1"], "P": ["chi1"],
            "A": [], "G": [],
        }

        folder = QuantumBiophysicsFolder(
            sequence=sequence,
            chi_mode="selective" if chi_mode == "selective" else "all",
            selective_chi_map=selective_chi_map,
            energy_backend=energy_backend,
            use_e2e_constraint=use_e2e_constraint,
            e2e_scale=e2e_scale,
        )
        folder.current_stage = 3

        angle_vec, observed = compute_qtf_angle_vector(trimmed_pdb, folder, chi_mode=chi_mode)

        return _compute_native_metrics(
            folder=folder,
            angle_vec=angle_vec,
            observed=observed,
            trimmed_pdb=trimmed_pdb,
            pdb_resseqs=pdb_resseqs,
            selected_chain_id=selected_chain_id,
            pdb_path=pdb_path,
            name=name,
            chi_mode=chi_mode,
            rmsd_mode=rmsd_mode,
            rmsd_residue_scope=rmsd_residue_scope,
            energy_backend=energy_backend,
            use_e2e_constraint=use_e2e_constraint,
            e2e_scale=e2e_scale,
            gromacs_minimize=gromacs_minimize,
            gromacs_forcefield=gromacs_forcefield,
            gromacs_water=gromacs_water,
            gromacs_nsteps=gromacs_nsteps,
            gromacs_emtol=gromacs_emtol,
            gromacs_maxwarn=gromacs_maxwarn,
            rebuilt_ca_pdb_path=rebuilt_ca_pdb_path,
            rebuilt_ca_centroid_pdb_path=rebuilt_ca_centroid_pdb_path,
            rebuilt_full_pdb_path=rebuilt_full_pdb_path,
            start=start,
            end=end,
        )
    finally:
        try:
            os.unlink(trimmed_pdb)
        except OSError:
            pass


def make_folder(
    *,
    sequence: str,
    energy_backend: str,
    use_e2e_constraint: bool,
    e2e_scale: float,
    chi_mode: Optional[str] = None,
    selective_chi_map: Optional[Dict[str, List[str]]] = None,
) -> QuantumBiophysicsFolder:
    """Construct a QuantumBiophysicsFolder with the shared runtime options."""
    model_aliases = {
        "custom": "pheat-custom-energy-v1",
        "openmm": "pheat-openmm-prepared",
    }
    score_model = model_aliases.get(str(energy_backend).strip().lower(), energy_backend)
    folder_kwargs: Dict[str, Any] = {
        "sequence": sequence,
        "score_model": score_model,
    }
    if chi_mode is not None:
        folder_kwargs["chi_mode"] = chi_mode
    if selective_chi_map is not None:
        folder_kwargs["selective_chi_map"] = selective_chi_map
    return QuantumBiophysicsFolder(**folder_kwargs)


def gromacs_postprocess_structure(
    *,
    enabled: bool,
    full_pdb_path: str,
    gromacs_dir: str,
    forcefield: str,
    water: str,
    nsteps: int,
    emtol: float,
    maxwarn: int,
    coords: Coords,
    labels: Labels,
    ca_coords: Coords,
    sidechain_centroid_fn: Callable[[Coords, Labels], Coords],
    nonlocal_clash_fn: Callable[[Coords, Labels], Dict[str, object]],
    local_clash_fn: Callable[[Coords, Labels], Dict[str, object]],
) -> Dict[str, Any]:
    """Run GROMACS minimization and refresh structural diagnostics if it succeeds."""
    gromacs_info: Dict[str, Any] = {}
    if enabled:
        gromacs_info = qtf_gromacs.minimize_pdb_with_gromacs(
            full_pdb_path,
            gromacs_dir,
            forcefield=forcefield,
            water=water,
            nsteps=nsteps,
            emtol=emtol,
            maxwarn=maxwarn,
        )
        if gromacs_info.get("gromacs_status") == "ok":
            min_coords, min_labels = qtf_gromacs.parse_pdb_atoms(
                str(gromacs_info["gromacs_minimized_full_pdb_path"])
            )
            min_ca = qtf_gromacs.ca_coords(min_coords, min_labels)
            if len(min_ca) == len(ca_coords):
                coords = min_coords
                labels = min_labels
                ca_coords = min_ca

    return {
        "coords": coords,
        "labels": labels,
        "ca_coords": ca_coords,
        "sidechain_centroids": sidechain_centroid_fn(coords, labels),
        "nonlocal_clash_metrics": nonlocal_clash_fn(coords, labels),
        "local_clash_metrics": local_clash_fn(coords, labels),
        "ring_penetration_metrics": qtf_gromacs.ring_penetration_metrics(coords, labels),
        "gromacs_info": gromacs_info,
    }
