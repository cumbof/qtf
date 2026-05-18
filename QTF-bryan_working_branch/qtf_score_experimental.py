#!/usr/bin/env python3
import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mdtraj as md
import numpy as np
import pandas as pd
from Bio.PDB import PDBIO, PDBParser, Select
from pathlib import Path
import QTF.runner as runner


AA3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

def pdb_id_from_path(p) -> str:
    return Path(str(p)).stem.upper()

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
        if residue.id[0] != ' ':
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
) -> Tuple[str, List[int], str]:
    """
    Extract first MODEL only, one chain, optional residue range, and return:
      - path to a temporary trimmed PDB
      - list of original PDB residue numbers
      - 1-letter sequence
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("native", src_pdb)
    model = next(structure.get_models())

    chains = [c for c in model]
    if chain_id is None:
        protein_chains = [c for c in chains if any(r.id[0] == ' ' for r in c)]
        if len(protein_chains) != 1:
            raise ValueError(f"PDB has multiple protein chains {[c.id for c in protein_chains]}; pass --chain")
        chain_id = protein_chains[0].id

    chain = model[chain_id]

    seq_chars = []
    pdb_resseqs = []
    for res in chain:
        if res.id[0] != ' ':
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

    return out.name, pdb_resseqs, "".join(seq_chars)


def compute_qtf_angle_vector(
    trimmed_pdb: str,
    folder: runner.QuantumBiophysicsFolder,
    chi_mode: str = "all",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Extract torsions from the PDB and pack them into folder.dof_map order.
    chi_mode:
      - 'beam' : keep only chi1, zero all higher chis
      - 'all'  : keep all extractable chi torsions
    """
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
        vals = values[0]
        for quad, val in zip(atom_quads, vals):
            res_idx = topo.atom(int(quad[1])).residue.index
            angle_map[(res_idx, name)] = float(val)
            observed[(res_idx, name)] = float(val)

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


def eval_energy_terms(folder: runner.QuantumBiophysicsFolder, angle_vec: np.ndarray):
    """
    Evaluate the exact stage-3 QTF energy function used in beam search.
    """
    dummy_params = np.zeros(folder.n_params, dtype=float)
    orig_get_angles = folder._get_angles
    try:
        folder._get_angles = lambda _params: angle_vec
        folder.current_stage = 3
        total = float(folder.energy_function(dummy_params, return_terms=True))
        terms = {
            str(k): float(v)
            for k, v in (getattr(folder, "last_energy_terms", {}) or {}).items()
        }
        return total, terms
    finally:
        folder._get_angles = orig_get_angles


def build_ca_coords(folder: runner.QuantumBiophysicsFolder, angle_vec: np.ndarray) -> np.ndarray:
    """
    Rebuild the structure in QTF geometry and return CA coordinates in Å.
    """
    orig_get_angles = folder._get_angles
    try:
        folder._get_angles = lambda _params: angle_vec
        coords, _, _ = folder.build_full_structure(angle_vec)
        ca = np.array([coords[i] for i, lbl in enumerate(folder.static_labels) if lbl[1] == "CA"])
        return ca
    finally:
        folder._get_angles = orig_get_angles


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


def get_tuning_settings():
    return {
        "hbond_scale": float(os.getenv("QTF_HBOND_SCALE", "0.75")),
        "sasa_scale": float(os.getenv("QTF_SASA_SCALE", "0.7")),
        "vdw_rep_scale": float(os.getenv("QTF_VDW_REP_SCALE", "0.1")),
        "vdw_attr_scale": float(os.getenv("QTF_VDW_ATTR_SCALE", "0.1")),
        "rotamer_scale": float(os.getenv("QTF_ROTAMER_SCALE", "1.0")),
        "pi_stack_scale": float(os.getenv("QTF_PI_STACK_SCALE", "1.0")),
    }


def calculate_metrics(ca_coords: np.ndarray) -> Dict[str, float]:
    end_to_end = float(np.linalg.norm(ca_coords[0] - ca_coords[-1]))
    centroid = np.mean(ca_coords, axis=0)
    rg = float(np.sqrt(np.mean(np.sum((ca_coords - centroid) ** 2, axis=1))))
    return {"end_to_end": end_to_end, "radius_of_gyration": rg}


def score_one(
    name: str,
    pdb_path: str,
    chain: Optional[str],
    start: Optional[int],
    end: Optional[int],
    forcefield: str,
    chi_mode: str,
) -> Dict:
    trimmed_pdb, pdb_resseqs, sequence = extract_subset_pdb(pdb_path, chain, start, end)

    try:
        selective_chi_map = {
            "Y": ["chi1", "chi2"], "W": ["chi1", "chi2"], "F": ["chi1", "chi2"], "H": ["chi1", "chi2"],
            "D": ["chi1"], "E": ["chi1"], "N": ["chi1"], "Q": ["chi1"],
            "T": ["chi1"], "S": ["chi1"],
            "V": ["chi1"], "I": ["chi1"], "L": ["chi1"], "M": ["chi1"],
            "K": ["chi1"], "R": ["chi1"], "C": ["chi1"], "P": ["chi1"],
            "A": [], "G": [],
        }

        folder = runner.QuantumBiophysicsFolder(
            sequence=sequence,
            force_field=forcefield,
            chi_mode="selective" if chi_mode == "selective" else "all",
            selective_chi_map=selective_chi_map,
        )
        folder.current_stage = 3

        angle_vec, observed = compute_qtf_angle_vector(trimmed_pdb, folder, chi_mode=chi_mode)

        total_energy, terms = eval_energy_terms(folder, angle_vec)
        rebuilt_ca = build_ca_coords(folder, angle_vec)
        rebuilt_metrics = calculate_metrics(rebuilt_ca)

        native_traj = md.load(trimmed_pdb)
        native_ca_idx = native_traj.topology.select("name CA")
        native_ca = native_traj.xyz[0, native_ca_idx, :] * 10.0
        native_metrics = calculate_metrics(native_ca)

        rebuilt_vs_native_ca_rmsd = kabsch_rmsd(rebuilt_ca, native_ca)

        tuning = get_tuning_settings()
        protein_name = name
        reference_pdb_id = pdb_id_from_path(pdb_path)
        reference_pdb_path = str(pdb_path)
        experiment_id = (
            f"{protein_name}_ff-{forcefield}_chi-{chi_mode}"
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
            "chain": chain or "",
            "residue_start": start if start is not None else pdb_resseqs[0],
            "residue_end": end if end is not None else pdb_resseqs[-1],
            "sequence": sequence,
            "forcefield": forcefield,
            "chi_mode": chi_mode,
            "hbond_scale": tuning["hbond_scale"],
            "sasa_scale": tuning["sasa_scale"],
            "vdw_rep_scale": tuning["vdw_rep_scale"],
            "vdw_attr_scale": tuning["vdw_attr_scale"],
            "rotamer_scale": tuning["rotamer_scale"],
            "pi_stack_scale": tuning["pi_stack_scale"],
            "n_residues": len(sequence),
            "total_energy": total_energy,
            "native_end_to_end": native_metrics["end_to_end"],
            "native_rg": native_metrics["radius_of_gyration"],
            "rebuilt_end_to_end": rebuilt_metrics["end_to_end"],
            "rebuilt_rg": rebuilt_metrics["radius_of_gyration"],
            "rebuilt_vs_native_ca_rmsd": rebuilt_vs_native_ca_rmsd,
            "n_observed_torsions": len(observed),
            "n_qtf_angles": int(folder.total_angles),
        }

        for k, v in terms.items():
            row[f"term_{k}"] = v

        return row
    finally:
        try:
            os.unlink(trimmed_pdb)
        except OSError:
            pass


def load_panel(panel_path: str) -> List[Dict]:
    path = Path(panel_path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    raise ValueError("panel must be .json or .csv")


def main():
    ap = argparse.ArgumentParser(description="Score experimental PDB structures with the QTF energy function.")
    ap.add_argument("--panel", help="JSON or CSV with columns/name,pdb_path,chain,residue_start,residue_end")
    ap.add_argument("--name")
    ap.add_argument("--pdb_path")
    ap.add_argument("--chain", default=None)
    ap.add_argument("--residue_start", type=int, default=None)
    ap.add_argument("--residue_end", type=int, default=None)
    ap.add_argument("--forcefield", default="amber", choices=["amber", "charmm", "opls"])
    ap.add_argument("--chi_mode", default="selective", choices=["beam", "selective", "all"])
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    rows = []
    if args.panel:
        specs = load_panel(args.panel)
        for spec in specs:
            rows.append(
                score_one(
                    name=spec["name"],
                    pdb_path=spec["pdb_path"],
                    chain=spec.get("chain") or None,
                    start=spec.get("residue_start"),
                    end=spec.get("residue_end"),
                    forcefield=spec.get("forcefield", args.forcefield),
                    chi_mode=spec.get("chi_mode", args.chi_mode),
                )
            )
    else:
        if not args.name or not args.pdb_path:
            raise SystemExit("single-structure mode requires --name and --pdb_path")
        rows.append(
            score_one(
                name=args.name,
                pdb_path=args.pdb_path,
                chain=args.chain,
                start=args.residue_start,
                end=args.residue_end,
                forcefield=args.forcefield,
                chi_mode=args.chi_mode,
            )
        )

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(rows, indent=2))

    print(df.to_string(index=False))
    print(f"\nWrote {args.out_csv}")
    if args.out_json:
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
