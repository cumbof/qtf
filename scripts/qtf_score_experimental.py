#!/usr/bin/env python3
import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mdtraj as md
import numpy as np
import pandas as pd
from Bio.PDB import PDBIO, PDBParser, Select
from qtf.core.folder import QuantumBiophysicsFolder
from qtf.utils import workflow as utils
from qtf.utils.workflow import AA3_TO_1, pdb_id_from_path


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
) -> Tuple[str, List[int], str, str]:
    """
    Extract first MODEL only, one chain, optional residue range, and return:
      - path to a temporary trimmed PDB
      - list of original PDB residue numbers
      - 1-letter sequence
      - selected chain id
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

    if chain_id and chain_id in model:
        chain = model[chain_id]
    else:
        chains = list(model.get_chains())
        if not chains:
            raise ValueError("No chains found in PDB")
        chain = model["A"] if "A" in model else chains[0]
        chain_id = chain.id

    print(f"[info] using chain: {chain_id}")

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

    return out.name, pdb_resseqs, "".join(seq_chars), chain_id


def compute_qtf_angle_vector(
    trimmed_pdb: str,
    folder: QuantumBiophysicsFolder,
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

    omega_quads, omega_values = md.compute_omega(traj)
    fixed_omegas = np.full(max(0, folder.n_residues - 1), np.pi, dtype=float)
    if omega_values.size:
        for quad, val in zip(omega_quads, omega_values[0]):
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


def build_ca_coords(folder: QuantumBiophysicsFolder, angle_vec: np.ndarray) -> np.ndarray:
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


def build_full_coords(folder: QuantumBiophysicsFolder, angle_vec: np.ndarray):
    """
    Rebuild the full structure in QTF geometry and return coordinates, labels, and bonds.
    """
    orig_get_angles = folder._get_angles
    try:
        folder._get_angles = lambda _params: angle_vec
        coords, labels, bonds = folder.build_full_structure(angle_vec)
        return coords, labels, bonds
    finally:
        folder._get_angles = orig_get_angles






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


# `kabsch_rmsd` and the `core_ca_*` helpers used to be defined here
# (B5: duplicate of qtf/analysis/stability.py::kabsch_rmsd and
# qtf/utils/workflow.py::core_ca_*). They are now imported below
# alongside `pdb_id_from_path` and `AA3_TO_1`.
from qtf.analysis.stability import kabsch_rmsd  # noqa: E402, F401


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


def get_tuning_settings():
    return {
        "hbond_scale": 0.75,
        "sasa_scale": 0.7,
        "vdw_rep_scale": 0.01,
        "vdw_attr_scale": 0.1,
        "rotamer_scale": 1.0,
        "pi_stack_scale": 1.0,
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
    chi_mode: str,
    rmsd_mode: str = "ca",
    rmsd_residue_scope: str = "core",
    energy_backend: str = "custom",
    use_e2e_constraint: bool = True,
    e2e_scale: float = 1.0,
    rosetta_repack: bool = False,
    rosetta_fa_min: bool = False,
    rosetta_cen_min: bool = False,
    gromacs_minimize: bool = False,
    gromacs_forcefield: str = "amber99sb-ildn",
    gromacs_water: str = "tip3p",
    gromacs_nsteps: int = 5000,
    gromacs_emtol: float = 100.0,
    gromacs_maxwarn: int = 2,
    rebuilt_ca_pdb_path: Optional[str] = None,
    rebuilt_ca_centroid_pdb_path: Optional[str] = None,
    rebuilt_full_pdb_path: Optional[str] = None,
) -> Dict:
    return utils.score_native_structure(
        name=name,
        pdb_path=pdb_path,
        chain=chain,
        start=start,
        end=end,
        chi_mode=chi_mode,
        rmsd_mode=rmsd_mode,
        rmsd_residue_scope=rmsd_residue_scope,
        energy_backend=energy_backend,
        use_e2e_constraint=use_e2e_constraint,
        e2e_scale=e2e_scale,
        rosetta_repack=rosetta_repack,
        rosetta_fa_min=rosetta_fa_min,
        rosetta_cen_min=rosetta_cen_min,
        gromacs_minimize=gromacs_minimize,
        gromacs_forcefield=gromacs_forcefield,
        gromacs_water=gromacs_water,
        gromacs_nsteps=gromacs_nsteps,
        gromacs_emtol=gromacs_emtol,
        gromacs_maxwarn=gromacs_maxwarn,
        rebuilt_ca_pdb_path=rebuilt_ca_pdb_path,
        rebuilt_ca_centroid_pdb_path=rebuilt_ca_centroid_pdb_path,
        rebuilt_full_pdb_path=rebuilt_full_pdb_path,
    )


def load_panel(panel_path: str) -> List[Dict]:
    path = Path(panel_path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    raise ValueError("panel must be .json or .csv")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score experimental PDB structures with the QTF energy function.")
    ap.add_argument("--panel", help="JSON or CSV with columns/name,pdb_path,chain,residue_start,residue_end")
    ap.add_argument("--name")
    ap.add_argument("--pdb_path")
    ap.add_argument("--chain", default=None)
    ap.add_argument("--residue_start", type=int, default=None)
    ap.add_argument("--residue_end", type=int, default=None)
    ap.add_argument("--chi_mode", default="selective", choices=["beam", "selective", "all"])
    ap.add_argument("--rmsd_mode", default="ca", choices=["ca", "heavy"],
                    help="RMSD atom selection: all CA atoms or all heavy atoms")
    ap.add_argument("--rmsd_residue_scope", default="core", choices=["core", "all"],
                    help="Residue range used for RMSD; core excludes the first and last residues")
    ap.add_argument("--energy_backend", default="custom", choices=["custom", "rosetta", "openmm"])
    ap.add_argument("--use_e2e_constraint", type=int, default=1)
    ap.add_argument("--e2e_scale", type=float, default=1.0)
    ap.add_argument("--rosetta_repack", type=int, default=0)
    ap.add_argument("--rosetta_fa_min", type=int, default=0)
    ap.add_argument("--rosetta_cen_min", type=int, default=0)
    ap.add_argument("--gromacs_minimize", type=int, default=None,
                    help="1 to add hydrogens/topology and minimize the rebuilt native structure with GROMACS")
    ap.add_argument("--gromacs_forcefield", default="amber99sb-ildn")
    ap.add_argument("--gromacs_water", default="tip3p")
    ap.add_argument("--gromacs_nsteps", type=int, default=5000)
    ap.add_argument("--gromacs_emtol", type=float, default=100.0)
    ap.add_argument("--gromacs_maxwarn", type=int, default=2)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args(argv)
    if args.gromacs_minimize is None:
        args.gromacs_minimize = 1

    base_output_dir = Path(args.out_json).parent if args.out_json else Path(args.out_csv).parent
    base_output_dir.mkdir(parents=True, exist_ok=True)
    pdb_output_dir = base_output_dir / "raw_pdbs"
    pdb_output_dir.mkdir(parents=True, exist_ok=True)
    gromacs_output_dir = base_output_dir / "gromacs_pdbs"
    gromacs_output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.panel:
        specs = utils.load_panel(args.panel)
        for spec in specs:
            start = spec.get("residue_start")
            end = spec.get("residue_end")
            rebuilt_ca_pdb_path, rebuilt_ca_centroid_pdb_path, rebuilt_full_pdb_path = utils.make_rebuilt_output_paths(
                pdb_output_dir, spec["name"], start, end
            )
            rows.append(
                score_one(
                    name=spec["name"],
                    pdb_path=spec["pdb_path"],
                    chain=spec.get("chain") or None,
                    start=start,
                    end=end,
                    chi_mode=spec.get("chi_mode", args.chi_mode),
                    rmsd_mode=args.rmsd_mode,
                    rmsd_residue_scope=args.rmsd_residue_scope,
                    energy_backend=args.energy_backend,
                    use_e2e_constraint=bool(args.use_e2e_constraint),
                    e2e_scale=args.e2e_scale,
                    rosetta_repack=bool(args.rosetta_repack),
                    rosetta_fa_min=bool(args.rosetta_fa_min),
                    rosetta_cen_min=bool(args.rosetta_cen_min),
                    gromacs_minimize=bool(args.gromacs_minimize),
                    gromacs_forcefield=args.gromacs_forcefield,
                    gromacs_water=args.gromacs_water,
                    gromacs_nsteps=args.gromacs_nsteps,
                    gromacs_emtol=args.gromacs_emtol,
                    gromacs_maxwarn=args.gromacs_maxwarn,
                    rebuilt_ca_pdb_path=str(rebuilt_ca_pdb_path),
                    rebuilt_ca_centroid_pdb_path=str(rebuilt_ca_centroid_pdb_path),
                    rebuilt_full_pdb_path=str(rebuilt_full_pdb_path),
                )
            )
    else:
        if not args.name or not args.pdb_path:
            raise SystemExit("single-structure mode requires --name and --pdb_path")
        rebuilt_ca_pdb_path, rebuilt_ca_centroid_pdb_path, rebuilt_full_pdb_path = utils.make_rebuilt_output_paths(
            pdb_output_dir, args.name, args.residue_start, args.residue_end
        )
        rows.append(
            score_one(
                name=args.name,
                pdb_path=args.pdb_path,
                chain=args.chain,
                start=args.residue_start,
                end=args.residue_end,
                chi_mode=args.chi_mode,
                rmsd_mode=args.rmsd_mode,
                rmsd_residue_scope=args.rmsd_residue_scope,
                energy_backend=args.energy_backend,
                use_e2e_constraint=bool(args.use_e2e_constraint),
                e2e_scale=args.e2e_scale,
                rosetta_repack=bool(args.rosetta_repack),
                rosetta_fa_min=bool(args.rosetta_fa_min),
                rosetta_cen_min=bool(args.rosetta_cen_min),
                gromacs_minimize=bool(args.gromacs_minimize),
                gromacs_forcefield=args.gromacs_forcefield,
                gromacs_water=args.gromacs_water,
                gromacs_nsteps=args.gromacs_nsteps,
                gromacs_emtol=args.gromacs_emtol,
                gromacs_maxwarn=args.gromacs_maxwarn,
                rebuilt_ca_pdb_path=str(rebuilt_ca_pdb_path),
                rebuilt_ca_centroid_pdb_path=str(rebuilt_ca_centroid_pdb_path),
                rebuilt_full_pdb_path=str(rebuilt_full_pdb_path),
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
