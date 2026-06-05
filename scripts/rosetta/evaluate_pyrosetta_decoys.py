#!/usr/bin/env python3
"""
Score/evaluate PyRosetta-generated PDB decoys against a native/reference PDB.

Important:
- For multi-model NMR PDBs, this script uses ONLY the first MODEL by default.
- RMSD is computed on CA atoms excluding terminal residues by default:
  residue 2 through residue N-1.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# `kabsch_rmsd` used to be defined here (B5: duplicate of
# `qtf.analysis.stability.kabsch_rmsd`). The canonical implementation
# is now imported and used everywhere; the local copy has been
# removed.
from qtf.analysis.stability import kabsch_rmsd  # noqa: E402, F401


def parse_pdb_ca_coords(
    pdb_path: Path,
    chain: Optional[str] = None,
    model_index: int = 1,
) -> np.ndarray:
    """
    Minimal CA parser.

    If MODEL records are present, only reads the selected 1-indexed MODEL.
    If no MODEL records are present, reads the whole file as a single model.
    """
    coords = []
    seen_model = False
    in_selected_model = model_index == 1

    with pdb_path.open() as f:
        for line in f:
            rec = line[:6].strip()

            if rec == "MODEL":
                seen_model = True
                try:
                    current_model = int(line[10:14].strip())
                except ValueError:
                    current_model = 1
                in_selected_model = current_model == model_index
                continue

            if rec == "ENDMDL":
                if seen_model and in_selected_model:
                    break
                in_selected_model = False
                continue

            if seen_model and not in_selected_model:
                continue

            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue

            chain_id = line[21].strip()
            if chain is not None and chain_id != chain:
                continue

            try:
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except ValueError:
                continue

    if not coords:
        raise ValueError(f"No CA atoms found in {pdb_path} chain={chain!r} model={model_index}")
    return np.asarray(coords, dtype=float)


def core_ca_slice(model_ca: np.ndarray, native_ca: np.ndarray, trim_terminal: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    n = min(len(model_ca), len(native_ca))
    if n <= 2 * trim_terminal:
        raise ValueError(f"Not enough CA atoms ({n}) to trim {trim_terminal} residues from each end.")
    start = trim_terminal
    end = n - trim_terminal
    return (
        model_ca[start:end],
        native_ca[start:end],
        {
            "rmsd_ca_start_residue_1indexed": start + 1,
            "rmsd_ca_end_residue_1indexed": end,
            "rmsd_ca_n_aligned": end - start,
        },
    )


def infer_generation_index(path: Path) -> Optional[int]:
    m = re.search(r"_(\d+)\.pdb$", path.name)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else None


def init_pyrosetta(mute: bool = True):
    try:
        import pyrosetta
        flags = "-mute all" if mute else ""
        pyrosetta.init(flags)
        return pyrosetta
    except ImportError as e:
        raise RuntimeError(
            "PyRosetta is required to compute energies. Install/activate PyRosetta, "
            "or run with --no_energy to compute RMSD only."
        ) from e


def score_pdb_with_pyrosetta(pdb_path: Path, scorefxn_name: str, pyrosetta_mod) -> float:
    from pyrosetta import rosetta
    pose = pyrosetta_mod.pose_from_pdb(str(pdb_path))
    if scorefxn_name:
        scorefxn = rosetta.core.scoring.ScoreFunctionFactory.create_score_function(scorefxn_name)
    else:
        scorefxn = rosetta.core.scoring.get_score_function()
    return float(scorefxn(pose))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoy_dir", required=True)
    ap.add_argument("--native_pdb", required=True)
    ap.add_argument("--pattern", default="*.pdb")
    ap.add_argument("--chain", default=None)
    ap.add_argument("--native_model", type=int, default=1, help="1-indexed MODEL to use from native NMR PDB.")
    ap.add_argument("--trim_terminal", type=int, default=1)
    ap.add_argument("--scorefxn", default="ref2015")
    ap.add_argument("--no_energy", action="store_true")
    ap.add_argument("--out_csv", default="pyrosetta_decoy_eval.csv")
    ap.add_argument("--out_json", default="pyrosetta_decoy_summary.json")
    args = ap.parse_args()

    decoy_dir = Path(args.decoy_dir)
    native_pdb = Path(args.native_pdb)
    pdbs = sorted(decoy_dir.glob(args.pattern))
    if not pdbs:
        raise FileNotFoundError(f"No decoy PDB files found in {decoy_dir} matching {args.pattern!r}")

    native_ca = parse_pdb_ca_coords(native_pdb, chain=args.chain, model_index=args.native_model)
    pyrosetta_mod = None if args.no_energy else init_pyrosetta(mute=True)

    rows: List[Dict] = []
    for pdb in pdbs:
        try:
            model_ca = parse_pdb_ca_coords(pdb, chain=args.chain, model_index=1)
            model_core, native_core, rmsd_meta = core_ca_slice(model_ca, native_ca, args.trim_terminal)
            rmsd = kabsch_rmsd(model_core, native_core)
        except Exception as e:
            rows.append({
                "pdb_path": str(pdb),
                "generation_index": infer_generation_index(pdb),
                "status": "rmsd_failed",
                "error": str(e),
            })
            continue

        energy = math.nan
        if pyrosetta_mod is not None:
            try:
                energy = score_pdb_with_pyrosetta(pdb, args.scorefxn, pyrosetta_mod)
            except Exception as e:
                rows.append({
                    "pdb_path": str(pdb),
                    "generation_index": infer_generation_index(pdb),
                    "rmsd_to_native_A": rmsd,
                    **rmsd_meta,
                    "status": "energy_failed",
                    "error": str(e),
                })
                continue

        rows.append({
            "pdb_path": str(pdb),
            "pdb_name": pdb.name,
            "generation_index": infer_generation_index(pdb),
            "energy": energy,
            "rmsd_to_native_A": rmsd,
            **rmsd_meta,
            "status": "ok",
            "error": "",
        })

    df = pd.DataFrame(rows)
    ok = df[df["status"].eq("ok")].copy()
    if ok.empty:
        df.to_csv(args.out_csv, index=False)
        raise RuntimeError(f"No successfully evaluated decoys. Wrote failures to {args.out_csv}")

    ok = ok.sort_values(["rmsd_to_native_A"], ascending=True).reset_index(drop=True)
    ok["rmsd_rank"] = np.arange(1, len(ok) + 1)

    if "energy" in ok.columns and ok["energy"].notna().any():
        energy_ranked = ok.sort_values(["energy"], ascending=True).reset_index(drop=True)
        energy_ranked["energy_rank"] = np.arange(1, len(energy_ranked) + 1)
        ok = ok.merge(energy_ranked[["pdb_path", "energy_rank"]], on="pdb_path", how="left")
        best_energy_row = energy_ranked.iloc[0]
        lowest_energy = float(best_energy_row["energy"])
        lowest_energy_rmsd = float(best_energy_row["rmsd_to_native_A"])
        lowest_energy_pdb = str(best_energy_row["pdb_path"])
        lowest_energy_generation_index = None if pd.isna(best_energy_row.get("generation_index")) else int(best_energy_row.get("generation_index"))
    else:
        lowest_energy = math.nan
        lowest_energy_rmsd = math.nan
        lowest_energy_pdb = None
        lowest_energy_generation_index = None

    best_rmsd_row = ok.sort_values(["rmsd_to_native_A"], ascending=True).iloc[0]
    best_rmsd = float(best_rmsd_row["rmsd_to_native_A"])
    best_rmsd_energy = math.nan if pd.isna(best_rmsd_row.get("energy")) else float(best_rmsd_row.get("energy"))
    ranking_gap = lowest_energy_rmsd - best_rmsd if not math.isnan(lowest_energy_rmsd) else math.nan

    out_rows = ok.sort_values(["energy_rank"] if "energy_rank" in ok.columns else ["rmsd_rank"]).reset_index(drop=True)
    out_rows.to_csv(args.out_csv, index=False)

    summary = {
        "native_pdb": str(native_pdb),
        "native_model": int(args.native_model),
        "decoy_dir": str(decoy_dir),
        "n_decoys_found": int(len(pdbs)),
        "n_decoys_evaluated": int(len(ok)),
        "scorefxn": None if args.no_energy else args.scorefxn,
        "trim_terminal": int(args.trim_terminal),
        "best_rmsd": best_rmsd,
        "best_rmsd_pdb": str(best_rmsd_row["pdb_path"]),
        "best_rmsd_generation_index": None if pd.isna(best_rmsd_row.get("generation_index")) else int(best_rmsd_row.get("generation_index")),
        "best_rmsd_energy": best_rmsd_energy,
        "lowest_energy": lowest_energy,
        "lowest_energy_pdb": lowest_energy_pdb,
        "lowest_energy_generation_index": lowest_energy_generation_index,
        "lowest_energy_rmsd": lowest_energy_rmsd,
        "ranking_gap": ranking_gap,
        "rmsd_ca_excludes_terminal_residues": args.trim_terminal > 0,
        "rmsd_ca_start_residue_1indexed": int(best_rmsd_row["rmsd_ca_start_residue_1indexed"]),
        "rmsd_ca_end_residue_1indexed": int(best_rmsd_row["rmsd_ca_end_residue_1indexed"]),
        "rmsd_ca_n_aligned": int(best_rmsd_row["rmsd_ca_n_aligned"]),
    }

    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote per-decoy table: {args.out_csv}")
    print(f"Wrote summary: {args.out_json}")


if __name__ == "__main__":
    main()
